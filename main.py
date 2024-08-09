import asyncio
import logging
import os
import sys
from datetime import datetime, date, time, timedelta

import pandas as pd
import pytz
import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from openai import OpenAI
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath("./PatternPy"))
from PatternPy.tradingpatterns.tradingpatterns import (
    calculate_support_resistance,
    detect_channel,
    detect_double_top_bottom,
    detect_head_shoulder,
    detect_multiple_tops_bottoms,
    detect_triangle_pattern,
    detect_trendline,
    detect_wedge,
    find_pivots
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
logging.info("Environment variables loaded")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
logging.info("OpenAI client initialized")

# Initialize FastAPI app
app = FastAPI()
logging.info("FastAPI app initialized")

# Pydantic model for stock request
class StockRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    multiplier: str = Field(..., description="Time multiplier for the timespan")
    timespan: str = Field(..., description="Time span (minute, hour, day, week, month, quarter, year)")
    from_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    to_date: str = Field(..., description="End date in YYYY-MM-DD format")
    include_extended_hours: bool = Field(..., description="Include pre and post market data")

def validate_dates(from_date: str, to_date: str, current_date: date) -> tuple:
    try:
        from_date = datetime.strptime(from_date, "%Y-%m-%d").date()
        to_date = datetime.strptime(to_date, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError("Invalid date format. Please use YYYY-MM-DD.") from e

    if from_date > current_date:
        logging.warning(f"From date {from_date} is in the future. Adjusting to today's date ({current_date}).")
        from_date = current_date

    if to_date > current_date:
        logging.warning(f"To date {to_date} is in the future. Adjusting to today's date ({current_date}).")
        to_date = current_date

    if from_date > to_date:
        raise ValueError(f"From date ({from_date}) cannot be after to date ({to_date}).")

    return from_date, to_date

async def fetch_daily_data(session, url, current_date, include_extended_hours):
    async with session.get(url) as response:
        data = await response.json()
        if response.status == 200 and data.get('status') == 'OK':
            result = {
                't': int(datetime.strptime(data['from'], "%Y-%m-%d").timestamp() * 1000),
                'o': data['open'],
                'h': data['high'],
                'l': data['low'],
                'c': data['close'],
                'v': data['volume'],
            }
            if include_extended_hours:
                result['preMarket'] = data.get('preMarket')
                result['afterHours'] = data.get('afterHours')
            logging.info(f"Received data for {current_date}")
            return result
        elif response.status != 404:
            logging.error(f"API request failed with status code {response.status}")
            logging.error(f"Response content: {data}")
        else:
            logging.info(f"No data available for {current_date} (likely a weekend or holiday)")
        return None

async def fetch_current_day_data(session, url, current_date, include_extended_hours):
    async with session.get(url) as response:
        data = await response.json()
        if response.status == 200 and data.get('results'):
            minute_data = data['results']
            ny_tz = pytz.timezone('America/New_York')
            market_open = time(9, 30)
            market_close = time(16, 0)

            if include_extended_hours:
                filtered_data = minute_data
            else:
                filtered_data = [bar for bar in minute_data if market_open <= datetime.fromtimestamp(bar['t']/1000, tz=ny_tz).time() <= market_close]

            if filtered_data:
                result = {
                    't': filtered_data[-1]['t'],
                    'o': filtered_data[0]['o'],
                    'h': max(bar['h'] for bar in filtered_data),
                    'l': min(bar['l'] for bar in filtered_data),
                    'c': filtered_data[-1]['c'],
                    'v': sum(bar['v'] for bar in filtered_data)
                }
                logging.info(f"Received and processed current day data for {current_date}")
                return result
            else:
                logging.info(f"No data available within specified hours for {current_date}")
        else:
            logging.error(f"Failed to fetch current day data: {data}")
        return None

async def get_stock_data(ticker, multiplier, timespan, from_date, to_date, include_extended_hours):
    logging.info(f"Fetching stock data for {ticker} from {from_date} to {to_date}")
    api_key = os.getenv('POLYGON_API_KEY')
    
    async with aiohttp.ClientSession() as session:
        if timespan.lower() == 'day':
            base_url = "https://api.polygon.io/v1/open-close"
            current_date = from_date
            today = date.today()
            
            tasks = []
            while current_date <= to_date:
                if current_date < today:
                    url = f"{base_url}/{ticker}/{current_date}?adjusted=true&apiKey={api_key}"
                    tasks.append(fetch_daily_data(session, url, current_date, include_extended_hours))
                elif current_date == today:
                    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{current_date}/{current_date}?adjusted=true&sort=asc&limit=1440&apiKey={api_key}"
                    tasks.append(fetch_current_day_data(session, url, current_date, include_extended_hours))
                current_date += timedelta(days=1)
            
            all_results = await asyncio.gather(*tasks)
            all_results = [result for result in all_results if result is not None]
        else:
            base_url = "https://api.polygon.io/v2/aggs/ticker"
            url = f"{base_url}/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
            async with session.get(url) as response:
                if response.status != 200:
                    data = await response.json()
                    logging.error(f"API request failed with status code {response.status}")
                    logging.error(f"Response content: {data}")
                    raise HTTPException(status_code=response.status, detail=f"API request failed: {data.get('error', 'Unknown error')}")
                data = await response.json()
                all_results = data.get('results', [])

    logging.info(f"Total data points fetched: {len(all_results)}")

    if not all_results:
        logging.warning(f"No data available for {ticker} from {from_date} to {to_date}")
        return []  # Return an empty list instead of raising an exception

    return all_results

def is_market_hours(dt):
    market_open = time(9, 30)
    market_close = time(16, 0)
    return market_open <= dt.time() <= market_close and dt.weekday() < 5

def process_data(data, include_extended_hours, timespan):
    ny_tz = pytz.timezone('America/New_York')
    
    processed_data = []
    for bar in data:
        bar_time = datetime.fromtimestamp(bar['t'] / 1000, tz=ny_tz)
        
        if timespan.lower() == 'day':
            processed_bar = {
                'open': bar['o'],
                'high': bar['h'],
                'low': bar['l'],
                'close': bar['c'],
                'volume': bar['v'],
                'timestamp': bar_time
            }
            if include_extended_hours:
                processed_bar['preMarket'] = bar.get('preMarket')
                processed_bar['afterHours'] = bar.get('afterHours')
            processed_data.append(processed_bar)
        elif include_extended_hours or is_market_hours(bar_time):
            processed_data.append({
                'open': bar['o'],
                'high': bar['h'],
                'low': bar['l'],
                'close': bar['c'],
                'volume': bar['v'],
                'timestamp': bar_time
            })
    
    logging.info(f"Processed {len(processed_data)} data points out of {len(data)} total")
    
    if not processed_data:
        if timespan.lower() == 'day' or include_extended_hours:
            logging.warning("No data points available for the specified date range.")
        else:
            logging.warning("No data points within regular market hours for the specified date range.")
    
    return processed_data

def is_market_open():
    ny_tz = pytz.timezone('America/New_York')
    current_time = datetime.now(ny_tz).time()
    current_day = datetime.now(ny_tz).weekday()
    market_open = time(9, 30)
    market_close = time(16, 0)
    return market_open <= current_time <= market_close and current_day < 5

async def analyze_stock(stock_request: StockRequest):
    logging.info(f"Analyzing stock: {stock_request.ticker}")

    # Get current date and time in New York
    ny_tz = pytz.timezone('America/New_York')
    current_datetime = datetime.now(ny_tz)
    current_date = current_datetime.date()

    logging.info(f"Analysis performed as of {current_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    try:
        from_date, to_date = validate_dates(stock_request.from_date, stock_request.to_date, current_date)
        logging.info(f"Adjusted date range: from {from_date} to {to_date}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    ticker = stock_request.ticker.upper()
    multiplier = stock_request.multiplier
    timespan = stock_request.timespan
    include_extended_hours = stock_request.include_extended_hours

    try:
        data = await get_stock_data(ticker, multiplier, timespan, from_date, to_date, include_extended_hours)
        processed_data = process_data(data, include_extended_hours, timespan)
    except HTTPException as e:
        logging.error(f"HTTPException in get_stock_data: {str(e.detail)}")
        raise Exception(str(e.detail)) from e
    except Exception as e:
        logging.error(f"Unexpected error in get_stock_data: {str(e)}")
        raise Exception(f"An unexpected error occurred: {str(e)}") from e

    if not processed_data:
        error_message = f"No {'market hours' if not include_extended_hours else ''} data available for {ticker} from {from_date} to {to_date}. "
        if not include_extended_hours and timespan.lower() != 'day':
            error_message += "Try including pre and post market data or adjusting your date range."
        else:
            error_message += "Please check your date range or try a different stock."
        logging.error(error_message)
        raise HTTPException(status_code=404, detail=error_message)

    # Convert processed_data to DataFrame for pattern recognition
    df = pd.DataFrame(processed_data)
    df = df.rename(columns={
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'open': 'Open',
        'volume': 'Volume',
        'timestamp': 'date'
    })
    df.set_index('date', inplace=True)

    # Perform pattern recognition
    patterns_with_times = {}

    df, hs_patterns = detect_head_shoulder(df)
    patterns_with_times.update(hs_patterns)

    df, mtb_patterns = detect_multiple_tops_bottoms(df)
    patterns_with_times.update(mtb_patterns)

    df, sr_patterns = calculate_support_resistance(df)
    patterns_with_times.update(sr_patterns)

    df, triangle_patterns = detect_triangle_pattern(df)
    patterns_with_times.update(triangle_patterns)

    df, wedge_patterns = detect_wedge(df)
    patterns_with_times.update(wedge_patterns)

    df, channel_patterns = detect_channel(df)
    patterns_with_times.update(channel_patterns)

    df, double_patterns = detect_double_top_bottom(df)
    patterns_with_times.update(double_patterns)

    df, trendline_patterns = detect_trendline(df)
    patterns_with_times.update(trendline_patterns)

    # Rename columns for find_pivots function
    df = df.rename(columns={'High': 'high', 'Low': 'low'})
    df, pivot_patterns = find_pivots(df)
    patterns_with_times.update(pivot_patterns)
    df = df.rename(columns={'high': 'High', 'low': 'Low'})  # Rename back for consistency

    # Collect detected patterns with timestamps
    detected_patterns = {}
    for pattern, timestamps in patterns_with_times.items():
        if timestamps:
            detected_patterns[pattern] = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps]

    # Get current time in New York
    current_time = current_datetime.strftime("%Y-%m-%d %I:%M %p ET")
    market_status = "open" if is_market_open() else "closed"

    # Prepare summary data
    latest_data = processed_data[-1]
    opening_data = processed_data[0]
    high_price = max(bar['high'] for bar in processed_data)
    low_price = min(bar['low'] for bar in processed_data)
    total_volume = sum(bar['volume'] for bar in processed_data)

    # Prepare prompt for GPT-4
    analysis_header = "# TTG AI - MARI Stock Chart & Pattern Analysis for: {} from {} to {} using a {} {} window.\n\n".format(ticker, from_date, to_date, multiplier, timespan)
    analysis_header += "**Chart Details:** {} {} chart from {} to {}\n".format(multiplier, timespan, from_date, to_date)
    analysis_header += "**Current Time:** {}, Market is currently {}\n".format(current_time, market_status)
    
    if timespan.lower() == 'day':
        market_hours_note = "This analysis includes regular market hours data." if not include_extended_hours else "This analysis includes pre-market, regular hours, and after-hours data."
    else:
        market_hours_note = "This analysis includes pre and post market data." if include_extended_hours else "This analysis only includes regular market hours data (9:30 AM to 4:00 PM ET)."
    analysis_header += f"**Data Range:** {market_hours_note}\n\n"

    patterns_str = "\n".join([f"- {k}:\n  " + '\n  '.join([f"  - Detected at {t}" for t in v]) for k, v in detected_patterns.items()])

    prompt = """
{analysis_header}
## Summary Data:

- Opening Price: ${opening_price:.2f}
- Current/Latest Price: ${latest_price:.2f}
- Day's High: ${high_price:.2f}
- Day's Low: ${low_price:.2f}
- Total Volume: {total_volume:,} shares

## Detected Patterns:
{patterns_str}

Please provide a comprehensive & detailed analysis of the stock's performance, trends and patterns. Include the following sections:

1. Price Movement Analysis
2. Volume Analysis
3. Technical Patterns Interpretation (including when patterns were detected)
4. Short-Term Predictions and Observations
5. Conclusion
6. TLDR - Too Long Didn't Read

Important: The current time is {current_time} and the market is {market_status}. Adjust your analysis accordingly,
avoiding phrases like "end of day" if the market is still open, and considering the current market status.

{market_hours_note}

Focus on the most recent price movements and volumes, and provide relevant short-term predictions or observations.
When discussing patterns, refer to the times they were detected to provide context.

Please format your response using Markdown syntax, including appropriate headers, bullet points, and emphasis where necessary.
""".format(
    analysis_header=analysis_header,
    opening_price=opening_data['open'],
    latest_price=latest_data['close'],
    high_price=high_price,
    low_price=low_price,
    total_volume=total_volume,
    patterns_str=patterns_str,
    current_time=current_time,
    market_status=market_status,
    market_hours_note=market_hours_note
)

    # Get GPT-4 interpretation
    try:
        logging.info(f"Sending request to OpenAI for {ticker} analysis")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial analyst expert in stock market analysis."},
                {"role": "user", "content": prompt}
            ]
        )
        logging.info(f"Received response from OpenAI for {ticker} analysis")

        analysis = response.choices[0].message.content

        logging.info(f"Analysis for {ticker} completed successfully")
        return analysis
    except Exception as e:
        logging.error(f"Error in OpenAI API call: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while getting AI analysis: {str(e)}") from e

@app.post("/analyze_stock")
async def api_analyze_stock(stock_request: StockRequest, request: Request):
    analysis = await analyze_stock(stock_request)

    # Return the Markdown response
    return Response(content=analysis, media_type="text/markdown")

async def cli_analyze_stock():
    print("Welcome to the Stock Analyzer!")
    while True:
        ticker = input("Enter the stock ticker (e.g., AAPL): ").strip().upper()
        multiplier = input("Enter the multiplier (e.g., 1): ").strip()
        timespan = input("Enter the timespan (minute, hour, day, week, month, quarter, year): ").strip().lower()
        from_date = input("Enter the start date (YYYY-MM-DD): ").strip()
        to_date = input("Enter the end date (YYYY-MM-DD): ").strip()
        include_extended_hours = input("Include pre and post market data? (yes/no): ").strip().lower() == 'yes'

        try:
            stock_request = StockRequest(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_date=from_date,
                to_date=to_date,
                include_extended_hours=include_extended_hours
            )
            analysis = await analyze_stock(stock_request)
            print("\nAnalysis Result:")
            print(analysis)
        except HTTPException as e:
            print(f"\nError: {e.detail}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {str(e)}")

        another = input("\nWould you like to analyze another stock? (y/n): ").strip().lower()
        if another != 'y':
            break

    print("Thank you for using the Stock Analyzer!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        asyncio.run(cli_analyze_stock())
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
