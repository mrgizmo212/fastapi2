import asyncio
import logging
import os
import uuid
from datetime import datetime, time, timedelta
import pytz

import plotly.graph_objects as go
import plotly.io as pio
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field

# Set up loggings
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

# Create a directory for storing charts
CHARTS_DIR = "chart_html"
os.makedirs(CHARTS_DIR, exist_ok=True)
logging.info(f"Chart directory created: {CHARTS_DIR}")

# Mount the charts directory
app.mount("/charts", StaticFiles(directory=CHARTS_DIR), name="charts")
logging.info("Charts directory mounted")

# Pydantic model for stock request
class StockRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    multiplier: str = Field(..., description="Time multiplier for the timespan")
    timespan: str = Field(..., description="Time span (minute, hour, day, week, month, quarter, year)")
    from_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    to_date: str = Field(..., description="End date in YYYY-MM-DD format")
    include_extended_hours: bool = Field(..., description="Include pre and post market data")

async def get_stock_data_chunks(ticker, multiplier, timespan, from_date, to_date, chunk_size=1000):
    logging.info(f"Fetching stock data for {ticker} from {from_date} to {to_date}")
    api_key = os.getenv('POLYGON_API_KEY')
    base_url = "https://api.polygon.io/v2/aggs/ticker"
    
    current_date = datetime.strptime(from_date, "%Y-%m-%d")
    end_date = datetime.strptime(to_date, "%Y-%m-%d")
    
    while current_date <= end_date:
        next_date = min(current_date + timedelta(days=chunk_size), end_date)
        url = f"{base_url}/{ticker}/range/{multiplier}/{timespan}/{current_date.strftime('%Y-%m-%d')}/{next_date.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        if 'results' in data and data['results']:
            yield data['results']
        
        current_date = next_date + timedelta(days=1)

def is_market_hours(dt):
    market_open = time(9, 30)
    market_close = time(16, 0)
    return market_open <= dt.time() <= market_close and dt.weekday() < 5

def process_data_chunk(chunk, include_extended_hours):
    if not chunk:
        return None
    
    processed_bars = []
    for bar in chunk:
        bar_time = datetime.fromtimestamp(bar['t'] / 1000, tz=pytz.timezone('America/New_York'))
        if include_extended_hours or is_market_hours(bar_time):
            processed_bars.append({
                'open': bar['o'],
                'high': bar['h'],
                'low': bar['l'],
                'close': bar['c'],
                'volume': bar['v'],
                'timestamp': bar_time
            })
    
    if not processed_bars:
        return None
    
    return {
        'open': processed_bars[0]['open'],
        'high': max(bar['high'] for bar in processed_bars),
        'low': min(bar['low'] for bar in processed_bars),
        'close': processed_bars[-1]['close'],
        'volume': sum(bar['volume'] for bar in processed_bars),
        'start_time': processed_bars[0]['timestamp'],
        'end_time': processed_bars[-1]['timestamp']
    }

def create_candlestick_chart(processed_data, ticker):
    logging.info(f"Creating candlestick chart for {ticker}")
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(x=[d['start_time'] for d in processed_data],
                    open=[d['open'] for d in processed_data],
                    high=[d['high'] for d in processed_data],
                    low=[d['low'] for d in processed_data],
                    close=[d['close'] for d in processed_data],
                    name='Price',
                    increasing_line_color='#00FFFF', decreasing_line_color='#FF69B4'),
                    row=1, col=1)
        
        # Volume chart
        fig.add_trace(go.Bar(x=[d['start_time'] for d in processed_data],
                             y=[d['volume'] for d in processed_data],
                             name='Volume',
                             marker_color='#9370DB'),
                      row=2, col=1)

        # Add OHLC annotation
        latest_data = processed_data[-1]
        ohlc_text = f"Open: {latest_data['open']:.2f}<br>High: {latest_data['high']:.2f}<br>Low: {latest_data['low']:.2f}<br>Close: {latest_data['close']:.2f}"
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            text=ohlc_text,
            showarrow=False,
            font=dict(color="#FFFFFF", size=12),
            align="left",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#FFFFFF",
            borderwidth=1,
            borderpad=4
        )

        fig.update_layout(
            height=800, 
            width=1200, 
            title_text=f"{ticker} Stock Analysis",
            paper_bgcolor='#000000',
            plot_bgcolor='#000000',
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
            xaxis_title=None,
            xaxis2_title=None,
            yaxis=dict(side="right", title=None, tickformat=',.2f', tickfont=dict(color='#FFFFFF')),
            yaxis2=dict(side="right", title=None, tickfont=dict(color='#FFFFFF')),
            font=dict(color='#FFFFFF'),
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(font=dict(color='#FFFFFF'))
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333333', zeroline=False, tickformat='%Y-%m-%d %H:%M', tickfont=dict(color='#FFFFFF'))
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333333', zeroline=False)

        # Generate a unique filename for the HTML file
        filename = f"{ticker}_{uuid.uuid4()}.html"
        filepath = os.path.join(CHARTS_DIR, filename)

        # Save the chart as an interactive HTML file
        pio.write_html(fig, file=filepath, auto_open=False)
        logging.info(f"Candlestick chart saved as interactive HTML: {filepath}")

        return filename
    except Exception as e:
        logging.error(f"Error in create_candlestick_chart: {str(e)}")
        raise

def is_market_open():
    ny_time = datetime.now(pytz.timezone('America/New_York'))
    market_open = time(9, 30)
    market_close = time(16, 0)
    return market_open <= ny_time.time() <= market_close and ny_time.weekday() < 5

async def analyze_stock(stock_request: StockRequest):
    logging.info(f"Analyzing stock: {stock_request.ticker}")
    ticker = stock_request.ticker.upper()
    multiplier = stock_request.multiplier
    timespan = stock_request.timespan
    from_date = stock_request.from_date
    to_date = stock_request.to_date
    include_extended_hours = stock_request.include_extended_hours

    processed_data = []
    async for chunk in get_stock_data_chunks(ticker, multiplier, timespan, from_date, to_date):
        processed_chunk = process_data_chunk(chunk, include_extended_hours)
        if processed_chunk:
            processed_data.append(processed_chunk)

    if not processed_data:
        logging.error(f"No data available for {ticker} from {from_date} to {to_date}")
        raise HTTPException(status_code=404, detail=f"No data available for {ticker} from {from_date} to {to_date}. Please check your date range and ensure it's not in the future.")

    # Create Plotly candlestick chart
    try:
        chart_filename = create_candlestick_chart(processed_data, ticker)
        logging.info(f"Candlestick chart created: {chart_filename}")
    except Exception as e:
        logging.error(f"Failed to create candlestick chart: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create candlestick chart: {str(e)}")

    # Get current time in New York
    ny_time = datetime.now(pytz.timezone('America/New_York'))
    current_time = ny_time.strftime("%Y-%m-%d %I:%M %p ET")
    market_status = "open" if is_market_open() else "closed"

    # Prepare summary data
    latest_data = processed_data[-1]
    opening_data = processed_data[0]
    high_price = max(chunk['high'] for chunk in processed_data)
    low_price = min(chunk['low'] for chunk in processed_data)
    total_volume = sum(chunk['volume'] for chunk in processed_data)

    # Prepare prompt for GPT-4o
    analysis_header = f"TTG AI - MARI Stock Chart Analysis for: {ticker} on a {multiplier} {timespan} chart from {from_date} to {to_date}\n"
    analysis_header += f"Current Time: {current_time}, Market is currently {market_status}\n"
    analysis_header += f"{'Including' if include_extended_hours else 'Excluding'} pre and post market data\n\n"

    prompt = f"""
    {analysis_header}
    Analyze the following summary data for {ticker} from {from_date} to {to_date}:

    Opening Price: {opening_data['open']}
    Current/Latest Price: {latest_data['close']}
    Day's High: {high_price}
    Day's Low: {low_price}
    Total Volume: {total_volume}

    A candlestick chart of this stock has been generated. Please provide insights on the stock's performance,
    trends, and any notable events or patterns you can discern from this summary data. Start your analysis with the header provided above.
    
    Important: The current time is {current_time} and the market is {market_status}. Please adjust your analysis accordingly,
    avoiding phrases like "end of day" if the market is still open, and considering the current market status in your analysis.
    
    {'This analysis includes pre and post market data.' if include_extended_hours else 'This analysis only includes regular market hours data (9:30 AM to 4:00 PM ET).'}
    
    Focus on the most recent price movements and volumes, and provide any relevant short-term predictions or observations.
    """

    # Get GPT-4o interpretation
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

        analysis = analysis_header + response.choices[0].message.content

        logging.info(f"Analysis for {ticker} completed successfully")
        return analysis, chart_filename
    except Exception as e:
        logging.error(f"Error in OpenAI API call: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while getting AI analysis: {str(e)}")

@app.post("/analyze_stock")
async def api_analyze_stock(stock_request: StockRequest, request: Request):
    analysis, chart_filename = await analyze_stock(stock_request)
    
    # Get the base URL of the current request
    base_url = str(request.base_url)
    
    # Construct the full URL to the chart
    chart_url = f"{base_url}charts/{chart_filename}"
    
    # Create the HTML response with embedded JavaScript to open the chart
    html_content = f"""
    <html>
    <head>
        <title>Stock Analysis Result</title>
        <script>
            function openChartWindow() {{
                window.open('{chart_url}', 'ChartWindow', 'width=1200,height=800');
            }}
            // Automatically open the chart when the page loads
            window.onload = openChartWindow;
        </script>
    </head>
    <body style="background-color: #1e1e1e; color: #e1e1e1; font-family: Arial, sans-serif; padding: 20px;">
        <h2>Analysis Summary:</h2>
        <p>{analysis}</p>
        <button onclick="openChartWindow()" style="background-color: #00FFFF; color: #000000; border: none; padding: 10px 20px; cursor: pointer;">View Chart Again</button>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content, status_code=200)

async def cli_analyze_stock():
    print("Welcome to the Stock Analyzer!")
    while True:
        ticker = input("Enter the stock ticker (e.g., AAPL): ").strip().upper()
        multiplier = input("Enter the multiplier (e.g., 1): ").strip()
        timespan = input("Enter the timespan (minute, hour, day, week, month, quarter, year): ").strip().lower()
        from_date = input("Enter the start date (YYYY-MM-DD): ").strip()
        to_date = input("Enter the end date (YYYY-MM-DD): ").strip()
        include_extended_hours = input("Include pre and post market data? (yes/no): ").strip().lower() == 'yes'

        stock_request = StockRequest(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_date=from_date,
            to_date=to_date,
            include_extended_hours=include_extended_hours
        )

        try:
            analysis, chart_filename = await analyze_stock(stock_request)
            print("\nAnalysis Result:")
            print(analysis)
            print(f"\nCandlestick chart saved as: {os.path.join(CHARTS_DIR, chart_filename)}")
            print("You can open this HTML file in your web browser to view the interactive chart.")
        except HTTPException as e:
            print(f"\nError: {e.detail}")

        another = input("\nWould you like to analyze another stock? (y/n): ").strip().lower()
        if another != 'y':
            break

    print("Thank you for using the Stock Analyzer!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        asyncio.run(cli_analyze_stock())
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
