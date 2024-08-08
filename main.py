import asyncio
import logging
import os
import uuid
from datetime import datetime

import plotly.graph_objects as go
import plotly.io as pio
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field

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

# Function to get stock data from Polygon API
def get_stock_data(ticker, multiplier, timespan, from_date, to_date):
    logging.info(f"Fetching stock data for {ticker} from {from_date} to {to_date}")
    api_key = os.getenv('POLYGON_API_KEY')
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    response = requests.get(url)
    logging.info(f"Received response from Polygon API for {ticker}")
    return response.json()

# Function to create Plotly candlestick chart
def create_candlestick_chart(stock_data, ticker):
    logging.info(f"Creating candlestick chart for {ticker}")
    try:
        dates = [datetime.fromtimestamp(bar['t']/1000) for bar in stock_data['results']]
        logging.info("Dates processed")
        
        # Calculate intraday OHLC
        intraday_open = stock_data['results'][0]['o']
        intraday_high = max(bar['h'] for bar in stock_data['results'])
        intraday_low = min(bar['l'] for bar in stock_data['results'])
        intraday_close = stock_data['results'][-1]['c']
        current_price = intraday_close

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, row_heights=[0.7, 0.3])
        logging.info("Subplots created")

        # Candlestick chart
        fig.add_trace(go.Candlestick(x=dates,
                    open=[bar['o'] for bar in stock_data['results']],
                    high=[bar['h'] for bar in stock_data['results']],
                    low=[bar['l'] for bar in stock_data['results']],
                    close=[bar['c'] for bar in stock_data['results']],
                    name='Price',
                    increasing_line_color='#00FFFF', decreasing_line_color='#FF69B4'),
                    row=1, col=1)
        logging.info("Candlestick trace added")

        # Volume chart
        colors = ['#00FFFF' if close >= open else '#FF69B4' for close, open in zip([bar['c'] for bar in stock_data['results']], [bar['o'] for bar in stock_data['results']])]
        fig.add_trace(go.Bar(x=dates,
                             y=[bar['v'] for bar in stock_data['results']],
                             name='Volume',
                             marker_color=colors),
                      row=2, col=1)
        logging.info("Volume trace added")

        # Add intraday OHLC and current price annotation
        ohlc_text = f"Current: {current_price:.2f}<br>Open: {intraday_open:.2f}<br>High: {intraday_high:.2f}<br>Low: {intraday_low:.2f}<br>Close: {intraday_close:.2f}"
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

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333333', zeroline=False, tickformat='%H:%M', tickfont=dict(color='#FFFFFF'))
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333333', zeroline=False)

        # Show dates on both subplots
        fig.update_xaxes(showticklabels=True, row=1, col=1)
        fig.update_xaxes(showticklabels=True, row=2, col=1)

        logging.info("Chart layout updated")

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

async def analyze_stock(stock_request: StockRequest):
    logging.info(f"Analyzing stock: {stock_request.ticker}")
    ticker = stock_request.ticker.upper()
    multiplier = stock_request.multiplier
    timespan = stock_request.timespan
    from_date = stock_request.from_date
    to_date = stock_request.to_date

    # Get stock data
    stock_data = get_stock_data(ticker, multiplier, timespan, from_date, to_date)

    if 'results' not in stock_data or not stock_data['results']:
        logging.error(f"No data available for {ticker} from {from_date} to {to_date}")
        raise HTTPException(status_code=404, detail=f"No data available for {ticker} from {from_date} to {to_date}. Please check your date range and ensure it's not in the future.")

    # Create Plotly candlestick chart
    try:
        chart_filename = create_candlestick_chart(stock_data, ticker)
        logging.info(f"Candlestick chart created: {chart_filename}")
    except Exception as e:
        logging.error(f"Failed to create candlestick chart: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create candlestick chart: {str(e)}")

    # Prepare prompt for GPT-4o
    analysis_header = f"TTG AI - MARI Stock Chart Analysis for: {ticker} on a {multiplier} {timespan} chart from {from_date} to {to_date}\n\n"

    # Include aggregate bar data in the prompt
    aggregate_data = "\n".join([f"Date: {datetime.fromtimestamp(bar['t']/1000).strftime('%Y-%m-%d %H:%M')}, Open: {bar['o']}, High: {bar['h']}, Low: {bar['l']}, Close: {bar['c']}, Volume: {bar['v']}" for bar in stock_data['results']])

    prompt = f"""
    {analysis_header}
    Analyze the following stock data for {ticker} from {from_date} to {to_date}:

    Aggregate Bar Data:
    {aggregate_data}

    A candlestick chart of this stock has been generated. Please provide insights on the stock's performance,
    trends, and any notable events or patterns you can discern from the data. Start your analysis with the header provided above.
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
async def api_analyze_stock(stock_request: StockRequest):
    analysis, chart_filename = await analyze_stock(stock_request)
    return {
        "analysis": analysis,
        "chart_url": f"/charts/{chart_filename}"
    }

async def cli_analyze_stock():
    print("Welcome to the Stock Analyzer!")
    while True:
        ticker = input("Enter the stock ticker (e.g., AAPL): ").strip().upper()
        multiplier = input("Enter the multiplier (e.g., 1): ").strip()
        timespan = input("Enter the timespan (minute, hour, day, week, month, quarter, year): ").strip().lower()
        from_date = input("Enter the start date (YYYY-MM-DD): ").strip()
        to_date = input("Enter the end date (YYYY-MM-DD): ").strip()

        stock_request = StockRequest(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_date=from_date,
            to_date=to_date
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
