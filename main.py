import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from datetime import datetime
import base64
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import asyncio
import io
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize FastAPI app
app = FastAPI()

# Pydantic model for request body
class StockRequest(BaseModel):
    ticker: str
    multiplier: str
    timespan: str
    from_date: str
    to_date: str

# Function to get stock data from Polygon API
def get_stock_data(ticker, multiplier, timespan, from_date, to_date):
    api_key = os.getenv('POLYGON_API_KEY')
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    response = requests.get(url)
    return response.json()

# Function to create chart image
def create_chart_image(data):
    # Convert data to pandas DataFrame
    df = pd.DataFrame(data['results'])
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    df.set_index('t', inplace=True)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'n', 'vw']

    # Create the candlestick chart
    fig, ax = plt.subplots(figsize=(10, 6))
    mpf.plot(df, type='candle', style='charles', ax=ax)
    
    # Save the figure to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert the buffer to base64
    chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    plt.close(fig)
    
    return chart_image

# Function to keep connection alive
async def keep_alive():
    await asyncio.sleep(300)  # Sleep for 5 minutes (300 seconds)

@app.post("/analyze_stock")
async def analyze_stock(request: StockRequest, background_tasks: BackgroundTasks):
    ticker = request.ticker.upper()  # Convert ticker to uppercase
    multiplier = request.multiplier
    timespan = request.timespan
    from_date = request.from_date
    to_date = request.to_date

    # Get stock data
    stock_data = get_stock_data(ticker, multiplier, timespan, from_date, to_date)

    if 'results' not in stock_data or not stock_data['results']:
        raise HTTPException(status_code=404, detail=f"No data available for {ticker} from {from_date} to {to_date}. Please check your date range and ensure it's not in the future.")

    # Create chart image
    chart_image_base64 = create_chart_image(stock_data)

    # Prepare prompt for GPT-4o
    analysis_header = f"TTG AI - MARI Stock Chart Analysis for: {ticker} on a {multiplier} {timespan} chart from {from_date} to {to_date}\n\n"
    
    # Include aggregate bar data in the prompt
    aggregate_data = "\n".join([f"Date: {datetime.fromtimestamp(bar['t']/1000).strftime('%Y-%m-%d')}, Open: {bar['o']}, High: {bar['h']}, Low: {bar['l']}, Close: {bar['c']}, Volume: {bar['v']}" for bar in stock_data['results']])
    
    prompt = f"""
    {analysis_header}
    Analyze the following stock data for {ticker} from {from_date} to {to_date}:
    
    Aggregate Bar Data:
    {aggregate_data}
    
    A chart image of this stock has been generated. Please provide insights on the stock's performance, 
    trends, and any notable events or patterns you can discern from the data. Start your analysis with the header provided above.
    """

    # Get GPT-4o interpretation
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial analyst expert in stock market analysis."},
                {"role": "user", "content": prompt}
            ]
        )

        analysis = analysis_header + response.choices[0].message.content

        # Add keep-alive task
        background_tasks.add_task(keep_alive)

        # Return only the chart image and analysis to the user
        return {
            "chart_image": chart_image_base64,
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while getting AI analysis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
