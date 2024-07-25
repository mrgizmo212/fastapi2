import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import base64
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import asyncio

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

# Function to create TradingView chart image
def create_tradingview_chart_image(ticker):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)
    
    url = f"https://www.tradingview.com/chart/?symbol={ticker}"
    driver.get(url)

    # Wait for the chart to load
    wait = WebDriverWait(driver, 20)
    chart = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "chart-markup-table")))
    
    # Give extra time for the chart to render completely
    time.sleep(5)

    # Take screenshot
    chart_image = driver.get_screenshot_as_base64()
    
    driver.quit()

    return chart_image

# Function to keep connection alive
async def keep_alive():
    await asyncio.sleep(300)  # Sleep for 5 minutes (300 seconds)

@app.post("/analyze_stock")
async def analyze_stock(request: StockRequest, background_tasks: BackgroundTasks):
    ticker = request.ticker.upper()
    multiplier = request.multiplier
    timespan = request.timespan
    from_date = request.from_date
    to_date = request.to_date

    # Get stock data
    stock_data = get_stock_data(ticker, multiplier, timespan, from_date, to_date)

    if 'results' not in stock_data or not stock_data['results']:
        raise HTTPException(status_code=404, detail=f"No data available for {ticker} from {from_date} to {to_date}. Please check your date range and ensure it's not in the future.")

    # Create TradingView chart image
    chart_image_base64 = create_tradingview_chart_image(ticker)

    # Prepare prompt for GPT-4o
    prompt = f"""
    Analyze the following stock data for {ticker} from {from_date} to {to_date}:
    
    {stock_data}
    
    A TradingView chart image of this stock has been generated. Please provide insights on the stock's performance, 
    trends, and any notable events or patterns you can discern from the data.
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

        analysis = response.choices[0].message.content

        # Add keep-alive task
        background_tasks.add_task(keep_alive)

        return {
            "chart_image": chart_image_base64,
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while getting AI analysis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
