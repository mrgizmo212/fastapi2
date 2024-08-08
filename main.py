import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import asyncio
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import uuid

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize FastAPI app
app = FastAPI()

# Create a directory for storing images
IMAGES_DIR = "chart_images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# Mount the images directory
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# Pydantic model for request body
class StockRequest(BaseModel):
    ticker: str = Field(..., to_lower=True)
    multiplier: str = Field(..., to_lower=True)
    timespan: str = Field(..., to_lower=True)
    from_date: str
    to_date: str

    class Config:
        alias_generator = lambda string: string.lower()
        allow_population_by_field_name = True

# Function to get stock data from Polygon API
def get_stock_data(ticker, multiplier, timespan, from_date, to_date):
    api_key = os.getenv('POLYGON_API_KEY')
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    response = requests.get(url)
    return response.json()

# Function to capture TradingView chart using Selenium
def capture_tradingview_chart(ticker, timespan, from_date, to_date):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)

    try:
        # Convert timespan to TradingView format
        tv_timespan = {
            "minute": "1",
            "hour": "60",
            "day": "D",
            "week": "W",
            "month": "M"
        }.get(timespan.lower(), "D")

        url = f"https://www.tradingview.com/chart/?symbol={ticker}&interval={tv_timespan}&range={from_date}/{to_date}"
        driver.get(url)

        # Wait for the chart to load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CLASS_NAME, "chart-markup-table"))
        )

        # Give extra time for the chart to render completely
        driver.implicitly_wait(5)

        # Generate a unique filename
        filename = f"{ticker}_{uuid.uuid4()}.png"
        filepath = os.path.join(IMAGES_DIR, filename)

        # Capture the screenshot and save it
        driver.save_screenshot(filepath)
        
        return filename
    finally:
        driver.quit()

# Function to keep connection alive
async def keep_alive():
    await asyncio.sleep(300)  # Sleep for 5 minutes (300 seconds)

@app.post("/analyze_stock")
async def analyze_stock(request: Request, stock_request: StockRequest, background_tasks: BackgroundTasks):
    ticker = stock_request.ticker.upper()  # Convert ticker to uppercase
    multiplier = stock_request.multiplier
    timespan = stock_request.timespan
    from_date = stock_request.from_date
    to_date = stock_request.to_date

    # Get stock data
    stock_data = get_stock_data(ticker, multiplier, timespan, from_date, to_date)

    if 'results' not in stock_data or not stock_data['results']:
        raise HTTPException(status_code=404, detail=f"No data available for {ticker} from {from_date} to {to_date}. Please check your date range and ensure it's not in the future.")

    # Capture TradingView chart
    chart_filename = capture_tradingview_chart(ticker, timespan, from_date, to_date)

    # Prepare prompt for GPT-4o
    analysis_header = f"TTG AI - MARI Stock Chart Analysis for: {ticker} on a {multiplier} {timespan} chart from {from_date} to {to_date}\n\n"
    
    # Include aggregate bar data in the prompt
    aggregate_data = "\n".join([f"Date: {datetime.fromtimestamp(bar['t']/1000).strftime('%Y-%m-%d')}, Open: {bar['o']}, High: {bar['h']}, Low: {bar['l']}, Close: {bar['c']}, Volume: {bar['v']}" for bar in stock_data['results']])
    
    prompt = f"""
    {analysis_header}
    Analyze the following stock data for {ticker} from {from_date} to {to_date}:
    
    Aggregate Bar Data:
    {aggregate_data}
    
    A TradingView chart of this stock has been generated. Please provide insights on the stock's performance, 
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

        # Get the base URL
        base_url = str(request.base_url)

        # Prepare the markdown output
        markdown_output = f"""
{analysis}

![{ticker} TradingView Chart]({base_url}images/{chart_filename})
        """

        # Return the markdown output
        return {"markdown_output": markdown_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while getting AI analysis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
