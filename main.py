import os
import sys
import requests
import json
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Load API keys from .env file
load_dotenv()
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set OpenAI API key
client = OpenAI(api_key=OPENAI_API_KEY)

# Create FastAPI app
app = FastAPI()

# Pydantic model for request
class StockRequest(BaseModel):
    ticker: str

# Get current date and date 30 days ago
current_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

def fetch_stock_data(ticker):
    base_url = "https://api.polygon.io"
    headers = {"Authorization": f"Bearer {POLYGON_API_KEY}"}

    # Fetch aggregates data
    aggs_url = f"{base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{current_date}"
    aggs_data = requests.get(aggs_url, headers=headers).json()

    # Fetch company info
    company_info_url = f"{base_url}/v3/reference/tickers/{ticker}"
    company_info_data = requests.get(company_info_url, headers=headers).json()

    # Fetch last quote
    last_quote_url = f"{base_url}/v2/last/nbbo/{ticker}"
    last_quote_data = requests.get(last_quote_url, headers=headers).json()

    try:
        previous_close = aggs_data['results'][-2]['c'] if len(aggs_data['results']) > 1 else None
        opening_price = aggs_data['results'][-1]['o'] if aggs_data['results'] else None
        current_price = last_quote_data['results']['P'] if 'results' in last_quote_data else None
        current_volume = aggs_data['results'][-1]['v'] if aggs_data['results'] else None
        average_volume = sum(day['v'] for day in aggs_data['results']) / len(aggs_data['results']) if aggs_data['results'] else None
        intraday_high = aggs_data['results'][-1]['h'] if aggs_data['results'] else None
        intraday_low = aggs_data['results'][-1]['l'] if aggs_data['results'] else None
        company_name = company_info_data['results']['name'] if 'results' in company_info_data else 'N/A'
    except (KeyError, IndexError):
        return None

    return {
        "previous_close": previous_close,
        "opening_price": opening_price,
        "current_price": current_price,
        "current_volume": current_volume,
        "average_volume": average_volume,
        "intraday_high": intraday_high,
        "intraday_low": intraday_low,
        "company_name": company_name,
    }

def fetch_news(ticker):
    news_url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&apiKey={POLYGON_API_KEY}"
    news_data = requests.get(news_url).json()

    news_headlines = []
    if 'results' in news_data:
        for article in news_data['results'][:5]:  # Limit to 5 most recent headlines
            article_date = article['published_utc'].split('T')[0]
            news_headlines.append(f"{article_date}: \"{article['title']}\" ({article['article_url']})")
    return "\n".join(news_headlines)

def generate_rationale(ticker, stock_data, news_headlines):
    messages = [
        {"role": "system", "content": "You are a financial analyst. Provide a comprehensive analysis based on the given data."},
        {"role": "user", "content": f"""
        Analyze the following stock performance data and news headlines for ticker {ticker}:

        Price Metrics:
        - Previous Close Price: ${stock_data['previous_close']}
        - Opening Price: ${stock_data['opening_price']}
        - Current Price: ${stock_data['current_price']}

        Volume Metrics:
        - Current Volume: {stock_data['current_volume']} shares
        - Average Volume: {stock_data['average_volume']} shares

        Intraday Metrics:
        - Intraday High: ${stock_data['intraday_high']}
        - Intraday Low: ${stock_data['intraday_low']}

        News Headlines:
        {news_headlines}

        Provide a comprehensive rationale for whether to Buy, Sell, or Hold this stock, considering all the provided data. Include an analysis of the stock's recent performance, the impact of news, and any potential risks or opportunities. Conclude with a clear recommendation.
        """}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=4096,
    )

    return response.choices[0].message.content

def main(ticker):
    # Fetch initial stock data
    stock_data = fetch_stock_data(ticker)
    if stock_data is None:
        return {"error": "Failed to fetch stock data. Please check the ticker symbol and try again."}

    # Fetch news headlines
    news_headlines = fetch_news(ticker)

    # Generate rationale
    rationale = generate_rationale(ticker, stock_data, news_headlines)

    # Generate the final report
    report = {
        "ticker": ticker,
        "company_name": stock_data['company_name'],
        "price_metrics": {
            "previous_close": stock_data['previous_close'],
            "opening_price": stock_data['opening_price'],
            "current_price": stock_data['current_price']
        },
        "volume_metrics": {
            "current_volume": stock_data['current_volume'],
            "average_volume": stock_data['average_volume']
        },
        "intraday_metrics": {
            "high": stock_data['intraday_high'],
            "low": stock_data['intraday_low']
        },
        "news_headlines": news_headlines,
        "analysis": rationale
    }

    return report

@app.post("/analyze")
async def analyze_stock(request: StockRequest):
    try:
        result = main(request.ticker)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # Run as FastAPI server
        uvicorn.run(app, host="127.0.0.1", port=8000)
    elif len(sys.argv) == 2:
        # Run as CLI
        ticker = sys.argv[1]
        print(json.dumps(main(ticker)))
    else:
        print(json.dumps({"error": "Usage: python stock_analysis.py <ticker> or python stock_analysis.py api"}))
        sys.exit(1)
