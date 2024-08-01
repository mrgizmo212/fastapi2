from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio as aioredis
from datetime import datetime
import pandas as pd
from typing import List
from pydantic import BaseModel
import numpy as np
from backend.schemas.stock import StockData, CompanyFinancials, MarketNews
from backend.services.model import train_lstm_model, predict_next_day, save_model, load_saved_model
from backend.services.data_fetcher import DataFetcher
from backend.services.preprocessing import preprocess_data
from backend.utils.logging import setup_logging, logger

app = FastAPI()

setup_logging()

data_fetcher = DataFetcher()

MODEL_PATH = "lstm_model.h5"
SCALER_PATH = "lstm_scaler.pkl"

class PredictionRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str

class PredictionResponse(BaseModel):
    ticker: str
    predicted_price: float

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost", encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

# ... (rest of the code remains the same)

@app.get("/stocks/{ticker}", response_model=List[StockData])
@cache(expire=3600)
async def get_stock_data(ticker: str, start_date: str, end_date: str):
    try:
        current_date = datetime.now().date()
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        if end_date_obj > current_date:
            raise HTTPException(status_code=400, detail=f"End date cannot be in the future. Current date is {current_date}")

        data = data_fetcher.get_stock_data(ticker, start_date, end_date)
        if data is None or len(data) == 0:
            raise HTTPException(status_code=404, detail=f"No stock data found for {ticker} between {start_date} and {end_date}")
        return data
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/train/{ticker}")
async def train_model(ticker: str, start_date: str, end_date: str, background_tasks: BackgroundTasks):
    try:
        current_date = datetime.now().date()
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        if end_date_obj > current_date:
            raise HTTPException(status_code=400, detail=f"End date cannot be in the future. Current date is {current_date}")

        stock_data = data_fetcher.get_stock_data(ticker, start_date, end_date)
        if stock_data is None or len(stock_data) == 0:
            raise HTTPException(status_code=404, detail=f"No stock data found for {ticker} between {start_date} and {end_date}")
        
        logger.info(f"Retrieved {len(stock_data)} stock data points")

        company_data = data_fetcher.get_company_financials(ticker)
        if company_data is None:
            logger.warning(f"No company data found for {ticker}")

        news_data = data_fetcher.get_ticker_news(ticker)
        if news_data is None:
            logger.warning(f"No news data found for {ticker}")
            news_data = []

        logger.info(f"Retrieved {len(news_data)} news items")
        
        df = preprocess_data(stock_data, company_data, news_data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Preprocessed data is empty")
        
        if len(df) < 60:  # Assuming we need at least 60 data points for meaningful training
            raise HTTPException(status_code=400, detail=f"Insufficient data points after preprocessing. Got {len(df)}, need at least 60.")
        
        logger.info(f"Preprocessed data shape: {df.shape}")
        
        background_tasks.add_task(train_and_save_model, df)
        
        return {"message": f"Model training for {ticker} started in the background"}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error initiating model training for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock_price(request: PredictionRequest):
    try:
        model, scaler = load_saved_model(MODEL_PATH, SCALER_PATH)
        if model is None or scaler is None:
            raise HTTPException(status_code=404, detail="Model not found. Please train the model first.")
        
        stock_data = data_fetcher.get_stock_data(request.ticker, request.start_date, request.end_date)
        company_data = data_fetcher.get_company_financials(request.ticker)
        news_data = data_fetcher.get_ticker_news(request.ticker)
        
        df = preprocess_data(stock_data, company_data, news_data)
        logger.info(f"Preprocessed data shape: {df.shape}")
        logger.info(f"Preprocessed data sample:\n{df.head()}")
        
        predicted_price = predict_next_day(model, scaler, df)
        
        if pd.isna(predicted_price) or np.isinf(predicted_price):
            logger.warning(f"Invalid prediction: {predicted_price}. Using last known price.")
            predicted_price = df['c'].iloc[-1]  # Use the last known closing price
        
        logger.info(f"Predicted price for {request.ticker}: {predicted_price}")
        
        return PredictionResponse(ticker=request.ticker, predicted_price=float(predicted_price))
    except Exception as e:
        logger.error(f"Error predicting stock price: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/company/{ticker}", response_model=CompanyFinancials)
async def get_company_financials(ticker: str):
    try:
        financials = data_fetcher.get_company_financials(ticker)
        if financials is None:
            raise HTTPException(status_code=404, detail=f"Company details not found for {ticker}")
        return financials
    except Exception as e:
        logger.error(f"Error fetching company details for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/news/{ticker}", response_model=List[MarketNews])
async def get_ticker_news(ticker: str, limit: int = Query(10, ge=1, le=100)):
    try:
        news = data_fetcher.get_ticker_news(ticker, limit)
        if news is None or len(news) == 0:
            raise HTTPException(status_code=404, detail=f"No news found for ticker {ticker}")
        return news
    except Exception as e:
        logger.error(f"Error fetching news for ticker {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/news", response_model=List[MarketNews])
async def get_market_news(limit: int = Query(10, ge=1, le=100)):
    try:
        news = data_fetcher.get_market_news(limit)
        if news is None or len(news) == 0:
            raise HTTPException(status_code=404, detail="No market news available")
        return news
    except Exception as e:
        logger.error(f"Error fetching market news: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def train_and_save_model(df: pd.DataFrame):
    try:
        if df.empty:
            logger.error("Empty DataFrame provided for model training")
            return
        
        model, scaler = train_lstm_model(df)
        save_model(model, scaler, MODEL_PATH, SCALER_PATH)
        logger.info("Model training completed and saved successfully")
    except Exception as e:
        logger.error(f"Error during model training and saving: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
