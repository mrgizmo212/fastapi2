import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Depends, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import aiohttp
import asyncio
import xml.etree.ElementTree as ET
import json
import pytz
import uvicorn

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Security
security = HTTPBearer()
API_KEY = os.getenv("API_KEY")

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API Key not configured")
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return credentials.credentials

# Pydantic models
class EarningsItem(BaseModel):
    id: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    ticker: Optional[str] = None
    exchange: Optional[str] = None
    name: Optional[str] = None
    currency: Optional[str] = None
    eps: Optional[float] = Field(None, description="Earnings per share")
    eps_est: Optional[float] = Field(None, description="Estimated earnings per share")
    eps_prior: Optional[float] = Field(None, description="Prior earnings per share")
    revenue: Optional[float] = Field(None, description="Revenue")
    revenue_est: Optional[float] = Field(None, description="Estimated revenue")
    revenue_prior: Optional[float] = Field(None, description="Prior revenue")

class EarningsResponse(BaseModel):
    earnings: List[EarningsItem]

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL")
EARNINGS_API_KEY = os.getenv("EARNINGS_API_KEY")

def get_ny_time():
    ny_tz = pytz.timezone('America/New_York')
    return datetime.now(ny_tz)

def parse_float(value: str) -> Optional[float]:
    try:
        return float(value) if value else None
    except ValueError:
        return None

def parse_xml_to_earnings(xml_string: str) -> List[EarningsItem]:
    root = ET.fromstring(xml_string)
    earnings_list = []

    for item in root.findall('.//item'):
        earnings_item = EarningsItem(
            id=item.find('id').text if item.find('id') is not None else None,
            date=item.find('date').text if item.find('date') is not None else None,
            time=item.find('time').text if item.find('time') is not None else None,
            ticker=item.find('ticker').text if item.find('ticker') is not None else None,
            exchange=item.find('exchange').text if item.find('exchange') is not None else None,
            name=item.find('name').text if item.find('name') is not None else None,
            currency=item.find('currency').text if item.find('currency') is not None else None,
            eps=parse_float(item.find('eps').text if item.find('eps') is not None else None),
            eps_est=parse_float(item.find('eps_est').text if item.find('eps_est') is not None else None),
            eps_prior=parse_float(item.find('eps_prior').text if item.find('eps_prior') is not None else None),
            revenue=parse_float(item.find('revenue').text if item.find('revenue') is not None else None),
            revenue_est=parse_float(item.find('revenue_est').text if item.find('revenue_est') is not None else None),
            revenue_prior=parse_float(item.find('revenue_prior').text if item.find('revenue_prior') is not None else None)
        )
        earnings_list.append(earnings_item)

    return earnings_list

def parse_json_to_earnings(json_data: Dict[str, Any]) -> List[EarningsItem]:
    earnings_list = []
    for item in json_data.get('earnings', []):
        earnings_item = EarningsItem(
            id=item.get('id'),
            date=item.get('date'),
            time=item.get('time'),
            ticker=item.get('ticker'),
            exchange=item.get('exchange'),
            name=item.get('name'),
            currency=item.get('currency'),
            eps=parse_float(item.get('eps')),
            eps_est=parse_float(item.get('eps_est')),
            eps_prior=parse_float(item.get('eps_prior')),
            revenue=parse_float(item.get('revenue')),
            revenue_est=parse_float(item.get('revenue_est')),
            revenue_prior=parse_float(item.get('revenue_prior'))
        )
        earnings_list.append(earnings_item)
    return earnings_list

async def fetch_earnings_data(params: Dict[str, Any]) -> List[EarningsItem]:
    url = f"{API_BASE_URL}/calendar/earnings"
    params["token"] = EARNINGS_API_KEY
    
    logger.info(f"Fetching earnings data from Benzinga API with params: {params}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            content = await response.text()
            content_type = response.headers.get('Content-Type', '')
            
            logger.info(f"Received response from Benzinga API. Content-Type: {content_type}")
            if 'xml' in content_type:
                logger.info("Parsing XML response")
                return parse_xml_to_earnings(content)
            else:
                logger.info("Parsing JSON response")
                return parse_json_to_earnings(json.loads(content))

async def get_earnings_data(date_from: str, date_to: str, tickers: Optional[str] = None, importance: int = 4, pagesize: int = 1000) -> List[EarningsItem]:
    params = {
        "parameters[date_from]": date_from,
        "parameters[date_to]": date_to,
        "pagesize": pagesize,
        "page": 0,  # Start with the first page
        "parameters[importance]": importance
    }
    if tickers:
        params["parameters[tickers]"] = tickers

    logger.info(f"Calling get_earnings_data with params: {params}")
    earnings_data = await fetch_earnings_data(params)
    
    # Filter out OTC exchanges
    filtered_earnings = [item for item in earnings_data if item.exchange != "OTC"]
    
    return filtered_earnings

@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.get("/earnings/next90days", response_model=EarningsResponse)
async def get_next_90_days_earnings(
    date_from: Optional[str] = Query(None, description="Start date for earnings query (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date for earnings query (YYYY-MM-DD)"),
    tickers: Optional[str] = Query(None, description="Comma-separated list of tickers to filter by"),
    importance: int = Query(4, ge=0, le=5, description="The importance level to filter by (default: 4)"),
    api_key: str = Depends(verify_api_key)
):
    try:
        ny_time = get_ny_time()
        current_date = ny_time.date()

        if not date_from:
            date_from = current_date.strftime("%Y-%m-%d")
        if not date_to:
            date_to = (current_date + timedelta(days=90)).strftime("%Y-%m-%d")

        logger.info(f"Fetching earnings from {date_from} to {date_to}")

        earnings_data = await get_earnings_data(date_from, date_to, tickers, importance)
        
        # Sort earnings by date and time
        earnings_data.sort(key=lambda x: (
            datetime.strptime(x.date, "%Y-%m-%d").date() if x.date else datetime.max.date(),
            datetime.strptime(x.time, "%H:%M:%S").time() if x.time else datetime.max.time()
        ))
        
        logger.info(f"Returning {len(earnings_data)} earnings entries")
        return EarningsResponse(earnings=earnings_data)

    except aiohttp.ClientError as e:
        logger.error(f"API request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data from external API: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in get_next_90_days_earnings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/calendar/earnings", response_model=EarningsResponse)
async def get_earnings(
    page: int = Query(0, description="Page offset"),
    pagesize: int = Query(1000, ge=10, le=1000, description="Number of results returned (min 10, max 1000)"),
    date_from: Optional[str] = Query(None, description="Date to query from point in time (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Date to query to point in time (YYYY-MM-DD)"),
    importance: int = Query(4, ge=0, le=5, description="The importance level to filter by (default: 4)"),
    tickers: Optional[str] = Query(None, description="Comma-separated list of tickers to filter by"),
    response_format: str = Query("json", description="Response format: 'xml' or 'json'"),
    api_key: str = Depends(verify_api_key)
):
    params = {
        "page": page,
        "pagesize": pagesize,
        "token": EARNINGS_API_KEY,
        "format": response_format,
        "parameters[importance]": importance
    }

    if date_from:
        params["parameters[date_from]"] = date_from
    if date_to:
        params["parameters[date_to]"] = date_to
    if tickers:
        params["parameters[tickers]"] = tickers

    try:
        earnings_list = await fetch_earnings_data(params)
        
        # Filter out OTC exchanges
        earnings_list = [item for item in earnings_list if item.exchange != "OTC"]
        
        # Sort earnings by date and time
        earnings_list.sort(key=lambda x: (
            datetime.strptime(x.date, "%Y-%m-%d").date() if x.date else datetime.max.date(),
            datetime.strptime(x.time, "%H:%M:%S").time() if x.time else datetime.max.time()
        ))
        
        return EarningsResponse(earnings=earnings_list)

    except aiohttp.ClientError as e:
        logger.error(f"API request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data from external API: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in get_earnings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
