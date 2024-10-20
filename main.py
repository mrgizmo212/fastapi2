import os
import requests
import xml.etree.ElementTree as ET
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import pytz

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Existing models
class TickerInput(BaseModel):
    symbol: str

class Security(BaseModel):
    cik: Optional[str]
    exchange: Optional[str]
    isin: Optional[str]
    name: Optional[str]
    symbol: Optional[str]

class AnalystInsight(BaseModel):
    action: Optional[str]
    analyst_insights: Optional[str]
    date: Optional[str]
    firm: Optional[str]
    firm_id: Optional[str]
    id: str
    pt: Optional[str]
    rating: Optional[str]
    rating_id: Optional[str]
    security: Security
    updated: int

class AnalystInsightsResponse(BaseModel):
    analyst_insights: List[AnalystInsight]

class EconomicEvent(BaseModel):
    id: str
    date: str
    time: str
    country: str
    event_name: str
    event_period: str
    period_year: int
    actual: Optional[str]
    actual_t: Optional[str]
    consensus: Optional[str]
    consensus_t: Optional[str]
    prior: Optional[str]
    prior_t: Optional[str]
    importance: int
    updated: int
    description: str

class EconomicsResponse(BaseModel):
    economics: List[EconomicEvent]

# New model for Earnings
class EarningsData(BaseModel):
    id: str
    date: str
    date_confirmed: str
    time: str
    ticker: str
    exchange: str
    name: str
    currency: str
    period: str
    period_year: int
    eps_type: Optional[str]
    eps: Optional[str]
    eps_est: Optional[str]
    eps_prior: Optional[str]
    eps_surprise: Optional[str]
    eps_surprise_percent: Optional[str]
    revenue_type: Optional[str]
    revenue: Optional[str]
    revenue_est: Optional[str]
    revenue_prior: Optional[str]
    revenue_surprise: Optional[str]
    revenue_surprise_percent: Optional[str]
    importance: int
    notes: Optional[str]
    updated: int

class EarningsResponse(BaseModel):
    earnings: List[EarningsData]

# Utility functions
def parse_xml(xml_string: str) -> Dict[str, Any]:
    root = ET.fromstring(xml_string)
    result = {}
    for child in root:
        if child.attrib.get('is_array') == 'true':
            result[child.tag] = [parse_xml_item(item) for item in child]
        else:
            result[child.tag] = parse_xml_item(child)
    return result

def parse_xml_item(element: ET.Element) -> Dict[str, Any]:
    result = {}
    for child in element:
        if child.text is not None:
            result[child.tag] = child.text.strip()
        else:
            result[child.tag] = None
    return result

async def fetch_data(url: str, params: dict, token_env: str = "API_TOKEN") -> Dict[str, Any]:
    try:
        params["token"] = os.getenv(token_env)
        if "symbols" in params and params["symbols"]:
            params["symbols"] = params["symbols"].upper()
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        logger.info(f"Raw API response: {response.text[:200]}...")  # Log first 200 chars of response
        
        content_type = response.headers.get('Content-Type', '')
        if 'application/json' in content_type:
            return response.json()
        elif 'application/xml' in content_type or 'text/xml' in content_type:
            return parse_xml(response.text)
        else:
            logger.error(f"Unexpected content type: {content_type}")
            return {"error": "Unexpected content type", "raw_response": response.text[:1000]}
    except requests.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data from external API: {str(e)}")

def sort_economics_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort economic events by date, closest to current date first."""
    now = datetime.now(pytz.UTC)
    return sorted(events, key=lambda x: abs(datetime.strptime(x['date'], '%Y-%m-%d').replace(tzinfo=pytz.UTC) - now))

# Existing endpoints
@app.post("/bulls_bears/")
async def get_bulls_bears(ticker: TickerInput):
    url = os.getenv("API_URL")
    params = {"symbols": ticker.symbol.upper()}
    try:
        result = await fetch_data(url, params)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_bulls_bears: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

@app.get("/analyst/insights")
async def get_analyst_insights(
    page: int = Query(1, description="Page number for pagination"),
    pageSize: int = Query(10, description="Number of items per page"),
    symbols: Optional[str] = Query(None, description="Stock ticker symbols separated by commas"),
    analyst: Optional[str] = Query(None, description="One or more analyst ids separated by a comma"),
    rating_id: Optional[str] = Query(None, description="One or more rating ids separated by a comma")
):
    url = f"{os.getenv('API_BASE_URL')}/analyst/insights"
    params = {
        "page": page,
        "pageSize": pageSize,
        "symbols": symbols.upper() if symbols else None,
        "analyst": analyst,
        "rating_id": rating_id
    }
    
    try:
        result = await fetch_data(url, params)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_analyst_insights: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

@app.get("/economics")
async def get_economics(
    page: int = Query(0, description="Page offset"),
    pagesize: int = Query(1000, ge=10, le=1000, description="Number of results returned (min 10, max 1000)"),
    date_from: Optional[str] = Query(None, description="Date to query from point in time (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Date to query to point in time (YYYY-MM-DD)"),
    importance: int = Query(3, ge=0, le=5, description="The importance level to filter by"),
    updated: Optional[int] = Query(None, description="Records last Updated Unix timestamp (UTC)"),
    country: str = Query("USA", description="3-Digit Country Code"),
    event_category: str = Query("Central Banks", description="One or more categories separated by a comma")
):
    url = "https://api.benzinga.com/api/v2.1/calendar/economics"
    
    utc_now = datetime.now(pytz.UTC)
    
    if date_from is None:
        date_from = utc_now.strftime('%Y-%m-%d')
    
    if date_to is None:
        date_to = (utc_now + timedelta(days=90)).strftime('%Y-%m-%d')

    params = {
        "page": page,
        "pagesize": pagesize,
        "parameters[date_from]": date_from,
        "parameters[date_to]": date_to,
        "parameters[importance]": importance,
        "parameters[updated]": updated,
        "country": country,
        "event_category": event_category
    }
    
    try:
        result = await fetch_data(url, params, token_env="NEW_API_TOKEN")
        if 'economics' in result and isinstance(result['economics'], list):
            result['economics'] = sort_economics_events(result['economics'])
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_economics: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

# New Earnings endpoint
@app.get("/calendar/earnings", response_model=EarningsResponse)
async def get_earnings(
    page: int = Query(0, description="Page offset"),
    pagesize: int = Query(1000, ge=10, le=1000, description="Number of results returned (min 10, max 1000)"),
    date: Optional[str] = Query(None, description="Date to query for calendar data"),
    date_from: Optional[str] = Query(None, description="Date to query from point in time"),
    date_to: Optional[str] = Query(None, description="Date to query to point in time"),
    date_sort: str = Query("date", description="Field sort option for earnings calendar"),
    tickers: Optional[str] = Query(None, description="Comma-separated list of tickers to filter by"),
    importance: Optional[int] = Query(None, ge=0, le=5, description="The importance level to filter by"),
    updated: Optional[int] = Query(None, description="Records last Updated Unix timestamp (UTC)")
):
    url = "https://api.benzinga.com/api/v1/calendar/earnings"
    
    params = {
        "page": page,
        "pagesize": pagesize,
        "parameters[date]": date,
        "parameters[date_from]": date_from,
        "parameters[date_to]": date_to,
        "parameters[date_sort]": date_sort,
        "parameters[tickers]": tickers.upper() if tickers else None,
        "parameters[importance]": importance,
        "parameters[updated]": updated
    }
    
    # Remove None values from params
    params = {k: v for k, v in params.items() if v is not None}
    
    try:
        result = await fetch_data(url, params)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_earnings: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
