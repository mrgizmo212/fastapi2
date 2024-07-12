from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Optional
import requests
import datetime

API_KEY = '0cv62bvSH9Xe_qHUVgkq4AE1ha9n7E7S'
BASE_URL = 'https://api.polygon.io'

class TickerRequest(BaseModel):
    ticker: str
    expiration_date: Optional[str] = None

    @validator("expiration_date", pre=True, always=True)
    def validate_expiration_date(cls, value):
        if value is None or value.lower() == "string":
            return None
        try:
            datetime.datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid expiration date format. Use YYYY-MM-DD.")
        return value

def get_underlying_price(ticker):
    url = f'{BASE_URL}/v2/last/trade/{ticker}?apiKey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    if 'results' in data:
        return data['results']['p']
    else:
        raise HTTPException(status_code=400, detail=f"Error fetching last trade for {ticker}: {data}")

def get_options_chain(ticker):
    url = f'{BASE_URL}/v3/snapshot/options/{ticker}?limit=100&apiKey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    if 'results' in data:
        return data['results']
    else:
        raise HTTPException(status_code=400, detail=f"Error fetching options chain for {ticker}: {data}")

def filter_itm_options(options, underlying_price):
    itm_options = []
    for option in options:
        strike_price = option['details']['strike_price']
        contract_type = option['details']['contract_type']
        if (contract_type == 'call' and strike_price < underlying_price) or (contract_type == 'put' and strike_price > underlying_price):
            itm_options.append(option)
    return itm_options

def sort_by_volume(options):
    return sorted(options, key=lambda x: x['day']['volume'] if 'day' in x and 'volume' in x['day'] else 0, reverse=True)

def get_top_two_options(itm_options):
    return itm_options[:2]


def post(request: TickerRequest):
    ticker = request.ticker.upper()
    expiration_date = request.expiration_date

    try:
        underlying_price = get_underlying_price(ticker)
        options_chain = get_options_chain(ticker)

        if expiration_date:
            options_chain = [option for option in options_chain if option['details']['expiration_date'] == expiration_date]
            if not options_chain:
                raise HTTPException(status_code=404, detail="No options contracts found for the given expiration date.")
        else:
            # Sort by expiration date and select the closest one
            options_chain = sorted(options_chain, key=lambda x: x['details']['expiration_date'])
            if options_chain:
                closest_expiration_date = options_chain[0]['details']['expiration_date']
                options_chain = [option for option in options_chain if option['details']['expiration_date'] == closest_expiration_date]

        itm_options = filter_itm_options(options_chain, underlying_price)

        if not itm_options:
            raise HTTPException(status_code=404, detail="No ITM options found.")

        sorted_options = sort_by_volume(itm_options)
        top_two_options = get_top_two_options(sorted_options)

        return {
            "ticker": ticker,
            "underlying_price": underlying_price,
            "options": [
                {
                    "ticker": option['details']['ticker'],
                    "strike_price": option['details']['strike_price'],
                    "volume": option['day']['volume'] if 'day' in option and 'volume' in option['day'] else 'N/A',
                    "type": option['details']['contract_type'],
                    "expiration_date": option['details']['expiration_date'],
                    "last_trade_price": option['day']['close'] if 'day' in option and 'close' in option['day'] else 'N/A',
                    "implied_volatility": option['implied_volatility'] if 'implied_volatility' in option else 'N/A'
                } for option in top_two_options
            ]
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

