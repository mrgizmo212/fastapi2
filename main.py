import asyncio
import statistics
from datetime import datetime
from typing import List, Dict, Tuple

import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Constants
API_KEY = 'kQWDqA0jsVPfH5vj5dnSiHH5j7HPFw6w'  # Replace with your Polygon.io API key
BASE_URL = 'https://api.polygon.io'
AUTH_TOKEN = 'Y0buhusiLO2AXc2yCWTlLeCNh9XnfW4stQ72tLpQ3QZK'
TICKERS_URL = 'https://ttg-triangle.sliplane.app/mid-to-mega-ttg-triangle'

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Helper Functions
async def get_tickers(session: aiohttp.ClientSession, url: str, auth_token: str) -> Dict:
    headers = {'Authorization': f'Bearer {auth_token}'}
    async with session.get(url, headers=headers) as response:
        if response.status != 200:
            text = await response.text()
            raise HTTPException(status_code=response.status, detail=f"Failed to get tickers: {text}")
        data = await response.json()
    return data

async def get_intraday_data(session: aiohttp.ClientSession, ticker: str) -> List[Dict]:
    date = datetime.now().strftime('%Y-%m-%d')
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/3/minute/{date}/{date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 5000,
        'apiKey': API_KEY
    }
    async with session.get(url, params=params) as response:
        if response.status != 200:
            text = await response.text()
            raise HTTPException(status_code=response.status, detail=f"Error fetching data for {ticker}: {text}")
        data = await response.json()
    if data.get('resultsCount', 0) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    return data['results']

def evaluate_ttgt_setup(ticker: str, data: List[Dict]) -> Tuple[int, Dict]:
    if not data or len(data) < 20:
        return 0, {'error': 'Not enough data to analyze'}

    closing_prices = [bar['c'] for bar in data]
    high_prices = [bar['h'] for bar in data]
    low_prices = [bar['l'] for bar in data]
    volumes = [bar['v'] for bar in data]
    timestamps = [datetime.fromtimestamp(bar['t'] / 1000) for bar in data]

    mid_index = len(high_prices) // 2
    previous_high = max(high_prices[:mid_index])
    score = 0
    details = {'previous_high': previous_high}

    # Detect breakout
    breakout_index = next((i for i in range(mid_index, len(high_prices))
                           if high_prices[i] > previous_high and volumes[i] > statistics.mean(volumes[:mid_index]) * 1.5), None)
    if breakout_index is None:
        details['breakout'] = False
        return score, details

    details.update({
        'breakout': True,
        'breakout_time': timestamps[breakout_index].strftime('%H:%M'),
        'breakout_price': high_prices[breakout_index],
        'breakout_volume': volumes[breakout_index]
    })
    score += 5

    # Check for pullback
    pullback_start = breakout_index + 1
    pullback_data = [(high_prices[i], low_prices[i], volumes[i]) 
                     for i in range(pullback_start, len(high_prices)) 
                     if low_prices[i] <= previous_high]
    
    if pullback_data:
        details['pullback'] = True
        pullback_volumes = [v for _, _, v in pullback_data]
        details['pullback_volume'] = 'Decreasing' if pullback_volumes == sorted(pullback_volumes, reverse=True) else 'Not Decreasing'
        score += 5 if details['pullback_volume'] == 'Decreasing' else 0

        # Check for descending triangle
        pullback_highs = [h for h, _, _ in pullback_data]
        pullback_lows = [l for _, l, _ in pullback_data]
        if all(pullback_highs[i] < pullback_highs[i-1] for i in range(1, len(pullback_highs))) and \
           max(pullback_lows) - min(pullback_lows) < 0.005 * previous_high:
            details['descending_triangle'] = True
            score += 5
        else:
            details['descending_triangle'] = False

        # Verify support levels
        if min(pullback_lows) >= previous_high:
            details['support_level_holding'] = True
            score += 5
        else:
            details['support_level_holding'] = False

        # Volume analysis
        if max(pullback_volumes) < volumes[breakout_index]:
            details['volume_analysis'] = 'Pullback volume less than breakout volume'
            score += 5
        else:
            details['volume_analysis'] = 'Pullback volume not less than breakout volume'
    else:
        details['pullback'] = False

    details['final_score'] = score
    return score, details

# Main analysis functions
async def analyze_tickers(tickers: List[str]) -> Dict[str, Dict]:
    async with aiohttp.ClientSession() as session:
        tasks = [evaluate_ticker(session, ticker.upper()) for ticker in tickers]
        results = await asyncio.gather(*tasks)
    return dict(zip(tickers, results))

async def evaluate_ticker(session: aiohttp.ClientSession, ticker: str) -> Dict:
    try:
        data = await get_intraday_data(session, ticker)
        score, details = evaluate_ttgt_setup(ticker, data)
        return {'score': score, 'details': details}
    except Exception as e:
        return {'error': str(e)}

# API Endpoints
@app.get("/fetch_and_analyze_tickers")
async def fetch_and_analyze_tickers_endpoint():
    try:
        async with aiohttp.ClientSession() as session:
            data = await get_tickers(session, TICKERS_URL, AUTH_TOKEN)
        
        if not isinstance(data, dict) or 'rows' not in data:
            raise ValueError(f"Unexpected data format: {data}")
        
        tickers = [row[1] for row in data['rows'] if len(row) > 1]
        analysis_results = await analyze_tickers(tickers)
        
        sorted_results = dict(sorted(analysis_results.items(), key=lambda item: item[1].get('score', 0), reverse=True))
        
        for row in data['rows']:
            ticker = row[1] if len(row) > 1 else None
            if ticker in sorted_results:
                row.append(sorted_results[ticker])
        
        data['headers'].append("TTG Triangle Analysis")
        
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching and analyzing tickers: {str(e)}")

# Run the application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
