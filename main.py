import asyncio
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Tuple
import aiohttp
from datetime import datetime
import statistics
import argparse

app = FastAPI()

API_KEY = 'kQWDqA0jsVPfH5vj5dnSiHH5j7HPFw6w'  # Replace with your Polygon.io API key
BASE_URL = 'https://api.polygon.io'

AUTH_TOKEN = 'Y0buhusiLO2AXc2yCWTlLeCNh9XnfW4stQ72tLpQ3QZK'
TICKERS_URL = 'https://ttg-triangle.sliplane.app/mid-to-mega-ttg-triangle'

async def get_tickers(session: aiohttp.ClientSession, url: str, auth_token: str) -> Dict:
    headers = {'Authorization': f'Bearer {auth_token}'}
    async with session.get(url, headers=headers) as response:
        if response.status != 200:
            text = await response.text()
            raise Exception(f"Failed to get tickers from {url}: {text}")
        data = await response.json()
    return data

async def fetch_and_analyze_tickers():
    """
    Fetch tickers and automatically analyze them using the TTG Triangle setup.
    """
    try:
        async with aiohttp.ClientSession() as session:
            data = await get_tickers(session, TICKERS_URL, AUTH_TOKEN)
        
        if not isinstance(data, dict) or 'rows' not in data:
            raise ValueError(f"Unexpected data format: {data}")
        
        # Extract tickers from the 'rows' data
        tickers = [row[1] for row in data['rows'] if len(row) > 1]
        
        # Analyze the tickers
        analysis_results = await analyze_tickers(tickers)
        
        # Sort results by score
        sorted_results = dict(sorted(analysis_results.items(), key=lambda item: item[1].get('score', 0), reverse=True))
        
        # Combine the original data with the analysis results
        for row in data['rows']:
            ticker = row[1] if len(row) > 1 else None
            if ticker in sorted_results:
                row.append(sorted_results[ticker])
        
        # Add a new header for the analysis results
        data['headers'].append("TTG Triangle Analysis")
        
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching and analyzing tickers: {str(e)}")

@app.get("/fetch_and_analyze_tickers")
async def fetch_and_analyze_tickers_endpoint():
    """
    Fetch tickers, analyze them, and return the combined results.
    """
    return await fetch_and_analyze_tickers()

async def analyze_tickers(tickers: List[str]) -> Dict[str, Dict]:
    results = {}
    tickers = [ticker.upper() for ticker in tickers]  # Ensure tickers are uppercase
    async with aiohttp.ClientSession() as session:
        tasks = [evaluate_ticker(session, ticker) for ticker in tickers]
        ticker_results = await asyncio.gather(*tasks)
    for ticker, result in zip(tickers, ticker_results):
        results[ticker] = result
    return results

async def evaluate_ticker(session: aiohttp.ClientSession, ticker: str) -> Dict:
    """
    Evaluate TTG Triangle setup for a single ticker intraday.
    """
    try:
        print(f"Analyzing {ticker}...")
        data = await get_intraday_data(session, ticker)
        score, details = evaluate_ttgt_setup(ticker, data)
        return {'score': score, 'details': details}
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return {'error': str(e)}

async def get_intraday_data(session: aiohttp.ClientSession, ticker: str) -> List[Dict]:
    """
    Retrieve intraday price and volume data for the given ticker.
    """
    date = datetime.now().strftime('%Y-%m-%d')
    multiplier = 3  # 3-minute bars
    timespan = 'minute'

    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{date}/{date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 5000,
        'apiKey': API_KEY
    }
    async with session.get(url, params=params) as response:
        if response.status != 200:
            text = await response.text()
            raise HTTPException(status_code=500, detail=f"Error fetching data for {ticker}: {text}")
        data = await response.json()
    if data.get('resultsCount', 0) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    return data['results']

def evaluate_ttgt_setup(ticker: str, data: List[Dict]) -> Tuple[int, Dict]:
    """
    Evaluate the TTG Triangle setup intraday for the given data.
    Return a score and details based on how well it meets the criteria.
    """
    score = 0
    details = {}
    if not data or len(data) < 20:
        details['error'] = 'Not enough data to analyze'
        return score, details

    closing_prices = [bar['c'] for bar in data]
    high_prices = [bar['h'] for bar in data]
    low_prices = [bar['l'] for bar in data]
    volumes = [bar['v'] for bar in data]
    timestamps = [datetime.fromtimestamp(bar['t'] / 1000) for bar in data]

    # 1. Identify previous highs before breakout
    # Use the first half of the day to find previous highs
    mid_index = len(high_prices) // 2
    previous_high = max(high_prices[:mid_index])
    details['previous_high'] = previous_high

    # 2. Detect breakout above previous high with increased volume
    breakout = False
    breakout_index = None
    average_volume = statistics.mean(volumes[:mid_index]) if mid_index > 0 else volumes[0]
    for i in range(mid_index, len(high_prices)):
        if high_prices[i] > previous_high and volumes[i] > average_volume * 1.5:
            breakout = True
            breakout_index = i
            breakout_price = high_prices[i]
            breakout_volume = volumes[i]
            details['breakout_time'] = timestamps[i].strftime('%H:%M')
            details['breakout_price'] = breakout_price
            details['breakout_volume'] = breakout_volume
            score += 5
            break
    if not breakout:
        details['breakout'] = False
        return score, details  # Cannot proceed without a breakout
    details['breakout'] = True

    # 3. Check for pullback toward previous high on decreasing volume
    pullback = False
    pullback_start = breakout_index + 1
    pullback_volumes = []
    pullback_highs = []
    pullback_lows = []

    for i in range(pullback_start, len(high_prices)):
        if low_prices[i] <= previous_high:
            pullback = True
            pullback_volumes.append(volumes[i])
            pullback_highs.append(high_prices[i])
            pullback_lows.append(low_prices[i])
        else:
            break  # Pullback ended
    if pullback:
        details['pullback'] = True
        # Check if volumes are decreasing during pullback
        if pullback_volumes == sorted(pullback_volumes, reverse=True):
            score += 5
            details['pullback_volume'] = 'Decreasing'
        else:
            details['pullback_volume'] = 'Not Decreasing'

        # 4. Check for descending triangle formation
        lower_highs = all(pullback_highs[i] < pullback_highs[i - 1] for i in range(1, len(pullback_highs)))
        if pullback_lows:
            horizontal_support = max(pullback_lows) - min(pullback_lows) < 0.005 * previous_high  # Adjust threshold as needed
        else:
            horizontal_support = False
        if lower_highs and horizontal_support:
            score += 5
            details['descending_triangle'] = True
        else:
            details['descending_triangle'] = False
    else:
        details['pullback'] = False

    # 5. Verify support levels
    if pullback and pullback_lows:
        support_level = min(pullback_lows)
        if support_level >= previous_high:
            score += 5
            details['support_level_holding'] = True
        else:
            details['support_level_holding'] = False
    else:
        details['support_level_holding'] = False

    # 6. Volume analysis
    # Confirm decreasing volume in pullback compared to breakout volume
    if pullback and pullback_volumes and max(pullback_volumes) < breakout_volume:
        score += 5
        details['volume_analysis'] = 'Pullback volume less than breakout volume'
    else:
        details['volume_analysis'] = 'Pullback volume not less than breakout volume'

    details['final_score'] = score

    return score, details

def main():
    parser = argparse.ArgumentParser(description='TTG Triangle Intraday Analysis')
    parser.add_argument('tickers', nargs='*', help='Ticker symbols (individual or comma-separated)')
    args = parser.parse_args()

    if args.tickers:
        # Process tickers input
        tickers_input = args.tickers
        tickers = []
        for item in tickers_input:
            tickers.extend(item.replace(',', ' ').split())
        tickers = [ticker.upper() for ticker in tickers]  # Ensure uppercase
        
        # Run analysis on provided tickers
        results = asyncio.run(analyze_tickers(tickers))
        
        # Print results
        for ticker, result in results.items():
            print(f"\nTicker: {ticker}")
            if 'score' in result:
                print(f"Score: {result['score']}")
                print("Details:")
                for key, value in result['details'].items():
                    print(f"  {key}: {value}")
            else:
                print(f"Error: {result['error']}")
    else:
        # Fetch and analyze tickers from the provided endpoint
        results = asyncio.run(fetch_and_analyze_tickers())
        print(results)

if __name__ == '__main__':
    import sys
    if 'uvicorn' in sys.modules:
        # Running with uvicorn
        pass
    elif len(sys.argv) > 1:
        main()
    else:
        # Start FastAPI app if no arguments are provided
        import uvicorn
        uvicorn.run("ttg_triangle_analysis:app", host='0.0.0.0', port=8000, reload=True)
