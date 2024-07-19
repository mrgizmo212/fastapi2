import json
import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Load environment variables
load_dotenv()

app = FastAPI()

class ScreenerRequest(BaseModel):
    filters: Optional[List[str]] = []
    signal: Optional[str] = None
    sort_by: Optional[str] = "ticker"
    sort_direction: Optional[str] = "asc"

class ScreenerResponse(BaseModel):
    headers: List[str]
    rows: List[List[str]]

class FinvizScreenerAction:
    def __init__(self):
        self.base_url = "https://finviz-screener-elite-adam-heimann.p.rapidapi.com"
        self.headers = {
            "X-RapidAPI-Key": os.getenv("RAPIDAPI_KEY"),
            "X-RapidAPI-Host": "finviz-screener-elite-adam-heimann.p.rapidapi.com"
        }
        self.load_filters()

    def load_filters(self):
        try:
            with open('finviz_filters.json', 'r', encoding='utf-8') as f:
                self.filters = json.load(f)
            print("Filters loaded successfully.")
        except Exception as e:
            print(f"Error loading filters: {e}")
            exit(1)

    def run_query(self, filters, signal, sort_by, sort_direction):
        params = {}
        if filters:
            params["filters"] = ",".join(filters)
        if signal:
            params["signal"] = signal
        
        params["order"] = sort_by
        params["desc"] = str(sort_direction.lower() == "desc").lower()

        try:
            response = requests.get(f"{self.base_url}/table", headers=self.headers, params=params)
            response.raise_for_status()
            results = response.json()
            return results.get('headers', []), results.get('rows', [])
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error accessing Finviz API: {str(e)}")

finviz_action = FinvizScreenerAction()

@app.post("/screen", response_model=ScreenerResponse)
async def screen_stocks(request: ScreenerRequest):
    headers, rows = finviz_action.run_query(
        request.filters,
        request.signal,
        request.sort_by,
        request.sort_direction
    )
    return ScreenerResponse(headers=headers, rows=rows)

@app.get("/filters")
async def get_filters():
    return finviz_action.filters['filters']

@app.get("/signals")
async def get_signals():
    return finviz_action.filters.get('signals', [])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
