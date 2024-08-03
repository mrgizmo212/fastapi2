import os
import sys
import requests
import json
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import uvicorn

from gpt_researcher import GPTResearcher
import asyncio
import argparse

# Load environment variables
load_dotenv()

# Set up FastAPI app
app = FastAPI()

class Query(BaseModel):
    query: str
    report_type: str = "research_report"
    date: datetime | None = None
    sources: list[str] = []

    @field_validator('date', mode='before')
    def parse_date(cls, value):
        if isinstance(value, str):
            try:
                utc_time = datetime.fromisoformat(value.rstrip('Z')).replace(tzinfo=ZoneInfo("UTC"))
                et_time = utc_time.astimezone(ZoneInfo("America/New_York"))
                print(f"Original UTC time: {utc_time}, Converted ET time: {et_time}")
                return et_time
            except ValueError as e:
                print(f"Date parsing error: {e}")
                raise ValueError("Invalid date format")
        return value

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

def get_current_time_et():
    return datetime.now(ZoneInfo("America/New_York"))

def get_referenced_date(query: str, current_date: datetime) -> datetime:
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    query_lower = query.lower()
   
    for i, day in enumerate(days):
        if day in query_lower:
            days_diff = (current_date.weekday() - i) % 7
            return (current_date - timedelta(days=days_diff)).replace(hour=0, minute=0, second=0, microsecond=0)
   
    if "yesterday" in query_lower:
        return (current_date - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif "today" in query_lower:
        return current_date.replace(hour=0, minute=0, second=0, microsecond=0)
   
    return current_date.replace(hour=0, minute=0, second=0, microsecond=0)

async def get_report(query: str, report_type: str, sources: list, query_date: datetime = None) -> tuple:
    start_time = time.time()
    
    if query_date:
        current_date = query_date
    else:
        current_date = get_current_time_et()
   
    referenced_date = get_referenced_date(query, current_date)
   
    date_context = f"As of {current_date.strftime('%A, %B %d, %Y %I:%M %p ET')}, "
    if referenced_date.date() != current_date.date():
        date_context += f"regarding {referenced_date.strftime('%A, %B %d, %Y')}, "
    contextualized_query = date_context + query

    researcher = GPTResearcher(query=contextualized_query, report_type=report_type, source_urls=sources)
    await researcher.conduct_research()
    report = await researcher.write_report()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return report, execution_time

@app.post("/research")
async def research(query: Query):
    print(f"Received query: {query}")
    report, execution_time = await get_report(query.query, query.report_type, query.sources, query.date)
    return {"report": report, "date": query.date, "execution_time": execution_time}

@app.post("/research_direct")
async def research_direct(query: str, report_type: str = "research_report", sources: list = []):
    start_time = time.time()
    researcher = GPTResearcher(query=query, report_type=report_type, source_urls=sources)
    await researcher.conduct_research()
    report = await researcher.write_report()
    end_time = time.time()
    execution_time = end_time - start_time
    return {"report": report, "execution_time": execution_time}

def run_fastapi():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)

def run_terminal():
    parser = argparse.ArgumentParser(description="GPT Researcher")
    parser.add_argument("query", type=str, help="Research query")
    parser.add_argument("--report_type", type=str, default="research_report", help="Type of report")
    parser.add_argument("--date", type=lambda d: datetime.fromisoformat(d).replace(tzinfo=ZoneInfo("America/New_York")),
                        help="Query date (YYYY-MM-DD HH:MM:SS in America/New_York)")
    parser.add_argument("--sources", nargs='+', default=[], help="List of source URLs")
    args = parser.parse_args()
    report, execution_time = asyncio.run(get_report(args.query, args.report_type, args.sources, args.date))
    print(f"Report:\n{report}")
    print(f"Execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_terminal()
    else:
        run_fastapi()
