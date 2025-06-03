from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from utils.news_utils import fetch_sf_news, analyze_news_relevance, NewsItem
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI()

# MongoDB connection
MONGO_URL = "mongodb://localhost:27018"
client = AsyncIOMotorClient(MONGO_URL)
db = client.crime_forecast

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Crime Forecast API"}

@app.get("/api/news/crime", response_model=List[NewsItem])
async def get_crime_news(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format")
):
    """Get crime-related news for San Francisco"""
    try:
        # Convert string dates to datetime
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Validate date range
        if end_dt < start_dt:
            raise HTTPException(
                status_code=400,
                detail="End date must be after start date"
            )
        
        # Fetch news
        news_items = await fetch_sf_news(start_dt, end_dt)
        
        if not news_items:
            return []
        
        # Analyze and filter relevant news
        relevant_news = await analyze_news_relevance(news_items)
        
        return relevant_news
        
    except ValueError as e:
        if "OPENAI_API_KEY" in str(e):
            raise HTTPException(
                status_code=500,
                detail="API key configuration error"
            )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

@app.get("/predictions/heatmap")
async def get_heatmap_data():
    return {"message": "Heatmap data endpoint"}
