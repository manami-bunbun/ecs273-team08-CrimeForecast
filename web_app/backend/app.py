from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from utils.news_utils import fetch_sf_news, analyze_news_relevance
from utils.llm_advice import analyze_area_safety
from utils.data_schema import NewsItem, HeatmapData, LLMAdvice, AreaAnalysis
from dotenv import load_dotenv
import os
from utils.store_analysis import analysis_cache

# Load environment variables
load_dotenv()

app = FastAPI()

# MongoDB connection
MONGO_URL = "mongodb://localhost:27017"
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


# call this for frontend
@app.post("/api/analysis/batch/{area_name}")
async def create_area_analysis(
    area_name: str,
    latitude: float = Query(..., description="Area latitude"),
    longitude: float = Query(..., description="Area longitude"),
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format")
):
 
    try:
        # Get news data
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        if end_dt < start_dt:
            raise HTTPException(
                status_code=400,
                detail="End date must be after start date"
            )
            
        news_items = await fetch_sf_news(start_dt, end_dt)
        if news_items:
            news_items = await analyze_news_relevance(news_items)

        # Get heatmap data (mock for now)
        heatmap_data = HeatmapData(
            latitude=latitude,
            longitude=longitude,
            risk_score=0.7,  # TODO: Get actual risk score
            district="San Fransisco"  # TODO: Get actual district
        )

        # Get LLM analysis
        llm_advice = await analyze_area_safety(
            area_name=area_name,
            heatmap_data=heatmap_data,
            news_items=news_items
        )

        # Store in cache
        analysis_cache.store_analysis(
            area_name=area_name,
            start_date=start_date,
            end_date=end_date,
            news=news_items,
            advice=llm_advice
        )

        return heatmap_data, news_items, llm_advice

    except ValueError as e:
        if "OPENAI_API_KEY" in str(e):
            raise HTTPException(
                status_code=500,
                detail="API key configuration error"
            )
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )


# for visualization: call this for frontend
@app.get("/api/news", response_model=List[NewsItem])
async def get_cached_news(
    area_name: str = Query(..., description="Area name"),
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format")
):
    """Get cached news for an area"""
    news = analysis_cache.get_news(area_name, start_date, end_date)
    if news is None:
        raise HTTPException(
            status_code=404,
            detail="No cached news found. Please run analysis first."
        )
    return news

# for visualization: call this for frontend
@app.get("/api/advice", response_model=LLMAdvice)
async def get_cached_advice(
    area_name: str = Query(..., description="Area name"),
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format")
):
    """Get cached LLM advice for an area"""
    advice = analysis_cache.get_advice(area_name, start_date, end_date)
    if advice is None:
        raise HTTPException(
            status_code=404,
            detail="No cached advice found. Please run analysis first."
        )
    return advice
