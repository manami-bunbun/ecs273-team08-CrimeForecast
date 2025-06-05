from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from utils.news_utils import fetch_sf_news, analyze_news_relevance, analyze_news_relevance_gpt
from utils.data_schema import NewsItem, TrendAnalysis, LLMAnalysis, AnalysisResponse
from dotenv import load_dotenv
import os
from utils.trend_analysis import analyze_crime_trends
from utils.llm_analysis import analyze_trends_and_news
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="SF Crime Forecast API")

MONGO_URL = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_URL)
db = client.crime_forecast

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


# Get relevant news articles for the last month before the specified end date
@app.get("/api/news", response_model=List[NewsItem])
async def get_news(
    end_date: str = Query(..., description="End date in YYYY-MM-DD format")
):
    try:
        news_items = await fetch_sf_news(end_date)
        analyzed_news = await analyze_news_relevance_gpt(news_items)
        return analyzed_news[:5]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get comprehensive crime trend analysis with news integration and LLM insights
@app.get("/api/analysis", response_model=AnalysisResponse)
async def get_trend_analysis(
    end_date: str = Query(..., description="End date for analysis (YYYY-MM-DD)")
) -> AnalysisResponse:
    try:
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date_obj = end_date_obj - timedelta(days=30)
        start_date = start_date_obj.strftime("%Y-%m-%d")

        logger.info(f"Fetching crime data from {start_date} to {end_date}")

        crime_data = []
        async for doc in db.incidents.find({
            "incident_datetime": {
                "$gte": start_date_obj,
                "$lte": end_date_obj
            }
        }):
            crime_data.append(doc)
            
        if not crime_data:
            raise HTTPException(
                status_code=404,
                detail="No crime data found for the specified time range"
            )
            
        df = pd.DataFrame(crime_data)
        
        try:
            # Get news data
            news_items = await fetch_sf_news(end_date)
            
            # Get trend analysis
            trend_data = await analyze_crime_trends(df, news_items=news_items)
            
            # Get LLM analysis
            llm_analysis = await analyze_trends_and_news(trend_data)
            
            return AnalysisResponse(
                trends=trend_data.model_dump(),
                news=news_items[:5] if news_items else [],
                llm_analysis=llm_analysis.model_dump()
            )
            
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error in analysis pipeline: {str(e)}"
            )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )
