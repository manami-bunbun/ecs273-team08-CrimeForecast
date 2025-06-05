"""
This file is the main file for the backend of the crime forecast web app API.
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from utils.news_utils import fetch_sf_news
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
    return {"message": "Crime Forecast Web App"}

# Get crime data for the bar chart visualization
@app.get("/api/crime-data")
async def get_crime_data(
    end_date: Optional[str] = None
):
    try:
        query = {}
        latest_doc = await db.incidents.find_one(
            sort=[("incident_datetime", -1)],
            projection={"incident_datetime": 1, "_id": 0}
        )

        if not latest_doc or "incident_datetime" not in latest_doc:
            raise HTTPException(status_code=500, detail="No data available in the database.")

        latest_date = datetime.fromisoformat(latest_doc["incident_datetime"].replace("Z", ""))

        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

            # Check if the given end_date exceeds the latest available data
            if end_date_obj > latest_date:
                raise HTTPException(
                    status_code=400,
                    detail=f"Selected date exceeds the latest available data. Data is only available up to {latest_date.strftime('%Y-%m-%d')}."
                )


            start_date_obj = end_date_obj - timedelta(days=30)
            query["incident_datetime"] = {
                "$gte": start_date_obj.isoformat(),
                "$lte": end_date_obj.isoformat()
            }
        
        count = await db.incidents.count_documents(query)
        if count == 0:
            return []  

        cursor = db.incidents.find(
            query,
            {"incident_category": 1, "incident_datetime": 1, "_id": 0}
        )
        
        data = await cursor.to_list(length=None)
        
        # Clean and validate data before returning
        valid_data = []
        for item in data:
            try:
                # Ensure incident_datetime is valid
                if isinstance(item.get('incident_datetime'), str):
                    datetime.fromisoformat(item['incident_datetime'].replace('Z', ''))
                else:
                    continue
                    
                valid_data.append({
                    'incident_datetime': item['incident_datetime'],
                    'incident_category': item.get('incident_category', 'Unknown')
                })
            except (ValueError, TypeError):
                continue
                
        return valid_data
    except Exception as e:
        logger.error(f"Error fetching crime data: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# Get crime locations for the heat map visualization
@app.get("/api/crime-locations")
async def get_crime_locations(
    category: str,
    end_date: Optional[str] = None
):
    try:
        # Parse end date
        if end_date:
            try:
                end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid date format. Please use YYYY-MM-DD"
                )
        else:
            end_date_obj = datetime.now()
            
        start_date_obj = end_date_obj - timedelta(days=30)
        
        # Build query with date range
        query = {
            "incident_datetime": {
                "$gte": start_date_obj.isoformat(),
                "$lte": end_date_obj.isoformat()
            }
        }
        
        if category and category != "All":
            query["incident_category"] = category

        # First check if any data exists for the date range
        count = await db.incidents.count_documents(query)
        if count == 0:
            return []  # Return empty list if no data found

        cursor = db.incidents.find(
            query,
            {
                "latitude": 1,
                "longitude": 1,
                "incident_category": 1,
                "incident_datetime": 1,  # Add this to verify dates
                "_id": 0
            }
        )
        
        locations = await cursor.to_list(length=None)
        
        # Filter out records with invalid coordinates or dates
        valid_locations = []
        for loc in locations:
            try:
                lat = float(loc.get('latitude', 0))
                lon = float(loc.get('longitude', 0))
                
                # Verify the date is within range
                incident_date = datetime.fromisoformat(loc['incident_datetime'].replace('Z', ''))
                if not (start_date_obj <= incident_date <= end_date_obj):
                    continue
                
                # Check if coordinates are valid
                if (lat != 0 and lon != 0 and 
                    -90 <= lat <= 90 and 
                    -180 <= lon <= 180 and
                    not pd.isna(lat) and 
                    not pd.isna(lon)):
                    valid_locations.append({
                        'latitude': lat,
                        'longitude': lon,
                        'incident_category': loc.get('incident_category', 'Unknown'),
                        'incident_datetime': loc['incident_datetime']
                    })
            except (ValueError, TypeError):
                continue
                
        if not valid_locations:
            return []  # Return empty list if no valid locations found
            
        return valid_locations
    except Exception as e:
        logger.error(f"Error fetching crime locations: {e}")
        raise HTTPException(status_code=500, detail=str(e))




# Run trend analysis for the trend section to feed the LLM (to avoid hallucination)
# Get news for the news section (LLM selected 5 news among these)
@app.get("/api/analysis", response_model=AnalysisResponse)
async def get_trend_analysis(
    end_date: str = Query(..., description="End date for analysis (YYYY-MM-DD)")
) -> AnalysisResponse:
    try:
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date_obj = end_date_obj - timedelta(days=30)

        logger.info(f"Fetching crime data from {start_date_obj.isoformat()} to {end_date_obj.isoformat()}")

        # Convert dates to ISO format strings for MongoDB query 
        query = {
            "incident_datetime": {
                "$gte": start_date_obj.isoformat(),
                "$lte": end_date_obj.isoformat()
            }
        }

        crime_data = []
        async for doc in db.incidents.find(query):
            crime_data.append(doc)
            
        if not crime_data:
            logger.warning(f"No data found for date range: {start_date_obj.isoformat()} to {end_date_obj.isoformat()}")
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
