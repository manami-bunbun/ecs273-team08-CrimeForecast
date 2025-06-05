from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class TrendData(BaseModel):
    time_period: str
    crime_trends: Dict[str, float]
    temporal_patterns: Dict[str, Dict[str, int]]
    location_patterns: Dict[str, int]
    relevant_news: Optional[List[Dict]] = None

async def analyze_crime_trends(
    df: pd.DataFrame,
    time_window: str = "30D",
    news_items: Optional[List[Dict]] = None
) -> TrendData:
    try:
        # Convert datetime and sort
        df['incident_datetime'] = pd.to_datetime(df['incident_datetime'])
        df = df.sort_values('incident_datetime')
        
        # Get the latest date in the dataset
        end_date = df['incident_datetime'].max()
        
        # Define time periods
        last_month_start = end_date - pd.Timedelta(days=30)
        comparison_start = last_month_start - pd.Timedelta(days=150) 
        
        # Split data into periods
        last_month_data = df[df['incident_datetime'] >= last_month_start]
        comparison_data = df[(df['incident_datetime'] >= comparison_start) & 
                           (df['incident_datetime'] < last_month_start)]
        
        # Calculate crime counts for both periods
        last_month_counts = last_month_data['incident_category'].value_counts()
        comparison_counts = comparison_data['incident_category'].value_counts() / 5  
        
        # Calculate trends (percentage change)
        crime_trends = {}
        for category in set(last_month_counts.index) | set(comparison_counts.index):
            last_month_count = last_month_counts.get(category, 0)
            avg_previous_count = comparison_counts.get(category, 0)
            
            if avg_previous_count > 0:
                change = ((last_month_count - avg_previous_count) / avg_previous_count) * 100
            else:
                change = 100 if last_month_count > 0 else 0
                
            crime_trends[category] = round(change, 2)
        
        # Temporal patterns (last month only)
        hourly_patterns = last_month_data['incident_datetime'].dt.hour.value_counts().to_dict()
        daily_patterns = last_month_data['incident_datetime'].dt.day_name().value_counts().to_dict()
        
        temporal_patterns = {
            "hourly": {str(k): v for k, v in hourly_patterns.items()},
            "daily": daily_patterns
        }
        
        # Location patterns (last month only)
        location_patterns = last_month_data['police_district'].value_counts().to_dict()
        
        # Format time period string
        time_period = f"{last_month_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        
        return TrendData(
            time_period=time_period,
            crime_trends=crime_trends,
            temporal_patterns=temporal_patterns,
            location_patterns=location_patterns,
            relevant_news=news_items[:5] if news_items else None
        )
        
    except Exception as e:
        logger.error(f"Error in trend analysis: {e}")
        raise ValueError(f"Error in trend analysis: {e}") 
