from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI
import os
import json
from pydantic import BaseModel


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30.0
)

class TrendAnalysis(BaseModel):
    time_period: str
    crime_trends: Dict[str, float]
    pattern_description: str
    contributing_factors: List[str]
    news_context: str
    recommendations: List[str]

async def analyze_crime_trends(
    df: pd.DataFrame,
    time_window: str = "30D",
    news_items: Optional[List[Dict]] = None
) -> TrendAnalysis:
    try:
        # Convert datetime and sort
        df['incident_datetime'] = pd.to_datetime(df['incident_datetime'])
        df = df.sort_values('incident_datetime')
        
        # Calculate time windows
        end_date = df['incident_datetime'].max()
        start_date = end_date - pd.Timedelta(time_window)
        
        # Get data for current and previous periods
        current_period = df[df['incident_datetime'] >= start_date]
        prev_start = start_date - pd.Timedelta(time_window)
        previous_period = df[(df['incident_datetime'] >= prev_start) & (df['incident_datetime'] < start_date)]
        
        # Calculate crime type trends
        current_counts = current_period['incident_category'].value_counts()
        previous_counts = previous_period['incident_category'].value_counts()
        
        # Calculate percent changes
        crime_trends = {}
        for crime_type in set(current_counts.index) | set(previous_counts.index):
            curr = current_counts.get(crime_type, 0)
            prev = previous_counts.get(crime_type, 0)
            if prev > 0:
                pct_change = ((curr - prev) / prev) * 100
            else:
                pct_change = 100 if curr > 0 else 0
            crime_trends[crime_type] = round(pct_change, 2)
        
        # Temporal patterns
        current_period['hour'] = current_period['incident_datetime'].dt.hour
        current_period['day_of_week'] = current_period['incident_datetime'].dt.day_name()
        
        hourly_patterns = current_period.groupby('hour').size()
        daily_patterns = current_period.groupby('day_of_week').size()
        
        # Location patterns
        location_patterns = current_period.groupby('analysis_neighborhood').size()
        
        # Prepare data for LLM analysis
        trend_data = {
            "time_period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "crime_trends": crime_trends,
            "temporal_patterns": {
                "hourly": hourly_patterns.to_dict(),
                "daily": daily_patterns.to_dict()
            },
            "location_patterns": location_patterns.to_dict()
        }
        
        if news_items:
            trend_data["relevant_news"] = [
                {
                    "title": item["title"],
                    "summary": item["summary"],
                    "relevance_score": item.get("relevance_score", 0)
                }
                for item in news_items
            ]
        
        # Generate LLM analysis
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a crime analysis expert. You must respond with valid JSON only."},
                {"role": "user", "content": """Please analyze the crime data and respond with a JSON object in this exact format:
                    {
                        "pattern_description": "Natural language description of main trends",
                        "contributing_factors": ["factor1", "factor2", "factor3"],
                        "news_context": "How recent news events relate to patterns",
                        "recommendations": ["rec1", "rec2", "rec3"]
                    }"""},
                {"role": "user", "content": f"Crime trend data to analyze:\n\n{json.dumps(trend_data, indent=2)}"}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        analysis = json.loads(response.choices[0].message.content.strip())
        
        return TrendAnalysis(
            time_period=trend_data["time_period"],
            crime_trends=crime_trends,
            pattern_description=analysis["pattern_description"],
            contributing_factors=analysis["contributing_factors"],
            news_context=analysis["news_context"],
            recommendations=analysis["recommendations"]
        )
        
    except Exception as e:
        raise ValueError(f"Error in trend analysis: {str(e)}") 
