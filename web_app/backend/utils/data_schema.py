from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

class NewsItem(BaseModel):
    title: str
    summary: str
    link: Optional[str] = None
    published_date: Optional[str] = None
    relevance_score: float = 0.0

class CrimeTrend(BaseModel):
    category: str
    count: int
    change_percentage: float
    trend_direction: str

class TrendAnalysis(BaseModel):
    time_period: str
    crime_trends: Dict[str, float]
    temporal_patterns: Dict[str, Dict[str, int]]
    location_patterns: Dict[str, int]
    pattern_description: Optional[str] = None
    contributing_factors: Optional[List[str]] = None
    news_context: Optional[str] = None
    recommendations: Optional[List[str]] = None

class LLMAnalysis(BaseModel):
    trend_summary: str
    relevant_news: List[Dict]
    safety_recommendations: List[str]

class AnalysisResponse(BaseModel):
    trends: Dict
    news: List[Dict]
    llm_analysis: Dict 
