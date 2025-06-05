from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

class NewsItem(BaseModel):
    title: str
    link: str
    published_date: str
    summary: str
    relevance_score: float = 0.0

class CrimeTrend(BaseModel):
    category: str
    count: int
    change_percentage: float
    trend_direction: str

class TrendAnalysis(BaseModel):
    time_period: str
    crime_trends: Dict[str, float]
    pattern_description: str
    contributing_factors: List[str]
    news_context: str
    recommendations: List[str]

class LLMAnalysis(BaseModel):
    trend_summary: str
    relevant_news: List[NewsItem]
    safety_recommendations: List[str]
    risk_areas: List[str]
    confidence_score: float

class AnalysisResponse(BaseModel):
    trends: TrendAnalysis
    news: List[NewsItem]
    llm_analysis: LLMAnalysis

class HeatmapData(BaseModel):
    latitude: float
    longitude: float
    risk_score: float
    district: str

class LLMAdvice(BaseModel):
    area_name: str
    risk_level: str
    summary: str
    recommendations: List[str]
    relevant_factors: List[str]

class AreaAnalysis(BaseModel):
    area_name: str
    heatmap_data: HeatmapData
    news_items: List[NewsItem]
    llm_advice: LLMAdvice 
