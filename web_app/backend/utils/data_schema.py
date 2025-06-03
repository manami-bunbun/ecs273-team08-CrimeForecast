from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Dict

class NewsItem(BaseModel):
    title: str
    link: str
    published_date: datetime
    summary: str
    relevance_score: float

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
