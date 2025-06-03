from typing import Dict, Optional
from datetime import datetime, timedelta
from .models import NewsItem, LLMAdvice
from typing import List

class AnalysisCache:
    def __init__(self):
        self._news_cache: Dict[str, List[NewsItem]] = {}
        self._advice_cache: Dict[str, LLMAdvice] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_duration = timedelta(minutes=30)  # Cache for 30 minutes

    def _generate_key(self, area_name: str, start_date: str, end_date: str) -> str:
        return f"{area_name}_{start_date}_{end_date}"

    def get_news(self, area_name: str, start_date: str, end_date: str) -> Optional[List[NewsItem]]:
        key = self._generate_key(area_name, start_date, end_date)
        if key in self._news_cache and datetime.now() - self._cache_time[key] < self._cache_duration:
            return self._news_cache[key]
        return None

    def get_advice(self, area_name: str, start_date: str, end_date: str) -> Optional[LLMAdvice]:
        key = self._generate_key(area_name, start_date, end_date)
        if key in self._advice_cache and datetime.now() - self._cache_time[key] < self._cache_duration:
            return self._advice_cache[key]
        return None

    def store_analysis(self, area_name: str, start_date: str, end_date: str, 
                      news: List[NewsItem], advice: LLMAdvice):
        key = self._generate_key(area_name, start_date, end_date)
        self._news_cache[key] = news
        self._advice_cache[key] = advice
        self._cache_time[key] = datetime.now()

# Global cache instance
analysis_cache = AnalysisCache() 
