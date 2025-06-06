"""
This file collects news from Google News to feed the LLM.
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv
import os
import aiohttp
from pydantic import BaseModel
import json
from .data_schema import NewsItem
from bs4 import BeautifulSoup
import asyncio
from urllib.parse import quote, urlencode
import re
import logging
import glob



load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


class NewsItem(BaseModel):
    title: str
    link: str
    published_date: str
    summary: str


    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "link": self.link,
            "published_date": self.published_date
        }


# Convert relative date strings to ISO format dates
def parse_relative_date(date_str: str) -> str:

    now = datetime.now()
    
    if 'hour' in date_str or 'min' in date_str:
        return now.isoformat()
    
    match = re.search(r'(\d+)\s*(day|week|month|year)', date_str)
    if not match:
        return now.isoformat()
        
    num = int(match.group(1))
    unit = match.group(2)
    
    if unit == 'day':
        delta = timedelta(days=num)
    elif unit == 'week':
        delta = timedelta(weeks=num)
    elif unit == 'month':
        delta = timedelta(days=num*30)
    else:  # year
        delta = timedelta(days=num*365)
    
    date = now - delta
    return date.isoformat()


# fetch San Francisco news from Google News
async def fetch_sf_news(end_date: str) -> List[Dict]:
    try:
        # Convert end_date string to datetime (timezone naive)
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
        # Calculate start date (1 month before end_date)
        start_date_dt = end_date_dt - timedelta(days=30)
        
        # Format dates for Google News query
        after = start_date_dt.strftime("%Y-%m-%d")
        before = end_date_dt.strftime("%Y-%m-%d")
        
        # Google News search URL construction with date range
        search_query = "San Francisco crime"
        base_url = "https://news.google.com/search"
        params = {
            "q": search_query,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en"
        }
        url = f"{base_url}?{urlencode(params)}"
        
        logger.info(f"Fetching news from URL: {url}")
        logger.info(f"Date range: {after} to {before}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html = await response.text()
                
                soup = BeautifulSoup(html, 'html.parser')
                articles = soup.find_all('article')
                
                results = []
                for article in articles:
                    try:
                        # Find title and link
                        title_element = None
                        for selector in ['a.JtKRv', 'h3 a', 'h4 a']:
                            title_element = article.select_one(selector)
                            if title_element:
                                break
                                
                        if not title_element:
                            continue
                            
                        title = title_element.get_text(strip=True)
                        if not title:
                            continue
                            
                        logger.info(f"Found article: {title}")
                        
                        #date
                        time_element = article.find('time', class_='hvbAAd')
                        relative_date = time_element.get_text(strip=True) if time_element else None
                        datetime_attr = time_element.get('datetime') if time_element else None
                        
                        if datetime_attr:
                            published_date = datetime_attr.replace('Z', '')
                            article_date = datetime.fromisoformat(published_date)
                            
         
                            if article_date.date() < start_date_dt.date() or article_date.date() > end_date_dt.date():
                                logger.info(f"Article date {article_date.date()} outside range {start_date_dt.date()} to {end_date_dt.date()}, skipping")
                                continue
                                
                            logger.info(f"Article date: {published_date}")
                        else:
                            continue
                        
              
                        if not any(keyword.lower() in title.lower() for keyword in ["SF", "San Francisco", "Bay Area"]):
                            # logger.info("Article not relevant to SF, skipping")
                            continue
                            
                
                        link = title_element.get('href', '')
                        if link.startswith('./'):
                            link = 'https://news.google.com' + link[1:]
                        elif not link.startswith('http'):
                            link = 'https://news.google.com' + link
                            
        
                        news_item = {
                            "title": title,
                            "link": link,
                            "published_date": published_date,
                            "summary": title, 
                            "relevance_score": 0.0
                        }
                        
                        results.append(news_item)
                        # logger.info("Article added to results")
                        
                    except Exception as e:
                        logger.error(f"Error processing article: {str(e)}")
                        continue
                
                logger.info(f"Returning {len(results)} articles")
                return results
                
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        return []
