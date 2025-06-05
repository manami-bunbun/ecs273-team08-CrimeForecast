from typing import List, Dict, Optional
from datetime import datetime, timedelta
from openai import OpenAI
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30.0
)


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
                            logger.info("Article not relevant to SF, skipping")
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
                        logger.info("Article added to results")
                        
                    except Exception as e:
                        logger.error(f"Error processing article: {str(e)}")
                        continue
                
                logger.info(f"Returning {len(results)} articles")
                return results
                
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        return []


# async def analyze_news_relevance(news_items: List[Dict], trends: dict) -> List[Dict]:
#     analyzed_news = []
#     for item in news_items:
#         # Calculate relevance score based on trend keywords in title and summary
#         score = 0.0
#         text = (item.get('title', '') + " " + item.get('summary', '')).lower()
        
#         # Check for crime types from trends
#         if 'crime_trends' in trends:
#             for crime_type, trend in trends['crime_trends'].items():
#                 if crime_type.lower() in text:
#                     score += 0.5
        
#         # Check for trend-related keywords
#         trend_keywords = ["crime", "incident", "arrest", "police", "safety", "victim", 
#                          "violence", "theft", "robbery", "assault", "burglary"]
        
#         for keyword in trend_keywords:
#             if keyword in text:
#                 score += 0.2
                
#         # Add location relevance
#         location_keywords = ["san francisco", "sf", "bay area", "mission", "tenderloin", 
#                            "soma", "downtown", "civic center"]
#         for keyword in location_keywords:
#             if keyword in text:
#                 score += 0.3
                
     
#         item['relevance_score'] = min(score, 1.0)
#         analyzed_news.append(item)
        
#     # Sort by relevance score
#     analyzed_news.sort(key=lambda x: x['relevance_score'], reverse=True)
#     return analyzed_news


async def analyze_news_relevance_gpt(news_items: List[Dict]) -> List[Dict]:
    if not news_items:
        return []
        
    try:
        for news in news_items:
            if 'relevance_score' not in news:
                news['relevance_score'] = 1.0
        
        news_texts = []
        for i, news in enumerate(news_items, 1):
            news_text = f"{i}. Title: {news.get('title', '')}\nSummary: {news.get('summary', '')}\n"
            news_texts.append(news_text)
            
        all_news = '\n'.join(news_texts)
        prompt = (
            "Analyze these news articles about crime in San Francisco:\n\n"
            f"{all_news}\n\n"
            "For each article, provide:\n"
            "1. A brief summary of the key points\n"
            "2. The implications for public safety\n"
            "3. Any actionable recommendations for residents\n\n"
            "Format your response as a JSON object with numbered articles as keys."
        )

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes news articles about crime and safety in San Francisco."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        analysis = response.choices[0].message.content
        
        try:
            analysis_dict = json.loads(analysis)
            for i, news in enumerate(news_items, 1):
                key = str(i)
                if key in analysis_dict:
                    if isinstance(analysis_dict[key], dict):
                        news['gpt_analysis'] = analysis_dict[key]
                    else:
                        news['gpt_analysis'] = {"analysis": analysis_dict[key]}
                        
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing GPT response as JSON: {str(e)}")

            for news in news_items:
                news['gpt_analysis'] = {"analysis": analysis}
                
        return news_items
        
    except Exception as e:
        logger.error(f"Error in LLM analysis: {str(e)}")

        return news_items 
