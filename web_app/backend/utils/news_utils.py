from typing import List, Dict
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
import os
import aiohttp
from pydantic import BaseModel
import json

load_dotenv()

class NewsItem(BaseModel):
    title: str
    link: str
    published_date: datetime
    summary: str
    relevance_score: float

# fetch San Francisco news from NewsAPI
async def fetch_sf_news(start_date: datetime, end_date: datetime) -> List[Dict]:
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        raise ValueError("NEWS_API_KEY not found in environment variables")
    
    from_date = start_date.strftime('%Y-%m-%d') #TODO: check
    to_date = end_date.strftime('%Y-%m-%d')
    
    # Keywords for San Francisco news
    keywords = '("San Francisco" OR "SF" OR "Bay Area")'
    
    # NewsAPI endpoint
    url = f'https://newsapi.org/v2/everything'
    params = {
        'q': keywords,
        'from': from_date,
        'to': to_date,
        'sortBy': 'publishedAt',
        'language': 'en',
        'apiKey': api_key,
        'pageSize': 20  # Get only 20 most recent articles
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get('articles', [])
                    
                    # Format articles
                    sf_news = []
                    for article in articles:
                        sf_news.append({
                            'title': article.get('title', ''),
                            'link': article.get('url', ''),
                            'published_date': datetime.strptime(article.get('publishedAt', ''), '%Y-%m-%dT%H:%M:%SZ'),
                            'summary': article.get('description', '')
                        })
                    
                    return sf_news
                else:
                    print(f"Error fetching news: {response.status}")
                    return []
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


async def analyze_news_relevance(news_items: List[Dict]) -> List[NewsItem]:
    """Analyze news relevance using GPT API in a single batch"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
        
    client = OpenAI(api_key=api_key)
    

    news_texts = []
    for i, news in enumerate(news_items, 1):
        news_texts.append(
            "Article {}:\nTitle: {}\nSummary: {}".format(
                i,
                news['title'],
                news['summary']
            )
        )
    
    prompt = """
            Analyze the following news articles for their relevance to crime prediction in San Francisco.
            Rate each article's relevance on a scale of 0.0 to 1.0, where:
            0.0 = Not relevant to crime/safety
            1.0 = Highly relevant to crime/safety

            {}

            Return a JSON array of objects with article numbers and scores, sorted by relevance (highest first), including only the top 5 most relevant articles.
            Example format:
            [
                {"article": 1, "score": 0.9},
                {"article": 4, "score": 0.8},
                ...
            ]
            """.format('\n'.join(news_texts))
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a crime news analyst. Analyze multiple news articles and return a JSON array of the top 5 most relevant articles with their scores."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0
        )
        
        scores = json.loads(response.choices[0].message.content.strip())
        
        # top 5 articles
        analyzed_news = []
        for score_obj in scores:
            article_idx = score_obj['article'] - 1  
            news = news_items[article_idx]
            analyzed_news.append(NewsItem(
                title=news['title'],
                link=news['link'],
                published_date=news['published_date'],
                summary=news['summary'],
                relevance_score=score_obj['score']
            ))
        
        return analyzed_news
        
    except Exception as e:
        print("Error analyzing news batch: {}".format(e))
        return [] 
