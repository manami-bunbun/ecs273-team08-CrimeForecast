"""
This file is the LLM analysis based on the trend analysis output and news.
"""
from typing import List, Dict
from openai import OpenAI
import os
from dotenv import load_dotenv
from .data_schema import LLMAnalysis
import json
import logging
from pydantic import BaseModel
from .trend_analysis import TrendData

logger = logging.getLogger(__name__)
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30.0
)

def format_news_for_prompt(news_items: List[Dict]) -> str:
    if not news_items:
        return "No recent news articles available."
        
    return "\n".join([
        f"- {item.get('title', '')} ({item.get('published_date', 'N/A')}): {item.get('summary', 'No summary available')}"
        for item in news_items
    ])



async def analyze_trends_and_news(trend_data: TrendData) -> LLMAnalysis:
    try:
        news_context = format_news_for_prompt(trend_data.relevant_news if trend_data.relevant_news else [])
        
        response = client.chat.completions.create(
            model="gpt-4o",
            # we use LLM to generate the prompt here
            messages=[ 
                {"role": "system", "content": """You are a crime analysis expert. Your task is to:
                1. Analyze crime trends and news
                2. Select the top 5 most relevant news items that are most important for public safety
                3. Provide practical safety advice
                
                Important constraints:
                    - You must only refer to crime types or patterns that are explicitly mentioned in the provided trend summary.
                    - Do not invent or speculate about additional categories, explanations, or causes.
                    - If no relevant news article can be identified, include 'no match' as a placeholder.
                    - You must respond with valid JSON only, without any additional text or formatting."""},
                {"role": "user", "content": """Please analyze the following data and respond with a JSON object in this exact format:
                        {
                            "trend_summary": "A clear and concise 2-3 sentence summary of the main crime trends",
                            "selected_news_indices": [0, 1, 2, 3, 4],
                            "safety_recommendations": [
                                "Specific and actionable safety recommendation 1",
                                "Specific and actionable safety recommendation 2",
                                "Specific and actionable safety recommendation 3"
                            ]
                        }
                        
                        Note: selected_news_indices should contain the indices (0-based) of the 5 most relevant news items from the provided list."""},
                {"role": "user", "content": f"""Crime trends in San Francisco from {trend_data.time_period}:

                        Crime Trends: {json.dumps(trend_data.crime_trends, indent=2)}
                        Temporal Patterns: {json.dumps(trend_data.temporal_patterns, indent=2)}
                        Location Patterns: {json.dumps(trend_data.location_patterns, indent=2)}
                        Recent News:\n{news_context}"""}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        try:
            content = response.choices[0].message.content.strip()

            # Remove any potential markdown formatting
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            analysis = json.loads(content)
            
            # Get the selected news items using the indices provided by LLM
            selected_indices = analysis.get("selected_news_indices", [])[:5]  # Ensure max 5 items
            selected_news = []
            if trend_data.relevant_news:
                for idx in selected_indices:
                    if idx < len(trend_data.relevant_news):
                        selected_news.append(trend_data.relevant_news[idx])
            
            return LLMAnalysis(
                trend_summary=analysis["trend_summary"],
                relevant_news=selected_news,
                safety_recommendations=analysis["safety_recommendations"]
            )
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}, Content: {content}")
            raise ValueError(f"Invalid JSON response from OpenAI API: {str(e)}")
        except KeyError as e:
            logger.error(f"Missing required field in analysis: {str(e)}")
            raise ValueError(f"Missing required field in analysis: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error in LLM analysis: {e}")
        return LLMAnalysis(
            trend_summary="Error in analysis",
            relevant_news=[],
            safety_recommendations=["Stay aware of your surroundings", "Report suspicious activity", "Keep valuables secure"]
        )

# Parse the LLM response into structured analysis
def parse_llm_response(response_text: str) -> LLMAnalysis:
    lines = response_text.split("\n")
    
    return LLMAnalysis(
        trend_summary=lines[0] if lines else "No summary available",
        relevant_news=[],  
        safety_recommendations=[line.strip("- ") for line in lines if line.startswith("- ")]
    ) 
