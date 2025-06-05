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
        for item in sorted(news_items, key=lambda x: x.get('relevance_score', 0), reverse=True)[:5]
    ])

async def analyze_trends_and_news(trend_data: TrendData) -> LLMAnalysis:
    try:
        news_context = format_news_for_prompt(trend_data.relevant_news if trend_data.relevant_news else [])
        
        response = client.chat.completions.create(
            model="gpt-4o",
            # we use LLM to generate the prompt here
            messages=[ 
                {"role": "system", "content": "You are a crime analysis expert. Provide a concise analysis focusing on practical safety advice. You must respond with valid JSON only, without any additional text or formatting."},
                {"role": "user", "content": """Please analyze the following data and respond with a JSON object in this exact format:
                        {
                            "trend_summary": "A clear and concise 2-3 sentence summary of the main crime trends",
                            "safety_recommendations": [
                                "Specific and actionable safety recommendation 1",
                                "Specific and actionable safety recommendation 2",
                                "Specific and actionable safety recommendation 3"
                            ]
                        }"""},
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
            
            return LLMAnalysis(
                trend_summary=analysis["trend_summary"],
                relevant_news=trend_data.relevant_news[:5] if trend_data.relevant_news else [],
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
