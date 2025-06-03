from typing import List, Dict
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from .models import NewsItem, HeatmapData, LLMAdvice

load_dotenv()


#  Returns advice and recommendations for the specified area
async def analyze_area_safety(
    area_name: str,
    heatmap_data: HeatmapData,
    news_items: List[NewsItem]
) -> LLMAdvice:

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    
    news_context = "\n".join([
        f"- {item.title}: {item.summary}"
        for item in news_items
    ])
    
    heatmap_context = (
        f"Area: {area_name}\n"
        f"Risk Score: {heatmap_data.risk_score}\n"
        f"District: {heatmap_data.district}\n"
        f"Location: ({heatmap_data.latitude}, {heatmap_data.longitude})"
    )
    
    prompt = f"""
    Analyze the safety situation for the following area based on crime risk data and recent news:

    {heatmap_context}

    Recent relevant news:
    {news_context}

    Provide a comprehensive safety analysis including:
    1. Current risk level assessment
    2. Summary of the situation
    3. Key factors contributing to safety/risk
    4. Specific recommendations for safety

    Return the analysis in the following JSON format:
    {{
        "risk_level": "low/medium/high",
        "summary": "brief situation summary",
        "recommendations": ["rec1", "rec2", ...],
        "relevant_factors": ["factor1", "factor2", ...]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a crime and safety analysis expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        
        analysis = json.loads(response.choices[0].message.content.strip())
        
        return LLMAdvice(
            area_name=area_name,
            risk_level=analysis["risk_level"],
            summary=analysis["summary"],
            recommendations=analysis["recommendations"],
            relevant_factors=analysis["relevant_factors"]
        )
        
    except Exception as e:
        raise ValueError(f"Error in LLM analysis: {str(e)}") 
