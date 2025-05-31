from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime
import pandas as pd
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient

app = FastAPI()

# MongoDB 
MONGO_URL = "mongodb://localhost:27018"
client = AsyncIOMotorClient(MONGO_URL)
db = client.crime_forecast

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    location: dict
    timestamp: str
    crime_probabilities: dict
    risk_scores: dict


@app.get("/")
async def root():
    return {"message": "Crime Forecast API"}

