from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

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

