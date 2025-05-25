import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class CrimeDataPreprocessor:
    @staticmethod
    def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Convert timestamps
        df['incident_datetime'] = pd.to_datetime(df['incident_datetime'])
        
        df['year'] = df['incident_datetime'].dt.year
        df['month'] = df['incident_datetime'].dt.month
        df['day'] = df['incident_datetime'].dt.day
        df['hour'] = df['incident_datetime'].dt.hour
        df['day_of_week'] = df['incident_datetime'].dt.day_name()
        df['is_weekend'] = df['incident_datetime'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Categorize time periods
        df['time_period'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        return df

    @staticmethod
    def process_location_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        1. Converts latitude/longitude to numeric values
        2. Creates GeoJSON format location field for MongoDB
        3. Handles missing or invalid coordinates
        """
        df = df.copy()
        
        # Convert coordinates to numeric
        if 'latitude' in df.columns:
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        if 'longitude' in df.columns:
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            
        # Create GeoJSON location field for MongoDB
        mask = df['latitude'].notna() & df['longitude'].notna()
        df.loc[mask, 'location'] = df[mask].apply(
            lambda row: {
                'type': 'Point',
                'coordinates': [float(row['longitude']), float(row['latitude'])]
            },
            axis=1
        )
        
        return df

    @staticmethod
    def clean_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize categorical data
        1. Handles missing values in categorical columns
        2. Standardizes text (strips whitespace, converts to uppercase)
        3. Processes the following columns:
           - police_district
           - incident_category
           - incident_subcategory
           - incident_description
        """
        df = df.copy()
        
        categorical_columns = [
            'police_district',
            'incident_category',
            'incident_subcategory',
            'incident_description'
        ]
        
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('UNKNOWN')
                df[col] = df[col].str.strip().str.upper()
        
        return df

    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        required_columns = [
            'incident_datetime',
            'incident_category',
            'police_district',
            'latitude',
            'longitude'
        ]
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        # Validate datetime format
        try:
            pd.to_datetime(df['incident_datetime'])
        except Exception as e:
            logger.error(f"Invalid datetime format: {e}")
            return False
            
        return True 
