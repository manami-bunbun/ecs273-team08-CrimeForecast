from pymongo import MongoClient, GEOSPHERE
import logging
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class MongoDBManager:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client.crime_forecast
        self.collection = self.db.incidents
        self.create_indexes()

    def insert_incidents(self, incidents: List[Dict]) -> None:
        try:
            # Ensure datetime is in ISO format string
            for incident in incidents:
                if 'incident_datetime' in incident:
                    if isinstance(incident['incident_datetime'], datetime):
                        incident['incident_datetime'] = incident['incident_datetime'].isoformat()
                    elif isinstance(incident['incident_datetime'], str):
                        # Validate and standardize date format
                        try:
                            dt = datetime.fromisoformat(incident['incident_datetime'].replace('Z', ''))
                            incident['incident_datetime'] = dt.isoformat()
                        except ValueError:
                            logger.warning(f"Invalid datetime format: {incident['incident_datetime']}")
                            continue

            self.collection.insert_many(incidents)
            logger.info(f"Inserted {len(incidents)} incidents into MongoDB")
        except Exception as e:
            logger.error(f"Error inserting incidents: {e}")
            raise

    def get_incidents(self, query: Dict = None, projection: Dict = None) -> pd.DataFrame:
        try:
            cursor = self.collection.find(
                query if query else {},
                projection if projection else None
            )
            data = list(cursor)
            
            # Convert to DataFrame and handle datetime
            df = pd.DataFrame(data)
            if not df.empty and 'incident_datetime' in df.columns:
                df['incident_datetime'] = pd.to_datetime(df['incident_datetime'])
            
            return df
        except Exception as e:
            logger.error(f"Error retrieving incidents: {e}")
            raise

    def get_incidents_by_date_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            # Ensure dates are in ISO format
            query = {
                "incident_datetime": {
                    "$gte": start_date.isoformat(),
                    "$lte": end_date.isoformat()
                }
            }
            return self.get_incidents(query)
        except Exception as e:
            logger.error(f"Error retrieving incidents by date range: {e}")
            raise

    def get_incidents_by_location(self, lat: float, lon: float, radius_meters: int = 1000) -> pd.DataFrame:
        try:
            # Validate coordinates
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                raise ValueError("Invalid coordinates")

            query = {
                "location": {
                    "$near": {
                        "$geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat]
                        },
                        "$maxDistance": radius_meters
                    }
                }
            }
            return self.get_incidents(query)
        except Exception as e:
            logger.error(f"Error retrieving incidents by location: {e}")
            raise

    def create_indexes(self) -> None:
        try:
            # Date index
            self.collection.create_index("incident_datetime")
            
            # Category index
            self.collection.create_index("incident_category")
            
            # Geospatial index
            self.collection.create_index([("location", GEOSPHERE)])
            
            logger.info("Successfully created MongoDB indexes")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            raise

    def close(self) -> None:
        """Close MongoDB connection"""
        try:
            self.client.close()
            logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")
            raise 
