from pymongo import MongoClient
import logging
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class MongoDBManager:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client.team8
        self.collection = self.db.sf_crime_incidents

    def insert_incidents(self, incidents: List[Dict]) -> None:
        try:
            self.collection.insert_many(incidents)
            logger.info(f"Inserted {len(incidents)} incidents into MongoDB")
        except Exception as e:
            logger.error(f"Error inserting incidents: {e}")
            raise

    def get_incidents(self, query: Dict = None) -> pd.DataFrame:
        try:
            cursor = self.collection.find(query if query else {})
            return pd.DataFrame(list(cursor))
        except Exception as e:
            logger.error(f"Error retrieving incidents: {e}")
            raise

    def create_indexes(self) -> None:
        try:
            self.collection.create_index("incident_datetime")
            self.collection.create_index("incident_category")
            self.collection.create_index("latitude")
            self.collection.create_index("longitude")
            
            logger.info("Successfully created MongoDB indexes")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            raise

    def close(self) -> None:
        """
        Close MongoDB connection
        """
        self.client.close() 
