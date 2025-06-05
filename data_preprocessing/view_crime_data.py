from pymongo import MongoClient
import pandas as pd
from tabulate import tabulate
from datetime import datetime
import logging
import os
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_datetime(dt):
    try:
        return pd.to_datetime(dt, unit='ms')
    except ValueError:
        try:
            return pd.to_datetime(dt)
        except Exception as e:
            logger.error(f"Failed to parse datetime: {dt}, Error: {e}")
            return None

class DataViewer:
    def __init__(self):
        self.csv_viewer = CSVViewer()
        self.mongodb_viewer = MongoDBViewer()

    def view_all_stats(self):
        print("\n=== CSV Data ===")
        self.csv_viewer.get_basic_stats()
        self.csv_viewer.view_recent_incidents(5)
        
        print("\n=== MongoDB Data ===")
        self.mongodb_viewer.get_basic_stats()
        self.mongodb_viewer.view_recent_incidents(5)

class CSVViewer:
    def __init__(self):
        # Find the most recent CSV 
        csv_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "csv")
        csv_files = glob.glob(os.path.join(csv_dir, "sf_crime_*.csv.gz"))
        
        if csv_files:
            self.csv_path = max(csv_files, key=os.path.getctime)
            logger.info(f"Using CSV file: {self.csv_path}")
        else:
            self.csv_path = None
            logger.warning("No CSV files found in data/csv directory")
        
        self.df = None
        try:
            if self.csv_path and os.path.exists(self.csv_path):
                self.df = pd.read_csv(self.csv_path)
                logger.info(f"Successfully loaded CSV data from {self.csv_path}")
            else:
                logger.warning(f"No CSV file found at {self.csv_path}")
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")

    def get_basic_stats(self):
        if self.df is not None:
            print(f"Total records: {len(self.df)}")
            if 'incident_datetime' in self.df.columns:
                self.df['incident_datetime'] = self.df['incident_datetime'].apply(convert_datetime)
                date_range = self.df['incident_datetime'].agg(['min', 'max'])
                print(f"Date range: {date_range['min']} to {date_range['max']}")
        else:
            print("No CSV data available")

    def view_recent_incidents(self, n=5):
        if self.df is not None and len(self.df) > 0:
            if 'incident_datetime' in self.df.columns:
                self.df['incident_datetime'] = self.df['incident_datetime'].apply(convert_datetime)
                recent = self.df.sort_values('incident_datetime', ascending=False).head(n)
                print(f"\nMost recent {n} incidents from CSV:")
                print(tabulate(recent, headers='keys', tablefmt='psql', showindex=False))
        else:
            print("No CSV data available")

class MongoDBViewer:
    def __init__(self):
        try:
            self.client = MongoClient('mongodb://localhost:27017/')
            self.db = self.client['sf_crime']
            self.collection = self.db['incidents']
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def get_basic_stats(self):
        """Get basic statistics about MongoDB data"""
        try:
            total_records = self.collection.count_documents({})
            print(f"Total records: {total_records}")
            
            date_range = self.collection.aggregate([
                {
                    "$group": {
                        "_id": None,
                        "min_date": {"$min": "$incident_datetime"},
                        "max_date": {"$max": "$incident_datetime"}
                    }
                }
            ])
            
            date_info = next(date_range, None)
            if date_info:
                min_date = convert_datetime(date_info['min_date'])
                max_date = convert_datetime(date_info['max_date'])
                print(f"Date range: {min_date} to {max_date}")
            
        except Exception as e:
            logger.error(f"Error getting MongoDB stats: {e}")

    def view_recent_incidents(self, n=5):
        """View n most recent incidents from MongoDB"""
        try:
            recent = list(self.collection.find().sort("incident_datetime", -1).limit(n))
            if recent:
                df = pd.DataFrame(recent)
                df = df.drop('_id', axis=1) 
                if 'incident_datetime' in df.columns:
                    df['incident_datetime'] = df['incident_datetime'].apply(convert_datetime)
                print(f"\nMost recent {n} incidents from MongoDB:")
                print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
            else:
                print("No incidents found in MongoDB")
        except Exception as e:
            logger.error(f"Error viewing MongoDB incidents: {e}")

if __name__ == "__main__":
    try:
        viewer = DataViewer()
        viewer.view_all_stats()
    except Exception as e:
        logger.error(f"Error in main: {e}") 
 