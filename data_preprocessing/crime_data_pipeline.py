from pymongo import MongoClient, GEOSPHERE
import pandas as pd
import os
import logging
from datetime import datetime
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_datetime(dt):
    """Convert various datetime formats to pandas datetime"""
    try:
        return pd.to_datetime(dt, unit='ms')
    except ValueError:
        try:
            return pd.to_datetime(dt)
        except Exception as e:
            logger.error(f"Failed to parse datetime: {dt}, Error: {e}")
            return None

def format_date(date):
    return date.strftime("%Y%m%d")

def get_csv_path(df):
    if 'incident_datetime' not in df.columns:
        return "sf_crime_data_latest.csv.gz"
    
    df['incident_datetime'] = df['incident_datetime'].apply(convert_datetime)
    min_date = df['incident_datetime'].min()
    max_date = df['incident_datetime'].max()
    
    return f"sf_crime_{format_date(min_date)}_{format_date(max_date)}.csv.gz"

def save_csv_with_date_range(df, csv_dir):
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    csv_filename = get_csv_path(df)
    csv_path = os.path.join(csv_dir, csv_filename)

    if 'incident_datetime' in df.columns:
        df['incident_datetime'] = df['incident_datetime'].astype(str)
    
    df.to_csv(csv_path, compression='gzip', index=False)
    logger.info(f"Saved CSV data to {csv_path}")
    return csv_path

def process_crime_data():
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['sf_crime']
        collection = db['incidents']
        
        logger.info("Dropping existing collection...")
        db.drop_collection('incidents')
        
        collection = db['incidents']
        collection.create_index("incident_number", unique=True)
        collection.create_index([("location", GEOSPHERE)])
        logger.info("Created new collection with indexes")
        
        offset = 0
        batch_size = 50000  # Maximum records per request for API v2.0
        target_total = 100000  # Total number of records we want to fetch
        all_data = []
        
        while len(all_data) < target_total:
            remaining = target_total - len(all_data)
            limit = min(batch_size, remaining)
            
            url = f"https://data.sfgov.org/resource/wg3w-h783.json?$limit={limit}&$offset={offset}&$order=incident_datetime DESC"
            logger.info(f"Fetching data from API with offset {offset}...")
            
            response = requests.get(url)
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}")
                break
                
            batch_data = response.json()
            if not batch_data:  # No more data
                break
                
            all_data.extend(batch_data)
            logger.info(f"Fetched {len(batch_data)} records. Total records so far: {len(all_data)}")
            
            if len(batch_data) < limit:  # Last batch
                break
                
            offset += limit
        
        df = pd.DataFrame(all_data)
        
        if len(df) == 0:
            logger.error("No data received from API")
            return
            
        logger.info(f"Total records loaded: {len(df)}")
        

        if 'incident_datetime' in df.columns:
            df['incident_datetime'] = df['incident_datetime'].apply(convert_datetime)
            df = df.sort_values('incident_datetime', ascending=False)
            df = df.drop_duplicates(subset=['incident_number'], keep='first')
            logger.info(f"After removing duplicates: {len(df)} records")
        

        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        

        df['location'] = df.apply(
            lambda row: {
                'type': 'Point',
                'coordinates': [float(row['longitude']), float(row['latitude'])]
            } if pd.notnull(row['longitude']) and pd.notnull(row['latitude']) else None,
            axis=1
        )
        

        null_coords = df[df['location'].isnull()].shape[0]
        null_dates = df[df['incident_datetime'].isnull()].shape[0]
        logger.info(f"Records with missing coordinates: {null_coords}")
        logger.info(f"Records with missing dates: {null_dates}")

        csv_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "csv")
        save_csv_with_date_range(df, csv_dir)
        

        df['incident_datetime'] = df['incident_datetime'].astype(str)
        

        records = df.replace({pd.NA: None}).to_dict('records')
        

        logger.info(f"Inserting {len(records)} records into MongoDB...")
        collection.insert_many(records)
        
        logger.info("MongoDB rebuild complete")
        
    except Exception as e:
        logger.error(f"Error processing crime data: {e}")
        raise

if __name__ == "__main__":
    process_crime_data()
