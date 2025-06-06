import pandas as pd
import gzip
from datetime import datetime
import logging
from pymongo import MongoClient, GEOSPHERE
import os
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Convert various datetime formats to pandas datetime
def convert_datetime(dt):
    try:
        return pd.to_datetime(dt)
    except Exception as e:
        logger.error(f"Failed to parse datetime: {dt}, Error: {e}")
        return None

# Process crime data before inserting into MongoDB
def process_crime_data(df):
    # Convert timestamps
    df['incident_datetime'] = pd.to_datetime(df['incident_datetime'])
    
    # Convert coordinates to numeric
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Create GeoJSON location field
    mask = df['latitude'].notna() & df['longitude'].notna()
    df.loc[mask, 'location'] = df[mask].apply(
        lambda row: {
            'type': 'Point',
            'coordinates': [float(row['longitude']), float(row['latitude'])]
        },
        axis=1
    )
    
    # Clean categorical data
    categorical_columns = [
        'incident_category',
        'incident_subcategory',
        'incident_description'
    ]
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('UNKNOWN')
            df[col] = df[col].str.strip().str.upper()
    
    return df

# Find the most recent crime data CSV file
def get_latest_csv():
    # Get the absolute path to the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))  # utils directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # project root
    csv_dir = os.path.join(project_root, "data", "csv")
    
    logger.info(f"Looking for CSV files in: {csv_dir}")
    csv_files = glob.glob(os.path.join(csv_dir, "sf_crime_*.csv.gz"))
    
    if not csv_files:
        raise FileNotFoundError(f"No crime data CSV files found in {csv_dir}")
    
    latest_csv = max(csv_files, key=os.path.getctime)
    logger.info(f"Using latest CSV file: {latest_csv}")
    return latest_csv

# Import crime data into MongoDB
async def import_crime_data():
    try:
        # MongoDB connection
        client = MongoClient("mongodb://localhost:27017")
        db = client.crime_forecast
        
        # Drop existing collection
        logger.info("Dropping existing collection...")
        db.incidents.drop()
        
        # Create indexes
        db.incidents.create_index("incident_datetime")
        db.incidents.create_index("incident_category")
        db.incidents.create_index([("location", GEOSPHERE)])
        logger.info("Created indexes")
        
        # Read latest CSV file
        csv_path = get_latest_csv()
        logger.info("Reading CSV file...")
        
        with gzip.open(csv_path, 'rt') as f:
            df = pd.read_csv(f)
        
        logger.info(f"Read {len(df)} records")
        
        # Process data
        df = process_crime_data(df)
        
        # Convert datetime to string for MongoDB
        df['incident_datetime'] = df['incident_datetime'].astype(str)
        
        # Convert to records and handle NA values
        records = df.replace({pd.NA: None}).to_dict('records')
        
        # Insert records in batches
        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            db.incidents.insert_many(batch)
            logger.info(f"Inserted {i + len(batch)} records")
        
        logger.info("Import complete!")
        
    except Exception as e:
        logger.error(f"Error importing crime data: {e}")
        raise
    finally:
        client.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(import_crime_data()) 
 