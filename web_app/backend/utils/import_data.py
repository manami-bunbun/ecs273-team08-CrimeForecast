import pandas as pd
import gzip
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
from datetime import datetime
import os

async def import_crime_data():
    # MongoDB connection
    MONGO_URL = "mongodb://localhost:27017"
    client = AsyncIOMotorClient(MONGO_URL)
    db = client.crime_forecast

    
    await db.incidents.drop()

    
    csv_path = "../../data/csv/sf_crime_20240529_20250523.csv.gz"
    
    print("Reading CSV file...")
    with gzip.open(csv_path, 'rt') as f:
        df = pd.read_csv(f)
    
    print(f"Read {len(df)} records")


    df['incident_datetime'] = pd.to_datetime(df['incident_datetime'])
    

    records = df.to_dict('records')
    
    print("Inserting records into MongoDB...")

    batch_size = 1000
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        await db.incidents.insert_many(batch)
        print(f"Inserted {i + len(batch)} records")
    

    print("Creating index...")
    await db.incidents.create_index("incident_datetime")
    
    print("Import complete!")

if __name__ == "__main__":
    asyncio.run(import_crime_data()) 
