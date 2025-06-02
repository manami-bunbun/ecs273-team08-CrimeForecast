import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns

def read_file():
    """
    Reads the latest sf_crime_*.csv.gz file from ../data/csv/ and returns a cleaned DataFrame.
    """
    os.makedirs("outputs", exist_ok=True)
    try:
        csv_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data", "csv")
        print("📂 CSV directory:", csv_dir)
        csv_files = glob.glob(os.path.join(csv_dir, "sf_crime_*.csv.gz"))
        print("📄 CSV files found:", csv_files)

        if not csv_files:
            print("❌ No CSV files found.")
            return None

        file_path = max(csv_files, key=os.path.getctime)

        df = pd.read_csv(file_path, compression='gzip', low_memory=False)
        print(f"✅ Loaded file: {file_path}")

        df['incident_datetime'] = pd.to_datetime(df['incident_datetime'], errors='coerce')
        df['incident_hour'] = df['incident_datetime'].dt.hour
        df['incident_month'] = df['incident_datetime'].dt.month

        df = df.dropna(subset=['latitude', 'longitude', 'incident_datetime', 'incident_category'])

        print(f"✅ df shape: {df.shape}")
        print(f"📊 df columns: {df.columns.tolist()}")

        return df

    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

if __name__ == "__main__":
    df = read_file()
    if df is not None:
        print("✅ DataFrame loaded successfully and ready for use.")
    else:
        print("❌ Failed to load DataFrame.")
