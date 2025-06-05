# ecs273-team08-CrimeForecast

##  Project Structure & Task Overview
See [`Implementation_Plan.md` ](https://github.com/manami-bunbun/ecs273-team08-CrimeForecast/blob/main/Implementation_Plan.md)

## Description
(Describe the repository in a few paragraphs)



## Installation

- **Python version**: 3.11.x

### Setup (with pyenv + venv / conda)

```zsh
# install correct Python version
pyenv install 3.11.8
pyenv local 3.11.8

# create virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

deactivate virtual environment
```zsh
deactivate
```

### Environment Variables
Copy the .env.example template and fill in required values 
```
cp .env.example .env
```

You need open ai api key to run this system. [Detail](https://platform.openai.com/api-keys)




## Execution

### Data setup
**Data Pipeline Execution**
```zsh
# Run the main data pipeline
python3 data_preprocessing/crime_data_pipeline.py

# View processed data
python3 data_preprocessing/view_crime_data.py
```

### 1. MongoDB Setup


1. Start MongoDB service:
   ```zsh
   # For macOS
   brew services start mongodb-community

   # For Ubuntu
   sudo systemctl start mongodb
   ```

2. Verify MongoDB is running:
   ```zsh
   mongosh
   ```

3. Import crime data to MongoDB:
   ```zsh
   # Navigate to the import script directory
   cd web_app/backend/utils
   
   # Run the import script
   python import_data.py
   ```

   This script will:
   - Create necessary indexes
   - Import crime data from the CSV file
   - Process and clean the data
   - Insert records into MongoDB

4. Verify data import:
   ```zsh
   # Connect to MongoDB shell
   mongosh
   
   # Switch to crime_forecast database
   use crime_forecast
   
   # Check the number of imported records
   db.incidents.count()
   ```

### 2. Backend Setup & Run

1. Navigate to backend directory:
   ```zsh
   cd web_app/backend
   ```


2. Run the backend server:
   ```bash
   uvicorn app:app --reload --port 8001
   ```

The backend will be available at `http://localhost:8001`

### 3. Frontend Setup & Run

1. Navigate to frontend directory:
   ```zsh
   cd web_app/frontend
   ```

2. Install dependencies:
   ```zsh
   npm install
   ```

3. Run the development server:
   ```zsh
   npm run dev
   ```

The frontend will be available at `http://localhost:5173`

### 4. Accessing the Application

1. Open your browser and visit `http://localhost:5173`

2. The application should now show:
   - Crime heatmap visualization
   - Crime category statistics
   - Related news selected by LLM and LLM advice



