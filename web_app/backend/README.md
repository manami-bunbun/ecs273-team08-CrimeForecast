# Backend Setup and Running Instructions

## Prerequisites

- Python 3.11 or later
- MongoDB 7.0
- pip (Python package manager)

## Installation

1. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

2. Install required packages:
```bash
pip install -r ../../requirements.txt
pip install fastapi uvicorn motor python-dotenv
```

## MongoDB Setup

1. Create a directory for MongoDB data:
```bash
mkdir -p data/mongodb
```

2. Start MongoDB server:
```bash
mongod --dbpath data/mongodb
```

Note: If you encounter any port conflicts, you can kill existing MongoDB processes:
```bash
pkill mongod
```

## Running the Backend Server

1. Make sure MongoDB is running in a separate terminal

2. Start the FastAPI server:
```bash
uvicorn app:app --reload
```

The server will start at http://127.0.0.1:8000

## API Documentation

Once the server is running, you can access:
- API documentation: http://127.0.0.1:8000/docs
- Alternative documentation: http://127.0.0.1:8000/redoc

## Troubleshooting

1. If you see "Address already in use" error:
   - Kill the existing process using the port
   - Try using a different port: `uvicorn app:app --reload --port 8001`

2. If MongoDB fails to start:
   - Ensure no other MongoDB instance is running
   - Check if the data directory exists and has proper permissions
   - Verify MongoDB version compatibility with your OS

3. If you encounter module import errors:
   - Ensure you're in the correct directory (web_app/backend)
   - Verify all required packages are installed
   - Check if your virtual environment is activated 
