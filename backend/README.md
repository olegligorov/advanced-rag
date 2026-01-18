# Backend - Advanced RAG Server

FastAPI REST API server for the Advanced RAG system with hybrid search, semantic chunking, and re-ranking.

## Setup

### 1. Create Virtual Environment
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
cp .env.example .env
```

## Running the Server

```bash
# Option 1: Direct Python
python main.py

# Option 2: Using uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will start on `http://localhost:8000`

## API Documentation

Once the server is running:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Health Check
```bash
GET /api/health

Response:
{
  "status": "ok",
  "message": "RAG server is running"
}
```

## Testing with curl

```bash
curl http://localhost:8000/api/health
```


Send a test query to the RAG pipeline:
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I configure Pod limits?" }'
```

## Project Structure

```
backend/
├── main.py              # FastAPI server entry point
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
└── README.md           # This file
```