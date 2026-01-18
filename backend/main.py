"""
FastAPI server for Plug and Play RAG System
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List

from config import DATA_PATH, RERANK_TOP_N
from models.rag_pipeline import RAGPipeline

rag_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the RAG pipeline on server startup.
    Loads all models and builds indices once.
    """
    print("Starting up... Initializing RAG Pipeline...")
    global rag_pipeline
    rag_pipeline = RAGPipeline(dataDirectory=DATA_PATH)
    print("RAG Pipeline ready!")

    yield

    print("Shutting down...")

app = FastAPI(
    title="Plug and Play RAG API",
    description="Plug and Play documentation RAG system",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")
    top_n: int = Field(RERANK_TOP_N, ge=1, le=20, description="Number of documents to retrieve")

class SourceResponse(BaseModel):
    rank: int
    source: str
    snippet: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceResponse]


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint to verify server and pipeline status.
    """
    return {
        "status": "ok",
        "message": "server is running",
        "pipeline_loaded": rag_pipeline is not None
    }

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query the RAG system with a question.
    Returns retrieved and re-ranked documents.
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Please wait for startup to complete."
        )

    try:
        result = rag_pipeline.query(query=request.question, top_n=request.top_n)

        sources = [
            SourceResponse(
                rank=source["rank"],
                source=source["source"],
                snippet=source["snippet"]
            )
            for source in result["sources"]
        ]

        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=sources,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
