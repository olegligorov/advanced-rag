"""
Configuration file for RAG system
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(BASE_DIR, "k8s_data", "concepts")
CACHE_DIR = os.path.join(BASE_DIR, "backend", ".cache", "indices")

# Model configurations
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama3"

# Retrieval parameters
VECTOR_RETRIEVAL_K = 25
BM25_RETRIEVAL_K = 25
RERANK_TOP_N = 5

# Chunking parameters
CHUNK_PERCENTILE = 95

# Server configuration
FLASK_HOST = os.getenv("HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("PORT", 8000))

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TEMPERATURE = 0.2

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
