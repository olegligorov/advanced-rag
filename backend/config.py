"""
Configuration file for RAG system
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

# Set to None to disable filtering, or use a value like 0.3-0.5 for quality filtering
# RERANK_SCORE_THRESHOLD = 0.5
RERANK_SCORE_THRESHOLD = 1
MIN_RETRIEVED_DOCS = 1

# Chunking parameters
CHUNK_PERCENTILE = 95

# Server configuration
FLASK_HOST = os.getenv("HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("PORT", 8000))

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_TEMPERATURE = 0.2

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Evaluation settings
EVALUATION_LLM = LLM_MODEL  # Reuse same LLM for evaluation
TEST_DATASET_PATH = os.path.join(BASE_DIR, "backend", "datasets", "k8s_qa_test_set.json")
EVALUATION_RESULTS_DIR = os.path.join(BASE_DIR, "backend", "results")

# Change to false, then it will use ollama model
USE_PROXY=True
PROXY_SONNET_MODEL=os.getenv("PROXY_SONNET_MODEL", "claude-sonnet-4-5-20250929")
PROXY_URL=os.getenv("PROXY_URL", "http://localhost:3030")
PROXY_API_KEY=os.getenv("PROXY_API_KEY")
 