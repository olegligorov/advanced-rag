# Kubernetes Advanced RAG System

A full-stack Retrieval-Augmented Generation (RAG) system demonstrating advanced information retrieval techniques using Kubernetes documentation as a knowledge base. This project combines semantic and keyword search with hybrid retrieval and neural re-ranking for high-quality question answering.

## Features

### RAG Pipeline
- **Hybrid Chunking**: Combines header-based and semantic chunking using sentence embeddings and cosine distance
- **Dense Retrieval**: FAISS vector search with HuggingFace embeddings for semantic matching
- **Sparse Retrieval**: BM25 keyword search for traditional information retrieval
- **Hybrid Search**: Reciprocal Rank Fusion (RRF) to merge dense and sparse results
- **Neural Re-ranking**: Cross-encoder model for final ranking optimization
- **Streaming Responses**: Server-Sent Events (SSE) for real-time answer generation

### Full-Stack Application
- **Backend**: FastAPI server with REST API and streaming support
- **Frontend**: Modern React 19 + TypeScript interface with real-time chat
- **Evaluation**: RAGAS-based metrics for quality assessment
- **Caching**: Persistent caching of indices and embeddings

## Performance Metrics

### Quality Metrics
- **Average Faithfulness**: 0.986 (10/10 valid)
- **Average Answer Relevancy**: 0.931

### Retrieval Metrics
| Metric    | K=1  | K=3  | K=5  |
|-----------|------|------|------|
| Hit@K     | 0.85 | 0.92 | 1.00 |
| Recall@K  | 0.80 | 0.90 | 1.00 |

## Project Structure

```
kubernetes_advanced_rag/
├── backend/                      # Python FastAPI server
│   ├── models/
│   │   └── rag_pipeline.py      # Main RAG orchestrator
│   ├── rag/
│   │   ├── chunking.py          # Semantic chunking
│   │   ├── retrieval.py         # Hybrid retrieval (FAISS + BM25)
│   │   ├── reranking.py         # Cross-encoder re-ranking
│   │   └── generation.py        # LLM answer generation
│   ├── evaluation/
│   │   ├── metrics.py           # RAGAS metrics
│   │   └── evaluator.py         # Evaluation orchestrator
│   ├── scripts/
│   │   ├── run_evaluation.py   # Run evaluation on test sets
│   │   └── generate_test_dataset.py
│   ├── main.py                  # FastAPI server entry point
│   └── config.py                # Configuration constants
├── client/                       # React TypeScript frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── chat/            # Chat UI components
│   │   │   ├── welcome/         # Welcome screen & suggestions
│   │   │   └── ui/              # shadcn/ui components
│   │   ├── hooks/
│   │   │   └── useChat.ts       # Chat state management
│   │   └── services/
│   │       └── api.ts           # Axios API client
│   └── package.json
├── k8s_data/concepts/           # Kubernetes documentation (knowledge base)
├── advanced_rag_playground.ipynb # Original Jupyter notebook
└── README.md
```

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+
- Ollama (for local LLM) or Claude API access

### Backend Setup

1. Navigate to backend directory:
   ```bash
   cd backend
   ```

2. Create and activate virtual environment:
   ```bash
   python3 -m venv venv
   ```

   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

   On Windows:
   ```bash
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. Check config.py and set up the needed values.

6. Run the LLM model

7. Start the server:
   ```bash
   python main.py
   ```

   Or using uvicorn:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

The API will be available at `http://localhost:8000`
API Documentation: `http://localhost:8000/docs`

### Frontend Setup

1. Navigate to client directory:
   ```bash
   cd client
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start development server:
   ```bash
   vite
   ```

The application will be available at `http://localhost:3001`

## API Documentation

### Endpoints

#### Health Check
```http
GET /api/health
```

#### Query (Standard)
```http
POST /api/query
Content-Type: application/json

{
  "query": "What are Kubernetes pods?"
}
```

Response:
```json
{
  "answer": "Kubernetes pods are...",
  "sources": [
    {
      "content": "...",
      "metadata": {
        "source": "pods.md",
        "score": 0.85
      }
    }
  ],
  "query": "What are Kubernetes pods?"
}
```

#### Query (Streaming)
```http
POST /api/query/stream
Content-Type: application/json

{
  "query": "What are Kubernetes pods?"
}
```

Response: Server-Sent Events (SSE) stream with real-time answer generation.

## Configuration

Key configuration parameters in [backend/config.py](backend/config.py):

```python
# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama3"  # or "claude-sonnet-4.5"

# Retrieval Parameters
VECTOR_RETRIEVAL_K = 25
BM25_RETRIEVAL_K = 25
RERANK_TOP_N = 5
RERANK_SCORE_THRESHOLD = 0.3

# Chunking
CHUNK_PERCENTILE = 95
```

## Running Evaluation

Evaluate the RAG system on test datasets:

```bash
cd backend
python scripts/run_evaluation.py --dataset datasets/k8s_qa_test_set.json --output results/
```

Results are saved to timestamped JSON files in `backend/results/` with metrics including:
- Faithfulness
- Answer Relevancy
- Precision@K
- Recall@K
- Hit@K

## Technology Stack

### Backend
- **Framework**: FastAPI + Uvicorn
- **LLM/Embeddings**: HuggingFace Transformers, Ollama (llama3)
- **Vector Database**: FAISS (CPU)
- **Keyword Search**: BM25 (rank-bm25)
- **Evaluation**: RAGAS metrics
- **ML Libraries**: LangChain, sentence-transformers, PyTorch

### Frontend
- **Framework**: React 19 + TypeScript
- **Build Tool**: Vite 6
- **Styling**: Tailwind CSS v4
- **Components**: shadcn/ui + Radix UI
- **State Management**: TanStack Query v5
- **HTTP Client**: Axios

## Development Notes

### Device Support
- The system is optimized for macOS with Apple Silicon (MPS support)
- For other platforms, adjust the device parameter in model initialization:
  - CPU: `device="cpu"`
  - CUDA: `device="cuda"`

### Jupyter Notebook
The original research and experiments are available in [advanced_rag_playground.ipynb](advanced_rag_playground.ipynb). Run the notebook cells sequentially to:
- Load and chunk documents
- Build vector and BM25 retrievers
- Perform hybrid retrieval and re-ranking
- Visualize results and inspect answers

## Research References

### RAG Techniques
- [Awesome Generative AI Guide - RAG Research](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/research_updates/rag_research_table.md)
- [Advanced RAG Paper](https://arxiv.org/html/2510.12323v1)

### Reciprocal Rank Fusion (RRF)
- [Mathematical Intuition Behind RRF](https://medium.com/@devalshah1619/mathematical-intuition-behind-reciprocal-rank-fusion-rrf-explained-in-2-mins-002df0cc5e2a)

### Document Chunking
- [The Art of Document Chunking for LLM Applications](https://agentset.ai/blog/the-art-of-document-chunking-for-llm-applications)
- [RAG 2.0: Advanced Chunking Strategies](https://medium.com/@visrow/rag-2-0-advanced-chunking-strategies-with-examples-d87d03adf6d1)

### Hybrid Search
- [Building a Hybrid Search System](https://medium.com/@hitendra.patel2986/i-built-a-hybrid-search-system-that-beats-standard-rag-by-35-1968791ae539)

## Future Improvements

### Query Expansion
Implement query expansion techniques to improve retrieval performance:

```python
def expand_query(user_query: str):
    """Generate multiple search queries to improve recall"""
    prompt = f"Generate 3 different search queries to find information for: {user_query}. Output only the queries, one per line."
    response = ollama.generate(model='llama3', prompt=prompt)

    queries = response['response'].strip().split('\n')
    queries.append(user_query)  # Always include the original
    return queries
```

### Conversation Memory
Implement memory mechanisms to retain chat history and context across interactions for better UX.

### Multi-Mode Support
Add conversation mode alongside single-query mode to support different user workflows.
