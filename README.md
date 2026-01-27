# Kubernetes Advanced RAG Playground

This project demonstrates advanced Retrieval-Augmented Generation (RAG) techniques using Kubernetes documentation as a knowledge base. It combines semantic and keyword search, hybrid retrieval, and neural re-ranking for high-quality information retrieval.

## Features
- **Semantic Chunking**: Splits documents into topic-based segments using sentence embeddings and cosine distance.
- **Dense Retrieval**: Uses FAISS and HuggingFace embeddings for semantic search.
- **Sparse Retrieval**: BM25 keyword search for traditional IR.
- **Hybrid Search**: Reciprocal Rank Fusion (RRF) to combine dense and sparse results.
- **Neural Re-ranking**: Cross-encoder model for final ranking of results.
- **Visualization**: Plots for semantic distance, chunking, and hybrid search overlap.

## Faithfulness and Relevancy Results
Average Faithfulness: 0.986 (10/10 valid)
Average Answer Relevancy: 0.931

## Hit@K and Recall@K
| Metric       | K=1   | K=3   | K=5   |
|--------------|-------|-------|-------|
| Hit@K  | 0.85  | 0.92  | 1  |
| Recall@K     | 0.80  | 0.90  | 1  |

## Folder Structure
- `advanced_rag_playground.ipynb`: Main notebook with all code and experiments.
- `k8s_data/concepts/`: Markdown files from Kubernetes documentation, used as the knowledge base.
- `requirements.txt`: Python dependencies for the project.

## Setup
1. Clone this repository.
2. Create and activate a virtual environment:
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
4. Download or place Kubernetes markdown docs in `k8s_data/concepts/`.
5. Open `advanced_rag_playground.ipynb` in Jupyter or VS Code.

## Usage
- Run the notebook cells sequentially to:
  - Load and chunk documents
  - Build vector and BM25 retrievers
  - Perform hybrid retrieval and re-ranking
  - Visualize results and inspect top answers

## Requirements
See `requirements.txt` for all dependencies. Key libraries:
- langchain_community
- langchain_core
- sentence-transformers
- scikit-learn
- matplotlib
- numpy
- pandas
- faiss-cpu
- langchain_huggingface

## Notes
- The notebook is optimized for macOS and supports Apple Silicon (MPS) for neural models.
- Switch the device parameter in the CrossEncoder initialization to "cpu" or "cuda" if not using a Mac with MPS support.


## Reads: 
RAG: https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/research_updates/rag_research_table.md
https://arxiv.org/html/2510.12323v1


RRF: https://medium.com/@devalshah1619/mathematical-intuition-behind-reciprocal-rank-fusion-rrf-explained-in-2-mins-002df0cc5e2a

Chunking: https://agentset.ai/blog/the-art-of-document-chunking-for-llm-applications
https://medium.com/@visrow/rag-2-0-advanced-chunking-strategies-with-examples-d87d03adf6d1

Hybrid Search: https://medium.com/@hitendra.patel2986/i-built-a-hybrid-search-system-that-beats-standard-rag-by-35-1968791ae539 

## Future improvements
Check query expansion techniques to improve retrieval performance.
smt like
```
def expand_query(user_query: str):
    # Ask the LLM to generate 3 different search queries
    prompt = f"Generate 3 different search queries to find information for: {user_query}. Output only the queries, one per line."
    response = ollama.generate(model='llama3', prompt=prompt)
    
    # Split the response into a list of strings
    queries = response['response'].strip().split('\n')
    queries.append(user_query) # Always include the original
    return queries
```

For a better UX, consider implementing memory mechanisms to retain chat history and context across interactions.

Or create it as a conversation mode, so both will be supported
