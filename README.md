# Kubernetes Advanced RAG Playground

This project demonstrates advanced Retrieval-Augmented Generation (RAG) techniques using Kubernetes documentation as a knowledge base. It combines semantic and keyword search, hybrid retrieval, and neural re-ranking for high-quality information retrieval.

## Features
- **Semantic Chunking**: Splits documents into topic-based segments using sentence embeddings and cosine distance.
- **Dense Retrieval**: Uses FAISS and HuggingFace embeddings for semantic search.
- **Sparse Retrieval**: BM25 keyword search for traditional IR.
- **Hybrid Search**: Reciprocal Rank Fusion (RRF) to combine dense and sparse results.
- **Neural Re-ranking**: Cross-encoder model for final ranking of results.
- **Visualization**: Plots for semantic distance, chunking, and hybrid search overlap.

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
