# **Project: Domain-Specific Reasoning Engine (Advanced RAG)**

## **1\. Project Overview**

Goal: Build a "Reasoning Engine" that answers complex user queries from a closed-domain dataset (e.g., technical manuals, legal docs).  
Differentiation: Unlike basic RAG wrappers, this project implements custom NLP segmentation algorithms (Semantic Chunking) and advanced IR ranking pipelines (Hybrid Search \+ Cross-Encoder Re-ranking).

## **2\. Architecture & Tech Stack**

### **High-Level Architecture**

The system follows a "Funnel Architecture" to maximize Precision and Recall:

1. **Ingestion:** Raw Text \-\> NLP Semantic Segmentation \-\> Intelligent Chunks.  
2. **Indexing:** Chunks are indexed twice:  
   * **Dense Index:** Vector Embeddings (for semantic meaning).  
   * **Sparse Index:** Inverted Index/BM25 (for exact keywords).  
3. **Retrieval:**  
   * Query Expansion (NLP) transforms user input.  
   * Parallel Search (Vector \+ BM25) retrieves top 50 candidates.  
   * **Fusion:** Reciprocal Rank Fusion (RRF) merges results.  
4. **Ranking:** Cross-Encoder model re-scores the top 50 to find the best 5\.  
5. **Generation:** LLM answers the query based *only* on the top 5 chunks.

### **Architecture Diagram:** ![Architecture Diagram](image.png)

### **Tech Stack**

* **Language:** Python
* **Orchestration:** LangChain (or pure Python functions)  
* **Vector DB:** ChromaDB (Local)  
* **Embeddings:** sentence-transformers/all-MiniLM-L6-v2  
* **Sparse Search:** rank\_bm25  
* **Re-Ranking:** cross-encoder/ms-marco-MiniLM-L-6-v2  
* **LLM:** Llama 3 (via Ollama or Groq) or OpenAI  
* **Frontend:** React or Angular or Streamlit

## **3\. Implementation Roadmap**

### **Phase 1: Environment & Data Setup**

* \[ \] Set up virtual environment and install dependencies (chromadb, sentence-transformers, rank\_bm25, langchain, setup FE).  
* \[ \] Acquire a clean dataset (e.g., a PDF of a textbook or technical manual).  
* \[ \] Implement a PDF text extractor to get raw text.

### **Phase 2: The NLP Engine (Semantic Chunking)**

*Crucial for NLP Subject Credit*

* \[ \] **Do not** use standard fixed-size splitters.  
* \[ \] Implement **Semantic Chunking**:  
  1. Split text into sentences.  
  2. Embed every sentence individually.  
  3. Calculate Cosine Similarity between Sentence $N$ and $N+1$.  
  4. Detect "valleys" (drops in similarity) to identify topic shifts.  
  5. Group sentences into chunks based on these boundaries.

### **Phase 3: The IR Engine (Hybrid Search)**

*Crucial for IR Subject Credit*

* \[ \] Initialize ChromaDB for Vector Search (Dense).  
* \[ \] Initialize BM25 for Keyword Search (Sparse).  
* \[ \] Implement **Reciprocal Rank Fusion (RRF)** function:  
  * Take the ranked list from Vector search and BM25.  
  * Apply formula: $Score = 1 / (k + rank)$.  
  * Sort and return merged unique documents.

### **Phase 4: Re-Ranking (Precision Layer)**

* \[ \] Load a Cross-Encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2).  
* \[ \] Create a function that takes {Query} \+ {Top\_20\_RRF\_Results}.  
* \[ \] Score every pair.  
* \[ \] Return the Top 5 highest-scored chunks.

### **Phase 5: Generation & RAG**

* \[ \] Setup LLM Client (Groq/Ollama).  
* \[ \] Construct the Prompt Template:  
  Context: {top\_5\_chunks}  
  Question: {user\_query}  
  Instructions: Answer strictly based on the context. If the answer is not there, say "I don't know".

### **Phase 6: Evaluation & UI**

* \[ \] Build a simple UI with a chat interface.  
* \[ \] (Optional) Implement an Evaluation script using RAGAS to measure Faithfulness and Relevance.

## **4\. Key Algorithms & Logic**

### **A. Semantic Chunking Logic**

Standard chunking breaks context. We must group sentences that are semantically similar.

* **Threshold:** Define a percentile (e.g., 20th percentile of similarity scores) as the breakpoint.  
* **Window:** Compare current sentence to the next sentence (Window=1).

### **B. Reciprocal Rank Fusion (RRF)**

We fuse two lists to avoid bias towards either vectors or keywords. 
 
$$ RRF\_score(d) = \sum_{r \in R} \frac{1}{k + rank(d, r)} $$

* Where $k$ is a constant (usually 60).  
* High BM25 rank \+ Low Vector rank \= Medium Score.  
* High BM25 rank \+ High Vector rank \= Very High Score.

### **C. Cross-Encoder Re-ranking**

Bi-encoders (used for retrieval) are fast because they embed query and document separately. Cross-encoders are slow but accurate because they process \[Query\] \[SEP\] \[Document\] simultaneously in the Transformer self-attention layers. We only use this on the final small set of candidates.

## **5\. Directory Structure**

/project-root  
├── /data                \# Raw PDFs and processed .json chunks  
├── /src  
│   ├── ingestion.py     \# Semantic Chunking & PDF loading  
│   ├── retrieval.py     \# Hybrid Search & RRF implementation  
│   ├── ranking.py       \# Cross-Encoder logic  
│   ├── generation.py    \# LLM Interface  
│   └── utils.py         \# Helper functions  
├── app.py               \# Streamlit Frontend  
├── Project.md           \# This file  
└── requirements.txt  

## 5. Evaluation Strategy

To ensure scientific rigor, the system is evaluated on two distinct levels: the quality of the information retrieved (IR) and the quality of the generated response (NLP). A "Golden Dataset" of ground-truth Question-Answer pairs is used for testing.

### Level 1: Information Retrieval (IR) Metrics
*Evaluating the Hybrid Search & Re-ranking Engine*

* **Hit Rate @ 5:**
    * *Definition:* A binary metric measuring if the ground-truth document appears *at least once* in the top 5 retrieved chunks.
    * *Goal:* Ensures the LLM actually receives the correct context required to answer the question.
* **Precision @ K:**
    * *Definition:* The ratio of relevant documents retrieved to the total number of documents retrieved (at $k=5$).
    * *Goal:* Measures the "signal-to-noise" ratio of the retrieval pipeline.
* **Recall @ K:**
    * *Definition:* The ratio of relevant documents retrieved to the total number of relevant documents existing in the database.
    * *Goal:* Measures the system's ability to find *all* distributed information (e.g., if an answer requires combining facts from two different pages).

### Level 2: Generation (NLP) Metrics
*Evaluating the LLM using the [RAGAS Framework](https://github.com/explodinggradients/ragas)*

* **Faithfulness (Hallucination Index):**
    * *Definition:* Measures whether the generated answer is derived *solely* from the retrieved context.
    * *Goal:* To detect and penalize "hallucinations" (facts invented by the model that are not present in the source text).
* **Answer Relevance:**
    * *Definition:* Measures the semantic similarity between the generated answer and the user's original query.
    * *Goal:* Ensures the system provides a direct answer rather than a generic or evasive response.