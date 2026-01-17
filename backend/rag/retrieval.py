from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import hashlib
from typing import List

from .reranking import Reranker
from config import EMBEDDING_MODEL, VECTOR_RETRIEVAL_K, BM25_RETRIEVAL_K, RERANK_TOP_N
class HybridRetriever:
    """
    Hybrid retrieval system combining dense vector search and sparse keyword search.

    This retriever implements a state-of-the-art approach that combines:
    - Dense retrieval: FAISS vector database with semantic embeddings
    - Sparse retrieval: BM25 algorithm for keyword matching
    - Reciprocal Rank Fusion (RRF): Fuses results from both retrievers
    - Cross-encoder re-ranking: Final re-ranking for optimal relevance

    The hybrid approach is more robust than either method alone because:
    - Vector search captures semantic meaning and context
    - BM25 excels at exact term matching and rare keywords
    - RRF fusion leverages strengths of both methods
    - Re-ranking provides final quality control
    """

    def __init__(self, semantic_docs, hf_embeddings=None):
        """
        Initialize HybridRetriever with vector and BM25 retrievers.

        Args:
            semantic_docs (list[Document]): List of chunked documents to index for retrieval.
            hf_embeddings (HuggingFaceEmbeddings, optional): Pre-loaded HuggingFaceEmbeddings instance.
                If None, creates a new embedding model using EMBEDDING_MODEL from config.
                Passing a pre-loaded model saves memory when sharing across components.
        """
        if hf_embeddings is None:
            hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        vector_db = FAISS.from_documents(
            documents=semantic_docs,
            embedding=hf_embeddings
        )
        self.vector_retriever = vector_db.as_retriever(search_kwargs={"k": VECTOR_RETRIEVAL_K})

        self.bm25_retriever = BM25Retriever.from_documents(documents=semantic_docs)
        self.bm25_retriever.k = BM25_RETRIEVAL_K
        
        # Reranked
        self.reranker = Reranker()
        
    def rrf(self, vector_results, bm25_results, k=60):
        """
        Implements a Reciprocal Rank Fusion.
        k is smoothing constant. Standard is 60.
        RRF = sum(1 / (k + rank))
        Optimized to avoid duplicate Document objects.
    
        Studies have shown that k = 60 performs well across various datasets and retrieval tasks.
        It provides a good balance between the influence of top-ranked and lower-ranked items. For example:
        - For rank 1: 1/(1+60) ≈ 0.0164
        - For rank 10: 1/(10+60) ≈ 0.0143
        - For rank 100: 1/(100+60) ≈ 0.00625

        k = 60 helps break ties effectively, especially for lower-ranked items where small differences in the original rankings might not be significant.
        This value has shown to be robust across different types of retrieval systems and data distributions.
        """
    
        rrf_scores = {}
        doc_map = {}
        
        for rank, doc in enumerate(vector_results, start=1):
            doc_id = self._get_id(doc)
            doc_map[doc_id] = doc
        #   todo check if some alfa value would help here
	    #  * but for technical documentation maybe not needed
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
            
        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = self._get_id(doc)
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
            
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, score in sorted_ids]    
        
    def _get_id(self, doc):
        """
        Generate a deterministic unique ID for a document.

        Uses MD5 hash of document content to ensure uniqueness. If 'source' metadata
        exists (e.g., filename), combines it with a truncated hash for readability.

        This is critical for RRF fusion to correctly identify duplicate documents
        retrieved by both vector and BM25 methods.

        Args:
            doc (Document): LangChain Document object

        Returns:
            str: Unique identifier in format "source_hash16" or just "hash" if no source

        Example:
            doc with source="configmaps.md" and content "..."
            -> "configmaps.md_a1b2c3d4e5f6g7h8"
        """
        
        content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
        if 'source' in doc.metadata:
            return f"{doc.metadata['source']}_{content_hash[:16]}"
        
        return content_hash
    
    def search(self, query: str, top_n=RERANK_TOP_N) -> List:
        """
        Perform hybrid search with re-ranking to retrieve most relevant documents.

        Pipeline:
        1. Vector Retrieval: Retrieve top-k documents using FAISS semantic search
        2. BM25 Retrieval: Retrieve top-k documents using BM25 keyword matching
        3. RRF Fusion: Combine and rank results using Reciprocal Rank Fusion
        4. Re-ranking: Apply cross-encoder model to re-rank fused results
        5. Return: Top-n highest scoring documents after re-ranking

        This multi-stage approach ensures both semantic relevance and keyword precision.

        Args:
            query (str): User's search query or question
            top_n (int, optional): Number of final documents to return after re-ranking.
                Defaults to RERANK_TOP_N (5).

        Returns:
            list[Document]: Top-n most relevant documents, sorted by relevance score.
                Each document includes page_content and metadata (source, etc.)

        Example:
            retriever.search("How do I configure Pod resource limits?", top_n=3)
            -> [doc1, doc2, doc3] sorted by relevance
        """        
        
        # First retrieve vector and bm25 results
        v_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # Fuse them with rrf
        fused_results = self.rrf(vector_results=v_docs, bm25_results=bm25_docs)
        
        # print(self._visualize_topdocs(top_docs))
        
        # Re-rank the results from the fusion
        top_docs = self.reranker.rerank(query, fused_results, top_n=top_n)

        # todo maybe visualize the top_docs
        return top_docs        
        
    
    