from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import hashlib
from typing import List

from .reranking import Reranker
class HybridRetriever:
    def __init__(self, semantic_docs, hf_embeddings=None):
        """
        Initialize HybridRetriever with vector and BM25 retrievers.

        Args:
            semantic_docs: List of chunked documents
            hf_embeddings: Optional HuggingFaceEmbeddings instance. If None, creates a new one.
        """
        # TODO set this as a config variable
        if hf_embeddings is None:
            hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vector_db = FAISS.from_documents(
            documents=semantic_docs,
            embedding=hf_embeddings
        )
        self.vector_retriever = vector_db.as_retriever(search_kwargs={"k": 25})
        
        self.bm25_retriever = BM25Retriever.from_documents(documents=semantic_docs)
        # TODO set this as a config variable
        self.bm25_retriever.k = 25
        
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
        Generates a deterministic ID for a document.
        Uses 'source' metadata if available, combined with a content hash.
        """
        
        content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
        if 'source' in doc.metadata:
            return f"{doc.metadata['source']}_{content_hash[:16]}"
        
        return content_hash
    
    def search(self, query: str, top_n =5) -> List:
        """
        Perform hybrid search with re-ranking.
        
        Args:
            query: search query
            top_n: Number of final documents to return after re-ranking
            
        Returns:
            List of top_n re-ranked documents
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
        
    
    