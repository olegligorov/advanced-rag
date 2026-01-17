
from sentence_transformers import CrossEncoder
import torch
from typing import List

class Reranker:
    """
    Cross-encoder based document re-ranker.
    Re-ranks retrieved documents based on query-document relevance.
    """
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize the re-ranker with a cross-encoder model.
        
        Args:
            model_name: HuggingFace cross-encoder model name
        """
        # Auto detect best available device
        self.device = self._detect_device()
        
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=self.device)
        
        print(f"Reranker initialized: {model_name} on {self.device}")
        
    def _detect_device(self) -> str:
        """
        Auto detect the best available device.
        Priority: MPS (Mac) > CUDA (GPU) > CPU
        """
        
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        
        return 'cpu'
    
    def rerank(self, query: str, documents: List, top_n: int = 5) -> List:
        """
        Re-rank documents using cross-encoder model.
        
        Args:
            query: Search query string
            documents: List of LangChain Document objects
            top_n: Number of top documents to return
        
        Returns:
            List of top-n re-ranked documents (sorted by relevance)
        """
        
        if not documents:
            return []
        
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        
        # Sort by scores
        
        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        top_docs = [doc for score, doc in scored_docs[:top_n]]
        
        return top_docs
    
    def rerank_with_scores(self, query: str, documents: List, top_n: int = 5):
        """
        Re-rank documents and return both documents and scores.
        Useful for debugging and visualization.
        
        Args:
            query: Search query string
            documents: List of LangChain Document objects
            top_n: Number of top documents to return
        
        Returns:
            List of tuples: [(score, document), ...]
        """
        if not documents:
            return []
        
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        
        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        
        return scored_docs[:top_n]