from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os
import pickle
from pathlib import Path

from rag.chunking import SemanticChunker
from rag.retrieval import HybridRetriever
from rag.generation import Generator
from config import EMBEDDING_MODEL, CHUNK_PERCENTILE, CACHE_DIR

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation (RAG) pipeline orchestrator.

    This class manages the complete RAG workflow:
    1. Document loading from markdown files
    2. Semantic chunking to split documents at natural boundaries
    3. Building hybrid retrieval indices (vector + BM25)
    4. Query processing with retrieval and re-ranking
    5. LLM-based answer generation with retrieved context

    Key features:
    - Memory-efficient: Shares a single embedding model across all components
    - Modular: Each RAG component (chunking, retrieval, re-ranking) is independent
    - Production-ready: Loads all models and indices once on initialization

    The pipeline is initialized once (typically on server startup).
    """

    def __init__(self, dataDirectory, use_cache=True):
        """
        Initialize the RAG pipeline with document loading, chunking, and indexing.

        This initialization process runs once and prepares all components:
        1. Load embedding model (shared across components for memory efficiency)
        2. Load markdown documents from specified directory
        3. Chunk documents using semantic boundaries
        4. Build retrieval indices (FAISS vector DB + BM25 index)

        Args:
            dataDirectory (str): Path to directory containing markdown (.md) documents.
                Will recursively load all .md files from this directory.
            use_cache (bool): If True, attempts to load cached indices. If False or cache
                doesn't exist, builds indices from scratch and saves them.
        """
        print("Initializing RAG Pipeline...")

        self.data_directory = dataDirectory
        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.chunks_cache_path = self.cache_dir / "semantic_chunks.pkl"
        self.faiss_cache_path = self.cache_dir / "faiss_index"
        self.bm25_cache_path = self.cache_dir / "bm25_retriever.pkl"

        print("Loading embedding model...")
        self._hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        print("Initializing LLM generator...")
        self._generator = Generator()

        if use_cache and self._cache_exists():
            print("Loading indices from cache...")
            try:
                self._load_from_cache()
                print("RAG Pipeline initialized successfully from cache!")
                return
            except Exception as e:
                print(f"Cache loading failed: {e}. Building indices from scratch...")

        print("Building indices from scratch...")

        # 1. Load documents
        print("Loading documents...")
        self.__loader = DirectoryLoader(
            dataDirectory,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader)

        self._raw_docs = self.load_documents(self.__loader)

        # 2. Chunk documents 
        self.__chunker = SemanticChunker(embedding_model=self._hf_embeddings._client)
        self.semantic_docs = self.__chunker.create_semantic_chunks(
            self._raw_docs,
            percentile_threshold=CHUNK_PERCENTILE
        )

        # 3. Initialize retriever
        print("Building retrieval indices...")
        self._retriever = HybridRetriever(
            semantic_docs=self.semantic_docs,
            hf_embeddings=self._hf_embeddings
        )

        # Save to cache
        if use_cache:
            print("Saving indices to cache...")
            self._save_to_cache()

        print("RAG Pipeline initialized successfully!")
        
    def load_documents(self, loader):
        """
        Load raw documents using the configured document loader.

        Uses LangChain's DirectoryLoader with UnstructuredMarkdownLoader to:
        - Recursively find all .md files in the specified directory
        - Parse markdown formatting and structure
        - Extract metadata (source file path, etc.)
        - Create Document objects with page_content and metadata

        Args:
            loader (DirectoryLoader): Configured LangChain DirectoryLoader instance

        Returns:
            list[Document]: List of loaded Document objects with raw (unchunked) content.
                Each document represents one markdown file.
        """
        raw_docs = loader.load()
        print(f"Loaded {len(raw_docs)} documents.")
        return raw_docs

    def _cache_exists(self):
        """
        Check if all required cache files exist.

        Returns:
            bool: True if all cache files exist, False otherwise
        """
        return (
            self.chunks_cache_path.exists() and
            self.faiss_cache_path.exists() and
            self.bm25_cache_path.exists()
        )

    def _save_to_cache(self):
        """
        Save semantic chunks and retriever indices to disk for faster loading.

        Saves:
        - Semantic chunks (pickled documents)
        - FAISS vector index
        - BM25 retriever (pickled)
        """
        try:
            # Save semantic chunks
            with open(self.chunks_cache_path, 'wb') as f:
                pickle.dump(self.semantic_docs, f)

            # Save FAISS index
            self._retriever.vector_retriever.vectorstore.save_local(str(self.faiss_cache_path))

            # Save BM25 retriever
            with open(self.bm25_cache_path, 'wb') as f:
                pickle.dump(self._retriever.bm25_retriever, f)

            print(f"Indices cached successfully at {self.cache_dir}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def _load_from_cache(self):
        """
        Load semantic chunks and retriever indices from cache.

        This is much faster than rebuilding indices from scratch.
        Typically reduces startup time from ~30 seconds to ~2-3 seconds.
        """
        from langchain_community.vectorstores import FAISS

        # Load semantic chunks
        with open(self.chunks_cache_path, 'rb') as f:
            self.semantic_docs = pickle.load(f)

        print(f"Loaded {len(self.semantic_docs)} cached semantic chunks")

        # Recreate retriever with cached data
        print("Loading FAISS index from cache...")
        vector_db = FAISS.load_local(
            str(self.faiss_cache_path),
            self._hf_embeddings,
            allow_dangerous_deserialization=True
        )

        print("Loading BM25 index from cache...")
        with open(self.bm25_cache_path, 'rb') as f:
            bm25_retriever = pickle.load(f)

        # Create HybridRetriever with cached components
        from rag.retrieval import HybridRetriever
        self._retriever = HybridRetriever.__new__(HybridRetriever)
        self._retriever.vector_retriever = vector_db.as_retriever(search_kwargs={"k": 25})
        self._retriever.bm25_retriever = bm25_retriever

        # Initialize reranker (this is fast, doesn't need caching)
        from rag.reranking import Reranker
        self._retriever.reranker = Reranker()

    def retrieve(self, query: str, top_n: int = 5):
        """
        Retrieve relevant documents without generating an answer.

        Executes the hybrid retrieval and re-ranking workflow:
        1. Embeds the query using the same embedding model as documents
        2. Retrieves candidates via vector search (FAISS) and keyword search (BM25)
        3. Fuses results using Reciprocal Rank Fusion (RRF)
        4. Re-ranks fused results using cross-encoder model
        5. Returns top-n most relevant document chunks

        Args:
            query (str): User's question or search query
            top_n (int, optional): Number of top-ranked documents to return.
                Defaults to 5.

        Returns:
            list[Document]: Top-n most relevant document chunks, sorted by relevance.

        Example:
            >>> pipeline = RAGPipeline("./k8s_docs")
            >>> docs = pipeline.retrieve("How do I set memory limits?", top_n=3)
        """
        return self._retriever.search(query=query, top_n=top_n)

    def query(self, query: str, top_n: int = 5):
        """
        Process a user query through the complete RAG pipeline.

        Full RAG workflow:
        1. Hybrid Retrieval: Retrieve relevant documents using vector + BM25 search
        2. Re-ranking: Re-rank results with cross-encoder for maximum relevance
        3. Generation: Generate natural language answer using LLM with retrieved context

        Args:
            query (str): User's question or search query
            top_n (int, optional): Number of top-ranked documents to retrieve.
                Defaults to 5. These documents provide context for the LLM.

        Returns:
            dict with 'answer', 'sources'
        """
        # Retrieve relevant documents
        retrieved_docs = self._retriever.search(query=query, top_n=top_n)

        # Generate answer using LLM with retrieved context
        result = self._generator.generate(
            query=query,
            documents=retrieved_docs,
            include_sources=True
        )

        return result
