from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from rag.chunking import SemanticChunker
from rag.retrieval import HybridRetriever
from config import EMBEDDING_MODEL, CHUNK_PERCENTILE

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation (RAG) pipeline orchestrator.

    This class manages the complete RAG workflow:
    1. Document loading from markdown files
    2. Semantic chunking to split documents at natural boundaries
    3. Building hybrid retrieval indices (vector + BM25)
    4. Query processing with retrieval and re-ranking

    Key features:
    - Memory-efficient: Shares a single embedding model across all components
    - Modular: Each RAG component (chunking, retrieval, re-ranking) is independent
    - Production-ready: Loads all models and indices once on initialization

    The pipeline is initialized once (typically on server startup).
    """

    def __init__(self, dataDirectory):
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
        """
        # TODO check which things we will need and which should be stores
        # TODO remove unused vals from function returns

        print("Initializing RAG Pipeline...")

        print("Loading embedding model...")
        # Try to use one model to decrease ram usage
        self._hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self._sentence_transformer = self._hf_embeddings.client

        # 1. Load documents
        print("Loading documents...")
        self.__loader = DirectoryLoader(
            dataDirectory,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader)

        self._raw_docs = self.load_documents(self.__loader)

        # 2. Chunk documents (pass shared embedding model)
        self.__chunker = SemanticChunker(embedding_model=self._sentence_transformer)
        self.semantic_docs = self.__chunker.create_semantic_chunks(
            self._raw_docs,
            percentile_threshold=CHUNK_PERCENTILE
        )

        # 3. Initialize retriever (pass shared HF embeddings)
        print("Building retrieval indices...")
        self._retriever = HybridRetriever(
            semantic_docs=self.semantic_docs,
            hf_embeddings=self._hf_embeddings
        )

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

    def query(self, query: str, top_n: int = 5):
        """
        Process a user query through the complete RAG retrieval pipeline.

        Executes the full hybrid retrieval and re-ranking workflow:
        1. Embeds the query using the same embedding model as documents
        2. Retrieves candidates via vector search (FAISS) and keyword search (BM25)
        3. Fuses results using Reciprocal Rank Fusion (RRF)
        4. Re-ranks fused results using cross-encoder model
        5. Returns top-n most relevant document chunks

        This method is the main interface for retrieving context documents
        that will be used to generate answers with an LLM.

        Args:
            query (str): User's question or search query
            top_n (int, optional): Number of top-ranked documents to return.
                Defaults to 5. These documents typically provide enough context
                for LLM generation without exceeding context window limits.

        Returns:
            list[Document]: Top-n most relevant document chunks, sorted by relevance.
                Each document contains:
                - page_content: The text content of the chunk
                - metadata: Source file, chunk position, etc.

        Example:
            >>> pipeline = RAGPipeline("./k8s_docs")
            >>> docs = pipeline.query("How do I set memory limits?", top_n=3)
            >>> for doc in docs:
            ...     print(f"Source: {doc.metadata['source']}")
            ...     print(f"Content: {doc.page_content[:100]}...")
        """

        top_docs = self._retriever.search(query=query, top_n=top_n)

        return top_docs
