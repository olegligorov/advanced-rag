from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from rag.chunking import SemanticChunker
from rag.retrieval import HybridRetriever

class RAGPipeline:
    def __init__(self, dataDirectory):
        # TODO check which things we will need and which should be stores
        # TODO remove unused vals from function returns

        print("Initializing RAG Pipeline...")

        print("Loading embedding model...")
        # self._sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        # self._hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Try to use one model to decrease ram usage
        self._hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
        self.semantic_docs = self.__chunker.create_semantic_chunks(self._raw_docs)

        # 3. Initialize retriever (pass shared HF embeddings)
        print("Building retrieval indices...")
        self._retriever = HybridRetriever(
            semantic_docs=self.semantic_docs,
            hf_embeddings=self._hf_embeddings
        )

        print("RAG Pipeline initialized successfully!")
        
    def load_documents(self, loader):
        raw_docs = loader.load()
        print(f"Loaded {len(raw_docs)} documents.")
        return raw_docs

    def query(self, query: str, top_n: int = 5):
        """
        Process a query through the RAG Pipeline.
        
        Args:
            query: User's question
            top_n: Number of docs to retrieve
        
        Returns:
            Retrieved and re-ranked documents
        """
        
        top_docs = self._retriever.search(query=query, top_n=top_n)
        
        return top_docs
