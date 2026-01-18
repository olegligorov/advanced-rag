"""
LLM generation module for RAG system.

Handles answer generation using retrieved context documents and Ollama LLM.
"""

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from typing import List, Dict
from backend.templates import SYSTEM_TEXT_TEMPLATE
from config import LLM_MODEL, OLLAMA_HOST, OLLAMA_TEMPERATURE


class Generator:
    """
    LLM-based answer generator for RAG system.

    Takes retrieved context documents and a user query, then generates
    a natural language answer using an LLM (via Ollama).
    """

    def __init__(self, model_name: str = LLM_MODEL, temperature: float = OLLAMA_TEMPERATURE):
        """
        Initialize the generator with an Ollama LLM.

        Args:
            model_name (str): Name of the Ollama model (e.g., "llama3", "mistral")
            temperature (float): Sampling temperature (0.0 = deterministic, 1.0 = creative)
        """
        self.llm = Ollama(
            model=model_name,
            base_url=OLLAMA_HOST,
            temperature=temperature
        )

        self.system_template =  SYSTEM_TEXT_TEMPLATE
        self.doc_prompt = PromptTemplate.from_template(
            "--- Document (Source: {source}) ---\n{page_content}\n--- END OF DOCUMENT ---"
        )
        
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            ("human", "{question}")
        ])
        
        self.combine_docs_chain = create_stuff_documents_chain(
            self.llm, 
            self.qa_prompt, 
            document_prompt=self.doc_prompt
        )

        print(f"Generator initialized: {model_name} (temperature={temperature})")

    def generate(self, query: str, documents: List, include_sources: bool = True) -> Dict:
        """
        Generate an answer to the query using retrieved documents as context.

        Args:
            query (str): User's question
            documents (list[Document]): Retrieved context documents (from retriever)
            include_sources (bool): Whether to include source documents in response

        Returns:
            dict: Response containing:
                - answer (str): Generated answer
                - sources (list[dict]): Source documents with metadata (if include_sources=True)

        Example:
            >>> generator = Generator()
            >>> docs = [...]  # Retrieved documents
            >>> result = generator.generate("How do I set Pod limits?", docs)
            >>> print(result['answer'])
        """

        answer = self.combine_docs_chain.invoke({
            "question": query,
            "context": documents
        })

        response = {
            "answer": answer.strip(),
        }

        # Add sources if requested
        if include_sources:
            response["sources"] = self._format_sources(documents)

        return response

    def _format_sources(self, documents: List) -> List[Dict]:
        """
        Format documents as source references for the response.

        Args:
            documents (list[Document]): Retrieved documents

        Returns:
            list[dict]: List of source dictionaries with metadata
        """
        sources = []

        print("Here?")
        for idx, doc in enumerate(documents, 1):
            sources.append({
                "rank": idx,
                "source": doc.metadata.get("source", "unknown"),
                "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        print("returning?")

        return sources
