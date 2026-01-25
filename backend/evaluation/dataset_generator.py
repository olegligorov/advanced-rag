"""
Synthetic Test Dataset Generator for RAG Evaluation.

This module generates question-answer pairs from Kubernetes documentation
using an LLM to create evaluation datasets.
"""

import random
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from config import LLM_MODEL, OLLAMA_HOST, PROXY_API_KEY, PROXY_SONNET_MODEL, PROXY_URL, USE_PROXY
from langchain_anthropic import ChatAnthropic

class DatasetGenerator:
    """
    Generates synthetic Q&A pairs from document chunks for evaluation.

    Uses an LLM to create questions that can be answered from document
    chunks, along with ground truth answers extracted from those chunks.
    """

    def __init__(self):
        """Initialize the dataset generator with an LLM."""
        if USE_PROXY == False:
            self.llm = Ollama(
                model=LLM_MODEL,
                base_url=OLLAMA_HOST,
                temperature=0.7
            )
        else:
            self.llm = ChatAnthropic(
                model=PROXY_SONNET_MODEL,
                base_url=PROXY_URL,
                api_key=PROXY_API_KEY,
                temperature=0.7,
                max_tokens=4096
            )

    def _get_llm_response(self, prompt: str) -> str:
        """
        Get response from LLM and normalize to string.

        Handles both Ollama (returns string) and ChatAnthropic (returns AIMessage).
        """
        response = self.llm.invoke(prompt)
        # Handle both string (Ollama) and AIMessage (ChatAnthropic) responses
        return response.content if hasattr(response, 'content') else response

    def generate_qa_pair(self, document: Document, question_id: str) -> Dict:
        """
        Generate a single question-answer pair from a document chunk.

        Args:
            document: LangChain Document containing text chunk and metadata
            question_id: Unique identifier for this Q&A pair

        Returns:
            dict: Test case with question, ground_truth, and metadata
        """
        # Extract context from document
        context = document.page_content
        source = document.metadata.get("source", "unknown")

        # Skip chunks that are too short or just headings
        if len(context.strip()) < 200:
            raise ValueError(f"Document chunk too short ({len(context)} chars) - likely just a heading")

        # Truncate very long contexts to avoid token limits
        if len(context) > 2000:
            context = context[:2000] + "..."

        # Prompt to generate question
        question_prompt = f"""Based on the following text from Kubernetes documentation, generate ONE specific technical question that can be answered using ONLY the information in this text.

Text:
{context}

Generate a clear, technical question that:
1. Can be definitively answered from the text above
2. Is specific and unambiguous (not vague like "what is mentioned here?")
3. Focuses on concrete facts, concepts, or procedures in Kubernetes
4. Would be useful for testing a RAG system

Output ONLY the question, no preamble or meta-commentary.

Question:"""

        # Generate question
        question = self._get_llm_response(question_prompt).strip()
        # Clean up question (remove quotes, extra whitespace)
        question = question.strip('"\'').strip()

        # Prompt to extract ground truth answer
        answer_prompt = f"""Based on the following text from Kubernetes documentation, provide a concise answer to this question using ONLY information from the text.

Text:
{context}

Question: {question}

Provide a direct, factual answer (2-3 sentences maximum) using only the information above:

Answer:"""

        # Generate ground truth answer
        ground_truth = self._get_llm_response(answer_prompt).strip()
        ground_truth = ground_truth.strip('"\'').strip()

        # Determine category based on source filename
        category = self._categorize_source(source)

        # Create test case
        test_case = {
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "expected_contexts": [source],
            "category": category,
            "difficulty": "medium"  # Default difficulty
        }

        return test_case

    def generate_qa_pairs(self, documents: List[Document], num_samples: int) -> List[Dict]:
        """
        Generate multiple Q&A pairs from a list of documents.

        Args:
            documents: List of LangChain Documents (semantic chunks)
            num_samples: Number of Q&A pairs to generate

        Returns:
            list: List of test cases ready for evaluation

        Example:
            >>> from models.rag_pipeline import RAGPipeline
            >>> pipeline = RAGPipeline("./k8s_data/concepts")
            >>> docs = pipeline.semantic_docs
            >>> generator = DatasetGenerator()
            >>> test_cases = generator.generate_qa_pairs(docs, num_samples=50)
        """
        if num_samples > len(documents):
            print(f"Warning: Requested {num_samples} samples but only {len(documents)} documents available.")
            print(f"Generating {len(documents)} samples instead.")
            num_samples = len(documents)

        test_cases = []
        available_docs = documents.copy()
        random.shuffle(available_docs)

        print(f"\nGenerating {num_samples} Q&A pairs from documents...")

        doc_index = 0
        attempts = 0
        max_attempts = num_samples * 3  # Try up to 3x to avoid infinite loops

        while len(test_cases) < num_samples and attempts < max_attempts:
            if doc_index >= len(available_docs):
                print(f"\nWarning: Ran out of documents. Only generated {len(test_cases)} Q&A pairs.")
                break

            doc = available_docs[doc_index]
            doc_index += 1
            attempts += 1

            try:
                question_id = f"k8s_{len(test_cases) + 1:03d}"
                test_case = self.generate_qa_pair(doc, question_id)
                test_cases.append(test_case)

                print(f"[{len(test_cases)}/{num_samples}] Generated: {test_case['question'][:60]}...")

            except ValueError as e:
                # Skip chunks that are too short
                print(f"[Attempt {attempts}] Skipped: {str(e)}")
                continue
            except Exception as e:
                print(f"[Attempt {attempts}] Error generating Q&A pair: {e}")
                continue

        print(f"\nSuccessfully generated {len(test_cases)} Q&A pairs (attempted {attempts} documents).")

        return test_cases

    def _categorize_source(self, source: str) -> str:
        """
        Categorize the test case based on source filename.

        Args:
            source: Source filename (e.g., "configmap.md", "pods.md")

        Returns:
            str: Category label (e.g., "configuration", "core-concepts")
        """
        source_lower = source.lower()

        if any(keyword in source_lower for keyword in ["pod", "container", "deployment", "replicaset"]):
            return "workloads"
        elif any(keyword in source_lower for keyword in ["configmap", "secret", "volume"]):
            return "configuration"
        elif any(keyword in source_lower for keyword in ["service", "ingress", "network"]):
            return "networking"
        elif any(keyword in source_lower for keyword in ["resource", "limit", "quota"]):
            return "resource-management"
        elif any(keyword in source_lower for keyword in ["rbac", "security", "policy"]):
            return "security"
        else:
            return "general"
