"""
Synthetic Test Dataset Generator for RAG Evaluation.

This module generates question-answer pairs from Kubernetes documentation
using an LLM to create evaluation datasets.
"""

import random
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from config import LLM_MODEL, OLLAMA_HOST


class DatasetGenerator:
    """
    Generates synthetic Q&A pairs from document chunks for evaluation.

    Uses an LLM to create questions that can be answered from document
    chunks, along with ground truth answers extracted from those chunks.
    """

    def __init__(self):
        """Initialize the dataset generator with an LLM."""
        self.llm = Ollama(
            model=LLM_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0.7  # Slightly higher for diverse question generation
        )

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

        # Truncate very long contexts to avoid token limits
        if len(context) > 2000:
            context = context[:2000] + "..."

        # Prompt to generate question
        question_prompt = f"""Based on the following text from Kubernetes documentation, generate ONE specific technical question that can be answered using ONLY the information in this text.

Text:
{context}

Generate a clear, technical question that:
1. Can be definitively answered from the text above
2. Is specific to Kubernetes concepts
3. Would be useful for testing a RAG system

Question:"""

        # Generate question
        question = self.llm.invoke(question_prompt).strip()

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
        ground_truth = self.llm.invoke(answer_prompt).strip()
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

        # Randomly sample documents to ensure diversity
        sampled_docs = random.sample(documents, num_samples)

        test_cases = []

        print(f"\nGenerating {num_samples} Q&A pairs from documents...")

        for i, doc in enumerate(sampled_docs, 1):
            try:
                question_id = f"k8s_{i:03d}"
                test_case = self.generate_qa_pair(doc, question_id)
                test_cases.append(test_case)

                print(f"[{i}/{num_samples}] Generated: {test_case['question'][:60]}...")

            except Exception as e:
                print(f"[{i}/{num_samples}] Error generating Q&A pair: {e}")
                continue

        print(f"\nSuccessfully generated {len(test_cases)} Q&A pairs.")

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
