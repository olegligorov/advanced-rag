"""
Multi-Context Test Dataset Generator for RAG Evaluation.

This module generates question-answer pairs that may require multiple documents
to fully answer, creating more realistic evaluation datasets.
"""

import random
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from config import LLM_MODEL, OLLAMA_HOST, PROXY_API_KEY, PROXY_SONNET_MODEL, PROXY_URL, USE_PROXY
from langchain_anthropic import ChatAnthropic
from pathlib import Path


class MultiContextDatasetGenerator:
    """
    Generates synthetic Q&A pairs from document chunks for evaluation.

    Creates a mix of:
    - Single-source questions (can be answered from one document)
    - Multi-source questions (require multiple documents for complete answer)
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
        return response.content if hasattr(response, 'content') else response

    def _find_related_documents(self, doc: Document, all_docs: List[Document], max_related: int = 3) -> List[Document]:
        """
        Find documents related to the given document (same topic/concept).

        Args:
            doc: The source document
            all_docs: All available documents
            max_related: Maximum number of related docs to return

        Returns:
            List of related documents
        """
        source_path = Path(doc.metadata.get("source", ""))
        source_dir = source_path.parent
        source_name = source_path.stem

        related = []

        for other_doc in all_docs:
            if other_doc == doc:
                continue

            other_path = Path(other_doc.metadata.get("source", ""))
            other_dir = other_path.parent
            other_name = other_path.stem

            # Same directory or related filenames
            if (other_dir == source_dir or
                any(keyword in source_name.lower() for keyword in ["pod", "container", "deployment"]) and
                any(keyword in other_name.lower() for keyword in ["pod", "container", "deployment"])):
                related.append(other_doc)

            if len(related) >= max_related:
                break

        return related

    def generate_single_source_qa(self, document: Document, question_id: str) -> Dict:
        """
        Generate a Q&A pair from a single document.

        Args:
            document: LangChain Document containing text chunk and metadata
            question_id: Unique identifier for this Q&A pair

        Returns:
            dict: Test case with question, ground_truth, and single expected_context
        """
        context = document.page_content
        source = document.metadata.get("source", "unknown")

        if len(context.strip()) < 200:
            raise ValueError(f"Document chunk too short ({len(context)} chars)")

        if len(context) > 2000:
            context = context[:2000] + "..."

        question_prompt = f"""Based on the following text from Kubernetes documentation, generate ONE specific technical question that can be answered using ONLY the information in this text.

Text:
{context}

Generate a clear, technical question that:
1. Can be definitively answered from the text above
2. Is specific and unambiguous
3. Focuses on concrete facts, concepts, or procedures in Kubernetes
4. Would be useful for testing a RAG system

Output ONLY the question, no preamble.

Question:"""

        question = self._get_llm_response(question_prompt).strip()
        question = question.strip('"\'').strip()

        answer_prompt = f"""Based on the following text, provide a concise answer to this question using ONLY information from the text.

Text:
{context}

Question: {question}

Provide a direct, factual answer (2-3 sentences maximum):

Answer:"""

        ground_truth = self._get_llm_response(answer_prompt).strip()
        ground_truth = ground_truth.strip('"\'').strip()

        category = self._categorize_source(source)

        test_case = {
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "expected_contexts": [source],
            "category": category,
            "difficulty": "medium",
            "question_type": "single_source"
        }

        return test_case

    def generate_multi_source_qa(self, documents: List[Document], question_id: str) -> Dict:
        """
        Generate a Q&A pair that requires multiple documents to answer.

        Args:
            documents: List of related LangChain Documents (2-3 docs)
            question_id: Unique identifier for this Q&A pair

        Returns:
            dict: Test case with question, ground_truth, and multiple expected_contexts
        """
        if len(documents) < 2:
            raise ValueError("Need at least 2 documents for multi-source question")

        # Combine contexts
        combined_context = "\n\n---\n\n".join([
            f"Document {i+1} ({Path(doc.metadata.get('source', 'unknown')).name}):\n{doc.page_content[:800]}"
            for i, doc in enumerate(documents[:3])
        ])

        sources = [doc.metadata.get("source", "unknown") for doc in documents]

        question_prompt = f"""Based on the following documents from Kubernetes documentation, generate ONE technical question that requires information from MULTIPLE documents to answer completely.

{combined_context}

Generate a question that:
1. Requires synthesizing information across these documents
2. Cannot be fully answered using just one document
3. Is specific and technical about Kubernetes concepts
4. Would test whether a RAG system retrieves multiple relevant documents

Examples of good multi-source questions:
- "How do Pods and Services work together in Kubernetes networking?"
- "What is the relationship between PersistentVolumes and PersistentVolumeClaims?"
- "How do resource requests and limits affect Pod scheduling and QoS?"

Output ONLY the question, no preamble.

Question:"""

        question = self._get_llm_response(question_prompt).strip()
        question = question.strip('"\'').strip()

        answer_prompt = f"""Based on the following documents, provide a comprehensive answer to this question by synthesizing information across all documents.

{combined_context}

Question: {question}

Provide a complete answer (3-4 sentences) that draws from multiple documents:

Answer:"""

        ground_truth = self._get_llm_response(answer_prompt).strip()
        ground_truth = ground_truth.strip('"\'').strip()

        # Use the category of the first document
        category = self._categorize_source(sources[0])

        test_case = {
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "expected_contexts": sources[:3],  # Up to 3 expected sources
            "category": category,
            "difficulty": "hard",
            "question_type": "multi_source"
        }

        return test_case

    def generate_mixed_qa_pairs(
        self,
        documents: List[Document],
        num_samples: int,
        multi_source_ratio: float = 0.3
    ) -> List[Dict]:
        """
        Generate a mix of single-source and multi-source Q&A pairs.

        Args:
            documents: List of LangChain Documents (semantic chunks)
            num_samples: Total number of Q&A pairs to generate
            multi_source_ratio: Fraction of questions that should be multi-source (default: 0.3 = 30%)

        Returns:
            list: List of test cases with mixed single/multi-source questions

        Example:
            >>> generator = MultiContextDatasetGenerator()
            >>> test_cases = generator.generate_mixed_qa_pairs(docs, num_samples=50, multi_source_ratio=0.3)
            # Will generate ~35 single-source and ~15 multi-source questions
        """
        if num_samples > len(documents):
            print(f"Warning: Requested {num_samples} samples but only {len(documents)} documents available.")
            num_samples = len(documents)

        num_multi = int(num_samples * multi_source_ratio)
        num_single = num_samples - num_multi

        print(f"\nGenerating {num_samples} Q&A pairs:")
        print(f"  - {num_single} single-source questions")
        print(f"  - {num_multi} multi-source questions")

        test_cases = []
        available_docs = documents.copy()
        random.shuffle(available_docs)

        doc_index = 0
        attempts = 0
        max_attempts = num_samples * 5

        # Generate single-source questions
        print("\n[1/2] Generating single-source questions...")
        while len([tc for tc in test_cases if tc.get("question_type") == "single_source"]) < num_single and attempts < max_attempts:
            if doc_index >= len(available_docs):
                print(f"\nWarning: Ran out of documents for single-source questions.")
                break

            doc = available_docs[doc_index]
            doc_index += 1
            attempts += 1

            try:
                question_id = f"k8s_{len(test_cases) + 1:03d}"
                test_case = self.generate_single_source_qa(doc, question_id)
                test_cases.append(test_case)

                current_single = len([tc for tc in test_cases if tc.get("question_type") == "single_source"])
                print(f"  [{current_single}/{num_single}] {test_case['question'][:70]}...")

            except ValueError as e:
                continue
            except Exception as e:
                print(f"  Error: {e}")
                continue

        # Generate multi-source questions
        print("\n[2/2] Generating multi-source questions...")
        attempts = 0
        while len([tc for tc in test_cases if tc.get("question_type") == "multi_source"]) < num_multi and attempts < max_attempts:
            if doc_index >= len(available_docs):
                print(f"\nWarning: Ran out of documents for multi-source questions.")
                break

            primary_doc = available_docs[doc_index]
            doc_index += 1
            attempts += 1

            try:
                # Find 1-2 related documents
                related_docs = self._find_related_documents(primary_doc, available_docs[doc_index:], max_related=2)

                if not related_docs:
                    # If no related docs found, just use 2 random nearby docs
                    related_docs = available_docs[doc_index:doc_index+2]

                if len(related_docs) < 1:
                    continue

                docs_for_question = [primary_doc] + related_docs[:2]
                question_id = f"k8s_{len(test_cases) + 1:03d}"
                test_case = self.generate_multi_source_qa(docs_for_question, question_id)
                test_cases.append(test_case)

                current_multi = len([tc for tc in test_cases if tc.get("question_type") == "multi_source"])
                print(f"  [{current_multi}/{num_multi}] {test_case['question'][:70]}...")
                print(f"      Sources: {len(test_case['expected_contexts'])} docs")

            except ValueError as e:
                continue
            except Exception as e:
                print(f"  Error: {e}")
                continue

        print(f"\n✓ Successfully generated {len(test_cases)} Q&A pairs")

        # Print final stats
        single_count = len([tc for tc in test_cases if tc.get("question_type") == "single_source"])
        multi_count = len([tc for tc in test_cases if tc.get("question_type") == "multi_source"])
        print(f"  Final: {single_count} single-source, {multi_count} multi-source")

        return test_cases

    def _categorize_source(self, source: str) -> str:
        """Categorize the test case based on source filename."""
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
