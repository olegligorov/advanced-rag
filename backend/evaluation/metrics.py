"""
RAGAS-based evaluation metrics for RAG system.

This module provides wrappers around RAGAS metrics for computing:
- Faithfulness: Measures if the answer is grounded in retrieved contexts (hallucination detection)
- Answer Relevance: Measures if the answer addresses the question
- Precision@K: Measures retrieval quality (how many retrieved docs are relevant)
- Recall@K: Measures retrieval completeness (how many relevant docs were retrieved)
"""

from typing import List, Dict, Set
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from config import LLM_MODEL, OLLAMA_HOST, EMBEDDING_MODEL, PROXY_API_KEY, PROXY_SONNET_MODEL, PROXY_URL, USE_PROXY
from langchain_anthropic import ChatAnthropic
from pathlib import Path

def _init_ragas_llm():
    """Initialize LLM for RAGAS metric computation."""
    # return Ollama(
    #     model=LLM_MODEL,
    #     base_url=OLLAMA_HOST,
    #     temperature=0.0  # Use 0 for evaluation consistency
    # )
    
    if USE_PROXY == False:
        return Ollama(
            model=LLM_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0.0
        )
    else:
        return ChatAnthropic(
            model=PROXY_SONNET_MODEL,
            base_url=PROXY_URL,
            api_key=PROXY_API_KEY,
            temperature=0.0,
            max_tokens=4096
        )


def _init_ragas_embeddings():
    """Initialize embeddings for RAGAS metric computation."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def compute_faithfulness(question: str, answer: str, contexts: List[str]) -> float:
    """
    Compute faithfulness score using RAGAS.

    Faithfulness measures whether the generated answer is grounded in the
    retrieved contexts. A low score indicates potential hallucination.

    Args:
        question: The user's question
        answer: The generated answer from the RAG system
        contexts: List of retrieved context strings

    Returns:
        float: Faithfulness score between 0.0 and 1.0 (higher is better)
               - 1.0: Perfectly grounded in context
               - < 0.7: Potential hallucination

    Example:
        >>> contexts = ["A Pod is the smallest deployable unit in Kubernetes."]
        >>> answer = "A Pod is the smallest unit in Kubernetes."
        >>> score = compute_faithfulness("What is a Pod?", answer, contexts)
        >>> print(score)  # Should be high (>0.8)
    """
    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    })

    llm = _init_ragas_llm()
    embeddings = _init_ragas_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness],
        llm=llm,
        embeddings=embeddings,
    )

    return result["faithfulness"]


def compute_answer_relevance(question: str, answer: str) -> float:
    """
    Compute answer relevance score using RAGAS.

    Answer relevance measures whether the answer actually addresses
    the question asked.

    Args:
        question: The user's question
        answer: The generated answer from the RAG system

    Returns:
        float: Answer relevance score between 0.0 and 1.0 (higher is better)
               - 1.0: Highly relevant to question
               - < 0.6: Answer doesn't address the question well

    Example:
        >>> question = "What is a Pod?"
        >>> answer = "A Pod is the smallest deployable unit in Kubernetes."
        >>> score = compute_answer_relevance(question, answer)
        >>> print(score)  # Should be high (>0.8)
    """
    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [[""]],  # Dummy contexts, not needed for answer_relevancy
    })

    llm = _init_ragas_llm()
    embeddings = _init_ragas_embeddings()

    result = evaluate(
        dataset,
        metrics=[answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )

    return result["answer_relevancy"]


def compute_all_metrics(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """
    Compute all available metrics for a single question-answer pair.

    This is more efficient than calling individual metric functions
    as it only initializes models once.

    Args:
        question: The user's question
        answer: The generated answer from the RAG system
        contexts: List of retrieved context strings

    Returns:
        dict: Dictionary with metric names as keys and scores as values
              {
                  "faithfulness": 0.87,
                  "answer_relevancy": 0.82
              }

    Example:
        >>> contexts = ["A Pod is the smallest deployable unit in Kubernetes."]
        >>> answer = "A Pod is the smallest unit in Kubernetes."
        >>> metrics = compute_all_metrics("What is a Pod?", answer, contexts)
        >>> print(metrics)
        {"faithfulness": 0.95, "answer_relevancy": 0.90}
    """
    try:
        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        })

        llm = _init_ragas_llm()
        embeddings = _init_ragas_embeddings()

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
        )

        # Extract scores, handle potential type issues
        # RAGAS returns lists even for single samples
        faith_val = result["faithfulness"]
        rel_val = result["answer_relevancy"]
        print(f"faith_val: {faith_val}, rel_val: {rel_val}")

        # Convert to float, handling both list and scalar returns
        faithfulness_score = float(faith_val[0]) if isinstance(faith_val, list) else float(faith_val)
        relevancy_score = float(rel_val[0]) if isinstance(rel_val, list) else float(rel_val)

        return {
            "faithfulness": faithfulness_score,
            "answer_relevancy": relevancy_score,
        }
    except Exception as e:
        print(f"Error in compute_all_metrics: {e}")
        import traceback
        traceback.print_exc()
        # Return default low scores on error
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
        }

def compute_precision_at_k(retrieved_sources: List[str], expected_sources: List[str], k: int = None) -> float:
    """
    Compute Precision@K for retrieval quality evaluation.

    Precision@K measures what fraction of the top-K retrieved documents are relevant.
    A document is considered relevant if its source file matches one of the expected sources.

    Formula: Precision@K = (# of relevant docs in top-K) / K

    Args:
        retrieved_sources: List of source file paths retrieved by the system (in rank order)
        expected_sources: List of expected/ground-truth source file paths or filenames
        k: Number of top results to consider. If None, uses len(retrieved_sources)

    Returns:
        float: Precision score between 0.0 and 1.0
               - 1.0: All retrieved docs are relevant
               - 0.5: Half of retrieved docs are relevant
               - 0.0: No retrieved docs are relevant

    Example:
        >>> retrieved = ["/path/to/pods.md", "/path/to/services.md", "/path/to/volumes.md"]
        >>> expected = ["pods.md", "containers.md"]
        >>> compute_precision_at_k(retrieved, expected, k=3)
        0.333  # Only 1 out of 3 retrieved docs (pods.md) is relevant

    Notes:
        - Matching is done by filename (basename) to handle different path formats
        - Case-insensitive matching
        - If k is larger than retrieved_sources, uses actual length
    """
    if not retrieved_sources:
        return 0.0

    # Use actual length if k not specified or exceeds available docs
    if k is None:
        k = len(retrieved_sources)
    else:
        k = min(k, len(retrieved_sources))

    # Extract filenames from paths for matching (handle both full paths and filenames)
    retrieved_filenames: Set[str] = {
        Path(src).name.lower() for src in retrieved_sources[:k]
    }

    expected_filenames: Set[str] = {
        Path(src).name.lower() for src in expected_sources
    }

    # Count how many retrieved docs are in the expected set
    relevant_count = len(retrieved_filenames.intersection(expected_filenames))

    precision = relevant_count / k
    return precision


def compute_recall_at_k(retrieved_sources: List[str], expected_sources: List[str], k: int = None) -> float:
    """
    Compute Recall@K for retrieval quality evaluation.

    Recall@K measures what fraction of all relevant documents were retrieved in top-K.

    Formula: Recall@K = (# of relevant docs in top-K) / (total # of relevant docs)

    Args:
        retrieved_sources: List of source file paths retrieved by the system (in rank order)
        expected_sources: List of expected/ground-truth source file paths or filenames
        k: Number of top results to consider. If None, uses len(retrieved_sources)

    Returns:
        float: Recall score between 0.0 and 1.0
               - 1.0: All relevant docs were retrieved
               - 0.5: Half of relevant docs were retrieved
               - 0.0: No relevant docs were retrieved

    Example:
        >>> retrieved = ["/path/to/pods.md", "/path/to/services.md"]
        >>> expected = ["pods.md", "containers.md", "volumes.md"]  # 3 relevant docs
        >>> compute_recall_at_k(retrieved, expected, k=2)
        0.333  # Only 1 out of 3 relevant docs (pods.md) was retrieved
    """
    if not expected_sources:
        return 0.0

    if not retrieved_sources:
        return 0.0

    # Use actual length if k not specified or exceeds available docs
    if k is None:
        k = len(retrieved_sources)
    else:
        k = min(k, len(retrieved_sources))

    # Extract filenames from paths for matching
    retrieved_filenames: Set[str] = {
        Path(src).name.lower() for src in retrieved_sources[:k]
    }

    expected_filenames: Set[str] = {
        Path(src).name.lower() for src in expected_sources
    }

    # Count how many expected docs were retrieved
    relevant_retrieved = len(retrieved_filenames.intersection(expected_filenames))

    recall = relevant_retrieved / len(expected_filenames)
    return recall
