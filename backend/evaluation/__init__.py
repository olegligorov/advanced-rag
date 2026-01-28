"""
Evaluation module for RAG system.

This module provides functionality for evaluating the RAG system's
generation quality using metrics like faithfulness and answer relevance.
"""

from .metrics import compute_faithfulness, compute_answer_relevance, compute_all_metrics

__all__ = [
    "compute_faithfulness",
    "compute_answer_relevance",
    "compute_all_metrics",
]
