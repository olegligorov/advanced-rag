"""
RAG Evaluation Orchestrator.

This module provides the RAGEvaluator class for evaluating the RAG system
against test datasets using RAGAS metrics.
"""

import json
from typing import Dict, List
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from models.rag_pipeline import RAGPipeline
from evaluation.metrics import compute_all_metrics
from config import DATA_PATH
import math


class RAGEvaluator:
    """
    Orchestrates evaluation of the RAG system against test datasets.

    This class handles:
    - Loading test datasets
    - Running queries through the RAG pipeline
    - Computing evaluation metrics (faithfulness, answer_relevance)
    - Aggregating results and identifying failure cases
    - Generating evaluation reports
    """

    def __init__(self, rag_pipeline: RAGPipeline = None):
        """
        Initialize the RAG evaluator.

        Args:
            rag_pipeline: Existing RAGPipeline instance. If None, creates a new one.
        """
        if rag_pipeline is None:
            print("Initializing RAG pipeline for evaluation...")
            self.rag_pipeline = RAGPipeline(DATA_PATH, use_cache=True)
        else:
            self.rag_pipeline = rag_pipeline

    def evaluate_single(self, question: str, ground_truth: str = None) -> Dict:
        """
        Evaluate a single question through the RAG pipeline.

        Args:
            question: The question to ask
            ground_truth: Optional ground truth answer for reference

        Returns:
            dict: Evaluation results with metrics and generated answer
        """
        print(f"\nEvaluating: {question}")

        # Run query through RAG pipeline with full contexts
        result = self.rag_pipeline.query_with_contexts(question, top_n=5)

        # Compute metrics
        metrics = compute_all_metrics(
            question=result["question"],
            answer=result["answer"],
            contexts=result["contexts"]
        )

        # Compile results
        evaluation_result = {
            "question": question,
            "generated_answer": result["answer"],
            "ground_truth": ground_truth,
            "retrieved_sources": [src["source"] for src in result["sources"]],
            "metrics": metrics
        }

        print(f"  Faithfulness: {metrics['faithfulness']:.3f}")
        print(f"  Answer Relevancy: {metrics['answer_relevancy']:.3f}")

        return evaluation_result

    def evaluate_dataset(self, dataset_path: str, output_path: str = None) -> Dict:
        """
        Evaluate the RAG system on a full test dataset.

        Args:
            dataset_path: Path to JSON file containing test cases
            output_path: Optional path to save evaluation results

        Returns:
            dict: Complete evaluation report with aggregate metrics and per-question results

        Dataset Format:
            {
                "test_cases": [
                    {
                        "question_id": "k8s_001",
                        "question": "What is a Pod?",
                        "ground_truth": "Expected answer...",
                        "expected_contexts": ["pods.md"],
                        "category": "core-concepts"
                    }
                ]
            }
        """
        print(f"\nLoading test dataset from: {dataset_path}")

        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        test_cases = dataset.get("test_cases", [])
        print(f"Loaded {len(test_cases)} test cases")

        if len(test_cases) == 0:
            raise ValueError("Dataset contains no test cases")

        # Evaluate each test case
        per_question_results = []
        failure_cases = []

        print("\nEvaluating test cases...")
        for test_case in tqdm(test_cases, desc="Evaluating"):
            question_id = test_case.get("question_id", "unknown")
            question = test_case["question"]
            ground_truth = test_case.get("ground_truth", "")

            try:
                # Run query through RAG pipeline
                result = self.rag_pipeline.query_with_contexts(question, top_n=5)

                # Compute metrics
                metrics = compute_all_metrics(
                    question=result["question"],
                    answer=result["answer"],
                    contexts=result["contexts"]
                )

                # Store result
                per_question_result = {
                    "question_id": question_id,
                    "question": question,
                    "generated_answer": result["answer"],
                    "ground_truth": ground_truth,
                    "faithfulness": metrics["faithfulness"],
                    "answer_relevancy": metrics["answer_relevancy"],
                    "retrieved_sources": [src["source"] for src in result["sources"]],
                    "category": test_case.get("category", "general")
                }

                per_question_results.append(per_question_result)

                # Identify failure cases (low faithfulness = hallucination risk)
                if metrics["faithfulness"] < 0.7:
                    failure_cases.append({
                        "question_id": question_id,
                        "question": question,
                        "faithfulness": metrics["faithfulness"],
                        "issue": "Low faithfulness - possible hallucination"
                    })

            except Exception as e:
                print(f"\nError evaluating question {question_id}: {e}")
                per_question_results.append({
                    "question_id": question_id,
                    "question": question,
                    "error": str(e)
                })

        # Compute aggregate metrics
        valid_results = [r for r in per_question_results if "error" not in r]

        if len(valid_results) == 0:
            raise RuntimeError("No valid results to aggregate")

        # Filter out NaN values for faithfulness
        valid_faithfulness = [r["faithfulness"] for r in valid_results if not math.isnan(r["faithfulness"])]
        valid_relevancy = [r["answer_relevancy"] for r in valid_results if not math.isnan(r["answer_relevancy"])]

        avg_faithfulness = sum(valid_faithfulness) / len(valid_faithfulness) if valid_faithfulness else 0.0
        avg_relevancy = sum(valid_relevancy) / len(valid_relevancy) if valid_relevancy else 0.0

        # Generate evaluation report
        evaluation_report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "dataset": str(dataset_path),
                "num_samples": len(test_cases),
                "num_successful": len(valid_results),
                "num_failed": len(test_cases) - len(valid_results)
            },
            "aggregate_metrics": {
                "faithfulness": round(avg_faithfulness, 3),
                "answer_relevancy": round(avg_relevancy, 3)
            },
            "per_question_results": per_question_results,
            "failure_cases": failure_cases
        }

        # Print summary
        num_nan_faithfulness = len(valid_results) - len(valid_faithfulness)
        num_nan_relevancy = len(valid_results) - len(valid_relevancy)

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Test Cases: {len(test_cases)}")
        print(f"Successful: {len(valid_results)}")
        print(f"Failed: {len(test_cases) - len(valid_results)}")
        print(f"\nAverage Faithfulness: {avg_faithfulness:.3f} ({len(valid_faithfulness)}/{len(valid_results)} valid)")
        if num_nan_faithfulness > 0:
            print(f"  Note: {num_nan_faithfulness} questions had NaN faithfulness (RAGAS parsing errors)")
        if num_nan_relevancy > 0:
            print(f"  Note: {num_nan_relevancy} questions had NaN answer relevancy (RAGAS parsing errors)")
        print(f"Average Answer Relevancy: {avg_relevancy:.3f}")
        print(f"\nFailure Cases (faithfulness < 0.7): {len(failure_cases)}")
        print("=" * 60)

        # Save results if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(evaluation_report, f, indent=2)

            print(f"\nEvaluation results saved to: {output_file}")

        return evaluation_report
