"""
CLI script to run lightweight retrieval evaluation (Precision, Recall, Hit@K only).

This script is much faster than run_evaluation.py because it skips the expensive
RAGAS metrics (faithfulness and answer_relevancy) that require LLM calls.

Usage:
    python scripts/run_retrieval_evaluation.py --dataset datasets/k8s_qa_test_multi_context.json --output results/
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rag_pipeline import RAGPipeline
from evaluation.metrics import compute_precision_at_k, compute_recall_at_k, compute_hit_at_k
from config import DATA_PATH


def evaluate_retrieval_only(pipeline: RAGPipeline, dataset_path: str, output_path: str = None):
    """
    Evaluate retrieval quality metrics only (no LLM-based metrics).

    Args:
        pipeline: RAGPipeline instance
        dataset_path: Path to test dataset JSON
        output_path: Optional path to save results

    Returns:
        dict: Evaluation report with retrieval metrics
    """
    print(f"\nLoading test dataset from: {dataset_path}")

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    test_cases = dataset.get("test_cases", [])
    print(f"Loaded {len(test_cases)} test cases")

    if len(test_cases) == 0:
        raise ValueError("Dataset contains no test cases")

    per_question_results = []

    print("\nEvaluating retrieval quality (Precision, Recall, Hit@K)...")

    for test_case in tqdm(test_cases, desc="Evaluating"):
        question_id = test_case.get("question_id", "unknown")
        question = test_case["question"]
        expected_contexts = test_case.get("expected_contexts", [])

        if not expected_contexts:
            print(f"\nWarning: Question {question_id} has no expected_contexts, skipping...")
            continue

        try:
            # Query pipeline to get retrieved sources (no answer generation needed)
            result = pipeline.query_with_contexts(question, top_n=5)
            retrieved_sources = [src["source"] for src in result["sources"]]

            # Compute retrieval metrics
            hit_at_1 = compute_hit_at_k(retrieved_sources, expected_contexts, k=1)
            hit_at_3 = compute_hit_at_k(retrieved_sources, expected_contexts, k=3)
            hit_at_5 = compute_hit_at_k(retrieved_sources, expected_contexts, k=5)

            recall_at_1 = compute_recall_at_k(retrieved_sources, expected_contexts, k=1)
            recall_at_3 = compute_recall_at_k(retrieved_sources, expected_contexts, k=3)
            recall_at_5 = compute_recall_at_k(retrieved_sources, expected_contexts, k=5)

            precision_at_1 = compute_precision_at_k(retrieved_sources, expected_contexts, k=1)
            precision_at_3 = compute_precision_at_k(retrieved_sources, expected_contexts, k=3)
            precision_at_5 = compute_precision_at_k(retrieved_sources, expected_contexts, k=5)

            per_question_result = {
                "question_id": question_id,
                "question": question,
                "retrieved_sources": retrieved_sources,
                "expected_sources": expected_contexts,
                "hit_at_1": hit_at_1,
                "hit_at_3": hit_at_3,
                "hit_at_5": hit_at_5,
                "recall_at_1": recall_at_1,
                "recall_at_3": recall_at_3,
                "recall_at_5": recall_at_5,
                "precision_at_1": precision_at_1,
                "precision_at_3": precision_at_3,
                "precision_at_5": precision_at_5,
                "category": test_case.get("category", "general")
            }

            per_question_results.append(per_question_result)

        except Exception as e:
            print(f"\nError evaluating question {question_id}: {e}")
            per_question_results.append({
                "question_id": question_id,
                "question": question,
                "error": str(e)
            })

    valid_results = [r for r in per_question_results if "error" not in r]

    if len(valid_results) == 0:
        raise RuntimeError("No valid results to aggregate")

    # Aggregate metrics
    avg_hit_1 = sum(r["hit_at_1"] for r in valid_results) / len(valid_results)
    avg_hit_3 = sum(r["hit_at_3"] for r in valid_results) / len(valid_results)
    avg_hit_5 = sum(r["hit_at_5"] for r in valid_results) / len(valid_results)

    avg_recall_1 = sum(r["recall_at_1"] for r in valid_results) / len(valid_results)
    avg_recall_3 = sum(r["recall_at_3"] for r in valid_results) / len(valid_results)
    avg_recall_5 = sum(r["recall_at_5"] for r in valid_results) / len(valid_results)

    avg_precision_1 = sum(r["precision_at_1"] for r in valid_results) / len(valid_results)
    avg_precision_3 = sum(r["precision_at_3"] for r in valid_results) / len(valid_results)
    avg_precision_5 = sum(r["precision_at_5"] for r in valid_results) / len(valid_results)

    evaluation_report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset": str(dataset_path),
            "num_samples": len(test_cases),
            "num_successful": len(valid_results),
            "num_failed": len(test_cases) - len(valid_results),
            "evaluation_type": "retrieval_only"
        },
        "aggregate_metrics": {
            "hit_at_1": round(avg_hit_1, 3),
            "hit_at_3": round(avg_hit_3, 3),
            "hit_at_5": round(avg_hit_5, 3),
            "recall_at_1": round(avg_recall_1, 3),
            "recall_at_3": round(avg_recall_3, 3),
            "recall_at_5": round(avg_recall_5, 3),
            "precision_at_1": round(avg_precision_1, 3),
            "precision_at_3": round(avg_precision_3, 3),
            "precision_at_5": round(avg_precision_5, 3)
        },
        "per_question_results": per_question_results
    }

    # Print summary
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Test Cases: {len(test_cases)}")
    print(f"Successful: {len(valid_results)}")
    print(f"Failed: {len(test_cases) - len(valid_results)}")

    print(f"\n--- Hit@K (Binary Success Rate) ---")
    print(f"Hit@1: {avg_hit_1:.3f} ({avg_hit_1*100:.1f}%)")
    print(f"  → Found at least one correct document in top 1: {avg_hit_1*100:.1f}% of questions")
    print(f"Hit@3: {avg_hit_3:.3f} ({avg_hit_3*100:.1f}%)")
    print(f"  → Found at least one correct document in top 3: {avg_hit_3*100:.1f}% of questions")
    print(f"Hit@5: {avg_hit_5:.3f} ({avg_hit_5*100:.1f}%)")
    print(f"  → Found at least one correct document in top 5: {avg_hit_5*100:.1f}% of questions")

    print(f"\n--- Recall@K (Coverage of Relevant Docs) ---")
    print(f"Recall@1: {avg_recall_1:.3f} ({avg_recall_1*100:.1f}%)")
    print(f"  → Retrieved {avg_recall_1*100:.1f}% of all relevant docs in top 1")
    print(f"Recall@3: {avg_recall_3:.3f} ({avg_recall_3*100:.1f}%)")
    print(f"  → Retrieved {avg_recall_3*100:.1f}% of all relevant docs in top 3")
    print(f"Recall@5: {avg_recall_5:.3f} ({avg_recall_5*100:.1f}%)")
    print(f"  → Retrieved {avg_recall_5*100:.1f}% of all relevant docs in top 5")

    print(f"\n--- Precision@K (Relevance of Retrieved Docs) ---")
    print(f"Precision@1: {avg_precision_1:.3f} ({avg_precision_1*100:.1f}%)")
    print(f"  → {avg_precision_1*100:.1f}% of top 1 retrieved docs were relevant")
    print(f"Precision@3: {avg_precision_3:.3f} ({avg_precision_3*100:.1f}%)")
    print(f"  → {avg_precision_3*100:.1f}% of top 3 retrieved docs were relevant")
    print(f"Precision@5: {avg_precision_5:.3f} ({avg_precision_5*100:.1f}%)")
    print(f"  → {avg_precision_5*100:.1f}% of top 5 retrieved docs were relevant")
    print("=" * 60)

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(evaluation_report, f, indent=2)

        print(f"\nResults saved to: {output_file.absolute()}")

    return evaluation_report


def main():
    parser = argparse.ArgumentParser(
        description="Fast retrieval evaluation (Precision, Recall, Hit@K only - no LLM calls)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to test dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory for evaluation results (default: results/)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rebuild RAG indices from scratch (don't use cache)"
    )

    args = parser.parse_args()

    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)

    print("=" * 60)
    print("RAG - RETRIEVAL EVALUATION (FAST)")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Output directory: {args.output}")
    print(f"Use cache: {not args.no_cache}")
    print("=" * 60)
    print("\nNOTE: This script only evaluates retrieval metrics.")
    print("For faithfulness/answer_relevancy, use run_evaluation.py instead.")
    print("=" * 60)

    # Initialize RAG pipeline
    print("\nStep 1: Initializing RAG pipeline...")
    try:
        pipeline = RAGPipeline(DATA_PATH, use_cache=not args.no_cache)
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        sys.exit(1)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(args.output) / f"retrieval_eval_{timestamp}.json"

    # Run evaluation
    print("\nStep 2: Running retrieval evaluation...")

    try:
        report = evaluate_retrieval_only(
            pipeline=pipeline,
            dataset_path=str(dataset_path),
            output_path=str(output_file)
        )
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    metrics = report["aggregate_metrics"]

    # Hit@K recommendations
    if metrics["hit_at_5"] >= 0.9:
        print("✓ Excellent retrieval success rate (Hit@5 >= 90%)")
        print("  System finds relevant documents for most queries")
    elif metrics["hit_at_5"] >= 0.7:
        print("! Good retrieval success rate (Hit@5 70-90%)")
        print("  Consider improving embeddings or chunking strategy")
    else:
        print("✗ Low retrieval success rate (Hit@5 < 70%)")
        print("  Review chunking, embeddings, and indexing strategy")

    # Precision recommendations
    if metrics["precision_at_3"] >= 0.6:
        print("\n✓ Good retrieval precision (P@3 >= 60%)")
        print("  Most retrieved documents are relevant")
    elif metrics["precision_at_3"] >= 0.4:
        print("\n! Acceptable retrieval precision (P@3 40-60%)")
        print("  Consider adding reranking or better filtering")
    else:
        print("\n✗ Low retrieval precision (P@3 < 40%)")
        print("  Too many irrelevant documents - improve retrieval quality")

    # Recall recommendations
    if metrics["recall_at_5"] >= 0.8:
        print("\n✓ Excellent recall (R@5 >= 80%)")
        print("  System retrieves most relevant documents")
    elif metrics["recall_at_5"] >= 0.6:
        print("\n! Good recall (R@5 60-80%)")
        print("  System captures most relevant documents")
    else:
        print("\n✗ Low recall (R@5 < 60%)")
        print("  System misses many relevant documents - increase top_k or improve chunking")

    print("\n" + "=" * 60)
    print(f"Full report saved to: {output_file.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
