"""
CLI script to run RAG evaluation on test datasets.

Usage:
    python scripts/run_evaluation.py --dataset datasets/k8s_qa_test_set.json --output results/
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rag_pipeline import RAGPipeline
from evaluation.evaluator import RAGEvaluator
from config import DATA_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system on test dataset using RAGAS metrics"
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
    parser.add_argument(
        "--skip-llm-metrics",
        action="store_true",
        help="Skip expensive LLM-based metrics (faithfulness, answer_relevancy) - only compute retrieval metrics"
    )

    args = parser.parse_args()

    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)

    print("=" * 60)
    print("RAG - EVALUATION")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Output directory: {args.output}")
    print(f"Use cache: {not args.no_cache}")
    print("=" * 60)

    # Initialize RAG pipeline
    print("\nStep 1: Initializing RAG pipeline...")
    try:
        pipeline = RAGPipeline(DATA_PATH, use_cache=not args.no_cache)
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        sys.exit(1)

    # Initialize evaluator
    print("\nStep 2: Initializing evaluator...")
    evaluator = RAGEvaluator(rag_pipeline=pipeline)

    if args.skip_llm_metrics:
        print("\nNOTE: Skipping expensive LLM-based metrics (faithfulness, answer_relevancy)")
        print("Only retrieval metrics will be computed (faster evaluation)")

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(args.output) / f"eval_{timestamp}.json"

    # Run evaluation
    print("\nStep 3: Running evaluation...")
    if args.skip_llm_metrics:
        print("Fast mode: Only computing retrieval metrics (no LLM calls)")
    else:
        print("This may take several minutes depending on dataset size...")
        print("(RAGAS faithfulness computation requires LLM calls per question)")

    try:
        report = evaluator.evaluate_dataset(
            dataset_path=str(dataset_path),
            output_path=str(output_file),
            skip_llm_metrics=args.skip_llm_metrics
        )
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)

    if report["failure_cases"]:
        print(f"\nFailure Cases ({len(report['failure_cases'])} total):")
        print("(Questions with faithfulness < 0.7 may indicate hallucinations)")
        print()
        for i, failure in enumerate(report["failure_cases"][:5], 1):
            print(f"{i}. {failure['question']}")
            print(f"   Faithfulness: {failure['faithfulness']:.3f}")
            print(f"   Issue: {failure['issue']}")
            print()

        if len(report["failure_cases"]) > 5:
            print(f"   ... and {len(report['failure_cases']) - 5} more (see full report)")
    else:
        print("\nNo failure cases detected!")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if not args.skip_llm_metrics:
        avg_faith = report["aggregate_metrics"]["faithfulness"]
        avg_rel = report["aggregate_metrics"]["answer_relevancy"]

        if avg_faith >= 0.8:
            print("✓ Faithfulness score is excellent (>= 0.8)")
            print("  Answers are well-grounded in retrieved context")
        elif avg_faith >= 0.7:
            print("! Faithfulness score is acceptable (0.7-0.8)")
            print("  Consider reviewing failure cases for improvements")
        else:
            print("✗ Faithfulness score needs improvement (< 0.7)")
            print("  High risk of hallucinations - review chunking and prompts")

        if avg_rel >= 0.8:
            print("✓ Answer relevancy is excellent (>= 0.8)")
            print("  Answers address questions well")
        elif avg_rel >= 0.7:
            print("! Answer relevancy is acceptable (0.7-0.8)")
            print("  Answers mostly relevant but could be more focused")
        else:
            print("✗ Answer relevancy needs improvement (< 0.7)")
            print("  Answers may be off-topic - review retrieval quality")
    else:
        print("LLM-based metrics skipped (use --skip-llm-metrics flag to enable)")
        print("\nRetrieval metrics computed successfully!")
        metrics = report["aggregate_metrics"]
        if metrics.get("hit_at_5") and metrics["hit_at_5"] >= 0.8:
            print("✓ Good retrieval performance (Hit@5 >= 80%)")
        elif metrics.get("hit_at_5") and metrics["hit_at_5"] >= 0.6:
            print("! Acceptable retrieval performance (Hit@5 60-80%)")
        else:
            print("✗ Retrieval needs improvement - consider tuning embeddings/chunking")

    print("\n" + "=" * 60)
    print(f"Full report saved to: {output_file.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
