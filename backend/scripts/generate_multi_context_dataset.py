"""
CLI script to generate test dataset with single and multi-source questions.

Usage:
    # Generate 50 questions with 30% multi-source
    python scripts/generate_multi_context_dataset.py --num-samples 50 --multi-ratio 0.3

    # Generate 100 questions with 40% multi-source
    python scripts/generate_multi_context_dataset.py --num-samples 100 --multi-ratio 0.4
"""

import argparse
import json
import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rag_pipeline import RAGPipeline
from evaluation.multi_context_dataset_generator import MultiContextDatasetGenerator
from config import DATA_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Generate mixed single/multi-source Q&A test dataset from Kubernetes documentation"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Total number of Q&A pairs to generate (default: 50)"
    )
    parser.add_argument(
        "--multi-ratio",
        type=float,
        default=0.7,
        help="Fraction of questions that require multiple sources (default: 0.7 = 70%%)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/k8s_qa_multi_context_50.json",
        help="Output path for the generated dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Validate ratio
    if not 0.0 <= args.multi_ratio <= 1.0:
        print("Error: --multi-ratio must be between 0.0 and 1.0")
        sys.exit(1)

    # Set random seed for reproducibility
    random.seed(args.seed)

    print("=" * 70)
    print("RAG - MULTI-CONTEXT TEST DATASET GENERATOR")
    print("=" * 70)
    print(f"Total samples: {args.num_samples}")
    print(f"Multi-source ratio: {args.multi_ratio:.1%} ({int(args.num_samples * args.multi_ratio)} questions)")
    print(f"Single-source: {int(args.num_samples * (1 - args.multi_ratio))} questions")
    print(f"Output path: {args.output}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)

    # Initialize RAG pipeline to access chunked documents
    print("\nStep 1: Initializing RAG pipeline...")
    try:
        pipeline = RAGPipeline(DATA_PATH, use_cache=True)
        print(f"✓ Loaded {len(pipeline.semantic_docs)} semantic chunks from documents")
    except Exception as e:
        print(f"✗ Error initializing RAG pipeline: {e}")
        sys.exit(1)

    # Initialize dataset generator
    print("\nStep 2: Initializing multi-context dataset generator...")
    generator = MultiContextDatasetGenerator()
    print("✓ Generator ready")

    # Generate Q&A pairs
    print("\nStep 3: Generating Q&A pairs...")
    print(f"This will take several minutes as each question requires LLM calls...")
    try:
        test_cases = generator.generate_mixed_qa_pairs(
            documents=pipeline.semantic_docs,
            num_samples=args.num_samples,
            multi_source_ratio=args.multi_ratio
        )
    except Exception as e:
        print(f"✗ Error generating Q&A pairs: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create dataset structure
    dataset = {
        "metadata": {
            "generated_by": "multi_context_dataset_generator",
            "num_samples": len(test_cases),
            "num_single_source": len([tc for tc in test_cases if tc.get("question_type") == "single_source"]),
            "num_multi_source": len([tc for tc in test_cases if tc.get("question_type") == "multi_source"]),
            "multi_source_ratio": args.multi_ratio,
            "source": "kubernetes documentation",
            "random_seed": args.seed,
            "description": f"Mixed dataset with {int((1-args.multi_ratio)*100)}% single-source and {int(args.multi_ratio*100)}% multi-source questions for comprehensive retrieval evaluation"
        },
        "test_cases": test_cases
    }

    # Save to file
    print("\nStep 4: Saving dataset...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"✓ Dataset saved successfully to: {output_path}")
    except Exception as e:
        print(f"✗ Error saving dataset: {e}")
        sys.exit(1)

    # Print detailed summary
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total Q&A pairs: {len(test_cases)}")

    # Question type breakdown
    single_count = len([tc for tc in test_cases if tc.get("question_type") == "single_source"])
    multi_count = len([tc for tc in test_cases if tc.get("question_type") == "multi_source"])
    print(f"\nQuestion type breakdown:")
    print(f"  Single-source (1 document): {single_count} ({single_count/len(test_cases)*100:.1f}%)")
    print(f"  Multi-source (2-3 documents): {multi_count} ({multi_count/len(test_cases)*100:.1f}%)")

    # Multi-source document count
    if multi_count > 0:
        doc_counts = {}
        for tc in test_cases:
            if tc.get("question_type") == "multi_source":
                count = len(tc.get("expected_contexts", []))
                doc_counts[count] = doc_counts.get(count, 0) + 1

        print(f"\nMulti-source document counts:")
        for count, freq in sorted(doc_counts.items()):
            print(f"  {count} documents: {freq} questions")

    # Category breakdown
    categories = {}
    for tc in test_cases:
        cat = tc.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\nCategory breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Difficulty breakdown
    difficulties = {}
    for tc in test_cases:
        diff = tc.get("difficulty", "unknown")
        difficulties[diff] = difficulties.get(diff, 0) + 1

    print("\nDifficulty breakdown:")
    for diff, count in sorted(difficulties.items()):
        print(f"  {diff}: {count}")

    print(f"\nDataset file: {output_path.absolute()}")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("Run fast retrieval evaluation (Precision, Recall, Hit@K):")
    print(f"  python scripts/run_retrieval_evaluation.py --dataset {args.output}")
    print("\nOr run full evaluation (including faithfulness/relevancy):")
    print(f"  python scripts/run_evaluation.py --dataset {args.output}")
    print("\nOr run full evaluation but skip expensive LLM metrics:")
    print(f"  python scripts/run_evaluation.py --dataset {args.output} --skip-llm-metrics")
    print("=" * 70)


if __name__ == "__main__":
    main()
