"""
CLI script to generate synthetic test dataset for RAG evaluation.

Usage:
    python scripts/generate_test_dataset.py --num-samples 100 --output datasets/k8s_qa_test_set.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rag_pipeline import RAGPipeline
from evaluation.dataset_generator import DatasetGenerator
from config import DATA_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Q&A test dataset from Kubernetes documentation"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of Q&A pairs to generate (default: 50)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/k8s_qa_test_set.json",
        help="Output path for the generated dataset (default: datasets/k8s_qa_test_set.json)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    import random
    random.seed(args.seed)

    print("=" * 60)
    print("RAG - TEST DATASET GENERATOR")
    print("=" * 60)
    print(f"Samples to generate: {args.num_samples}")
    print(f"Output path: {args.output}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)

    # Initialize RAG pipeline to access chunked documents
    print("\nStep 1: Initializing RAG pipeline...")
    try:
        pipeline = RAGPipeline(DATA_PATH, use_cache=True)
        print(f"Loaded {len(pipeline.semantic_docs)} semantic chunks from documents")
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        sys.exit(1)

    # Initialize dataset generator
    print("\nStep 2: Initializing dataset generator...")
    generator = DatasetGenerator()

    # Generate Q&A pairs
    print("\nStep 3: Generating Q&A pairs...")
    try:
        test_cases = generator.generate_qa_pairs(
            documents=pipeline.semantic_docs,
            num_samples=args.num_samples
        )
    except Exception as e:
        print(f"Error generating Q&A pairs: {e}")
        sys.exit(1)

    # Create dataset structure
    dataset = {
        "metadata": {
            "generated_by": "dataset_generator.py",
            "num_samples": len(test_cases),
            "source": "kubernetes documentation",
            "random_seed": args.seed
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
        print(f"Dataset saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving dataset: {e}")
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total Q&A pairs: {len(test_cases)}")

    # Category breakdown
    categories = {}
    for tc in test_cases:
        cat = tc.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\nCategory breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    print(f"\nDataset file: {output_path.absolute()}")
    print("\nNext step: Run evaluation with this dataset:")
    print(f"  python scripts/run_evaluation.py --dataset {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
