# RAG Evaluation Test Datasets

This directory contains test datasets for evaluating the Kubernetes RAG system.

## Dataset Schema

Test datasets follow this JSON structure:

```json
{
  "metadata": {
    "generated_by": "dataset_generator.py",
    "num_samples": 100,
    "source": "kubernetes documentation",
    "random_seed": 42
  },
  "test_cases": [
    {
      "question_id": "k8s_001",
      "question": "What is a Pod in Kubernetes?",
      "ground_truth": "A Pod is the smallest deployable unit in Kubernetes...",
      "expected_contexts": ["pods.md"],
      "category": "workloads",
      "difficulty": "easy"
    }
  ]
}
```

## Field Descriptions

### Test Case Fields

- **question_id** (string): Unique identifier for the test case (e.g., "k8s_001")
- **question** (string): The question to ask the RAG system
- **ground_truth** (string): Expected answer extracted from documentation
- **expected_contexts** (list): Source files that should contain the answer
- **category** (string): Question category for analysis
  - workloads: Pods, Deployments, StatefulSets, etc.
  - configuration: ConfigMaps, Secrets, Volumes
  - networking: Services, Ingress, NetworkPolicies
  - resource-management: Resource limits, quotas
  - security: RBAC, SecurityContext, Policies
  - general: Other topics
- **difficulty** (string): Difficulty level (easy, medium, hard)

## Generating Datasets

### Generate Synthetic Dataset

```bash
cd backend
python scripts/generate_test_dataset.py --num-samples 100 --output datasets/k8s_qa_test_set.json
```

Parameters:
- `--num-samples`: Number of Q&A pairs to generate (default: 50)
- `--output`: Output file path
- `--seed`: Random seed for reproducibility (default: 42)

### Creating Manual Golden Dataset

For higher quality evaluation, create a manually curated dataset:

1. Start with a synthetic dataset as a template
2. Review and edit questions for clarity and correctness
3. Verify ground truth answers against documentation
4. Add edge cases (multi-hop reasoning, ambiguous queries, etc.)
5. Save as `k8s_qa_golden_set.json`

## Available Datasets

- **k8s_qa_test_set.json** - Main synthetic test dataset (generated)
- **k8s_qa_golden_set.json** - Manually curated golden dataset (if created)
- **k8s_qa_test_small.json** - Small dataset for quick testing (10-20 samples)

## Usage in Evaluation

```bash
# Run evaluation on a dataset
python scripts/run_evaluation.py --dataset datasets/k8s_qa_test_set.json --output results/

# Results will be saved to results/eval_[timestamp].json
```

## Quality Guidelines

For manually creating or reviewing test cases:

1. **Questions should be:**
   - Clear and unambiguous
   - Specific to Kubernetes concepts
   - Answerable from the documentation
   - Representative of real user queries

2. **Ground truth answers should:**
   - Be factually accurate
   - Come directly from documentation
   - Be 2-3 sentences maximum
   - Focus on the core answer

3. **Expected contexts should:**
   - List the source files containing the answer
   - Be verifiable against actual documentation

## Categories

Organize test cases by category to analyze system performance across different Kubernetes topics:

- **workloads**: Core workload resources (Pods, Deployments, Jobs, etc.)
- **configuration**: Configuration and storage (ConfigMaps, Secrets, Volumes)
- **networking**: Networking concepts (Services, Ingress, DNS)
- **resource-management**: Resource quotas, limits, scheduling
- **security**: Security policies, RBAC, authentication
- **general**: Cross-cutting concerns or other topics


Average Faithfulness: 0.986 (10/10 valid)
Average Answer Relevancy: 0.931

with k8s_qa_test_small.json (10 samples)
results/eval_20260125_155222.json