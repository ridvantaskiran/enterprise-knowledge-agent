
import json
import sys
from pathlib import Path

# Allow importing rag.py from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag import load_index, retrieve


DATASET_PATH = Path(__file__).parent / "test_dataset.json"


def evaluate_retrieval(k=3):
    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    records, matrix = load_index()

    recalls = []
    reciprocal_ranks = []
    precisions = []

    for test in dataset:
        question = test["question"]
        relevant_sources = set(test["relevant_sources"])

        results = retrieve(
            question,
            records,
            matrix,
            top_k=k
        )

        retrieved_sources = [
            result["source"] for result in results
        ]

        # Count each source only once for document-level evaluation.
        unique_sources = list(dict.fromkeys(retrieved_sources))

        relevant_found = (
            set(unique_sources) & relevant_sources
        )

        recall = len(relevant_found) / len(relevant_sources)

        precision = (
            len(relevant_found) / len(unique_sources)
            if unique_sources else 0
        )

        reciprocal_rank = 0

        for rank, source in enumerate(
            retrieved_sources, start=1
        ):
            if source in relevant_sources:
                reciprocal_rank = 1 / rank
                break

        recalls.append(recall)
        precisions.append(precision)
        reciprocal_ranks.append(reciprocal_rank)

        print(f"\nQuestion: {question}")
        print(f"Relevant documents: {relevant_sources}")
        print(f"Retrieved: {unique_sources}")
        print(f"Recall@{k}: {recall:.3f}")
        print(f"MRR contribution: {reciprocal_rank:.3f}")

    print("\n" + "=" * 45)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 45)

    print(f"Recall@{k}: {sum(recalls) / len(recalls):.3f}")
    print(f"Precision@{k}: {sum(precisions) / len(precisions):.3f}")
    print(f"MRR: {sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}")


if __name__ == "__main__":
    evaluate_retrieval(k=2)
