
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag import load_index, retrieve

DATASET_PATH = Path(__file__).parent / "chunk_dataset.json"


def evaluate(k=3):
    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    records, matrix = load_index()

    recalls = []
    reciprocal_ranks = []
    precisions = []

    for test in dataset:
        question = test["question"]

        relevant = {
            (item["source"], item["chunk_id"])
            for item in test["relevant_chunks"]
        }

        results = retrieve(
            question,
            records,
            matrix,
            top_k=k
        )

        retrieved = [
            (item["source"], item["chunk_id"])
            for item in results
        ]

        found = set(retrieved) & relevant

        recall = len(found) / len(relevant)
        precision = len(found) / len(retrieved) if retrieved else 0

        reciprocal_rank = 0

        for rank, chunk in enumerate(retrieved, start=1):
            if chunk in relevant:
                reciprocal_rank = 1 / rank
                break

        recalls.append(recall)
        precisions.append(precision)
        reciprocal_ranks.append(reciprocal_rank)

        print(f"\nQuestion: {question}")
        print(f"Relevant chunks: {relevant}")
        print(f"Retrieved chunks: {retrieved}")
        print(f"Recall@{k}: {recall:.3f}")
        print(f"MRR: {reciprocal_rank:.3f}")

    print("\n" + "=" * 45)
    print("CHUNK-LEVEL EVALUATION")
    print("=" * 45)

    print(f"Recall@{k}: {sum(recalls) / len(recalls):.3f}")
    print(f"Precision@{k}: {sum(precisions) / len(precisions):.3f}")
    print(f"MRR: {sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}")


if __name__ == "__main__":
    evaluate(k=3)
