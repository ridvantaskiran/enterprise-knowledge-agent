
import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
INDEX_PATH = STORAGE_DIR / "knowledge_index.json"

EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)

client = OpenAI()


def load_documents():
    """Read Markdown and text files from the data folder."""
    documents = []

    for path in sorted(DATA_DIR.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue

        text = path.read_text(encoding="utf-8").strip()

        if text:
            documents.append({
                "source": path.name,
                "text": text,
            })

    if not documents:
        raise ValueError(
            f"No .md or .txt documents found in {DATA_DIR}"
        )

    return documents


def split_into_chunks(text, chunk_size=900, overlap=150):
    """
    Split text into overlapping character-based chunks.
    """
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk_size or overlap")

    words = text.split()
    chunks = []
    current = []
    current_length = 0

    for word in words:
        extra_length = len(word) + (1 if current else 0)

        if current and current_length + extra_length > chunk_size:
            chunks.append(" ".join(current))

            # Keep the tail of the previous chunk as overlap.
            tail = []
            tail_length = 0

            for previous_word in reversed(current):
                extra = len(previous_word) + (
                    1 if tail else 0
                )
                if tail and tail_length + extra > overlap:
                    break

                tail.insert(0, previous_word)
                tail_length += extra

            current = tail
            current_length = len(" ".join(current))

        if current:
            current_length += 1

        current.append(word)
        current_length += len(word)

    if current:
        chunks.append(" ".join(current))

    return [chunk for chunk in chunks if chunk.strip()]


def embed_texts(texts, batch_size=64):
    """Embed multiple texts in batches to reduce API calls."""
    vectors = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        # Preserve the input order.
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in ordered)

    return np.asarray(vectors, dtype=np.float32)


def main():
    documents = load_documents()
    records = []

    for document in documents:
        chunks = split_into_chunks(
            document["text"],
            chunk_size=350,
            overlap=60
        )

        for chunk_number, chunk in enumerate(chunks):
            records.append({
                "source": document["source"],
                "chunk_id": chunk_number,
                "text": chunk,
            })

    if not records:
        raise ValueError("No text chunks were generated.")

    print(f"Loaded {len(documents)} documents.")
    print(f"Created {len(records)} chunks.")
    print("Generating embeddings...")

    vectors = embed_texts([record["text"] for record in records])

    # Normalize vectors so dot product equals cosine similarity.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)

    for record, vector in zip(records, vectors):
        record["embedding"] = vector.tolist()

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "embedding_model": EMBEDDING_MODEL,
        "records": records,
    }

    INDEX_PATH.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved index to {INDEX_PATH}")
    print(f"Embedding dimensions: {vectors.shape[1]}")
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
