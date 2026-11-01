
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "storage" / "knowledge_index.json"

payload = json.loads(
    INDEX_PATH.read_text(encoding="utf-8")
)

for record in payload["records"]:
    print("\n" + "=" * 70)
    print(f"Source: {record['source']}")
    print(f"Chunk ID: {record['chunk_id']}")
    print(f"Text: {record['text']}")
