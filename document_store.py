import json
from pathlib import Path


DOCUMENT_DIR = Path("documents")
DOCUMENT_DIR.mkdir(exist_ok=True)


def save_document(document_id: str, data: dict):
    file_path = DOCUMENT_DIR / f"{document_id}.json"

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)


def load_document(document_id: str):
    file_path = DOCUMENT_DIR / f"{document_id}.json"

    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)