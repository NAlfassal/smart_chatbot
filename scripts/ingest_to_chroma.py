import os
import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
INPUT_JSONL = KNOWLEDGE_DIR / "chunks_final.jsonl"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "regulations"

def clean_metadata(value):
    if value is None:
        return "NA"
    return str(value)

def main():
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_JSONL}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

    # ✅ امسحي الكولكشن بالكامل (بدون where={})
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        # إذا ماكانت موجودة أصلاً، عادي
        pass

    # ✅ أنشئيها من جديد
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn
    )

    docs, metadatas, ids = [], [], []

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)

            text = row["text"].strip()
            if len(text) < 50:
                continue

            uid = str(i)

            docs.append(text)
            metadatas.append({
                "doc_name": clean_metadata(row.get("doc_name")),
                "page": clean_metadata(row.get("page")),
                "article_no": clean_metadata(row.get("article_no")),
            })
            ids.append(uid)

    print(f"Chunks ready: {len(docs)}")

    collection.add(
        documents=docs,
        metadatas=metadatas,
        ids=ids
    )

    print("Ingestion done")
    print("DB path:", CHROMA_DIR)

if __name__ == "__main__":
    main()
