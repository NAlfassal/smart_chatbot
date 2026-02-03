import os
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "sfda_collection"


JSON_PATH = os.path.join(BASE_DIR, "sfda_articles.json")  


def normalize_source_name(name: str) -> str:
   
    s = str(name or "").strip()
    s = s.replace("-", " ")
    s = re.sub(r"\s{2,}", " ", s)
    return s


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_documents(data: dict):
    docs = []

    for doc_name, sections in data.items():
        src = normalize_source_name(doc_name)

        # sections عبارة عن dict: key -> text
        for key, text in (sections or {}).items():
            key_str = str(key).strip()
            text_str = str(text or "").strip()
            if not text_str:
                continue

            # ✅ لو المفتاح رقم => مادة
            if re.fullmatch(r"\d+", key_str):
                docs.append(
                    Document(
                        page_content=f"المادة {key_str}\n\n{text_str}",
                        metadata={
                            "source": src,
                            "type": "pdf",
                            "article": key_str,
                            "section": f"المادة {key_str}",
                        },
                    )
                )
            else:
                # ✅ لو المفتاح عنوان => قسم
                docs.append(
                    Document(
                        page_content=f"{key_str}\n\n{text_str}",
                        metadata={
                            "source": src,
                            "type": "pdf",
                            "section": key_str,
                        },
                    )
                )

    return docs


def main():
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"JSON file not found: {JSON_PATH}")

    data = load_json(JSON_PATH)
    docs = build_documents(data)
    print(f"✅ Prepared {len(docs)} documents from JSON.")

    embeddings_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"},
    )

    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings_model,
        persist_directory=CHROMA_PATH,
    )

    # ✅ مهم: add_documents بدون chunks (لأن المواد جاهزة)
    db.add_documents(docs)
    print("✅ Ingestion complete. Saved to Chroma.")


if __name__ == "__main__":
    main()
