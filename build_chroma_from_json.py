import json
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

CHROMA_PATH = "chroma_db"
COLLECTION = "sfda_collection"
JSON_PATH = "knowledge/sfda_articles.json"

def main():
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"JSON not found: {JSON_PATH}")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for source_name, articles in data.items():
        if not isinstance(articles, dict):
            continue

        for article_key, text in articles.items():
            if not text or not str(text).strip():
                continue

            # ✅ نخزن رقم المادة كـ string "20"
            article_num = str(article_key).strip()

            docs.append(
                Document(
                    page_content=str(text).strip(),
                    metadata={
                        "source": str(source_name),
                        "article": article_num,               # <-- مهم
                        "key": f"{source_name}||{article_num}" # اختياري للتتبع
                    },
                )
            )

    print(f"Prepared documents: {len(docs)}")

    embeddings_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"},
    )

    # ✅ افتح القاعدة
    vector_store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings_model,
        persist_directory=CHROMA_PATH,
    )

    # ✅ امسح كل اللي داخل collection (عشان ما يصير خلط)
    try:
        vector_store._collection.delete(where={})
        print("✅ Cleared existing collection.")
    except Exception as e:
        print("Could not clear collection (maybe empty):", e)

    # ✅ أضف الدوكيومنتس
    vector_store.add_documents(docs)

    # ✅ تحقق
    try:
        print("Chroma docs count:", vector_store._collection.count())
        check = vector_store.get(where={"article": "20"}, include=["metadatas"], limit=5)
        print("Check article=20 metadatas:", check.get("metadatas"))
    except Exception as e:
        print("Debug error:", e)

    print("Done. ChromaDB built from JSON.")

if __name__ == "__main__":
    main()
