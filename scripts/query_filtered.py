from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "regulations"

KEYWORDS = ["محظور", "محظورة", "مقيد", "مقيدة", "قائمة", "المادة الرابعة", "الماده الرابعه", "الهيئة", "تنشر"]

def score_text(text: str) -> int:
    t = text.replace("ـ", " ")
    return sum(1 for k in KEYWORDS if k in t)

def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)

    query = "قائمة المواد المحظورة والمواد المقيدة الاستخدام في منتجات التجميل المادة الرابعة"
    results = collection.query(
        query_texts=[query],
        n_results=25,
        where={"doc_name": "اللائحة-التنفيذية-لنظام-منتجات-التجميل.pdf"}
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    ranked = sorted(
        [(score_text(d), d, m) for d, m in zip(docs, metas)],
        key=lambda x: x[0],
        reverse=True
    )

    print("Query:", query)
    print("Top ranked results (after keyword rerank):")
    for i, (s, d, m) in enumerate(ranked[:10], start=1):
        print(f"\n--- {i} --- score={s}")
        print("meta:", m)
        print(d[:600])

if __name__ == "__main__":
    main()
