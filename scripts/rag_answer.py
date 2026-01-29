from pathlib import Path
import os
import chromadb
from chromadb.utils import embedding_functions

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "regulations"

KEYWORDS = ["محظور", "محظورة", "مقيد", "مقيدة", "قائمة", "المادة الرابعة", "الماده الرابعه", "تنشر", "الهيئة"]

def score_text(text: str) -> int:
    t = text.replace("ـ", " ")
    return sum(1 for k in KEYWORDS if k in t)

def main():
    # 1) Load Chroma
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)

    # 2) Query
    query = "ما هي القوائم المتعلقة بالمواد المحظورة والمواد المقيدة في منتجات التجميل؟ وكيف يتم تحديثها؟"
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

    top = ranked[:3]  # best 3 chunks
    context_blocks = []
    citations = []

    for idx, (s, d, m) in enumerate(top, start=1):
        doc_name = m.get("doc_name", "NA")
        page = m.get("page", "NA")
        article = m.get("article_no", "NA")
        context_blocks.append(
            f"[Source {idx}] doc={doc_name} page={page} article={article}\n{d}"
        )
        citations.append(f"Source {idx}: {doc_name} (page {page}, article {article})")

    context = "\n\n".join(context_blocks)

    # 3) LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)  # غيّري الموديل إذا تبين

    system = SystemMessage(
        content=(
            "أنت مساعد امتثال. أجب بالعربية فقط وبشكل مختصر وواضح.\n"
            "استخدم المعلومات الموجودة في CONTEXT فقط ولا تخمن.\n"
            "في نهاية الإجابة ضع قسم (المراجع) واذكر doc + page + article لكل مصدر استخدمته."
        )
    )

    user = HumanMessage(
        content=f"QUESTION:\n{query}\n\nCONTEXT:\n{context}"
    )

    answer = llm.invoke([system, user]).content

    print("\n===== ANSWER =====\n")
    print(answer)

    print("\n===== USED SOURCES (debug) =====\n")
    for c in citations:
        print("-", c)

if __name__ == "__main__":
    main()
