from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

import os
import json
import pandas as pd
from pathlib import Path
import re

load_dotenv()

DATA_PATH = "knowledge"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "sfda_collection"

# ✅ مهم: ملفات اللوائح الأفضل تكون JSON منظم (dict of dict)
# مثال: sfda_articles.json داخل knowledge/
REG_JSON_NAME_HINTS = ["sfda_articles", "articles", "لوائح", "اللائحة"]


# -------------------------------
# Helpers
# -------------------------------
def normalize_spaces(text: str) -> str:
    text = str(text)
    text = " ".join(text.split())
    return text.strip()


def normalize_source_name(name: str) -> str:
    """لإزالة الشرطات وتوحيد اسم المصدر للعرض"""
    s = normalize_spaces(name)
    s = s.replace("-", " ")
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def jsonl_iter(path: Path):
    """Yield json objects from a .jsonl file safely."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"⚠️ Skipping bad JSONL line {line_no} in {path.name}: {e}")


def row_to_text(row: dict) -> str:
    """Convert a record dict to a readable text block."""
    parts = []
    for k, v in row.items():
        if v is None:
            continue
        if str(k).startswith("_"):
            continue
        parts.append(f"{k}: {v}")
    return "\n".join(parts).strip()


def is_regulation_json(data: dict) -> bool:
    """
    Detect your regulation JSON format:
    {
      "اسم مستند": { "1": "...", "2": "...", "الاستدعاءات": "...", ... },
      ...
    }
    """
    if not isinstance(data, dict) or not data:
        return False

    # if any value is a dict that contains numeric keys -> likely regulation format
    for _, v in data.items():
        if isinstance(v, dict):
            for kk in v.keys():
                if re.fullmatch(r"\d+", str(kk).strip()):
                    return True
    return False


def build_regulation_docs(data: dict, source_file: str):
    """
    Build Documents where:
    - each numeric key => article doc with metadata article="4"
    - each non-numeric key => section doc
    """
    docs = []
    for doc_name, sections in data.items():
        if not isinstance(sections, dict):
            continue

        src = normalize_source_name(doc_name)

        for key, text in sections.items():
            key_str = str(key).strip()
            text_str = str(text or "").strip()
            if not text_str:
                continue

            # numeric => Article
            if re.fullmatch(r"\d+", key_str):
                docs.append(
                    Document(
                        page_content=f"المادة {key_str}\n\n{text_str}",
                        metadata={
                            "source": src,
                            "source_file": source_file,
                            "type": "pdf",
                            "category": "regulation",
                            "article": key_str,
                            "section": f"المادة {key_str}",
                        },
                    )
                )
            else:
                # section title
                docs.append(
                    Document(
                        page_content=f"{key_str}\n\n{text_str}",
                        metadata={
                            "source": src,
                            "source_file": source_file,
                            "type": "pdf",
                            "category": "regulation",
                            "section": key_str,
                        },
                    )
                )
    return docs


# -------------------------------
# 1) Load Excel (.xlsx) -> Documents (each row is a Document)
# -------------------------------
print("📊 Loading Excel documents...")
excel_docs: list[Document] = []
knowledge_dir = Path(DATA_PATH)

for xlsx_path in knowledge_dir.glob("*.xlsx"):
    print(f"   • Excel file: {xlsx_path.name}")

    sheets = pd.read_excel(xlsx_path, sheet_name=None)

    for sheet_name, df in sheets.items():
        df = df.copy().fillna("")
        df.columns = [normalize_spaces(c) for c in df.columns]

        for i, row in df.iterrows():
            row_dict = {
                col: normalize_spaces(row[col])
                for col in df.columns
                if normalize_spaces(row[col]) != ""
            }
            if not row_dict:
                continue

            text = row_to_text(row_dict)

            excel_docs.append(
                Document(
                    page_content=f"[محظورات التجميل | Excel:{xlsx_path.name} | Sheet:{sheet_name} | Row:{i}]\n{text}",
                    metadata={
                        "source": normalize_source_name(xlsx_path.name),
                        "type": "excel",
                        "category": "banned",
                        "sheet": sheet_name,
                        "row_index": int(i),
                    },
                )
            )

print(f"✅ Loaded {len(excel_docs)} Excel row-documents.")


# -------------------------------
# 2) Load JSON / JSONL files -> Documents
#   ✅ هنا التعديل الأكبر: ندعم JSON اللوائح (dict داخل dict) بشكل صحيح
# -------------------------------
print("🧩 Loading JSON/JSONL documents...")
json_docs: list[Document] = []
regulation_docs: list[Document] = []

json_files = sorted(list(knowledge_dir.glob("*.json")))
jsonl_files = sorted(list(knowledge_dir.glob("*.jsonl")))

# JSON
for jp in json_files:
    print(f"   • JSON file: {jp.name}")
    try:
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ Could not load {jp.name}: {e}")
        continue

    # ✅ إذا هذا JSON حق اللوائح (dict of dict) => نبنيه كمستندات مواد
    if is_regulation_json(data):
        docs = build_regulation_docs(data, source_file=jp.name)
        regulation_docs.extend(docs)
        print(f"     ✅ Detected regulation JSON -> built {len(docs)} docs (articles/sections).")
        continue

    # باقي الـ JSON العادي (dict أو list)
    if isinstance(data, dict):
        for k, v in data.items():
            if v is None:
                continue
            content = normalize_spaces(v) if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            if not content:
                continue
            json_docs.append(
                Document(
                    page_content=f"[JSON:{jp.name} | Key:{k}]\n{content}",
                    metadata={
                        "source": normalize_source_name(jp.name),
                        "type": "json",
                        "category": "generic_json",
                        "json_key": str(k),
                    },
                )
            )

    elif isinstance(data, list):
        for idx, rec in enumerate(data):
            if not isinstance(rec, dict):
                content = normalize_spaces(rec)
                if not content:
                    continue
                json_docs.append(
                    Document(
                        page_content=f"[JSON:{jp.name} | Index:{idx}]\n{content}",
                        metadata={
                            "source": normalize_source_name(jp.name),
                            "type": "json",
                            "category": "generic_json",
                            "index": idx,
                        },
                    )
                )
                continue

            text = row_to_text(rec) or json.dumps(rec, ensure_ascii=False)
            if not text:
                continue

            meta = {
                "source": normalize_source_name(jp.name),
                "type": "json",
                "category": "generic_json",
                "index": idx,
            }
            for mk in ["doc_name", "page", "pages", "article_no", "article_key", "sheet", "dataset", "category"]:
                if mk in rec and rec[mk] is not None:
                    meta[mk] = rec[mk]

            json_docs.append(Document(page_content=f"[JSON:{jp.name} | Index:{idx}]\n{text}", metadata=meta))
    else:
        print(f"⚠️ {jp.name} format not supported. Skipping.")


# JSONL
for jlp in jsonl_files:
    print(f"   • JSONL file: {jlp.name}")
    for idx, rec in enumerate(jsonl_iter(jlp)):
        if isinstance(rec, dict):
            text = row_to_text(rec) or json.dumps(rec, ensure_ascii=False)
            meta = {
                "source": normalize_source_name(jlp.name),
                "type": "jsonl",
                "category": "generic_jsonl",
                "index": idx,
            }
            for mk in ["doc_name", "page", "pages", "article_no", "article_key", "sheet", "dataset", "category"]:
                if mk in rec and rec[mk] is not None:
                    meta[mk] = rec[mk]
            json_docs.append(Document(page_content=f"[JSONL:{jlp.name} | Index:{idx}]\n{text}", metadata=meta))
        else:
            content = normalize_spaces(rec)
            if not content:
                continue
            json_docs.append(
                Document(
                    page_content=f"[JSONL:{jlp.name} | Index:{idx}]\n{content}",
                    metadata={
                        "source": normalize_source_name(jlp.name),
                        "type": "jsonl",
                        "category": "generic_jsonl",
                        "index": idx,
                    },
                )
            )

print(f"✅ Loaded {len(regulation_docs)} regulation docs from JSON.")
print(f"✅ Loaded {len(json_docs)} generic JSON/JSONL docs.")


# -------------------------------
# 3) (اختياري) Load PDF documents (raw)
#    ✅ مهم: نخليه OPTIONAL لأن عندك لوائح جاهزة من JSON
#    لو تبين تخليه: اتركيه True
# -------------------------------
LOAD_RAW_PDFS = False  # ❗ خليها False عشان ما نخرب المواد بالـ chunks

pdf_chunks: list[Document] = []

if LOAD_RAW_PDFS:
    print("📄 Loading RAW PDF documents...")
    pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
    pdf_docs = pdf_loader.load()
    print(f"✅ Loaded {len(pdf_docs)} PDF documents.")

    for d in pdf_docs:
        if "source" not in d.metadata:
            d.metadata["source"] = d.metadata.get("file_path", "unknown")
        if "page" not in d.metadata:
            d.metadata["page"] = d.metadata.get("page_number", "N/A")
        d.metadata["source"] = normalize_source_name(d.metadata["source"])
        d.metadata["type"] = "pdf"
        d.metadata["category"] = d.metadata.get("category", "raw_pdf")

    print("✂️ Splitting RAW PDFs into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pdf_chunks = splitter.split_documents(pdf_docs)
    print(f"✅ Split RAW PDFs into {len(pdf_chunks)} chunks.")


# -------------------------------
# 4) Build embeddings + Chroma
# -------------------------------
print("🧠 Loading embedding model on CPU...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cpu"},
)
print("✅ Embeddings model loaded.")

print("💾 Building Chroma vector store...")
db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# ✅ IMPORTANT: لا نجزئ regulation_docs (مواد) — نخليها كاملة
# ✅ لكن نجزئ excel/json لأنها قد تكون طويلة/متنوعة
splitter_generic = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

excel_chunks = splitter_generic.split_documents(excel_docs) if excel_docs else []
json_chunks = splitter_generic.split_documents(json_docs) if json_docs else []

all_docs = regulation_docs + pdf_chunks + excel_chunks + json_chunks

print(f"📦 Total documents to add: {len(all_docs)}")

# ---- Add in batches to avoid Chroma max batch size error ----
BATCH_SIZE = 2000
for i in range(0, len(all_docs), BATCH_SIZE):
    batch = all_docs[i:i + BATCH_SIZE]
    print(f" Adding batch {i//BATCH_SIZE + 1} | size={len(batch)}")
    db.add_documents(batch)

print("✅ All batches added successfully.")
print("✅ Ingestion complete. Chroma DB has been saved successfully.")
print(f"   Collection: {COLLECTION_NAME}")
print(f"   Persist dir: {CHROMA_PATH}")
