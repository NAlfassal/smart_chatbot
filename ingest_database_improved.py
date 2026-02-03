"""
Improved Database Ingestion Module for SFDA Cosmetics Chatbot.

This module handles loading documents from various sources (PDF, JSON, JSONL, Excel)
and ingesting them into a ChromaDB vector store with proper error handling and logging.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Any

import pandas as pd
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO if config.DEBUG else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextNormalizer:
    """Handles text normalization operations."""

    @staticmethod
    def normalize_spaces(text: str) -> str:
        """
        Normalize whitespace in text.

        Args:
            text: Input text to normalize

        Returns:
            Text with normalized spaces
        """
        text = str(text)
        text = " ".join(text.split())
        return text.strip()

    @staticmethod
    def normalize_source_name(name: str) -> str:
        """
        Normalize source file names for display.

        Args:
            name: Source file name

        Returns:
            Normalized source name
        """
        s = TextNormalizer.normalize_spaces(name)
        s = s.replace("-", " ")
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip()


class DocumentLoader:
    """Handles loading documents from various sources."""

    def __init__(self, knowledge_dir: Path):
        """
        Initialize document loader.

        Args:
            knowledge_dir: Path to knowledge directory
        """
        self.knowledge_dir = knowledge_dir
        if not self.knowledge_dir.exists():
            raise FileNotFoundError(f"Knowledge directory not found: {self.knowledge_dir}")

    def load_excel_documents(self) -> List[Document]:
        """
        Load documents from Excel files.

        Returns:
            List of Document objects from Excel files
        """
        logger.info("📊 Loading Excel documents...")
        excel_docs: List[Document] = []

        for xlsx_path in self.knowledge_dir.glob("*.xlsx"):
            try:
                logger.info(f"   • Processing Excel file: {xlsx_path.name}")
                sheets = pd.read_excel(xlsx_path, sheet_name=None)

                for sheet_name, df in sheets.items():
                    df = df.copy().fillna("")
                    df.columns = [TextNormalizer.normalize_spaces(c) for c in df.columns]

                    for i, row in df.iterrows():
                        row_dict = {
                            col: TextNormalizer.normalize_spaces(row[col])
                            for col in df.columns
                            if TextNormalizer.normalize_spaces(row[col]) != ""
                        }
                        if not row_dict:
                            continue

                        text = self._row_to_text(row_dict)
                        excel_docs.append(
                            Document(
                                page_content=f"[محظورات التجميل | Excel:{xlsx_path.name} | Sheet:{sheet_name} | Row:{i}]\n{text}",
                                metadata={
                                    "source": TextNormalizer.normalize_source_name(xlsx_path.name),
                                    "type": "excel",
                                    "category": "banned",
                                    "sheet": sheet_name,
                                    "row_index": int(i),
                                },
                            )
                        )
            except Exception as e:
                logger.error(f"Error processing Excel file {xlsx_path.name}: {e}")
                continue

        logger.info(f"✅ Loaded {len(excel_docs)} Excel row-documents.")
        return excel_docs

    def load_json_documents(self) -> tuple[List[Document], List[Document]]:
        """
        Load documents from JSON and JSONL files.

        Returns:
            Tuple of (regulation_docs, generic_json_docs)
        """
        logger.info("🧩 Loading JSON/JSONL documents...")
        json_docs: List[Document] = []
        regulation_docs: List[Document] = []

        # Process JSON files
        for jp in sorted(self.knowledge_dir.glob("*.json")):
            try:
                logger.info(f"   • Processing JSON file: {jp.name}")
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if self._is_regulation_json(data):
                    docs = self._build_regulation_docs(data, source_file=jp.name)
                    regulation_docs.extend(docs)
                    logger.info(f"     ✅ Detected regulation JSON -> built {len(docs)} docs")
                else:
                    docs = self._process_generic_json(data, jp)
                    json_docs.extend(docs)

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in {jp.name}: {e}")
            except Exception as e:
                logger.error(f"Error processing JSON file {jp.name}: {e}")

        # Process JSONL files
        for jlp in sorted(self.knowledge_dir.glob("*.jsonl")):
            try:
                logger.info(f"   • Processing JSONL file: {jlp.name}")
                for idx, rec in enumerate(self._jsonl_iter(jlp)):
                    doc = self._process_jsonl_record(rec, jlp, idx)
                    if doc:
                        json_docs.append(doc)
            except Exception as e:
                logger.error(f"Error processing JSONL file {jlp.name}: {e}")

        logger.info(f"✅ Loaded {len(regulation_docs)} regulation docs from JSON.")
        logger.info(f"✅ Loaded {len(json_docs)} generic JSON/JSONL docs.")
        return regulation_docs, json_docs

    def load_pdf_documents(self, load_raw: bool = False) -> List[Document]:
        """
        Load documents from PDF files.

        Args:
            load_raw: Whether to load raw PDFs

        Returns:
            List of Document chunks from PDFs
        """
        if not load_raw:
            logger.info("📄 Skipping RAW PDF loading (LOAD_RAW_PDFS=False)")
            return []

        logger.info("📄 Loading RAW PDF documents...")
        try:
            pdf_loader = PyPDFDirectoryLoader(str(self.knowledge_dir))
            pdf_docs = pdf_loader.load()
            logger.info(f"✅ Loaded {len(pdf_docs)} PDF documents.")

            # Add metadata
            for d in pdf_docs:
                if "source" not in d.metadata:
                    d.metadata["source"] = d.metadata.get("file_path", "unknown")
                if "page" not in d.metadata:
                    d.metadata["page"] = d.metadata.get("page_number", "N/A")
                d.metadata["source"] = TextNormalizer.normalize_source_name(d.metadata["source"])
                d.metadata["type"] = "pdf"
                d.metadata["category"] = d.metadata.get("category", "raw_pdf")

            # Split into chunks
            logger.info("✂️ Splitting RAW PDFs into chunks...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            pdf_chunks = splitter.split_documents(pdf_docs)
            logger.info(f"✅ Split RAW PDFs into {len(pdf_chunks)} chunks.")
            return pdf_chunks

        except Exception as e:
            logger.error(f"Error loading PDF documents: {e}")
            return []

    @staticmethod
    def _row_to_text(row: dict) -> str:
        """Convert a record dict to a readable text block."""
        parts = []
        for k, v in row.items():
            if v is None:
                continue
            if str(k).startswith("_"):
                continue
            parts.append(f"{k}: {v}")
        return "\n".join(parts).strip()

    @staticmethod
    def _is_regulation_json(data: dict) -> bool:
        """
        Detect regulation JSON format.

        Args:
            data: JSON data to check

        Returns:
            True if this is a regulation JSON format
        """
        if not isinstance(data, dict) or not data:
            return False

        for _, v in data.items():
            if isinstance(v, dict):
                for kk in v.keys():
                    if re.fullmatch(r"\d+", str(kk).strip()):
                        return True
        return False

    @staticmethod
    def _build_regulation_docs(data: dict, source_file: str) -> List[Document]:
        """
        Build regulation documents from JSON structure.

        Args:
            data: JSON data containing regulations
            source_file: Source file name

        Returns:
            List of regulation documents
        """
        docs = []
        for doc_name, sections in data.items():
            if not isinstance(sections, dict):
                continue

            src = TextNormalizer.normalize_source_name(doc_name)

            for key, text in sections.items():
                key_str = str(key).strip()
                text_str = str(text or "").strip()
                if not text_str:
                    continue

                # Numeric key => Article
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
                    # Section title
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

    def _process_generic_json(self, data: Any, jp: Path) -> List[Document]:
        """Process generic JSON data."""
        docs = []

        if isinstance(data, dict):
            for k, v in data.items():
                if v is None:
                    continue
                content = TextNormalizer.normalize_spaces(v) if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                if not content:
                    continue
                docs.append(
                    Document(
                        page_content=f"[JSON:{jp.name} | Key:{k}]\n{content}",
                        metadata={
                            "source": TextNormalizer.normalize_source_name(jp.name),
                            "type": "json",
                            "category": "generic_json",
                            "json_key": str(k),
                        },
                    )
                )

        elif isinstance(data, list):
            for idx, rec in enumerate(data):
                doc = self._process_json_list_item(rec, jp, idx)
                if doc:
                    docs.append(doc)

        return docs

    def _process_json_list_item(self, rec: Any, jp: Path, idx: int) -> Optional[Document]:
        """Process a single item from a JSON list."""
        if not isinstance(rec, dict):
            content = TextNormalizer.normalize_spaces(rec)
            if not content:
                return None
            return Document(
                page_content=f"[JSON:{jp.name} | Index:{idx}]\n{content}",
                metadata={
                    "source": TextNormalizer.normalize_source_name(jp.name),
                    "type": "json",
                    "category": "generic_json",
                    "index": idx,
                },
            )

        text = self._row_to_text(rec) or json.dumps(rec, ensure_ascii=False)
        if not text:
            return None

        meta = {
            "source": TextNormalizer.normalize_source_name(jp.name),
            "type": "json",
            "category": "generic_json",
            "index": idx,
        }

        # Add optional metadata fields
        for mk in ["doc_name", "page", "pages", "article_no", "article_key", "sheet", "dataset", "category"]:
            if mk in rec and rec[mk] is not None:
                meta[mk] = rec[mk]

        return Document(page_content=f"[JSON:{jp.name} | Index:{idx}]\n{text}", metadata=meta)

    @staticmethod
    def _jsonl_iter(path: Path) -> Iterator[Any]:
        """Yield json objects from a .jsonl file safely."""
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception as e:
                    logger.warning(f"Skipping bad JSONL line {line_no} in {path.name}: {e}")

    def _process_jsonl_record(self, rec: Any, jlp: Path, idx: int) -> Optional[Document]:
        """Process a single JSONL record."""
        if isinstance(rec, dict):
            text = self._row_to_text(rec) or json.dumps(rec, ensure_ascii=False)
            meta = {
                "source": TextNormalizer.normalize_source_name(jlp.name),
                "type": "jsonl",
                "category": "generic_jsonl",
                "index": idx,
            }
            for mk in ["doc_name", "page", "pages", "article_no", "article_key", "sheet", "dataset", "category"]:
                if mk in rec and rec[mk] is not None:
                    meta[mk] = rec[mk]
            return Document(page_content=f"[JSONL:{jlp.name} | Index:{idx}]\n{text}", metadata=meta)
        else:
            content = TextNormalizer.normalize_spaces(rec)
            if not content:
                return None
            return Document(
                page_content=f"[JSONL:{jlp.name} | Index:{idx}]\n{content}",
                metadata={
                    "source": TextNormalizer.normalize_source_name(jlp.name),
                    "type": "jsonl",
                    "category": "generic_jsonl",
                    "index": idx,
                },
            )


class VectorStoreManager:
    """Manages ChromaDB vector store operations."""

    def __init__(self):
        """Initialize vector store manager."""
        logger.info("🧠 Loading embedding model...")
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": config.EMBEDDING_DEVICE},
        )
        logger.info("✅ Embeddings model loaded.")

    def build_vector_store(self, documents: List[Document]) -> Chroma:
        """
        Build and populate ChromaDB vector store.

        Args:
            documents: List of documents to ingest

        Returns:
            Chroma vector store instance
        """
        logger.info("💾 Building Chroma vector store...")

        try:
            db = Chroma(
                collection_name=config.COLLECTION_NAME,
                embedding_function=self.embeddings_model,
                persist_directory=config.CHROMA_PATH,
            )

            logger.info(f"📦 Total documents to add: {len(documents)}")

            # Add in batches to avoid Chroma max batch size error
            for i in range(0, len(documents), config.BATCH_SIZE):
                batch = documents[i:i + config.BATCH_SIZE]
                logger.info(f" Adding batch {i//config.BATCH_SIZE + 1} | size={len(batch)}")
                db.add_documents(batch)

            logger.info("✅ All batches added successfully.")
            return db

        except Exception as e:
            logger.error(f"Error building vector store: {e}")
            raise


def main():
    """Main ingestion pipeline."""
    try:
        logger.info("=" * 60)
        logger.info("Starting SFDA Cosmetics Database Ingestion")
        logger.info("=" * 60)

        # Initialize loader
        knowledge_dir = Path(config.DATA_PATH)
        loader = DocumentLoader(knowledge_dir)

        # Load documents from all sources
        excel_docs = loader.load_excel_documents()
        regulation_docs, json_docs = loader.load_json_documents()
        pdf_chunks = loader.load_pdf_documents(load_raw=False)

        # Split documents (but not regulation docs - keep them whole)
        splitter_generic = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

        excel_chunks = splitter_generic.split_documents(excel_docs) if excel_docs else []
        json_chunks = splitter_generic.split_documents(json_docs) if json_docs else []

        # Combine all documents
        all_docs = regulation_docs + pdf_chunks + excel_chunks + json_chunks

        logger.info(f"\n📊 Document Summary:")
        logger.info(f"   - Regulation docs: {len(regulation_docs)}")
        logger.info(f"   - PDF chunks: {len(pdf_chunks)}")
        logger.info(f"   - Excel chunks: {len(excel_chunks)}")
        logger.info(f"   - JSON chunks: {len(json_chunks)}")
        logger.info(f"   - Total: {len(all_docs)}\n")

        # Build vector store
        manager = VectorStoreManager()
        db = manager.build_vector_store(all_docs)

        logger.info("=" * 60)
        logger.info("✅ Ingestion complete. Chroma DB has been saved successfully.")
        logger.info(f"   Collection: {config.COLLECTION_NAME}")
        logger.info(f"   Persist dir: {config.CHROMA_PATH}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Fatal error during ingestion: {e}")
        raise


if __name__ == "__main__":
    main()
