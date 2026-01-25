
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

DATA_PATH = "knowledge"
CHROMA_PATH = "chroma_db"

print(" Loading PDF documents")
loader = PyPDFDirectoryLoader(DATA_PATH)
docs = loader.load()
print(f"✅ Loaded {len(docs)} documents.")

for doc in docs:
    if "source" not in doc.metadata:
        doc.metadata["source"] = doc.metadata.get("file_path", "unknown")
    if "page" not in doc.metadata:
        doc.metadata["page"] = doc.metadata.get("page_number", "N/A")

print("✂️ Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f"✅ Split into {len(chunks)} chunks.")

print(" Loading embedding model on CPU...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': "cpu"}
)
print("✅ Embeddings model loaded.")

print("💾 Building Chroma vector store...")
db = Chroma(
    collection_name="sfda_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH
)
db.add_documents(chunks)
print(" Ingestion complete. Chroma DB has been saved successfully.")
