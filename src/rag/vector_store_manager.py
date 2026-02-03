from langchain_chroma import Chroma
from src.config import settings
from src.rag.embeddings_manager import get_embeddings


def get_vector_store():
    embeddings = get_embeddings()
    return Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(settings.CHROMA_PATH),
    )
