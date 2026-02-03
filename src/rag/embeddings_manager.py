from langchain_huggingface import HuggingFaceEmbeddings
from src.config import settings


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": settings.EMBEDDING_DEVICE},
    )