from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL, EMBEDDING_DEVICE
from src.utils.logger import logger

def get_embeddings():
    logger.info(f"Loading Embeddings: {EMBEDDING_MODEL} on device: {EMBEDDING_DEVICE}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
    )