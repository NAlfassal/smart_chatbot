from langchain_chroma import Chroma
from src.config import CHROMA_PATH, COLLECTION_NAME
from src.rag.embeddings_manager import get_embeddings
from src.utils.logger import logger 

def get_vector_store():
    try:
        embeddings = get_embeddings()
        
        persist_dir = str(CHROMA_PATH)
        
        logger.info(f"Connecting to ChromaDB at: {persist_dir} | Collection: {COLLECTION_NAME}")

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        return vector_store
        
    except Exception as e:
        logger.critical(f"Failed to connect to ChromaDB: {str(e)}")
        raise e