"""
Configuration module for SFDA Cosmetics Chatbot.
Centralizes all configuration settings and environment variables.
"""

import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = os.getenv("DATA_PATH", "knowledge")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")

# Database configuration
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sfda_collection")

# Embedding model configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# LLM configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-chat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "700"))
LLM_BASE_URL = "https://openrouter.ai/api/v1"

# RAG configuration
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "8"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2000"))

# Application configuration
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Regulation JSON configuration
REG_JSON_NAME_HINTS = ["sfda_articles", "articles", "لوائح", "اللائحة"]

# Categories
CategoryType = Literal["regulation", "banned", "generic_json", "generic_jsonl", "raw_pdf"]

# Source type
SourceType = Literal["pdf", "json", "jsonl", "excel"]

# Validation
def validate_config() -> None:
    """Validate required configuration settings."""
    errors = []

    if not OPENROUTER_API_KEY and not OPENAI_API_KEY:
        errors.append("Either OPENROUTER_API_KEY or OPENAI_API_KEY must be set in .env file")

    if not Path(DATA_PATH).exists():
        errors.append(f"Data directory not found: {DATA_PATH}")

    if CHUNK_SIZE <= CHUNK_OVERLAP:
        errors.append(f"CHUNK_SIZE ({CHUNK_SIZE}) must be greater than CHUNK_OVERLAP ({CHUNK_OVERLAP})")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

# Run validation on import
if __name__ != "__main__":
    try:
        validate_config()
    except ValueError as e:
        print(f"⚠️ Configuration Warning: {e}")
