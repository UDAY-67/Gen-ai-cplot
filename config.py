"""
Application Configuration Module
Centralizes all configuration variables, environment variable loading, and defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Mistral AI Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
DEFAULT_MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

AVAILABLE_MISTRAL_MODELS = [
    "mistral-small-latest",
    "mistral-medium-latest",
    "mistral-large-latest",
    "open-mistral-7b",
    "open-mixtral-8x7b",
    "codestral-latest",
]

# Embeddings & Vector Search Defaults
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))

# Database & Storage
DB_PATH = BASE_DIR / os.getenv("DB_PATH", "data/study_copilot.db")
FAISS_INDEX_DIR = BASE_DIR / os.getenv("FAISS_INDEX_DIR", "data/faiss_index")

# Generation Defaults
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1024


def validate_api_key(api_key: str = None) -> bool:
    """Check if the provided or configured API key is validly formatted."""
    key = api_key or MISTRAL_API_KEY
    if not key or key == "your_mistral_api_key_here" or len(key) < 10:
        return False
    return True
