"""
Embeddings Service
Generates dense vector embeddings for text chunks and queries using sentence-transformers.
"""

from typing import List, Optional
import numpy as np
from config import EMBEDDING_MODEL_NAME

# Global cache for the embedding model to avoid repeated disk reads
_EMBEDDING_MODEL_CACHE = {}


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """
    Load or retrieve cached SentenceTransformer model.

    Args:
        model_name: HuggingFace model identifier (e.g. 'all-MiniLM-L6-v2').

    Returns:
        SentenceTransformer model instance.
    """
    global _EMBEDDING_MODEL_CACHE
    if model_name not in _EMBEDDING_MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
            # Load model onto CPU/GPU
            _EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers package is not installed. "
                "Please run: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{model_name}': {str(e)}")

    return _EMBEDDING_MODEL_CACHE[model_name]


def generate_embeddings(
    texts: List[str],
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = 32
) -> np.ndarray:
    """
    Convert a list of text strings into a 2D float32 numpy array of normalized vector embeddings.

    Args:
        texts: List of strings to encode.
        model_name: Name of embedding model.
        batch_size: Batch size for encoding.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim) with dtype float32.
    """
    if not texts:
        return np.empty((0, 384), dtype="float32")

    model = get_embedding_model(model_name)
    # normalize_embeddings=True allows cosine similarity to be computed via simple inner product (Dot Product)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embeddings.astype("float32")


def generate_query_embedding(
    query: str,
    model_name: str = EMBEDDING_MODEL_NAME
) -> np.ndarray:
    """
    Generate normalized embedding for a single search query.

    Args:
        query: Question or search string.
        model_name: Name of embedding model.

    Returns:
        np.ndarray of shape (1, embedding_dim) with dtype float32.
    """
    return generate_embeddings([query], model_name=model_name)
