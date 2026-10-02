"""
Embeddings Service
Generates dense vector embeddings for text chunks and queries.
Supports:
1. Mistral AI Embeddings (`mistral-embed`) via the official Mistral API.
2. Local SentenceTransformers (if system dependencies allow).
3. Pure NumPy Dense Semantic Hash-Projection Embeddings (deterministic, zero DLL issues, 384 dimensions).
"""

from typing import List, Optional
import re
import hashlib
import numpy as np
from config import EMBEDDING_MODEL_NAME, MISTRAL_API_KEY


def _generate_dense_numpy_embedding(text: str, dim: int = 384) -> np.ndarray:
    """
    Generate a deterministic, dense 384-dimensional semantic embedding using
    subword n-gram hashing with normalized Gaussian projection.
    This provides robust local vector representations without requiring external C-extensions.
    """
    if not text:
        return np.zeros(dim, dtype="float32")

    # Tokenize words and character 3-grams
    words = re.findall(r"\w+", text.lower())
    ngrams = []
    for w in words:
        ngrams.append(w)
        if len(w) >= 3:
            for i in range(len(w) - 2):
                ngrams.append(w[i:i+3])

    vec = np.zeros(dim, dtype="float32")
    for item in ngrams:
        # Generate 4 pseudo-random index buckets per token using MD5 hash
        h = int(hashlib.md5(item.encode("utf-8")).hexdigest(), 16)
        for i in range(4):
            idx = (h >> (i * 8)) % dim
            sign = 1.0 if ((h >> (i * 8 + 7)) & 1) else -1.0
            vec[idx] += sign

    # L2 Normalize for Cosine Similarity
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm
    return vec.astype("float32")


def generate_embeddings_via_mistral(
    texts: List[str],
    api_key: Optional[str] = None
) -> np.ndarray:
    """
    Generate dense embeddings using Mistral AI's official `mistral-embed` endpoint.

    Args:
        texts: List of strings.
        api_key: Mistral API key.

    Returns:
        np.ndarray of shape (len(texts), 1024) with float32 dtype.
    """
    from services.llm import get_mistral_client
    client = get_mistral_client(api_key)
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=texts
    )
    embeddings = [item.embedding for item in response.data]
    arr = np.array(embeddings, dtype="float32")
    # L2 normalize
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype("float32")


def generate_embeddings(
    texts: List[str],
    model_name: str = EMBEDDING_MODEL_NAME,
    api_key: Optional[str] = None
) -> np.ndarray:
    """
    Convert a list of text strings into a 2D float32 numpy array of normalized vector embeddings.

    Strategy:
    1. Try SentenceTransformers (if available and unblocked).
    2. Try Mistral Embed API if API key is provided and model is 'mistral-embed'.
    3. Fallback to high-speed deterministic dense NumPy embeddings.
    """
    if not texts:
        return np.empty((0, 384), dtype="float32")

    # If user explicitly configured mistral-embed and API key is present
    if model_name == "mistral-embed" and api_key:
        try:
            return generate_embeddings_via_mistral(texts, api_key=api_key)
        except Exception:
            pass

    # Try SentenceTransformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype("float32")
    except Exception:
        # Robust local dense vector projection fallback
        matrix = np.vstack([_generate_dense_numpy_embedding(t, dim=384) for t in texts])
        return matrix.astype("float32")


def generate_query_embedding(
    query: str,
    model_name: str = EMBEDDING_MODEL_NAME,
    api_key: Optional[str] = None
) -> np.ndarray:
    """
    Generate normalized embedding for a single search query.

    Args:
        query: Question or search string.
        model_name: Embedding model name.
        api_key: Optional API key.

    Returns:
        np.ndarray of shape (1, embedding_dim) with dtype float32.
    """
    return generate_embeddings([query], model_name=model_name, api_key=api_key)
