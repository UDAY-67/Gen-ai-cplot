"""
Vector Store Service (FAISS)
Handles index creation, vector storage, persistence, and semantic similarity search.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pickle
import numpy as np
from config import FAISS_INDEX_DIR, DEFAULT_TOP_K


class FAISSVectorStore:
    """
    In-memory and disk-persisted vector store utilizing FAISS for fast similarity search.
    Uses Inner Product (Cosine Similarity on normalized vectors).
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.chunks_metadata: List[Dict[str, Any]] = []
        self._init_index()

    def _init_index(self):
        """Initialize empty FAISS IndexFlatIP."""
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.dimension)
        except ImportError:
            raise ImportError(
                "faiss-cpu is not installed. Please run: pip install faiss-cpu"
            )

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> int:
        """
        Add text chunks and their pre-computed embeddings to the index.

        Args:
            chunks: List of metadata dicts corresponding to each embedding row.
            embeddings: 2D numpy float32 array of shape (N, dimension).

        Returns:
            Total number of chunks now indexed.
        """
        if len(chunks) == 0:
            return len(self.chunks_metadata)

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) does not match "
                f"number of embedding rows ({embeddings.shape[0]})."
            )

        # Ensure correct dimension
        if embeddings.shape[1] != self.dimension:
            # Dynamically update dimension if model dimension differs
            self.dimension = embeddings.shape[1]
            self._init_index()

        self.index.add(embeddings)
        self.chunks_metadata.extend(chunks)
        return len(self.chunks_metadata)

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Search for top-K most similar chunks for a given query vector.

        Args:
            query_embedding: 2D numpy float32 array of shape (1, dimension).
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of chunk dicts augmented with 'score' (cosine similarity, typically between 0.0 and 1.0).
        """
        if self.index is None or self.index.ntotal == 0 or len(self.chunks_metadata) == 0:
            return []

        # Clamp top_k to number of indexed elements
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks_metadata):
                chunk_copy = dict(self.chunks_metadata[idx])
                chunk_copy["score"] = float(round(score, 4))
                results.append(chunk_copy)

        return results

    def save(self, dir_path: Path = FAISS_INDEX_DIR) -> None:
        """Persist FAISS index and chunk metadata to disk."""
        import faiss
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        index_file = dir_path / "index.faiss"
        meta_file = dir_path / "metadata.pkl"

        if self.index is not None:
            faiss.write_index(self.index, str(index_file))

        with open(meta_file, "wb") as f:
            pickle.dump(self.chunks_metadata, f)

    def load(self, dir_path: Path = FAISS_INDEX_DIR) -> bool:
        """Load persisted FAISS index and chunk metadata from disk if available."""
        import faiss
        dir_path = Path(dir_path)
        index_file = dir_path / "index.faiss"
        meta_file = dir_path / "metadata.pkl"

        if not index_file.exists() or not meta_file.exists():
            return False

        try:
            self.index = faiss.read_index(str(index_file))
            self.dimension = self.index.d
            with open(meta_file, "rb") as f:
                self.chunks_metadata = pickle.load(f)
            return True
        except Exception:
            return False

    def clear(self) -> None:
        """Reset the vector store in memory."""
        self._init_index()
        self.chunks_metadata = []

    def get_total_chunks(self) -> int:
        """Return total indexed chunks."""
        return self.index.ntotal if self.index is not None else 0

    def get_indexed_documents(self) -> List[str]:
        """Return unique list of document names indexed."""
        docs = set()
        for chunk in self.chunks_metadata:
            if "doc_name" in chunk:
                docs.add(chunk["doc_name"])
        return sorted(list(docs))
