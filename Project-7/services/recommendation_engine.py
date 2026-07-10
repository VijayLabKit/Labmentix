"""
Semantic recommendation engine backed by FAISS.

Builds TF-IDF text embeddings for hotels and places of interest and indexes
them with FAISS for cosine-similarity search. This enables "semantic"
recommendations such as:

    >>> engine.recommend_hotels("luxury beach resort with pool and spa", top_k=3)
    >>> engine.recommend_places("historic forts and museums", top_k=5)

Design notes
------------
* TF-IDF (via scikit-learn) is used instead of a hosted embedding model so
  the engine works fully offline -- there is no dependency on downloading
  pretrained weights, which keeps the project runnable in restricted/
  air-gapped environments while still demonstrating a real vector-search
  pipeline.
* Vectors are L2-normalised so that FAISS's inner-product index
  (``IndexFlatIP``) computes cosine similarity.
* If the optional ``faiss`` dependency is not installed, the engine
  transparently falls back to a NumPy-based cosine-similarity search so the
  rest of the application keeps working (degraded but functional).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from utils.logger import get_logger

logger = get_logger("recommendation_engine")

try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when faiss missing
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False
    logger.warning("faiss is not installed; falling back to NumPy cosine similarity search.")


def _hotel_document(hotel: Dict[str, Any]) -> str:
    """Build the text document used to embed a hotel record."""
    amenities = " ".join(hotel.get("amenities", []))
    return (
        f"{hotel.get('name', '')} {hotel.get('city', '')} "
        f"{hotel.get('stars', '')} star hotel with amenities {amenities}"
    )


def _place_document(place: Dict[str, Any]) -> str:
    """Build the text document used to embed a place/attraction record."""
    from utils.helpers import categorise_place

    categories = " ".join(categorise_place(place.get("type", "")))
    return (
        f"{place.get('name', '')} {place.get('city', '')} {place.get('type', '')} "
        f"attraction categories {categories} rated {place.get('rating', '')}"
    )


class _VectorIndex:
    """Thin wrapper that uses FAISS if available, else NumPy cosine search."""

    def __init__(self, vectors: np.ndarray) -> None:
        self._vectors = vectors.astype("float32")
        self._dimension = self._vectors.shape[1] if self._vectors.size else 0
        self._index = None
        if _FAISS_AVAILABLE and self._dimension:
            self._index = faiss.IndexFlatIP(self._dimension)
            self._index.add(self._vectors)

    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, indices)`` for the ``top_k`` nearest vectors."""
        if self._dimension == 0:
            return np.array([[]]), np.array([[]], dtype=int)

        query = query_vector.astype("float32").reshape(1, -1)
        top_k = min(top_k, self._vectors.shape[0])

        if self._index is not None:
            scores, indices = self._index.search(query, top_k)
            return scores, indices

        # NumPy fallback: cosine similarity via dot product (vectors are
        # already L2-normalised).
        similarities = self._vectors @ query.T  # shape (n, 1)
        similarities = similarities.flatten()
        top_indices = np.argsort(-similarities)[:top_k]
        top_scores = similarities[top_indices]
        return top_scores.reshape(1, -1), top_indices.reshape(1, -1)


class RecommendationEngine:
    """Semantic recommendation engine for hotels and places.

    Args:
        hotels: List of hotel records (as loaded from ``hotels.json``).
        places: List of place records (as loaded from ``places.json``).
    """

    def __init__(self, hotels: List[Dict[str, Any]], places: List[Dict[str, Any]]) -> None:
        self._hotels = hotels
        self._places = places

        self._hotel_vectorizer = TfidfVectorizer(stop_words="english")
        self._place_vectorizer = TfidfVectorizer(stop_words="english")

        self._hotel_index = self._build_index(hotels, _hotel_document, self._hotel_vectorizer)
        self._place_index = self._build_index(places, _place_document, self._place_vectorizer)

        logger.info(
            "RecommendationEngine initialised | hotels={} places={} faiss={}",
            len(hotels),
            len(places),
            _FAISS_AVAILABLE,
        )

    @staticmethod
    def _build_index(records: List[Dict[str, Any]], doc_fn, vectorizer: TfidfVectorizer) -> _VectorIndex:
        if not records:
            return _VectorIndex(np.zeros((0, 0), dtype="float32"))

        documents = [doc_fn(record) for record in records]
        matrix = vectorizer.fit_transform(documents).toarray()
        matrix = normalize(matrix, norm="l2", axis=1)
        return _VectorIndex(matrix)

    def recommend_hotels(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return the ``top_k`` hotels most semantically similar to ``query_text``.

        Args:
            query_text: Free-text description of what the traveller wants
                (e.g. ``"affordable hotel with pool near the beach"``).
            top_k: Maximum number of results to return.

        Returns:
            A list of hotel dicts, each annotated with a ``semantic_score``
            in ``[-1, 1]`` (cosine similarity), sorted descending.
        """
        return self._recommend(query_text, top_k, self._hotels, self._hotel_vectorizer, self._hotel_index)

    def recommend_places(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return the ``top_k`` places most semantically similar to ``query_text``.

        Args:
            query_text: Free-text description of the traveller's interests
                (e.g. ``"historic forts and cultural museums"``).
            top_k: Maximum number of results to return.

        Returns:
            A list of place dicts, each annotated with a ``semantic_score``
            in ``[-1, 1]`` (cosine similarity), sorted descending.
        """
        return self._recommend(query_text, top_k, self._places, self._place_vectorizer, self._place_index)

    @staticmethod
    def _recommend(
        query_text: str,
        top_k: int,
        records: List[Dict[str, Any]],
        vectorizer: TfidfVectorizer,
        index: _VectorIndex,
    ) -> List[Dict[str, Any]]:
        if not records or not query_text.strip():
            return []

        query_vector = vectorizer.transform([query_text]).toarray()
        query_vector = normalize(query_vector, norm="l2", axis=1)[0]

        scores, indices = index.search(query_vector, top_k)
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(records):
                continue
            record = dict(records[int(idx)])
            record["semantic_score"] = round(float(score), 4)
            results.append(record)
        return results
