"""Corpus embedding service built on top of a small TF-IDF corpus.

This module intentionally focuses on corpus/document embeddings (not single
token embeddings). It is designed for clarity and beginner-friendly behavior.
"""

from functools import lru_cache
from typing import TypedDict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

from utils.helpers import normalize_whitespace


# A small custom corpus to demonstrate corpus-level embeddings.
CUSTOM_CORPUS: list[str] = [
    "Natural language processing helps computers understand human language.",
    "Machine learning models learn patterns from data.",
    "Word embeddings capture semantic similarity between terms.",
    "FastAPI makes building Python APIs simple and fast.",
    "Tokenization splits text into meaningful pieces.",
    "Lemmatization reduces words to dictionary forms.",
    "Stemming cuts words using handcrafted rules.",
    "Named entities include people organizations and locations.",
    "Bengaluru is a growing technology hub in India.",
    "Seattle and New York host many AI conferences.",
    "Data scientists visualize vectors with PCA and t SNE.",
    "Similarity search can find related words quickly.",
    "Python libraries like spaCy and scikit learn power NLP workflows.",
    "A good corpus improves vocabulary quality and context coverage.",
]


class EmbeddingStore(TypedDict):
    """In-memory cached structure for vectorizer output and metadata."""

    vectorizer: TfidfVectorizer
    documents: list[str]
    doc_vectors: np.ndarray
    strengths: np.ndarray
    feature_names: list[str]


@lru_cache(maxsize=1)
def _build_store() -> EmbeddingStore:
    """Build and cache corpus TF-IDF vectors.

    Why this design:
    - TF-IDF is simple and explainable for beginners.
    - Caching avoids repeated fitting on every API call.
    """
    normalized_docs = [normalize_whitespace(doc).lower() for doc in CUSTOM_CORPUS]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
    )
    doc_term_matrix = vectorizer.fit_transform(normalized_docs)

    # Row i represents one corpus document in TF-IDF feature space.
    doc_vectors = doc_term_matrix.toarray().astype(np.float32)
    strengths = np.asarray(doc_vectors.sum(axis=1)).reshape(-1)

    return {
        "vectorizer": vectorizer,
        "documents": CUSTOM_CORPUS,
        "doc_vectors": doc_vectors,
        "strengths": strengths,
        "feature_names": vectorizer.get_feature_names_out().tolist(),
    }


def _normalize_text(text: str) -> str:
    """Normalize user input text for stable vectorization."""
    normalized = normalize_whitespace(text).lower()
    if not normalized:
        raise ValueError("Please provide non-empty text.")
    return normalized


def _cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one query vector and matrix rows."""
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float32)

    row_norms = np.linalg.norm(matrix, axis=1)
    return np.dot(matrix, query_vector) / ((row_norms * query_norm) + 1e-12)


def _nearest_corpus_neighbors(query_vector: np.ndarray, top_k: int) -> list[dict]:
    """Return nearest corpus entries for a query vector by cosine similarity."""
    store = _build_store()
    scores = _cosine_similarity(query_vector, store["doc_vectors"])
    sorted_indices = np.argsort(scores)[::-1]

    neighbors: list[dict] = []
    for index in sorted_indices:
        if len(neighbors) >= top_k:
            break
        if not np.isfinite(scores[index]):
            continue

        neighbors.append(
            {
                "doc_id": int(index),
                "text": store["documents"][index],
                "score": round(float(scores[index]), 4),
            }
        )

    return neighbors


def get_embedding_info() -> dict:
    """Return summary information about corpus embedding setup."""
    store = _build_store()
    return {
        "technique": "TF-IDF corpus/document embeddings",
        "corpus_size": len(store["documents"]),
        "vocabulary_size": len(store["feature_names"]),
        "vector_dimension": int(store["doc_vectors"].shape[1]),
    }


def get_corpus(limit: int = 80) -> list[dict]:
    """Return corpus documents for UI discovery and sampling."""
    store = _build_store()
    safe_limit = max(1, min(limit, len(store["documents"])))
    return [
        {
            "doc_id": idx,
            "text": store["documents"][idx],
        }
        for idx in range(safe_limit)
    ]


def get_corpus_embedding(text: str, top_k: int = 5) -> dict:
    """Return vector and nearest corpus entries for input text."""
    store = _build_store()
    normalized = _normalize_text(text)

    query_sparse = store["vectorizer"].transform([normalized])
    query_vector = query_sparse.toarray()[0].astype(np.float32)

    if float(np.linalg.norm(query_vector)) == 0.0:
        raise ValueError(
            "Input text has no overlap with the current corpus vocabulary. "
            "Try words related to NLP, machine learning, cities, or APIs."
        )

    return {
        "query_text": text,
        "dimension": int(query_vector.shape[0]),
        "vector": [round(float(value), 6) for value in query_vector.tolist()],
        "neighbors": _nearest_corpus_neighbors(query_vector, top_k),
        "corpus_size": len(store["documents"]),
    }


def project_embeddings(method: str = "pca", limit: int = 14) -> dict:
    """Project corpus document embeddings to 2D with PCA or t-SNE."""
    store = _build_store()
    normalized_method = method.lower()

    if normalized_method not in {"pca", "tsne"}:
        raise ValueError("Method must be either 'pca' or 'tsne'.")

    ranked_indices = np.argsort(store["strengths"])[::-1]
    safe_limit = max(2, min(limit, len(ranked_indices)))
    selected_indices = ranked_indices[:safe_limit]

    matrix = store["doc_vectors"][selected_indices]
    if matrix.shape[0] < 2:
        raise RuntimeError("Not enough corpus entries to project embeddings.")

    if normalized_method == "tsne":
        perplexity = max(2, min(10, matrix.shape[0] - 1))
        reducer = TSNE(
            n_components=2,
            random_state=42,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        )
    else:
        reducer = PCA(n_components=2, random_state=42)

    reduced = reducer.fit_transform(matrix)

    points: list[dict] = []
    for output_index, doc_index in enumerate(selected_indices):
        points.append(
            {
                "label": f"Doc {doc_index}",
                "text": store["documents"][doc_index],
                "x": round(float(reduced[output_index, 0]), 6),
                "y": round(float(reduced[output_index, 1]), 6),
                "importance": round(float(store["strengths"][doc_index]), 6),
            }
        )

    return {
        "method": normalized_method,
        "points": points,
    }
