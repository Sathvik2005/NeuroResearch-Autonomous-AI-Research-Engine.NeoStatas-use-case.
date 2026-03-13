from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional

import numpy as np

try:
    faiss = import_module("faiss")
except Exception:
    faiss = None

from models.embeddings import embed_text, embed_texts


@dataclass
class VectorStore:
    index: Optional[object] = None
    chunks: List[str] = field(default_factory=list)
    metadata: List[Dict] = field(default_factory=list)
    vectors: Optional[np.ndarray] = None
    dim: int = 0


def create_vector_store(chunks: List[str], metadatas: Optional[List[Dict]] = None) -> VectorStore:
    store = VectorStore()
    add_to_vector_store(store, chunks, metadatas)
    return store


def add_to_vector_store(
    store: VectorStore, chunks: List[str], metadatas: Optional[List[Dict]] = None
) -> VectorStore:
    if not chunks:
        return store

    vectors = embed_texts(chunks)
    if vectors.size == 0:
        return store

    vectors = np.array(vectors, dtype="float32")

    if faiss is not None:
        if store.index is None:
            store.dim = vectors.shape[1]
            store.index = faiss.IndexFlatL2(store.dim)
        store.index.add(vectors)
    else:
        if store.vectors is None:
            store.vectors = vectors
            store.dim = vectors.shape[1]
        else:
            store.vectors = np.vstack([store.vectors, vectors])

    store.chunks.extend(chunks)
    store.metadata.extend(metadatas or [{} for _ in chunks])
    return store


def search_vector_store(store: VectorStore, query: str, top_k: int = 4) -> List[Dict]:
    if not store or not query:
        return []

    query_vector = embed_text(query)
    if query_vector.size == 0:
        return []

    if faiss is not None and store.index is not None:
        query_vector = np.array([query_vector], dtype="float32")
        distances, indices = store.index.search(query_vector, top_k)
        ranked = list(zip(distances[0], indices[0]))
    else:
        if store.vectors is None or store.vectors.size == 0:
            return []
        similarity_scores = np.dot(store.vectors, query_vector)
        top_indices = np.argsort(-similarity_scores)[:top_k]
        ranked = [(float(1 - similarity_scores[idx]), int(idx)) for idx in top_indices]

    results = []
    for distance, idx in ranked:
        if idx < 0 or idx >= len(store.chunks):
            continue
        results.append(
            {
                "chunk": store.chunks[idx],
                "metadata": store.metadata[idx] if idx < len(store.metadata) else {},
                "score": float(distance),
            }
        )
    return results
