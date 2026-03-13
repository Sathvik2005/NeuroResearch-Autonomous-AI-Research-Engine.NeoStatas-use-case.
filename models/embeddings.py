from functools import lru_cache
import hashlib
from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from config.config import settings


@lru_cache(maxsize=1)
def get_embedding_model():
    if SentenceTransformer is None:
        return None
    return SentenceTransformer(settings.embedding_model)


def _fallback_embed_single(text: str, dim: int = 384) -> np.ndarray:
    vector = np.zeros(dim, dtype="float32")
    tokens = text.lower().split()
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).hexdigest()
        idx = int(digest, 16) % dim
        vector[idx] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.array([], dtype="float32")

    model = get_embedding_model()
    if model is not None:
        embeddings = model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings, dtype="float32")

    embeddings = [_fallback_embed_single(text) for text in texts]
    return np.array(embeddings, dtype="float32")


def embed_text(text: str) -> np.ndarray:
    if not text:
        return np.array([], dtype="float32")
    return embed_texts([text])[0]
