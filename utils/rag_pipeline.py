import logging
from typing import Dict, List, Tuple

from config.config import settings
from utils.document_loader import load_documents
from utils.text_splitter import split_documents
from utils.vector_store import VectorStore, create_vector_store, search_vector_store

logger = logging.getLogger(__name__)


def build_vector_store_from_uploads(uploaded_files) -> Tuple[VectorStore, int, int]:
    try:
        documents = load_documents(uploaded_files)
        chunks, metadata = split_documents(documents)
        store = create_vector_store(chunks, metadata)
        return store, len(documents), len(chunks)
    except Exception as exc:
        logger.exception("Failed to build vector store: %s", exc)
        raise


def retrieve_rag_context(
    query: str,
    vector_store: VectorStore,
    top_k: int = settings.default_top_k,
) -> Tuple[str, List[str], List[Dict]]:
    try:
        matches = search_vector_store(vector_store, query, top_k=top_k)
        if not matches:
            return "", [], []

        context_parts = []
        sources = []

        for item in matches:
            source = item.get("metadata", {}).get("source", "unknown")
            sources.append(source)
            context_parts.append(item.get("chunk", ""))

        unique_sources = sorted(set(sources))
        context = "\n\n".join(context_parts)
        return context, unique_sources, matches
    except Exception as exc:
        logger.exception("RAG retrieval failed: %s", exc)
        return "", [], []


def compose_rag_prompt(query: str, context: str, mode: str = "concise") -> str:
    return (
        "Use the retrieved document context to answer the question. "
        "If context is insufficient, be explicit about limitations.\n\n"
        f"Response mode: {mode}\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved context:\n{context}"
    )
