from typing import Dict, List, Tuple


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> List[str]:
    if not text:
        return []

    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - chunk_overlap)

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break

    return chunks


def split_documents(
    documents: List[Dict[str, str]], chunk_size: int = 500, chunk_overlap: int = 80
) -> Tuple[List[str], List[Dict[str, int]]]:
    all_chunks: List[str] = []
    metadata: List[Dict[str, int]] = []

    for doc in documents:
        doc_chunks = split_text(doc.get("text", ""), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, chunk in enumerate(doc_chunks):
            all_chunks.append(chunk)
            metadata.append({"source": doc.get("name", "unknown"), "chunk_index": idx})

    return all_chunks, metadata
