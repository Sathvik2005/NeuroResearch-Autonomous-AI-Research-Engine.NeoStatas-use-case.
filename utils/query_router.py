from typing import Literal

Route = Literal["rag", "web", "hybrid", "direct", "research", "deep_research"]


def route_query(
    query: str,
    has_documents: bool,
    research_mode: bool = False,
    deep_research_mode: bool = False,
) -> Route:
    normalized = (query or "").lower()
    web_markers = ["latest", "recent", "news", "today", "trend", "trends", "current"]
    compare_markers = ["compare", "vs", "versus", "difference", "research"]

    if deep_research_mode:
        return "deep_research"

    if research_mode:
        return "research"

    has_web_intent = any(marker in normalized for marker in web_markers)
    has_compare_intent = any(marker in normalized for marker in compare_markers)

    if has_documents and (has_web_intent or has_compare_intent):
        return "hybrid"
    if has_web_intent:
        return "web"
    if has_documents:
        return "rag"
    return "direct"
