import logging
from typing import Dict, List, Tuple

from tavily import TavilyClient

from config.config import settings

logger = logging.getLogger(__name__)


def summarize_search_results(results: List[Dict], max_chars: int = 2200) -> str:
    lines = []
    for idx, result in enumerate(results, start=1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = (result.get("content", "") or "").replace("\n", " ").strip()
        snippet = content[:300]
        lines.append(f"{idx}. {title} | {url}\nSnippet: {snippet}")

    summary = "\n\n".join(lines)
    return summary[:max_chars]


def perform_web_search(
    query: str,
    max_results: int = None,
    api_key: str = None,
) -> Tuple[str, List[str], List[Dict]]:
    try:
        key_to_use = api_key or settings.tavily_api_key
        if not key_to_use:
            raise ValueError("TAVILY_API_KEY is not configured.")

        client = TavilyClient(api_key=key_to_use)
        result = client.search(
            query=query,
            max_results=max_results or settings.max_search_results,
            include_answer=False,
            include_raw_content=False,
        )
        hits = result.get("results", []) if isinstance(result, dict) else []
        summary = summarize_search_results(hits)
        sources = [item.get("url", "") for item in hits if item.get("url")]
        return summary, sources, hits
    except Exception as exc:
        logger.exception("Web search failed: %s", exc)
        return "", [], []
