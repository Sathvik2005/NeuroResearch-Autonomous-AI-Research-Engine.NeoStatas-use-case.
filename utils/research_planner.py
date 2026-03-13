import logging
from typing import Dict, List, Optional

from models.llm import generate_response

logger = logging.getLogger(__name__)


def _parse_plan(raw_text: str, max_steps: int = 6) -> List[str]:
    lines = [line.strip(" -\t") for line in raw_text.splitlines() if line.strip()]
    steps = []
    for line in lines:
        cleaned = line
        if ". " in line[:4]:
            cleaned = line.split(". ", 1)[1].strip()
        if cleaned and cleaned not in steps:
            steps.append(cleaned)
        if len(steps) >= max_steps:
            break
    return steps


def generate_research_plan(
    query: str,
    provider: Optional[str] = None,
    max_steps: int = 5,
    provider_keys: Optional[Dict[str, str]] = None,
) -> List[str]:
    try:
        prompt = (
            "Create a concise multi-step research plan for this query. "
            "Return only numbered steps.\n\n"
            f"Query: {query}"
        )
        plan_text = generate_response(
            prompt=prompt,
            mode="concise",
            provider=provider,
            provider_keys=provider_keys,
        )
        steps = _parse_plan(plan_text, max_steps=max_steps)
        if steps:
            return steps
    except Exception as exc:
        logger.exception("Research planning failed: %s", exc)

    return [
        "Clarify the topic scope and key terminology.",
        "Collect evidence from local documents.",
        "Collect recent evidence from web sources.",
        "Synthesize findings across sources.",
        "Summarize implications and future outlook.",
    ]
