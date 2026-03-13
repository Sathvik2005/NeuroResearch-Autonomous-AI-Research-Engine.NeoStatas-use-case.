import logging
from typing import Dict, List, Optional

from models.llm import generate_response
from utils.rag_pipeline import retrieve_rag_context
from utils.research_planner import generate_research_plan
from utils.web_search import perform_web_search

logger = logging.getLogger(__name__)


def _build_report_prompt(
    query: str,
    local_context: str,
    web_context: str,
    steps: Optional[List[str]] = None,
) -> str:
    steps_text = "\n".join([f"- {step}" for step in (steps or [])])
    return (
        "Generate a structured research report using the provided evidence. "
        "Use this exact format with clear section headings:\n"
        "Overview\n"
        "Key Concepts\n"
        "Major Technologies\n"
        "Applications\n"
        "Future Trends\n\n"
        f"Query:\n{query}\n\n"
        f"Research plan:\n{steps_text or 'N/A'}\n\n"
        f"Local document evidence:\n{local_context or 'No local evidence found.'}\n\n"
        f"Web evidence:\n{web_context or 'No web evidence found.'}"
    )


def run_research_agent(
    query: str,
    vector_store,
    response_mode: str = "detailed",
    deep_research: bool = False,
    provider: Optional[str] = None,
    provider_keys: Optional[Dict[str, str]] = None,
    tavily_api_key: Optional[str] = None,
) -> Dict:
    try:
        all_sources: List[str] = []
        local_evidence_blocks: List[str] = []
        web_evidence_blocks: List[str] = []

        steps = (
            generate_research_plan(
                query,
                provider=provider,
                provider_keys=provider_keys,
            )
            if deep_research
            else []
        )
        execution_steps = steps if steps else [query]

        for step in execution_steps:
            local_context, local_sources, _ = retrieve_rag_context(step, vector_store, top_k=4)
            web_context, web_sources, _ = perform_web_search(
                step,
                max_results=4,
                api_key=tavily_api_key,
            )

            if local_context:
                local_evidence_blocks.append(local_context)
            if web_context:
                web_evidence_blocks.append(web_context)

            all_sources.extend(local_sources)
            all_sources.extend(web_sources)

        prompt = _build_report_prompt(
            query=query,
            local_context="\n\n".join(local_evidence_blocks),
            web_context="\n\n".join(web_evidence_blocks),
            steps=steps,
        )

        answer = generate_response(
            prompt=prompt,
            mode="detailed" if response_mode == "detailed" else "concise",
            provider=provider,
            provider_keys=provider_keys,
        )

        return {
            "answer": answer,
            "sources": sorted(set([src for src in all_sources if src])),
            "plan": steps,
        }
    except Exception as exc:
        logger.exception("Research agent failed: %s", exc)
        return {
            "answer": "The research workflow failed due to an internal error.",
            "sources": [],
            "plan": [],
        }
