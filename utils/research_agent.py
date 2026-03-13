import logging
from typing import Dict, List, Optional

from models.llm import generate_response
from utils.rag_pipeline import retrieve_rag_context
from utils.research_planner import generate_research_plan
from utils.web_search import perform_web_search

logger = logging.getLogger(__name__)


def _pick_top_points(text_blocks: List[str], max_points: int = 4) -> List[str]:
    candidates: List[str] = []
    for block in text_blocks:
        if not block:
            continue
        sentences = [segment.strip() for segment in block.replace("\n", " ").split(".") if segment.strip()]
        for sentence in sentences:
            if 25 <= len(sentence) <= 220 and sentence not in candidates:
                candidates.append(sentence)
            if len(candidates) >= max_points:
                return candidates
    return candidates


def _extractive_report_fallback(query: str, local_blocks: List[str], web_blocks: List[str]) -> str:
    local_points = _pick_top_points(local_blocks, max_points=4)
    web_points = _pick_top_points(web_blocks, max_points=4)

    overview_text = (
        local_points[0]
        if local_points
        else web_points[0]
        if web_points
        else f"A complete model-generated synthesis is unavailable right now, but collected evidence is summarized below for: {query}."
    )

    key_concepts = local_points[:2] + [point for point in web_points[:2] if point not in local_points[:2]]
    if not key_concepts:
        key_concepts = ["No high-confidence concept extraction could be completed from current evidence."]

    technologies = [
        "Large Language Models",
        "Retrieval-Augmented Generation",
        "Vector Similarity Search",
        "Hybrid Web and Document Retrieval",
    ]

    applications = [
        "Research assistants and knowledge copilots",
        "Document-grounded Q&A and summarization",
        "Enterprise search and decision support",
    ]

    trends = [
        "More robust retrieval orchestration and fallback handling",
        "Hybrid local-plus-web evidence pipelines",
        "Structured reporting with stronger source attribution",
    ]

    return (
        "Overview\n"
        f"{overview_text}.\n\n"
        "Key Concepts\n"
        + "\n".join([f"- {item}" for item in key_concepts])
        + "\n\n"
        + "Major Technologies\n"
        + "\n".join([f"- {item}" for item in technologies])
        + "\n\n"
        + "Applications\n"
        + "\n".join([f"- {item}" for item in applications])
        + "\n\n"
        + "Future Trends\n"
        + "\n".join([f"- {item}" for item in trends])
    )


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
    all_sources: List[str] = []
    local_evidence_blocks: List[str] = []
    web_evidence_blocks: List[str] = []
    errors: List[str] = []

    try:
        steps = (
            generate_research_plan(
                query,
                provider=provider,
                provider_keys=provider_keys,
            )
            if deep_research
            else []
        )
    except Exception as exc:
        logger.exception("Research planning phase failed: %s", exc)
        steps = []
        errors.append(f"planning: {exc}")

    execution_steps = steps if steps else [query]

    for step in execution_steps:
        try:
            local_context, local_sources, _ = retrieve_rag_context(step, vector_store, top_k=4)
            if local_context:
                local_evidence_blocks.append(local_context)
            all_sources.extend(local_sources)
        except Exception as exc:
            logger.exception("RAG step failed for '%s': %s", step, exc)
            errors.append(f"rag({step}): {exc}")

        try:
            web_context, web_sources, _ = perform_web_search(
                step,
                max_results=4,
                api_key=tavily_api_key,
            )
            if web_context:
                web_evidence_blocks.append(web_context)
            all_sources.extend(web_sources)
        except Exception as exc:
            logger.exception("Web step failed for '%s': %s", step, exc)
            errors.append(f"web({step}): {exc}")

    prompt = _build_report_prompt(
        query=query,
        local_context="\n\n".join(local_evidence_blocks),
        web_context="\n\n".join(web_evidence_blocks),
        steps=steps,
    )

    try:
        answer = generate_response(
            prompt=prompt,
            mode="detailed" if response_mode == "detailed" else "concise",
            provider=provider,
            provider_keys=provider_keys,
        )
    except Exception as exc:
        logger.exception("Research synthesis failed: %s", exc)
        errors.append(f"synthesis: {exc}")

        fallback_prompt = (
            "Provide a helpful answer to the query using available evidence. "
            "If evidence is missing, answer from general knowledge and mention limitations.\n\n"
            f"Query: {query}\n\n"
            f"Local evidence: {' | '.join(local_evidence_blocks[:2]) or 'none'}\n\n"
            f"Web evidence: {' | '.join(web_evidence_blocks[:2]) or 'none'}"
        )

        try:
            answer = generate_response(
                prompt=fallback_prompt,
                mode="concise" if response_mode == "concise" else "detailed",
                provider=provider,
                provider_keys=provider_keys,
            )
        except Exception as fallback_exc:
            logger.exception("Research fallback synthesis failed: %s", fallback_exc)
            errors.append(f"fallback: {fallback_exc}")
            deterministic_answer = _extractive_report_fallback(
                query=query,
                local_blocks=local_evidence_blocks,
                web_blocks=web_evidence_blocks,
            )
            return {
                "answer": deterministic_answer,
                "sources": sorted(set([src for src in all_sources if src])),
                "plan": steps,
                "errors": errors,
            }

    return {
        "answer": answer,
        "sources": sorted(set([src for src in all_sources if src])),
        "plan": steps,
        "errors": errors,
    }
