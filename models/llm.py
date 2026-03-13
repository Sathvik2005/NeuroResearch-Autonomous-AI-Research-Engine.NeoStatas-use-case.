import logging
from typing import Optional

from groq import Groq
from openai import OpenAI, RateLimitError

from config.config import settings

logger = logging.getLogger(__name__)


MODE_INSTRUCTIONS = {
    "concise": "Answer in 2 to 4 sentences. Be clear, factual, and compact.",
    "detailed": "Provide a detailed explanation with clear sections and practical examples.",
}


def _build_prompt(prompt: str, mode: str) -> str:
    instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["concise"])
    return f"{instruction}\n\nUser request:\n{prompt}"


def _call_openai(prompt: str, model_name: Optional[str]) -> str:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=model_name or settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def _call_groq(prompt: str, model_name: Optional[str]) -> str:
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is not configured.")
    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=model_name or settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def generate_response(
    prompt: str,
    mode: str = "concise",
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    selected_provider = (provider or settings.llm_provider).lower()
    full_prompt = _build_prompt(prompt, mode)

    try:
        if selected_provider == "openai":
            return _call_openai(full_prompt, model_name)
        if selected_provider == "groq":
            return _call_groq(full_prompt, model_name)
        raise ValueError(f"Unsupported LLM provider: {selected_provider}")
    except RateLimitError as exc:
        logger.exception("LLM generation failed with rate/quota error: %s", exc)
        if selected_provider == "openai" and settings.groq_api_key:
            logger.info("Falling back from OpenAI to Groq due to OpenAI quota/rate-limit error.")
            return _call_groq(full_prompt, None)
        raise
    except Exception as exc:
        logger.exception("LLM generation failed: %s", exc)
        raise
