import logging
from typing import Dict, Optional

from groq import Groq
from openai import OpenAI, RateLimitError

from config.config import settings

logger = logging.getLogger(__name__)


MODE_INSTRUCTIONS = {
    "concise": "Answer in 2 to 4 sentences. Be clear, factual, and compact.",
    "detailed": "Provide a detailed explanation with clear sections and practical examples.",
}


def _provider_sequence(selected_provider: str, provider_keys: Optional[Dict[str, str]] = None):
    sequence = [selected_provider]
    if selected_provider == "openai":
        sequence.append("groq")
    elif selected_provider == "groq":
        sequence.append("openai")
    else:
        sequence = ["openai", "groq"]

    runtime_keys = provider_keys or {}
    available = [provider for provider in sequence if _has_provider_key(provider, runtime_keys)]
    return available


def _has_provider_key(provider: str, provider_keys: Optional[Dict[str, str]] = None) -> bool:
    runtime_keys = provider_keys or {}
    if provider == "openai":
        return bool(runtime_keys.get("openai") or settings.openai_api_key)
    if provider == "groq":
        return bool(runtime_keys.get("groq") or settings.groq_api_key)
    return False


def _build_prompt(prompt: str, mode: str) -> str:
    instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["concise"])
    return f"{instruction}\n\nUser request:\n{prompt}"


def _call_openai(prompt: str, model_name: Optional[str], api_key: Optional[str] = None) -> str:
    key_to_use = api_key or settings.openai_api_key
    if not key_to_use:
        raise ValueError("OPENAI_API_KEY is not configured.")
    client = OpenAI(api_key=key_to_use)
    response = client.chat.completions.create(
        model=model_name or settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def _call_groq(prompt: str, model_name: Optional[str], api_key: Optional[str] = None) -> str:
    key_to_use = api_key or settings.groq_api_key
    if not key_to_use:
        raise ValueError("GROQ_API_KEY is not configured.")
    client = Groq(api_key=key_to_use)
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
    provider_keys: Optional[Dict[str, str]] = None,
) -> str:
    selected_provider = (provider or settings.llm_provider).lower()
    full_prompt = _build_prompt(prompt, mode)
    runtime_keys = provider_keys or {}

    candidate_providers = _provider_sequence(selected_provider, runtime_keys)
    if not candidate_providers:
        raise ValueError("No valid LLM API key found. Configure GROQ_API_KEY or OPENAI_API_KEY.")

    provider_errors = []

    for candidate in candidate_providers:
        try:
            if candidate == "openai":
                return _call_openai(full_prompt, model_name, runtime_keys.get("openai"))
            if candidate == "groq":
                return _call_groq(full_prompt, model_name, runtime_keys.get("groq"))
        except RateLimitError as exc:
            logger.exception("OpenAI quota/rate-limit error: %s", exc)
            provider_errors.append(f"{candidate}: rate-limit or quota")
            continue
        except Exception as exc:
            logger.exception("LLM generation failed for provider %s: %s", candidate, exc)
            provider_errors.append(f"{candidate}: {exc}")
            continue

    summarized = "; ".join(provider_errors[:2])
    raise RuntimeError(f"All LLM providers failed. {summarized}")
