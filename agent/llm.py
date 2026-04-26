from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

AVAILABLE_MODELS = {
    "haiku":         ("anthropic", "claude-haiku-4-5-20251001"),
    "sonnet":        ("anthropic", "claude-sonnet-4-6"),
    "mistral:7b-q4": ("ollama",   "mistral:7b-instruct-q4_0"),
    "qwen2.5:3b":    ("ollama",   "qwen2.5:3b"),
}

DEFAULT_MODEL = "sonnet"


def build_llm(model_name: str | None = None, temperature: float = 0, max_tokens: int = 1024):
    """Return a LangChain-compatible LLM — Claude API or local Ollama.

    Args:
        model_name: Key from AVAILABLE_MODELS. Falls back to DEFAULT_MODEL.
        temperature: Sampling temperature (0 = deterministic).
        max_tokens: Maximum tokens in the response.
    """
    resolved = model_name or DEFAULT_MODEL
    provider, model_id = AVAILABLE_MODELS.get(resolved, ("anthropic", resolved))

    if provider == "ollama":
        return ChatOllama(
            model=model_id,
            temperature=temperature,
            think=False,
            num_predict=max_tokens,
        )

    return ChatAnthropic(
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
    )
