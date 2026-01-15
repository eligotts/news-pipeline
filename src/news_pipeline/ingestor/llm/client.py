from __future__ import annotations

"""OpenRouter client wrapper.

Provides an OpenAI-compatible client that uses OpenRouter's API for LLM calls.
Uses the same OpenAI SDK but points to OpenRouter's base URL.
"""

from typing import Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def create_openrouter_client(
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
) -> Optional["OpenAI"]:
    """Create an OpenAI-compatible client pointing to OpenRouter.

    Args:
        api_key: OpenRouter API key
        base_url: OpenRouter API base URL (defaults to production)

    Returns:
        OpenAI client configured for OpenRouter, or None if SDK not available
    """
    if OpenAI is None:
        return None
    return OpenAI(api_key=api_key, base_url=base_url)
