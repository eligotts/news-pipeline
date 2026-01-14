from __future__ import annotations

import os
from dataclasses import dataclass, field


def _openrouter_model_default() -> str:
    return os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview")


@dataclass(frozen=True)
class Settings:
    google_cloud_project: str = os.getenv("GOOGLE_CLOUD_PROJECT", "praxis-db")
    pubsub_subscription: str = os.getenv("PUBSUB_SUBSCRIPTION", "news_ingestion_sub")

    supabase_dsn: str | None = os.getenv("SUPABASE_DSN")

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

    # OpenRouter settings for LLM calls
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    openrouter_model: str = field(default_factory=_openrouter_model_default)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "10"))
    max_concurrent_batches: int = int(os.getenv("MAX_CONCURRENT_BATCHES", "3"))
    idle_timeout: float = float(os.getenv("IDLE_TIMEOUT", "300"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    redis_url: str | None = os.getenv("REDIS_URL")
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))


def get_settings() -> Settings:
    settings = Settings()
    missing: list[str] = []
    if not settings.supabase_dsn:
        missing.append("SUPABASE_DSN")
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not settings.openrouter_api_key:
        missing.append("OPENROUTER_API_KEY")
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
    return settings
