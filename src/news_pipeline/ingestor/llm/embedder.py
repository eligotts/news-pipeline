from __future__ import annotations

import logging
from typing import List
from openai import OpenAI

from ..core import retry_with_backoff, CircuitBreaker

logger = logging.getLogger(__name__)

# Circuit breaker for OpenAI embedding API
_embedding_circuit_breaker = CircuitBreaker(
    name="openai_embeddings",
    failure_threshold=5,
    recovery_timeout=60.0,
)


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0)
    def embed(self, texts: List[str], dimensions: int | None = None) -> List[List[float]]:
        # Check circuit breaker before making request
        if not _embedding_circuit_breaker.allow_request():
            raise RuntimeError("OpenAI embedding API circuit breaker is open")

        try:
            resp = self._client.embeddings.create(
                input=texts,
                model=self._model,
                dimensions=dimensions,
            )
            _embedding_circuit_breaker.record_success()
            return [d.embedding for d in resp.data]
        except Exception as exc:
            _embedding_circuit_breaker.record_failure()
            raise
