from __future__ import annotations

from typing import List
from openai import OpenAI


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: List[str], dimensions: int | None = None) -> List[List[float]]:
        resp = self._client.embeddings.create(
            input=texts,
            model=self._model,
            dimensions=dimensions,
        )
        return [d.embedding for d in resp.data]
