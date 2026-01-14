from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Union

from pydantic import BaseModel, Field, HttpUrl, validator


class ArticlePayload(BaseModel):
    title: str = Field(..., min_length=1)
    description: Optional[str] = None
    url: HttpUrl
    published_at: Optional[str] = None
    source_name: Optional[str] = None
    source: Optional[str] = None
    category: Optional[List[str]] = None
    keywords: Optional[Union[str, List[str]]] = None
    image_url: Optional[str] = None
    content: Optional[str] = None
    author: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    lang: Optional[str] = None
    region: Optional[str] = None

    @validator("category", pre=True)
    def _coerce_category(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            # allow comma-separated strings
            return [chunk.strip() for chunk in v.split(",") if chunk.strip()]
        return list(v)

    @validator("keywords", pre=True)
    def _coerce_keywords(cls, v):
        if v is None:
            return None
        if isinstance(v, list):
            return ", ".join(str(item) for item in v if item)
        return v

    @validator("published_at")
    def _validate_published_at(cls, v):
        if v is None:
            return None
        # ensure parseable by datetime.fromisoformat after replacing Z
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("published_at must be ISO-8601 compatible") from exc
        return v
