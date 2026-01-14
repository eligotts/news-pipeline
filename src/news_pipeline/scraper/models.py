from pydantic import BaseModel
from typing import Optional


class Article(BaseModel):
    """
    Pydantic model for article data.
    This schema is used by the LLM extraction strategy.
    """

    url: str
    title: str
    content: str
    description: Optional[str] = ""
    author: Optional[str] = ""
    published_at: str
    source: str
    imageUrl: Optional[str] = ""
    imageAlt: Optional[str] = ""
