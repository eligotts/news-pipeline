"""News ingestor module - processes articles from Pub/Sub through the full pipeline."""

from .config import get_settings, Settings
from .core import Database
from .pipeline import IngestionProcessor, IngestionResult

__all__ = [
    "get_settings",
    "Settings",
    "Database",
    "IngestionProcessor",
    "IngestionResult",
]
