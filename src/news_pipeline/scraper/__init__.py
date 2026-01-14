"""News scraper module - fetches articles from NewsData.io and publishes to Pub/Sub."""

from .core import scrape_and_publish

__all__ = ["scrape_and_publish"]
