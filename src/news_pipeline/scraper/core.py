"""Core scraper logic - fetches articles and publishes to Pub/Sub."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from google.cloud import pubsub_v1

from .sources.newsdata import NewsDataClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration from environment
MAX_HOURS_AGO = int(os.getenv("MAX_HOURS_AGO", "24"))
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# GCP settings
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "praxis-db")
TOPIC_NAME = os.getenv("PUBSUB_TOPIC", "news_scraper_topic")

# API key from environment
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")


def _get_publisher():
    """Lazily initialize Pub/Sub publisher."""
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)
    return publisher, topic_path


async def publish_article(
    article: Dict[str, Any],
    publisher: pubsub_v1.PublisherClient,
    topic_path: str,
) -> bool:
    """Publish a single article to Pub/Sub with retry logic."""
    try:
        # Set the source field from source_name before encoding
        article_data = article.copy()
        article_data["source"] = article_data.get("source_name", article_data.get("source", ""))

        # Convert to JSON string and encode
        data = json.dumps(article_data, ensure_ascii=False).encode("utf-8")

        # Publish with retry logic
        for attempt in range(MAX_RETRIES):
            try:
                future = publisher.publish(topic_path, data)
                future.result(timeout=30)  # Wait for confirmation
                logger.info(f"Published article: {article['title'][:60]} from {article.get('source', 'unknown')}")
                return True
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed to publish article after {MAX_RETRIES} attempts: {e}")
                    return False
                await asyncio.sleep(RETRY_DELAY)

    except Exception as e:
        logger.error(f"Error publishing article: {e}")
        return False


async def scrape_news(hours_ago: int | None = None) -> tuple[int, int]:
    """
    Main scraping function that coordinates scraping from NewsData.io.

    Args:
        hours_ago: How many hours back to scrape (default: MAX_HOURS_AGO from env)

    Returns:
        Tuple of (total_articles, published_articles)
    """
    if not NEWSDATA_API_KEY:
        raise ValueError("NEWSDATA_API_KEY environment variable is not set")

    hours = hours_ago or MAX_HOURS_AGO
    total_articles = 0
    published_articles = 0

    publisher, topic_path = _get_publisher()

    try:
        async with NewsDataClient(NEWSDATA_API_KEY) as client:
            articles = await client.fetch_recent_articles(hours_ago=hours)
            total_articles = len(articles)

            # Publish all articles
            for article in articles:
                if await publish_article(article, publisher, topic_path):
                    published_articles += 1

            # Log results
            logger.info(f"Scraping complete. Processed {total_articles} articles, published {published_articles}.")

    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        raise

    return total_articles, published_articles


async def scrape_and_publish(hours_ago: int | None = None) -> Dict[str, int]:
    """
    High-level entry point for the scraper.

    Args:
        hours_ago: How many hours back to scrape (default: MAX_HOURS_AGO from env)

    Returns:
        Dict with 'total' and 'published' counts
    """
    total, published = await scrape_news(hours_ago)
    return {"total": total, "published": published}


# CLI entry point
if __name__ == "__main__":
    try:
        result = asyncio.run(scrape_and_publish())
        print(f"Scraper completed: {result}")
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
