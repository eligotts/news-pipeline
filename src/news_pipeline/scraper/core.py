"""Core scraper logic - fetches articles and publishes to Pub/Sub."""

from __future__ import annotations

import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from google.cloud import pubsub_v1

from .sources.newsdata import NewsDataClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_HOURS_AGO = int(os.getenv("MAX_HOURS_AGO", "24"))
MAX_RETRIES = 3
RETRY_DELAY = 5

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "praxis-db")
TOPIC_NAME = os.getenv("PUBSUB_TOPIC", "news_scraper_topic")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")


def _get_publisher() -> tuple[pubsub_v1.PublisherClient, str]:
    """Create Pub/Sub publisher and topic path."""
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)
    return publisher, topic_path


async def publish_article(
    article: dict,
    publisher: pubsub_v1.PublisherClient,
    topic_path: str,
) -> bool:
    """Publish a single article to Pub/Sub with retry logic."""
    try:
        article_data = article.copy()
        article_data["source"] = article_data.get("source_name", article_data.get("source", ""))
        data = json.dumps(article_data, ensure_ascii=False).encode("utf-8")

        for attempt in range(MAX_RETRIES):
            try:
                future = publisher.publish(topic_path, data)
                future.result(timeout=30)
                logger.info(f"Published: {article['title'][:60]} from {article.get('source', 'unknown')}")
                return True
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed to publish after {MAX_RETRIES} attempts: {e}")
                    return False
                await asyncio.sleep(RETRY_DELAY)
    except Exception as e:
        logger.error(f"Error publishing article: {e}")


async def scrape_news(hours_ago: int | None = None) -> tuple[int, int]:
    """Scrape news and publish to Pub/Sub. Returns (total, published) counts."""
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


async def scrape_and_publish(hours_ago: int | None = None) -> dict[str, int]:
    """High-level entry point for the scraper."""
    total, published = await scrape_news(hours_ago)
    return {"total": total, "published": published}


if __name__ == "__main__":
    asyncio.run(scrape_and_publish())
