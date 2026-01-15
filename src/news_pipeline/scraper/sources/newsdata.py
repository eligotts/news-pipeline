import asyncio
import logging
from datetime import datetime, timedelta

import aiohttp

logger = logging.getLogger(__name__)


class NewsDataClient:
    """Client for fetching articles from NewsData.io API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsdata.io/api/1"
        self.excluded_categories = ["entertainment", "sports", "food", "tourism"]
        self.session: aiohttp.ClientSession | None = None
        self.sources = {
            "New York Times": "nytimes",
            "The Huffington Post": "huffpost",
            "The Guardian": "theguardian",
            "New Yorker": "newyorker",
            "Al Jazeera": "aljazeera_us",
            "BBC": "bbc",
            "MSNBC": "msnbc",
            "Wall Street Journal": "wsj",
            "Politico": "politico",
            "The Washington Post": "washingtonpost",
            "The Hill": "thehill",
            "Time": "time",
            "NPR": "npr",
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "session"):
            await self.session.close()

    async def fetch_recent_articles(self, hours_ago: int = 24) -> list[dict]:
        """Fetch recent articles from all configured sources with limited concurrency."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_ago)
        logger.info(
            f"Scraping articles from {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"to {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent API requests

        async def fetch_source(source_name: str, source_id: str) -> tuple[str, list[dict]]:
            async with semaphore:
                try:
                    articles = await self._fetch_source_articles(source_id, hours_ago)
                    for article in articles:
                        article["source_name"] = source_name
                        article["source"] = source_id
                    return source_name, articles
                except Exception as e:
                    logger.error(f"Error fetching articles from {source_name}: {e}")
                    return source_name, []

        results = await asyncio.gather(
            *[fetch_source(name, sid) for name, sid in self.sources.items()]
        )

        all_articles = []
        source_counts = {}
        for source_name, articles in results:
            all_articles.extend(articles)
            source_counts[source_name] = len(articles)

        logger.info("Article counts by source:")
        for source, count in source_counts.items():
            logger.info(f"  {source}: {count}")
        logger.info(f"Total: {len(all_articles)}")

        return all_articles

    def _parse_article(self, article: dict) -> dict:
        """Parse raw API article into normalized format."""
        title = article.get("title", "")
        description = article.get("description", "")
        content = article.get("content", "")

        # Ensure content is never empty
        if not content:
            content = f"{title}\n\n{description}" if description else title

        # Handle author field - convert list to string or use empty string
        creator = article.get("creator")
        if creator:
            author = ", ".join(creator) if isinstance(creator, list) else creator
        else:
            author = ""

        return {
            "title": title,
            "description": description,
            "url": article.get("link", ""),
            "published_at": article.get("pubDate", ""),
            "source_name": article.get("source_name", ""),
            "source": article.get("source_name", ""),
            "category": article.get("category", []),
            "keywords": article.get("keywords", ""),
            "image_url": article.get("image_url", ""),
            "content": content,
            "author": author,
        }

    async def _fetch_source_articles(self, source_id: str, hours_ago: int = 24) -> list[dict]:
        """Fetch articles from a specific source with pagination support."""
        all_articles = []
        next_page = None

        while True:
            url = f"{self.base_url}/latest"
            params = {
                "apikey": self.api_key,
                "domain": source_id,
                "language": "en",
                "excludecategory": ",".join(self.excluded_categories),
                "timeframe": str(hours_ago),
            }

            if next_page:
                params["page"] = next_page

            # Log the URL being fetched (without API key for security)
            logger.info(f"Fetching from {source_id}, page={next_page or 'initial'}")

            try:
                async with self.session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Error fetching from {source_id}: {response.status}")
                        break

                    data = await response.json()

                    if data.get("status") != "success":
                        logger.error(f"API error for {source_id}: {data.get('message', 'Unknown error')}")
                        break

                    articles = data.get("results", [])
                    filtered_articles = []

                    for article in articles:
                        try:
                            filtered_articles.append(self._parse_article(article))
                        except Exception as e:
                            logger.warning(f"Error processing article from {source_id}: {e}")
                            continue

                    all_articles.extend(filtered_articles)

                    # Check for next page
                    next_page = data.get("nextPage")
                    if not next_page:
                        break

                    # Add a small delay between requests to avoid rate limiting
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error fetching articles from {source_id}: {e}")
                break

        return all_articles
