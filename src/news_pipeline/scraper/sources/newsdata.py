import aiohttp
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class NewsDataClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsdata.io/api/1"
        # Categories we want to exclude (entertainment, sports, etc.)
        self.excluded_categories = ["entertainment", "sports", "food", "tourism"]
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

    async def fetch_recent_articles(self, hours_ago: int = 3) -> List[Dict[str, Any]]:
        """
        Fetch recent articles from all configured sources.
        """
        # Log the time range being scraped
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_ago)
        logger.info(
            f"\nScraping articles from {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        all_articles = []
        source_counts = {}

        for source_name, source_id in self.sources.items():
            try:
                articles = await self._fetch_source_articles(source_id)
                for article in articles:
                    article["source_name"] = source_name
                    article["source"] = source_id
                all_articles.extend(articles)
                source_counts[source_name] = len(articles)
            except Exception as e:
                logger.error(f"Error fetching articles from {source_name}: {e}")
                source_counts[source_name] = 0
                continue

        # Print article counts per source
        logger.info("\nArticle Counts by Source:")
        for source, count in source_counts.items():
            logger.info(f"{source}: {count} articles")
        logger.info(f"Total articles: {len(all_articles)}")

        return all_articles

    async def _fetch_source_articles(self, source_id: str) -> List[Dict[str, Any]]:
        """
        Fetch articles from a specific source with pagination support.
        """
        all_articles = []
        next_page = None

        while True:
            url = f"{self.base_url}/latest"
            params = {
                "apikey": self.api_key,
                "domain": source_id,
                "language": "en",
                "excludecategory": ",".join(self.excluded_categories),
                "timeframe": "24",
            }

            # Add next page token if available
            if next_page:
                params["page"] = next_page

            # Log the complete URL being fetched
            full_url = f"{url}?apikey={self.api_key}&domain={source_id}&language=en&excludecategory={params['excludecategory']}&timeframe=20"
            if next_page:
                full_url += f"&page={next_page}"
            logger.info(f"Fetching from URL: {full_url}")

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
                            # For AP News, use first 500 chars of content if description is missing
                            description = article.get("description", "")
                            if not description and source_id == "apnews":
                                content = article.get("content", "")
                                description = content[:500] + "..." if len(content) > 500 else content

                            # Handle author field - use source name as default if no creator
                            creator = article.get("creator")
                            if creator:
                                # Convert list of authors to comma-separated string, or use single author
                                author = ", ".join(creator) if isinstance(creator, list) else creator
                            else:
                                author = article.get(
                                    "source_name", ""
                                )  # Use source name (e.g. "Associated Press") as default author

                            # Ensure content is never empty by using title + description as fallback
                            title = article.get("title", "")
                            content = article.get("content", "")
                            if not content:
                                content = f"{title}\n\n{description}" if description else title

                            filtered_articles.append(
                                {
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
                            )

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
