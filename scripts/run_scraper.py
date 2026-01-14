#!/usr/bin/env python
"""Run the news scraper to fetch articles and publish to Pub/Sub."""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env before importing anything else
from dotenv import load_dotenv
load_dotenv()

from news_pipeline.scraper.core import scrape_and_publish


async def main():
    """Run the scraper."""
    print("[Scraper] Starting news scraper...")
    result = await scrape_and_publish()
    print(f"[Scraper] Complete: {result['total']} articles fetched, {result['published']} published to Pub/Sub")
    return result


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Scraper] Interrupted by user")
    except Exception as e:
        print(f"[Scraper] Error: {e}")
        raise
