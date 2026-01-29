#!/usr/bin/env python
"""Run the complete news pipeline: scraper -> ingestor.

This is the entrypoint for scheduled Cloud Run Jobs.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add scripts dir so sibling scripts can be imported
sys.path.insert(0, str(Path(__file__).parent))

# Load .env for local development (no-op in production)
from dotenv import load_dotenv
load_dotenv()


async def run_scraper():
    """Run the scraper to fetch articles and publish to Pub/Sub."""
    from news_pipeline.scraper.core import scrape_and_publish

    print("[Pipeline] Starting scraper...")
    result = await scrape_and_publish()
    print(f"[Pipeline] Scraper complete: {result['total']} articles fetched, {result['published']} published")
    return result


def run_ingestor():
    """Run the ingestor pipeline."""
    from news_pipeline.ingestor.run import main as ingestor_main

    print("[Pipeline] Starting ingestor...")
    # Reset argv to avoid issues
    sys.argv = ["run_pipeline.py", "--start-step", "drain_pubsub", "--end-step", "topic_jobs"]
    ingestor_main()
    print("[Pipeline] Ingestor complete")


def run_fallback():
    """Populate fallback articles."""
    from populate_fallback import main as fallback_main

    print("[Pipeline] Populating fallback articles...")
    sys.argv = ["run_pipeline.py"]
    fallback_main()
    print("[Pipeline] Fallback complete")


def main():
    print("=" * 60)
    print("[Pipeline] Starting news pipeline")
    print("=" * 60)

    # Step 1: Scraper
    asyncio.run(run_scraper())

    # Step 2: Ingestor
    run_ingestor()

    # Step 3: Fallback articles
    run_fallback()

    print("=" * 60)
    print("[Pipeline] Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
