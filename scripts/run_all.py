#!/usr/bin/env python
"""Run full pipeline: scrape -> ingest (steps 1-5).

This is the main entry point for automated/cron execution.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env before importing anything else
from dotenv import load_dotenv
load_dotenv()


async def main():
    """Run the complete news pipeline."""
    parser = argparse.ArgumentParser(description="Run full news pipeline")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--bypass-stance-llm", action="store_true", help="Bypass LLM for stance jobs")
    args = parser.parse_args()

    from news_pipeline.scraper.core import scrape_and_publish
    from news_pipeline.ingestor.run_ingestion import main as run_ingestion

    print("=" * 60)
    print("NEWS PIPELINE - Full Run")
    print("=" * 60)

    # Step 1: Scrape news and publish to Pub/Sub
    print("\n[Step 1/2] Running scraper...")
    print("-" * 40)
    scraper_result = await scrape_and_publish()
    print(f"[Scraper] Complete: {scraper_result['total']} articles, {scraper_result['published']} published")

    # Step 2: Run ingestion pipeline (drain_pubsub -> topic_jobs)
    print("\n[Step 2/2] Running ingestor pipeline...")
    print("-" * 40)

    # Set up args for ingestion
    sys.argv = [
        "run_all.py",
        "--start-step", "drain_pubsub",
        "--end-step", "topic_jobs",
    ]
    if args.verbose:
        sys.argv.append("--verbose")
    if args.bypass_stance_llm:
        sys.argv.append("--bypass-stance-llm")

    run_ingestion()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted by user")
    except Exception as e:
        print(f"[Pipeline] Error: {e}")
        raise
