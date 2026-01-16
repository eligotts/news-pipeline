#!/usr/bin/env python3
"""
Backfill topic_search_term for existing topics.

Selects the most important topics (by cluster linkage count) and generates
search terms via LLM for keyword aggregation.

Usage:
    uv run python scripts/backfill_topic_search_terms.py --limit 500
    uv run python scripts/backfill_topic_search_terms.py --limit 100 --dry-run
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Backfill topic search terms")
    parser.add_argument("--limit", type=int, default=500, help="Max topics to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't commit changes")
    parser.add_argument("--batch-size", type=int, default=50, help="Commit every N topics")
    args = parser.parse_args()

    load_dotenv()

    from news_pipeline.ingestor.core import Database
    from news_pipeline.ingestor.config import get_settings
    from news_pipeline.ingestor.clustering import TopicOrchestrator

    settings = get_settings()
    db = Database(settings.supabase_dsn)
    db.connect()

    try:
        # Find topics that:
        # 1. Don't have search terms yet
        # 2. Are non-seed (seed topics are bucket categories)
        # 3. Have the most cluster linkages (most "important")
        # 4. Have descriptions (needed for good search term generation)
        logger.info("Finding topics without search terms...")

        topics = db.query_all("""
            SELECT t.id, t.name, t.description
            FROM topic t
            LEFT JOIN topic_search_term tst ON tst.topic_id = t.id
            WHERE tst.id IS NULL
              AND t.source != 'seed'
              AND t.description IS NOT NULL
              AND t.description != ''
            ORDER BY (
                SELECT COUNT(*) FROM cluster_topic ct WHERE ct.topic_id = t.id
            ) DESC
            LIMIT %s
        """, (args.limit,))

        if not topics:
            logger.info("No topics need search terms")
            return

        logger.info(f"Found {len(topics)} topics to process")

        # Show sample of what we'll process
        logger.info("Top 5 topics by importance:")
        for tid, name, desc in topics[:5]:
            cluster_count = db.query_one(
                "SELECT COUNT(*) FROM cluster_topic WHERE topic_id = %s", (tid,)
            )[0]
            logger.info(f"  [{tid}] {name} ({cluster_count} clusters)")

        # Create orchestrator for LLM access
        orchestrator = TopicOrchestrator(db, settings)

        processed = 0
        total_terms = 0

        for i, (topic_id, name, description) in enumerate(topics):
            try:
                terms_count = orchestrator.generate_topic_search_terms(
                    db, topic_id, name, description
                )
                total_terms += terms_count
                processed += 1

                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{len(topics)} topics, {total_terms} terms generated")

                # Commit in batches
                if processed % args.batch_size == 0:
                    if not args.dry_run:
                        db.commit()
                        logger.info(f"Committed batch at {processed} topics")
                    else:
                        db.rollback()
                        logger.info(f"[DRY RUN] Would commit at {processed} topics")

            except Exception as exc:
                logger.warning(f"Failed to generate terms for topic {topic_id} ({name}): {exc}")
                continue

        # Final commit
        if not args.dry_run:
            db.commit()
            logger.info(f"Final commit complete")
        else:
            db.rollback()
            logger.info(f"[DRY RUN] Would commit final batch")

        logger.info(f"Backfill complete: {processed} topics, {total_terms} search terms generated")

        # Show sample results
        sample = db.query_all("""
            SELECT t.name, array_agg(tst.term ORDER BY tst.weight DESC)
            FROM topic t
            JOIN topic_search_term tst ON tst.topic_id = t.id
            GROUP BY t.id, t.name
            ORDER BY t.id DESC
            LIMIT 5
        """)
        if sample:
            logger.info("Sample results:")
            for name, terms in sample:
                logger.info(f"  {name}: {terms[:5]}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
