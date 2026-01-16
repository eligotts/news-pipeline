#!/usr/bin/env python3
"""
Backfill article_topic entries for existing articles.

This script assigns topics directly to articles based on vector similarity,
enabling retrieval of singleton articles through the direct traversal pipeline.

Usage:
    python scripts/backfill_article_topics.py --batch-size 100 --limit 1000
"""

import argparse
import logging
import os
import sys
from typing import List, Tuple

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Topic assignment thresholds (same as in pipeline/features.py)
ARTICLE_TOPIC_SIMILARITY_THRESHOLD = 0.35
ARTICLE_TOPIC_LIMIT = 5


def _vec_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


def get_articles_needing_topics(db, batch_size: int, offset: int = 0) -> List[Tuple[int, str]]:
    """Get articles that have v_title but no article_topic entries."""
    rows = db.query_all(
        """
        SELECT a.id, a.v_title::text
        FROM public.article a
        LEFT JOIN public.article_topic at ON at.article_id = a.id
        WHERE a.v_title IS NOT NULL
          AND at.article_id IS NULL
        ORDER BY a.id
        LIMIT %s OFFSET %s
        """,
        (batch_size, offset),
    )
    return [(int(row[0]), row[1]) for row in rows]


def assign_topics_to_article(db, article_id: int, v_title_str: str) -> int:
    """Assign topics to a single article based on vector similarity."""
    # Find top topics by vector similarity
    rows = db.query_all(
        """
        SELECT t.id, 1 - (t.embedding <=> %s::vector) as similarity
        FROM public.topic t
        WHERE t.embedding IS NOT NULL
          AND t.status IN ('active', 'candidate')
        ORDER BY t.embedding <=> %s::vector ASC
        LIMIT %s
        """,
        (v_title_str, v_title_str, ARTICLE_TOPIC_LIMIT * 2),
    )

    if not rows:
        return 0

    assigned = 0
    for topic_id, similarity in rows:
        if similarity < ARTICLE_TOPIC_SIMILARITY_THRESHOLD:
            continue
        if assigned >= ARTICLE_TOPIC_LIMIT:
            break

        db.execute(
            """
            INSERT INTO public.article_topic (article_id, topic_id, weight, source)
            VALUES (%s, %s, %s, 'backfill')
            ON CONFLICT (article_id, topic_id) DO NOTHING
            """,
            (article_id, int(topic_id), float(similarity)),
        )
        assigned += 1

    return assigned


def main():
    parser = argparse.ArgumentParser(description="Backfill article_topic entries")
    parser.add_argument("--batch-size", type=int, default=100, help="Articles per batch")
    parser.add_argument("--limit", type=int, default=None, help="Max articles to process (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Don't commit changes")
    args = parser.parse_args()

    load_dotenv()

    from news_pipeline.ingestor.core import Database
    from news_pipeline.ingestor.config import get_settings

    settings = get_settings()
    db = Database(settings.supabase_dsn)
    db.connect()

    try:
        # Count articles needing topics
        count_row = db.query_one(
            """
            SELECT COUNT(*)
            FROM public.article a
            LEFT JOIN public.article_topic at ON at.article_id = a.id
            WHERE a.v_title IS NOT NULL
              AND at.article_id IS NULL
            """
        )
        total_needing = count_row[0] if count_row else 0
        logger.info(f"Found {total_needing} articles needing topic assignment")

        if args.limit:
            total_to_process = min(args.limit, total_needing)
        else:
            total_to_process = total_needing

        logger.info(f"Will process {total_to_process} articles")

        processed = 0
        total_topics_assigned = 0
        offset = 0

        while processed < total_to_process:
            batch = get_articles_needing_topics(db, args.batch_size, offset)
            if not batch:
                break

            batch_assigned = 0
            for article_id, v_title_str in batch:
                if processed >= total_to_process:
                    break

                assigned = assign_topics_to_article(db, article_id, v_title_str)
                batch_assigned += assigned
                processed += 1

                if processed % 100 == 0:
                    logger.info(f"Processed {processed}/{total_to_process} articles, {total_topics_assigned + batch_assigned} topics assigned")

            total_topics_assigned += batch_assigned

            if not args.dry_run:
                db.commit()
                logger.info(f"Committed batch: {len(batch)} articles, {batch_assigned} topics")
            else:
                db.rollback()
                logger.info(f"[DRY RUN] Would commit: {len(batch)} articles, {batch_assigned} topics")

            offset += args.batch_size

        logger.info(f"Backfill complete: {processed} articles processed, {total_topics_assigned} topic assignments created")

    finally:
        db.close()


if __name__ == "__main__":
    main()
