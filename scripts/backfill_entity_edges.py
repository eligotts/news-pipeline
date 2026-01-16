#!/usr/bin/env python3
"""
Backfill entity_edge table from existing article_entity data.

Strategy: Focus on high-salience entity pairs from articles with 2-10 entities.
This captures meaningful relationships without exploding combinatorially.

Usage:
    uv run python scripts/backfill_entity_edges.py --limit 5000
    uv run python scripts/backfill_entity_edges.py --limit 1000 --dry-run
"""

import argparse
import logging
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Backfill entity edges")
    parser.add_argument("--limit", type=int, default=5000, help="Max articles to process")
    parser.add_argument("--min-salience", type=float, default=0.3, help="Min entity salience to include")
    parser.add_argument("--max-entities", type=int, default=10, help="Skip articles with more entities (too noisy)")
    parser.add_argument("--batch-size", type=int, default=500, help="Commit every N articles")
    parser.add_argument("--dry-run", action="store_true", help="Don't commit changes")
    args = parser.parse_args()

    load_dotenv()

    from news_pipeline.ingestor.core import Database
    from news_pipeline.ingestor.config import get_settings

    settings = get_settings()
    db = Database(settings.supabase_dsn)
    db.connect()

    try:
        # Check current state
        current_edges = db.query_one("SELECT COUNT(*) FROM entity_edge")[0]
        logger.info(f"Current entity_edge count: {current_edges}")

        # Find articles with 2+ high-salience entities
        # Order by recency so we prioritize recent content
        logger.info(f"Finding articles with 2-{args.max_entities} entities (salience >= {args.min_salience})...")

        articles = db.query_all("""
            SELECT article_id, COUNT(*) as entity_count
            FROM article_entity
            WHERE salience >= %s
            GROUP BY article_id
            HAVING COUNT(*) BETWEEN 2 AND %s
            ORDER BY article_id DESC
            LIMIT %s
        """, (args.min_salience, args.max_entities, args.limit))

        if not articles:
            logger.info("No articles found matching criteria")
            return

        logger.info(f"Found {len(articles)} articles to process")

        processed = 0
        edges_created = 0
        edges_updated = 0

        for article_id, entity_count in articles:
            # Get entities for this article
            entities = db.query_all("""
                SELECT entity_id, salience
                FROM article_entity
                WHERE article_id = %s AND salience >= %s
                ORDER BY salience DESC
            """, (article_id, args.min_salience))

            if len(entities) < 2:
                continue

            # Create edges for all pairs
            for (e1_id, e1_sal), (e2_id, e2_sal) in combinations(entities, 2):
                # Ensure consistent ordering (smaller ID first)
                src_id, dst_id = min(e1_id, e2_id), max(e1_id, e2_id)
                weight = float(e1_sal) * float(e2_sal)

                # Upsert edge
                result = db.execute("""
                    INSERT INTO entity_edge (src_entity_id, dst_entity_id, relationship, weight, evidence_count, last_article_id)
                    VALUES (%s, %s, 'cooccurrence', %s, 1, %s)
                    ON CONFLICT (src_entity_id, dst_entity_id) DO UPDATE SET
                        weight = entity_edge.weight + EXCLUDED.weight * 0.1,
                        evidence_count = entity_edge.evidence_count + 1,
                        last_article_id = GREATEST(entity_edge.last_article_id, EXCLUDED.last_article_id),
                        updated_at = now()
                    RETURNING (xmax = 0) as inserted
                """, (src_id, dst_id, weight, article_id))

                # Check if it was insert vs update
                row = db.query_one("SELECT 1")  # Dummy to consume result
                edges_created += 1  # Simplified - count all operations

            processed += 1

            if processed % 100 == 0:
                logger.info(f"Processed {processed}/{len(articles)} articles, {edges_created} edge operations")

            # Commit in batches
            if processed % args.batch_size == 0:
                if not args.dry_run:
                    db.commit()
                    logger.info(f"Committed batch at {processed} articles")
                else:
                    db.rollback()
                    logger.info(f"[DRY RUN] Would commit at {processed} articles")

        # Final commit
        if not args.dry_run:
            db.commit()
            logger.info("Final commit complete")
        else:
            db.rollback()
            logger.info("[DRY RUN] Would commit final batch")

        # Show results
        new_edge_count = db.query_one("SELECT COUNT(*) FROM entity_edge")[0]
        logger.info(f"Backfill complete: {processed} articles processed")
        logger.info(f"Entity edges: {current_edges} -> {new_edge_count} (+{new_edge_count - current_edges})")

        # Show top entities by edge count
        top_entities = db.query_all("""
            WITH edge_counts AS (
                SELECT entity_id, SUM(cnt) as total
                FROM (
                    SELECT src_entity_id as entity_id, COUNT(*) as cnt FROM entity_edge GROUP BY 1
                    UNION ALL
                    SELECT dst_entity_id as entity_id, COUNT(*) as cnt FROM entity_edge GROUP BY 1
                ) t
                GROUP BY entity_id
            )
            SELECT e.name, ec.total
            FROM edge_counts ec
            JOIN entity e ON e.id = ec.entity_id
            ORDER BY ec.total DESC
            LIMIT 10
        """)

        if top_entities:
            logger.info("Top entities by edge count:")
            for name, count in top_entities:
                logger.info(f"  {name}: {count} edges")

    finally:
        db.close()


if __name__ == "__main__":
    main()
