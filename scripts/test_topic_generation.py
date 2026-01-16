#!/usr/bin/env python3
"""
Test script for topic generation pipeline.

This script runs just the topic_jobs step on a few clusters to verify
that topic generation, search term creation, and article-topic assignment
are working correctly.

Usage:
    uv run python scripts/test_topic_generation.py --clusters 3
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test topic generation pipeline")
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters to process")
    parser.add_argument("--verbose", action="store_true", help="Extra verbose output")
    args = parser.parse_args()

    load_dotenv()

    from news_pipeline.ingestor.core import Database
    from news_pipeline.ingestor.config import get_settings
    from news_pipeline.ingestor.clustering import TopicOrchestrator

    settings = get_settings()
    db = Database(settings.supabase_dsn)
    db.connect()

    try:
        # Show current state
        print("\n" + "="*60)
        print("BEFORE: Current database state")
        print("="*60)

        topic_count = db.query_one("SELECT COUNT(*) FROM topic")[0]
        search_term_count = db.query_one("SELECT COUNT(*) FROM topic_search_term")[0]
        article_topic_count = db.query_one("SELECT COUNT(*) FROM article_topic")[0]

        print(f"  Topics: {topic_count}")
        print(f"  Topic search terms: {search_term_count}")
        print(f"  Article-topic links: {article_topic_count}")

        # Show recent topics
        print("\n  Recent topics (last 5):")
        recent_topics = db.query_all("""
            SELECT id, name, status, source,
                   (SELECT COUNT(*) FROM topic_search_term WHERE topic_id = t.id) as term_count
            FROM topic t
            ORDER BY id DESC
            LIMIT 5
        """)
        for tid, name, status, source, term_count in recent_topics:
            print(f"    [{tid}] {name[:50]} ({status}, {source}, {term_count} terms)")

        # Find clusters to process
        print("\n" + "="*60)
        print(f"Finding {args.clusters} clusters without topics...")
        print("="*60)

        clusters = db.query_all("""
            SELECT c.id, c.summary, c.size, c.ts_end
            FROM public.cluster c
            LEFT JOIN public.cluster_topic ct ON ct.cluster_id = c.id
            WHERE c.ts_end IS NOT NULL
              AND c.ts_end >= now() - interval '72 hours'
              AND ct.cluster_id IS NULL
              AND c.centroid_vec IS NOT NULL
              AND c.size > 1
            ORDER BY c.size DESC
            LIMIT %s
        """, (args.clusters,))

        if not clusters:
            print("No clusters found needing topics. Try running cluster_jobs first.")
            return

        print(f"Found {len(clusters)} clusters to process:")
        for cid, summary, size, ts_end in clusters:
            summary_preview = (summary or "No summary")[:80]
            print(f"  Cluster {cid} (size={size}): {summary_preview}...")

        # Run topic orchestrator
        print("\n" + "="*60)
        print("Running topic discovery and assignment...")
        print("="*60)

        orchestrator = TopicOrchestrator(db, settings)
        inserted, processed = orchestrator.discover_and_assign_topics(
            lookback_hours=72,
            limit=args.clusters,
            max_workers=1  # Single worker for easier debugging
        )
        db.commit()

        print(f"\nResults: {inserted} new topics created, {processed} clusters processed")

        # Show after state
        print("\n" + "="*60)
        print("AFTER: Updated database state")
        print("="*60)

        new_topic_count = db.query_one("SELECT COUNT(*) FROM topic")[0]
        new_search_term_count = db.query_one("SELECT COUNT(*) FROM topic_search_term")[0]
        new_article_topic_count = db.query_one("SELECT COUNT(*) FROM article_topic")[0]

        print(f"  Topics: {topic_count} -> {new_topic_count} (+{new_topic_count - topic_count})")
        print(f"  Topic search terms: {search_term_count} -> {new_search_term_count} (+{new_search_term_count - search_term_count})")
        print(f"  Article-topic links: {article_topic_count} -> {new_article_topic_count} (+{new_article_topic_count - article_topic_count})")

        # Show new topics with their search terms
        if inserted > 0:
            print("\n  New topics created:")
            new_topics = db.query_all("""
                SELECT t.id, t.name, t.description
                FROM topic t
                ORDER BY t.id DESC
                LIMIT %s
            """, (inserted,))

            for tid, name, description in new_topics:
                print(f"\n    [{tid}] {name}")
                if description:
                    print(f"        Description: {description[:80]}...")

                # Show search terms for this topic
                terms = db.query_all("""
                    SELECT term FROM topic_search_term
                    WHERE topic_id = %s
                    ORDER BY weight DESC
                    LIMIT 10
                """, (tid,))
                if terms:
                    term_list = [t[0] for t in terms]
                    print(f"        Search terms: {', '.join(term_list)}")

        # Show cluster-topic assignments
        print("\n  Cluster-topic assignments:")
        for cid, _, _, _ in clusters:
            assignments = db.query_all("""
                SELECT t.name, ct.weight
                FROM cluster_topic ct
                JOIN topic t ON t.id = ct.topic_id
                WHERE ct.cluster_id = %s
                ORDER BY ct.weight DESC
            """, (cid,))
            if assignments:
                print(f"\n    Cluster {cid}:")
                for name, weight in assignments:
                    print(f"      - {name} (weight={weight:.2f})")

        print("\n" + "="*60)
        print("Topic generation test complete!")
        print("="*60)

    finally:
        db.close()


if __name__ == "__main__":
    main()
