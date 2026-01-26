#!/usr/bin/env python3
"""
Populate fallback articles cache by querying the recommendation engine.

This script:
1. Finds the most prominent topics from recent large clusters
2. Calls the recommendation API with those topics
3. Caches the results in the fallback_articles table for fast retrieval

Usage:
    python scripts/populate_fallback.py
    python scripts/populate_fallback.py --min-cluster-size 5 --num-clusters 10
"""

import argparse
import json
import logging
import os
import sys
import uuid
from typing import List

import requests

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_db():
    """Create database connection."""
    from news_pipeline.ingestor.core.db import Database

    dsn = os.getenv("SUPABASE_DSN")
    if not dsn:
        raise RuntimeError("SUPABASE_DSN environment variable not set")

    db = Database(dsn)
    db.connect()
    return db


def get_top_cluster_topics(db, min_cluster_size: int = 10, num_clusters: int = 5) -> List[str]:
    """
    Get prominent topics from recent large clusters.

    Since cluster_topic no longer has a weight column, we:
    1. Find the N most recent clusters with size > min_cluster_size
    2. Get all topics linked to those clusters
    3. Rank topics by how many of those clusters they appear in
    4. Return the top topic names
    """
    query = """
        WITH recent_clusters AS (
            SELECT id
            FROM public.cluster
            WHERE size > %s
              AND ts_end >= now() - interval '7 days'
            ORDER BY ts_end DESC
            LIMIT %s
        ),
        topic_counts AS (
            SELECT
                t.id,
                t.name,
                COUNT(DISTINCT ct.cluster_id) as cluster_count
            FROM public.cluster_topic ct
            JOIN recent_clusters rc ON rc.id = ct.cluster_id
            JOIN public.topic t ON t.id = ct.topic_id
            GROUP BY t.id, t.name
        )
        SELECT name
        FROM topic_counts
        ORDER BY cluster_count DESC
        LIMIT %s
    """
    rows = db.query_all(query, (min_cluster_size, num_clusters * 3, num_clusters))
    return [row[0] for row in rows]


def call_recommendation_api(topics: List[str], api_url: str, api_key: str = None) -> List[dict]:
    """Call the recommendation API and return articles."""

    prompt = f"most interesting news about: {', '.join(topics)}"

    payload = {
        "prompt": prompt,
        "topics": topics,
        "center_x": 0.0,
        "center_y": 0.0,
        "radius": 2.0,  # Wide radius to get diverse results
        "recency_hours": 168,  # 7 days
        "neighbor_top_n": 5,
        "traversal_limit": 50,
        "cluster_limit": 3,
        "dense_limit": 20,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    logger.info(f"Calling recommendation API at {api_url}/v1/recommendations")
    logger.info(f"Request topics: {topics}")

    response = requests.post(
        f"{api_url}/v1/recommendations",
        json=payload,
        headers=headers,
        timeout=60,  # Longer timeout for LLM reranking
    )
    response.raise_for_status()

    data = response.json()
    articles = data.get("articles", [])
    logger.info(f"Received {len(articles)} articles from API")

    return articles


def populate_fallback_table(db, articles: List[dict], topics: List[str], prompt: str):
    """Clear and populate the fallback_articles table."""

    if not articles:
        logger.warning("No articles to insert")
        return

    # Generate a request ID for this batch
    request_id = str(uuid.uuid4())

    # Clear existing entries
    logger.info("Clearing existing fallback articles...")
    db.execute("TRUNCATE TABLE public.fallback_articles")
    db.commit()

    # Insert new articles
    logger.info(f"Inserting {len(articles)} fallback articles...")

    insert_query = """
        INSERT INTO public.fallback_articles (
            request_id,
            prompt,
            topics,
            position_x,
            position_y,
            radius,
            recency_hours,
            neighbor_top_n,
            traversal_limit,
            cluster_limit,
            dense_limit,
            article_rank,
            article_id,
            cluster_id,
            title,
            lede,
            url,
            publisher,
            ts_pub,
            x,
            y,
            score,
            reasons,
            sources,
            cluster_summary,
            raw_article
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
    """

    for rank, article in enumerate(articles, 1):
        params = (
            request_id,
            prompt,
            topics,
            0.0,  # position_x
            0.0,  # position_y
            2.0,  # radius
            168,  # recency_hours
            5,    # neighbor_top_n
            50,   # traversal_limit
            3,    # cluster_limit
            20,   # dense_limit
            rank,
            article.get("article_id"),
            article.get("cluster_id"),
            article.get("title"),
            article.get("lede"),
            article.get("url"),
            article.get("publisher"),
            article.get("ts_pub"),
            article.get("x"),
            article.get("y"),
            article.get("score"),
            article.get("reasons", []),
            article.get("sources", []),
            article.get("cluster_summary"),
            json.dumps(article),
        )
        db.execute(insert_query, params)

    db.commit()
    logger.info(f"Successfully inserted {len(articles)} fallback articles")


def main():
    parser = argparse.ArgumentParser(description="Populate fallback articles cache")
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=10,
        help="Minimum cluster size to consider (default: 10)",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=5,
        help="Number of top topics to use (default: 5)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.getenv("RECOMMENDER_API_URL", "http://localhost:8001"),
        help="Recommendation API URL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only fetch topics, don't call API or update database",
    )
    args = parser.parse_args()

    api_key = os.getenv("RECOMMENDER_API_KEY")

    logger.info("Connecting to database...")
    db = get_db()

    try:
        # 1. Get prominent topics from recent clusters
        logger.info(f"Fetching top topics from clusters (min_size={args.min_cluster_size}, num={args.num_clusters})...")
        topics = get_top_cluster_topics(db, args.min_cluster_size, args.num_clusters)

        if not topics:
            logger.warning("No qualifying clusters/topics found")
            return 1

        logger.info(f"Found topics: {topics}")

        if args.dry_run:
            logger.info("Dry run - skipping API call and database update")
            return 0

        # 2. Call recommendation API
        prompt = f"most interesting news about: {', '.join(topics)}"
        articles = call_recommendation_api(topics, args.api_url, api_key)

        if not articles:
            logger.warning("No articles returned from API")
            return 1

        # 3. Populate fallback table
        populate_fallback_table(db, articles, topics, prompt)

        logger.info("Done!")
        return 0

    except requests.RequestException as e:
        logger.error(f"API request failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        db.rollback()
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
