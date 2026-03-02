#!/usr/bin/env python3
"""
Populate fallback articles cache using OpenAI web search + Gemini + Recommendation engine.

Pipeline:
1. Hit OpenAI Responses API with web_search_preview to get current trending news
2. Send results to OpenRouter Gemini 3 Flash to craft a detailed prompt
3. Use that prompt with the recommendation API to get articles
4. Populate the fallback_articles table

Usage:
    python scripts/populate_fallback.py
    python scripts/populate_fallback.py --dry-run
"""

import argparse
import json
import logging
import os
import sys
import uuid
from typing import List, Optional

import requests
from openai import OpenAI

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


def fetch_current_news(client: OpenAI) -> str:
    """
    Use OpenAI Responses API with web_search_preview to get current trending news.

    Returns the text response with current news summaries grounded in live web results.
    """
    logger.info("Fetching current news via OpenAI web search...")

    response = client.responses.create(
        model="gpt-4o",
        tools=[{"type": "web_search_preview", "search_context_size": "high"}],
        input=(
            "What are the top 15-20 most important news stories happening "
            "right now across all categories? For each story, include:\n"
            "- The specific event or development\n"
            "- Key people, organizations, and countries involved\n"
            "- Why it matters\n"
            "- Any related ongoing developments\n\n"
            "Cover: politics, international affairs, economy/markets, "
            "technology, science, health, conflict/security, climate/environment, "
            "and any major breaking stories. Be specific with names, places, "
            "and details - not generic summaries. Focus on what is NEW and "
            "BREAKING in the last 24-48 hours."
        ),
    )

    content = response.output_text
    logger.info(f"OpenAI web search returned {len(content)} chars of news context")
    return content


def craft_prompt_with_gemini(
    news_context: str,
    client: OpenAI,
    model: str,
) -> dict:
    """
    Send current news context to Gemini via OpenRouter to craft a detailed
    recommendation prompt and extract topic names.

    Returns dict with 'prompt' and 'topics' keys.
    """
    logger.info("Crafting recommendation prompt via Gemini...")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search query specialist for a news recommendation engine. "
                    "Your job is to take current news context and produce TWO things:\n\n"
                    "1. A detailed natural language PROMPT that will be used for "
                    "full-text search against a news article database. This prompt "
                    "should be keyword-rich, mentioning specific people, organizations, "
                    "countries, events, policies, and developments. It should read like "
                    "a comprehensive search query that captures ALL the major stories. "
                    "Include entity names, place names, policy names, and specific "
                    "details that would match article text. EMPHASIZE recent and "
                    "breaking developments.\n\n"
                    "2. A list of 8-12 TOPIC labels that categorize the major themes. "
                    "These should be broad but specific enough to match topic names "
                    "like 'Ukraine Conflict', 'US Economy', 'AI Technology', "
                    "'Middle East Crisis', 'Climate Policy', etc.\n\n"
                    "Return ONLY valid JSON with this exact structure:\n"
                    '{"prompt": "your detailed search prompt here", '
                    '"topics": ["Topic 1", "Topic 2", ...]}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is a summary of current trending news. Craft a detailed "
                    f"recommendation prompt and topic list from this:\n\n"
                    f"{news_context}"
                ),
            },
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    parsed = json.loads(content)

    prompt = parsed.get("prompt", "")
    topics = parsed.get("topics", [])

    logger.info(f"Gemini crafted prompt ({len(prompt)} chars) with {len(topics)} topics")
    logger.info(f"Topics: {topics}")

    return {"prompt": prompt, "topics": topics}


def call_recommendation_api(
    prompt: str,
    topics: List[str],
    api_url: str,
    api_key: Optional[str] = None,
) -> List[dict]:
    """Call the recommendation API with the crafted prompt and return articles."""

    payload = {
        "prompt": prompt,
        "topics": topics,
        "center_x": 0.0,
        "center_y": 0.0,
        "radius": 2.0,
        "recency_hours": 168,
        "neighbor_top_n": 5,
        "traversal_limit": 50,
        "cluster_limit": 3,
        "dense_limit": 20,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    logger.info(f"Calling recommendation API at {api_url}/v1/recommendations")
    logger.info(f"Prompt preview: {prompt[:200]}...")

    response = requests.post(
        f"{api_url}/v1/recommendations",
        json=payload,
        headers=headers,
        timeout=60,
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

    request_id = str(uuid.uuid4())

    logger.info("Clearing existing fallback articles...")
    db.execute("TRUNCATE TABLE public.fallback_articles")
    db.commit()

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
            0.0,   # position_x
            0.0,   # position_y
            2.0,   # radius
            168,   # recency_hours
            5,     # neighbor_top_n
            50,    # traversal_limit
            3,     # cluster_limit
            20,    # dense_limit
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
    parser = argparse.ArgumentParser(description="Populate fallback articles via OpenAI web search + Gemini pipeline")
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.getenv("RECOMMENDER_API_URL", "https://praxis-backend-production.up.railway.app"),
        help="Recommendation API URL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run web search + Gemini steps but don't call recommender or update DB",
    )
    args = parser.parse_args()

    # Validate required API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.error("OPENAI_API_KEY not set in .env")
        return 1

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        logger.error("OPENROUTER_API_KEY not set in .env")
        return 1

    openrouter_model = os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview")
    recommender_api_key = os.getenv("RECOMMENDER_API_KEY")

    # Create clients
    openai_client = OpenAI(api_key=openai_key)
    openrouter_client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")

    # Step 1: Get current news via OpenAI web search
    current_news = fetch_current_news(openai_client)
    logger.info(f"=== Current News ===\n{current_news[:500]}...\n")

    # Step 2: Craft detailed prompt via Gemini
    result = craft_prompt_with_gemini(current_news, openrouter_client, openrouter_model)
    prompt = result["prompt"]
    topics = result["topics"]

    logger.info(f"=== Crafted Prompt ===\n{prompt}\n")
    logger.info(f"=== Topics ===\n{topics}\n")

    if args.dry_run:
        logger.info("Dry run - skipping recommender API call and database update")
        return 0

    # Step 3: Call recommendation API with crafted prompt
    articles = call_recommendation_api(prompt, topics, args.api_url, recommender_api_key)

    if not articles:
        logger.warning("No articles returned from recommendation API")
        return 1

    # Step 4: Populate fallback table
    logger.info("Connecting to database...")
    db = get_db()

    try:
        populate_fallback_table(db, articles, topics, prompt)
        logger.info("Done!")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        db.rollback()
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
