from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import Settings
from .db import Database
from .openai_embed import OpenAIEmbedder

from .openrouter_client import create_openrouter_client

logger = logging.getLogger(__name__)


# JSON Schema for structured output
TOPIC_PROPOSAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "topic_proposal",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "bucket_topics": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "1-2 topic IDs from bucket_topics list for primary categorization"
                },
                "existing_topics": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Topic IDs from similar_existing_topics that match well"
                },
                "topics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Topic name"
                            },
                            "one_liner": {
                                "type": "string",
                                "description": "Brief topic description"
                            },
                            "definition": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Detailed topic definition points"
                            },
                            "key_entities": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Key entities related to the topic"
                            },
                            "weight": {
                                "type": "number",
                                "description": "Topic relevance weight between 0 and 1"
                            }
                        },
                        "required": ["name", "one_liner", "definition", "key_entities", "weight"],
                        "additionalProperties": False
                    },
                    "description": "New topic proposals to fill gaps"
                }
            },
            "required": ["bucket_topics", "existing_topics", "topics"],
            "additionalProperties": False
        }
    }
}

# TODO(spec A.2): add topic promotion workflow and parent assignment heuristics once defined.


class TopicEmbedder:
    def __init__(self, openai_embedder: Optional[OpenAIEmbedder]) -> None:
        if openai_embedder is None:
            raise ValueError("OpenAIEmbedder is required for TopicEmbedder")
        self._openai = openai_embedder

    def embed(self, card: Dict[str, Any]) -> List[float]:
        text_parts = [
            card.get("name", ""),
            card.get("one_liner", ""),
            "; ".join(card.get("definition", [])),
            "Entities: " + ", ".join(card.get("key_entities", [])),
        ]
        text = "\n".join([part for part in text_parts if part]).strip()
        if not text:
            text = card.get("name") or ""

        def _pad(vec: Iterable[float]) -> List[float]:
            values = [float(x) for x in vec]
            if len(values) >= 768:
                return values[:768]
            return values + [0.0] * (768 - len(values))

        try:
            return _pad(self._openai.embed([text], dimensions=768)[0])
        except Exception as exc:  # pragma: no cover - external request failure
            raise RuntimeError("OpenAI topic embedding failed") from exc

    @staticmethod
    def to_literal(vec: Iterable[float]) -> str:
        return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


class TopicOrchestrator:
    """Manages LLM-driven topic catalog discovery and cluster topic assignment."""

    def __init__(self, db, settings: Settings) -> None:
        if not settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY must be configured for topic orchestration")

        self.db = db
        self.settings = settings
        self._openrouter_client = create_openrouter_client(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )

        openai_embedder = OpenAIEmbedder(api_key=settings.openai_api_key, model=settings.openai_embed_model)
        self.embedder = TopicEmbedder(openai_embedder)

    # ------------------------------------------------------------------
    # Unified topic discovery and assignment (LLM-driven)
    # ------------------------------------------------------------------
    def discover_and_assign_topics(self, lookback_hours: int = 72, limit: int = 40, max_workers: int = 3) -> Tuple[int, int]:
        """
        Unified pipeline that discovers new topics and assigns existing topics to clusters in one step.
        Returns tuple of (new_topics_inserted, clusters_processed).
        """
        logger.info("topic_discover_and_assign_start", lookback_hours=lookback_hours, limit=limit)

        clusters = self.db.query_all(
            """
            SELECT c.id, c.summary, c.top_headlines, c.ts_end, c.centroid_vec
            FROM public.cluster c
            LEFT JOIN public.cluster_topic ct ON ct.cluster_id = c.id
            WHERE c.ts_end IS NOT NULL
              AND c.ts_end >= now() - (%s * interval '1 hour')
              AND ct.cluster_id IS NULL
              AND c.centroid_vec IS NOT NULL
              AND c.size > 1
            ORDER BY COALESCE(c.size, 0) DESC
            LIMIT %s
            """,
            (lookback_hours, limit),
        )
        if not clusters:
            print(f"[Topics] No clusters found to process")
            return (0, 0)

        print(f"\n[Topics] Starting topic discovery and assignment for {len(clusters)} clusters")
        
        # Process clusters in parallel
        cluster_data = [
            (int(cluster_id), summary, headlines, centroid_vec)
            for cluster_id, summary, headlines, _, centroid_vec in clusters
        ]
        
        inserted = 0
        processed = 0
        workers = max(1, min(max_workers, len(cluster_data)))
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._process_topic_cluster, cluster_id, summary, headlines, centroid_vec): cluster_id
                for cluster_id, summary, headlines, centroid_vec in cluster_data
            }
            for future in as_completed(futures):
                cluster_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        new_topics_count, success = result
                        inserted += new_topics_count
                        if success:
                            processed += 1
                except Exception as exc:
                    logger.warning("topic_cluster_processing_failed cluster_id=%s error=%s", cluster_id, str(exc))

        print(f"\n[Topics] Processing complete!")
        print(f"  Total clusters processed: {processed}")
        print(f"  New topics created: {inserted}")
        logger.info("topic_discover_and_assign_complete examined=%s inserted=%s processed=%s", len(clusters), inserted, processed)
        return (inserted, processed)

    def _process_topic_cluster(
        self, cluster_id: int, summary: Optional[str], headlines: Optional[List[str]], centroid_vec: Any
    ) -> Optional[Tuple[int, bool]]:
        """Process a single cluster for topic discovery. Creates its own DB connection for thread safety."""
        # Get DSN from the original database connection
        dsn = self.db._dsn if hasattr(self.db, '_dsn') else self.settings.supabase_dsn
        if not dsn:
            raise RuntimeError("Database DSN not available")
        db = Database(dsn)
        db.connect()
        try:
            logger.debug("topic_discover_and_assign_cluster cluster_id=%s", cluster_id)
            
            # First, get all bucket topics (seed topics)
            bucket_topics = db.query_all(
                """
                SELECT t.id, t.name, t.description
                FROM public.topic t
                WHERE t.source = 'seed'
                ORDER BY t.name ASC
                """,
            )
            
            # Get top 10 existing topics by vector similarity, excluding bucket topics
            similar_topics = db.query_all(
                """
                SELECT t.id, t.name, t.description
                FROM public.topic t
                WHERE t.embedding IS NOT NULL
                  AND t.source != 'seed'
                ORDER BY t.embedding <=> %s::vector ASC
                LIMIT 10
                """,
                (centroid_vec,),
            )

            # Get entities
            entities = db.query_all(
                """
                SELECT e.name
                FROM public.article_entity ae
                JOIN public.entity e ON e.id = ae.entity_id
                JOIN public.article_cluster ac ON ac.article_id = ae.article_id
                WHERE ac.cluster_id = %s
                GROUP BY e.name
                ORDER BY MAX(ae.salience) DESC
                LIMIT 6
                """,
                (cluster_id,),
            )
            entity_names = [name for (name,) in entities]

            # Build topic lists for LLM
            bucket_topics_list = [
                {"topic_id": int(tid), "name": name, "description": description}
                for tid, name, description in bucket_topics
            ]
            similar_topics_list = [
                {"topic_id": int(tid), "name": name, "description": description}
                for tid, name, description in similar_topics
            ]
            
            result = self._propose_and_match_topics(
                cluster_id=int(cluster_id),
                summary=summary,
                headlines=headlines,
                entities=entity_names,
                bucket_topics=bucket_topics_list,
                similar_topics=similar_topics_list,
            )

            # Clear existing topic assignments for this cluster
            bucket_topic_ids_selected = result.get("bucket_topics", [])
            existing_topic_ids = result.get("existing_topics", [])
            new_topics = result.get("topics", [])
            
            if bucket_topic_ids_selected or existing_topic_ids or new_topics:
                db.execute("DELETE FROM public.cluster_topic WHERE cluster_id = %s", (cluster_id,))

            # Process bucket topic matches (higher weight since these are primary categorization)
            linked_bucket = 0
            for topic_id in bucket_topic_ids_selected:
                try:
                    # Verify topic exists and is a bucket topic
                    topic_check = db.query_one(
                        "SELECT id FROM public.topic WHERE id = %s AND source = 'seed'",
                        (topic_id,)
                    )
                    if not topic_check:
                        logger.warning("bucket_topic_not_found cluster_id=%s topic_id=%s", cluster_id, topic_id)
                        continue
                    
                    # Link cluster to bucket topic with higher weight
                    db.execute(
                        """
                        INSERT INTO public.cluster_topic(cluster_id, topic_id, weight)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (cluster_id, topic_id) DO UPDATE SET weight = EXCLUDED.weight
                        """,
                        (cluster_id, topic_id, 0.8),  # Higher weight for bucket topics
                    )
                    linked_bucket += 1
                    logger.debug("bucket_topic_linked cluster_id=%s topic_id=%s", cluster_id, topic_id)
                except Exception as exc:
                    logger.warning("bucket_topic_link_failed cluster_id=%s topic_id=%s error=%s", cluster_id, topic_id, exc)
                    continue

            # Process existing topic matches (non-bucket similar topics)
            linked_existing = 0
            for topic_id in existing_topic_ids:
                try:
                    # Verify topic exists
                    topic_check = db.query_one("SELECT id FROM public.topic WHERE id = %s", (topic_id,))
                    if not topic_check:
                        logger.warning("topic_not_found cluster_id=%s topic_id=%s", cluster_id, topic_id)
                        continue
                    
                    # Link cluster to existing topic with default weight
                    db.execute(
                        """
                        INSERT INTO public.cluster_topic(cluster_id, topic_id, weight)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (cluster_id, topic_id) DO UPDATE SET weight = EXCLUDED.weight
                        """,
                        (cluster_id, topic_id, 0.5),  # Default weight for existing matches
                    )
                    linked_existing += 1
                    logger.debug("topic_linked_existing cluster_id=%s topic_id=%s", cluster_id, topic_id)
                except Exception as exc:
                    logger.warning("topic_link_failed cluster_id=%s topic_id=%s error=%s", cluster_id, topic_id, exc)
                    continue

            # Process new topic proposals
            inserted_count = 0
            if new_topics:
                reasons = {}
                for card in new_topics:
                    topic_name = card.get("name", "Unknown")
                    try:
                        vec = self.embedder.embed(card)
                    except Exception as exc:
                        logger.warning("topic_embed_failed name=%s error=%s", card.get("name"), exc)
                        continue
                    
                    # Insert new topic directly without duplicate checking
                    topic_id = db.query_one(
                        """
                        INSERT INTO public.topic(name, description, status, embedding, source, version)
                        VALUES (%s, %s, %s, %s::vector, %s, %s)
                        RETURNING id
                        """,
                        (
                            card.get("name"),
                            card.get("one_liner"),
                            "candidate",
                            TopicEmbedder.to_literal(vec),
                            "llm",
                            1,
                        ),
                    )[0]
                    inserted_count += 1
                    weight = float(card.get("weight", 0.5))
                    logger.info("topic_discovery_inserted cluster_id=%s topic_id=%s name=%s", cluster_id, topic_id, card.get("name"))
                    
                    # Link the new topic to the cluster
                    reason = card.get("reason")
                    db.execute(
                        """
                        INSERT INTO public.cluster_topic(cluster_id, topic_id, weight)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (cluster_id, topic_id) DO UPDATE SET weight = EXCLUDED.weight
                        """,
                        (cluster_id, topic_id, weight),
                    )
                    if reason:
                        reasons[str(topic_id)] = reason
                
                if reasons:
                    db.execute(
                        """
                        UPDATE public.cluster
                        SET meta = COALESCE(meta, '{}'::jsonb) || jsonb_build_object('topic_reasons', %s::jsonb)
                        WHERE id = %s
                        """,
                        (json.dumps(reasons), cluster_id),
                    )

            db.commit()
            logger.info(
                "topic_discover_and_assign_complete_cluster cluster_id=%s bucket_matched=%s existing_matched=%s new_proposed=%s",
                cluster_id, linked_bucket, linked_existing, len(new_topics)
            )
            return (inserted_count, True)
        except Exception as exc:
            db.rollback()
            logger.warning("topic_cluster_failed cluster_id=%s error=%s", cluster_id, str(exc))
            return None
        finally:
            db.close()

    def _propose_and_match_topics(
        self,
        cluster_id: int,
        summary: Optional[str],
        headlines: Optional[List[str]],
        entities: List[str],
        bucket_topics: List[Dict[str, Any]],
        similar_topics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """LLM call that returns bucket topic assignments, existing topic matches, and new topic proposals."""
        context = {
            "cluster_id": cluster_id,
            "summary": summary,
            "headlines": headlines,
            "entities": entities,
            "bucket_topics": bucket_topics,
            "similar_existing_topics": similar_topics,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are analyzing a news cluster to assign topics. Your response must be a JSON object with three fields:\n\n"
                    "1. 'bucket_topics': REQUIRED. An array of 1-2 topic_ids from the bucket_topics list that best categorize this cluster. "
                    "These are high-level category buckets. You MUST select at least 1 bucket topic.\n\n"
                    "2. 'existing_topics': An array of topic_ids from similar_existing_topics that match this cluster well. "
                    "These are more specific topics. Return empty array if none match well.\n\n"
                    "3. 'topics': An array of new topic proposals to fill gaps not covered by existing topics. Do not formulate overly specific new topics that would be too narrow to be attached to other similar clusters."
                    "Each proposal: {\"name\": str, \"one_liner\": str, \"definition\": [str], \"key_entities\": [str], \"weight\": float (0-1)}. "
                    "Return empty array if existing topics fully cover the cluster.\n\n"
                    "Response format: {\"bucket_topics\": [int], \"existing_topics\": [int], \"topics\": [...]}"
                ),
            },
            {"role": "user", "content": json.dumps(context)},
        ]
        try:
            resp = self._openrouter_client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.settings.openrouter_model,
                messages=messages,
                temperature=0.2,
                max_tokens=800,
                timeout=self.settings.llm_timeout_seconds,
                response_format=TOPIC_PROPOSAL_SCHEMA,
            )
            content = resp.choices[0].message.content  # type: ignore[index]
            data = json.loads(content or "{}")
            
            # Extract bucket_topics field (required, at least 1)
            bucket_topics_result = []
            if "bucket_topics" in data and isinstance(data["bucket_topics"], list):
                bucket_topics_result = [
                    int(tid) for tid in data["bucket_topics"]
                    if isinstance(tid, (int, str)) and str(tid).isdigit()
                ]
            
            # Extract existing_topics field
            existing_topics = []
            if "existing_topics" in data and isinstance(data["existing_topics"], list):
                existing_topics = [
                    int(tid) for tid in data["existing_topics"]
                    if isinstance(tid, (int, str)) and str(tid).isdigit()
                ]
            
            # Extract new topics field
            topics = []
            if "topics" in data and isinstance(data["topics"], list):
                topics = [card for card in data["topics"] if isinstance(card, dict)]
            
            # Warn if no bucket topics were returned
            if not bucket_topics_result:
                logger.warning("LLM returned no bucket_topics for cluster %s", cluster_id)
            
            return {
                "bucket_topics": bucket_topics_result,
                "existing_topics": existing_topics,
                "topics": topics,
            }
        except Exception as exc:
            logger.error("Failed to propose and match topics for cluster %s: %s", cluster_id, exc)
            raise RuntimeError(f"Topic proposal and matching LLM failed for cluster {cluster_id}") from exc

