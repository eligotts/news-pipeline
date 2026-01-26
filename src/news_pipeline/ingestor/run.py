from __future__ import annotations

"""Unified end-to-end ingestion runner.

Executes the full ingestion maintenance loop in the order prescribed by
`tech_spec.md` and `plan.md`:

1. Drain Pub/Sub queue and persist new articles via `IngestionProcessor`.
2. Run the article embedding worker to populate vectors + clusters + topics.
3. Run the stance worker to enrich entity relations with stance/role metadata.
4. Refresh cluster materialization, confirm/split via LLM, and generate summaries.
5. Run dynamic topic discovery + assignment jobs.
6. Recompute entity co-occurrence windows, neighbor lists, and communities.
7. Warm entity/topic caches for serving layer.

Each step fails fast and aborts the sequence if an exception is raised.
"""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import structlog
from dotenv import load_dotenv

from .clustering import ClusterMaintenance, TopicOrchestrator
from .config import get_settings
from .core import Database
from .llm import CoordinateRanker, EntityPipeline, OpenAIEmbedder
from .pipeline import EmbeddingBackend, EmbeddingWorker, FeatureScheduler, IngestionProcessor, IngestionResult

try:  # Optional dependency; required for queue draining.
    from google.cloud import pubsub_v1  # type: ignore
except Exception:  # pragma: no cover
    pubsub_v1 = None  # type: ignore


logger = structlog.get_logger()


DEFAULT_PARALLEL_BATCH_SIZE = 10


PIPELINE_STEPS: Tuple[str, ...] = (
    "drain_pubsub",
    "embedding_jobs",
    # NOTE: stance_jobs removed - stance extraction now inline during entity extraction
    "cluster_jobs",
    "topic_jobs",
    # NOTE: graph_jobs removed - entity edges now recorded inline during entity extraction
    # NOTE: cache_jobs removed - not using Redis cache layer
)
_STEP_INDEX = {name: idx for idx, name in enumerate(PIPELINE_STEPS)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full ingestion pipeline end-to-end")
    parser.add_argument("--max-messages", type=int, default=None, help="Maximum Pub/Sub messages to drain (default: pull 100 at a time until queue is empty)")
    parser.add_argument("--max-clusters", type=int, help="Maximum clusters to process (default: unlimited)")
    parser.add_argument("--max-topic-clusters", type=int, help="Maximum clusters to assign topics to (default: unlimited)")
    parser.add_argument(
        "--parallel-batch-size",
        type=int,
        default=DEFAULT_PARALLEL_BATCH_SIZE,
        help="Items per parallel batch when fan-out processing (default: 100)",
    )
    parser.add_argument("--metrics-endpoint", type=str, help="Optional Prometheus endpoint for before/after diff")
    parser.add_argument("--skip-cache", action="store_true", help="Skip cache warmers (entity/topic)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--start-step",
        type=str,
        choices=PIPELINE_STEPS,
        help="First pipeline step to execute (default: first step)",
    )
    parser.add_argument(
        "--end-step",
        type=str,
        choices=PIPELINE_STEPS,
        help="Final pipeline step to execute (default: last step)",
    )
    return parser.parse_args()


def _load_env_and_settings():
    load_dotenv()
    settings = get_settings()
    return settings


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(level))


def _init_db(settings) -> Database:
    db = Database(settings.supabase_dsn)  # type: ignore[arg-type]
    db.connect()
    return db


def _prepare_embeddings(settings) -> Tuple[EmbeddingBackend, OpenAIEmbedder]:
    openai_embedder = OpenAIEmbedder(api_key=settings.openai_api_key, model=settings.openai_embed_model)
    backend = EmbeddingBackend(openai_embedder=openai_embedder)
    return backend, openai_embedder


def _init_entity_pipeline(settings, embedder: OpenAIEmbedder) -> EntityPipeline:
    from .llm.client import create_openrouter_client

    llm_client = create_openrouter_client(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
    return EntityPipeline(
        llm_client=llm_client,
        llm_model=settings.openrouter_model,
        embedder=embedder,
        llm_timeout_seconds=settings.llm_timeout_seconds,
    )


def _drain_pubsub(
    settings,
    embed_backend: EmbeddingBackend,
    embedder: OpenAIEmbedder,
    limit: int | None,
    batch_size: int,
    max_workers: int,
) -> Tuple[List[IngestionResult], Dict[str, int]]:
    if pubsub_v1 is None:
        raise RuntimeError("google-cloud-pubsub is required to drain queue; install dependency")
    subscriber = pubsub_v1.SubscriberClient()
    sub_path = subscriber.subscription_path(settings.google_cloud_project, settings.pubsub_subscription)
    
    # Always pull 100 at a time (or less if limit is smaller)
    pull_batch_size = 50 if limit is None else min(50, limit)
    all_results: List[IngestionResult] = []
    all_stats = {"processed": 0, "duplicates": 0, "created": 0, "failed": 0, "url_canon_exists": 0, "validation_errors": 0, "processing_errors": 0}
    total_pulled = 0
    
    while True:
        # Pull messages
        response = subscriber.pull(request={"subscription": sub_path, "max_messages": pull_batch_size}, timeout=30)
        messages = list(response.received_messages)
        
        if not messages:
            logger.info("No more messages in subscription queue")
            break
        
        logger.info(f"Pulled {len(messages)} messages from Pub/Sub (batch {total_pulled // pull_batch_size + 1}, total pulled: {total_pulled + len(messages)})")
        total_pulled += len(messages)
        
        # Process this batch
        batch_results, batch_stats = _process_message_batch(
            messages, settings, embed_backend, embedder, batch_size, max_workers, subscriber, sub_path
        )
        
        all_results.extend(batch_results)
        all_stats["processed"] += batch_stats["processed"]
        all_stats["duplicates"] += batch_stats["duplicates"]
        all_stats["created"] += batch_stats["created"]
        all_stats["failed"] += batch_stats["failed"]
        all_stats["url_canon_exists"] += batch_stats.get("url_canon_exists", 0)
        all_stats["validation_errors"] += batch_stats.get("validation_errors", 0)
        all_stats["processing_errors"] += batch_stats.get("processing_errors", 0)
        
        # If limit was specified and we've reached it, stop
        if limit is not None and total_pulled >= limit:
            logger.info(f"Reached max-messages limit of {limit}")
            break
    
    logger.info(f"Drain complete: pulled {total_pulled} messages total")
    return all_results, all_stats


def _process_message_batch(
    messages: List,
    settings,
    embed_backend: EmbeddingBackend,
    embedder: OpenAIEmbedder,
    batch_size: int,
    max_workers: int,
    subscriber: Any,
    sub_path: str,
) -> Tuple[List[IngestionResult], Dict[str, int]]:
    """Process a batch of messages, acking only successful ones and nacking failures for redelivery."""
    stats = {"processed": 0, "duplicates": 0, "created": 0, "failed": 0, "url_canon_exists": 0, "validation_errors": 0, "processing_errors": 0}

    # Build ack_id -> message index mapping for tracking per-message success
    ack_id_to_idx = {rm.ack_id: idx for idx, rm in enumerate(messages)}
    all_ack_ids = list(ack_id_to_idx.keys())

    # Extend ack deadline IMMEDIATELY before any processing begins.
    # Default deadline is ~10s which is too short for our processing.
    # This gives us 10 minutes to process before Pub/Sub redelivers.
    if all_ack_ids:
        try:
            subscriber.modify_ack_deadline(
                request={
                    "subscription": sub_path,
                    "ack_ids": all_ack_ids,
                    "ack_deadline_seconds": 600,  # 10 minutes
                }
            )
            logger.debug(f"Extended ack deadline for {len(all_ack_ids)} messages to 600s")
        except Exception as e:
            logger.warning(f"Failed to extend ack deadline: {e}")

    chunks = _chunked(messages, batch_size)

    # Track which message indices succeeded vs failed
    successful_indices: set = set()
    failed_indices: set = set()

    # Wrapper to track success/failure per chunk and return per-message success info
    def handler_with_tracking(chunk_with_idx):
        chunk_idx, chunk = chunk_with_idx
        try:
            results, chunk_stats, msg_success = _process_ingestion_chunk_with_tracking(
                chunk, settings, embed_backend, embedder
            )
            return results, chunk_stats, msg_success, chunk_idx, True
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx}: {e}", exc_info=True)
            # Mark all messages in chunk as failed
            msg_success = {rm.ack_id: False for rm in chunk}
            return [], {"processed": 0, "duplicates": 0, "created": 0}, msg_success, chunk_idx, False

    # Process chunks in parallel with tracking
    chunk_outputs_with_status = _execute_parallel_batches(
        [(idx, chunk) for idx, chunk in enumerate(chunks)],
        handler_with_tracking,
        max_workers
    )

    results: List[IngestionResult] = []

    for chunk_results, chunk_stats, msg_success, chunk_idx, success in chunk_outputs_with_status:
        results.extend(chunk_results)
        stats["processed"] += chunk_stats["processed"]
        stats["duplicates"] += chunk_stats["duplicates"]
        stats["created"] += chunk_stats["created"]
        stats["url_canon_exists"] += chunk_stats.get("url_canon_exists", 0)
        stats["validation_errors"] += chunk_stats.get("validation_errors", 0)
        stats["processing_errors"] += chunk_stats.get("processing_errors", 0)

        # Track per-message success for selective acking
        for ack_id, succeeded in msg_success.items():
            if ack_id in ack_id_to_idx:
                if succeeded:
                    successful_indices.add(ack_id_to_idx[ack_id])
                else:
                    failed_indices.add(ack_id_to_idx[ack_id])

    stats["failed"] = len(failed_indices)

    # Collect ack_ids for successful and failed messages
    success_ack_ids = [rm.ack_id for idx, rm in enumerate(messages) if idx in successful_indices]
    failed_ack_ids = [rm.ack_id for idx, rm in enumerate(messages) if idx in failed_indices]

    # Ack only successful messages
    if success_ack_ids:
        try:
            subscriber.acknowledge(request={"subscription": sub_path, "ack_ids": success_ack_ids})
            logger.info(f"Acknowledged {len(success_ack_ids)} successful messages")
        except Exception as e:
            logger.error(f"Error acknowledging successful messages: {e}")

    # Nack failed messages to allow redelivery (set ack deadline to 0)
    if failed_ack_ids:
        try:
            subscriber.modify_ack_deadline(
                request={
                    "subscription": sub_path,
                    "ack_ids": failed_ack_ids,
                    "ack_deadline_seconds": 0,  # Immediate redelivery
                }
            )
            logger.warning(f"Nacked {len(failed_ack_ids)} failed messages for redelivery")
        except Exception as e:
            logger.error(f"Error nacking failed messages: {e}")

    return results, stats


def _drain_embedding_jobs(
    db: Database,
    settings,
    seq_backend: EmbeddingBackend,
    batch_size: int,
    max_workers: int,
) -> int:
    total = 0
    fetch_limit = max(batch_size * max_workers, batch_size)
    while True:
        # Select pending jobs. The atomic UPDATE in _process_single_job ensures
        # that even if multiple workers get the same job_id, only one will process it.
        rows = db.query_all(
            """
            SELECT article_id
            FROM public.article_embedding_job
            WHERE status = 'pending'
            ORDER BY priority ASC, queued_at ASC
            LIMIT %s
            """,
            (fetch_limit,),
        )
        job_ids = [int(row[0]) for row in rows]
        if not job_ids:
            break
        chunks = _chunked(job_ids, batch_size)
        handler = lambda chunk: _process_embedding_chunk(chunk, settings, seq_backend, batch_size)
        chunk_counts = _execute_parallel_batches(chunks, handler, max_workers)
        processed = sum(chunk_counts)
        total += processed
        if processed == 0:
            break
    return total


# NOTE: _drain_stance_jobs removed - stance extraction now inline during entity extraction


def _run_cluster_jobs(db: Database, settings, cluster_ids: Iterable[int] | None = None, max_clusters: int | None = None, max_workers: int = 3) -> Dict[str, int]:
    maint = ClusterMaintenance(db, settings)
    counts = {"refreshed": 0, "confirmed": 0, "summaries": 0}

    # If max_clusters is specified, query for that many cluster IDs (only clusters with size > 1)
    if max_clusters is not None and cluster_ids is None:
        rows = db.query_all(
            """
            SELECT id
            FROM public.cluster
            WHERE (ts_end IS NULL OR ts_end >= now() - interval '168 hours')
              AND size > 1
            ORDER BY ts_end DESC NULLS LAST
            LIMIT %s
            """,
            (max_clusters,),
        )
        cluster_ids = [int(row[0]) for row in rows]

    # materialization updates metadata for each cluster
    counts["refreshed"] = maint.refresh_materialization(recency_hours=168, cluster_ids=cluster_ids)
    db.commit()

    # Generate summaries - if max_clusters is specified, limit summaries to that count; otherwise process all
    counts["summaries"] = maint.generate_summaries(lookback_hours=48, limit=max_clusters, max_workers=max_workers)
    db.commit()
    return counts


def _run_topic_jobs(db: Database, settings, max_topic_clusters: int | None = None, max_workers: int = 3) -> Dict[str, int]:
    orchestrator = TopicOrchestrator(db, settings)
    # If max_topic_clusters not specified, process all clusters (use very large limit)
    limit = max_topic_clusters if max_topic_clusters is not None else 999999
    inserted, processed = orchestrator.discover_and_assign_topics(lookback_hours=72, limit=limit, max_workers=max_workers)
    db.commit()
    return {"discovered": inserted, "assigned": processed}


def _resolve_step_bounds(start_step: str | None, end_step: str | None) -> Tuple[int, int]:
    start_idx = 0 if start_step is None else _STEP_INDEX[start_step]
    end_idx = len(PIPELINE_STEPS) - 1 if end_step is None else _STEP_INDEX[end_step]
    if start_idx > end_idx:
        raise ValueError(
            f"start-step '{start_step}' occurs after end-step '{end_step}'"
        )
    return start_idx, end_idx


def _chunked(items: Sequence[Any], size: int) -> List[List[Any]]:
    if size <= 0:
        raise ValueError("parallel batch size must be positive")
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _execute_parallel_batches(chunks: List[List[Any]], worker, max_workers: int):
    if not chunks:
        return []
    workers = max(1, min(max_workers, len(chunks)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, chunk) for chunk in chunks]
        return [future.result() for future in futures]


def _process_ingestion_chunk_with_tracking(
    messages,
    settings,
    embed_backend: EmbeddingBackend,
    embedder: OpenAIEmbedder,
):
    """Process messages and return per-message success tracking for selective acking."""
    db = Database(settings.supabase_dsn)  # type: ignore[arg-type]
    db.connect()
    scheduler = FeatureScheduler(db, embed_backend)
    entity_pipeline = _init_entity_pipeline(settings, embedder)

    # Initialize coordinate ranker
    coord_ranker = CoordinateRanker(
        openrouter_api_key=settings.openrouter_api_key,  # type: ignore[arg-type]
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
    )

    processor = IngestionProcessor(db, scheduler, entity_pipeline, coord_ranker)
    stats = {"processed": 0, "duplicates": 0, "created": 0, "url_canon_exists": 0, "validation_errors": 0, "processing_errors": 0}
    results: List[IngestionResult] = []
    # Track success/failure per message by ack_id
    msg_success: Dict[str, bool] = {}

    try:
        for rm in messages:
            ack_id = rm.ack_id
            try:
                payload = json.loads(rm.message.data.decode("utf-8"))
                res = processor.process_article(payload)
                db.commit()
                results.append(res)
                stats["processed"] += 1
                msg_success[ack_id] = True  # Success
                if res.duplicate:
                    stats["duplicates"] += 1
                if res.created:
                    stats["created"] += 1
                if res.url_canon_exists:
                    stats["url_canon_exists"] += 1
            except ValueError as e:
                # Validation errors (invalid payload, bad timestamp, etc)
                # These are permanent failures - ack them to avoid infinite retry
                db.rollback()
                stats["validation_errors"] += 1
                msg_success[ack_id] = True  # Ack validation errors (permanent failures)
                error_msg = str(e)
                try:
                    raw_payload = json.loads(rm.message.data.decode("utf-8"))
                    article_info = raw_payload.get("article", raw_payload)
                    title = article_info.get("title", "unknown")[:60]
                    url = article_info.get("url", "unknown")[:80]
                except Exception:
                    title, url = "unknown", "unknown"
                logger.warning(
                    f"Validation error processing article (will not retry): {error_msg}",
                    extra={"title": title, "url": url, "error": error_msg}
                )
            except Exception as e:
                # Other processing errors - these might be transient, allow retry
                db.rollback()
                stats["processing_errors"] += 1
                msg_success[ack_id] = False  # Nack for retry
                logger.error(f"Error processing article (will retry): {e}", exc_info=True)
        return results, stats, msg_success
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _process_embedding_chunk(
    job_ids: Sequence[int],
    settings,
    backend: EmbeddingBackend,
    batch_size: int,
) -> int:
    if not job_ids:
        return 0
    db = Database(settings.supabase_dsn)  # type: ignore[arg-type]
    db.connect()
    try:
        worker = EmbeddingWorker(db, backend, batch_size=batch_size)
        return worker.process_jobs(job_ids)
    finally:
        db.close()


# NOTE: _process_stance_chunk removed - stance extraction now inline during entity extraction


def main() -> None:
    args = _parse_args()
    settings = _load_env_and_settings()
    _configure_logging(args.verbose)
    if args.parallel_batch_size <= 0:
        raise ValueError("parallel-batch-size must be positive")

    start_idx, end_idx = _resolve_step_bounds(args.start_step, args.end_step)

    def step_enabled(step: str) -> bool:
        return start_idx <= _STEP_INDEX[step] <= end_idx

    db = _init_db(settings)
    try:
        embed_backend, embedder = _prepare_embeddings(settings)
        max_workers = max(1, settings.max_concurrent_batches)

        ingest_results: List[IngestionResult] = []
        ingest_stats: Dict[str, int] = {"processed": 0, "duplicates": 0, "created": 0, "url_canon_exists": 0, "validation_errors": 0, "processing_errors": 0}
        if step_enabled("drain_pubsub"):
            ingest_results, ingest_stats = _drain_pubsub(
                settings,
                embed_backend,
                embedder,
                limit=args.max_messages,
                batch_size=args.parallel_batch_size,
                max_workers=max_workers,
            )
            # Log detailed breakdown to help debug missing articles
            total_msgs = ingest_stats["processed"] + ingest_stats.get("validation_errors", 0) + ingest_stats.get("processing_errors", 0)
            logger.info(
                "ingestion_complete",
                stats=ingest_stats,
                last_ids=[res.article_id for res in ingest_results[-5:]],
            )
            # Print detailed accounting of where messages went
            logger.info(
                "ingestion_accounting",
                total_messages_attempted=total_msgs,
                successfully_processed=ingest_stats["processed"],
                new_articles_created=ingest_stats["created"],
                url_canon_already_existed=ingest_stats["url_canon_exists"],
                content_sig_duplicates=ingest_stats["duplicates"],
                validation_errors=ingest_stats.get("validation_errors", 0),
                processing_errors=ingest_stats.get("processing_errors", 0),
                expected_embedding_jobs=ingest_stats["created"],  # Only new articles get embedding jobs
                note="url_canon_exists articles are UPDATED but do NOT create embedding jobs"
            )
        else:
            logger.info("ingestion_skipped", reason="outside selected step range")

        emb_processed = 0
        if step_enabled("embedding_jobs"):
            emb_processed = _drain_embedding_jobs(
                db,
                settings,
                embed_backend,
                batch_size=args.parallel_batch_size,
                max_workers=max_workers,
            )
            logger.info("embedding_jobs_complete", processed=emb_processed)
        else:
            logger.info("embedding_jobs_skipped", reason="outside selected step range")

        # NOTE: stance_jobs step removed - stance extraction now inline during entity extraction

        cluster_counts: Dict[str, int] = {"refreshed": 0, "confirmed": 0, "summaries": 0}
        if step_enabled("cluster_jobs"):
            cluster_counts = _run_cluster_jobs(db, settings, max_clusters=args.max_clusters, max_workers=max_workers)
            logger.info("cluster_jobs_complete", counts=cluster_counts)
        else:
            logger.info("cluster_jobs_skipped", reason="outside selected step range")

        topic_counts: Dict[str, int] = {"discovered": 0, "assigned": 0}
        if step_enabled("topic_jobs"):
            topic_counts = _run_topic_jobs(db, settings, max_topic_clusters=args.max_topic_clusters, max_workers=max_workers)
            logger.info("topic_jobs_complete", counts=topic_counts)
        else:
            logger.info("topic_jobs_skipped", reason="outside selected step range")

        # NOTE: graph_jobs and cache_jobs removed
        # Entity edges are now recorded inline during entity extraction (see llm/entities.py)

        logger.info(
            "run_complete",
            ingest=ingest_stats,
            embeddings=emb_processed,
            clusters=cluster_counts,
            topics=topic_counts,
            step_range={
                "start": args.start_step or PIPELINE_STEPS[start_idx],
                "end": args.end_step or PIPELINE_STEPS[end_idx],
            },
        )
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI entry point
        logger.error("run_failed", error=str(exc))
        raise
