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

from ._archive.cache import CacheManager, cache_hot_entities, cache_hot_topics
from ._archive.cooccur import refresh_entity_cooccur
from ._archive.communities import refresh_entity_communities
from ._archive.neighbors import refresh_entity_neighbors
from .cluster_jobs import ClusterMaintenance
from .config import get_settings
from .coordinate_ranker import CoordinateRanker
from .db import Database
from .feature_jobs import EmbeddingBackend, EmbeddingWorker, FeatureScheduler
from .ner_link import EntityPipeline
from .openai_embed import OpenAIEmbedder
from .processor import IngestionProcessor, IngestionResult
from .stance_worker import StanceWorker
from .topics_dynamic import TopicOrchestrator

try:  # Optional dependency; required for queue draining.
    from google.cloud import pubsub_v1  # type: ignore
except Exception:  # pragma: no cover
    pubsub_v1 = None  # type: ignore


logger = structlog.get_logger()


DEFAULT_PARALLEL_BATCH_SIZE = 10


PIPELINE_STEPS: Tuple[str, ...] = (
    "drain_pubsub",
    "embedding_jobs",
    "stance_jobs",
    "cluster_jobs",
    "topic_jobs",
    "graph_jobs",
    "cache_jobs",
)
_STEP_INDEX = {name: idx for idx, name in enumerate(PIPELINE_STEPS)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full ingestion pipeline end-to-end")
    parser.add_argument("--max-messages", type=int, default=None, help="Maximum Pub/Sub messages to drain (default: pull 100 at a time until queue is empty)")
    parser.add_argument("--max-stance-jobs", type=int, help="Maximum stance jobs to process (default: unlimited)")
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
    parser.add_argument("--bypass-stance-llm", action="store_true", help="Bypass LLM calls for stance jobs and use default values")
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
    from .openrouter_client import create_openrouter_client

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
    """Process a single batch of messages and ack all messages regardless of processing outcome."""
    stats = {"processed": 0, "duplicates": 0, "created": 0, "failed": 0, "url_canon_exists": 0, "validation_errors": 0, "processing_errors": 0}
    
    # Collect all ack_ids - we'll ack all messages regardless of success/failure
    all_ack_ids = [rm.ack_id for rm in messages]

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
    
    # Wrapper to track success/failure per chunk
    def handler_with_tracking(chunk_with_idx):
        chunk_idx, chunk = chunk_with_idx
        try:
            return _process_ingestion_chunk(chunk, settings, embed_backend, embedder), chunk_idx, True
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx}: {e}", exc_info=True)
            return ([], {"processed": 0, "duplicates": 0, "created": 0}), chunk_idx, False
    
    # Process chunks in parallel with tracking
    chunk_outputs_with_status = _execute_parallel_batches(
        [(idx, chunk) for idx, chunk in enumerate(chunks)],
        handler_with_tracking,
        max_workers
    )

    results: List[IngestionResult] = []
    failed_count = 0
    
    for (chunk_results, chunk_stats), chunk_idx, success in chunk_outputs_with_status:
        results.extend(chunk_results)
        stats["processed"] += chunk_stats["processed"]
        stats["duplicates"] += chunk_stats["duplicates"]
        stats["created"] += chunk_stats["created"]
        stats["url_canon_exists"] += chunk_stats.get("url_canon_exists", 0)
        stats["validation_errors"] += chunk_stats.get("validation_errors", 0)
        stats["processing_errors"] += chunk_stats.get("processing_errors", 0)

        # Count failed chunks
        if not success:
            failed_count += len(chunks[chunk_idx])
    
    stats["failed"] = failed_count

    # Ack all messages regardless of processing outcome
    if all_ack_ids:
        try:
            subscriber.acknowledge(request={"subscription": sub_path, "ack_ids": all_ack_ids})
            logger.info(f"Acknowledged {len(all_ack_ids)} messages (processed: {len(results)}, failed: {failed_count})")
        except Exception as e:
            logger.error(f"Error acknowledging messages: {e}")
    
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


def _drain_stance_jobs(
    db: Database,
    settings,
    batch_size: int,
    max_workers: int,
    max_jobs: int | None = None,
    bypass_llm: bool = False,
) -> int:
    total = 0
    while True:
        if max_jobs is not None and total >= max_jobs:
            break
        remaining = None if max_jobs is None else max(0, max_jobs - total)
        fetch_limit = batch_size * max_workers if remaining is None else min(remaining, batch_size * max_workers)
        if fetch_limit == 0:
            break
        rows = db.query_all(
            """
            SELECT article_id, entity_id
            FROM public.article_entity_stance_job
            WHERE status = 'pending'
            ORDER BY queued_at ASC
            LIMIT %s
            """,
            (fetch_limit,),
        )
        jobs = [(int(article_id), int(entity_id)) for article_id, entity_id in rows]
        if not jobs:
            break
        chunks = _chunked(jobs, batch_size)
        handler = lambda chunk: _process_stance_chunk(chunk, settings, bypass_llm)
        chunk_counts = _execute_parallel_batches(chunks, handler, max_workers)
        processed = sum(chunk_counts)
        total += processed
        if processed == 0:
            break
    return total


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
    ## Lets skip this step for now
    # counts["confirmed"] = maint.confirm_clusters(recency_hours=24, limit=50)
    # db.commit()
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


def _run_graph_jobs(db: Database) -> Dict[str, int]:
    refresh_entity_cooccur(db, window_days=7)
    db.commit()
    refresh_entity_cooccur(db, window_days=30)
    db.commit()
    refresh_entity_neighbors(db)
    db.commit()
    refresh_entity_communities(db)
    db.commit()
    return {"cooccur_7": 1, "cooccur_30": 1, "neighbors": 1, "communities": 1}


def _run_cache_jobs(db: Database, settings) -> Dict[str, int]:
    cache = CacheManager(settings)
    ent = cache_hot_entities(db, cache, lookback_hours=72, limit=1000)
    db.commit()
    top = cache_hot_topics(db, cache, lookback_hours=72, limit=500)
    db.commit()
    return {"entities": ent, "topics": top}


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


def _process_ingestion_chunk(
    messages,
    settings,
    embed_backend: EmbeddingBackend,
    embedder: OpenAIEmbedder,
):
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
    try:
        for rm in messages:
            try:
                payload = json.loads(rm.message.data.decode("utf-8"))
                res = processor.process_article(payload)
                db.commit()
                results.append(res)
                stats["processed"] += 1
                if res.duplicate:
                    stats["duplicates"] += 1
                if res.created:
                    stats["created"] += 1
                if res.url_canon_exists:
                    stats["url_canon_exists"] += 1
            except ValueError as e:
                # Validation errors (invalid payload, bad timestamp, etc)
                db.rollback()
                stats["validation_errors"] += 1
                error_msg = str(e)
                # Try to extract some identifying info from the message
                try:
                    raw_payload = json.loads(rm.message.data.decode("utf-8"))
                    article_info = raw_payload.get("article", raw_payload)
                    title = article_info.get("title", "unknown")[:60]
                    url = article_info.get("url", "unknown")[:80]
                except Exception:
                    title, url = "unknown", "unknown"
                logger.warning(
                    f"Validation error processing article: {error_msg}",
                    extra={"title": title, "url": url, "error": error_msg}
                )
            except Exception as e:
                # Other processing errors
                db.rollback()
                stats["processing_errors"] += 1
                logger.error(f"Error processing article: {e}", exc_info=True)
        return results, stats
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


def _process_stance_chunk(
    jobs: Sequence[tuple[int, int]],
    settings,
    bypass_llm: bool = False,
) -> int:
    if not jobs:
        return 0
    from .openrouter_client import create_openrouter_client

    db = Database(settings.supabase_dsn)  # type: ignore[arg-type]
    db.connect()
    try:
        # Only create OpenRouter client if not bypassing LLM
        client = None if bypass_llm else create_openrouter_client(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
        worker = StanceWorker(db, client, settings.openrouter_model, settings.llm_timeout_seconds, bypass_llm=bypass_llm)
        return worker.process_jobs(jobs)
    finally:
        db.close()


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

        stance_processed = 0
        if step_enabled("stance_jobs"):
            stance_processed = _drain_stance_jobs(
                db,
                settings,
                batch_size=args.parallel_batch_size,
                max_workers=max_workers,
                max_jobs=args.max_stance_jobs,
                bypass_llm=args.bypass_stance_llm,
            )
            logger.info("stance_jobs_complete", processed=stance_processed, bypass_llm=args.bypass_stance_llm)
        else:
            logger.info("stance_jobs_skipped", reason="outside selected step range")

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

        graph_counts: Dict[str, int] = {"cooccur_7": 0, "cooccur_30": 0, "neighbors": 0, "communities": 0}
        if step_enabled("graph_jobs"):
            graph_counts = _run_graph_jobs(db)
            logger.info("graph_jobs_complete", counts=graph_counts)
        else:
            logger.info("graph_jobs_skipped", reason="outside selected step range")

        cache_counts = {"entities": 0, "topics": 0}
        if step_enabled("cache_jobs"):
            if args.skip_cache:
                logger.info("cache_jobs_skipped", reason="skip_cache flag set")
            else:
                cache_counts = _run_cache_jobs(db, settings)
                logger.info("cache_jobs_complete", counts=cache_counts)
        else:
            logger.info("cache_jobs_skipped", reason="outside selected step range")

        logger.info(
            "run_complete",
            ingest=ingest_stats,
            embeddings=emb_processed,
            stance=stance_processed,
            clusters=cluster_counts,
            topics=topic_counts,
            graph=graph_counts,
            cache=cache_counts,
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
