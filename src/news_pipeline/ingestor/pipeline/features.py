from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import psycopg

from ..clustering import Clusterer, assign_topics_for_cluster

logger = logging.getLogger(__name__)


# Topic assignment thresholds
ARTICLE_TOPIC_SIMILARITY_THRESHOLD = 0.35  # Minimum similarity to assign topic
ARTICLE_TOPIC_LIMIT = 5  # Max topics per article


def assign_article_topics(db, article_id: int, v_title: List[float]) -> int:
    """
    Assign topics directly to an article based on vector similarity.

    This enables retrieval of articles regardless of cluster membership,
    solving the singleton cluster problem (83% of articles unreachable).

    Returns number of topics assigned.
    """
    if not v_title:
        return 0

    vec_literal = _vec_literal(v_title)

    # Find top topics by vector similarity
    # Using cosine distance: 1 - similarity, so lower is better
    # We compute similarity as: 1 - (distance / 2) for L2, or use <=> for cosine
    # NOTE: status filter removed - all topics are now valid
    rows = db.query_all(
        """
        SELECT t.id, 1 - (t.embedding <=> %s::vector) as similarity
        FROM public.topic t
        WHERE t.embedding IS NOT NULL
        ORDER BY t.embedding <=> %s::vector ASC
        LIMIT %s
        """,
        (vec_literal, vec_literal, ARTICLE_TOPIC_LIMIT * 2),  # Fetch extra to filter
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
            VALUES (%s, %s, %s, 'inferred')
            ON CONFLICT (article_id, topic_id) DO UPDATE SET
                weight = EXCLUDED.weight,
                source = EXCLUDED.source
            """,
            (article_id, int(topic_id), float(similarity)),
        )
        assigned += 1

    if assigned > 0:
        logger.debug("article_topics_assigned article_id=%s count=%s", article_id, assigned)

    return assigned


def _vec_literal(vec: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


def _ensure_dimensions(vec: Iterable[float], length: int) -> List[float]:
    values = list(float(x) for x in vec)
    if len(values) >= length:
        return values[:length]
    values.extend([0.0] * (length - len(values)))
    return values


 

class EmbeddingBackend:
    """Thin wrapper around the OpenAI embedding API with fixed 768-d outputs."""

    def __init__(self, openai_embedder) -> None:
        if openai_embedder is None:
            raise ValueError("OpenAIEmbedder is required for EmbeddingBackend")
        self._openai = openai_embedder

    def embed(self, title: str, lede: str, body: str) -> Tuple[List[float], List[float]]:
        """Return float vectors for (v_title, v_text) using OpenAI embeddings."""
        title_text = f"{title}\n{(lede or '')[:300]}".strip()
        body_text = f"{title}\n{(body or '')[:2000]}".strip()
        try:
            title_vec = self._openai.embed([title_text], dimensions=768)[0]
            text_vec = self._openai.embed([body_text], dimensions=768)[0]
        except Exception as exc:  # pragma: no cover - propagate service failure
            raise RuntimeError("OpenAI embedding request failed") from exc
        return _ensure_dimensions(title_vec, 768), _ensure_dimensions(text_vec, 768)


class FeatureScheduler:
    def __init__(
        self,
        db,
        backend: EmbeddingBackend,
    ) -> None:
        self.db = db
        self.backend = backend
        self.clusterer = Clusterer()

    def schedule(self, article_id: int, title: str, lede: Optional[str], body: Optional[str]) -> bool:
        self.db.execute(
            """
            INSERT INTO public.article_embedding_job(article_id, status, attempts)
            VALUES (%s, 'pending', 0)
            ON CONFLICT (article_id)
            DO UPDATE SET status = 'pending', attempts = CASE WHEN public.article_embedding_job.status = 'failed' THEN 0 ELSE public.article_embedding_job.attempts END
            """,
            (article_id,),
        )
        return False

    def _post_embed_refresh(self, article_id: int, title: str, lede: Optional[str], title_vec: List[float]) -> None:
        row = self.db.query_one(
            "SELECT ts_pub, publisher_id FROM public.article WHERE id = %s",
            (article_id,),
        )
        ts_pub = row[0] if row else None
        publisher_id = row[1] if row else None
        assign_res = self.clusterer.assign_cluster(
            db=self.db,
            article_id=article_id,
            title=title,
            lede=lede,
            v_title=title_vec,
            ts_pub=ts_pub,
            publisher_id=publisher_id,
        )
        assign_topics_for_cluster(self.db, assign_res.cluster_id)


MAX_EMBEDDING_JOB_ATTEMPTS = 5  # Move to dead status after this many failures


class EmbeddingWorker:
    def __init__(
        self,
        db,
        backend: EmbeddingBackend,
        batch_size: int = 16,
        max_attempts: int = MAX_EMBEDDING_JOB_ATTEMPTS,
    ) -> None:
        self.db = db
        self.backend = backend
        self.batch_size = batch_size
        self.clusterer = Clusterer()
        self.max_attempts = max_attempts

    def run_once(self) -> int:
        rows = self.db.query_all(
            """
            SELECT article_id
            FROM public.article_embedding_job
            WHERE status = 'pending'
            ORDER BY priority ASC, queued_at ASC
            LIMIT %s
            """,
            (self.batch_size,),
        )
        article_ids = [int(row[0]) for row in rows]
        return self.process_jobs(article_ids)

    def process_jobs(self, article_ids: Sequence[int]) -> int:
        processed = 0
        for article_id in article_ids:
            processed += self._process_single_job(article_id)
        return processed

    def _process_single_job(self, article_id: int) -> int:
        try:
            # Atomically claim the job by only updating if status is still 'pending'
            # This prevents multiple workers from processing the same job
            rows_updated = self.db.execute_update(
                """
                UPDATE public.article_embedding_job
                SET status = 'processing', attempts = attempts + 1
                WHERE article_id = %s AND status = 'pending'
                """,
                (article_id,),
            )

            # If no rows were updated, another worker already claimed this job
            if rows_updated == 0:
                return 0

            # Check if max attempts exceeded - move to dead status
            row = self.db.query_one(
                "SELECT attempts FROM public.article_embedding_job WHERE article_id = %s",
                (article_id,),
            )
            if row and row[0] > self.max_attempts:
                logger.error(
                    "embedding_job_max_attempts_exceeded article_id=%s attempts=%s max=%s",
                    article_id, row[0], self.max_attempts
                )
                self.db.execute(
                    """
                    UPDATE public.article_embedding_job
                    SET status = 'dead', last_error = %s
                    WHERE article_id = %s
                    """,
                    (f"Max attempts ({self.max_attempts}) exceeded", article_id),
                )
                self.db.commit()
                return 0

            art = self.db.query_one(
                "SELECT title, lede, body, ts_pub, publisher_id FROM public.article WHERE id = %s",
                (article_id,),
            )
            if not art:
                self.db.execute("DELETE FROM public.article_embedding_job WHERE article_id = %s", (article_id,))
                self.db.commit()
                return 0
            title, lede, body, ts_pub, publisher_id = art
            title_vec, text_vec = self.backend.embed(title or "", lede or "", body or "")
            v_title = _vec_literal(title_vec)
            v_text = _vec_literal(text_vec)
            self.db.execute(
                "UPDATE public.article SET v_title = %s::vector, v_text = %s::vector WHERE id = %s",
                (v_title, v_text, article_id),
            )
            assign_res = self.clusterer.assign_cluster(
                db=self.db,
                article_id=article_id,
                title=title or "",
                lede=lede,
                v_title=title_vec,
                ts_pub=ts_pub,
                publisher_id=publisher_id,
            )
            # Assign topics directly to article (enables singleton retrieval)
            assign_article_topics(self.db, article_id, title_vec)
            self.db.execute("DELETE FROM public.article_embedding_job WHERE article_id = %s", (article_id,))
            self.db.commit()
            return 1
        except (psycopg.OperationalError, psycopg.InterfaceError) as exc:
            # Connection error - connection is lost or broken
            logger.error("embedding_job_failed_connection article_id=%s error=%s", article_id, exc)
            self._handle_job_failure(article_id, f"Connection error: {str(exc)[:450]}")
            return 0
        except Exception as exc:
            # Other errors (not connection-related)
            logger.error("embedding_job_failed article_id=%s error=%s", article_id, exc)
            self._handle_job_failure(article_id, str(exc)[:500])
            return 0

    def _handle_job_failure(self, article_id: int, error_msg: str) -> None:
        """Handle job failure - update status to 'pending' for retry or 'dead' if max attempts exceeded."""
        try:
            self.db.rollback()
        except (psycopg.OperationalError, psycopg.InterfaceError):
            logger.warning("connection_lost_during_rollback article_id=%s", article_id)

        try:
            if not self.db.is_connected():
                self.db.reconnect()

            # Check current attempts to decide between 'pending' (retry) or 'dead' (give up)
            row = self.db.query_one(
                "SELECT attempts FROM public.article_embedding_job WHERE article_id = %s",
                (article_id,),
            )
            attempts = row[0] if row else 0

            if attempts >= self.max_attempts:
                # Max attempts exceeded - move to dead status (no more retries)
                logger.error(
                    "embedding_job_dead article_id=%s attempts=%s max=%s error=%s",
                    article_id, attempts, self.max_attempts, error_msg[:100]
                )
                self.db.execute_update(
                    """
                    UPDATE public.article_embedding_job
                    SET status = 'dead', last_error = %s
                    WHERE article_id = %s AND status = 'processing'
                    """,
                    (f"Max attempts exceeded. Last error: {error_msg}", article_id),
                )
            else:
                # Return to pending for retry
                self.db.execute_update(
                    """
                    UPDATE public.article_embedding_job
                    SET status = 'pending', last_error = %s
                    WHERE article_id = %s AND status = 'processing'
                    """,
                    (error_msg, article_id),
                )
            self.db.commit()
        except Exception as update_exc:
            logger.error("failed_to_update_job_status article_id=%s error=%s", article_id, update_exc)
