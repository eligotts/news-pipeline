from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import psycopg

from .clustering import Clusterer
from .topics import assign_topics_for_cluster

logger = logging.getLogger(__name__)


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


class EmbeddingWorker:
    def __init__(
        self,
        db,
        backend: EmbeddingBackend,
        batch_size: int = 16,
    ) -> None:
        self.db = db
        self.backend = backend
        self.batch_size = batch_size
        self.clusterer = Clusterer()

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
            # dont think we need to re assign topics EVERY time we assign an article to a cluster
            # assign_topics_for_cluster(self.db, assign_res.cluster_id)
            self.db.execute("DELETE FROM public.article_embedding_job WHERE article_id = %s", (article_id,))
            self.db.commit()
            return 1
        except (psycopg.OperationalError, psycopg.InterfaceError) as exc:
            # Connection error - connection is lost or broken
            logger.error("embedding_job_failed_connection article_id=%s error=%s", article_id, exc)
            # Try to rollback gracefully (will handle connection loss)
            self.db.rollback()
            # Try to reconnect and update job status
            try:
                self.db.reconnect()
                # Only update status if we successfully claimed the job (status was 'pending')
                # Use a conditional update to avoid overwriting if another worker already handled it
                self.db.execute_update(
                    """
                    UPDATE public.article_embedding_job 
                    SET status = 'failed', last_error = %s 
                    WHERE article_id = %s AND status = 'processing'
                    """,
                    (f"Connection error: {str(exc)[:450]}", article_id),
                )
                self.db.commit()
            except Exception as update_exc:
                # If we can't update status, log it but don't fail
                logger.error("failed_to_update_job_status_after_connection_error article_id=%s error=%s", article_id, update_exc)
            return 0
        except Exception as exc:
            # Other errors (not connection-related)
            logger.error("embedding_job_failed article_id=%s error=%s", article_id, exc)
            # Try to rollback if connection is still valid
            try:
                self.db.rollback()
            except (psycopg.OperationalError, psycopg.InterfaceError):
                # Connection lost during rollback, reconnect
                logger.warning("connection_lost_during_rollback article_id=%s", article_id)
                self.db.reconnect()
            
            # Try to update job status
            try:
                if not self.db.is_connected():
                    self.db.reconnect()
                # Only update status if we successfully claimed the job (status was 'pending')
                # Use a conditional update to avoid overwriting if another worker already handled it
                self.db.execute_update(
                    """
                    UPDATE public.article_embedding_job 
                    SET status = 'failed', last_error = %s 
                    WHERE article_id = %s AND status = 'processing'
                    """,
                    (str(exc)[:500], article_id),
                )
                self.db.commit()
            except Exception as update_exc:
                # If we can't update status, log it but don't fail
                logger.error("failed_to_update_job_status article_id=%s error=%s", article_id, update_exc)
            return 0
