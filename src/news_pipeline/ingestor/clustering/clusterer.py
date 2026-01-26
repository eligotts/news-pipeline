from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional


logger = logging.getLogger(__name__)


def vec_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def approx_cos_from_l2(dist: float) -> float:
    # If vectors are normalized, ||u-v||^2 = 2(1 - cos)
    return max(-1.0, min(1.0, 1.0 - (dist * dist) / 2.0))


@dataclass
class ClusterAssignResult:
    cluster_id: int
    is_new: bool


class Clusterer:
    def __init__(self, tau_cos: float = 0.80, candidate_limit: int = 300, recency_hours: int = 72):
        self.tau_cos = tau_cos
        self.candidate_limit = candidate_limit
        self.recency_hours = recency_hours

    def assign_cluster(
        self,
        db,
        article_id: int,
        title: str,
        lede: str | None,
        v_title: Optional[List[float]],
        ts_pub,
        publisher_id: Optional[int],
    ) -> ClusterAssignResult:
        if v_title is None or not v_title:
            row = db.query_one("SELECT v_title FROM public.article WHERE id = %s", (article_id,))
            if not row or row[0] is None:
                raise RuntimeError("Vector features unavailable for clustering")
            stored = row[0]
            if isinstance(stored, str):
                v_title = [float(x) for x in stored.strip("[]").split(",") if x]
            else:
                v_title = list(stored)
            if not v_title:
                raise RuntimeError("Vector features unavailable for clustering (empty vector)")

        if ts_pub is None:
            row = db.query_one("SELECT ts_pub FROM public.article WHERE id = %s", (article_id,))
            ts_pub = row[0] if row else None
        if publisher_id is None:
            row = db.query_one("SELECT publisher_id FROM public.article WHERE id = %s", (article_id,))
            publisher_id = row[0] if row else None
        if ts_pub is None:
            raise RuntimeError("Publication timestamp required for clustering")

        # Track any existing membership to avoid duplicate inserts when reprocessing
        existing_cluster_id: Optional[int] = None
        existing_row = db.query_one("SELECT cluster_id FROM public.article_cluster WHERE article_id = %s", (article_id,))
        if existing_row and existing_row[0] is not None:
            existing_cluster_id = int(existing_row[0])

        # Fetch ANN candidates by L2 distance over a recent window
        qvec = vec_literal(v_title)
        rows = db.query_all(
            """
            SELECT id, ts_pub, v_title <-> %s::vector AS dist
            FROM public.article
            WHERE id <> %s
              AND ts_pub >= now() - (%s * interval '1 hour')
              AND v_title IS NOT NULL
            ORDER BY v_title <-> %s::vector ASC
            LIMIT %s
            """,
            (qvec, article_id, self.recency_hours, qvec, self.candidate_limit),
        )

        # Collect all matches with cosine >= 0.6
        matching_article_ids: List[int] = []
        cosine_threshold = 0.6
        for rid, _, dist in rows:
            cos = approx_cos_from_l2(float(dist))
            if cos >= cosine_threshold:
                matching_article_ids.append(int(rid))

        # Get unique cluster_ids from matching articles
        unique_cluster_ids: set[int] = set()
        for match_article_id in matching_article_ids:
            row = db.query_one("SELECT cluster_id FROM public.article_cluster WHERE article_id = %s", (match_article_id,))
            if row and row[0] is not None:
                unique_cluster_ids.add(int(row[0]))

        if not unique_cluster_ids:
            # No matches found, create new cluster
            if existing_cluster_id is not None:
                # Article already belongs to a cluster, refresh its rep score
                if ts_pub is not None:
                    with db.cursor() as cur:
                        cur.execute(
                            "UPDATE public.article_cluster SET rep_score = %s WHERE article_id = %s AND cluster_id = %s",
                            (self._rep_score_now(1.0, ts_pub, publisher_id, existing_cluster_id, db), article_id, existing_cluster_id),
                        )
                return ClusterAssignResult(cluster_id=existing_cluster_id, is_new=False)
            # Materialize new cluster with summary = title initially
            with db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.cluster(centroid_vec, ts_start, ts_end, size, top_headlines, summary)
                    VALUES (%s::vector, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (qvec, ts_pub, ts_pub, 1, [title][:1], title),
                )
                cluster_id = cur.fetchone()[0]
                cur.execute(
                    "INSERT INTO public.article_cluster(article_id, cluster_id, rep_score) VALUES (%s, %s, %s)",
                    (article_id, cluster_id, self._rep_score_now(1.0, ts_pub, publisher_id, cluster_id, db)),
                )
            return ClusterAssignResult(cluster_id=cluster_id, is_new=True)

        # Assign article to all matching clusters
        # Sort cluster IDs to ensure consistent lock ordering across workers (prevents deadlocks)
        first_cluster_id = None
        for cluster_id in sorted(unique_cluster_ids):
            if first_cluster_id is None:
                first_cluster_id = cluster_id

            # Acquire exclusive lock on cluster row BEFORE inserting into article_cluster
            # This prevents deadlocks from FK checks acquiring FOR KEY SHARE locks
            db.execute("SELECT id FROM public.cluster WHERE id = %s FOR UPDATE", (cluster_id,))

            # Upsert membership for current article
            rep_score = self._rep_score_now(None, ts_pub, publisher_id, cluster_id, db)
            with db.cursor() as cur:
                cur.execute(
                    "INSERT INTO public.article_cluster(article_id, cluster_id, rep_score) VALUES (%s, %s, %s) ON CONFLICT (article_id, cluster_id) DO UPDATE SET rep_score = EXCLUDED.rep_score",
                    (article_id, cluster_id, rep_score),
                )

            # Update cluster metadata: centroid, ts bounds, size, headlines
            self._refresh_cluster_materialization(db, cluster_id)

        return ClusterAssignResult(cluster_id=first_cluster_id, is_new=False)

    def _refresh_cluster_materialization(self, db, cluster_id: int) -> None:
        # Acquire row lock first to prevent deadlocks when multiple workers
        # try to update the same cluster simultaneously
        db.execute("SELECT id FROM public.cluster WHERE id = %s FOR UPDATE", (cluster_id,))

        # Fetch members
        rows = db.query_all(
            """
            SELECT a.id, a.title, a.ts_pub, a.v_title
            FROM public.article a
            JOIN public.article_cluster ac ON ac.article_id = a.id
            WHERE ac.cluster_id = %s
            ORDER BY a.ts_pub DESC
            """,
            (cluster_id,),
        )
        # Compute centroid as mean of member vectors
        vecs: List[List[float]] = []
        headlines: List[str] = []
        ts_start = None
        ts_end = None
        for _, title, ts_pub, v in rows:
            if v is None:
                continue
            # psycopg returns pgvector as a list-like in many setups; if string, parse
            if isinstance(v, str):
                v_list = [float(x) for x in v.strip("[]").split(",") if x]
            else:
                v_list = list(v)
            if not v_list:
                continue
            vecs.append(v_list)
            if title and len(headlines) < 5:
                headlines.append(title)
            ts_start = ts_pub if ts_start is None or ts_pub < ts_start else ts_start
            ts_end = ts_pub if ts_end is None or ts_pub > ts_end else ts_end

        if not vecs:
            return
        dim = len(vecs[0])
        mean = [0.0] * dim
        for v in vecs:
            for i in range(dim):
                mean[i] += v[i]
        n = float(len(vecs))
        mean = [x / n for x in mean]

        with db.cursor() as cur:
            cur.execute(
                "UPDATE public.cluster SET centroid_vec = %s::vector, ts_start = %s, ts_end = %s, size = %s, top_headlines = %s WHERE id = %s",
                (vec_literal(mean), ts_start, ts_end, len(vecs), headlines[:5], cluster_id),
            )

    def _rep_score_now(
        self,
        cos_to_centroid: Optional[float],
        ts_pub,
        publisher_id: Optional[int],
        cluster_id: int,
        db,
    ) -> float:
        # recency decay
        # Do in SQL: exp(-extract(epoch from (now()-ts_pub))/86400)
        row = db.query_one("SELECT EXTRACT(EPOCH FROM (now() - %s))", (ts_pub,))
        age_sec = float(row[0]) if row else 0.0
        recency = math.exp(-age_sec / 86400.0)

        # reliability y from article
        row = db.query_one("SELECT y FROM public.article WHERE id = (SELECT article_id FROM public.article_cluster WHERE cluster_id = %s ORDER BY article_id LIMIT 1)", (cluster_id,))
        y = float(row[0]) if row else 0.7

        # src diversity: count same publisher in cluster
        if publisher_id is not None:
            row = db.query_one(
                """
                SELECT COUNT(*) FROM public.article a
                JOIN public.article_cluster ac ON ac.article_id = a.id
                WHERE ac.cluster_id = %s AND a.publisher_id = %s
                """,
                (cluster_id, publisher_id),
            )
            same_pub = int(row[0]) if row else 0
            src_div = 1.0 - min(same_pub / 3.0, 1.0)
        else:
            src_div = 1.0

        cos = cos_to_centroid if cos_to_centroid is not None else 0.9

        score = 0.60 * cos + 0.20 * recency + 0.15 * max(0.0, min(y, 1.0)) + 0.05 * src_div
        return float(score)
