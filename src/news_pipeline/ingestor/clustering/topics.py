from __future__ import annotations

import math
import logging
from typing import List, Tuple


logger = logging.getLogger(__name__)


def _softmax(xs: List[float], alpha: float) -> List[float]:
    # Stable softmax over alpha * xs
    ax = [alpha * x for x in xs]
    m = max(ax) if ax else 0.0
    exps = [math.exp(a - m) for a in ax]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def assign_topics_for_cluster(db, cluster_id: int, top_k: int = 3, alpha: float = 9.0) -> None:
    row = db.query_one("SELECT COUNT(*) FROM public.topic WHERE embedding IS NOT NULL")
    if not row or int(row[0]) == 0:
        logger.warning("assign_topics_skipped_no_topics cluster_id=%s", cluster_id)
        return
    # Ensure we have a centroid
    row = db.query_one("SELECT centroid_vec FROM public.cluster WHERE id = %s", (cluster_id,))
    if not row or row[0] is None:
        return
    centroid = row[0]
    if isinstance(centroid, str):
        qvec = centroid
    else:
        qvec = "[" + ",".join(f"{float(x):.6f}" for x in centroid) + "]"

    # If topic embeddings use a different dimensionality than the cluster centroid,
    # pad the vector to match so ANN queries succeed.
    target_dim = 768
    row = db.query_one("SELECT vector_dims(embedding) FROM public.topic WHERE embedding IS NOT NULL LIMIT 1")
    if row and row[0] and isinstance(row[0], int):
        target_dim = int(row[0])

    values = [float(x) for x in qvec.strip("[]").split(",") if x]
    current_dim = len(values)
    if target_dim > current_dim:
        values.extend([0.0] * (target_dim - current_dim))
        qvec = "[" + ",".join(f"{x:.6f}" for x in values) + "]"
    elif target_dim < current_dim:
        values = values[:target_dim]
        qvec = "[" + ",".join(f"{x:.6f}" for x in values) + "]"

    # Pick candidate topics by ANN over topic.embedding
    rows = db.query_all(
        """
        SELECT id, name, embedding <-> %s::vector AS dist
        FROM public.topic
        WHERE embedding IS NOT NULL
        ORDER BY embedding <-> %s::vector ASC
        LIMIT 12
        """,
        (qvec, qvec),
    )
    if not rows:
        return

    # Convert L2 distance to cosine approx and then weight by softmax
    cands: List[Tuple[int, float]] = []
    for tid, _name, dist in rows:
        d = float(dist)
        cos = 1.0 - (d * d) / 2.0
        cands.append((int(tid), cos))

    if not cands:
        return

    # Keep top_k by weight after softmax
    scores = [c for _, c in cands]
    probs = _softmax(scores, alpha=alpha)
    # Pair back
    weighted = list(zip([t for t, _ in cands], probs))
    # Sort by prob desc
    weighted.sort(key=lambda x: x[1], reverse=True)
    keep = weighted[:top_k]

    db.execute("DELETE FROM public.cluster_topic WHERE cluster_id = %s", (cluster_id,))
    with db.cursor() as cur:
        for tid, w in keep:
            cur.execute(
                """
                INSERT INTO public.cluster_topic(cluster_id, topic_id, weight)
                VALUES (%s, %s, %s)
                ON CONFLICT (cluster_id, topic_id) DO UPDATE SET weight = EXCLUDED.weight
                """,
                (cluster_id, tid, float(w)),
            )
