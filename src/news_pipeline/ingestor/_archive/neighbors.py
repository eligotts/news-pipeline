from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import logging


logger = logging.getLogger(__name__)


def refresh_entity_neighbors(db, w30: float = 0.6, w7: float = 0.4, k: int = 50, min_w: float = 0.1) -> None:
    logger.info("entity_neighbors_start")
    rows = db.query_all(
        """
        SELECT src_entity_id, dst_entity_id, "window", weight
        FROM public.entity_cooccur
        WHERE "window" IN (7, 30)
        """
    )
    if not rows:
        logger.info("entity_neighbors_no_edges")
        return

    # Build blended adjacency
    adj: DefaultDict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for src, dst, win, w in rows:
        src = int(src)
        dst = int(dst)
        w = float(w)
        blend = w30 * w if int(win) == 30 else w7 * w
        adj[src][dst] += blend
        adj[dst][src] += blend

    total_rel = 0
    with db.cursor() as cur:
        for idx, (eid, nbrs) in enumerate(adj.items(), start=1):
            ranked: List[Tuple[int, float]] = sorted(nbrs.items(), key=lambda x: x[1], reverse=True)
            keep = [(nid, float(w)) for nid, w in ranked if w >= min_w][:k]
            for rank, (nid, w) in enumerate(keep, start=1):
                cur.execute(
                    """
                    INSERT INTO public.entity_neighbors(entity_id, neighbor_id, weight, rank, updated_at)
                    VALUES (%s, %s, %s, %s, now())
                    ON CONFLICT (entity_id, neighbor_id)
                    DO UPDATE SET weight = EXCLUDED.weight, rank = EXCLUDED.rank, updated_at = now()
                    """,
                    (eid, int(nid), w, rank),
                )
                total_rel += 1
            if idx % 200 == 0 or idx == 1:
                logger.info("entity_neighbors_progress entities=%s relationships=%s", idx, total_rel)
    logger.info("entity_neighbors_complete entries=%s", total_rel)
