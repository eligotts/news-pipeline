from __future__ import annotations

from typing import Dict, List, Tuple

import logging


logger = logging.getLogger(__name__)

try:
    import igraph as ig  # type: ignore
except Exception:  # pragma: no cover
    ig = None  # type: ignore


def refresh_entity_communities(db) -> None:
    if ig is None:
        logger.info("entity_communities_skipped igraph_missing")
        return
    # Load 30-day edges
    logger.info("entity_communities_start")
    rows = db.query_all(
        """
        SELECT src_entity_id, dst_entity_id, weight
        FROM public.entity_cooccur
        WHERE "window" = 30 AND weight IS NOT NULL AND weight > 0
        """
    )
    if not rows:
        logger.info("entity_communities_no_edges")
        return

    # Map entity ids to 0..N-1 indices
    nodes: Dict[int, int] = {}
    edges: List[Tuple[int, int]] = []
    weights: List[float] = []
    for a, b, w in rows:
        a = int(a)
        b = int(b)
        if a not in nodes:
            nodes[a] = len(nodes)
        if b not in nodes:
            nodes[b] = len(nodes)
        edges.append((nodes[a], nodes[b]))
        weights.append(float(w))

    g = ig.Graph()
    g.add_vertices(len(nodes))
    g.add_edges(edges)
    g.es["weight"] = weights

    # Louvain (multilevel) community detection
    parts = g.community_multilevel(weights=weights)
    membership = parts.membership

    # Reverse map back to entity ids
    inv_nodes = {idx: eid for eid, idx in nodes.items()}
    # Persist to entity.meta.community_id
    with db.cursor() as cur:
        for idx, comm in enumerate(membership):
            eid = inv_nodes[idx]
            cur.execute(
                "UPDATE public.entity SET meta = COALESCE(meta, '{}'::jsonb) || jsonb_build_object('community_id', %s) WHERE id = %s",
                (int(comm), int(eid)),
            )
    logger.info("entity_communities_complete entities=%s communities=%s", len(nodes), len(set(membership)))
