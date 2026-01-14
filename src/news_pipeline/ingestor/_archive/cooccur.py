from __future__ import annotations

import math
from collections import defaultdict
import statistics
from typing import DefaultDict, Dict, Iterable, List, Set, Tuple

import logging


logger = logging.getLogger(__name__)


def _decay(dt_hours: float, tau_hours: float) -> float:
    return math.exp(-dt_hours / tau_hours)


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    idx = (len(values) - 1) * q
    lower = int(idx)
    upper = min(lower + 1, len(values) - 1)
    frac = idx - lower
    return values[lower] * (1 - frac) + values[upper] * frac


def _weight_beta(window_days: int) -> float:
    """Return smoothing constant for PMI-to-weight mapping based on window size."""
    if window_days <= 7:
        return 2.0
    if window_days <= 14:
        return 3.0
    if window_days <= 30:
        return 6.0
    return 7.0


def _weight_from_pmi(pmi: float, beta: float) -> float:
    """Map PMI to [0,1) while preserving differences among high-PMI pairs."""
    if pmi <= 0.0:
        return 0.0
    return pmi / (pmi + beta)


def refresh_entity_cooccur(
    db,
    window_days: int = 7,
    top_per_pair: int | None = None,
    min_support: int = 2,
) -> None:
    """Compute decayed co-occurrence over clusters ending within the window.

    Writes to public.entity_cooccur with src<dst and given window.
    """
    # Collect clusters and their unique entities
    logger.info("entity_cooccur_start window_days=%s", window_days)
    rows = db.query_all(
        """
        SELECT c.id, c.ts_end
        FROM public.cluster c
        WHERE c.ts_end IS NOT NULL
          AND c.ts_end >= now() - (%s * interval '1 day')
          AND c.size > 1
        """,
        (window_days,),
    )
    if not rows:
        logger.info("entity_cooccur_no_clusters window_days=%s", window_days)
        return

    # Preload article->entities mapping per cluster
    # For each cluster, collect unique entity IDs
    cl_entities: Dict[int, Set[int]] = {}
    cl_ts_end: Dict[int, float] = {}
    for cid, ts_end in rows:
        cl_ts_end[int(cid)] = float(ts_end.timestamp())
        ents = db.query_all(
            """
            SELECT DISTINCT ae.entity_id
            FROM public.article_entity ae
            JOIN public.article_cluster ac ON ac.article_id = ae.article_id
            WHERE ac.cluster_id = %s
            """,
            (cid,),
        )
        cl_entities[int(cid)] = {int(eid) for (eid,) in ents}

    if not cl_entities:
        logger.info("entity_cooccur_no_entities window_days=%s", window_days)
        return

    # Accumulate counts with decay
    tau_hours = (window_days / 2.0) * 24.0
    now_row = db.query_one("SELECT EXTRACT(EPOCH FROM now())")
    now_ts = float(now_row[0]) if now_row else 0.0
    cooc: DefaultDict[Tuple[int, int], float] = defaultdict(float)
    cnt: DefaultDict[int, float] = defaultdict(float)

    total_pairs = 0
    total_cluster_weight = 0.0
    cooc_support: DefaultDict[Tuple[int, int], int] = defaultdict(int)
    for idx, (cid, ents) in enumerate(cl_entities.items(), start=1):
        if not ents:
            continue
        dt_hours = (now_ts - cl_ts_end[cid]) / 3600.0
        w = _decay(dt_hours, tau_hours)
        total_cluster_weight += w
        elist = sorted(ents)
        if idx % 10 == 0 or idx == 1:
            logger.info(
                "entity_cooccur_cluster_progress cid=%s entities=%s weight=%.4f processed=%s",
                cid,
                len(elist),
                w,
                idx,
            )
        # Unique entities per cluster
        for i, ei in enumerate(elist):
            cnt[ei] += w
            for ej in elist[i + 1 :]:
                a, b = (ei, ej) if ei < ej else (ej, ei)
                cooc[(a, b)] += w
                cooc_support[(a, b)] += 1
                total_pairs += 1

    # Compute PMI-like weight and write top edges
    # N should represent the total cluster weight seen within the window
    N = max(1e-6, total_cluster_weight)
    edges: List[Tuple[int, int, float]] = []

    logger.info("entity_cooccur_pmi_start N=%.4f num_edges=%s", N, len(cooc))

    min_support = max(1, int(min_support))
    if window_days >= 30:
        min_support = max(min_support, 3)
    dropped_sparse = 0
    beta = _weight_beta(window_days)
    logger.info(
        "entity_cooccur_weight_params window_days=%s beta=%.3f min_support=%s",
        window_days,
        beta,
        min_support,
    )
    pmi_values: List[float] = []
    for idx, ((a, b), c_ab) in enumerate(cooc.items()):
        if cooc_support[(a, b)] < min_support:
            dropped_sparse += 1
            continue
        pmi = math.log((c_ab + 1e-6) * N / ((cnt[a] + 1e-6) * (cnt[b] + 1e-6)))
        # Preserve separation among high-PMI pairs with a slower saturation curve
        w = _weight_from_pmi(pmi, beta)
        pmi_values.append(pmi)
        edges.append((a, b, w))

        if idx < 5 or idx % 100 == 0:
            logger.info(
                "entity_cooccur_edge_detail a=%s b=%s c_ab=%.4f cnt_a=%.4f cnt_b=%.4f pmi=%.4f weight=%.4f",
                a, b, c_ab, cnt[a], cnt[b], pmi, w,
            )

    # Optionally keep only strongest edges
    if top_per_pair:
        edges.sort(key=lambda x: x[2], reverse=True)
        edges = edges[: top_per_pair]

    with db.cursor() as cur:
        cur.execute('DELETE FROM public.entity_cooccur WHERE "window" = %s', (int(window_days),))
        logger.info(
            "entity_cooccur_existing_cleared window_days=%s deleted=%s",
            window_days,
            cur.rowcount,
        )
        for idx, (a, b, w) in enumerate(edges, start=1):
            cur.execute(
                """
                INSERT INTO public.entity_cooccur(src_entity_id, dst_entity_id, "window", weight, updated_at)
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (src_entity_id, dst_entity_id, "window")
                DO UPDATE SET weight = EXCLUDED.weight, updated_at = now()
                """,
                (a, b, int(window_days), float(w)),
            )
            if idx % 500 == 0:
                logger.info("entity_cooccur_edge_progress inserted=%s", idx)
    logger.info(
        "entity_cooccur_complete window_days=%s clusters=%s pairs=%s edges=%s",
        window_days,
        len(cl_entities),
        total_pairs,
        len(edges),
    )
    if pmi_values:
        pmi_values.sort()
        logger.info(
            "entity_cooccur_pmi_stats window_days=%s min=%.4f median=%.4f p90=%.4f max=%.4f",
            window_days,
            pmi_values[0],
            statistics.median(pmi_values),
            _percentile(pmi_values, 0.9),
            pmi_values[-1],
        )
    if dropped_sparse:
        logger.info("entity_cooccur_sparse_pairs_dropped count=%s min_support=%s", dropped_sparse, min_support)
