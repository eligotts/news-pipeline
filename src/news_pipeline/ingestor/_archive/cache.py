from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from ..config import Settings

try:  # pragma: no cover - optional dependency
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


class CacheManager:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._ttl = settings.cache_ttl_seconds
        self._redis = None
        self._local: Dict[str, tuple[float, str]] = {}
        if settings.redis_url and redis is not None:
            self._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)  # type: ignore[arg-type]

    def _serialize(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    def _deserialize(self, payload: Optional[str]) -> Optional[Any]:
        if payload is None:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def get(self, key: str) -> Optional[Any]:
        if self._redis is not None:
            payload = self._redis.get(key)
            return self._deserialize(payload)
        entry = self._local.get(key)
        if not entry:
            return None
        expires_at, payload = entry
        if expires_at < time.time():
            self._local.pop(key, None)
            return None
        return self._deserialize(payload)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl_seconds = ttl or self._ttl
        payload = self._serialize(value)
        if self._redis is not None:
            self._redis.set(key, payload, ex=ttl_seconds)
            return
        self._local[key] = (time.time() + ttl_seconds, payload)

    def invalidate(self, prefix: str) -> None:
        if self._redis is not None:
            pattern = f"{prefix}*"
            for key in self._redis.scan_iter(match=pattern):  # type: ignore[attr-defined]
                self._redis.delete(key)
            return
        for key in list(self._local.keys()):
            if key.startswith(prefix):
                self._local.pop(key, None)


def cache_hot_entities(
    db,
    cache: CacheManager,
    lookback_hours: int = 72,
    limit: int = 1000,
    per_entity_articles: int = 15,
    ttl_seconds: Optional[int] = None,
) -> int:
    rows = db.query_all(
        """
        SELECT ae.entity_id, COUNT(*) AS cnt
        FROM public.article_entity ae
        JOIN public.article a ON a.id = ae.article_id
        WHERE a.ts_pub >= now() - interval %s
        GROUP BY ae.entity_id
        ORDER BY cnt DESC
        LIMIT %s
        """,
        (f"{lookback_hours} hours", limit),
    )
    cached = 0
    for entity_id, _ in rows:
        data = db.query_all(
            """
            SELECT ac.cluster_id, ac.rep_score, a.id, a.title, a.url, a.ts_pub
            FROM public.article_entity ae
            JOIN public.article a ON a.id = ae.article_id
            JOIN public.article_cluster ac ON ac.article_id = a.id
            WHERE ae.entity_id = %s
              AND a.ts_pub >= now() - interval %s
            ORDER BY ac.rep_score DESC NULLS LAST
            LIMIT %s
            """,
            (entity_id, f"{lookback_hours} hours", per_entity_articles),
        )
        payload = [
            {
                "cluster_id": int(cluster_id),
                "rep_score": float(rep_score) if rep_score is not None else None,
                "article_id": int(article_id),
                "title": title,
                "url": url,
                "ts_pub": ts_pub.isoformat() if ts_pub else None,
            }
            for cluster_id, rep_score, article_id, title, url, ts_pub in data
        ]
        cache.set(f"entity:{int(entity_id)}", payload, ttl_seconds)
        cached += 1
    return cached


def cache_hot_topics(
    db,
    cache: CacheManager,
    lookback_hours: int = 72,
    limit: int = 500,
    per_topic_clusters: int = 10,
    ttl_seconds: Optional[int] = None,
) -> int:
    rows = db.query_all(
        """
        SELECT ct.topic_id, COUNT(*) AS cnt
        FROM public.cluster_topic ct
        JOIN public.cluster c ON c.id = ct.cluster_id
        WHERE c.ts_end IS NOT NULL
          AND c.ts_end >= now() - interval %s
        GROUP BY ct.topic_id
        ORDER BY cnt DESC
        LIMIT %s
        """,
        (f"{lookback_hours} hours", limit),
    )
    cached = 0
    for topic_id, _ in rows:
        data = db.query_all(
            """
            SELECT ct.cluster_id, ct.weight, c.summary, c.top_headlines
            FROM public.cluster_topic ct
            JOIN public.cluster c ON c.id = ct.cluster_id
            WHERE ct.topic_id = %s
            ORDER BY ct.weight DESC
            LIMIT %s
            """,
            (topic_id, per_topic_clusters),
        )
        payload = [
            {
                "cluster_id": int(cluster_id),
                "weight": float(weight),
                "summary": summary,
                "top_headlines": headlines,
            }
            for cluster_id, weight, summary, headlines in data
        ]
        cache.set(f"topic:{int(topic_id)}", payload, ttl_seconds)
        cached += 1
    return cached
