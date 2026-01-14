from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .clustering import Clusterer
from .config import Settings
from .db import Database

_INVALID_UNICODE_ESCAPE_RE = re.compile(r"\\u(?![0-9a-fA-F]{4})")


def _sanitize_text(text: str) -> str:
    """Remove null bytes and other problematic Unicode characters."""
    if not isinstance(text, str):
        return text
    sanitized = text.replace('\x00', '').replace('\u0000', '')
    # Replace invalid unicode escape sequences like '\u1' with literal '\u1'
    sanitized = _INVALID_UNICODE_ESCAPE_RE.sub(r"\\u", sanitized)
    return sanitized


def _sanitize_json_data(data: Any) -> Any:
    """Recursively sanitize JSON data by removing null bytes from strings."""
    if isinstance(data, str):
        return _sanitize_text(data)
    elif isinstance(data, dict):
        return {k: _sanitize_json_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize_json_data(item) for item in data]
    else:
        return data

from .openrouter_client import create_openrouter_client

logger = logging.getLogger(__name__)


# JSON Schemas for structured output
CLUSTER_CONFIRM_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "cluster_confirmation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["confirmed", "split", "uncertain"],
                    "description": "Whether articles belong to the same event"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence level between 0 and 1"
                },
                "groups": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "description": "Article ID groups if splitting"
                },
                "reason": {
                    "type": "string",
                    "description": "Explanation for the decision"
                }
            },
            "required": ["decision", "confidence", "groups", "reason"],
            "additionalProperties": False
        }
    }
}

CLUSTER_SUMMARY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "cluster_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Neutral 2-3 sentence summary of the shared news event"
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key terms related to the event"
                }
            },
            "required": ["summary", "keywords"],
            "additionalProperties": False
        }
    }
}


@dataclass
class ClusterArticle:
    article_id: int
    title: str
    lede: Optional[str]
    publisher: Optional[str]
    ts_pub: Optional[datetime]
    rep_score: Optional[float]


class ClusterMaintenance:
    def __init__(self, db, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self.clusterer = Clusterer()
        self._llm_timeout = settings.llm_timeout_seconds
        self._llm_client = None
        if settings.openrouter_api_key:
            try:
                self._llm_client = create_openrouter_client(
                    api_key=settings.openrouter_api_key,
                    base_url=settings.openrouter_base_url,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("cluster_llm_init_failed", error=str(exc))
                self._llm_client = None

    # ------------------------------------------------------------------
    # Cluster materialization + rep scores
    # ------------------------------------------------------------------
    def refresh_materialization(self, recency_hours: int = 168, cluster_ids: Optional[Sequence[int]] = None) -> int:
        if cluster_ids is None:
            rows = self.db.query_all(
                f"""
                SELECT id
                FROM public.cluster
                WHERE ts_end IS NULL OR ts_end >= now() - interval '{recency_hours} hours'
                """,
            )
            cluster_ids = [int(row[0]) for row in rows]
        refreshed = 0
        for cid in cluster_ids:
            try:
                self.clusterer._refresh_cluster_materialization(self.db, cid)  # type: ignore[attr-defined]
                refreshed += 1
            except Exception as exc:
                logger.warning("cluster_refresh_failed cluster_id=%s error=%s", cid, str(exc))
        if cluster_ids:
            self._update_rep_scores(cluster_ids)
        return refreshed

    def _update_rep_scores(self, cluster_ids: Sequence[int]) -> None:
        cluster_filter = ""
        params: List[Any] = []
        if cluster_ids:
            cluster_filter = "WHERE ac.cluster_id = ANY(%s)"
            params.append(list(cluster_ids))
        sql = f"""
        WITH publisher_counts AS (
            SELECT ac.cluster_id, a.publisher_id, COUNT(*) AS cnt
            FROM public.article_cluster ac
            JOIN public.article a ON a.id = ac.article_id
            GROUP BY ac.cluster_id, a.publisher_id
        ),
        score_calc AS (
            SELECT
                ac.article_id,
                ac.cluster_id,
                0.60 * (CASE WHEN c.centroid_vec IS NULL OR a.v_title IS NULL THEN 0.8 ELSE (1 - (a.v_title <=> c.centroid_vec)) END)
                + 0.20 * exp(-extract(epoch FROM (now() - a.ts_pub)) / 86400)
                + 0.15 * LEAST(GREATEST(a.y, 0), 1)
                + 0.05 * (
                    CASE WHEN pc.cnt IS NULL THEN 1
                         ELSE 1 - LEAST(pc.cnt / 3.0, 1)
                    END
                  ) AS computed_score
            FROM public.article_cluster ac
            JOIN public.article a ON a.id = ac.article_id
            LEFT JOIN public.cluster c ON c.id = ac.cluster_id
            LEFT JOIN publisher_counts pc ON pc.cluster_id = ac.cluster_id AND pc.publisher_id = a.publisher_id
            {cluster_filter}
        )
        UPDATE public.article_cluster
        SET rep_score = score_calc.computed_score
        FROM score_calc
        WHERE article_cluster.article_id = score_calc.article_id
          AND article_cluster.cluster_id = score_calc.cluster_id
        """
        self.db.execute(sql, tuple(params) if params else None)

    # ------------------------------------------------------------------
    # Cluster confirmation via LLM
    # ------------------------------------------------------------------
    def confirm_clusters(self, recency_hours: int = 24, limit: int = 50) -> int:
        rows = self.db.query_all(
            f"""
            SELECT id
            FROM public.cluster
            WHERE ts_end IS NOT NULL
              AND ts_end >= now() - interval '{recency_hours} hours'
              AND (meta->>'confirmation_status') IS DISTINCT FROM 'confirmed'
            ORDER BY ts_end DESC
            LIMIT %s
            """,
            (limit,),
        )
        processed = 0
        refresh_targets: List[int] = []
        for (cluster_id,) in rows:
            articles = self._load_cluster_articles(int(cluster_id), limit=5)
            if len(articles) < 2:
                self._mark_cluster_meta(cluster_id, {
                    "confirmation_status": "confirmed",
                    "confirmation_reason": "singleton",
                    "confirmation_ts": datetime.utcnow().isoformat(),
                })
                continue
            decision = self._llm_confirm(cluster_id=int(cluster_id), articles=articles)
            new_ids = self._apply_confirmation(int(cluster_id), articles, decision)
            refresh_targets.append(int(cluster_id))
            refresh_targets.extend(new_ids)
            processed += 1
        if refresh_targets:
            unique_ids = list({cid for cid in refresh_targets if cid is not None})
            self.refresh_materialization(cluster_ids=unique_ids)
        return processed

    def _load_cluster_articles(self, cluster_id: int, limit: int = 5) -> List[ClusterArticle]:
        rows = self.db.query_all(
            """
            SELECT a.id, a.title, a.lede, p.name, a.ts_pub, ac.rep_score
            FROM public.article a
            JOIN public.article_cluster ac ON ac.article_id = a.id
            LEFT JOIN public.publisher p ON p.id = a.publisher_id
            WHERE ac.cluster_id = %s
            ORDER BY ac.rep_score DESC NULLS LAST, a.ts_pub DESC
            LIMIT %s
            """,
            (cluster_id, limit),
        )
        return [
            ClusterArticle(
                article_id=int(aid),
                title=title or "",
                lede=lede,
                publisher=publisher,
                ts_pub=ts_pub,
                rep_score=float(rep_score) if rep_score is not None else None,
            )
            for aid, title, lede, publisher, ts_pub, rep_score in rows
        ]

    def _llm_confirm(self, cluster_id: int, articles: List[ClusterArticle]) -> Dict[str, Any]:
        if self._llm_client is None:
            return {"decision": "confirmed", "confidence": 0.1}
        payload = {
            "cluster_id": cluster_id,
            "articles": [
                {
                    "id": art.article_id,
                    "title": art.title,
                    "lede": art.lede,
                    "publisher": art.publisher,
                    "ts_pub": art.ts_pub.isoformat() if art.ts_pub else None,
                }
                for art in articles
            ],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You verify if grouped news articles describe the same real-world event. "
                    "Respond with JSON: {\"decision\": 'confirmed'|'split'|'uncertain', \"confidence\": float, \"groups\": [[article_ids...]], \"reason\": str}."
                ),
            },
            {"role": "user", "content": json.dumps(payload)},
        ]
        try:
            resp = self._llm_client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.settings.openrouter_model,
                messages=messages,
                temperature=0.1,
                max_tokens=400,
                timeout=self._llm_timeout,
                response_format=CLUSTER_CONFIRM_SCHEMA,
            )
            content = resp.choices[0].message.content  # type: ignore[index]
            data = json.loads(content or "{}")
            if not isinstance(data, dict):
                raise ValueError("invalid response")
            return data
        except Exception as exc:
            logger.warning("cluster_llm_failure cluster_id=%s error=%s", cluster_id, str(exc))
            return {"decision": "confirmed", "confidence": 0.0, "groups": [], "reason": "llm_fallback"}

    def _apply_confirmation(self, cluster_id: int, articles: List[ClusterArticle], decision: Dict[str, Any]) -> None:
        status = decision.get("decision", "confirmed")
        confidence = float(decision.get("confidence", 0.0))
        reason = decision.get("reason")
        meta_update = {
            "confirmation_status": status,
            "confirmation_confidence": confidence,
            "confirmation_reason": reason,
            "confirmation_ts": datetime.utcnow().isoformat(),
        }
        new_cluster_ids: List[int] = []
        if status == "split":
            groups = decision.get("groups")
            if isinstance(groups, list) and groups:
                new_cluster_ids = self._split_cluster(cluster_id, groups)
        self._mark_cluster_meta(cluster_id, meta_update)
        return new_cluster_ids

    def _split_cluster(self, cluster_id: int, groups: Sequence[Sequence[int]]) -> List[int]:
        keep_ids = set(int(x) for x in (groups[0] if groups else []))
        all_articles = {art.article_id for art in self._load_cluster_articles(cluster_id, limit=50)}
        if keep_ids:
            self.db.execute(
                "UPDATE public.article_cluster SET cluster_id = %s WHERE article_id = ANY(%s)",
                (cluster_id, list(keep_ids)),
            )
        assigned = set(keep_ids)
        new_cluster_ids: List[int] = []
        for article_group in groups[1:]:
            ids = [int(x) for x in article_group]
            if not ids:
                continue
            new_cluster_id = self.db.query_one(
                "INSERT INTO public.cluster(meta) VALUES ('{}'::jsonb) RETURNING id",
            )[0]
            self.db.execute(
                "UPDATE public.article_cluster SET cluster_id = %s WHERE article_id = ANY(%s)",
                (new_cluster_id, ids),
            )
            assigned.update(ids)
            new_cluster_ids.append(int(new_cluster_id))
        leftovers = list(all_articles - assigned)
        if leftovers:
            self.db.execute(
                "UPDATE public.article_cluster SET cluster_id = %s WHERE article_id = ANY(%s)",
                (cluster_id, leftovers),
            )
        return new_cluster_ids

    def _mark_cluster_meta(self, cluster_id: int, meta_fragment: Dict[str, Any]) -> None:
        self.db.execute(
            """
            UPDATE public.cluster
            SET meta = COALESCE(meta, '{}'::jsonb) || %s::jsonb
            WHERE id = %s
            """,
            (json.dumps(meta_fragment), cluster_id),
        )

    # ------------------------------------------------------------------
    # Cluster summaries (LLM)
    # ------------------------------------------------------------------
    def generate_summaries(self, lookback_hours: int = 48, limit: int | None = None, max_workers: int = 3) -> int:
        sql = """
            SELECT id
            FROM public.cluster
            WHERE (summary IS NULL OR summary_updated_at IS NULL OR summary_updated_at <= now() - interval '6 hours')
              AND ts_end IS NOT NULL
              AND ts_end >= now() - (%s * interval '1 hour')
              AND size > 1
            ORDER BY ts_end DESC
        """
        params: List[Any] = [lookback_hours]
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
        
        rows = self.db.query_all(sql, tuple(params))
        cluster_ids = [int(row[0]) for row in rows]
        
        if not cluster_ids:
            return 0
        
        # Process clusters in parallel
        updated = 0
        workers = max(1, min(max_workers, len(cluster_ids)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._process_summary_cluster, cluster_id): cluster_id
                for cluster_id in cluster_ids
            }
            for future in as_completed(futures):
                cluster_id = futures[future]
                try:
                    if future.result():
                        updated += 1
                except Exception as exc:
                    logger.warning("cluster_summary_processing_failed cluster_id=%s error=%s", cluster_id, str(exc))
        
        return updated

    def _process_summary_cluster(self, cluster_id: int) -> bool:
        """Process a single cluster summary. Creates its own DB connection for thread safety."""
        # Get DSN from the original database connection
        dsn = self.db._dsn if hasattr(self.db, '_dsn') else self.settings.supabase_dsn
        if not dsn:
            raise RuntimeError("Database DSN not available")
        db = Database(dsn)
        db.connect()
        try:
            # Load cluster articles
            articles = self._load_cluster_articles_with_db(db, cluster_id, limit=5)
            entities = db.query_all(
                """
                SELECT e.name
                FROM public.article_entity ae
                JOIN public.entity e ON e.id = ae.entity_id
                JOIN public.article_cluster ac ON ac.article_id = ae.article_id
                WHERE ac.cluster_id = %s
                GROUP BY e.name
                ORDER BY MAX(ae.salience) DESC
                LIMIT 5
                """,
                (cluster_id,),
            )
            entity_names = [name for (name,) in entities]
            
            # Generate summary using LLM
            summary = self._llm_summary(cluster_id=int(cluster_id), articles=articles, entities=entity_names)
            
            # Sanitize summary text and keywords to remove null bytes
            summary_text = _sanitize_text(summary.get("summary", "") or "")
            keywords = summary.get("keywords", [])
            keywords = _sanitize_json_data(keywords)
            
            # Update database
            db.execute(
                """
                UPDATE public.cluster
                SET summary = %s,
                    summary_updated_at = now(),
                    meta = COALESCE(meta, '{}'::jsonb) || %s::jsonb
                WHERE id = %s
                """,
                (
                    summary_text,
                    json.dumps({"summary_keywords": keywords}),
                    cluster_id,
                ),
            )
            db.commit()
            return True
        except Exception as exc:
            db.rollback()
            logger.warning("cluster_summary_failed cluster_id=%s error=%s", cluster_id, str(exc))
            return False
        finally:
            db.close()

    def _load_cluster_articles_with_db(self, db: Database, cluster_id: int, limit: int = 5) -> List[ClusterArticle]:
        """Load cluster articles using a specific database connection."""
        rows = db.query_all(
            """
            SELECT a.id, a.title, a.lede, p.name, a.ts_pub, ac.rep_score
            FROM public.article a
            JOIN public.article_cluster ac ON ac.article_id = a.id
            LEFT JOIN public.publisher p ON p.id = a.publisher_id
            WHERE ac.cluster_id = %s
            ORDER BY ac.rep_score DESC NULLS LAST, a.ts_pub DESC
            LIMIT %s
            """,
            (cluster_id, limit),
        )
        return [
            ClusterArticle(
                article_id=int(aid),
                title=title or "",
                lede=lede,
                publisher=publisher,
                ts_pub=ts_pub,
                rep_score=float(rep_score) if rep_score is not None else None,
            )
            for aid, title, lede, publisher, ts_pub, rep_score in rows
        ]

    def _llm_summary(self, cluster_id: int, articles: List[ClusterArticle], entities: List[str]) -> Dict[str, Any]:
        context = {
            "cluster_id": cluster_id,
            "entities": entities,
            "articles": [
                {
                    "title": art.title,
                    "lede": art.lede,
                    "publisher": art.publisher,
                    "ts_pub": art.ts_pub.isoformat() if art.ts_pub else None,
                }
                for art in articles
            ],
        }
        if self._llm_client is None:
            headline = articles[0].title if articles else ""
            return {
                "summary": headline[:512],
                "keywords": entities[:5],
            }
        messages = [
            {
                "role": "system",
                "content": (
                    "Write a neutral 2-3 sentence summary of the shared news event described. "
                    "Return JSON: {\"summary\": str, \"keywords\": [str,...]}."
                ),
            },
            {"role": "user", "content": json.dumps(context)},
        ]
        try:
            resp = self._llm_client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.settings.openrouter_model,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
                timeout=self._llm_timeout,
                response_format=CLUSTER_SUMMARY_SCHEMA,
            )
            content = resp.choices[0].message.content  # type: ignore[index]
            # Sanitize the LLM response before parsing to remove null bytes
            if content:
                content = _sanitize_text(content)
            data = json.loads(content or "{}")
            if not isinstance(data, dict):
                raise ValueError("invalid response")
            return data
        except Exception as exc:
            logger.warning("cluster_summary_llm_failure cluster_id=%s error=%s", cluster_id, str(exc))
            headline = articles[0].title if articles else ""
            return {
                "summary": headline[:512],
                "keywords": entities[:5],
            }
