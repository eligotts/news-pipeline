from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from pydantic import ValidationError

from ..core import (
    Database,
    canonicalize_url,
    compute_idempotency_key,
    content_signature,
    detect_language,
)
from .features import FeatureScheduler
from ..llm import EntityPipeline, CoordinateRanker
from ..schema import ArticlePayload

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    article_id: int
    duplicate: bool
    created: bool
    url_canon_exists: bool = False  # True if article existed by url_canon (no embedding job scheduled)


class IngestionProcessor:
    """Primary entry point for normalizing and persisting raw article payloads."""

    def __init__(
        self,
        db: Database,
        feature_scheduler: FeatureScheduler,
        entity_pipeline: EntityPipeline,
        coordinate_ranker: Optional[CoordinateRanker] = None,
    ):
        self.db = db
        self.features = feature_scheduler
        self.entities = entity_pipeline
        self.coord_ranker = coordinate_ranker

    def _ensure_publisher(self, source_name: Optional[str], url_canon: str) -> Optional[int]:
        u = urlparse(url_canon)
        domain = (u.hostname or "").lower()
        pub_id = None
        if source_name:
            row = self.db.query_one("SELECT id FROM public.publisher WHERE name = %s", (source_name,))
            if row:
                pub_id = row[0]
        if pub_id is None and domain:
            row = self.db.query_one("SELECT id FROM public.publisher WHERE domain = %s", (domain,))
            if row:
                pub_id = row[0]
        if pub_id is None:
            self.db.execute(
                "INSERT INTO public.publisher(name, domain) VALUES (%s, %s) ON CONFLICT(domain) DO UPDATE SET name = EXCLUDED.name RETURNING id",
                (source_name or domain or "unknown", domain or None),
            )
            row = self.db.query_one("SELECT id FROM public.publisher WHERE domain = %s OR name = %s ORDER BY id DESC LIMIT 1", (domain, source_name or domain))
            if row:
                pub_id = row[0]
        return pub_id

    def _parse_ts(self, published_at: str | None) -> datetime:
        if not published_at:
            return datetime.utcnow()
        try:
            return datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        except Exception as exc:
            raise ValueError(f"Invalid published_at timestamp: {published_at}") from exc

    @staticmethod
    def _extract_content(description: Optional[str], content: Optional[str]) -> Tuple[Optional[str], Optional[str], str]:
        """Return (lede, cleaned_body, body_source) using clean payload content only."""
        description = description or ""
        content = content or ""
        lede = description if description else (content[:400] if content else None)
        cleaned_body = content or None
        return lede, cleaned_body, "content"

    def process_article(self, message: Dict[str, Any]) -> IngestionResult:
        """Normalize payload, upsert article row, enqueue embeddings, link entities."""
        raw_article = message.get("article", message)
        try:
            art = ArticlePayload.parse_obj(raw_article)
        except ValidationError as exc:
            raise ValueError(f"Invalid payload: {exc}") from exc
        raw_url = str(art.url).strip()
        url_canon = canonicalize_url(raw_url)
        ts_pub = self._parse_ts(art.published_at)
        source_name = art.source_name or art.source
        publisher_id = self._ensure_publisher(source_name, url_canon)

        lede, cleaned_body, body_source = self._extract_content(art.description, art.content)
        title = art.title
        byline = art.author or None
        image_url = art.image_url or None
        category = art.category or None

        lang = art.lang or detect_language(" ".join(filter(None, [title, lede or "", cleaned_body or ""]))) or "en"
        region = art.region

        sig = content_signature(title, lede or "")
        idem_key = compute_idempotency_key(url_canon, ts_pub, title)

        # Rank article coordinates using LLM (if ranker available)
        x_coord = float(art.x) if art.x is not None else 0.0
        y_coord = float(art.y) if art.y is not None else 0.7
        x_explanation = None
        y_explanation = None
        summary = None

        existing = self.db.query_one("SELECT id FROM public.article WHERE url_canon = %s", (url_canon,))
        article_id: Optional[int] = existing[0] if existing else None
        was_created = False
        url_canon_existed = existing is not None

        if url_canon_existed:
            logger.info(
                "Article already exists by url_canon - will update only, no embedding job scheduled",
                extra={
                    "existing_article_id": article_id,
                    "url_canon": url_canon,
                    "title": title[:80] if title else None,
                }
            )

        # Duplicate suppression within publisher/time window
        window_start = ts_pub - timedelta(days=1)
        window_end = ts_pub + timedelta(days=1)
        duplicate = self.db.query_one(
            """
            SELECT id FROM public.article
            WHERE publisher_id = %s
              AND content_sig = %s
              AND ts_pub BETWEEN %s AND %s
            LIMIT 1
            """,
            (publisher_id, sig, window_start, window_end),
        ) if publisher_id is not None else None

        if duplicate and article_id is None:
            logger.info(
                "Article is content-sig duplicate within publisher/time window",
                extra={
                    "duplicate_article_id": duplicate[0],
                    "url_canon": url_canon,
                    "title": title[:80] if title else None,
                    "publisher_id": publisher_id,
                    "content_sig": sig,
                }
            )
            return IngestionResult(article_id=int(duplicate[0]), duplicate=True, created=False, url_canon_exists=False)

        if article_id is None and self.coord_ranker is not None and cleaned_body:
            try:
                x_coord, y_coord, x_explanation, y_explanation, summary = self.coord_ranker.rank_article(
                    title=title or "",
                    content=cleaned_body,
                    source=source_name,
                )
            except Exception as exc:
                logger.warning(f"Coordinate ranking failed for article '{title}': {exc}")

        meta = {k: v for k, v in {
            "lang_source": "payload" if art.lang else "detected",
            "region": region,
            "body_source": body_source,
            "x_explanation": x_explanation,
            "y_explanation": y_explanation,
            "summary": summary,
        }.items() if v is not None}

        if article_id is None:
            row = self.db.query_one(
                """
                INSERT INTO public.article(
                  url, url_canon, publisher_id, title, lede, body, image_url, byline,
                  ts_pub, lang, region, category, x, y, content_sig, idempotency_key, meta
                ) VALUES (
                  %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (url) DO UPDATE SET
                  title = EXCLUDED.title,
                  lede = EXCLUDED.lede,
                  image_url = EXCLUDED.image_url,
                  byline = EXCLUDED.byline,
                  ts_pub = EXCLUDED.ts_pub,
                  meta = public.article.meta || EXCLUDED.meta
                RETURNING id
                """,
                (
                    raw_url,
                    url_canon,
                    publisher_id,
                    title,
                    lede,
                    cleaned_body,
                    image_url,
                    byline,
                    ts_pub,
                    lang,
                    region,
                    category,
                    x_coord,
                    y_coord,
                    sig,
                    idem_key,
                    meta,
                ),
            )
            article_id = int(row[0])
            was_created = True

            self.features.schedule(article_id, title, lede, cleaned_body)

            self.entities.extract_and_link(
                self.db,
                article_id,
                title=title,
                lede=lede,
                body=cleaned_body,
            )

        else:
            self.db.execute(
                """
                UPDATE public.article SET
                  title = %s,
                  lede = %s,
                  body = %s,
                  image_url = %s,
                  byline = %s,
                  ts_pub = %s,
                  lang = %s,
                  region = %s,
                  meta = meta || %s::jsonb
                WHERE id = %s
                """,
                (
                    title,
                    lede,
                    cleaned_body,
                    image_url,
                    byline,
                    ts_pub,
                    lang,
                    region,
                    meta,
                    article_id,
                ),
            )

        self.db.execute("SELECT public.refresh_article_lex(%s)", (article_id,))

        if was_created:
            logger.debug(
                "New article created with embedding job scheduled",
                extra={
                    "article_id": article_id,
                    "url_canon": url_canon,
                    "title": title[:80] if title else None,
                }
            )

        return IngestionResult(article_id=article_id, duplicate=False, created=was_created, url_canon_exists=url_canon_existed)
