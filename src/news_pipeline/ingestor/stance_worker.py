from __future__ import annotations

import argparse
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence

import structlog
from dotenv import load_dotenv

from .config import get_settings
from .db import Database

from .openrouter_client import create_openrouter_client

logger = structlog.get_logger()


# JSON Schema for structured output
STANCE_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "stance_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "stance": {
                    "type": "string",
                    "enum": ["pro", "critical", "neutral"],
                    "description": "How the article portrays the entity"
                },
                "role": {
                    "type": "string",
                    "enum": ["subject", "actor", "target", "mentioned"],
                    "description": "The role of the entity in the article"
                },
                "centrality": {
                    "type": "number",
                    "description": "How central the entity is to the article, between 0 and 1"
                },
                "in_quote": {
                    "type": "boolean",
                    "description": "Whether the entity is quoted in the article"
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A single-item array with concise reasoning for the stance assessment"
                }
            },
            "required": ["stance", "role", "centrality", "in_quote", "evidence"],
            "additionalProperties": False
        }
    }
}


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


def _derive_salience(role: str, centrality: float, in_quote: bool) -> float:
    role_score = 1.0 if role in {"subject", "actor"} else 0.0
    quote_score = 1.0 if in_quote else 0.0
    score = 0.5 * centrality + 0.3 * role_score + 0.2 * quote_score
    return float(max(0.0, min(score, 1.0)))


class StanceWorker:
    def __init__(self, db: Database, client: Optional[Any], model: str, timeout: int, bypass_llm: bool = False) -> None:
        self.db = db
        self.client = client
        self.model = model
        self._timeout = timeout
        self.bypass_llm = bypass_llm

    def run_once(self, limit: int = 10) -> int:
        rows = self.db.query_all(
            """
            SELECT article_id, entity_id
            FROM public.article_entity_stance_job
            WHERE status = 'pending'
            ORDER BY queued_at ASC
            LIMIT %s
            """,
            (limit,),
        )
        jobs = [(int(article_id), int(entity_id)) for article_id, entity_id in rows]
        return self.process_jobs(jobs)

    def process_jobs(self, jobs: Sequence[tuple[int, int]]) -> int:
        processed = 0
        for article_id, entity_id in jobs:
            processed += self._process_single_job(article_id, entity_id)
        return processed

    def _process_single_job(self, article_id: int, entity_id: int) -> int:
        try:
            self.db.execute(
                "UPDATE public.article_entity_stance_job SET status = 'processing', attempts = attempts + 1 WHERE article_id = %s AND entity_id = %s",
                (article_id, entity_id),
            )
            payload = self._build_payload(article_id, entity_id)
            if payload is None:
                self.db.execute(
                    "DELETE FROM public.article_entity_stance_job WHERE article_id = %s AND entity_id = %s",
                    (article_id, entity_id),
                )
                self.db.commit()
                return 0
            if self.bypass_llm:
                result = self._get_default_result()
            else:
                result = self._call_llm(payload)
            self._persist(article_id, entity_id, result)
            self.db.execute(
                "DELETE FROM public.article_entity_stance_job WHERE article_id = %s AND entity_id = %s",
                (article_id, entity_id),
            )
            self.db.commit()
            return 1
        except Exception:
            self.db.rollback()
            raise

    def _build_payload(self, article_id: int, entity_id: int) -> Optional[Dict[str, Any]]:
        row = self.db.query_one(
            """
            SELECT a.title, a.lede, a.body, jsonb_build_object('x', a.x, 'y', a.y) AS coords,
                   e.name, e.type, e.description
            FROM public.article a
            JOIN public.article_entity ae ON ae.article_id = a.id
            JOIN public.entity e ON e.id = ae.entity_id
            WHERE a.id = %s AND e.id = %s
            """,
            (article_id, entity_id),
        )
        if not row:
            return None
        title, lede, body, coords, name, etype, description = row

        # Sanitize text fields to remove null bytes before sending to LLM
        title = _sanitize_text(title) if title else None
        lede = _sanitize_text(lede) if lede else None
        body = _sanitize_text(body) if body else None
        name = _sanitize_text(name) if name else name
        description = _sanitize_text(description) if description else description

        text_parts = [p for p in [title, lede, body] if p]
        article_text = "\n\n".join(text_parts)[:4000]
        return {
            "article_id": article_id,
            "entity_id": entity_id,
            "entity_name": name,
            "entity_type": etype,
            "entity_description": description,
            "coords": coords,
            "text": article_text,
        }

    def _get_default_result(self) -> Dict[str, Any]:
        """Return default values when bypassing LLM call."""
        return {
            "stance": "neutral",
            "role": "other",
            "centrality": 1.0,
            "in_quote": False,
            "evidence": [],
            "salience": 0.75,
        }

    def _call_llm(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You extract structured relations between news articles and entities. "
                    "Always respond with a json object matching this schema: "
                    "{\"stance\": one of ['pro','critical','neutral'], "
                    "\"role\": one of ['subject','actor','target','mentioned'], "
                    "\"centrality\": float between 0 and 1, \"in_quote\": boolean, "
                    "\"evidence\": an array of a single string, which is a concise (max 1 sentence) reasoning for the stance assessment}. "
                    "IMPORTANT: 'stance' measures how the article portrays the entity - "
                    "'pro' if portrayed positively/favorably, 'critical' if portrayed negatively/unfavorably, "
                    "'neutral' if portrayed objectively without judgment."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "task": "Extract how this article portrays the entity, using the required JSON schema.",
                    "entity": payload["entity_name"],
                    "entity_type": payload["entity_type"],
                    "entity_description": payload.get("entity_description"),
                    "article_excerpt": payload["text"],
                }),
            },
        ]
        try:
            response = self.client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=350,
                timeout=self._timeout,
                response_format=STANCE_RESPONSE_SCHEMA,
            )
            content = response.choices[0].message.content  # type: ignore[index]
            # Sanitize the LLM response before parsing to remove null bytes and invalid escapes
            if content:
                content = _sanitize_text(content)
            data = json.loads(content)
        except Exception as exc:
            raise RuntimeError(f"Stance LLM extraction failed for article {payload['article_id']} entity {payload['entity_id']}") from exc
        return data

    def _persist(self, article_id: int, entity_id: int, result: Dict[str, Any]) -> None:
        stance = result.get("stance", "neutral")
        role = result.get("role", "mentioned")
        centrality = float(result.get("centrality", 0.2))
        in_quote = bool(result.get("in_quote", False))
        evidence = result.get("evidence", [])

        # Sanitize evidence data to remove null bytes that PostgreSQL can't handle
        evidence = _sanitize_json_data(evidence)

        # If bypassing LLM, use the provided salience directly (default 0.75), otherwise calculate it
        if self.bypass_llm:
            salience = float(result.get("salience", 0.75))
        else:
            salience = _derive_salience(role, centrality, in_quote)
        self.db.execute(
            """
            UPDATE public.article_entity
            SET stance = %s,
                role = %s,
                centrality = %s,
                in_quote = %s,
                evidence = %s::jsonb,
                salience = %s
            WHERE article_id = %s AND entity_id = %s
            """,
            (stance, role, centrality, in_quote, json.dumps(evidence), salience, article_id, entity_id),
        )


def _pending_jobs(db: Database) -> int:
    row = db.query_one(
        "SELECT COUNT(*) FROM public.article_entity_stance_job WHERE status = 'pending'",
    )
    return int(row[0]) if row else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the stance extraction worker loop")
    parser.add_argument("--max-iterations", type=int, default=0, help="Stop after this many idle polling cycles (0 = run indefinitely)")
    parser.add_argument("--idle-sleep", type=float, default=5.0, help="Seconds to sleep between polls when no work is found")
    parser.add_argument("--bypass-llm", action="store_true", help="Bypass LLM calls and use default values")
    return parser


def main() -> None:
    load_dotenv()
    settings = get_settings()
    logging.basicConfig(level=logging.INFO)
    parser = _build_parser()
    args = parser.parse_args()
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))

    db = Database(settings.supabase_dsn)  # type: ignore[arg-type]
    db.connect()

    client = None
    if not args.bypass_llm and settings.openrouter_api_key:
        client = create_openrouter_client(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )

    worker = StanceWorker(db, client, settings.openrouter_model, settings.llm_timeout_seconds, bypass_llm=args.bypass_llm)

    iteration = 0
    try:
        while True:
            processed = worker.run_once()
            if processed == 0:
                iteration += 1
                if args.max_iterations and iteration >= args.max_iterations:
                    logger.info("stance_worker_idle_exit", iterations=iteration)
                    break
                pending = _pending_jobs(db)
                logger.info("stance_worker_idle", pending=pending)
                time.sleep(args.idle_sleep)
            else:
                logger.info("stance_batch_completed", processed=processed)
                db.commit()
                iteration = 0
    finally:
        db.close()


if __name__ == "__main__":
    main()
