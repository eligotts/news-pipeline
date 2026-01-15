from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..core import sanitize_text

logger = logging.getLogger(__name__)


ALLOWED_TYPES = {"person", "org", "place", "other"}


# JSON Schemas for structured output
ENTITY_EXTRACT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "entity_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Short canonical label for the entity"
                            },
                            "description": {
                                "type": "string",
                                "description": "Max two sentences describing the entity"
                            },
                            "type": {
                                "type": "string",
                                "enum": ["person", "org", "place", "other"],
                                "description": "Entity type classification"
                            }
                        },
                        "required": ["name", "description", "type"],
                        "additionalProperties": False
                    },
                    "description": "Up to three distinct entities ordered by importance"
                }
            },
            "required": ["entities"],
            "additionalProperties": False
        }
    }
}

ENTITY_RESOLVE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "entity_resolution",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entity_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Array of candidate IDs that match, or -1 for no match"
                }
            },
            "required": ["entity_ids"],
            "additionalProperties": False
        }
    }
}


def _vec_literal(vec: Sequence[float], length: int) -> str:
    values = list(float(x) for x in vec[:length])
    if len(values) < length:
        values.extend([0.0] * (length - len(values)))
    return "[" + ",".join(f"{x:.6f}" for x in values) + "]"


def _normalise_type(raw: str | None) -> str:
    if not raw:
        return "other"
    value = raw.strip().lower()
    if value in ALLOWED_TYPES:
        return value
    if value in {"person", "people", "individual"}:
        return "person"
    if value in {"organization", "organisation", "company", "corp"}:
        return "org"
    if value in {"gpe", "location", "loc", "country", "city", "place"}:
        return "place"
    return "other"


def _trim_description(text: str, max_len: int = 400) -> str:
    text = sanitize_text(text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


@dataclass
class ExistingEntity:
    entity_id: int
    name: str
    description: str
    etype: str


@dataclass
class ProposedEntity:
    name: str
    description: str
    etype: str
    vector: List[float] = field(default_factory=list)
    candidates: List[ExistingEntity] = field(default_factory=list)


class EntityPipeline:
    def __init__(
        self,
        llm_client: Any | None,
        llm_model: str,
        embedder: Any | None,
        *,
        vector_dimensions: int = 768,
        top_entities: int = 3,
        neighbor_limit: int = 5,
        llm_timeout_seconds: Optional[int] = None,
    ) -> None:
        self._client = llm_client
        self._model = llm_model
        self._embedder = embedder
        self._dims = vector_dimensions
        self._top_entities = top_entities
        self._neighbor_limit = neighbor_limit
        self._timeout = llm_timeout_seconds

    def extract_and_link(self, db, article_id: int, title: str, lede: str | None, body: str | None) -> None:
        if self._client is None or self._embedder is None:
            logger.warning("entity_pipeline_dependencies_missing")
            return

        text_parts = [p.strip() for p in [title or "", lede or "", body or ""] if p]
        article_text = "\n\n".join(text_parts)
        if not article_text:
            return

        try:
            proposals_raw = self._extract_entities(article_text)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("entity_extract_failed: %s", exc)
            return

        if not proposals_raw:
            return

        descriptions = [p.description or p.name for p in proposals_raw]
        try:
            vectors = self._embedder.embed(descriptions, dimensions=self._dims)
        except Exception as exc:  # pragma: no cover - embed API failure
            logger.error("entity_description_embed_failed: %s", exc)
            return

        if len(vectors) != len(proposals_raw):
            logger.error(
                "entity_embed_length_mismatch expected=%s received=%s",
                len(proposals_raw),
                len(vectors),
            )
            return

        proposals: List[ProposedEntity] = []

        with db.cursor() as cur:  # type: ignore[attr-defined]
            for raw, vec in zip(proposals_raw, vectors):
                vec_list = [float(x) for x in vec[: self._dims]]
                candidates = self._fetch_candidates(cur, vec_list)
                proposals.append(
                    ProposedEntity(
                        name=raw.name,
                        description=raw.description,
                        etype=raw.etype,
                        vector=vec_list,
                        candidates=candidates,
                    )
                )

            decisions = self._resolve_matches(proposals)

            for idx, proposal in enumerate(proposals):
                match_id = decisions[idx] if idx < len(decisions) else -1
                if match_id in {cand.entity_id for cand in proposal.candidates}:
                    entity_id = match_id
                    self._ensure_alias(cur, entity_id, proposal.name)
                    if proposal.description:
                        self._maybe_update_description(cur, entity_id, proposal)
                else:
                    entity_id = self._insert_entity(cur, proposal)
                self._link_article(cur, article_id, entity_id)

    def _extract_entities(self, text: str) -> List[ProposedEntity]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You identify the three most salient real-world entities mentioned in news articles. "
                    "Always respond as a JSON object with an `entities` array. Each entity must include "
                    "`name` (short canonical label), `description` (max two sentences-think about a canonical description for this entity) and `type` "
                    "classified as one of ['person','org','place','other']."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "Extract up to three distinct entities ordered by importance.",
                        "article_text": text[:6000],
                    }
                ),
            },
        ]

        response = self._client.chat.completions.create(  # type: ignore[attr-defined]
            model=self._model,
            messages=messages,
            temperature=0,
            max_tokens=400,
            timeout=self._timeout,
            response_format=ENTITY_EXTRACT_SCHEMA,
        )
        content = response.choices[0].message.content  # type: ignore[index]
        data = json.loads(content or "{}")
        items = data.get("entities", []) if isinstance(data, dict) else []

        proposals: List[ProposedEntity] = []
        for item in items[: self._top_entities]:
            if not isinstance(item, dict):
                continue
            name = sanitize_text(item.get("name") or "").strip()
            description = _trim_description(item.get("description") or name)
            if not name or not description:
                continue
            etype = _normalise_type(item.get("type"))
            proposals.append(ProposedEntity(name=name, description=description, etype=etype))
        return proposals

    def _fetch_candidates(self, cur, vector: Sequence[float]) -> List[ExistingEntity]:
        literal = _vec_literal(vector, self._dims)
        cur.execute(
            """
            SELECT id, name, COALESCE(description, ''), type
            FROM public.entity
            WHERE v_description IS NOT NULL
            ORDER BY v_description <-> %s::vector
            LIMIT %s
            """,
            (literal, self._neighbor_limit),
        )
        rows = cur.fetchall() or []
        out: List[ExistingEntity] = []
        for row in rows:
            try:
                ent_id, name, description, etype = row
                out.append(
                    ExistingEntity(
                        entity_id=int(ent_id),
                        name=str(name or ""),
                        description=str(description or ""),
                        etype=_normalise_type(str(etype or "")),
                    )
                )
            except Exception:  # pragma: no cover - defensive
                continue
        return out

    def _resolve_matches(self, proposals: List[ProposedEntity]) -> List[int]:
        if not proposals:
            return []
        if self._client is None:
            return [-1] * len(proposals)

        payload = {
            "entities": []
        }
        for idx, proposal in enumerate(proposals):
            payload["entities"].append(
                {
                    "index": idx,
                    "proposed": {
                        "name": proposal.name,
                        "description": proposal.description,
                        "type": proposal.etype,
                    },
                    "candidates": [
                        {
                            "id": cand.entity_id,
                            "name": cand.name,
                            "description": cand.description,
                            "type": cand.etype,
                        }
                        for cand in proposal.candidates
                    ],
                }
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You reconcile proposed entities against an existing catalog. "
                    "Return JSON with a `entity_ids` array of integers, one per proposal in order. "
                    "Use the candidate id when it clearly refers to the same entity; otherwise respond with -1."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload),
            },
        ]

        try:
            response = self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=self._model,
                messages=messages,
                temperature=0,
                max_tokens=200,
                timeout=self._timeout,
                response_format=ENTITY_RESOLVE_SCHEMA,
            )
            content = response.choices[0].message.content  # type: ignore[index]
            data = json.loads(content or "{}")
        except Exception as exc:  # pragma: no cover - LLM fallback
            logger.warning("entity_resolution_failed: %s", exc)
            return [-1] * len(proposals)

        ids = data.get("entity_ids", []) if isinstance(data, dict) else []
        if not isinstance(ids, list):
            return [-1] * len(proposals)
        out: List[int] = []
        for val in ids[: len(proposals)]:
            try:
                out.append(int(val))
            except Exception:
                out.append(-1)
        if len(out) < len(proposals):
            out.extend([-1] * (len(proposals) - len(out)))
        return out

    def _insert_entity(self, cur, proposal: ProposedEntity) -> int:
        literal = _vec_literal(proposal.vector, self._dims)
        sanitized_name = sanitize_text(proposal.name)
        sanitized_desc = sanitize_text(proposal.description)
        cur.execute(
            """
            INSERT INTO public.entity (name, type, description, v_description)
            VALUES (%s, %s, %s, %s::vector)
            RETURNING id
            """,
            (sanitized_name, proposal.etype, sanitized_desc, literal),
        )
        entity_id = int(cur.fetchone()[0])
        self._ensure_alias(cur, entity_id, proposal.name)
        return entity_id

    def _maybe_update_description(self, cur, entity_id: int, proposal: ProposedEntity) -> None:
        cur.execute("SELECT description FROM public.entity WHERE id = %s", (entity_id,))
        row = cur.fetchone()
        current_desc = (row[0] or "") if row else ""
        if current_desc:
            return
        literal = _vec_literal(proposal.vector, self._dims)
        sanitized_desc = sanitize_text(proposal.description)
        cur.execute(
            """
            UPDATE public.entity
            SET description = %s,
                v_description = %s::vector
            WHERE id = %s
            """,
            (sanitized_desc, literal, entity_id),
        )

    def _ensure_alias(self, cur, entity_id: int, alias: str) -> None:
        name = sanitize_text(alias).strip()
        if not name:
            return
        cur.execute(
            """
            INSERT INTO public.entity_alias(entity_id, alias)
            VALUES (%s, %s)
            ON CONFLICT (entity_id, alias) DO NOTHING
            """,
            (entity_id, name),
        )

    def _link_article(self, cur, article_id: int, entity_id: int) -> None:
        cur.execute(
            """
            INSERT INTO public.article_entity(article_id, entity_id)
            VALUES (%s, %s)
            ON CONFLICT (article_id, entity_id) DO NOTHING
            """,
            (article_id, entity_id),
        )
        cur.execute(
            """
            INSERT INTO public.article_entity_stance_job(article_id, entity_id)
            VALUES (%s, %s)
            ON CONFLICT (article_id, entity_id) DO NOTHING
            """,
            (article_id, entity_id),
        )
