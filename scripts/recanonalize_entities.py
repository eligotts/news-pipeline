#!/usr/bin/env python3
"""
Re-canonicalize all entities using the improved canonicalize_entity_name(),
then merge any new duplicates that surface (e.g., "U.S." and "United States"
now both → "united states").

Run AFTER migrations/0002_entity_dedup.sql and BEFORE adding the UNIQUE index.

Usage:
    uv run python scripts/recanonalize_entities.py
    uv run python scripts/recanonalize_entities.py --dry-run
    uv run python scripts/recanonalize_entities.py --batch-size 500
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Re-canonicalize entities and merge duplicates")
    parser.add_argument("--batch-size", type=int, default=1000, help="Commit every N updates")
    parser.add_argument("--dry-run", action="store_true", help="Don't commit changes")
    args = parser.parse_args()

    load_dotenv()

    from news_pipeline.ingestor.core import Database
    from news_pipeline.ingestor.config import get_settings
    from news_pipeline.ingestor.llm.entities import canonicalize_entity_name

    settings = get_settings()
    db = Database(settings.supabase_dsn)
    db.connect()

    try:
        _recanonalize(db, args, canonicalize_entity_name)
        _merge_duplicates(db, args)
    finally:
        db.close()


def _recanonalize(db, args, canonicalize_fn):
    """Phase 1: Update name_canonical for every entity using the improved function."""
    logger.info("Phase 1: Re-canonicalizing all entities...")

    entities = db.query_all("SELECT id, name, name_canonical FROM public.entity ORDER BY id")
    logger.info(f"Found {len(entities)} entities to process")

    updated = 0
    unchanged = 0

    for i, (entity_id, name, old_canonical) in enumerate(entities):
        new_canonical = canonicalize_fn(name or "")

        if new_canonical != (old_canonical or ""):
            db.execute(
                "UPDATE public.entity SET name_canonical = %s WHERE id = %s",
                (new_canonical, entity_id),
            )
            updated += 1

            if updated <= 20:
                logger.info(f"  Updated entity {entity_id}: {old_canonical!r} -> {new_canonical!r} (name={name!r})")
        else:
            unchanged += 1

        if (i + 1) % args.batch_size == 0:
            if not args.dry_run:
                db.commit()
                logger.info(f"  Committed batch at {i + 1}/{len(entities)}")
            else:
                db.rollback()

    if not args.dry_run:
        db.commit()
    else:
        db.rollback()

    logger.info(f"Phase 1 complete: {updated} updated, {unchanged} unchanged")


def _merge_duplicates(db, args):
    """Phase 2: Merge entities that now share the same name_canonical."""
    logger.info("Phase 2: Merging newly-surfaced duplicates...")

    dupes = db.query_all("""
        SELECT name_canonical, COUNT(*) as cnt,
               array_agg(id ORDER BY id) as ids
        FROM public.entity
        WHERE name_canonical IS NOT NULL AND name_canonical != ''
        GROUP BY name_canonical
        HAVING COUNT(*) > 1
        ORDER BY cnt DESC
    """)

    if not dupes:
        logger.info("No duplicates found — nothing to merge")
        return

    logger.info(f"Found {len(dupes)} canonical names with duplicates")

    total_merged = 0

    for canonical, count, ids in dupes:
        winner_id = ids[0]  # Lowest id = oldest
        loser_ids = ids[1:]
        names = db.query_all(
            "SELECT id, name FROM public.entity WHERE id = ANY(%s) ORDER BY id",
            (ids,),
        )
        name_str = ", ".join(f"{eid}:{n!r}" for eid, n in names)
        logger.info(f"  Merging {canonical!r} ({count} entities): {name_str} -> winner={winner_id}")

        for loser_id in loser_ids:
            _redirect_entity(db, loser_id, winner_id)
            total_merged += 1

        if not args.dry_run:
            db.commit()
        else:
            db.rollback()

    if not args.dry_run:
        db.commit()
    else:
        db.rollback()

    logger.info(f"Phase 2 complete: merged {total_merged} duplicate entities")


def _redirect_entity(db, old_id: int, new_id: int):
    """Redirect all references from old_id to new_id, then delete old_id."""

    # article_entity: delete conflicts, then redirect
    db.execute("""
        DELETE FROM public.article_entity ae
        WHERE ae.entity_id = %s
          AND EXISTS (
              SELECT 1 FROM public.article_entity ae2
              WHERE ae2.article_id = ae.article_id
                AND ae2.entity_id = %s
          )
    """, (old_id, new_id))

    db.execute("""
        UPDATE public.article_entity
        SET entity_id = %s
        WHERE entity_id = %s
    """, (new_id, old_id))

    # entity_alias: delete conflicts, then redirect
    db.execute("""
        DELETE FROM public.entity_alias ea
        WHERE ea.entity_id = %s
          AND EXISTS (
              SELECT 1 FROM public.entity_alias ea2
              WHERE ea2.entity_id = %s
                AND ea2.alias = ea.alias
          )
    """, (old_id, new_id))

    db.execute("""
        UPDATE public.entity_alias
        SET entity_id = %s
        WHERE entity_id = %s
    """, (new_id, old_id))

    # entity_edge: remove self-loops and conflicts, then redirect
    # Delete edges that would become self-loops
    db.execute("""
        DELETE FROM public.entity_edge
        WHERE (src_entity_id = %s AND dst_entity_id = %s)
           OR (src_entity_id = %s AND dst_entity_id = %s)
    """, (old_id, new_id, new_id, old_id))

    # Delete edges from old_id that would conflict after redirect (src side)
    db.execute("""
        DELETE FROM public.entity_edge ee
        WHERE ee.src_entity_id = %s
          AND EXISTS (
              SELECT 1 FROM public.entity_edge ee2
              WHERE ee2.src_entity_id = %s
                AND ee2.dst_entity_id = ee.dst_entity_id
          )
    """, (old_id, new_id))

    # Delete edges from old_id that would conflict after redirect (dst side)
    db.execute("""
        DELETE FROM public.entity_edge ee
        WHERE ee.dst_entity_id = %s
          AND EXISTS (
              SELECT 1 FROM public.entity_edge ee2
              WHERE ee2.dst_entity_id = %s
                AND ee2.src_entity_id = ee.src_entity_id
          )
    """, (old_id, new_id))

    # Now redirect remaining edges
    db.execute("""
        UPDATE public.entity_edge
        SET src_entity_id = %s
        WHERE src_entity_id = %s
    """, (new_id, old_id))

    db.execute("""
        UPDATE public.entity_edge
        SET dst_entity_id = %s
        WHERE dst_entity_id = %s
    """, (new_id, old_id))

    # Fix ordering convention (src < dst) broken by redirects.
    # Delete misordered edges that conflict with a correctly-ordered copy.
    db.execute("""
        DELETE FROM public.entity_edge ee
        WHERE ee.src_entity_id > ee.dst_entity_id
          AND EXISTS (
              SELECT 1 FROM public.entity_edge ee2
              WHERE ee2.src_entity_id = ee.dst_entity_id
                AND ee2.dst_entity_id = ee.src_entity_id
          )
    """)
    db.execute("""
        UPDATE public.entity_edge
        SET src_entity_id = dst_entity_id,
            dst_entity_id = src_entity_id
        WHERE src_entity_id > dst_entity_id
    """)

    # Finally delete the orphaned entity
    db.execute("DELETE FROM public.entity WHERE id = %s", (old_id,))


if __name__ == "__main__":
    main()
