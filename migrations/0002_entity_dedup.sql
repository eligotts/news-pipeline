-- Migration: Entity Deduplication
-- Date: 2025-01-27
-- Description: Merge duplicate entities sharing the same name_canonical,
--              then add a UNIQUE partial index to prevent future duplicates.
--
-- Run on a DB backup first!  Verification queries at the bottom.

BEGIN;

-- ============================================================================
-- STEP 1: Build a merge map — for each duplicate canonical, keep the lowest id
-- ============================================================================

CREATE TEMP TABLE _entity_merge AS
SELECT
    e.id        AS old_id,
    winner.id   AS new_id
FROM public.entity e
JOIN (
    -- Winner = lowest id per name_canonical
    SELECT name_canonical, MIN(id) AS id
    FROM public.entity
    WHERE name_canonical IS NOT NULL AND name_canonical != ''
    GROUP BY name_canonical
    HAVING COUNT(*) > 1
) winner
    ON winner.name_canonical = e.name_canonical
   AND e.id != winner.id;

-- Index for fast lookups during redirect
CREATE INDEX ON _entity_merge (old_id);

-- ============================================================================
-- STEP 2: Redirect article_entity rows to the winner entity
-- ============================================================================

-- Delete loser rows where the article already has the winner OR has another
-- loser with a smaller entity_id mapping to the same winner.
-- This handles articles linked to multiple duplicates of the same entity.
DELETE FROM public.article_entity ae
USING _entity_merge m
WHERE ae.entity_id = m.old_id
  AND (
      EXISTS (
          SELECT 1 FROM public.article_entity ae2
          WHERE ae2.article_id = ae.article_id
            AND ae2.entity_id = m.new_id
      )
      OR EXISTS (
          SELECT 1 FROM public.article_entity ae2
          JOIN _entity_merge m2 ON m2.old_id = ae2.entity_id
          WHERE ae2.article_id = ae.article_id
            AND m2.new_id = m.new_id
            AND ae2.entity_id < ae.entity_id
      )
  );

-- Now at most one loser per (article, winner) — safe to update.
UPDATE public.article_entity ae
SET entity_id = m.new_id
FROM _entity_merge m
WHERE ae.entity_id = m.old_id;

-- ============================================================================
-- STEP 3: Redirect entity_alias rows
-- ============================================================================

-- Same pattern: delete aliases where the winner already has the same alias
-- OR another loser with smaller id mapping to same winner has the same alias.
DELETE FROM public.entity_alias ea
USING _entity_merge m
WHERE ea.entity_id = m.old_id
  AND (
      EXISTS (
          SELECT 1 FROM public.entity_alias ea2
          WHERE ea2.entity_id = m.new_id
            AND ea2.alias = ea.alias
      )
      OR EXISTS (
          SELECT 1 FROM public.entity_alias ea2
          JOIN _entity_merge m2 ON m2.old_id = ea2.entity_id
          WHERE m2.new_id = m.new_id
            AND ea2.alias = ea.alias
            AND ea2.entity_id < ea.entity_id
      )
  );

UPDATE public.entity_alias ea
SET entity_id = m.new_id
FROM _entity_merge m
WHERE ea.entity_id = m.old_id;

-- ============================================================================
-- STEP 4: Redirect entity_edge rows (save, delete, re-insert approach)
-- ============================================================================

-- 4a: Save redirected edges into temp table with merged IDs + correct ordering
CREATE TEMP TABLE _edge_redirected AS
SELECT DISTINCT ON (new_src, new_dst)
    LEAST(
        COALESCE(ms.new_id, ee.src_entity_id),
        COALESCE(md.new_id, ee.dst_entity_id)
    ) AS new_src,
    GREATEST(
        COALESCE(ms.new_id, ee.src_entity_id),
        COALESCE(md.new_id, ee.dst_entity_id)
    ) AS new_dst,
    ee.relationship,
    ee.weight,
    ee.evidence_count,
    ee.last_article_id,
    ee.updated_at
FROM public.entity_edge ee
LEFT JOIN _entity_merge ms ON ms.old_id = ee.src_entity_id
LEFT JOIN _entity_merge md ON md.old_id = ee.dst_entity_id
WHERE ms.old_id IS NOT NULL OR md.old_id IS NOT NULL
ORDER BY new_src, new_dst, ee.evidence_count DESC;

-- Exclude self-loops
DELETE FROM _edge_redirected WHERE new_src = new_dst;

-- 4b: Delete original edges that involve any loser
DELETE FROM public.entity_edge ee
WHERE EXISTS (SELECT 1 FROM _entity_merge m WHERE m.old_id = ee.src_entity_id)
   OR EXISTS (SELECT 1 FROM _entity_merge m WHERE m.old_id = ee.dst_entity_id);

-- 4c: Re-insert redirected edges (merge with any existing edge for that pair)
INSERT INTO public.entity_edge
    (src_entity_id, dst_entity_id, relationship, weight, evidence_count, last_article_id, updated_at)
SELECT new_src, new_dst, relationship, weight, evidence_count, last_article_id, updated_at
FROM _edge_redirected
ON CONFLICT (src_entity_id, dst_entity_id) DO UPDATE SET
    weight = GREATEST(entity_edge.weight, EXCLUDED.weight),
    evidence_count = entity_edge.evidence_count + EXCLUDED.evidence_count,
    last_article_id = GREATEST(entity_edge.last_article_id, EXCLUDED.last_article_id),
    updated_at = GREATEST(entity_edge.updated_at, EXCLUDED.updated_at);

DROP TABLE _edge_redirected;

-- 4d: Fix any pre-existing misordered edges (src > dst)
DELETE FROM public.entity_edge ee
WHERE ee.src_entity_id > ee.dst_entity_id
  AND EXISTS (
      SELECT 1 FROM public.entity_edge ee2
      WHERE ee2.src_entity_id = ee.dst_entity_id
        AND ee2.dst_entity_id = ee.src_entity_id
  );

UPDATE public.entity_edge
SET src_entity_id = dst_entity_id,
    dst_entity_id = src_entity_id
WHERE src_entity_id > dst_entity_id;

-- ============================================================================
-- STEP 5: Delete orphaned duplicate entity rows
-- ============================================================================

DELETE FROM public.entity e
USING _entity_merge m
WHERE e.id = m.old_id;

-- Cleanup temp table
DROP TABLE _entity_merge;

COMMIT;

-- ============================================================================
-- STEP 6: Add UNIQUE partial index (must run outside transaction for CONCURRENTLY)
-- ============================================================================

CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_entity_name_canonical_unique
ON public.entity (name_canonical)
WHERE name_canonical IS NOT NULL AND name_canonical != '';

-- Index for alias lookups (used by _find_canonical_match)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entity_alias_lower_alias
ON public.entity_alias (lower(alias));


-- ============================================================================
-- VERIFICATION QUERIES (run manually after migration)
-- ============================================================================
--
-- 1. No orphaned FK references:
--    SELECT COUNT(*) FROM article_entity ae LEFT JOIN entity e ON ae.entity_id = e.id WHERE e.id IS NULL;
--    SELECT COUNT(*) FROM entity_alias ea LEFT JOIN entity e ON ea.entity_id = e.id WHERE e.id IS NULL;
--    SELECT COUNT(*) FROM entity_edge ee LEFT JOIN entity e ON ee.src_entity_id = e.id WHERE e.id IS NULL;
--    SELECT COUNT(*) FROM entity_edge ee LEFT JOIN entity e ON ee.dst_entity_id = e.id WHERE e.id IS NULL;
--
-- 2. No duplicate canonicals remain:
--    SELECT name_canonical, COUNT(*) FROM entity
--    WHERE name_canonical IS NOT NULL AND name_canonical != ''
--    GROUP BY name_canonical HAVING COUNT(*) > 1;
--
-- 3. Unique index exists:
--    SELECT indexname FROM pg_indexes WHERE tablename = 'entity' AND indexname = 'idx_entity_name_canonical_unique';
