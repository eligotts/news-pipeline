-- Migration: Schema Optimization for Ingestion Pipeline
-- Date: 2025-01-24
-- Description: Simplify stance extraction, remove unused columns, add topic_edge table

-- ============================================================================
-- PART 1: DROP UNUSED COLUMNS
-- ============================================================================

-- article table: remove unused columns
ALTER TABLE public.article DROP COLUMN IF EXISTS minhash_sig;
ALTER TABLE public.article DROP COLUMN IF EXISTS keywords_raw;

-- article_entity table: remove old stance-related columns (replaced by stance_text)
ALTER TABLE public.article_entity DROP COLUMN IF EXISTS stance;
ALTER TABLE public.article_entity DROP COLUMN IF EXISTS role;
ALTER TABLE public.article_entity DROP COLUMN IF EXISTS centrality;
ALTER TABLE public.article_entity DROP COLUMN IF EXISTS in_quote;
ALTER TABLE public.article_entity DROP COLUMN IF EXISTS evidence;

-- topic table: remove unused hierarchy/versioning columns
ALTER TABLE public.topic DROP COLUMN IF EXISTS parent_id;
ALTER TABLE public.topic DROP COLUMN IF EXISTS status;
ALTER TABLE public.topic DROP COLUMN IF EXISTS version;
ALTER TABLE public.topic DROP COLUMN IF EXISTS parent_confidence;

-- entity table: remove type column
ALTER TABLE public.entity DROP COLUMN IF EXISTS type;

-- cluster_topic table: remove weight column
ALTER TABLE public.cluster_topic DROP COLUMN IF EXISTS weight;

-- topic_search_term table: remove unused columns
ALTER TABLE public.topic_search_term DROP COLUMN IF EXISTS weight;
ALTER TABLE public.topic_search_term DROP COLUMN IF EXISTS term_type;

-- ============================================================================
-- PART 2: ADD NEW COLUMNS
-- ============================================================================

-- article_entity table: add new stance_text column (free-form text description)
ALTER TABLE public.article_entity ADD COLUMN IF NOT EXISTS stance_text TEXT;

-- cluster table: track when topics were last assigned
ALTER TABLE public.cluster ADD COLUMN IF NOT EXISTS topics_assigned_at TIMESTAMPTZ;

-- ============================================================================
-- PART 3: CREATE NEW TABLES
-- ============================================================================

-- topic_edge: Records co-occurrence between topics assigned to the same cluster
CREATE TABLE IF NOT EXISTS public.topic_edge (
    src_topic_id BIGINT NOT NULL REFERENCES public.topic(id) ON DELETE CASCADE,
    dst_topic_id BIGINT NOT NULL REFERENCES public.topic(id) ON DELETE CASCADE,
    weight REAL NOT NULL DEFAULT 1.0,
    cluster_count INTEGER NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (src_topic_id, dst_topic_id)
);

-- Index for bidirectional lookups
CREATE INDEX IF NOT EXISTS idx_topic_edge_dst ON public.topic_edge(dst_topic_id);

-- ============================================================================
-- PART 4: DROP OBSOLETE TABLES
-- ============================================================================

-- No longer needed after combining entity+stance extraction
DROP TABLE IF EXISTS public.article_entity_stance_job;
