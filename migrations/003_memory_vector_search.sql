-- Craft AI v3.1 — Memory & Vector Search
-- Run this in Supabase SQL Editor (Dashboard → SQL Editor → New Query)

-- ============================================================
-- 1. Ensure pgvector extension is enabled
-- ============================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- 2. Ensure user_memory table has embedding column (vector)
-- ============================================================
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'user_memory' AND column_name = 'embedding'
    ) THEN
        ALTER TABLE user_memory ADD COLUMN embedding vector(1536);
    END IF;
END $$;

-- Create index for fast vector search
CREATE INDEX IF NOT EXISTS idx_user_memory_embedding ON user_memory
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ============================================================
-- 3. RPC function for vector similarity search
-- ============================================================
CREATE OR REPLACE FUNCTION match_memories(
    query_embedding vector(1536),
    match_user_id TEXT,
    match_limit INT DEFAULT 5,
    match_threshold FLOAT DEFAULT 0.7
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        um.id,
        um.content,
        1 - (um.embedding <=> query_embedding) AS similarity
    FROM user_memory um
    WHERE um.user_id = match_user_id
      AND um.embedding IS NOT NULL
      AND 1 - (um.embedding <=> query_embedding) > match_threshold
    ORDER BY um.embedding <=> query_embedding
    LIMIT match_limit;
END;
$$;

-- ============================================================
-- 4. Add memory_count to users table (for auto-summarize trigger)
-- ============================================================
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'users' AND column_name = 'memory_count'
    ) THEN
        ALTER TABLE users ADD COLUMN memory_count INT DEFAULT 0;
    END IF;
END $$;
