-- Craft AI v2.0 — Phase 1: Projects & Messages tables
-- Run this in Supabase SQL Editor (Dashboard → SQL Editor → New Query)

-- Projects table — each carousel = project
CREATE TABLE IF NOT EXISTS projects (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id BIGINT NOT NULL,
    title TEXT DEFAULT 'Новая карусель',
    status TEXT DEFAULT 'draft',
    idea_text TEXT,
    selected_headline TEXT,
    slides_data JSONB,
    template_id TEXT,
    instagram_username TEXT,
    language TEXT DEFAULT 'ru',
    translated_from UUID REFERENCES projects(id) ON DELETE SET NULL,
    carousel_images JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Index for fast user project listing
CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id);
CREATE INDEX IF NOT EXISTS idx_projects_updated_at ON projects(updated_at DESC);

-- Project messages — chat history inside project
CREATE TABLE IF NOT EXISTS project_messages (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    message_type TEXT DEFAULT 'text',
    message_data JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_project_messages_project_id ON project_messages(project_id);
CREATE INDEX IF NOT EXISTS idx_project_messages_created_at ON project_messages(created_at);

-- Auto-update updated_at on projects
CREATE OR REPLACE FUNCTION update_projects_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_projects_updated_at ON projects;
CREATE TRIGGER trigger_projects_updated_at
    BEFORE UPDATE ON projects
    FOR EACH ROW
    EXECUTE FUNCTION update_projects_updated_at();

-- Add plan column to users if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'users' AND column_name = 'plan'
    ) THEN
        ALTER TABLE users ADD COLUMN plan TEXT DEFAULT 'free';
    END IF;
END $$;
