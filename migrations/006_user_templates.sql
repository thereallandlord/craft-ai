-- User templates: personal and published templates stored in Supabase
-- System templates remain on filesystem (templates/ directory)

CREATE TABLE IF NOT EXISTS user_templates (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id BIGINT NOT NULL,
    template_id TEXT NOT NULL,
    name TEXT NOT NULL,
    slides JSONB NOT NULL DEFAULT '[]',
    settings JSONB NOT NULL DEFAULT '{}',
    is_published BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_user_templates_user ON user_templates(user_id);
CREATE INDEX IF NOT EXISTS idx_user_templates_published ON user_templates(is_published) WHERE is_published = true;
CREATE UNIQUE INDEX IF NOT EXISTS idx_user_templates_tid ON user_templates(user_id, template_id);
