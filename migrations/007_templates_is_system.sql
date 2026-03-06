-- Add is_system column to user_templates
-- System templates are visible to ALL users, only admins can create/toggle them

ALTER TABLE user_templates ADD COLUMN IF NOT EXISTS is_system BOOLEAN DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_user_templates_system ON user_templates(is_system) WHERE is_system = true;
