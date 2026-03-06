-- User settings for carousel editor: last template and Instagram usernames
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_template_id TEXT DEFAULT 'kamalov';
ALTER TABLE users ADD COLUMN IF NOT EXISTS instagram_usernames JSONB DEFAULT '[]'::jsonb;
