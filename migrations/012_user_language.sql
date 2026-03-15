-- Add language preference column to auth_accounts table
ALTER TABLE auth_accounts ADD COLUMN IF NOT EXISTS language TEXT DEFAULT NULL;
-- NULL = auto-detect, 'ru' = Russian, 'en' = English
