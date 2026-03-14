-- Add last_name and photo_url to users table for admin panel display
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_name TEXT DEFAULT '';
ALTER TABLE users ADD COLUMN IF NOT EXISTS photo_url TEXT DEFAULT '';
