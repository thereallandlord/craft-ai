-- Add model field to system_prompts for per-agent model selection
ALTER TABLE system_prompts ADD COLUMN IF NOT EXISTS model TEXT DEFAULT 'openai/gpt-4o';
