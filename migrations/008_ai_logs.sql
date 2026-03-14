-- AI call logs for admin monitoring
CREATE TABLE IF NOT EXISTS ai_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id BIGINT,
    endpoint TEXT NOT NULL,
    prompt_key TEXT,
    model TEXT NOT NULL,
    system_prompt TEXT,
    user_context TEXT,
    messages JSONB,
    user_message TEXT,
    ai_response TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    response_time_ms INTEGER DEFAULT 0,
    status TEXT DEFAULT 'success',
    error_message TEXT,
    topic_id UUID,
    sub_chat_id UUID,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ai_logs_user_id ON ai_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_ai_logs_created_at ON ai_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ai_logs_endpoint ON ai_logs(endpoint);
CREATE INDEX IF NOT EXISTS idx_ai_logs_status ON ai_logs(status);
