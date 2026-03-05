-- Craft AI v3 — Topics & Sub-chats
-- Run this in Supabase SQL Editor (Dashboard → SQL Editor → New Query)

-- ============================================================
-- 1. sub_chats table — conversation threads within a topic
-- ============================================================
CREATE TABLE IF NOT EXISTS sub_chats (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    topic_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL,
    chat_type TEXT NOT NULL CHECK (chat_type IN ('headlines', 'text', 'carousel')),
    title TEXT,
    selected_headline TEXT,
    parent_subchat_id UUID REFERENCES sub_chats(id) ON DELETE SET NULL,
    status TEXT DEFAULT 'active',
    template_id TEXT,
    carousel_images JSONB DEFAULT '[]',
    slides_data JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sub_chats_topic_id ON sub_chats(topic_id);
CREATE INDEX IF NOT EXISTS idx_sub_chats_user_id ON sub_chats(user_id);
CREATE INDEX IF NOT EXISTS idx_sub_chats_type ON sub_chats(topic_id, chat_type);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_sub_chats_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_sub_chats_updated_at ON sub_chats;
CREATE TRIGGER trigger_sub_chats_updated_at
    BEFORE UPDATE ON sub_chats
    FOR EACH ROW
    EXECUTE FUNCTION update_sub_chats_updated_at();

-- ============================================================
-- 2. Modify projects table — add active_subchat_id
-- ============================================================
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'projects' AND column_name = 'active_subchat_id'
    ) THEN
        ALTER TABLE projects ADD COLUMN active_subchat_id UUID;
    END IF;
END $$;

-- ============================================================
-- 3. Modify project_messages — add sub_chat_id
-- ============================================================
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'project_messages' AND column_name = 'sub_chat_id'
    ) THEN
        ALTER TABLE project_messages ADD COLUMN sub_chat_id UUID REFERENCES sub_chats(id) ON DELETE CASCADE;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_project_messages_sub_chat_id ON project_messages(sub_chat_id);

-- ============================================================
-- 4. system_prompts table — admin-editable AI prompts
-- ============================================================
CREATE TABLE IF NOT EXISTS system_prompts (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    prompt_key TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    description TEXT,
    content TEXT NOT NULL,
    variables JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT true,
    updated_by BIGINT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

DROP TRIGGER IF EXISTS trigger_system_prompts_updated_at ON system_prompts;
CREATE TRIGGER trigger_system_prompts_updated_at
    BEFORE UPDATE ON system_prompts
    FOR EACH ROW
    EXECUTE FUNCTION update_projects_updated_at();

-- Seed default prompts
INSERT INTO system_prompts (prompt_key, title, description, content, variables) VALUES
(
    'headlines',
    'Генератор заголовков',
    'Генерирует 5-7 вариантов заголовков для карусели',
    'Ты — AI контент-менеджер для Instagram-каруселей. Ты помогаешь создавать контент на основе идеи пользователя.

Сейчас этап: ГЕНЕРАЦИЯ ЗАГОЛОВКОВ.
На основе идеи пользователя предложи 5-7 цепляющих заголовков для карусели.

Формат ответа:
1. Заголовок 1
2. Заголовок 2
...

Пиши на языке пользователя. Заголовки должны быть короткими (3-7 слов), цепляющими, вызывать любопытство.',
    '[{"name": "user_context", "description": "Профиль и стиль пользователя"}]'::jsonb
),
(
    'text',
    'Автор текста слайдов',
    'Пишет текст слайдов карусели на основе выбранного заголовка',
    'Ты — AI контент-менеджер для Instagram-каруселей. Пользователь выбрал заголовок: "{selected_headline}". Теперь напиши текст для слайдов карусели.

Формат ответа СТРОГО:
Слайд 1
Заголовок: [заголовок карусели]
Описание: [подзаголовок или вступление]

Слайд 2
Заголовок: [пункт 1]
Описание: [раскрытие пункта 1]

... и так далее

Пиши кратко, ёмко. Каждый слайд — одна мысль. {slides_count} слайдов.',
    '[{"name": "slides_count", "description": "Количество слайдов"}, {"name": "selected_headline", "description": "Выбранный заголовок"}, {"name": "user_context", "description": "Профиль пользователя"}]'::jsonb
),
(
    'editing',
    'Редактор текста',
    'Помогает улучшить и отредактировать существующий текст карусели',
    'Ты — AI контент-менеджер. Текст карусели готов. Помоги пользователю улучшить или отредактировать текст.
Если пользователь просит что-то изменить — перепиши соответствующие слайды в том же формате.',
    '[{"name": "user_context", "description": "Профиль пользователя"}]'::jsonb
),
(
    'carousel',
    'Дизайнер каруселей',
    'Помогает с визуальным оформлением карусели',
    'Ты — AI помощник по дизайну каруселей. Помоги пользователю выбрать визуальное оформление: шаблон, цвета, шрифты. Отвечай кратко и по делу.',
    '[{"name": "user_context", "description": "Профиль пользователя"}]'::jsonb
)
ON CONFLICT (prompt_key) DO NOTHING;

-- ============================================================
-- 5. Data migration — create sub_chats for existing projects
-- ============================================================
INSERT INTO sub_chats (topic_id, user_id, chat_type, title, status, created_at)
SELECT p.id, p.user_id, 'headlines', p.title,
    CASE WHEN p.status IN ('text_ready', 'carousel_ready') THEN 'completed' ELSE 'active' END,
    p.created_at
FROM projects p
WHERE NOT EXISTS (
    SELECT 1 FROM sub_chats sc WHERE sc.topic_id = p.id
);

-- Link old messages to their new sub_chat
UPDATE project_messages pm
SET sub_chat_id = sc.id
FROM sub_chats sc
WHERE pm.project_id = sc.topic_id
AND pm.sub_chat_id IS NULL;
