-- Add 'analysis' to chat_type CHECK constraint for competitor analysis sub-chats
ALTER TABLE sub_chats DROP CONSTRAINT IF EXISTS sub_chats_chat_type_check;
ALTER TABLE sub_chats ADD CONSTRAINT sub_chats_chat_type_check
  CHECK (chat_type IN ('headlines', 'text', 'carousel', 'analysis'));
