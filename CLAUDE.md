# Craft AI — Контекст проекта

## Архитектура
- Backend: FastAPI + Uvicorn (`main.py`, ~3700 строк)
- Frontend: Vanilla HTML/JS SPA (`static/index.html`, ~10500 строк)
- DB: Supabase (PostgreSQL) — таблицы: users, user_templates, topics, ...
- Деплой: Railway (GitHub push → auto-deploy)
- Auth: Telegram (3 метода: Login Widget, Mini App, Bot Deep Link)

## Ключевые файлы
- `main.py` — весь backend (API, auth, шаблоны, генерация)
- `static/index.html` — весь frontend (SPA, редактор, галерея, чат)
- `migrations/006_user_templates.sql` — схема таблицы шаблонов

## Решённые проблемы (не повторять!)

### Шаблоны / Галерея
- **user_id тип**: DB колонка BIGINT, Pydantic модель str → всегда `int(template.user_id)` в backend, `String(currentUser.id)` в frontend
- **Двойной JSON.stringify**: onclick обработчики — НЕ делать `JSON.stringify(JSON.stringify(t))`, использовать `useTemplateById('id')`
- **templateCache = []**: НИКОГДА не обнулять кэш напрямую. Использовать `refreshTemplateCache()` для обновления
- **Большие base64 в preview**: `preview=light` должен возвращать thumbnail (200px) для background.photo и thumbnail (100px) для element.photo. Элементы >500KB — просто убирать
- **Pillow OOM**: НЕ пытаться ресайзить изображения >500KB base64 через Pillow на Railway — вызывает 502. Только удалять
- **localStorage кэш**: при изменении формата ответа API — менять ключ кэша (templateCacheV3 → V4 → ...)
- **Загрузка галереи**: показывать спиннер пока `_templatesFetched === false`, НЕ "Нет шаблонов"

### Auth
- **Bot deep link** (`/api/telegram/webhook`): `message.from` НЕ содержит `photo_url` — нужно запрашивать через `getUserProfilePhotos` Bot API
- **Login Widget и Mini App**: получают `photo_url` из данных Telegram напрямую
- **currentUser.id**: число (Telegram user_id), при передаче в API — `String()`

### API / Backend
- **FastAPI 422 detail**: возвращается как массив объектов, не строка. Обрабатывать: `detail.map(d => d.msg).join('; ')`
- **Supabase upsert**: для user_templates используй `on_conflict="user_id,template_id"`
- **Railway ресурсы**: ограниченная память, НЕ обрабатывать большие изображения на лету

### Frontend
- **rerenderGalleryIfVisible()**: вызывать после ЛЮБОГО обновления templateCache
- **loadTemplateSelectors()**: после фонового обновления вызывает rerenderGalleryIfVisible()
- **saveTemplate()**: НЕ делать silent fallback на localStorage — показывать ошибку пользователю
- **empty catch(e){}**: НИКОГДА — всегда логировать или показывать ошибку

## Правила разработки
1. После изменения формата API ответа — бампить версию localStorage ключа
2. Тестировать с реальным user_id через curl ПЕРЕД деплоем
3. Проверять размер ответа: `curl -s -o /dev/null -w "%{size_download}"` — должен быть <100KB для gallery
4. При работе с base64 изображениями — всегда проверять размер ПЕРЕД обработкой
5. Все ошибки в catch — логировать в console.error, НЕ глотать молча
6. AI-агенты: любая работа с ИИ (новый промпт, обработка, генерация) ОБЯЗАНА иметь запись в таблице `system_prompts` и отображаться во вкладке «Промпты» в админ-панели (`/admin`). Промпт должен загружаться через `get_system_prompt_v3(prompt_key)`, НЕ хардкодиться в коде. Это даёт полный контроль над промптами и моделями через UI.
