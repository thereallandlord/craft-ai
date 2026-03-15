"""Carousel Studio API v7.0
ПРАВИЛЬНАЯ ЛОГИКА:
- Первый слайд из slides → INTRO template (с подстановкой)
- Остальные слайды из slides → CONTENT template (с подстановкой)
- Ending слайды → как есть из шаблона (БЕЗ подстановки)

NEW в v7.0:
- Overlay система: darken/lighten + 4 типа градиентов (solid, bottom, top, both)
- Letter-spacing: -5 до +20px
- Circle shape для фото
- Google Fonts support (автоматическая загрузка)
- Custom Fonts upload (.ttf/.otf)
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass
import requests
import httpx
import base64
import json
import os
import io
import shutil
import grapheme
import arabic_reshaper
from bidi.algorithm import get_display
import re
import uuid
import jwt as pyjwt
from pilmoji.source import AppleEmojiSource

# Apple emoji source + image cache for manual emoji rendering
_emoji_source = AppleEmojiSource()
_emoji_img_cache: dict = {}

def _get_emoji_image(char: str, size: int):
    """Fetch and cache resized Apple emoji image."""
    key = (char, size)
    if key not in _emoji_img_cache:
        try:
            stream = _emoji_source.get_emoji(char)
            if stream:
                from PIL import Image as _Img
                img = _Img.open(stream).convert('RGBA')
                img = img.resize((size, size), _Img.Resampling.LANCZOS)
                _emoji_img_cache[key] = img
            else:
                _emoji_img_cache[key] = None
        except Exception as e:
            print(f"Emoji fetch error for {repr(char)}: {e}")
            _emoji_img_cache[key] = None
    return _emoji_img_cache[key]

# Async HTTP client with connection limits for image proxy
_http_client: httpx.AsyncClient | None = None

async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            timeout=httpx.Timeout(15.0),
            follow_redirects=True,
        )
    return _http_client

# Emoji detection regex
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002702-\U000027B0"  # dingbats
    "\U00002600-\U000026FF"  # misc symbols
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero width joiner
    "\U00002640\U00002642"   # gender symbols
    "\U000023CF-\U000023FF"  # misc technical
    "\U00002B50\U00002B55"   # stars
    "\U0000203C\U00002049"   # exclamation marks
    "\U000000A9\U000000AE"   # copyright/registered
    "]+"
)

def _has_emoji(text: str) -> bool:
    return bool(_EMOJI_RE.search(text))
import hashlib
import asyncio
import hmac
import time
import secrets
import numpy as np
from supabase import create_client, Client as SupabaseClient
import psycopg2
import psycopg2.extras

app = FastAPI(title="Carousel Studio", version="7.0")

@app.on_event("shutdown")
async def shutdown_http_client():
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()

# === Supabase ===
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
supabase: SupabaseClient | None = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print("✅ Supabase connected")
else:
    print("⚠️ Supabase not configured (SUPABASE_URL / SUPABASE_SERVICE_KEY missing)")
if DATABASE_URL:
    print("✅ DATABASE_URL configured (direct PG)")
else:
    print("⚠️ DATABASE_URL not set — ai_logs will not work")

# === Supabase Auth (JWT) ===
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

# JWKS client for ECC (ES256) JWT verification — Supabase migrated from HS256
_jwks_client = None
if SUPABASE_URL:
    _jwks_url = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
    try:
        _jwks_client = pyjwt.PyJWKClient(_jwks_url, cache_keys=True, lifespan=3600)
        print(f"✅ JWKS client configured: {_jwks_url}")
    except Exception as e:
        print(f"⚠️ JWKS client init failed: {e}")

# === Scrape Creators API ===
SCRAPE_CREATORS_API_KEY = os.getenv("SCRAPE_CREATORS_API_KEY", "")
SCRAPE_CREATORS_BASE = "https://api.scrapecreators.com"

# === Lava.top Payments ===
LAVA_TOP_API_KEY = os.getenv("LAVA_TOP_API_KEY", "")
LAVA_TOP_WEBHOOK_SECRET = os.getenv("LAVA_TOP_WEBHOOK_SECRET", "")
LAVA_TOP_OFFER_PRO = os.getenv("LAVA_TOP_OFFER_PRO_MONTHLY", "")
LAVA_TOP_OFFER_BUSINESS = os.getenv("LAVA_TOP_OFFER_BUSINESS_MONTHLY", "")
LAVA_TOP_API_URL = "https://gate.lava.top"
if LAVA_TOP_API_KEY:
    print("✅ Lava.top configured")
else:
    print("⚠️ Lava.top not configured (LAVA_TOP_API_KEY missing)")

PLATFORM_CONFIG = {
    "instagram": {
        "patterns": [r'instagram\.com/(p|reel|reels)/[\w-]+'],
        "post_endpoint": "/v1/instagram/post",
        "reel_endpoint": "/v2/instagram/reel",
        "transcript_endpoint": "/v2/instagram/media/transcript",
    },
    "youtube": {
        "patterns": [r'youtube\.com/watch\?v=', r'youtu\.be/', r'youtube\.com/shorts/'],
        "post_endpoint": "/v1/youtube/video",
        "transcript_endpoint": "/v1/youtube/video/transcript",
    },
    "tiktok": {
        "patterns": [r'tiktok\.com/@[\w.]+/video/', r'tiktok\.com/@[\w.]+/photo/', r'vm\.tiktok\.com/', r'tiktok\.com/t/'],
        "post_endpoint": "/v2/tiktok/video",
        "transcript_endpoint": "/v1/tiktok/video/transcript",
    },
    "twitter": {
        "patterns": [r'(twitter\.com|x\.com)/\w+/status/\d+'],
        "post_endpoint": "/v1/twitter/tweet",
        "transcript_endpoint": "/v1/twitter/tweet/transcript",
    },
    "linkedin": {
        "patterns": [r'linkedin\.com/(posts|feed/update)'],
        "post_endpoint": "/v1/linkedin/post",
        "transcript_endpoint": None,
    },
    "facebook": {
        "patterns": [r'facebook\.com/.+/(posts|videos|reel)'],
        "post_endpoint": "/v1/facebook/post",
        "transcript_endpoint": "/v1/facebook/post/transcript",
    },
}

def detect_platform(url: str) -> str | None:
    """Detect social media platform from URL."""
    for platform, cfg in PLATFORM_CONFIG.items():
        for pattern in cfg["patterns"]:
            if re.search(pattern, url):
                return platform
    return None

if SCRAPE_CREATORS_API_KEY:
    print("✅ SCRAPE_CREATORS_API_KEY configured")
else:
    print("⚠️ SCRAPE_CREATORS_API_KEY not set — competitor analysis will not work")


def _pg_query(query, params=None, fetchone=False):
    """Execute SQL directly via psycopg2, bypassing PostgREST."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            if cur.description:
                if fetchone:
                    row = cur.fetchone()
                    conn.commit()
                    return dict(row) if row else None
                else:
                    rows = cur.fetchall()
                    conn.commit()
                    return [dict(r) for r in rows]
            conn.commit()
            return None
    finally:
        conn.close()


# Auto-create ai_logs table on startup
if DATABASE_URL:
    try:
        _pg_query("""
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
            )
        """)
        _pg_query("CREATE INDEX IF NOT EXISTS idx_ai_logs_user_id ON ai_logs(user_id)")
        _pg_query("CREATE INDEX IF NOT EXISTS idx_ai_logs_created_at ON ai_logs(created_at DESC)")
        _pg_query("CREATE INDEX IF NOT EXISTS idx_ai_logs_endpoint ON ai_logs(endpoint)")
        _pg_query("CREATE INDEX IF NOT EXISTS idx_ai_logs_status ON ai_logs(status)")
        print("✅ ai_logs table ready")
    except Exception as e:
        print(f"⚠️ ai_logs table migration error: {e}")

    # Auto-migration: add last_name and photo_url to users table
    try:
        _pg_query("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_name TEXT DEFAULT ''")
        _pg_query("ALTER TABLE users ADD COLUMN IF NOT EXISTS photo_url TEXT DEFAULT ''")
        print("✅ users extra fields ready")
    except Exception as e:
        print(f"⚠️ users extra fields migration error: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def anon_token_and_headers(request: Request, call_next):
    response = await call_next(request)
    # Set anonymous tracking cookie if not present
    if "craft_anon_token" not in request.cookies and request.url.path.startswith("/api/"):
        anon_token = str(uuid.uuid4())
        response.set_cookie(
            "craft_anon_token", anon_token,
            httponly=True, samesite="lax", max_age=86400 * 365
        )
    # Allow Telegram Web to iframe the app
    response.headers["Content-Security-Policy"] = (
        "frame-ancestors 'self' https://web.telegram.org https://*.telegram.org"
    )
    return response

# NEW v7.0: Configurable paths for Railway Volume support
# Railway Volume монтируется в /app/data (настраивается через Railway Dashboard)
# Локально используются обычные папки в текущей директории
DATA_DIR = os.getenv("DATA_PATH", ".")  # Railway: /app/data, локально: текущая директория
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")  # Legacy, used only for migration
GIT_TEMPLATES_DIR = "templates"  # Legacy, used only for migration
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
FONTS_DIR = "fonts"  # fonts всегда локальные (часть кодовой базы)

# Создаём необходимые директории
os.makedirs("static", exist_ok=True)
os.makedirs(FONTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"📁 Uploads directory: {UPLOADS_DIR}")
print(f"📁 Output directory: {OUTPUT_DIR}")

CANVAS_W, CANVAS_H = 1080, 1350


class GenerateRequest(BaseModel):
    template_name: Optional[str] = None
    template_id: Optional[str] = None
    USERNAME: Optional[str] = None
    username: Optional[str] = None
    slides: Optional[List[Dict[str, Any]]] = None


class SlideData(BaseModel):
    slide: Dict[str, Any]
    settings: Dict[str, Any]
    slideNumber: int


class TemplateData(BaseModel):
    name: str
    template_id: Optional[str] = None  # NEW v7.0: уникальный идентификатор для API
    settings: Dict[str, Any] = {}
    slides: List[Dict[str, Any]]
    createdAt: Optional[str] = None
    user_id: Optional[str] = None  # If set, save to Supabase instead of filesystem


def generate_template_id(name: str) -> str:
    """
    Генерирует template_id из названия шаблона с транслитерацией
    Пример: "Мой Шаблон!" → "moy_shablon"
    """
    # Транслитерация русских букв
    translit = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
        'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
        'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
        'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch',
        'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya'
    }

    result = name.lower()
    for ru, en in translit.items():
        result = result.replace(ru, en)

    # Убираем все кроме букв, цифр, пробелов и дефисов
    result = re.sub(r'[^a-z0-9\s\-]', '', result)
    # Пробелы в подчеркивания
    result = re.sub(r'\s+', '_', result.strip())
    # Множественные подчеркивания в одно
    result = re.sub(r'_+', '_', result)

    return result or 'template'


class SlideRenderer:
    def __init__(self):
        self.font_cache = {}

    def download_google_font(self, family: str, weight: str = '400') -> str:
        """Скачать Google Font и закэшировать в fonts/google/"""
        os.makedirs('fonts/google', exist_ok=True)
        safe_name = family.replace(' ', '_')
        cache_path = f"fonts/google/{safe_name}-{weight}.ttf"

        if os.path.exists(cache_path):
            return cache_path

        try:
            import urllib.request
            # Получить CSS с Google Fonts API
            font_family_url = family.replace(' ', '+')
            css_url = f"https://fonts.googleapis.com/css2?family={font_family_url}:wght@{weight}&display=swap"

            req = urllib.request.Request(css_url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req, timeout=10)
            css_content = response.read().decode('utf-8')

            # Извлечь URL .ttf файла из CSS
            import re
            font_url_match = re.search(r'url\((https://[^)]+\.(?:ttf|woff2))\)', css_content)
            if font_url_match:
                font_url = font_url_match.group(1)
                urllib.request.urlretrieve(font_url, cache_path)
                print(f"Downloaded Google Font: {family} {weight} → {cache_path}")
                return cache_path
        except Exception as e:
            print(f"Failed to download Google Font {family} {weight}: {e}")

        return None

    def get_font(self, family: str, size: int, weight: str = '400'):
        weight_map = {
            '300': 'Light', '400': 'Regular', '500': 'Medium',
            '600': 'SemiBold', '700': 'Bold', '800': 'ExtraBold', '900': 'Black'
        }
        weight_name = weight_map.get(str(weight), 'Regular')
        font_name = f"Inter-{weight_name}" if family == 'Inter' and weight_name != 'Regular' else family
        key = f"{font_name}_{size}"

        if key not in self.font_cache:
            paths = [
                f"fonts/{font_name}.otf",
                f"fonts/{font_name}.ttf",
                os.path.join(UPLOADS_DIR, f"{family}.ttf"),  # NEW: Custom fonts
                os.path.join(UPLOADS_DIR, f"{family}.otf"),  # NEW: Custom fonts
                f"fonts/google/{family.replace(' ', '_')}-{weight}.ttf",  # NEW: Google Fonts cache
                "fonts/Inter.otf",
                "fonts/Inter-Regular.otf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if weight_name in ['Bold', 'ExtraBold', 'Black', 'SemiBold'] else None,
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            ]
            for path in paths:
                if path and os.path.exists(path):
                    try:
                        self.font_cache[key] = ImageFont.truetype(path, size)
                        break
                    except:
                        continue

            # NEW: Попытаться скачать Google Font если не нашли локально
            if key not in self.font_cache:
                google_font_path = self.download_google_font(family, weight)
                if google_font_path:
                    try:
                        self.font_cache[key] = ImageFont.truetype(google_font_path, size)
                    except:
                        pass

            if key not in self.font_cache:
                self.font_cache[key] = self.get_fallback_font(size, weight)

        return self.font_cache[key]

    # === Multilingual support ===

    def _contains_arabic_letters(self, text: str) -> bool:
        """Check if text contains actual Arabic letters (not just symbols like ﷺ)."""
        for ch in text:
            # Arabic letters: U+0621-U+064A (main), U+0671-U+06D3 (extended)
            if '\u0621' <= ch <= '\u064A' or '\u0671' <= ch <= '\u06D3':
                return True
        return False

    def _contains_arabic(self, text: str) -> bool:
        """Check if text contains Arabic script characters (including symbols)."""
        for ch in text:
            if '\u0600' <= ch <= '\u06FF' or '\u0750' <= ch <= '\u077F' or '\uFB50' <= ch <= '\uFDFF' or '\uFE70' <= ch <= '\uFEFF':
                return True
        return False

    def preprocess_text(self, text: str) -> str:
        """Reshape Arabic characters and apply BiDi algorithm for correct RTL rendering."""
        if self._contains_arabic_letters(text):
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        return text

    _notdef_cache = {}  # font_id -> notdef_mask_sum

    def has_glyph(self, font, char: str) -> bool:
        """Check if font contains a glyph for the character (not .notdef)."""
        try:
            if char.strip() == '':
                return True
            # Compare mask with a known-missing glyph (.notdef)
            font_id = id(font)
            if font_id not in self._notdef_cache:
                notdef_mask = font.getmask('\uFFFF')
                self._notdef_cache[font_id] = sum(notdef_mask)
            char_mask = font.getmask(char)
            return sum(char_mask) != self._notdef_cache[font_id]
        except:
            return False

    def get_fallback_font(self, size: int, weight: str = '400'):
        """Return a Noto fallback font for Unicode coverage."""
        weight_name = 'Bold' if str(weight) in ('600', '700', '800', '900') else 'Regular'
        fallback_chain = [
            f'fonts/NotoSansArabic-{weight_name}.ttf',
            f'fonts/NotoSans-{weight_name}.ttf',
            'fonts/NotoSansSymbols2-Regular.ttf',
        ]
        key = f"_fallback_{weight_name}_{size}"
        if key not in self.font_cache:
            for path in fallback_chain:
                if os.path.exists(path):
                    try:
                        self.font_cache[key] = ImageFont.truetype(path, size)
                        return self.font_cache[key]
                    except:
                        continue
            self.font_cache[key] = ImageFont.load_default()
        return self.font_cache[key]

    def select_font_for_text(self, text: str, primary_font, size: int, weight: str = '400'):
        """Use primary font if it has all glyphs, otherwise try fallback chain per-character."""
        for ch in text:
            if ch.strip() == '':
                continue
            if not self.has_glyph(primary_font, ch):
                # Try specific fallback fonts for this character
                weight_name = 'Bold' if str(weight) in ('600', '700', '800', '900') else 'Regular'
                for path in [f'fonts/NotoSansArabic-{weight_name}.ttf', f'fonts/NotoSans-{weight_name}.ttf', 'fonts/NotoSansSymbols2-Regular.ttf']:
                    fkey = f"_fb_{path}_{size}"
                    if fkey not in self.font_cache:
                        if os.path.exists(path):
                            try:
                                self.font_cache[fkey] = ImageFont.truetype(path, size)
                            except:
                                continue
                        else:
                            continue
                    fb = self.font_cache[fkey]
                    if self.has_glyph(fb, ch):
                        return fb
                return self.get_fallback_font(size, weight)
        return primary_font

    def load_image(self, source: str):
        if not source:
            return None
        try:
            if source.startswith('data:'):
                _, data = source.split(',', 1)
                return Image.open(io.BytesIO(base64.b64decode(data)))
            elif source.startswith('http'):
                r = requests.get(source, timeout=30)
                r.raise_for_status()
                return Image.open(io.BytesIO(r.content))
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def create_background(self, bg: dict) -> Image.Image:
        color = bg.get('color', '#ffffff')
        try:
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        except:
            r, g, b = 255, 255, 255

        canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), (r, g, b))

        if bg.get('type') == 'photo' and bg.get('photo'):
            img = self.load_image(bg['photo'])
            if img:
                pos = bg.get('photoPosition', {'x': 50, 'y': 50})
                zoom = max(1, bg.get('photoZoom', 1))

                img_ratio = img.width / img.height
                canvas_ratio = CANVAS_W / CANVAS_H

                if img_ratio > canvas_ratio:
                    base_height, base_width = CANVAS_H, int(CANVAS_H * img_ratio)
                else:
                    base_width, base_height = CANVAS_W, int(CANVAS_W / img_ratio)

                new_width, new_height = int(base_width * zoom), int(base_height * zoom)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                offset_x = int(max(0, new_width - CANVAS_W) * pos.get('x', 50) / 100)
                offset_y = int(max(0, new_height - CANVAS_H) * pos.get('y', 50) / 100)
                img = img.crop((offset_x, offset_y, offset_x + CANVAS_W, offset_y + CANVAS_H))

                if img.mode != 'RGB':
                    img = img.convert('RGB')
                canvas = img

        overlay = bg.get('overlay', 0)
        if overlay > 0:
            canvas = canvas.convert('RGBA')

            # NEW: Overlay mode - darken (black) or lighten (white)
            overlay_mode = bg.get('overlayMode', 'darken')
            overlay_color = (0, 0, 0) if overlay_mode == 'darken' else (255, 255, 255)

            # NEW: Overlay gradient type - solid, bottom, top, both
            overlay_gradient = bg.get('overlayGradient', bg.get('overlayType', 'solid'))
            if overlay_gradient == 'gradient':  # Legacy support
                overlay_gradient = 'bottom'

            if overlay_gradient == 'solid':
                # Solid overlay - равномерный по всему canvas
                overlay_img = Image.new('RGBA', (CANVAS_W, CANVAS_H),
                                        overlay_color + (int(255 * overlay / 100),))
                if overlay_mode == 'darken':
                    canvas = Image.alpha_composite(canvas, overlay_img)
                else:  # lighten - use screen blend
                    canvas = self.apply_lighten_blend(canvas, overlay_img)

            elif overlay_gradient == 'bottom':
                # Gradient bottom - усиливается снизу (от 40% высоты)
                gradient = Image.new('RGBA', (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
                draw = ImageDraw.Draw(gradient)
                for y in range(CANVAS_H):
                    progress = y / CANVAS_H
                    if progress > 0.4:
                        alpha = int(255 * overlay / 100 * ((progress - 0.4) / 0.6))
                        draw.line([(0, y), (CANVAS_W, y)], fill=overlay_color + (alpha,))
                if overlay_mode == 'darken':
                    canvas = Image.alpha_composite(canvas, gradient)
                else:  # lighten - use screen blend
                    canvas = self.apply_lighten_blend(canvas, gradient)

            elif overlay_gradient == 'top':
                # Gradient top - усиливается сверху (до 60% высоты)
                gradient = Image.new('RGBA', (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
                draw = ImageDraw.Draw(gradient)
                for y in range(CANVAS_H):
                    progress = y / CANVAS_H
                    if progress < 0.6:
                        alpha = int(255 * overlay / 100 * (1 - progress / 0.6))
                        draw.line([(0, y), (CANVAS_W, y)], fill=overlay_color + (alpha,))
                if overlay_mode == 'darken':
                    canvas = Image.alpha_composite(canvas, gradient)
                else:  # lighten - use screen blend
                    canvas = self.apply_lighten_blend(canvas, gradient)

            elif overlay_gradient == 'both':
                # Gradient both - с обеих сторон (параболический эффект)
                gradient = Image.new('RGBA', (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
                draw = ImageDraw.Draw(gradient)
                for y in range(CANVAS_H):
                    progress = y / CANVAS_H
                    # Parabolic: максимум на краях (0 и 1), минимум в центре (0.5)
                    edge_factor = 4 * (progress - 0.5) ** 2  # Fixed: removed "1 -" to get correct parabola
                    alpha = int(255 * overlay / 100 * edge_factor)
                    draw.line([(0, y), (CANVAS_W, y)], fill=overlay_color + (alpha,))
                if overlay_mode == 'darken':
                    canvas = Image.alpha_composite(canvas, gradient)
                else:  # lighten - use screen blend
                    canvas = self.apply_lighten_blend(canvas, gradient)

            canvas = canvas.convert('RGB')

        return canvas

    def apply_lighten_blend(self, base: Image.Image, overlay: Image.Image) -> Image.Image:
        """Apply lighten/screen blend mode for white overlay"""
        # Convert to numpy arrays for efficient pixel manipulation
        base_array = np.array(base, dtype=np.float32)
        overlay_array = np.array(overlay, dtype=np.float32)

        # Extract alpha channel from overlay
        alpha = overlay_array[:, :, 3:4] / 255.0  # Normalize to 0-1

        # Get RGB channels
        base_rgb = base_array[:, :, :3]
        overlay_rgb = overlay_array[:, :, :3]

        # Screen blend formula: result = 1 - (1 - base) * (1 - overlay)
        # In 0-255 range: result = 255 - ((255 - base) * (255 - overlay)) / 255
        screen_blend = 255 - ((255 - base_rgb) * (255 - overlay_rgb)) / 255

        # Mix based on alpha: result = base * (1 - alpha) + screen_blend * alpha
        result_rgb = base_rgb * (1 - alpha) + screen_blend * alpha

        # Clip to valid range and convert back
        result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)

        # Add back alpha channel (fully opaque)
        result = np.dstack([result_rgb, np.full((base_array.shape[0], base_array.shape[1]), 255, dtype=np.uint8)])

        return Image.fromarray(result, mode='RGBA')

    def parse_color(self, color: str):
        if not color:
            return (0, 0, 0)
        color = color.lstrip('#')
        try:
            return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        except:
            return (0, 0, 0)

    def draw_photo_element(self, canvas: Image.Image, el: dict):
        if not el.get('photo'):
            return

        img = self.load_image(el['photo'])
        if not img:
            return

        x, y = int(el.get('x', 0)), int(el.get('y', 0))
        w, h = int(el.get('width', 300)), int(el.get('height', 300))
        border_radius = el.get('borderRadius', 0)
        shape = el.get('shape', 'rectangle')  # NEW: 'rectangle' или 'circle'

        img_ratio = img.width / img.height
        el_ratio = w / h

        if img_ratio > el_ratio:
            new_h = h
            new_w = int(h * img_ratio)
        else:
            new_w = w
            new_h = int(w / img_ratio)

        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        left = (new_w - w) // 2
        top = (new_h - h) // 2
        img = img.crop((left, top, left + w, top + h))

        if shape == 'circle':
            # NEW: Круглая маска (ellipse)
            img = img.convert('RGBA')
            mask = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse([0, 0, w, h], fill=255)
            img.putalpha(mask)
            canvas.paste(img, (x, y), img)
        elif border_radius > 0:
            # Прямоугольник с скругленными углами (НЕ овал)
            img = img.convert('RGBA')
            mask = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(mask)
            # NEW v7.0: Use pixel-based radius, not percentage, to prevent oval
            # Max radius is 50px OR 20% of smallest side, whichever is smaller
            max_radius_px = 50
            max_radius_from_size = int(min(w, h) * 0.2)  # 20% of smallest side
            max_allowed = min(max_radius_px, max_radius_from_size)
            radius = int(border_radius)  # border_radius now in pixels (0-50)
            radius = min(radius, max_allowed)
            draw.rounded_rectangle([0, 0, w, h], radius=radius, fill=255)
            img.putalpha(mask)
            canvas.paste(img, (x, y), img)
        else:
            # Обычный прямоугольник
            if img.mode == 'RGBA':
                canvas.paste(img, (x, y), img)
            else:
                canvas.paste(img, (x, y))

    def get_text_width(self, text: str, font, letter_spacing: int, draw, font_size: int = 0) -> int:
        """Calculate text width with letter spacing, accounting for emoji."""
        if _has_emoji(text):
            # For emoji text: measure non-emoji parts + estimate emoji width
            total_width = 0
            for cluster in grapheme.graphemes(text):
                if _has_emoji(cluster):
                    total_width += (font_size or 48) + letter_spacing
                else:
                    bbox = draw.textbbox((0, 0), cluster, font=font)
                    total_width += (bbox[2] - bbox[0]) + letter_spacing
            return total_width - letter_spacing if total_width > 0 else 0
        elif letter_spacing == 0 or self._contains_arabic(text):
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0]
        else:
            total_width = 0
            for cluster in grapheme.graphemes(text):
                bbox = draw.textbbox((0, 0), cluster, font=font)
                total_width += (bbox[2] - bbox[0]) + letter_spacing
            return total_width - letter_spacing if total_width > 0 else 0

    def draw_text_element(self, canvas: Image.Image, el: dict, settings: dict, slide_num: int, total_slides: int, username_override: str = None):
        draw = ImageDraw.Draw(canvas)
        content = el.get('content', '')

        x, y = int(el.get('x', 0)), int(el.get('y', 0))
        font_family = el.get('fontFamily', 'Inter')
        font_size = int(el.get('fontSize', 48))
        font_weight = str(el.get('fontWeight', '400'))
        color = el.get('color', '#000000')
        highlight_color = el.get('highlightColor', '#c8ff00')
        opacity = float(el.get('opacity', 100)) / 100
        line_height = float(el.get('lineHeight', 1.2))
        max_width = int(el.get('maxWidth')) if el.get('maxWidth') else None
        align = el.get('align', 'left')
        letter_spacing = float(el.get('letterSpacing', 0))  # NEW: -5 до +20px (float для 0.5 шага)

        if el.get('type') == 'username':
            content = username_override or settings.get('username', '@username')
        elif el.get('type') == 'slidenum':
            content = f"{slide_num}/{total_slides}"

        font = self.get_font(font_family, font_size, font_weight)
        # NEW v7.0: Get bold font for *bold* text
        bold_font = self.get_font(font_family, font_size, '700') if font_weight != '700' else font

        base_color = self.parse_color(color)
        hl_color = self.parse_color(highlight_color)

        if opacity < 1:
            base_color = tuple(int(c * opacity) for c in base_color)
            hl_color = tuple(int(c * opacity) for c in hl_color)

        # Parse markup: *bold*, _highlight_, {#hex:colored text}
        segments = []
        pattern = r'\*([^*]+)\*|_([^_]+)_|\{(#[0-9a-fA-F]{3,8}):([^}]+)\}'
        last_end = 0
        for match in re.finditer(pattern, content):
            if match.start() > last_end:
                segments.append({'text': content[last_end:match.start()], 'hl': False, 'bold': False, 'color': None})

            if match.group(1):  # *text* = bold
                segments.append({'text': match.group(1), 'hl': False, 'bold': True, 'color': None})
            elif match.group(2):  # _text_ = highlight
                segments.append({'text': match.group(2), 'hl': True, 'bold': False, 'color': None})
            elif match.group(3) and match.group(4):  # {#hex:text} = colored
                segments.append({'text': match.group(4), 'hl': False, 'bold': False, 'color': match.group(3)})

            last_end = match.end()

        if last_end < len(content):
            segments.append({'text': content[last_end:], 'hl': False, 'bold': False, 'color': None})

        if not segments:
            segments = [{'text': content, 'hl': False, 'bold': False, 'color': None}]

        # Preprocess text for Arabic/RTL and auto-align
        has_arabic_letters = False
        for seg in segments:
            if self._contains_arabic_letters(seg['text']):
                has_arabic_letters = True
                seg['text'] = self.preprocess_text(seg['text'])
        if has_arabic_letters and align == 'left':
            align = 'right'

        lines = []
        current_words, current_segs = [], []
        for seg in segments:
            for pi, part in enumerate(seg['text'].split('\n')):
                if pi > 0:
                    lines.append(current_segs)
                    current_words, current_segs = [], []
                for word in part.split(' '):
                    if not word:
                        continue
                    test = ' '.join(current_words + [word])
                    test_width = self.get_text_width(test, font, letter_spacing, draw, font_size)
                    if max_width and test_width > max_width and current_words:
                        lines.append(current_segs)
                        current_words, current_segs = [word], [{'text': word, 'hl': seg['hl'], 'color': seg.get('color')}]
                    else:
                        if current_words:
                            current_segs.append({'text': ' ', 'hl': False, 'color': None})
                        current_words.append(word)
                        current_segs.append({'text': word, 'hl': seg['hl'], 'color': seg.get('color')})
        if current_segs:
            lines.append(current_segs)

        # Get primary font ascent for baseline alignment
        primary_ascent = font.getmetrics()[0]

        curr_y = y
        for line_segs in lines:
            if not line_segs:
                curr_y += int(font_size * line_height)
                continue

            line_text = ''.join(s['text'] for s in line_segs)
            line_font = self.select_font_for_text(line_text, font, font_size, font_weight)
            line_w = self.get_text_width(line_text, line_font, letter_spacing, draw, font_size)

            curr_x = x - line_w if align == 'right' else (x - line_w // 2 if align == 'center' else x)

            for seg in line_segs:
                # Color priority: custom color > highlight > base
                if seg.get('color'):
                    col = self.parse_color(seg['color'])
                    if opacity < 1:
                        col = tuple(int(c * opacity) for c in col)
                elif seg['hl']:
                    col = hl_color
                else:
                    col = base_color
                seg_has_emoji = _has_emoji(seg['text'])

                if seg_has_emoji:
                    # Manual emoji rendering — bypass pilmoji text layout
                    seg_font = bold_font if seg.get('bold') else font
                    seg_ascent = seg_font.getmetrics()[0]
                    emoji_y = curr_y + int(font_size * 0.15)
                    for cluster in grapheme.graphemes(seg['text']):
                        if _has_emoji(cluster):
                            try:
                                emoji_img = _get_emoji_image(cluster, font_size)
                                if emoji_img:
                                    canvas.paste(emoji_img, (int(curr_x), int(emoji_y)), emoji_img)
                            except Exception as e:
                                print(f"Emoji paste error: {e}")
                            curr_x += font_size + int(letter_spacing)
                        else:
                            draw.text((curr_x, curr_y), cluster, font=seg_font, fill=col)
                            bbox = draw.textbbox((0, 0), cluster, font=seg_font)
                            curr_x += (bbox[2] - bbox[0]) + int(letter_spacing)
                    if letter_spacing != 0:
                        curr_x -= int(letter_spacing)
                else:
                    seg_font = bold_font if seg.get('bold') else font
                    # Font fallback for Unicode/Arabic/special chars
                    seg_font = self.select_font_for_text(seg['text'], seg_font, font_size, font_weight)
                    # Baseline alignment: adjust Y when fallback font has different ascent
                    seg_ascent = seg_font.getmetrics()[0]
                    y_offset = primary_ascent - seg_ascent if seg_font != font else 0

                if not seg_has_emoji:
                    if letter_spacing != 0 and not self._contains_arabic(seg['text']):
                        for cluster in grapheme.graphemes(seg['text']):
                            draw.text((curr_x, curr_y + y_offset), cluster, font=seg_font, fill=col)
                            bbox = draw.textbbox((0, 0), cluster, font=seg_font)
                            curr_x += (bbox[2] - bbox[0]) + letter_spacing
                    else:
                        draw.text((curr_x, curr_y + y_offset), seg['text'], font=seg_font, fill=col)
                        bbox = draw.textbbox((0, 0), seg['text'], font=seg_font)
                        curr_x += bbox[2] - bbox[0]

            curr_y += int(font_size * line_height)

    def calculate_text_lines_count(self, el: dict, settings: dict, slide_num: int, total_slides: int, username_override: str = None) -> int:
        """Вычислить количество строк которые займёт текст"""
        content = el.get('content', '')

        if el.get('type') == 'username':
            content = username_override or settings.get('username', '@username')
        elif el.get('type') == 'slidenum':
            content = f"{slide_num}/{total_slides}"

        font_family = el.get('fontFamily', 'Inter')
        font_size = int(el.get('fontSize', 48))
        font_weight = str(el.get('fontWeight', '400'))
        max_width = int(el.get('maxWidth')) if el.get('maxWidth') else None
        letter_spacing = float(el.get('letterSpacing', 0))  # FIX: Add missing variable

        font = self.get_font(font_family, font_size, font_weight)

        # Создаём временный draw для вычисления
        temp_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_img)

        # Parse markup: *bold*, _highlight_, {#hex:colored text}
        segments = []
        pattern = r'\*([^*]+)\*|_([^_]+)_|\{(#[0-9a-fA-F]{3,8}):([^}]+)\}'
        last_end = 0
        for match in re.finditer(pattern, content):
            if match.start() > last_end:
                segments.append({'text': content[last_end:match.start()], 'hl': False})

            if match.group(1):
                segments.append({'text': match.group(1), 'hl': False})
            elif match.group(2):
                segments.append({'text': match.group(2), 'hl': True})
            elif match.group(4):
                segments.append({'text': match.group(4), 'hl': False})

            last_end = match.end()

        if last_end < len(content):
            segments.append({'text': content[last_end:], 'hl': False})

        if not segments:
            segments = [{'text': content, 'hl': False}]

        # Preprocess Arabic text for correct line counting
        for seg in segments:
            if self._contains_arabic_letters(seg['text']):
                seg['text'] = self.preprocess_text(seg['text'])

        # Разбиваем на строки
        lines = []
        current_words, current_segs = [], []
        for seg in segments:
            for pi, part in enumerate(seg['text'].split('\n')):
                if pi > 0:
                    lines.append(current_segs)
                    current_words, current_segs = [], []
                for word in part.split(' '):
                    if not word:
                        continue
                    test = ' '.join(current_words + [word])
                    # Use helper function that accounts for letter spacing
                    test_width = self.get_text_width(test, font, letter_spacing, draw, font_size)
                    if max_width and test_width > max_width and current_words:
                        lines.append(current_segs)
                        current_words, current_segs = [word], [{'text': word, 'hl': seg['hl']}]
                    else:
                        if current_words:
                            current_segs.append({'text': ' ', 'hl': False})
                        current_words.append(word)
                        current_segs.append({'text': word, 'hl': seg['hl']})
        if current_segs:
            lines.append(current_segs)

        return len(lines)

    def wrap_text_for_preview(self, content: str, font_family: str = 'Inter', font_size: int = 48,
                               font_weight: str = '400', max_width: int = None, letter_spacing: float = 0) -> str:
        """Return content with \\n inserted at PIL's word-wrap points, preserving markup."""
        if not content or not max_width:
            return content

        font = self.get_font(font_family, font_size, font_weight)
        temp_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_img)

        # Strip markup for measurement, track markup positions for reconstruction
        # Work with raw content: split by \n first, then word-wrap each paragraph
        paragraphs = content.split('\n')
        wrapped_paragraphs = []

        for para in paragraphs:
            if not para.strip():
                wrapped_paragraphs.append(para)
                continue

            # Strip markup for width measurement
            plain = re.sub(r'\{#[0-9a-fA-F]{3,8}:([^}]+)\}', r'\1', para)
            plain = re.sub(r'\*([^*]+)\*', r'\1', plain)

            # Word-wrap the plain text
            words = plain.split(' ')
            plain_lines = []
            current_words = []
            for word in words:
                if not word:
                    continue
                test = ' '.join(current_words + [word])
                test_width = self.get_text_width(test, font, letter_spacing, draw, font_size)
                if test_width > max_width and current_words:
                    plain_lines.append(' '.join(current_words))
                    current_words = [word]
                else:
                    current_words.append(word)
            if current_words:
                plain_lines.append(' '.join(current_words))

            if len(plain_lines) <= 1:
                wrapped_paragraphs.append(para)
                continue

            # Map plain line breaks back to original paragraph (with markup)
            # Strategy: consume words from original para matching plain_lines
            result_lines = []
            raw_idx = 0
            for pline in plain_lines:
                pline_words = pline.split(' ')
                target_word_count = len(pline_words)
                # Consume target_word_count plain words from para, keeping markup
                line_chars = []
                words_consumed = 0
                while words_consumed < target_word_count and raw_idx < len(para):
                    # Check for markup at current position
                    color_m = re.match(r'\{(#[0-9a-fA-F]{3,8}):([^}]+)\}', para[raw_idx:])
                    bold_m = re.match(r'\*([^*]+)\*', para[raw_idx:])
                    if color_m:
                        line_chars.append(color_m.group(0))
                        # Count plain words inside this markup
                        inner = color_m.group(2)
                        inner_words = [w for w in inner.split(' ') if w]
                        words_consumed += len(inner_words)
                        raw_idx += len(color_m.group(0))
                    elif bold_m:
                        line_chars.append(bold_m.group(0))
                        inner = bold_m.group(1)
                        inner_words = [w for w in inner.split(' ') if w]
                        words_consumed += len(inner_words)
                        raw_idx += len(bold_m.group(0))
                    elif para[raw_idx] == ' ':
                        line_chars.append(' ')
                        raw_idx += 1
                    else:
                        # Regular character — find end of word
                        word_end = raw_idx
                        while word_end < len(para) and para[word_end] != ' ' and para[word_end] not in '{*':
                            word_end += 1
                        line_chars.append(para[raw_idx:word_end])
                        words_consumed += 1
                        raw_idx = word_end
                # Skip trailing space between lines
                if raw_idx < len(para) and para[raw_idx] == ' ':
                    raw_idx += 1
                result_lines.append(''.join(line_chars).strip())

            # Append any remaining content
            if raw_idx < len(para):
                remaining = para[raw_idx:].strip()
                if remaining:
                    if result_lines:
                        result_lines[-1] += ' ' + remaining
                    else:
                        result_lines.append(remaining)

            wrapped_paragraphs.append('\n'.join(result_lines))

        return '\n'.join(wrapped_paragraphs)

    def render_slide(self, slide: dict, settings: dict, slide_num: int, total_slides: int, username_override: str = None) -> Image.Image:
        canvas = self.create_background(slide.get('background', {}))

        # NEW v7.0: Auto-height adaptation - PASS 1: Calculate line counts and Y offsets
        elements = slide.get('elements', [])
        y_offsets = {}  # {element_id: y_offset}

        # Сортируем элементы по Y для правильного смещения
        text_elements = [el for el in elements if el.get('type') != 'photo']
        sorted_text = sorted(text_elements, key=lambda e: e.get('y', 0))

        accumulated_offset = 0  # Накопленное смещение для элементов ниже

        for i, el in enumerate(sorted_text):
            auto_height = el.get('autoHeight', 'none')
            original_y = el.get('y', 0)

            # Применяем накопленное смещение от элементов выше
            y_offsets[el['id']] = accumulated_offset

            if auto_height == 'none':
                continue

            # Вычисляем количество строк
            lines_count = self.calculate_text_lines_count(el, settings, slide_num, total_slides, username_override)
            if lines_count <= 1:
                continue  # Одна строка - смещать нечего

            font_size = int(el.get('fontSize', 48))
            line_height = float(el.get('lineHeight', 1.2))

            # Реальная высота vs ожидаемая (1 строка)
            real_height = lines_count * font_size * line_height
            expected_height = font_size * line_height
            height_diff = int(real_height - expected_height)

            if auto_height == 'up':
                # Текст растёт вверх: сам элемент смещается вверх
                y_offsets[el['id']] = accumulated_offset - height_diff
            elif auto_height == 'down':
                # Текст растёт вниз: элементы ниже смещаются вниз
                accumulated_offset += height_diff
            elif auto_height == 'center':
                # Текст растёт по центру: сам элемент вверх, элементы ниже вниз
                half_diff = height_diff // 2
                y_offsets[el['id']] = accumulated_offset - half_diff
                accumulated_offset += half_diff

        # PASS 2: Render with offsets
        for el in elements:
            if el.get('type') == 'photo':
                self.draw_photo_element(canvas, el)
            else:
                offset = y_offsets.get(el['id'], 0)
                self.draw_text_element_with_offset(canvas, el, settings, slide_num, total_slides, username_override, offset)

        return canvas

    def draw_text_element_with_offset(self, canvas: Image.Image, el: dict, settings: dict, slide_num: int, total_slides: int, username_override: str = None, y_offset: int = 0):
        """Рендер текстового элемента с Y смещением"""
        # Создаём копию элемента чтобы не менять оригинал
        el_copy = el.copy()
        el_copy['y'] = el.get('y', 0) + y_offset
        self.draw_text_element(canvas, el_copy, settings, slide_num, total_slides, username_override)


renderer = SlideRenderer()

app.mount("/fonts", StaticFiles(directory="fonts"), name="fonts")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico", media_type="image/x-icon", headers={
        "Cache-Control": "no-cache, must-revalidate"
    })


@app.get("/")
async def index():
    return FileResponse("static/index.html", headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    })


@app.get("/google47e4976f6f044658.html")
async def google_verification():
    return FileResponse("static/google47e4976f6f044658.html")

@app.get("/privacy")
async def privacy_page():
    return FileResponse("static/privacy.html")

@app.get("/terms")
async def terms_page():
    return FileResponse("static/terms.html")

@app.get("/admin")
async def admin_page():
    return FileResponse("static/admin.html", headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    })


@app.post("/api/admin/login")
async def admin_login(request: Request):
    body = await request.json()
    pw = body.get("password", "")
    expected = os.getenv("ADMIN_PASSWORD", "")
    if not expected or pw != expected:
        raise HTTPException(status_code=403, detail="Wrong password")
    return {"ok": True}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "10.1",
        "fonts": os.listdir("fonts") if os.path.exists("fonts") else []
    }


@app.get("/fonts")
async def list_fonts():
    """Список всех доступных шрифтов (system, google, custom)"""
    # System fonts (встроенные)
    system_fonts = ["Inter", "Montserrat", "Roboto"]

    # Google Fonts (топ-100, можно расширить)
    google_fonts = [
        "Roboto", "Open Sans", "Lato", "Montserrat", "Oswald", "Raleway",
        "Poppins", "Ubuntu", "Merriweather", "Playfair Display", "PT Sans",
        "Nunito", "Work Sans", "Quicksand", "Crimson Text", "Bitter"
    ]

    # Custom fonts (загруженные пользователем)
    custom_fonts = []
    if os.path.exists('uploads'):
        for filename in os.listdir('uploads'):
            if filename.endswith(('.ttf', '.otf')):
                font_name = filename.rsplit('.', 1)[0]
                custom_fonts.append(font_name)

    return {
        "system": system_fonts,
        "google": google_fonts,
        "custom": custom_fonts
    }


@app.post("/upload-font")
async def upload_font(file: UploadFile):
    """Загрузить custom font (.ttf или .otf)"""
    if not file.filename.endswith(('.ttf', '.otf')):
        raise HTTPException(status_code=400, detail="Only .ttf and .otf files allowed")

    # Безопасное имя файла
    safe_filename = re.sub(r'[^a-zA-Z0-9_\-.]', '_', file.filename)
    file_path = os.path.join(UPLOADS_DIR, safe_filename)

    # Сохранить файл
    with open(file_path, 'wb') as f:
        content = await file.read()
        f.write(content)

    font_name = safe_filename.rsplit('.', 1)[0]

    return {
        "success": True,
        "fontName": font_name,
        "filename": safe_filename
    }


_thumbnail_cache = {}  # {hash(photo_uri[:100]): thumbnail_data_uri}

def _make_thumbnail(data_uri: str, max_w: int = 200) -> str:
    """Сжать base64 фото до маленького thumbnail для превью. С кэшем."""
    cache_key = hash(data_uri[:200])
    if cache_key in _thumbnail_cache:
        return _thumbnail_cache[cache_key]
    try:
        header, b64data = data_uri.split(',', 1)
        img_bytes = base64.b64decode(b64data)
        img = Image.open(io.BytesIO(img_bytes))
        ratio = max_w / img.width
        new_h = int(img.height * ratio)
        img = img.resize((max_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert('RGB').save(buf, format='JPEG', quality=50, optimize=True)
        thumb_b64 = base64.b64encode(buf.getvalue()).decode()
        result = f"data:image/jpeg;base64,{thumb_b64}"
    except Exception:
        result = ""
    _thumbnail_cache[cache_key] = result
    return result


@app.get("/templates")
async def list_templates(preview: str = "full", user_id: str = ""):
    """List templates from Supabase. Returns system + personal + published."""
    if not supabase:
        return {"templates": []}

    templates = []

    def format_row(row, ttype: str):
        all_slides = row.get("slides") or []
        if preview == 'light':
            # Первый слайд с thumbnail вместо полных base64 фото
            if all_slides:
                slide = dict(all_slides[0])
                # Сжать background.photo
                bg = slide.get('background')
                if bg and isinstance(bg, dict) and bg.get('photo', '').startswith('data:'):
                    bg_copy = dict(bg)
                    bg_copy['photo'] = _make_thumbnail(bg['photo'])
                    slide['background'] = bg_copy
                # Сжать element.photo в elements
                elements = slide.get('elements', [])
                new_elements = []
                for el in elements:
                    if el.get('type') == 'photo' and isinstance(el.get('photo'), str) and el['photo'].startswith('data:'):
                        el_copy = dict(el)
                        if len(el['photo']) > 500_000:
                            # >500KB base64 — слишком большой для Pillow в prod, убираем
                            el_copy['photo'] = ''
                        else:
                            el_copy['photo'] = _make_thumbnail(el['photo'], max_w=100)
                        new_elements.append(el_copy)
                    else:
                        new_elements.append(el)
                slide['elements'] = new_elements
                slides_data = [slide]
            else:
                slides_data = []
        else:
            slides_data = all_slides
        return {
            "name": row["name"],
            "template_id": row["template_id"],
            "createdAt": row.get("created_at", ""),
            "slidesCount": len(all_slides),
            "slides": slides_data,
            "type": ttype,
            "is_system": row.get("is_system", False),
            "is_published": row.get("is_published", False),
            "db_id": str(row["id"]),
            "owner_id": str(row.get("user_id", ""))
        }

    try:
        # 1. System templates (visible to all)
        sys_result = supabase.table("user_templates").select("*").eq("is_system", True).execute()
        for row in (sys_result.data or []):
            templates.append(format_row(row, "system"))

        if user_id:
            # 2. User's own non-system templates
            own_result = supabase.table("user_templates").select("*").eq("user_id", user_id).eq("is_system", False).execute()
            for row in (own_result.data or []):
                templates.append(format_row(row, "personal"))

            # 3. Published by others (non-system)
            pub_result = supabase.table("user_templates").select("*").eq("is_published", True).eq("is_system", False).neq("user_id", user_id).execute()
            for row in (pub_result.data or []):
                templates.append(format_row(row, "published"))
    except Exception as e:
        print(f"Error loading templates: {e}")

    return {"templates": templates}


@app.get("/templates/{identifier}")
async def get_template(identifier: str, user_id: str = ""):
    """Get single template by template_id from Supabase."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    try:
        result = supabase.table("user_templates").select("*").eq("template_id", identifier).execute()
        rows = result.data or []
        if rows:
            row = None
            # Приоритет: свой шаблон > системный > любой
            if user_id:
                own = [r for r in rows if str(r.get("user_id", "")) == user_id]
                if own:
                    row = own[0]
            if not row:
                sys_rows = [r for r in rows if r.get("is_system")]
                row = sys_rows[0] if sys_rows else rows[0]

            ttype = "system" if row.get("is_system") else ("published" if row.get("is_published") else "personal")
            return {
                "name": row["name"],
                "template_id": row["template_id"],
                "settings": row.get("settings") or {},
                "slides": row.get("slides") or [],
                "createdAt": row.get("created_at", ""),
                "type": ttype,
                "is_system": row.get("is_system", False),
                "is_published": row.get("is_published", False),
                "db_id": str(row["id"]),
                "owner_id": str(row.get("user_id", ""))
            }
    except Exception as e:
        print(f"Error getting template {identifier}: {e}")

    raise HTTPException(status_code=404, detail="Not found")


@app.post("/templates")
async def save_template(template: TemplateData):
    """Save template to Supabase. Requires user_id."""
    from datetime import datetime

    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    if not template.user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    # Приведение user_id к int (в БД колонка BIGINT)
    try:
        uid_int = int(template.user_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail=f"Invalid user_id: {template.user_id}")

    tid = template.template_id or generate_template_id(template.name)

    try:
        row = {
            "user_id": uid_int,
            "template_id": tid,
            "name": template.name,
            "slides": template.slides or [],
            "settings": template.settings or {},
            "updated_at": datetime.now().isoformat()
        }
        supabase.table("user_templates").upsert(row, on_conflict="user_id,template_id").execute()
        return {"success": True, "name": template.name, "template_id": tid, "type": "personal"}
    except Exception as e:
        print(f"Error saving template: {e}")
        raise HTTPException(status_code=500, detail=f"error.save: {str(e)}")


@app.put("/templates/{identifier}/publish")
async def toggle_publish_template(identifier: str, user_id: str = ""):
    """Toggle is_published on a user template (only owner can do this)."""
    if not supabase or not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    try:
        result = supabase.table("user_templates").select("id,is_published").eq("template_id", identifier).eq("user_id", user_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Template not found")
        row = result.data[0]
        new_state = not row.get("is_published", False)
        supabase.table("user_templates").update({"is_published": new_state}).eq("id", row["id"]).execute()
        return {"success": True, "is_published": new_state}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/templates/{identifier}/system")
async def toggle_system_template(identifier: str, user_id: str = ""):
    """Toggle is_system on a template. Admin only."""
    if not supabase or not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    if str(user_id) not in get_admin_ids():
        raise HTTPException(status_code=403, detail="Admin access required")
    try:
        result = supabase.table("user_templates").select("id,is_system").eq("template_id", identifier).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Template not found")
        row = result.data[0]
        new_state = not row.get("is_system", False)
        supabase.table("user_templates").update({"is_system": new_state}).eq("id", row["id"]).execute()
        return {"success": True, "is_system": new_state}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/templates/{identifier}")
async def delete_template(identifier: str, user_id: str = ""):
    """Delete template from Supabase."""
    if not supabase or not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    try:
        # Admin can delete any template, regular user only their own
        if str(user_id) in get_admin_ids():
            result = supabase.table("user_templates").delete().eq("template_id", identifier).execute()
        else:
            result = supabase.table("user_templates").delete().eq("template_id", identifier).eq("user_id", user_id).execute()
        if result.data and len(result.data) > 0:
            return {"success": True}
    except Exception as e:
        print(f"Error deleting template: {e}")

    raise HTTPException(status_code=404, detail="Not found")


@app.post("/admin/migrate-templates")
async def migrate_filesystem_templates(user_id: str = ""):
    """One-time migration: move filesystem templates to Supabase. Admin only."""
    from datetime import datetime

    if not supabase or not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    if str(user_id) not in get_admin_ids():
        raise HTTPException(status_code=403, detail="Admin access required")

    migrated = []
    errors = []

    for tdir in [TEMPLATES_DIR, GIT_TEMPLATES_DIR]:
        try:
            if not os.path.isdir(tdir):
                continue
            for f in os.listdir(tdir):
                if not f.endswith('.json'):
                    continue
                path = os.path.join(tdir, f)
                try:
                    with open(path, 'r', encoding='utf-8') as file:
                        d = json.load(file)

                    template_id = d.get('template_id')
                    if not template_id:
                        template_id = generate_template_id(d.get('name', f.replace('.json', '')))

                    row = {
                        "user_id": user_id,
                        "template_id": template_id,
                        "name": d.get('name', f.replace('.json', '')),
                        "slides": d.get('slides', []),
                        "settings": d.get('settings', {}),
                        "is_system": False,
                        "is_published": False,
                        "updated_at": datetime.now().isoformat()
                    }
                    supabase.table("user_templates").upsert(row, on_conflict="user_id,template_id").execute()
                    migrated.append(f)
                except Exception as e:
                    errors.append({"file": f, "error": str(e)})
        except Exception as e:
            errors.append({"dir": tdir, "error": str(e)})

    return {"migrated": migrated, "errors": errors, "total": len(migrated)}


@app.post("/render-slide")
async def render_slide(data: SlideData):
    try:
        img = renderer.render_slide(data.slide, data.settings, data.slideNumber, 10)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return {
            "success": True,
            "base64": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate_carousel(request: GenerateRequest, http_request: Request = None):
    """
    ПРАВИЛЬНАЯ ЛОГИКА v6.2:
    1. Первый слайд из request.slides → рендерится по INTRO template
    2. Остальные слайды из request.slides → рендерятся по CONTENT template
    3. Ending слайды → добавляются в конец (КАК ЕСТЬ из шаблона)

    NEW v7.0: Поиск шаблона по template_id с fallback на template_name
    """
    # --- Usage limit check ---
    _auth_id_for_limit = None
    _anon_token = None
    if http_request:
        _anon_token = _get_anon_token(http_request)
        # Try user_id from query params
        _uid = http_request.query_params.get("user_id", "")
        if _uid:
            try:
                tg_id = int(_uid)
                _acc = _get_auth_account_by_telegram(tg_id)
                if _acc:
                    _auth_id_for_limit = _acc["id"]
            except (ValueError, TypeError):
                pass
        # Try JWT
        if not _auth_id_for_limit:
            _auth_header = http_request.headers.get("Authorization", "")
            if _auth_header.startswith("Bearer "):
                _claims = verify_supabase_jwt(_auth_header[7:])
                if _claims and _claims.get("sub"):
                    _auth_id_for_limit = _claims["sub"]
    allowed, used, limit = await check_usage_limit(_auth_id_for_limit, _anon_token, "carousel_generate")
    if not allowed:
        raise HTTPException(status_code=429, detail={"error": "limit_reached", "used": used, "limit": limit, "action": "carousel_generate"})
    await increment_usage(_auth_id_for_limit, _anon_token, "carousel_generate")

    # NEW: приоритет template_id над template_name
    template_identifier = request.template_id or request.template_name
    username = request.username or request.USERNAME or "@username"

    if not template_identifier:
        raise HTTPException(status_code=400, detail="template_id or template_name required")

    # NEW: Поиск шаблона по ID или по имени
    template_path = None
    template = None

    # Сначала ищем по template_id или name в JSON файлах
    if os.path.exists(TEMPLATES_DIR):
        for filename in os.listdir(TEMPLATES_DIR):
            if not filename.endswith('.json'):
                continue

            path = os.path.join(TEMPLATES_DIR, filename)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    t = json.load(f)
                    # Ищем по template_id или по name
                    if t.get('template_id') == template_identifier or t.get('name') == template_identifier:
                        template_path = path
                        template = t
                        break
            except:
                continue

    # Fallback: старая логика поиска по имени файла (для обратной совместимости)
    if not template_path:
        safe = re.sub(r'[^a-zA-Z0-9_\-а-яА-ЯёЁ]', '_', template_identifier)
        fallback_path = os.path.join(TEMPLATES_DIR, f"{safe}.json")
        if os.path.exists(fallback_path):
            with open(fallback_path, 'r', encoding='utf-8') as f:
                template = json.load(f)
                template_path = fallback_path

    # Fallback: поиск в Supabase (пользовательские шаблоны)
    if not template and supabase:
        try:
            result = supabase.table("user_templates").select("*").eq("template_id", template_identifier).execute()
            if result.data:
                row = result.data[0]
                template = {
                    "name": row["name"],
                    "template_id": row["template_id"],
                    "settings": row.get("settings") or {},
                    "slides": row.get("slides") or [],
                }
        except Exception as e:
            print(f"Supabase template lookup error: {e}")

    if not template:
        raise HTTPException(status_code=404, detail=f"Template '{template_identifier}' not found")

    settings = template.get('settings', {})
    slides = template.get('slides', [])

    # NEW v7.0: Разделяем шаблоны (intro, content, content1-10, ending)
    intro_slides = [s for s in slides if s.get('type') == 'intro']
    content_slides = [s for s in slides if s.get('type') == 'content']
    ending_slides = [s for s in slides if s.get('type') == 'ending']

    # NEW: Собираем content1-10 templates
    content_specific = {}
    for i in range(1, 11):
        content_type = f'content{i}'
        matching = [s for s in slides if s.get('type') == content_type]
        if matching:
            content_specific[content_type] = matching[0]

    # Берём базовые шаблоны
    intro_template = intro_slides[0] if intro_slides else (content_slides[0] if content_slides else slides[0])

    # FIX: Smart fallback - если нет "content", используем "content1", потом любой слайд
    if content_slides:
        content_template = content_slides[0]
    elif 'content1' in content_specific:
        content_template = content_specific['content1']
    else:
        content_template = slides[0]  # Last resort fallback

    # NEW: Функция для выбора content template с "последний доступный" fallback
    def get_content_template(index: int):
        """
        Выбрать content{index} template или ближайший меньший.

        Логика: Если content{index} не существует, использовать максимальный
        доступный content{N} где N <= index.

        Пример: есть content1, content2, content3
        - index=2 → content2 (точное совпадение)
        - index=4 → content3 (ближайший: max([1,2,3]) = 3)
        - index=10 → content3 (ближайший: max([1,2,3]) = 3)
        """
        content_type = f'content{index}'
        if content_type in content_specific:
            return content_specific[content_type]

        # Найти максимальный доступный content{N} где N <= index
        available = []
        for i in range(1, index + 1):
            if f'content{i}' in content_specific:
                available.append(i)

        if available:
            max_num = max(available)
            return content_specific[f'content{max_num}']
        else:
            return content_template  # Fallback на content/content1

    result_slides = []

    if request.slides:
        # ПРАВИЛЬНЫЙ подсчёт
        total_slides = len(request.slides) + len(ending_slides)

        # Рендерим слайды с подстановкой
        for i, slide_vars in enumerate(request.slides):
            # NEW v7.0: Первый слайд → INTRO, остальные → content{i} с fallback на content
            if i == 0:
                template_to_use = intro_template
            else:
                template_to_use = get_content_template(i)

            slide = json.loads(json.dumps(template_to_use))

            # NEW: Override background from client request (color picker, photo, etc.)
            if '_background' in slide_vars:
                client_bg = slide_vars['_background']
                if isinstance(client_bg, dict):
                    if not slide.get('background'):
                        slide['background'] = {}
                    slide['background'].update(client_bg)

            # Подстановка PHOTO для background
            if 'PHOTO' in slide_vars and slide.get('background', {}).get('type') == 'photo':
                slide['background']['photo'] = slide_vars['PHOTO']

            # Подстановка переменных по varName
            for el in slide.get('elements', []):
                var_name = el.get('varName', '')

                if not var_name:
                    continue

                value = None
                color_value = None

                for key, val in slide_vars.items():
                    if key.upper() == var_name.upper():
                        value = val
                    elif key.upper() == f"{var_name.upper()}_COLOR":
                        color_value = val

                if value is not None:
                    if el.get('type') == 'photo':
                        el['photo'] = value
                    else:
                        el['content'] = value

                if color_value:
                    el['highlightColor'] = color_value

            result_slides.append(slide)

        # Добавляем ending слайды (БЕЗ изменений)
        for ending_slide in ending_slides:
            result_slides.append(json.loads(json.dumps(ending_slide)))
    else:
        result_slides = slides
        total_slides = len(slides)

    # Рендерим все слайды
    rendered = []

    for i, slide in enumerate(result_slides):
        img = renderer.render_slide(slide, settings, i + 1, total_slides, username)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        filename = f"slide_{i+1}_{uuid.uuid4().hex[:8]}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, 'wb') as f:
            f.write(buf.getvalue())

        rendered.append({
            "slide_number": i + 1,
            "url": f"https://app.craftopen.space/output/{filename}",
            "filename": filename,
            "base64": f"data:image/png;base64,{b64}"  # оставь на случай, если понадобится
        })

    return {
        "success": True,
        "slides": rendered
    }


@app.get("/output/{filename}")
async def get_output(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Not found")


# === Whisper Transcription ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    audio_data = await file.read()
    if len(audio_data) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        resp = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": (file.filename or "audio.webm", audio_data, file.content_type or "audio/webm")},
            data={"model": "whisper-1"}
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Chat Proxy ===
class ChatProxyRequest(BaseModel):
    webhook_url: str
    payload: Dict[str, Any]


@app.post("/api/chat")
async def chat_proxy(request: ChatProxyRequest):
    """Proxy chat requests to n8n webhook to avoid CORS issues."""
    try:
        resp = requests.post(
            request.webhook_url,
            json=request.payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="AI response timeout")
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


# === AI Chat via OpenRouter ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

SYSTEM_PROMPT_HEADLINES_GEN = """Ты — AI контент-менеджер для Instagram-каруселей. Ты помогаешь создавать контент на основе идеи пользователя.

Сейчас этап: ГЕНЕРАЦИЯ ЗАГОЛОВКОВ.
На основе идеи пользователя предложи 5-7 цепляющих заголовков для карусели.

Формат ответа:
1. Заголовок 1
2. Заголовок 2
...

Пиши на языке пользователя. Заголовки должны быть короткими (3-7 слов), цепляющими, вызывать любопытство."""

SYSTEM_PROMPT_TEXT_GEN = """Ты — AI контент-менеджер для Instagram-каруселей. Пользователь выбрал заголовок. Теперь напиши текст для слайдов карусели.

Формат ответа СТРОГО:
Слайд 1
Заголовок: [заголовок карусели]
Описание: [подзаголовок или вступление]

Слайд 2
Заголовок: [пункт 1]
Описание: [раскрытие пункта 1]

... и так далее

Пиши кратко, ёмко. Каждый слайд — одна мысль. {slides_count} слайдов."""

SYSTEM_PROMPT_TEXT_READY = """Ты — AI контент-менеджер. Текст карусели готов. Помоги пользователю улучшить или отредактировать текст.
Если пользователь просит что-то изменить — перепиши соответствующие слайды в том же формате."""

SYSTEM_PROMPT_COMPETITOR_REWRITE = """Ты — AI контент-менеджер для Instagram-каруселей. Пользователь хочет переработать пост конкурента в свою уникальную карусель.

Задача: перепиши пост конкурента в формат карусели из {slides_count} слайдов.
- НЕ копируй текст конкурента дословно — создай уникальный контент на ту же тему
- Используй стиль и тон пользователя (если контекст о пользователе доступен)
- Сделай текст более вовлекающим и адаптированным под аудиторию пользователя
- Каждый слайд — одна ключевая мысль

Формат ответа СТРОГО:
Слайд 1
Заголовок: [цепляющий заголовок карусели]
Описание: [вступление]

Слайд 2
Заголовок: [пункт 1]
Описание: [раскрытие]

... и так далее на {slides_count} слайдов.

Пиши кратко, ёмко. Пиши на языке оригинального поста, если пользователь не указал иное."""

OCR_SLIDES_PROMPT = """Извлеки ТОЛЬКО контентный текст с каждого слайда карусели.

ИГНОРИРУЙ элементы интерфейса Instagram:
- Имя пользователя и @username вверху слайда
- Аватар пользователя
- Счётчик слайдов (1/7, 2/10 и т.д.)
- Иконки (лайк, комментарий, поделиться, сохранить)
- Любые элементы навигации и UI

Извлекай ТОЛЬКО текст, который является частью дизайна/контента слайда.
Сохраняй оригинальный язык текста. Не добавляй своих комментариев.

Формат ответа:
Слайд 1:
<контентный текст с первого изображения>

Слайд 2:
<контентный текст со второго изображения>

Если на слайде нет контентного текста, напиши: (нет текста)"""

TRANSCRIPT_CLEAN_PROMPT = """Ты — текстовый редактор. Возьми сырую транскрипцию видео ниже и создай чистую, читаемую версию.

Задачи:
- Добавь правильную пунктуацию и заглавные буквы
- Раздели на логические абзацы
- Убери слова-паразиты (ну, типа, как бы, э-э, вот, получается, короче)
- Убери повторы и запинки
- Сохрани весь смысл и ключевую информацию
- НЕ добавляй ничего от себя, НЕ дополняй текст

Выведи только очищенный текст на том же языке, без комментариев."""

VIDEO_STRUCTURE_PROMPT = """Проанализируй транскрипцию видео и определи его структуру.

Формат ответа:

**Хук (первые секунды):** [чем цепляет внимание в начале]

**Основной посыл:** [главная мысль/тезис видео]

**Ключевые тезисы:**
- [тезис 1]
- [тезис 2]
- [тезис 3]
...

**Призыв к действию (CTA):** [что автор просит сделать зрителя, если есть. Если нет — "Не обнаружен"]

**Стиль подачи:** [тон, темп, уровень формальности, особенности речи]

Пиши на русском. Будь конкретен, приводи примеры из текста."""


def get_system_prompt(status: str, slides_count: int = 7, custom_prompts: dict = None) -> str:
    cp = custom_prompts or {}
    if status == 'draft':
        return cp.get('draft') or SYSTEM_PROMPT_HEADLINES_GEN
    elif status == 'headlines':
        custom = cp.get('headlines')
        if custom:
            return custom.replace('{slides_count}', str(slides_count))
        return SYSTEM_PROMPT_TEXT_GEN.replace('{slides_count}', str(slides_count))
    elif status == 'text_ready':
        return cp.get('text_ready') or SYSTEM_PROMPT_TEXT_READY
    return cp.get('draft') or SYSTEM_PROMPT_HEADLINES_GEN


def get_user_context(user_id: int, query: str = "") -> str:
    """Get user profile and memory for personalization."""
    if not supabase:
        return ""
    context_parts = []
    try:
        user = supabase.table("users").select("profile_summary").eq("user_id", str(user_id)).single().execute()
        if user.data and user.data.get("profile_summary"):
            context_parts.append(f"Профиль пользователя: {user.data['profile_summary']}")
    except Exception as e:
        print(f"[user_context] Failed to get profile for user {user_id}: {e}")
    # Vector search for relevant memories
    try:
        memories = get_relevant_memories(user_id, query)
        if memories:
            context_parts.append("Факты о пользователе:\n" + "\n".join(f"- {m}" for m in memories))
    except Exception as e:
        print(f"[user_context] Failed to get memories for user {user_id}: {e}")
    return "\n".join(context_parts)


def _create_embedding(text: str) -> list:
    """Create embedding via OpenAI text-embedding-3-small."""
    if not OPENAI_API_KEY:
        return []
    try:
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": "text-embedding-3-small", "input": text},
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()["data"][0]["embedding"]
    except Exception:
        pass
    return []


def get_relevant_memories(user_id: int, query: str, limit: int = 5) -> list:
    """Vector search for relevant user memories from Supabase."""
    if not supabase:
        return []
    if query:
        embedding = _create_embedding(query)
        if embedding:
            try:
                result = supabase.rpc("match_memories", {
                    "query_embedding": embedding,
                    "match_user_id": str(user_id),
                    "match_limit": limit,
                    "match_threshold": 0.6
                }).execute()
                return [r["content"] for r in (result.data or [])]
            except Exception as e:
                print(f"[memories] Vector search failed for user {user_id}: {e}")
    # Fallback: get recent memories without vector search
    try:
        result = supabase.table("user_memory").select("content").eq("user_id", str(user_id)).order("created_at", desc=True).limit(limit).execute()
        return [r["content"] for r in (result.data or [])]
    except Exception as e:
        print(f"[memories] Fallback memory fetch failed for user {user_id}: {e}")
        return []


def _extract_and_save_memories_sync(user_id: int, messages: list):
    """Background thread: extract important facts from conversation and save to memory."""
    import json as _json
    print(f"[Memory] Starting extraction for user {user_id}, {len(messages)} messages")
    if not OPENROUTER_API_KEY or not supabase:
        print(f"[Memory] Skipped — OPENROUTER_API_KEY={bool(OPENROUTER_API_KEY)}, supabase={bool(supabase)}")
        return
    try:
        # Load memory_extract prompt from DB
        prompt_content, memory_model = get_system_prompt_v3("memory_extract")
        if "ГЕНЕРАЦИЯ ЗАГОЛОВКОВ" in prompt_content or not prompt_content:
            memory_model = "openai/gpt-4o-mini"
            prompt_content = None
        if not prompt_content:
            prompt_content = (
                'Проанализируй последние сообщения диалога. Если есть информация о стиле, '
                'предпочтениях, бизнесе, нише, аудитории пользователя — верни JSON:\n'
                '{"memories": ["факт1", "факт2"]}\n'
                'Если ничего важного — верни:\n{"memories": []}'
            )

        recent = messages[-6:] if len(messages) > 6 else messages
        conversation = "\n".join(f"{m['role']}: {m['content']}" for m in recent if m.get('content'))

        ai_text, _usage = _call_openrouter([
            {"role": "system", "content": prompt_content},
            {"role": "user", "content": conversation}
        ], memory_model)
        print(f"[Memory] AI response: {ai_text[:200]}")
        _log_ai_call(user_id, "memory_extract", "memory_extract", memory_model,
                     prompt_content, None, None, conversation, ai_text, _usage)

        # Parse JSON
        try:
            start = ai_text.find('{')
            end = ai_text.rfind('}') + 1
            if start >= 0 and end > start:
                data = _json.loads(ai_text[start:end])
                memories = data.get("memories", [])
            else:
                memories = []
        except Exception as parse_err:
            print(f"[Memory] JSON parse error: {parse_err}")
            memories = []

        if not memories:
            print(f"[Memory] No memories extracted for user {user_id}")
            return

        print(f"[Memory] Extracted {len(memories)} memories: {memories}")

        saved = 0
        for mem in memories:
            if not mem or len(mem.strip()) < 5:
                continue
            embedding = _create_embedding(mem)
            memory_data = {
                "user_id": str(user_id),
                "content": mem.strip(),
                "metadata": {"source": "auto_extract"},
            }
            if embedding:
                memory_data["embedding"] = embedding
            else:
                print(f"[Memory] Warning: embedding failed, saving without vector for user {user_id}")
            try:
                supabase.table("user_memory").insert(memory_data).execute()
                saved += 1
            except Exception as save_err:
                print(f"[Memory] Save error: {save_err}")

        print(f"[Memory] Saved {saved} memories for user {user_id}")

        # Update memory_count and auto-summarize
        try:
            user = supabase.table("users").select("memory_count").eq("user_id", str(user_id)).single().execute()
            new_count = (user.data.get("memory_count") or 0) + saved if user.data else saved
            supabase.table("users").update({"memory_count": new_count}).eq("user_id", str(user_id)).execute()

            if saved > 0:
                _auto_summarize_profile_sync(user_id)
                print(f"[Memory] Auto-summarized profile for user {user_id}")
        except Exception as count_err:
            print(f"[Memory] Count update error: {count_err}")

    except Exception as e:
        import traceback
        print(f"Memory extraction error: {e}\n{traceback.format_exc()}")


def _auto_summarize_profile_sync(user_id: int):
    """Summarize all memories into a profile summary (sync)."""
    if not supabase or not OPENROUTER_API_KEY:
        return
    try:
        result = supabase.table("user_memory").select("content").eq("user_id", str(user_id)).order("created_at", desc=True).limit(30).execute()
        if not result.data:
            return
        memories_text = "\n".join(f"- {r['content']}" for r in result.data)

        prompt, profile_model = get_system_prompt_v3("profile_summarize", variables={"memories": memories_text})
        if not prompt or prompt == SYSTEM_PROMPT_HEADLINES_GEN:
            prompt = f"На основе следующих фактов о пользователе создай краткий профиль (3-5 предложений): его ниша, стиль контента, целевая аудитория, предпочтения.\n\nФакты:\n{memories_text}"
            profile_model = "openai/gpt-4o-mini"

        summary, _usage = _call_openrouter([
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Создай профиль пользователя."}
        ], profile_model)
        _log_ai_call(user_id, "profile_summarize", "profile_summarize", profile_model,
                     prompt, None, None, None, summary, _usage)

        supabase.table("users").update({"profile_summary": summary}).eq("user_id", str(user_id)).execute()
        print(f"[Memory] Profile summary updated for user {user_id}")
    except Exception as e:
        print(f"Profile summarize error: {e}")


def _run_memory_extraction(user_id: int, messages: list):
    """Launch memory extraction in a background thread."""
    import threading
    t = threading.Thread(target=_extract_and_save_memories_sync, args=(user_id, messages), daemon=True)
    t.start()


# === v3: Prompt loading from DB with cache ===
_prompt_cache = {}
_prompt_cache_time = 0
PROMPT_CACHE_TTL = 60  # seconds


def get_system_prompt_v3(chat_type: str, has_slides: bool = False, variables: dict = None):
    """Load prompt from system_prompts table with caching. Returns (content, model)."""
    global _prompt_cache, _prompt_cache_time

    # Direct mapping: chat_type = prompt_key (editing/carousel removed)
    prompt_key = chat_type  # headlines, text, memory_extract, profile_summarize, format_text

    now = time.time()
    if now - _prompt_cache_time > PROMPT_CACHE_TTL:
        try:
            sb = require_supabase()
            result = sb.table("system_prompts").select("prompt_key,content,model").eq("is_active", True).execute()
            _prompt_cache = {row["prompt_key"]: {"content": row["content"], "model": row.get("model", "openai/gpt-4o")} for row in (result.data or [])}
            _prompt_cache_time = now
        except Exception:
            pass

    # Fallback to hardcoded prompts
    fallbacks = {
        'headlines': SYSTEM_PROMPT_HEADLINES_GEN,
        'text': SYSTEM_PROMPT_TEXT_GEN,
        'competitor_rewrite': SYSTEM_PROMPT_COMPETITOR_REWRITE,
        'ocr_slides': OCR_SLIDES_PROMPT,
        'transcript_clean': TRANSCRIPT_CLEAN_PROMPT,
        'video_structure': VIDEO_STRUCTURE_PROMPT,
        'format_text': open(os.path.join(os.path.dirname(__file__), "prompts", "format_carousel_text.txt"), encoding="utf-8").read() if os.path.exists(os.path.join(os.path.dirname(__file__), "prompts", "format_carousel_text.txt")) else "Parse text into JSON array of slides with TITLE and DESCRIPTION fields."
    }

    prompt_data = _prompt_cache.get(prompt_key)
    if prompt_data:
        template = prompt_data["content"]
        model = prompt_data.get("model", "openai/gpt-4o")
    else:
        template = fallbacks.get(prompt_key, SYSTEM_PROMPT_HEADLINES_GEN)
        fallback_models = {'format_text': 'openai/gpt-4o-mini', 'memory_extract': 'openai/gpt-4o-mini', 'profile_summarize': 'openai/gpt-4o-mini', 'ocr_slides': 'google/gemini-2.0-flash-001', 'transcript_clean': 'openai/gpt-4o-mini', 'video_structure': 'openai/gpt-4o-mini'}
        model = fallback_models.get(prompt_key, "openai/gpt-4o")

    if variables:
        for key, value in variables.items():
            template = template.replace(f'{{{key}}}', str(value))

    return template, model


def _call_openrouter(messages: list, model: str = "openai/gpt-4o", temperature: float = 0.7) -> tuple:
    """Call OpenRouter API. Returns (content_str, usage_dict)."""
    _start = time.time()
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://app.craftopen.space",
            "X-Title": "Craft AI"
        },
        json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": 4096},
        timeout=120
    )
    _elapsed = int((time.time() - _start) * 1000)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"OpenRouter error: {resp.text}")
    data = resp.json()
    _usage = data.get("usage", {})
    return data["choices"][0]["message"]["content"], {
        "input_tokens": _usage.get("prompt_tokens", 0),
        "output_tokens": _usage.get("completion_tokens", 0),
        "total_tokens": _usage.get("total_tokens", 0),
        "response_time_ms": _elapsed,
    }


def _extract_slides_text(images: list, user_id: int = None) -> list:
    """Extract text from carousel slide images using Gemini Flash vision via OpenRouter.
    Returns list of dicts: [{"slide": 1, "text": "..."}, ...]
    Non-fatal: returns [] on any error.
    """
    image_items = [img for img in images if not img.get("is_video") and img.get("url")]
    if not image_items:
        return []

    # Load OCR prompt from admin panel (with fallback to hardcoded)
    ocr_prompt, _ocr_model = get_system_prompt_v3("ocr_slides")

    # Build multimodal message with image URLs
    content_parts = [
        {"type": "text", "text": ocr_prompt}
    ]
    for img in image_items:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": img["url"]}
        })

    messages = [{"role": "user", "content": content_parts}]

    try:
        content, usage = _call_openrouter(
            messages=messages,
            model=_ocr_model,
            temperature=0.1
        )
    except Exception as e:
        print(f"[OCR] Gemini URL mode failed: {e}, trying base64 fallback...")
        # Fallback: download images and send as base64
        try:
            import base64
            content_parts_b64 = [content_parts[0]]  # keep text prompt
            for img in image_items:
                try:
                    img_resp = requests.get(img["url"], timeout=15)
                    if img_resp.status_code == 200:
                        b64 = base64.b64encode(img_resp.content).decode("utf-8")
                        # Detect content type
                        ct = img_resp.headers.get("content-type", "image/jpeg")
                        content_parts_b64.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{ct};base64,{b64}"}
                        })
                    else:
                        print(f"[OCR] Failed to download image: HTTP {img_resp.status_code}")
                except Exception as dl_err:
                    print(f"[OCR] Image download error: {dl_err}")

            if len(content_parts_b64) <= 1:
                print("[OCR] No images downloaded, giving up")
                return []

            messages_b64 = [{"role": "user", "content": content_parts_b64}]
            content, usage = _call_openrouter(
                messages=messages_b64,
                model=_ocr_model,
                temperature=0.1
            )
        except Exception as e2:
            print(f"[OCR] Base64 fallback also failed: {e2}")
            return []

    # Parse response: split by "Слайд N:"
    slides_text = []
    parts = re.split(r'Слайд\s+(\d+)\s*:', content)
    # parts = ['preamble', '1', 'text1', '2', 'text2', ...]
    for i in range(1, len(parts) - 1, 2):
        slide_num = int(parts[i])
        text = parts[i + 1].strip()
        if text and text != "(нет текста)":
            slides_text.append({"slide": slide_num, "text": text})

    # Log the OCR call
    try:
        _log_ai_call(
            user_id=user_id or 0,
            endpoint="competitor_analyze",
            prompt_key="ocr_slides",
            model=_ocr_model,
            system_prompt="",
            user_context="",
            messages=f"[{len(image_items)} images]",
            user_message=f"OCR {len(image_items)} carousel slides",
            ai_response=content[:2000],
            usage=usage,
            status="success"
        )
    except Exception:
        pass

    print(f"[OCR] ✅ Extracted text from {len(slides_text)}/{len(image_items)} slides, "
          f"tokens: {usage.get('total_tokens', '?')}")
    return slides_text


def _log_ai_call(user_id, endpoint, prompt_key, model, system_prompt,
                 user_context, messages, user_message, ai_response,
                 usage, status="success", error_message=None,
                 topic_id=None, sub_chat_id=None):
    """Log AI call to ai_logs table via direct PG. Non-blocking, never raises."""
    if not DATABASE_URL:
        return
    try:
        _pg_query("""
            INSERT INTO ai_logs (user_id, endpoint, prompt_key, model, system_prompt,
                user_context, messages, user_message, ai_response,
                input_tokens, output_tokens, total_tokens, response_time_ms,
                status, error_message, topic_id, sub_chat_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, [
            int(user_id) if user_id else None,
            endpoint,
            prompt_key,
            model,
            (system_prompt or "")[:5000],
            (user_context or "")[:2000] or None,
            json.dumps(messages[-5:]) if messages else None,
            (user_message or "")[:2000] or None,
            (ai_response or "")[:5000],
            usage.get("input_tokens", 0) if usage else 0,
            usage.get("output_tokens", 0) if usage else 0,
            usage.get("total_tokens", 0) if usage else 0,
            usage.get("response_time_ms", 0) if usage else 0,
            status,
            error_message,
            str(topic_id) if topic_id else None,
            str(sub_chat_id) if sub_chat_id else None,
        ])
    except Exception as e:
        print(f"[ai_logs] Failed to log: {e}")


def _parse_headlines(text: str) -> list:
    """Parse headlines from AI response. Supports: 1. / 1) / 1: / - / • / * formats."""
    headlines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Numbered: "1. Text", "1) Text", "1: Text", "1- Text", "1 Text"
        match = re.match(r'^\d+[\.\):\-]?\s+(.+)', line)
        if match:
            headlines.append(match.group(1).strip())
            continue
        # Bulleted: "- Text", "• Text", "* Text"
        match = re.match(r'^[\-\•\*]\s+(.+)', line)
        if match:
            headlines.append(match.group(1).strip())
    return headlines


# === Format Text API (carousel text parser) ===

@app.post("/api/format-text")
async def format_text(request: Request):
    """Parse raw user text into structured carousel slides (TITLE + DESCRIPTION)."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    body = await request.json()
    user_text = body.get("text", "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="text is required")

    # Load prompt and model from system_prompts (Supabase) with fallback
    system_prompt, model = get_system_prompt_v3("format_text")
    print(f"[format-text] Using model: {model}, prompt length: {len(system_prompt)} chars")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text + "\n\nReturn ONLY valid JSON (no markdown, no ```json blocks)."}
    ]

    try:
        result, _usage = _call_openrouter(messages, model, temperature=0.3)
        _log_ai_call(None, "format_text", "format_text", model,
                     system_prompt, None, None, user_text, result, _usage)
        # Strip markdown code block wrapper if present
        result = result.strip()
        if result.startswith("```"):
            result = re.sub(r'^```(?:json)?\s*', '', result)
            result = re.sub(r'\s*```$', '', result)
        slides = json.loads(result)
        if not isinstance(slides, list):
            slides = [slides]
        print(f"[format-text] OK — parsed {len(slides)} slides")
        return {"success": True, "slides": slides}
    except json.JSONDecodeError as e:
        print(f"[format-text] ERROR JSON parse: {e}\nRaw response: {result[:500]}")
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response as JSON: {str(e)}")
    except Exception as e:
        print(f"[format-text] ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Format text error: {str(e)}")


@app.post("/api/wrap-text")
async def wrap_text(request: Request):
    """Calculate PIL word-wrap for text elements and return wrapped content."""
    body = await request.json()
    elements = body.get("elements", [])
    results = []
    for el in elements:
        content = el.get("content", "")
        wrapped = renderer.wrap_text_for_preview(
            content,
            font_family=el.get("fontFamily", "Inter"),
            font_size=int(el.get("fontSize", 48)),
            font_weight=str(el.get("fontWeight", "400")),
            max_width=int(el.get("maxWidth")) if el.get("maxWidth") else None,
            letter_spacing=float(el.get("letterSpacing", 0))
        )
        results.append({"id": el.get("id", ""), "wrapped": wrapped})
    return results


@app.post("/api/debug/text-width")
async def debug_text_width(request: Request):
    """Debug: measure text width with PIL to compare with browser Canvas measureText."""
    body = await request.json()
    text = body.get("text", "")
    font_family = body.get("fontFamily", "Inter")
    font_size = int(body.get("fontSize", 48))
    font_weight = str(body.get("fontWeight", "700"))
    letter_spacing = float(body.get("letterSpacing", 0))
    max_width = int(body.get("maxWidth", 984)) if body.get("maxWidth") else None

    font = renderer.get_font(font_family, font_size, font_weight)
    temp_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(temp_img)

    # Measure full text width
    full_width = renderer.get_text_width(text, font, letter_spacing, draw, font_size)

    # Also measure word-by-word wrapping
    words = text.split(' ')
    lines = []
    current_words = []
    for word in words:
        if not word:
            continue
        test = ' '.join(current_words + [word])
        w = renderer.get_text_width(test, font, letter_spacing, draw, font_size)
        if max_width and w > max_width and current_words:
            lines.append(' '.join(current_words))
            current_words = [word]
        else:
            current_words.append(word)
    if current_words:
        lines.append(' '.join(current_words))

    # Measure each word
    word_widths = {}
    for word in set(words):
        if word:
            word_widths[word] = renderer.get_text_width(word, font, letter_spacing, draw, font_size)

    return {
        "full_width": full_width,
        "max_width": max_width,
        "lines_count": len(lines),
        "lines": lines,
        "line_widths": [renderer.get_text_width(l, font, letter_spacing, draw, font_size) for l in lines],
        "word_widths": word_widths,
        "font": f"{font_weight} {font_size}px {font_family}",
    }


# === Competitor Analysis API ===

def _parse_instagram(data: dict) -> dict:
    """Parse Scrape Creators Instagram response into normalized format."""
    media = data.get("data", {}).get("xdt_shortcode_media") or data.get("data", {})
    if not media:
        return None

    # Caption
    caption_edges = media.get("edge_media_to_caption", {}).get("edges", [])
    caption = caption_edges[0]["node"]["text"] if caption_edges else media.get("caption", {}).get("text", "") if isinstance(media.get("caption"), dict) else str(media.get("caption", ""))

    # Owner
    owner = media.get("owner", {})

    # Images
    images = []
    typename = media.get("__typename", "")
    if typename == "XDTGraphSidecar" or "edge_sidecar_to_children" in media:
        children = media.get("edge_sidecar_to_children", {}).get("edges", [])
        for child in children:
            node = child.get("node", {})
            images.append({
                "url": node.get("display_url", ""),
                "is_video": node.get("is_video", False),
            })
    else:
        display_url = media.get("display_url", "")
        if display_url:
            images.append({
                "url": display_url,
                "is_video": media.get("is_video", False),
            })

    # Engagement
    like_count = media.get("edge_media_preview_like", {}).get("count", 0) if isinstance(media.get("edge_media_preview_like"), dict) else media.get("like_count", 0)
    comment_count = media.get("edge_media_to_parent_comment", {}).get("count", 0) if isinstance(media.get("edge_media_to_parent_comment"), dict) else media.get("comment_count", 0)

    is_video = media.get("is_video", False)
    post_type = "carousel" if (typename == "XDTGraphSidecar" or len(images) > 1) else ("reel" if is_video else "post")

    return {
        "platform": "instagram",
        "username": owner.get("username", ""),
        "profile_pic_url": owner.get("profile_pic_url", ""),
        "caption": caption,
        "images": images[:10],
        "post_type": post_type,
        "is_video": is_video,
        "like_count": like_count,
        "comment_count": comment_count,
        "taken_at": media.get("taken_at_timestamp") or media.get("taken_at"),
        "shortcode": media.get("shortcode", ""),
    }


def _parse_youtube(data: dict, source_url: str = "") -> dict:
    """Parse Scrape Creators YouTube response into normalized format."""
    video = data.get("data", data)
    if isinstance(video, list) and video:
        video = video[0]

    title = video.get("title", "")
    description = video.get("description", "")
    caption = f"{title}\n\n{description}" if description else title

    thumbnail = video.get("thumbnail", "") or video.get("thumbnailUrl", "")
    if isinstance(thumbnail, list) and thumbnail:
        thumbnail = thumbnail[-1].get("url", "") if isinstance(thumbnail[-1], dict) else str(thumbnail[-1])

    images = [{"url": thumbnail, "is_video": True}] if thumbnail else []

    # Check both API response URL and original source URL for /shorts/
    is_short = "/shorts/" in str(video.get("url", "")) or "/shorts/" in source_url
    return {
        "platform": "youtube",
        "username": video.get("channelName", "") or video.get("author", "") or video.get("ownerChannelName", ""),
        "profile_pic_url": video.get("channelThumbnail", "") or video.get("authorThumbnail", ""),
        "caption": caption,
        "images": images,
        "post_type": "short" if is_short else "video",
        "is_video": True,
        "like_count": video.get("likeCount", 0) or video.get("likes", 0),
        "comment_count": video.get("commentCount", 0) or video.get("comments", 0),
        "taken_at": None,
        "shortcode": video.get("videoId", "") or video.get("id", ""),
    }


def _parse_tiktok(data: dict) -> dict:
    """Parse Scrape Creators TikTok response into normalized format."""
    video = data.get("data", data)
    if isinstance(video, list) and video:
        video = video[0]

    author = video.get("author", {}) if isinstance(video.get("author"), dict) else {}
    stats = video.get("stats", {}) if isinstance(video.get("stats"), dict) else {}

    # Check for photo carousel (imagePost)
    image_post = video.get("imagePost", {})
    if not isinstance(image_post, dict):
        image_post = {}
    carousel_images_raw = image_post.get("images", [])
    if not isinstance(carousel_images_raw, list):
        carousel_images_raw = []

    # Extract carousel image URLs
    carousel_images = []
    for img in carousel_images_raw:
        if isinstance(img, str) and img:
            carousel_images.append(img)
        elif isinstance(img, dict):
            img_url = img.get("imageURL", "") or img.get("imageUri", "") or img.get("url", "")
            if img_url:
                carousel_images.append(img_url)

    is_carousel = len(carousel_images) > 0

    if is_carousel:
        images = [{"url": u, "is_video": False} for u in carousel_images[:20]]
        post_type = "carousel"
        is_video = False
    else:
        cover = video.get("cover", "") or video.get("originCover", "")
        if isinstance(cover, list) and cover:
            cover = cover[0] if isinstance(cover[0], str) else ""
        images = [{"url": cover, "is_video": True}] if cover else []
        post_type = "video"
        is_video = True

    return {
        "platform": "tiktok",
        "username": author.get("uniqueId", "") or video.get("author_name", "") or video.get("uniqueId", ""),
        "profile_pic_url": author.get("avatarThumb", "") or video.get("author_avatar", ""),
        "caption": video.get("desc", "") or video.get("title", "") or video.get("description", ""),
        "images": images,
        "post_type": post_type,
        "is_video": is_video,
        "like_count": stats.get("diggCount", 0) or video.get("likes", 0) or video.get("diggCount", 0),
        "comment_count": stats.get("commentCount", 0) or video.get("comments", 0) or video.get("commentCount", 0),
        "taken_at": video.get("createTime"),
        "shortcode": video.get("id", "") or video.get("video_id", ""),
    }


def _parse_twitter(data: dict) -> dict:
    """Parse Scrape Creators Twitter/X response into normalized format."""
    tweet = data.get("data", data)
    if isinstance(tweet, list) and tweet:
        tweet = tweet[0]

    user = tweet.get("user", {}) if isinstance(tweet.get("user"), dict) else {}
    media = tweet.get("media", []) if isinstance(tweet.get("media"), list) else []

    images = []
    is_video = False
    for m in media:
        if isinstance(m, dict):
            if m.get("type") == "video":
                is_video = True
                images.append({"url": m.get("thumbnail_url", "") or m.get("preview_image_url", ""), "is_video": True})
            elif m.get("type") == "photo" or m.get("media_url_https"):
                images.append({"url": m.get("media_url_https", "") or m.get("url", ""), "is_video": False})

    return {
        "platform": "twitter",
        "username": user.get("screen_name", "") or user.get("username", "") or tweet.get("author_username", ""),
        "profile_pic_url": user.get("profile_image_url_https", "") or user.get("profile_image_url", ""),
        "caption": tweet.get("full_text", "") or tweet.get("text", ""),
        "images": images[:10],
        "post_type": "tweet",
        "is_video": is_video,
        "like_count": tweet.get("favorite_count", 0) or tweet.get("likes", 0),
        "comment_count": tweet.get("reply_count", 0) or tweet.get("replies", 0),
        "taken_at": None,
        "shortcode": tweet.get("id_str", "") or str(tweet.get("id", "")),
    }


def _parse_linkedin(data: dict) -> dict:
    """Parse Scrape Creators LinkedIn response into normalized format."""
    post = data.get("data", data)
    if isinstance(post, list) and post:
        post = post[0]

    author = post.get("author", {}) if isinstance(post.get("author"), dict) else {}
    images = []
    post_images = post.get("images", []) or post.get("media", [])
    if isinstance(post_images, list):
        for img in post_images[:10]:
            if isinstance(img, str):
                images.append({"url": img, "is_video": False})
            elif isinstance(img, dict):
                images.append({"url": img.get("url", "") or img.get("src", ""), "is_video": img.get("type") == "video"})

    return {
        "platform": "linkedin",
        "username": author.get("name", "") or post.get("author_name", "") or post.get("authorName", ""),
        "profile_pic_url": author.get("profilePicture", "") or author.get("avatar", "") or post.get("authorAvatar", ""),
        "caption": post.get("text", "") or post.get("commentary", "") or post.get("content", ""),
        "images": images,
        "post_type": "post",
        "is_video": False,
        "like_count": post.get("likeCount", 0) or post.get("likes", 0) or post.get("numLikes", 0),
        "comment_count": post.get("commentCount", 0) or post.get("comments", 0) or post.get("numComments", 0),
        "taken_at": None,
        "shortcode": post.get("id", "") or post.get("urn", ""),
    }


def _parse_facebook(data: dict) -> dict:
    """Parse Scrape Creators Facebook response into normalized format."""
    post = data.get("data", data)
    if isinstance(post, list) and post:
        post = post[0]

    images = []
    post_images = post.get("images", []) or post.get("attachments", [])
    if isinstance(post_images, list):
        for img in post_images[:10]:
            if isinstance(img, str):
                images.append({"url": img, "is_video": False})
            elif isinstance(img, dict):
                images.append({"url": img.get("url", "") or img.get("src", ""), "is_video": img.get("type") == "video"})

    return {
        "platform": "facebook",
        "username": post.get("author_name", "") or post.get("pageName", "") or post.get("username", ""),
        "profile_pic_url": post.get("author_avatar", "") or post.get("profilePicture", ""),
        "caption": post.get("text", "") or post.get("message", "") or post.get("content", ""),
        "images": images,
        "post_type": "video" if post.get("is_video") else "post",
        "is_video": post.get("is_video", False),
        "like_count": post.get("likes", 0) or post.get("likeCount", 0) or post.get("reactions", 0),
        "comment_count": post.get("comments", 0) or post.get("commentCount", 0),
        "taken_at": None,
        "shortcode": post.get("id", "") or post.get("post_id", ""),
    }


_PLATFORM_PARSERS = {
    "instagram": _parse_instagram,
    "youtube": _parse_youtube,
    "tiktok": _parse_tiktok,
    "twitter": _parse_twitter,
    "linkedin": _parse_linkedin,
    "facebook": _parse_facebook,
}


# Allowed domains for image proxy (security whitelist)
_PROXY_DOMAIN_PREFIXES = [
    "scontent", "instagram", "fbcdn", "cdninstagram",  # Instagram/FB CDN
    "yt3.ggpht", "i.ytimg", "lh3.googleusercontent",   # YouTube
    "p16-sign.tiktokcdn", "p77-sign.tiktokcdn", "v16-webapp.tiktok",  # TikTok
    "pbs.twimg", "abs.twimg",                            # Twitter
    "media.licdn",                                        # LinkedIn
]


@app.get("/api/image-proxy")
async def image_proxy(url: str):
    """Proxy external images to avoid CORS. Returns image from CDN."""
    if not url or not url.startswith("http"):
        raise HTTPException(400, "Invalid URL")

    # Security: only proxy images from known CDN domains
    from urllib.parse import urlparse
    hostname = urlparse(url).hostname or ""
    if not any(hostname.startswith(prefix) or prefix in hostname for prefix in _PROXY_DOMAIN_PREFIXES):
        raise HTTPException(403, f"Domain not allowed: {hostname}")

    try:
        client = await get_http_client()
        resp = await client.get(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; CraftAI/1.0)"
        })
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, "Image fetch failed")
        content_type = resp.headers.get("content-type", "image/jpeg")
        return Response(
            content=resp.content,
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=3600"}  # Cache 1 hour
        )
    except httpx.TimeoutException:
        raise HTTPException(504, "Image proxy timeout")
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Image proxy error: {str(e)}")


@app.post("/api/competitor/analyze")
async def competitor_analyze(request: Request):
    """Analyze a social media post via Scrape Creators API. Creates Topic + analysis sub-chat."""
    if not SCRAPE_CREATORS_API_KEY:
        raise HTTPException(status_code=500, detail="SCRAPE_CREATORS_API_KEY not configured")

    sb = require_supabase()
    body = await request.json()
    url = body.get("url", "").strip()
    user_id = body.get("user_id")

    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    # --- Usage limit check ---
    _auth_id_for_limit = None
    _anon_token = _get_anon_token(request)
    try:
        tg_id = int(user_id)
        _acc = _get_auth_account_by_telegram(tg_id)
        if _acc:
            _auth_id_for_limit = _acc["id"]
    except (ValueError, TypeError):
        pass
    allowed, used, limit = await check_usage_limit(_auth_id_for_limit, _anon_token, "competitor_analysis")
    if not allowed:
        raise HTTPException(status_code=429, detail={"error": "limit_reached", "used": used, "limit": limit, "action": "competitor_analysis"})
    await increment_usage(_auth_id_for_limit, _anon_token, "competitor_analysis")

    # Detect platform
    platform = detect_platform(url)
    if not platform:
        raise HTTPException(status_code=400, detail="error.unsupported_link")

    config = PLATFORM_CONFIG[platform]

    # Normalize URL — remove query params, trailing slashes, fix /reels/ → /reel/
    from urllib.parse import urlparse, urlunparse
    parsed_url = urlparse(url)
    clean_path = parsed_url.path.rstrip('/')
    # Instagram: /reels/CODE → /reel/CODE (Scrape Creators only accepts singular)
    if platform == "instagram":
        clean_path = re.sub(r'/reels/', '/reel/', clean_path)
    url = urlunparse((parsed_url.scheme, parsed_url.netloc, clean_path, '', '', ''))

    is_reel_url = bool(re.search(r'/reel/', url))

    try:
        # 1. Fetch post data
        print(f"[competitor] Fetching {platform} post: {url[:80]}...")
        resp = requests.get(
            f"{SCRAPE_CREATORS_BASE}{config['post_endpoint']}",
            params={"url": url},
            headers={"x-api-key": SCRAPE_CREATORS_API_KEY},
            timeout=30
        )

        # Fallback: if post endpoint fails for reels, try reel-specific endpoint
        if resp.status_code != 200 and platform == "instagram" and is_reel_url and config.get("reel_endpoint"):
            print(f"[competitor] Post endpoint failed ({resp.status_code}), trying reel endpoint...")
            reel_resp = requests.get(
                f"{SCRAPE_CREATORS_BASE}{config['reel_endpoint']}",
                params={"url": url},
                headers={"x-api-key": SCRAPE_CREATORS_API_KEY},
                timeout=30
            )
            if reel_resp.status_code == 200:
                resp = reel_resp

        if resp.status_code != 200:
            print(f"[competitor] Scrape Creators error {resp.status_code}: {resp.text[:500]}")
            if resp.status_code == 404:
                raise HTTPException(status_code=404, detail="error.post_not_found")
            raise HTTPException(status_code=502, detail=f"error.post_load_failed: {resp.status_code}")

        raw_data = resp.json()

        # 2. Parse with platform-specific parser
        parser = _PLATFORM_PARSERS.get(platform)
        if platform == "youtube":
            analysis = parser(raw_data, source_url=url)
        else:
            analysis = parser(raw_data)
        if not analysis:
            raise HTTPException(status_code=404, detail="error.post_unavailable")

        # 3. Try to get transcript for video content
        transcript = None
        if analysis["is_video"] and config.get("transcript_endpoint"):
            try:
                print(f"[competitor] Fetching transcript for {platform} video...")
                tr_resp = requests.get(
                    f"{SCRAPE_CREATORS_BASE}{config['transcript_endpoint']}",
                    params={"url": url},
                    headers={"x-api-key": SCRAPE_CREATORS_API_KEY},
                    timeout=30
                )

                if tr_resp.status_code == 200:
                    # Try to parse as JSON, fall back to plain text
                    try:
                        tr_data = tr_resp.json()
                    except Exception:
                        # v1 endpoint returns plain text
                        transcript = tr_resp.text.strip()
                        if transcript:
                            print(f"[competitor] Got transcript (plain text): {len(transcript)} chars")
                    else:
                        # v2 endpoint: {"success": true, "transcripts": [{"text": "..."}]}
                        if isinstance(tr_data.get("transcripts"), list) and tr_data["transcripts"]:
                            transcript = tr_data["transcripts"][0].get("text", "")
                        # Fallback: try "data" field (other platforms)
                        elif isinstance(tr_data.get("data"), str):
                            transcript = tr_data["data"]
                        elif isinstance(tr_data.get("data"), dict):
                            transcript = tr_data["data"].get("transcript", "") or tr_data["data"].get("text", "")
                        elif isinstance(tr_data.get("data"), list):
                            segments = tr_data["data"]
                            transcript = " ".join(
                                seg.get("text", "") if isinstance(seg, dict) else str(seg)
                                for seg in segments
                            ).strip()
                        if transcript:
                            print(f"[competitor] Got transcript: {len(transcript)} chars")
                        else:
                            print(f"[competitor] Transcript response OK but empty: {str(tr_data)[:300]}")
                else:
                    print(f"[competitor] Transcript unavailable (status {tr_resp.status_code})")
            except Exception as e:
                print(f"[competitor] Transcript fetch error: {e}")

        analysis["transcript"] = transcript or ""
        analysis["source_url"] = url

        # 3b. Extract text from carousel images (OCR via Gemini Flash)
        slides_text = []
        if analysis["images"] and not analysis["is_video"]:
            try:
                slides_text = _extract_slides_text(analysis["images"], user_id=int(user_id))
                print(f"[competitor] OCR extracted text from {len(slides_text)} slides")
            except Exception as e:
                print(f"[competitor] OCR failed (non-fatal): {e}")
        analysis["slides_text"] = slides_text

        # 4. Create Topic
        topic_title = f"Анализ @{analysis['username']}" if analysis['username'] else f"Анализ {platform}"
        # Use slides_text for idea_text if available, fallback to caption
        if slides_text:
            idea_text = "\n\n".join([f"Слайд {s['slide']}: {s['text']}" for s in slides_text])[:2000]
        else:
            idea_text = analysis["caption"][:2000] if analysis["caption"] else url
        topic = sb.table("projects").insert({
            "user_id": int(user_id),
            "title": topic_title[:50],
            "status": "draft",
            "idea_text": idea_text,
        }).execute()
        topic_data = topic.data[0]
        topic_id = topic_data["id"]

        # 5. Create analysis sub-chat
        sub_chat = sb.table("sub_chats").insert({
            "topic_id": topic_id,
            "user_id": int(user_id),
            "chat_type": "analysis",
            "title": "Анализ",
            "slides_data": {
                "platform": analysis["platform"],
                "username": analysis["username"],
                "source_url": url,
                "post_type": analysis["post_type"],
                "scraped_at": time.time(),
            },
        }).execute()
        sub_chat_data = sub_chat.data[0]
        sub_chat_id = sub_chat_data["id"]

        # 6. Save analysis as first message
        sb.table("project_messages").insert({
            "project_id": topic_id,
            "sub_chat_id": sub_chat_id,
            "user_id": int(user_id),
            "role": "assistant",
            "content": analysis["caption"][:5000],
            "message_type": "analysis",
            "message_data": analysis,
        }).execute()

        # Update topic's active sub-chat
        sb.table("projects").update({"active_subchat_id": sub_chat_id}).eq("id", topic_id).execute()

        print(f"[competitor] ✅ Analyzed {platform} @{analysis['username']}: "
              f"{len(analysis['caption'])} chars, {len(analysis['images'])} images, "
              f"{analysis['like_count']} likes. Topic: {topic_id}")

        return {
            "topic": topic_data,
            "sub_chat": sub_chat_data,
            "analysis": analysis,
        }

    except HTTPException:
        raise
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="error.timeout")
    except requests.RequestException as e:
        print(f"[competitor] Network error: {e}")
        raise HTTPException(status_code=502, detail=f"error.network: {str(e)}")
    except Exception as e:
        print(f"[competitor] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"error.analysis: {str(e)}")


@app.post("/api/competitor/process-transcript")
async def process_transcript(request: Request):
    """Process raw transcript with AI: clean version + structure analysis."""
    body = await request.json()
    user_id = body.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    transcript = body.get("transcript", "").strip()
    topic_id = body.get("topic_id")
    sub_chat_id = body.get("sub_chat_id")

    if not transcript or len(transcript) < 30:
        raise HTTPException(status_code=400, detail="error.transcription_short")

    # Truncate to avoid token limits
    transcript_input = transcript[:8000]

    processed_transcript = ""
    video_structure = ""

    # 1. Clean transcript
    try:
        clean_prompt, clean_model = get_system_prompt_v3("transcript_clean")
        processed_transcript, _ = _call_openrouter(
            [{"role": "system", "content": clean_prompt},
             {"role": "user", "content": transcript_input}],
            model=clean_model, temperature=0.3
        )
        print(f"[process-transcript] Clean transcript: {len(processed_transcript)} chars")
    except Exception as e:
        print(f"[process-transcript] Clean error (non-fatal): {e}")

    # 2. Analyze structure
    try:
        struct_prompt, struct_model = get_system_prompt_v3("video_structure")
        video_structure, _ = _call_openrouter(
            [{"role": "system", "content": struct_prompt},
             {"role": "user", "content": transcript_input}],
            model=struct_model, temperature=0.5
        )
        print(f"[process-transcript] Structure analysis: {len(video_structure)} chars")
    except Exception as e:
        print(f"[process-transcript] Structure error (non-fatal): {e}")

    # 3. Update stored message_data if sub_chat_id provided
    if sub_chat_id and supabase and (processed_transcript or video_structure):
        try:
            msgs = supabase.table("project_messages").select("id,message_data").eq(
                "sub_chat_id", sub_chat_id
            ).eq("message_type", "analysis").order("created_at", desc=True).limit(1).execute()
            if msgs.data:
                msg = msgs.data[0]
                msg_data = msg.get("message_data", {}) or {}
                if processed_transcript:
                    msg_data["processed_transcript"] = processed_transcript
                if video_structure:
                    msg_data["video_structure"] = video_structure
                supabase.table("project_messages").update(
                    {"message_data": msg_data}
                ).eq("id", msg["id"]).execute()
                print(f"[process-transcript] Updated message {msg['id']} with processed data")
        except Exception as e:
            print(f"[process-transcript] DB update error (non-fatal): {e}")

    return {
        "processed_transcript": processed_transcript,
        "video_structure": video_structure,
    }


# === v3: Topics API ===

@app.get("/api/topics")
async def list_topics(user_id: int):
    """List all topics for a user with sub-chat counts."""
    sb = require_supabase()
    topics = sb.table("projects").select("*").eq("user_id", user_id).order("updated_at", desc=True).execute()
    result = []
    for t in (topics.data or []):
        sub_chats = sb.table("sub_chats").select("id,chat_type").eq("topic_id", t["id"]).execute()
        t["sub_chat_count"] = len(sub_chats.data or [])
        result.append(t)
    return result


@app.post("/api/topics")
async def create_topic(request: Request):
    """Create topic + headlines sub-chat + AI response in one call."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")
    sb = require_supabase()
    body = await request.json()
    user_id = body.get("user_id")
    message = body.get("message", "")

    if not user_id or not message:
        raise HTTPException(status_code=400, detail="user_id and message required")

    # 1. Create topic (project)
    topic = sb.table("projects").insert({
        "user_id": int(user_id),
        "title": message[:50],
        "status": "draft",
        "idea_text": message,
    }).execute()
    topic_data = topic.data[0]
    topic_id = topic_data["id"]

    # 2. Create headlines sub-chat
    sub_chat = sb.table("sub_chats").insert({
        "topic_id": topic_id,
        "user_id": int(user_id),
        "chat_type": "headlines",
        "title": "Идеи и темы",
    }).execute()
    sub_chat_data = sub_chat.data[0]
    sub_chat_id = sub_chat_data["id"]

    # 3. Save user message
    sb.table("project_messages").insert({
        "project_id": topic_id,
        "sub_chat_id": sub_chat_id,
        "user_id": int(user_id),
        "role": "user",
        "content": message,
        "message_type": "text"
    }).execute()

    # 4. Call AI
    system_prompt, prompt_model = get_system_prompt_v3("headlines")
    user_context = get_user_context(int(user_id), query=message)
    if user_context:
        system_prompt += f"\n\nКонтекст о пользователе:\n{user_context}"
        system_prompt += "\n\nИспользуй контекст о пользователе, чтобы предлагать идеи, максимально релевантные его нише, продукту и целевой аудитории. Персонализируй заголовки под его бизнес и стиль."
    else:
        system_prompt += """

ВАЖНО: Ты пока ничего не знаешь об этом пользователе — его нише, продукте, целевой аудитории.
Сначала ответь на запрос пользователя (предложи заголовки, если он дал тему).
Потом ОБЯЗАТЕЛЬНО добавь в конце сообщения дружелюбную рекомендацию:
— Расскажи, что ты можешь запоминать информацию о пользователе (нишу, продукт, аудиторию, tone of voice)
— Предложи наговорить голосовым сообщением (через кнопку диктовки 🎙️) всё о себе: чем занимается, кто его клиенты, какой продукт, какой стиль общения
— Скажи, что чем больше ты узнаешь, тем точнее и качественнее будут идеи и тексты каруселей
— Будь кратким, 2-3 предложения максимум"""

    ai_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]

    try:
        ai_text, _usage = _call_openrouter(ai_messages, body.get("model") or prompt_model)
        _log_ai_call(user_id, "topics", "headlines", body.get("model") or prompt_model,
                     system_prompt, user_context, ai_messages, message, ai_text, _usage,
                     topic_id=topic_id, sub_chat_id=sub_chat_id)
    except Exception as e:
        return {"topic": topic_data, "sub_chat": sub_chat_data, "ai_response": {"content": str(e), "message_type": "text", "message_data": None}}

    # Parse headlines
    headlines = _parse_headlines(ai_text)
    message_type = "headlines" if headlines else "text"
    message_data = {"headlines": headlines} if headlines else None

    # 5. Save AI response
    sb.table("project_messages").insert({
        "project_id": topic_id,
        "sub_chat_id": sub_chat_id,
        "user_id": int(user_id),
        "role": "assistant",
        "content": ai_text,
        "message_type": message_type,
        "message_data": message_data
    }).execute()

    # Update topic
    sb.table("projects").update({"active_subchat_id": sub_chat_id}).eq("id", topic_id).execute()

    # Extract memories from user message
    _run_memory_extraction(int(user_id), [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ai_text}
    ])

    return {
        "topic": topic_data,
        "sub_chat": sub_chat_data,
        "ai_response": {"content": ai_text, "message_type": message_type, "message_data": message_data}
    }


@app.get("/api/topics/{topic_id}")
async def get_topic(topic_id: str):
    """Get topic with all its sub-chats."""
    sb = require_supabase()
    topic = sb.table("projects").select("*").eq("id", topic_id).single().execute()
    if not topic.data:
        raise HTTPException(status_code=404, detail="Topic not found")
    sub_chats = sb.table("sub_chats").select("*").eq("topic_id", topic_id).order("created_at").execute()
    return {"topic": topic.data, "sub_chats": sub_chats.data or []}


@app.patch("/api/topics/{topic_id}")
async def rename_topic(topic_id: str, request: Request):
    """Rename a topic."""
    sb = require_supabase()
    body = await request.json()
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")
    result = sb.table("projects").update({"title": title}).eq("id", topic_id).execute()
    return result.data[0] if result.data else {}


@app.delete("/api/topics/{topic_id}")
async def delete_topic(topic_id: str):
    """Delete a topic (cascades sub-chats and messages)."""
    sb = require_supabase()
    sb.table("projects").delete().eq("id", topic_id).execute()
    return {"ok": True}


# === v3: Sub-chats API ===

@app.get("/api/topics/{topic_id}/sub-chats")
async def list_sub_chats(topic_id: str):
    """List all sub-chats in a topic."""
    sb = require_supabase()
    result = sb.table("sub_chats").select("*").eq("topic_id", topic_id).order("created_at").execute()
    # Add last message preview for each sub-chat
    sub_chats = []
    for sc in (result.data or []):
        last_msg = sb.table("project_messages").select("content,created_at").eq("sub_chat_id", sc["id"]).order("created_at", desc=True).limit(1).execute()
        sc["last_message"] = last_msg.data[0]["content"][:100] if last_msg.data else ""
        sc["last_message_at"] = last_msg.data[0]["created_at"] if last_msg.data else sc["created_at"]
        sub_chats.append(sc)
    return sub_chats


@app.post("/api/topics/{topic_id}/sub-chats")
async def create_sub_chat(topic_id: str, request: Request):
    """Create a new sub-chat. For text type, auto-generates slide text."""
    sb = require_supabase()
    body = await request.json()
    user_id = body.get("user_id")
    chat_type = body.get("chat_type")
    selected_headline = body.get("selected_headline")
    parent_subchat_id = body.get("parent_subchat_id")
    slides_count = body.get("slides_count", 7)

    if not user_id or not chat_type:
        raise HTTPException(status_code=400, detail="user_id and chat_type required")

    title = selected_headline or f"{'Текст' if chat_type == 'text' else 'Карусель'}"

    sub_chat = sb.table("sub_chats").insert({
        "topic_id": topic_id,
        "user_id": int(user_id),
        "chat_type": chat_type,
        "title": title,
        "selected_headline": selected_headline,
        "parent_subchat_id": parent_subchat_id,
    }).execute()
    sub_chat_data = sub_chat.data[0]
    sub_chat_id = sub_chat_data["id"]

    ai_response = None

    # For text sub-chat: auto-generate slide text
    if chat_type == 'text' and selected_headline and OPENROUTER_API_KEY:
        source = body.get("source", "")
        is_rewrite = source == "competitor_rewrite"
        prompt_key = "competitor_rewrite" if is_rewrite else "text"

        system_prompt, prompt_model = get_system_prompt_v3(prompt_key, variables={
            "slides_count": str(slides_count),
            "selected_headline": selected_headline,
        })
        user_context = get_user_context(int(user_id))
        if user_context:
            system_prompt += f"\n\nКонтекст о пользователе:\n{user_context}"

        user_message = (
            f"Перепиши этот пост конкурента в формат карусели из {slides_count} слайдов в моём стиле:\n\n{selected_headline}"
            if is_rewrite else
            f"Напиши текст карусели на тему: {selected_headline}"
        )

        try:
            ai_text, _usage = _call_openrouter([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ], body.get("model") or prompt_model)
            _log_ai_call(user_id, f"sub-chats/{'rewrite' if is_rewrite else 'text'}", prompt_key,
                         body.get("model") or prompt_model,
                         system_prompt, user_context, None, selected_headline[:500], ai_text, _usage,
                         topic_id=topic_id, sub_chat_id=sub_chat_id)

            sb.table("project_messages").insert({
                "project_id": topic_id,
                "sub_chat_id": sub_chat_id,
                "user_id": int(user_id),
                "role": "assistant",
                "content": ai_text,
                "message_type": "slides",
            }).execute()

            ai_response = {"content": ai_text, "message_type": "slides", "message_data": None}
        except Exception as e:
            ai_response = {"content": str(e), "message_type": "text", "message_data": None}

    # For carousel sub-chat: pull slides text from source message or parent sub-chat
    if chat_type == 'carousel':
        source_message_id = body.get("source_message_id")
        slides_text = None
        if source_message_id:
            # Get text from specific message
            msg = sb.table("project_messages").select("content").eq("id", source_message_id).limit(1).execute()
            if msg.data:
                slides_text = msg.data[0]["content"]
        elif parent_subchat_id:
            # Fallback: get last slides message from parent
            parent_msgs = sb.table("project_messages").select("content").eq("sub_chat_id", parent_subchat_id).eq("message_type", "slides").order("created_at", desc=True).limit(1).execute()
            if parent_msgs.data:
                slides_text = parent_msgs.data[0]["content"]
        if slides_text:
            sb.table("sub_chats").update({"slides_data": {"text": slides_text}}).eq("id", sub_chat_id).execute()
            sub_chat_data["slides_data"] = {"text": slides_text}

    # Update topic's active sub-chat
    sb.table("projects").update({"active_subchat_id": sub_chat_id}).eq("id", topic_id).execute()

    return {"sub_chat": sub_chat_data, "ai_response": ai_response}


@app.get("/api/sub-chats/{sub_chat_id}")
async def get_sub_chat(sub_chat_id: str):
    """Get a sub-chat with its messages."""
    sb = require_supabase()
    sub_chat = sb.table("sub_chats").select("*").eq("id", sub_chat_id).single().execute()
    if not sub_chat.data:
        raise HTTPException(status_code=404, detail="Sub-chat not found")
    messages = sb.table("project_messages").select("*").eq("sub_chat_id", sub_chat_id).order("created_at").execute()
    return {"sub_chat": sub_chat.data, "messages": messages.data or []}


@app.patch("/api/sub-chats/{sub_chat_id}")
async def rename_sub_chat(sub_chat_id: str, request: Request):
    """Rename a sub-chat."""
    sb = require_supabase()
    body = await request.json()
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")
    result = sb.table("sub_chats").update({"title": title}).eq("id", sub_chat_id).execute()
    return result.data[0] if result.data else {}


@app.delete("/api/sub-chats/{sub_chat_id}")
async def delete_sub_chat(sub_chat_id: str):
    """Delete a sub-chat (cascades messages)."""
    sb = require_supabase()
    sb.table("sub_chats").delete().eq("id", sub_chat_id).execute()
    return {"ok": True}


# === v3: Refactored AI Chat ===

@app.post("/api/ai/chat")
async def ai_chat(request: Request):
    """AI chat — supports both v2 (project_id) and v3 (sub_chat_id) modes."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    sb = require_supabase()
    body = await request.json()
    project_id = body.get("project_id")
    sub_chat_id = body.get("sub_chat_id")
    user_id = body.get("user_id")
    user_message = body.get("message", "")
    slides_count = body.get("slides_count", 7)
    custom_prompts = body.get("custom_prompts", {})

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    if not project_id and not sub_chat_id:
        raise HTTPException(status_code=400, detail="project_id or sub_chat_id required")

    # --- Usage limit check ---
    _auth_id_for_limit = None
    _anon_token = _get_anon_token(request)
    try:
        tg_id = int(user_id)
        _acc = _get_auth_account_by_telegram(tg_id)
        if _acc:
            _auth_id_for_limit = _acc["id"]
    except (ValueError, TypeError):
        pass
    allowed, used, limit = await check_usage_limit(_auth_id_for_limit, _anon_token, "ai_chat")
    if not allowed:
        raise HTTPException(status_code=429, detail={"error": "limit_reached", "used": used, "limit": limit, "action": "ai_chat"})
    await increment_usage(_auth_id_for_limit, _anon_token, "ai_chat")

    # === v3 mode: sub_chat_id ===
    if sub_chat_id:
        sub_chat = sb.table("sub_chats").select("*").eq("id", sub_chat_id).single().execute()
        if not sub_chat.data:
            raise HTTPException(status_code=404, detail="Sub-chat not found")
        sc = sub_chat.data
        topic_id = sc["topic_id"]
        chat_type = sc["chat_type"]

        # Determine prompt key (editing merged into text, carousel removed)
        prompt_key = chat_type  # headlines or text

        # Build system prompt from DB
        variables = {
            "slides_count": str(slides_count),
            "selected_headline": sc.get("selected_headline") or "",
        }
        system_prompt, prompt_model = get_system_prompt_v3(prompt_key, variables=variables)
        user_context = get_user_context(int(user_id), query=user_message)
        if user_context:
            system_prompt += f"\n\nКонтекст о пользователе:\n{user_context}"

        # Get message history from ALL sub-chats in this topic (shared short-term memory)
        messages_result = sb.table("project_messages").select("role,content").eq("project_id", topic_id).order("created_at").execute()
        history = messages_result.data or []

        ai_messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-20:]:
            ai_messages.append({"role": msg["role"], "content": msg["content"]})
        if user_message:
            ai_messages.append({"role": "user", "content": user_message})

        # Save user message
        if user_message:
            sb.table("project_messages").insert({
                "project_id": topic_id,
                "sub_chat_id": sub_chat_id,
                "user_id": int(user_id),
                "role": "user",
                "content": user_message,
                "message_type": "text"
            }).execute()

        # Call AI
        try:
            ai_text, _usage = _call_openrouter(ai_messages, body.get("model") or prompt_model)
            _log_ai_call(user_id, "ai/chat", prompt_key, body.get("model") or prompt_model,
                         system_prompt, user_context, ai_messages, user_message, ai_text, _usage,
                         topic_id=topic_id, sub_chat_id=sub_chat_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Determine message type
        message_type = "text"
        message_data = None

        if chat_type == "headlines":
            headlines = _parse_headlines(ai_text)
            if headlines:
                message_type = "headlines"
                message_data = {"headlines": headlines}
        elif chat_type == "text":
            message_type = "slides"

        # Save AI response
        sb.table("project_messages").insert({
            "project_id": topic_id,
            "sub_chat_id": sub_chat_id,
            "user_id": int(user_id),
            "role": "assistant",
            "content": ai_text,
            "message_type": message_type,
            "message_data": message_data
        }).execute()

        # Async memory extraction (non-blocking)
        try:
            all_msgs = [{"role": m["role"], "content": m["content"]} for m in history]
            if user_message:
                all_msgs.append({"role": "user", "content": user_message})
            all_msgs.append({"role": "assistant", "content": ai_text})
            _run_memory_extraction(int(user_id), all_msgs)
        except Exception:
            pass

        return {
            "content": ai_text,
            "message_type": message_type,
            "message_data": message_data,
            "sub_chat_id": sub_chat_id,
            "chat_type": chat_type
        }

    # === v2 legacy mode: project_id ===
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id required")

    project = sb.table("projects").select("*").eq("id", project_id).single().execute()
    if not project.data:
        raise HTTPException(status_code=404, detail="Project not found")

    status = project.data.get("status", "draft")

    # Try to find default sub-chat for backward compat
    default_sc = sb.table("sub_chats").select("id").eq("topic_id", project_id).order("created_at").limit(1).execute()
    legacy_sub_chat_id = default_sc.data[0]["id"] if default_sc.data else None

    # Get message history
    messages_result = sb.table("project_messages").select("role,content").eq("project_id", project_id).order("created_at").execute()
    history = messages_result.data or []

    system_prompt = get_system_prompt(status, slides_count, custom_prompts)
    user_context = get_user_context(int(user_id))
    if user_context:
        system_prompt += f"\n\nКонтекст о пользователе:\n{user_context}"

    ai_messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-20:]:
        ai_messages.append({"role": msg["role"], "content": msg["content"]})
    if user_message:
        ai_messages.append({"role": "user", "content": user_message})

    # Save user message
    if user_message:
        msg_insert = {
            "project_id": project_id,
            "user_id": int(user_id),
            "role": "user",
            "content": user_message,
            "message_type": "text"
        }
        if legacy_sub_chat_id:
            msg_insert["sub_chat_id"] = legacy_sub_chat_id
        sb.table("project_messages").insert(msg_insert).execute()

    # Call OpenRouter
    try:
        _v2_model = body.get("model", "openai/gpt-4o")
        ai_text, _usage = _call_openrouter(ai_messages, _v2_model)
        _log_ai_call(user_id, "ai/chat-v2", "legacy", _v2_model,
                     system_prompt, user_context, ai_messages, user_message, ai_text, _usage,
                     topic_id=project_id, sub_chat_id=legacy_sub_chat_id)

        message_type = "text"
        message_data = None
        new_status = status

        if status == 'draft':
            headlines = []
            for line in ai_text.split('\n'):
                line = line.strip()
                match = re.match(r'^\d+[\.\)]\s*(.+)', line)
                if match:
                    headlines.append(match.group(1).strip())
            if headlines:
                message_type = "headlines"
                message_data = {"headlines": headlines}
                new_status = "headlines"

        elif status == 'headlines':
            message_type = "slides"
            new_status = "text_ready"

        # Save AI response
        msg_insert = {
            "project_id": project_id,
            "user_id": int(user_id),
            "role": "assistant",
            "content": ai_text,
            "message_type": message_type,
            "message_data": message_data
        }
        if legacy_sub_chat_id:
            msg_insert["sub_chat_id"] = legacy_sub_chat_id
        sb.table("project_messages").insert(msg_insert).execute()

        update_data = {"status": new_status}
        if status == 'headlines' and project.data.get("selected_headline"):
            update_data["slides_data"] = ai_text
        sb.table("projects").update(update_data).eq("id", project_id).execute()

        return {
            "content": ai_text,
            "message_type": message_type,
            "message_data": message_data,
            "status": new_status
        }

    except requests.Timeout:
        raise HTTPException(status_code=504, detail="AI response timeout")
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/select-headline")
async def ai_select_headline(request: Request):
    """User selected a headline — update project and trigger text generation."""
    sb = require_supabase()
    body = await request.json()
    project_id = body.get("project_id")
    headline = body.get("headline")

    if not project_id or not headline:
        raise HTTPException(status_code=400, detail="project_id and headline required")

    sb.table("projects").update({
        "selected_headline": headline,
        "status": "headlines"
    }).eq("id", project_id).execute()

    return {"ok": True}


@app.post("/api/ai/translate")
async def ai_translate(request: Request):
    """Translate a project's slides to another language."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    sb = require_supabase()
    body = await request.json()
    project_id = body.get("project_id")
    target_language = body.get("target_language", "en")
    user_id = body.get("user_id")

    if not project_id:
        raise HTTPException(status_code=400, detail="project_id required")

    # Get source project
    project = sb.table("projects").select("*").eq("id", project_id).single().execute()
    if not project.data:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get slides text from messages
    messages = sb.table("project_messages").select("content,message_type").eq("project_id", project_id).eq("message_type", "slides").order("created_at", desc=True).limit(1).execute()
    slides_text = ""
    if messages.data:
        slides_text = messages.data[0]["content"]
    elif project.data.get("slides_data"):
        slides_text = json.dumps(project.data["slides_data"]) if isinstance(project.data["slides_data"], (list, dict)) else str(project.data["slides_data"])

    if not slides_text:
        raise HTTPException(status_code=400, detail="No slides text to translate")

    lang_names = {"en": "English", "ru": "Russian", "ar": "Arabic", "tr": "Turkish", "es": "Spanish", "fr": "French", "de": "German"}
    lang_name = lang_names.get(target_language, target_language)

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-4o",
                "messages": [
                    {"role": "system", "content": f"Translate the following carousel slides to {lang_name}. Keep the exact same format (Slide N, Title, Description). Only translate, don't change meaning or structure."},
                    {"role": "user", "content": slides_text}
                ],
                "temperature": 0.3,
                "max_tokens": 4096
            },
            timeout=120
        )

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"Translation error: {resp.text}")

        translated_text = resp.json()["choices"][0]["message"]["content"]

        # Create new project for translated version
        new_project = {
            "user_id": int(user_id) if user_id else project.data["user_id"],
            "title": f"{project.data['title']} ({target_language.upper()})",
            "status": "text_ready",
            "idea_text": project.data.get("idea_text"),
            "selected_headline": project.data.get("selected_headline"),
            "template_id": project.data.get("template_id"),
            "instagram_username": project.data.get("instagram_username"),
            "language": target_language,
            "translated_from": project_id,
        }
        result = sb.table("projects").insert(new_project).execute()
        new_project_id = result.data[0]["id"] if result.data else None

        if new_project_id:
            sb.table("project_messages").insert({
                "project_id": new_project_id,
                "user_id": int(user_id) if user_id else int(project.data["user_id"]),
                "role": "assistant",
                "content": translated_text,
                "message_type": "slides"
            }).execute()

        return {
            "translated_text": translated_text,
            "new_project_id": new_project_id
        }

    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Translation timeout")
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/save-memory")
async def ai_save_memory(request: Request):
    """Save a user preference/memory to user_memory table with embedding."""
    sb = require_supabase()
    body = await request.json()
    user_id = body.get("user_id")
    content = body.get("content")

    if not user_id or not content:
        raise HTTPException(status_code=400, detail="user_id and content required")

    # Create embedding via OpenAI
    embedding = None
    if OPENAI_API_KEY:
        try:
            emb_resp = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={"model": "text-embedding-3-small", "input": content},
                timeout=30
            )
            if emb_resp.status_code == 200:
                embedding = emb_resp.json()["data"][0]["embedding"]
        except Exception:
            pass

    memory_data = {
        "user_id": str(user_id),
        "content": content,
        "metadata": {"source": "web_chat"},
    }
    if embedding:
        memory_data["embedding"] = embedding

    sb.table("user_memory").insert(memory_data).execute()
    return {"ok": True}


# === Projects CRUD ===

def require_supabase():
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    return supabase


# === v3: Admin Prompts CRUD ===

@app.get("/api/admin/prompts")
async def list_prompts(request: Request):
    """List all system prompts. Admin only. Seeds defaults if empty."""
    _check_admin_token(request)
    sb = require_supabase()
    result = sb.table("system_prompts").select("*").order("prompt_key").execute()
    # Seed missing default prompts
    existing_keys = {row["prompt_key"] for row in (result.data or [])}
    defaults = [
            {"prompt_key": "headlines", "title": "Генератор идей и заголовков", "description": "Предлагает идеи для постов и генерирует заголовки карусели",
             "model": "google/gemini-2.5-flash",
             "content": """# ЛИЧНОСТЬ

Ты — AI-копирайтер Craft AI. Общайся дружелюбно, на «ты», кратко и по делу.
Не используй канцелярит. Пиши как умный друг-маркетолог, а не как робот.
Ты помнишь всё, что обсуждал с пользователем ранее — используй это.

# ЗАДАЧА

У тебя две функции в зависимости от запроса:

## ФУНКЦИЯ 1: ИДЕИ ДЛЯ ПОСТОВ

Если пользователь НЕ дал конкретную тему, а просит идеи («дай идеи», «что написать», «не знаю о чём пост», «предложи тему»):

1. Проанализируй всё что знаешь о пользователе (ниша, продукт, аудитория, стиль)
2. Учти что уже обсуждали ранее (не повторяй прошлые темы)
3. Предложи 5-7 конкретных идей для постов

Формат:
Вот что можно сделать:

1. **[Короткая тема]** — [почему это зайдёт, 1 предложение]
2. **[Короткая тема]** — [почему это зайдёт]
...

Выбирай какая нравится, или скажи свою тему — сделаю заголовки 🔥

## ФУНКЦИЯ 2: ГЕНЕРАЦИЯ ЗАГОЛОВКОВ

Если пользователь ДАЛ конкретную тему или выбрал идею:

Сгенерируй 5-7 заголовков для карусели.

Формат — СТРОГО пронумерованный список:
1. [заголовок]
2. [заголовок]
...

### Требования к заголовкам:
- Макс 60 символов
- Разговорный язык, на «ты»
- Конкретика, а не абстракции
- Каждый заголовок — с триггером: боль, контраст, цифра, провокация, вопрос, секрет, срочность
- Если знаешь ЦА/нишу — адаптируй под них
- Все заголовки разные по структуре (не повторяй шаблон)

### Если мало контекста:
Если ты ещё мало знаешь о пользователе — задай 1-2 коротких вопроса перед генерацией:
«Кстати, кто твоя аудитория? Так заголовки будут точнее»

## ЗАПРЕЩЕНО

- «Давайте разберём», «В современном мире», «Это не просто...», «Секрет успеха в том...»
- Канцелярит, вода, штампы
- Заголовки длиннее 60 символов
- Одинаковые структуры подряд
- Придумывать факты о пользователе""",
             "variables": [{"name": "user_context", "description": "Профиль и стиль пользователя"}]},
            {"prompt_key": "text", "title": "Автор и редактор текста слайдов", "description": "Пишет и дорабатывает текст слайдов карусели",
             "model": "google/gemini-2.5-flash",
             "content": """# ЛИЧНОСТЬ

Ты — AI-копирайтер Craft AI. Общайся дружелюбно, на «ты», кратко и по делу.
Не используй канцелярит. Пиши как умный друг-маркетолог, а не как робот.
Ты помнишь всё, что обсуждал с пользователем ранее — используй это.

# ЗАДАЧА

У тебя две функции:

## ФУНКЦИЯ 1: НАПИСАТЬ ТЕКСТ КАРУСЕЛИ (если текста ещё нет)

Данные:
- Заголовок: «{selected_headline}»
- Слайдов: {slides_count}

Структура:

**Слайд 1 — ОБЛОЖКА:** Заголовок = ТОЧНО «{selected_headline}» (без изменений!). Описание = подзаголовок-интрига (до 80 символов).

**Слайд 2 — ХУК:** Зачем это читать? Проблема, которую решает пост (до 150 символов).

**Слайды 3–N — КОНТЕНТ:** Заголовок пункта + раскрытие (до 200 символов).

**Предпоследний — ВЫВОД:** Суть карусели в 1-2 предложениях.

**Последний — CTA:** Призыв к действию.

## ФУНКЦИЯ 2: ДОРАБОТАТЬ ТЕКСТ (если текст уже есть в истории)

Если пользователь просит изменить текст:
- Конкретный слайд → перепиши только его
- Стиль → перепиши все в новом стиле
- Добавить/удалить слайд → сделай и покажи
- «Всё ок» → подтверди что текст финальный

НЕ переписывай всё без запроса. Меняй только то, что просят.

## ФОРМАТ ОТВЕТА (для обеих функций)

Слайд 1
Заголовок: ...
Описание: ...

Слайд 2
Заголовок: ...
Описание: ...

(и так далее)

## ПРАВИЛА

- Пиши на «ты», короткие предложения (max 15-20 слов)
- Каждое слово работает — без воды
- Как человек, не как робот
- Если есть контекст о ЦА/продукте/стиле — используй
- Если контекста нет — пиши универсально

## СТОП-СЛОВА

«Это не просто», «Представьте», «В современном мире», «Как известно», «Не секрет что», «Уникальный», «Инновационный», «Совершил революцию»

## КРИТИЧЕСКИЕ ПРАВИЛА

1. «{selected_headline}» → заголовок слайда 1 БЕЗ ИЗМЕНЕНИЙ
2. {slides_count} → точное количество слайдов
3. НЕ спрашивай заголовок и количество — они уже даны
4. Пустой контекст о пользователе — НЕ ошибка""",
             "variables": [{"name": "slides_count", "description": "Количество слайдов"}, {"name": "selected_headline", "description": "Выбранный заголовок"}, {"name": "user_context", "description": "Профиль пользователя"}]},
            {"prompt_key": "memory_extract", "title": "Извлечение памяти", "description": "Анализирует диалог и извлекает важные факты о пользователе",
             "model": "openai/gpt-4o-mini",
             "content": """Проанализируй последние сообщения диалога. Извлеки важную информацию о пользователе.

## Что сохранять:
- Целевая аудитория ("моя аудитория — мамы в декрете")
- Продукт/услуга ("я продаю курс по инвестициям")
- Ниша ("я работаю в сфере красоты")
- Стиль общения ("пиши дерзко", "без мата")
- Любые другие важные детали о бизнесе, клиентах, предпочтениях

## Что НЕ сохранять:
- Саму тему заголовка
- "да", "нет", "ок"
- Номера выбора
- Просьбы изменить текст слайдов
- Технические команды

## Формат ответа:
Если есть информация — верни JSON:
{"memories": ["факт1", "факт2"]}
Если ничего важного — верни:
{"memories": []}""",
             "variables": []},
            {"prompt_key": "profile_summarize", "title": "Суммаризация профиля", "description": "Создаёт краткий профиль пользователя из всех воспоминаний",
             "model": "openai/gpt-4o-mini",
             "content": "На основе следующих фактов о пользователе создай краткий профиль (3-5 предложений): его ниша, стиль контента, целевая аудитория, предпочтения.\n\nФакты:\n{memories}",
             "variables": [{"name": "memories", "description": "Все сохранённые факты о пользователе"}]},
            {"prompt_key": "format_text", "title": "Парсер текста карусели", "description": "Разбивает вставленный текст на слайды (TITLE + DESCRIPTION) без изменения содержания",
             "model": "openai/gpt-4o-mini",
             "content": """# РОЛЬ: ПАРСЕР ТЕКСТА КАРУСЕЛИ

Ты - парсер текста. Твоя задача - ИЗВЛЕЧЬ структуру из текста пользователя БЕЗ ИЗМЕНЕНИЯ самого текста.

## АБСОЛЮТНЫЕ ПРАВИЛА (КРИТИЧЕСКИ ВАЖНО!)

1. **НИ В КОЕМ СЛУЧАЕ НЕ МЕНЯЙ ТЕКСТ ПОЛЬЗОВАТЕЛЯ**
   - Не переформулируй
   - Не исправляй ошибки
   - Не улучшай стиль
   - Не добавляй свои слова
   - Копируй текст КАК ЕСТЬ

2. **НИ ОДИН СИМВОЛ НЕ ДОЛЖЕН ПРОПАСТЬ**
   - Если не понятно где заголовок - отправь ВСЁ в DESCRIPTION
   - Если не можешь разделить на слайды - отправь весь текст в 1 слайд
   - Лучше всё в DESCRIPTION, чем потерять текст

3. **ЕДИНСТВЕННОЕ РАЗРЕШЁННОЕ ИЗМЕНЕНИЕ: разбивка на абзацы**
   - Только в DESCRIPTION
   - Только если текст > 150 символов
   - Используй \\n\\n между абзацами

## АЛГОРИТМ ПАРСИНГА

### ШАГ 0: Очистка метаданных (ОБЯЗАТЕЛЬНО!)
Текст может содержать служебные строки из Telegram-бота. УДАЛИ их перед парсингом, НЕ включай в слайды:
- Строки с: 📄 Слайдов, 👤 Instagram, 🎨 Шаблон, 📷 Фото
- Подтверждающие вопросы: "Всё верно? Создаём карусель?", "да/нет"
- Инструкции: "Изменить никнейм или шаблон?", "Напиши: никнейм: @новый"
- Любые строки, которые явно НЕ являются контентом слайдов

### ШАГ 1: Определи слайды
Ищи маркеры:
- "Слайд 1:", "Слайд 2:" (с двоеточием или без)
- "1.", "2.", "3."
- Эмодзи + текст
- Пустые строки между блоками

### ШАГ 2: Для каждого слайда определи TITLE и DESCRIPTION

**ЕСЛИ есть явные метки "Заголовок:" и/или "Описание:":**
- "Заголовок:" → текст после метки идёт в TITLE (может быть на той же строке или на следующей)
- "Описание:" → текст после метки идёт в DESCRIPTION
- Если "Заголовок:" пустой (нет текста после метки), но есть текст до "Описание:" → этот текст идёт в TITLE
- Если нет "Описание:", весь текст после заголовка → DESCRIPTION
- Метки "Заголовок:" и "Описание:" сами НЕ включаются в результат

**ЕСЛИ нет явных меток:**
TITLE - это первая строка/фраза если:
- Она короткая (< 100 символов)
- После неё есть текст побольше
- Она выделена (жирным, заглавными и т.д.)

**ИСКЛЮЧЕНИЕ ДЛЯ ПЕРВОГО СЛАЙДА (slide_number: 1):**
- Если непонятно что TITLE - возьми первую фразу/строку и отправь в TITLE
- Первый слайд ВСЕГДА должен иметь TITLE (не пустой)
- Это обложка карусели - заголовок обязателен

**ДЛЯ ОСТАЛЬНЫХ СЛАЙДОВ (2, 3, 4...):**
- Если не уверен - оставь TITLE пустым "", всё в DESCRIPTION

### ШАГ 3: Остальное -> DESCRIPTION
Всё что не TITLE -> в DESCRIPTION (БЕЗ ИЗМЕНЕНИЙ!)

### ШАГ 4: Разбей DESCRIPTION на абзацы (опционально)
**ТОЛЬКО если DESCRIPTION > 150 символов:**
- Найди смысловые блоки (новая мысль, новый аспект)
- Раздели через \\n\\n
- НЕ разбивай каждое предложение
- 1 абзац = минимум 50 символов

**ЕСЛИ < 150 символов** -> оставь как есть

## ПРИМЕРЫ

### Пример 1: Текст из Telegram-бота с метками

Входящий:
📄 Слайдов: 3
👤 Instagram: @username

Слайд 1:
Заголовок: Почему важно пить воду правильно

Слайд 2:
Заголовок:
Это было сказано давно

Описание:
Когда человек пьёт стоя, вода резко попадает в желудок и создаёт нагрузку.

Слайд 3:
Заголовок: Попробуй сегодня

Описание:
Сядь. Сделай несколько глотков. Не спеши.
Иногда самые простые действия — самые правильные.

Всё верно? Создаём карусель? (да/нет)
📷 Фото: ❌ не выбрано

Правильный результат:
[
  {"TITLE": "Почему важно пить воду правильно", "DESCRIPTION": ""},
  {"TITLE": "Это было сказано давно", "DESCRIPTION": "Когда человек пьёт стоя, вода резко попадает в желудок и создаёт нагрузку."},
  {"TITLE": "Попробуй сегодня", "DESCRIPTION": "Сядь. Сделай несколько глотков. Не спеши.\\n\\nИногда самые простые действия — самые правильные."}
]

### Пример 2: Простой текст без меток
Входящий: "Инфляция съедает сбережения быстрее чем вы успеваете их заработать. В 2024 году официальная инфляция 7%, но реальная на продукты - 15%. Вклады в банках дают максимум 5-6% годовых."
Правильно: [{"TITLE": "Инфляция съедает сбережения быстрее чем вы успеваете их заработать", "DESCRIPTION": "В 2024 году официальная инфляция 7%, но реальная на продукты - 15%.\\n\\nВклады в банках дают максимум 5-6% годовых."}]

## ВЫХОДНОЙ ФОРМАТ

Верни ТОЛЬКО JSON массив (не объект, без markdown):

[
  {
    "TITLE": "Заголовок 1",
    "DESCRIPTION": "Текст как есть"
  },
  {
    "TITLE": "Заголовок 2",
    "DESCRIPTION": "Длинный текст.\\n\\nВторой абзац если нужно."
  }
]

ПРАВИЛА:
- Массив напрямую
- TITLE может быть пустым ""
- DESCRIPTION обязателен (может быть пустым "" для обложки)
- Текст пользователя НЕ ИЗМЕНЁН
- \\n\\n только для абзацев
- Метки "Заголовок:", "Описание:", "Слайд N:" НЕ включать в результат

## ЗАПРЕЩЕНО
- Менять слова пользователя
- Исправлять грамматику
- Добавлять свой текст
- Удалять текст пользователя (кроме метаданных!)
- Возвращать ```json``` блоки
- Добавлять комментарии
- Включать метаданные бота (📄, 👤, 🎨, 📷) в слайды

## В СОМНИТЕЛЬНЫХ СЛУЧАЯХ
Не можешь определить заголовок? -> TITLE: "", весь текст в DESCRIPTION
Не можешь разделить на слайды? -> Сделай 1 слайд со всем текстом
Текст странный/непонятный? -> Всё равно сохрани его КАК ЕСТЬ

ГЛАВНОЕ: Сохрани ВЕСЬ текст пользователя БЕЗ ИЗМЕНЕНИЙ!""",
             "variables": []},
            {"prompt_key": "ocr_slides", "title": "OCR: извлечение текста со слайдов",
             "description": "Извлекает контентный текст с изображений карусели (Gemini Vision). Игнорирует UI Instagram.",
             "model": "google/gemini-2.0-flash-001",
             "content": OCR_SLIDES_PROMPT,
             "variables": []},
            {"prompt_key": "competitor_rewrite", "title": "Рерайт конкурента",
             "description": "Переписывает пост конкурента в уникальную карусель в стиле пользователя",
             "model": "google/gemini-2.5-flash",
             "content": SYSTEM_PROMPT_COMPETITOR_REWRITE,
             "variables": [{"name": "slides_count", "description": "Количество слайдов"}]},
            {"prompt_key": "transcript_clean", "title": "Очистка транскрипции",
             "description": "Очищает сырую транскрипцию видео: пунктуация, абзацы, убирает слова-паразиты",
             "model": "openai/gpt-4o-mini",
             "content": TRANSCRIPT_CLEAN_PROMPT,
             "variables": []},
            {"prompt_key": "video_structure", "title": "Анализ структуры видео",
             "description": "Анализирует хук, основной посыл, тезисы, CTA и стиль подачи видео",
             "model": "openai/gpt-4o-mini",
             "content": VIDEO_STRUCTURE_PROMPT,
             "variables": []},
        ]
    # Upsert: insert missing, update existing defaults that haven't been manually edited
    defaults_by_key = {d["prompt_key"]: d for d in defaults}
    changed = False
    for d in defaults:
        if d["prompt_key"] not in existing_keys:
            # New prompt — insert
            try:
                sb.table("system_prompts").insert(d).execute()
                changed = True
            except Exception:
                pass
        else:
            # Existing prompt — update content/model from defaults if not manually edited
            existing = next((r for r in (result.data or []) if r["prompt_key"] == d["prompt_key"]), None)
            if existing and existing.get("updated_by") is None:
                try:
                    sb.table("system_prompts").update({
                        "content": d["content"],
                        "model": d["model"],
                        "title": d["title"],
                        "description": d.get("description", ""),
                    }).eq("prompt_key", d["prompt_key"]).execute()
                    changed = True
                except Exception:
                    pass
    # Delete prompts not in defaults (e.g. removed carousel, editing)
    default_keys = {d["prompt_key"] for d in defaults}
    for r in (result.data or []):
        if r["prompt_key"] not in default_keys:
            try:
                sb.table("system_prompts").delete().eq("prompt_key", r["prompt_key"]).execute()
                changed = True
            except Exception:
                pass
    if changed:
        result = sb.table("system_prompts").select("*").order("prompt_key").execute()
    return result.data or []


@app.put("/api/admin/prompts/{prompt_key}")
async def update_prompt(prompt_key: str, request: Request):
    """Update a system prompt. Admin only."""
    _check_admin_token(request)
    global _prompt_cache, _prompt_cache_time
    sb = require_supabase()
    body = await request.json()

    allowed_fields = {"title", "description", "content", "variables", "is_active", "model"}
    update_data = {k: v for k, v in body.items() if k in allowed_fields}
    if not update_data:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    update_data["updated_by"] = 0
    result = sb.table("system_prompts").update(update_data).eq("prompt_key", prompt_key).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Prompt not found")

    # Invalidate cache
    _prompt_cache.clear()
    _prompt_cache_time = 0

    return result.data[0]


@app.post("/api/admin/prompts")
async def create_prompt(request: Request):
    """Create a new system prompt. Admin only."""
    _check_admin_token(request)
    global _prompt_cache, _prompt_cache_time
    sb = require_supabase()
    body = await request.json()

    prompt_key = body.get("prompt_key")
    title = body.get("title")
    content = body.get("content")
    if not prompt_key or not title or not content:
        raise HTTPException(status_code=400, detail="prompt_key, title, and content required")

    result = sb.table("system_prompts").insert({
        "prompt_key": prompt_key,
        "title": title,
        "description": body.get("description", ""),
        "content": content,
        "variables": body.get("variables", []),
        "updated_by": int(user_id),
    }).execute()

    _prompt_cache.clear()
    _prompt_cache_time = 0
    return result.data[0] if result.data else {}


# === Admin: AI Logs & Dialogs ===

def _check_admin_token(request: Request):
    """Check admin access via X-Admin-Token header."""
    token = request.headers.get("X-Admin-Token", "")
    expected = os.getenv("ADMIN_PASSWORD", "")
    if not expected or token != expected:
        raise HTTPException(status_code=403, detail="Admin access required")


@app.post("/api/track/download")
async def track_download(request: Request):
    """Track carousel download. Increments carousels_used for the user."""
    try:
        body = await request.json()
        user_id = body.get("user_id")
        if user_id and supabase:
            resp = supabase.table("users").select("carousels_used").eq("user_id", str(user_id)).execute()
            current = (resp.data[0].get("carousels_used") or 0) if resp.data else 0
            supabase.table("users").update({"carousels_used": current + 1}).eq("user_id", str(user_id)).execute()
    except Exception as e:
        print(f"[track/download] Error: {e}")
    return {"ok": True}


@app.get("/api/admin/dashboard")
async def admin_dashboard(request: Request):
    """Dashboard stats. Admin only."""
    _check_admin_token(request)
    try:
        from datetime import datetime
        import urllib.request
        sb = require_supabase()
        # Users stats from Supabase
        all_users = sb.table("users").select("user_id,first_name,last_name,username,last_active,carousels_used").execute()
        users_data = all_users.data or []
        total_users = len(users_data)
        today_str = datetime.now().strftime("%Y-%m-%d")
        active_today_users = [
            {"user_id": u["user_id"], "first_name": u.get("first_name",""), "last_name": u.get("last_name",""), "username": u.get("username","")}
            for u in users_data if (u.get("last_active") or "")[:10] == today_str
        ]
        active_today = len(active_today_users)
        total_carousels = sum(u.get("carousels_used") or 0 for u in users_data)
        # AI logs stats from PostgreSQL (_pg_query returns list of dicts via RealDictCursor)
        tok_row = _pg_query("SELECT COALESCE(SUM(total_tokens),0) as val FROM ai_logs")
        total_tokens = tok_row[0]["val"] if tok_row else 0
        avg_row = _pg_query("""
            SELECT COALESCE(AVG(user_total), 0) as val FROM (
                SELECT SUM(total_tokens) as user_total FROM ai_logs
                WHERE user_id IS NOT NULL GROUP BY user_id
            ) sub
        """)
        avg_tokens = avg_row[0]["val"] if avg_row else 0
        # Cost calculation by model pricing (OpenRouter rates)
        cost_row = _pg_query("""
            SELECT COALESCE(SUM(
                CASE
                    WHEN model ILIKE '%%gemini%%flash%%' THEN input_tokens * 0.15 / 1000000.0 + output_tokens * 0.60 / 1000000.0
                    WHEN model ILIKE '%%gpt-4o-mini%%' THEN input_tokens * 0.15 / 1000000.0 + output_tokens * 0.60 / 1000000.0
                    WHEN model ILIKE '%%gpt-4o%%' THEN input_tokens * 2.50 / 1000000.0 + output_tokens * 10.0 / 1000000.0
                    WHEN model ILIKE '%%gemini%%pro%%' THEN input_tokens * 1.25 / 1000000.0 + output_tokens * 10.0 / 1000000.0
                    WHEN model ILIKE '%%claude%%' THEN input_tokens * 3.0 / 1000000.0 + output_tokens * 15.0 / 1000000.0
                    ELSE input_tokens * 0.50 / 1000000.0 + output_tokens * 1.50 / 1000000.0
                END
            ), 0) as total_cost_usd FROM ai_logs
        """)
        total_cost_usd = float(cost_row[0]["total_cost_usd"]) if cost_row else 0.0
        # USD/RUB rate from CBR
        try:
            cbr_resp = urllib.request.urlopen("https://www.cbr-xml-daily.ru/daily_json.js", timeout=3)
            usd_rub = json.loads(cbr_resp.read())["Valute"]["USD"]["Value"]
        except Exception:
            usd_rub = 88.0
        daily_rows = _pg_query("""
            SELECT created_at::date as day, COUNT(*) as cnt
            FROM ai_logs WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY day ORDER BY day
        """) or []
        return {
            "total_users": total_users,
            "active_today": active_today,
            "active_today_users": active_today_users,
            "total_carousels": total_carousels,
            "total_tokens": int(total_tokens),
            "total_cost_usd": round(total_cost_usd, 2),
            "usd_rub_rate": round(float(usd_rub), 2),
            "avg_tokens_per_user": round(float(avg_tokens)),
            "daily_usage": [{"date": str(r["day"]), "count": r["cnt"]} for r in daily_rows]
        }
    except Exception as e:
        print(f"[admin/dashboard] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/logs")
async def list_ai_logs(request: Request, limit: int = 100, offset: int = 0,
                       filter_user_id: int = None, filter_endpoint: str = None):
    """List AI call logs. Admin only."""
    _check_admin_token(request)
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")
    try:
        query = "SELECT * FROM ai_logs WHERE 1=1"
        params = []
        if filter_user_id:
            query += " AND user_id = %s"
            params.append(filter_user_id)
        if filter_endpoint:
            query += " AND endpoint = %s"
            params.append(filter_endpoint)
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        rows = _pg_query(query, params)
        return rows or []
    except HTTPException:
        raise
    except Exception as e:
        print(f"[admin/logs] Error: {e}")
        raise HTTPException(status_code=500, detail=f"error.load: {str(e)}")


@app.get("/api/admin/logs/{log_id}")
async def get_ai_log_detail(log_id: str, request: Request):
    """Get full AI log detail. Admin only."""
    _check_admin_token(request)
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")
    try:
        row = _pg_query("SELECT * FROM ai_logs WHERE id = %s", [log_id], fetchone=True)
        if not row:
            raise HTTPException(status_code=404, detail="Log not found")
        return row
    except HTTPException:
        raise
    except Exception as e:
        print(f"[admin/logs/{log_id}] Error: {e}")
        raise HTTPException(status_code=500, detail=f"error: {str(e)}")


@app.get("/api/admin/users")
async def list_admin_users(request: Request):
    """List all users with stats. Admin only."""
    _check_admin_token(request)
    try:
        sb = require_supabase()
        users = sb.table("users").select("user_id,first_name,last_name,username,photo_url,instagram_usernames,profile_summary,memory_count,last_active,created_at").order("last_active", desc=True).execute()
        users_data = users.data or []
        for u in users_data:
            try:
                cnt = sb.table("projects").select("id", count="exact").eq("user_id", u["user_id"]).execute()
                u["topic_count"] = cnt.count if cnt.count else 0
            except Exception:
                u["topic_count"] = 0
        return users_data
    except HTTPException:
        raise
    except Exception as e:
        print(f"[admin/users] Error: {e}")
        raise HTTPException(status_code=500, detail=f"error: {str(e)}")


@app.get("/api/admin/users/{target_user_id}/topics")
async def list_user_topics(target_user_id: int, request: Request):
    """List all topics for a specific user. Admin only."""
    _check_admin_token(request)
    sb = require_supabase()
    result = sb.table("projects").select("id,title,idea_text,status,created_at,updated_at").eq("user_id", target_user_id).order("updated_at", desc=True).execute()
    for topic in (result.data or []):
        msgs = sb.table("project_messages").select("id", count="exact").eq("project_id", topic["id"]).execute()
        topic["message_count"] = msgs.count if msgs.count else 0
    return result.data or []


@app.get("/api/admin/topics/{topic_id}/messages")
async def list_topic_messages(topic_id: str, request: Request):
    """Get all messages for a topic (all sub-chats). Admin only."""
    _check_admin_token(request)
    sb = require_supabase()
    result = sb.table("project_messages").select("*").eq("project_id", topic_id).order("created_at").execute()
    return result.data or []


# === Legacy: Projects API ===

@app.get("/api/projects")
async def list_projects(user_id: int):
    """List all projects for a user, sorted by updated_at DESC."""
    sb = require_supabase()
    result = sb.table("projects").select("*").eq("user_id", user_id).order("updated_at", desc=True).execute()
    return result.data


@app.post("/api/projects")
async def create_project(request: Request):
    """Create a new project."""
    sb = require_supabase()
    body = await request.json()
    user_id = body.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    project = {
        "user_id": int(user_id),
        "title": body.get("title", "Новая карусель"),
        "status": "draft",
        "idea_text": body.get("idea_text"),
        "template_id": body.get("template_id"),
        "instagram_username": body.get("instagram_username"),
        "language": body.get("language", "ru"),
    }
    result = sb.table("projects").insert(project).execute()
    return result.data[0] if result.data else {}


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Get project with its messages."""
    sb = require_supabase()
    project = sb.table("projects").select("*").eq("id", project_id).single().execute()
    messages = sb.table("project_messages").select("*").eq("project_id", project_id).order("created_at").execute()
    return {"project": project.data, "messages": messages.data}


@app.put("/api/projects/{project_id}")
async def update_project(project_id: str, request: Request):
    """Update a project."""
    sb = require_supabase()
    body = await request.json()
    allowed_fields = {"title", "status", "idea_text", "selected_headline", "slides_data",
                      "template_id", "instagram_username", "language", "carousel_images"}
    update_data = {k: v for k, v in body.items() if k in allowed_fields}
    if not update_data:
        raise HTTPException(status_code=400, detail="No valid fields to update")
    result = sb.table("projects").update(update_data).eq("id", project_id).execute()
    return result.data[0] if result.data else {}


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project (cascade deletes messages)."""
    sb = require_supabase()
    sb.table("projects").delete().eq("id", project_id).execute()
    return {"ok": True}


@app.post("/api/projects/{project_id}/messages")
async def add_project_message(project_id: str, request: Request):
    """Add a message to a project's chat."""
    sb = require_supabase()
    body = await request.json()
    message = {
        "project_id": project_id,
        "user_id": int(body.get("user_id", 0)),
        "role": body.get("role", "user"),
        "content": body.get("content", ""),
        "message_type": body.get("message_type", "text"),
        "message_data": body.get("message_data"),
    }
    result = sb.table("project_messages").insert(message).execute()
    return result.data[0] if result.data else {}


# === User Profile & Memory ===

@app.get("/api/user/profile")
async def get_user_profile(user_id: int):
    """Get fresh user profile summary."""
    sb = require_supabase()
    result = sb.table("users").select("profile_summary").eq("user_id", str(user_id)).single().execute()
    return {"profile_summary": result.data.get("profile_summary") if result.data else None}


@app.put("/api/user/profile")
async def update_user_profile(request: Request):
    """Update user profile summary."""
    sb = require_supabase()
    body = await request.json()
    user_id = body.get("user_id")
    profile_summary = body.get("profile_summary", "")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    sb.table("users").update({"profile_summary": profile_summary}).eq("user_id", str(user_id)).execute()
    return {"ok": True}


@app.get("/api/user/memories")
async def get_user_memories(user_id: int):
    """Get all user memories."""
    sb = require_supabase()
    result = sb.table("user_memory").select("id,content,created_at").eq("user_id", str(user_id)).order("created_at", desc=True).execute()
    return result.data or []


@app.delete("/api/user/memories/{memory_id}")
async def delete_user_memory(memory_id: str):
    """Delete a specific memory."""
    sb = require_supabase()
    sb.table("user_memory").delete().eq("id", memory_id).execute()
    return {"ok": True}


@app.get("/api/user/settings")
async def get_user_settings(user_id: int):
    """Get user settings (last_template_id, instagram_usernames)."""
    sb = require_supabase()
    result = sb.table("users").select("last_template_id, instagram_usernames").eq("user_id", str(user_id)).single().execute()
    return result.data or {"last_template_id": "kamalov", "instagram_usernames": []}


@app.patch("/api/user/settings")
async def update_user_settings(request: Request):
    """Update user settings (last_template_id, instagram_usernames)."""
    sb = require_supabase()
    body = await request.json()
    user_id = body.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    update = {}
    if "last_template_id" in body:
        update["last_template_id"] = body["last_template_id"]
    if "instagram_usernames" in body:
        update["instagram_usernames"] = body["instagram_usernames"]
    if update:
        sb.table("users").update(update).eq("user_id", str(user_id)).execute()
    # Language preference — stored in auth_accounts table
    if "language" in body:
        lang = body["language"]
        if lang in ("ru", "en", None):
            try:
                auth_id, _ = await resolve_auth_id(request)
                if auth_id:
                    sb.table("auth_accounts").update({"language": lang}).eq("id", auth_id).execute()
            except Exception as e:
                print(f"[i18n] Error saving language: {e}")
    return {"ok": True}


# === SaaS Auth: JWT verification, identity resolution, usage limits ===

# Plan limits: -1 = unlimited, missing key = unlimited (club)
PLAN_LIMITS = {
    "anonymous":  {"competitor_analysis": 0, "carousel_generate": 0, "ai_chat": 5},
    "free":       {"competitor_analysis": 3, "carousel_generate": 2, "ai_chat": 5},
    "pro":        {"competitor_analysis": 50, "carousel_generate": 50, "ai_chat": 500},
    "business":   {"competitor_analysis": -1, "carousel_generate": -1, "ai_chat": -1},
    "club":       {},
}


def verify_supabase_jwt(token: str) -> dict | None:
    """Verify a Supabase Auth JWT and return decoded claims.
    Tries JWKS (ES256/ECC) first, then falls back to legacy HS256 shared secret."""
    # 1. Try JWKS (ECC/ES256) — current Supabase signing method
    if _jwks_client:
        try:
            signing_key = _jwks_client.get_signing_key_from_jwt(token)
            return pyjwt.decode(token, signing_key.key, algorithms=["ES256"], audience="authenticated")
        except pyjwt.ExpiredSignatureError:
            print("[jwt] Token expired")
            return None
        except Exception as e:
            print(f"[jwt] JWKS verification failed, trying HS256 fallback: {e}")
    # 2. Fallback: legacy HS256 shared secret
    if SUPABASE_JWT_SECRET:
        try:
            return pyjwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], audience="authenticated")
        except pyjwt.ExpiredSignatureError:
            print("[jwt] Token expired (HS256)")
            return None
        except pyjwt.InvalidTokenError as e:
            print(f"[jwt] Invalid token (HS256): {e}")
            return None
    return None


# Cache: auth_id → auth_account dict, TTL 60s
_auth_cache: dict = {}
_AUTH_CACHE_TTL = 60


def _get_auth_account(auth_id: str) -> dict | None:
    """Get auth_account by id with short cache."""
    import time
    now = time.time()
    cached = _auth_cache.get(auth_id)
    if cached and now - cached["_ts"] < _AUTH_CACHE_TTL:
        return cached
    if not supabase:
        return None
    try:
        result = supabase.table("auth_accounts").select("*").eq("id", auth_id).execute()
        if result.data:
            account = result.data[0]
            account["_ts"] = now
            _auth_cache[auth_id] = account
            return account
    except Exception as e:
        print(f"[auth] Failed to get auth_account {auth_id}: {e}")
    return None


def _get_auth_account_by_telegram(telegram_id: int) -> dict | None:
    """Get auth_account by telegram_id."""
    if not supabase:
        return None
    try:
        result = supabase.table("auth_accounts").select("*").eq("telegram_id", telegram_id).execute()
        if result.data:
            return result.data[0]
    except Exception as e:
        print(f"[auth] Failed to get auth_account by telegram_id {telegram_id}: {e}")
    return None


async def resolve_auth_id(request: Request) -> tuple[str | None, str]:
    """Resolve the auth_id from request. Returns (auth_id, auth_type).
    auth_type: 'jwt', 'telegram', 'anonymous'
    """
    # 1. Check JWT (Authorization: Bearer <token>)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        claims = verify_supabase_jwt(token)
        if claims and claims.get("sub"):
            return (claims["sub"], "jwt")

    # 2. Check legacy user_id query param (Telegram ID)
    user_id_param = request.query_params.get("user_id", "")
    if not user_id_param:
        # Also check in JSON body for POST requests
        if request.method == "POST":
            try:
                body = await request.json()
                user_id_param = str(body.get("user_id", ""))
            except Exception:
                pass
    if user_id_param:
        try:
            telegram_id = int(user_id_param)
            account = _get_auth_account_by_telegram(telegram_id)
            if account:
                return (account["id"], "telegram")
        except (ValueError, TypeError):
            # user_id is not a number — might be a Supabase UUID (email/Google user)
            account = _get_auth_account(user_id_param)
            if account:
                return (account["id"], "email")

    # 3. Anonymous — use cookie token
    return (None, "anonymous")


def _get_anon_token(request: Request) -> str | None:
    """Get anonymous tracking token from cookie or header."""
    return request.cookies.get("craft_anon_token") or request.headers.get("X-Anon-Token")


def _get_user_plan(auth_id: str | None) -> str:
    """Get effective plan for user (with club override)."""
    if not auth_id:
        return "anonymous"
    account = _get_auth_account(auth_id)
    if not account:
        return "free"
    if account.get("is_club_member"):
        return "business"
    return account.get("plan", "free")


async def check_usage_limit(auth_id: str | None, anon_token: str | None, action_type: str) -> tuple[bool, int, int]:
    """Check if user is within usage limit. Returns (allowed, used, limit).
    limit = -1 means unlimited.
    """
    plan = _get_user_plan(auth_id)
    limits = PLAN_LIMITS.get(plan, {})
    limit = limits.get(action_type, -1)  # missing = unlimited
    if limit == -1:
        return (True, 0, -1)

    # Count today's usage
    if not supabase:
        return (True, 0, limit)
    try:
        import datetime as _dt
        today_str = _dt.date.today().isoformat()
        if auth_id:
            result = supabase.table("usage_tracking").select("count").eq(
                "auth_id", auth_id
            ).eq("action_type", action_type).eq("date", today_str).execute()
        elif anon_token:
            result = supabase.table("usage_tracking").select("count").eq(
                "anon_token", anon_token
            ).eq("action_type", action_type).eq("date", today_str).execute()
        else:
            return (limit > 0, 0, limit)
        used = result.data[0]["count"] if result.data else 0
        return (used < limit, used, limit)
    except Exception as e:
        print(f"[usage] Check error: {e}")
        return (True, 0, limit)


async def increment_usage(auth_id: str | None, anon_token: str | None, action_type: str):
    """Increment usage counter for today."""
    if not supabase:
        return
    try:
        import datetime
        today = datetime.date.today().isoformat()
        if auth_id:
            # Try upsert
            existing = supabase.table("usage_tracking").select("id,count").eq(
                "auth_id", auth_id
            ).eq("action_type", action_type).eq("date", today).execute()
            if existing.data:
                row = existing.data[0]
                supabase.table("usage_tracking").update({"count": row["count"] + 1}).eq("id", row["id"]).execute()
            else:
                supabase.table("usage_tracking").insert({
                    "auth_id": auth_id, "action_type": action_type, "date": today, "count": 1
                }).execute()
        elif anon_token:
            existing = supabase.table("usage_tracking").select("id,count").eq(
                "anon_token", anon_token
            ).eq("action_type", action_type).eq("date", today).execute()
            if existing.data:
                row = existing.data[0]
                supabase.table("usage_tracking").update({"count": row["count"] + 1}).eq("id", row["id"]).execute()
            else:
                supabase.table("usage_tracking").insert({
                    "anon_token": anon_token, "action_type": action_type, "date": today, "count": 1
                }).execute()
    except Exception as e:
        print(f"[usage] Increment error: {e}")


# === Telegram Auth ===
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_BOT_USERNAME = os.getenv("TELEGRAM_BOT_USERNAME", "")
def get_admin_ids():
    return [x.strip() for x in os.getenv("ADMIN_TELEGRAM_IDS", "").split(",") if x.strip()]
CLUB_CHAT_ID = "-1002841247853"

# Pending login tokens for bot deep link auth
_pending_login_tokens = {}  # token -> {"created": timestamp, "user": None or user_data}
_LOGIN_TOKEN_TTL = 300  # 5 minutes
_webhook_registered = False


def verify_telegram_hash(auth_data: dict) -> bool:
    """Verify data from Telegram Login Widget using HMAC-SHA256."""
    if not TELEGRAM_BOT_TOKEN:
        return False
    check_hash = auth_data.get("hash", "")
    # Build data-check-string: sorted key=value pairs excluding hash
    data_pairs = sorted(
        f"{k}={v}" for k, v in auth_data.items() if k != "hash"
    )
    data_check_string = "\n".join(data_pairs)
    # Secret key = SHA256(bot_token)
    secret_key = hashlib.sha256(TELEGRAM_BOT_TOKEN.encode()).digest()
    # HMAC-SHA256
    computed_hash = hmac.new(
        secret_key, data_check_string.encode(), hashlib.sha256
    ).hexdigest()
    return computed_hash == check_hash


def check_club_membership(user_id: int) -> bool:
    """Check if user is a member of the club channel via Telegram Bot API."""
    if not TELEGRAM_BOT_TOKEN:
        return False
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getChatMember",
            params={"chat_id": CLUB_CHAT_ID, "user_id": user_id},
            timeout=10,
        )
        data = resp.json()
        if resp.status_code == 200 and data.get("ok"):
            status = data["result"].get("status", "")
            return status in ("member", "administrator", "creator")
    except Exception:
        pass
    return False


def upsert_user_to_supabase(user_id: int, first_name: str = "", username: str = "", is_club_member: bool = False, last_name: str = "", photo_url: str = ""):
    """Upsert user into Supabase users table. Returns full user profile."""
    if not supabase:
        return None
    try:
        user_data = {
            "user_id": str(user_id),
            "first_name": first_name,
            "last_name": last_name,
            "username": username or "",
            "photo_url": photo_url,
            "last_active": "now()",
        }
        result = supabase.table("users").upsert(user_data, on_conflict="user_id").execute()
        if result.data:
            return result.data[0]
    except Exception as e:
        print(f"Supabase upsert error: {e}")
    return None


def _ensure_auth_account(telegram_id: int, display_name: str = "", avatar_url: str = "", is_club_member: bool = False, email: str = "") -> dict | None:
    """Ensure auth_accounts row exists for a Telegram user. Returns the auth_account."""
    if not supabase:
        return None
    try:
        # Check if already exists
        existing = supabase.table("auth_accounts").select("*").eq("telegram_id", telegram_id).execute()
        if existing.data:
            # Update club membership and name
            account = existing.data[0]
            updates = {"is_club_member": is_club_member, "club_checked_at": "now()"}
            if display_name:
                updates["display_name"] = display_name
            if avatar_url:
                updates["avatar_url"] = avatar_url
            supabase.table("auth_accounts").update(updates).eq("id", account["id"]).execute()
            account.update(updates)
            return account
        # Create new
        new_account = {
            "telegram_id": telegram_id,
            "display_name": display_name,
            "avatar_url": avatar_url,
            "plan": "free",
            "is_club_member": is_club_member,
            "club_checked_at": "now()",
        }
        if email:
            new_account["email"] = email
        result = supabase.table("auth_accounts").insert(new_account).execute()
        if result.data:
            account = result.data[0]
            # Link to users table
            supabase.table("users").update({"auth_id": account["id"]}).eq("user_id", str(telegram_id)).execute()
            return account
    except Exception as e:
        print(f"[auth] _ensure_auth_account error for telegram_id={telegram_id}: {e}")
    return None


@app.get("/api/auth/config")
async def auth_config():
    """Return auth config for frontend."""
    return {
        "bot_username": TELEGRAM_BOT_USERNAME,
        "supabase_url": SUPABASE_URL,
        "supabase_anon_key": SUPABASE_ANON_KEY,
    }


@app.post("/api/auth/register-account")
async def register_account(request: Request):
    """Create or link auth_accounts entry after Supabase Auth signup (email/Google).
    Called by frontend right after supabase.auth.signUp() or signInWithOAuth() succeeds.
    JWT must be present in Authorization header.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="JWT required")
    claims = verify_supabase_jwt(auth_header[7:])
    if not claims or not claims.get("sub"):
        raise HTTPException(status_code=401, detail="Invalid JWT")

    supabase_uid = claims["sub"]
    email = claims.get("email", "")

    sb = require_supabase()

    # Check if auth_account already exists with this id
    existing = sb.table("auth_accounts").select("*").eq("id", supabase_uid).execute()
    if existing.data:
        return {"auth_account": existing.data[0]}

    # Check if email already exists (e.g., user signed up with email, then Google with same email)
    if email:
        email_existing = sb.table("auth_accounts").select("*").eq("email", email).execute()
        if email_existing.data:
            # Update the id to match Supabase Auth UID
            account = email_existing.data[0]
            sb.table("auth_accounts").update({"id": supabase_uid}).eq("id", account["id"]).execute()
            account["id"] = supabase_uid
            return {"auth_account": account}

    # Create new auth_account with Supabase Auth UID as id
    display_name = claims.get("user_metadata", {}).get("full_name") or claims.get("user_metadata", {}).get("name") or email.split("@")[0] if email else ""
    avatar_url = claims.get("user_metadata", {}).get("avatar_url", "")

    new_account = {
        "id": supabase_uid,
        "email": email or None,
        "display_name": display_name,
        "avatar_url": avatar_url,
        "plan": "free",
        "is_club_member": False,
    }
    # Also set google_id if provider is google
    provider = claims.get("app_metadata", {}).get("provider", "")
    if provider == "google":
        google_id = claims.get("user_metadata", {}).get("provider_id") or claims.get("sub")
        new_account["google_id"] = google_id

    result = sb.table("auth_accounts").insert(new_account).execute()
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create auth account")

    # Also create a users row for backward compatibility
    account = result.data[0]
    try:
        # Use a negative hash of UUID as pseudo-user_id for non-Telegram users
        pseudo_id = abs(hash(supabase_uid)) % (10**15)
        sb.table("users").upsert({
            "user_id": str(pseudo_id),
            "auth_id": supabase_uid,
            "first_name": display_name,
            "photo_url": avatar_url,
            "last_active": "now()",
        }, on_conflict="user_id").execute()
    except Exception as e:
        print(f"[auth] users row creation error (non-fatal): {e}")

    return {"auth_account": account}


@app.post("/api/auth/link-telegram")
async def link_telegram(request: Request):
    """Link Telegram account to an existing auth_account (email/Google user).
    Requires JWT + Telegram auth data.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="JWT required")
    claims = verify_supabase_jwt(auth_header[7:])
    if not claims or not claims.get("sub"):
        raise HTTPException(status_code=401, detail="Invalid JWT")

    auth_id = claims["sub"]
    body = await request.json()
    telegram_data = body.get("telegram_data", {})
    if not telegram_data:
        raise HTTPException(status_code=400, detail="telegram_data required")

    # Verify Telegram hash
    if not verify_telegram_hash(telegram_data):
        raise HTTPException(status_code=401, detail="Invalid Telegram auth data")

    telegram_id = int(telegram_data.get("id", 0))
    if not telegram_id:
        raise HTTPException(status_code=400, detail="Invalid telegram user id")

    sb = require_supabase()

    # Check if this telegram_id is already linked to another account
    existing_tg = sb.table("auth_accounts").select("id").eq("telegram_id", telegram_id).execute()
    if existing_tg.data:
        old_auth_id = existing_tg.data[0]["id"]
        if old_auth_id != auth_id:
            raise HTTPException(status_code=409, detail="error.account_already_linked")

    # Update current auth_account with telegram_id
    is_club_member = check_club_membership(telegram_id)
    sb.table("auth_accounts").update({
        "telegram_id": telegram_id,
        "is_club_member": is_club_member,
        "club_checked_at": "now()",
    }).eq("id", auth_id).execute()

    # Clear cache
    _auth_cache.pop(auth_id, None)

    updated = sb.table("auth_accounts").select("*").eq("id", auth_id).execute()
    return {
        "auth_account": updated.data[0] if updated.data else None,
        "is_club_member": is_club_member,
    }


@app.post("/api/auth/link-telegram-by-id")
async def link_telegram_by_id(request: Request):
    """Link Telegram account to existing auth_account using verified telegram_id.
    Called after bot deep link flow confirms Telegram identity.
    Requires JWT (proves email/Google identity) + telegram_id (verified via bot).
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="JWT required")
    claims = verify_supabase_jwt(auth_header[7:])
    if not claims or not claims.get("sub"):
        raise HTTPException(status_code=401, detail="Invalid JWT")

    auth_id = claims["sub"]
    body = await request.json()
    telegram_id = body.get("telegram_id")
    if not telegram_id:
        raise HTTPException(status_code=400, detail="telegram_id required")
    telegram_id = int(telegram_id)

    sb = require_supabase()

    # Check if this telegram_id is already linked to another account
    existing_tg = sb.table("auth_accounts").select("id").eq("telegram_id", telegram_id).execute()
    if existing_tg.data:
        old_auth_id = existing_tg.data[0]["id"]
        if old_auth_id != auth_id:
            raise HTTPException(status_code=409, detail="error.account_already_linked")

    # Update current auth_account with telegram_id
    is_club_member = check_club_membership(telegram_id)
    sb.table("auth_accounts").update({
        "telegram_id": telegram_id,
        "is_club_member": is_club_member,
        "club_checked_at": "now()",
    }).eq("id", auth_id).execute()

    # Clear cache
    _auth_cache.pop(auth_id, None)

    updated = sb.table("auth_accounts").select("*").eq("id", auth_id).execute()
    return {
        "auth_account": updated.data[0] if updated.data else None,
        "is_club_member": is_club_member,
    }


@app.post("/api/auth/link-email")
async def link_email(request: Request):
    """Link Email/Password account to an existing auth_account (Telegram user).
    Requires user_id (Telegram ID) + supabase_token (Email JWT) in body.
    """
    body = await request.json()

    # 1. Resolve current auth_id from user_id (try telegram_id first, then auth_id)
    user_id_str = str(body.get("user_id", ""))
    auth_id = None
    sb = require_supabase()
    if user_id_str:
        try:
            telegram_id = int(user_id_str)
            account = _get_auth_account_by_telegram(telegram_id)
            if account:
                auth_id = account["id"]
        except (ValueError, TypeError):
            pass
        # Fallback: try as auth_id directly
        if not auth_id:
            existing = sb.table("auth_accounts").select("id").eq("id", user_id_str).execute()
            if existing.data:
                auth_id = existing.data[0]["id"]
    if not auth_id:
        raise HTTPException(status_code=401, detail="Not authenticated — user_id required")

    # 2. Get Supabase JWT from body (proves Email identity)
    supabase_token = body.get("supabase_token")
    if not supabase_token:
        raise HTTPException(status_code=400, detail="supabase_token required")

    claims = verify_supabase_jwt(supabase_token)
    if not claims or not claims.get("sub"):
        raise HTTPException(status_code=401, detail="Invalid Supabase token")

    supabase_uid = claims["sub"]
    email = claims.get("email", "")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")

    # 3. Check if supabase_uid already linked to another account
    existing = sb.table("auth_accounts").select("id").eq("id", supabase_uid).execute()
    if existing.data and existing.data[0]["id"] != auth_id:
        raise HTTPException(status_code=409, detail="error.account_already_linked")

    # Check email conflict
    email_existing = sb.table("auth_accounts").select("id").eq("email", email).execute()
    if email_existing.data and email_existing.data[0]["id"] != auth_id:
        raise HTTPException(status_code=409, detail="error.account_already_linked")

    # 4. Update auth_account: set email and change id to supabase_uid
    old_auth_id = auth_id
    updates = {"email": email, "id": supabase_uid}
    sb.table("auth_accounts").update(updates).eq("id", old_auth_id).execute()

    # Update users table reference
    sb.table("users").update({"auth_id": supabase_uid}).eq("auth_id", old_auth_id).execute()

    # Update usage_tracking references
    try:
        sb.table("usage_tracking").update({"auth_id": supabase_uid}).eq("auth_id", old_auth_id).execute()
    except Exception:
        pass  # Non-fatal

    # Clear cache
    _auth_cache.pop(old_auth_id, None)

    updated = sb.table("auth_accounts").select("*").eq("id", supabase_uid).execute()
    return {"auth_account": updated.data[0] if updated.data else None}


@app.post("/api/auth/link-google")
async def link_google(request: Request):
    """Link Google/Email account to an existing auth_account (Telegram user).
    Requires user_id (Telegram ID) + supabase_token (Google OAuth JWT) in body.
    """
    # Read body once (resolve_auth_id also reads body, so we do manual resolution)
    body = await request.json()

    # 1. Resolve current auth_id from user_id in body
    user_id_str = str(body.get("user_id", ""))
    auth_id = None
    if user_id_str:
        try:
            telegram_id = int(user_id_str)
            account = _get_auth_account_by_telegram(telegram_id)
            if account:
                auth_id = account["id"]
        except (ValueError, TypeError):
            pass
    if not auth_id:
        raise HTTPException(status_code=401, detail="Not authenticated — user_id required")

    # 2. Get Supabase JWT from body (proves Google/Email identity)
    supabase_token = body.get("supabase_token")
    if not supabase_token:
        raise HTTPException(status_code=400, detail="supabase_token required")

    claims = verify_supabase_jwt(supabase_token)
    if not claims or not claims.get("sub"):
        raise HTTPException(status_code=401, detail="Invalid Supabase token")

    supabase_uid = claims["sub"]
    email = claims.get("email", "")
    provider = claims.get("app_metadata", {}).get("provider", "")
    google_id = claims.get("user_metadata", {}).get("provider_id") if provider == "google" else None

    sb = require_supabase()

    # 3. Check if supabase_uid already linked to another account
    existing = sb.table("auth_accounts").select("id").eq("id", supabase_uid).execute()
    if existing.data and existing.data[0]["id"] != auth_id:
        raise HTTPException(status_code=409, detail="error.account_already_linked")

    # Check email conflict
    if email:
        email_existing = sb.table("auth_accounts").select("id").eq("email", email).execute()
        if email_existing.data and email_existing.data[0]["id"] != auth_id:
            raise HTTPException(status_code=409, detail="error.account_already_linked")

    # 4. Update auth_account: set email, google_id, and change id to supabase_uid
    old_auth_id = auth_id
    updates = {}
    if email:
        updates["email"] = email
    if google_id:
        updates["google_id"] = google_id

    # Change the auth_account id to match Supabase Auth UID
    # so JWT-based auth (Google/Email login) finds this same account
    updates["id"] = supabase_uid
    sb.table("auth_accounts").update(updates).eq("id", old_auth_id).execute()

    # Update users table reference
    sb.table("users").update({"auth_id": supabase_uid}).eq("auth_id", old_auth_id).execute()

    # Update usage_tracking references
    try:
        sb.table("usage_tracking").update({"auth_id": supabase_uid}).eq("auth_id", old_auth_id).execute()
    except Exception:
        pass  # Non-fatal

    # Clear cache
    _auth_cache.pop(old_auth_id, None)

    updated = sb.table("auth_accounts").select("*").eq("id", supabase_uid).execute()
    return {"auth_account": updated.data[0] if updated.data else None}


@app.get("/api/auth/me")
async def auth_me(request: Request):
    """Get current user profile based on JWT or user_id."""
    auth_id, auth_type = await resolve_auth_id(request)
    if not auth_id:
        return {"user": None, "auth_type": "anonymous"}

    account = _get_auth_account(auth_id)
    if not account:
        return {"user": None, "auth_type": auth_type}

    # Also get legacy user data
    user_data = {}
    if account.get("telegram_id"):
        try:
            result = supabase.table("users").select("*").eq("user_id", str(account["telegram_id"])).execute()
            if result.data:
                user_data = result.data[0]
        except Exception:
            pass

    is_admin = str(account.get("telegram_id", "")) in get_admin_ids() if account.get("telegram_id") else False

    return {
        "user": {
            "id": account.get("telegram_id") or user_data.get("user_id"),
            "auth_id": account["id"],
            "email": account.get("email"),
            "telegram_id": account.get("telegram_id"),
            "google_id": account.get("google_id"),
            "first_name": account.get("display_name") or user_data.get("first_name", ""),
            "last_name": user_data.get("last_name", ""),
            "username": user_data.get("username", ""),
            "photo_url": account.get("avatar_url") or user_data.get("photo_url", ""),
            "is_club_member": account.get("is_club_member", False),
            "is_admin": is_admin,
            "plan": account.get("plan", "free"),
            "carousels_used": user_data.get("carousels_used", 0),
            "carousels_limit": user_data.get("carousels_limit", 10),
            "profile_summary": user_data.get("profile_summary"),
        },
        "auth_type": auth_type,
    }


# ========================
# PAYMENTS (Lava.top)
# ========================

PLAN_OFFER_MAP = {
    "pro": LAVA_TOP_OFFER_PRO,
    "business": LAVA_TOP_OFFER_BUSINESS,
}


@app.post("/api/payments/create-invoice")
async def create_invoice(request: Request):
    """Create a Lava.top invoice for plan upgrade."""
    if not LAVA_TOP_API_KEY:
        raise HTTPException(status_code=503, detail="Payments not configured")

    auth_id, _ = await resolve_auth_id(request)
    if not auth_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Read body manually since resolve_auth_id may have consumed it
    body = await request.json()
    plan = body.get("plan", "")
    currency = body.get("currency", "USD")

    if plan not in PLAN_OFFER_MAP:
        raise HTTPException(status_code=400, detail=f"Invalid plan: {plan}. Must be 'pro' or 'business'")

    offer_id = PLAN_OFFER_MAP[plan]
    if not offer_id:
        raise HTTPException(status_code=503, detail=f"Offer for plan '{plan}' not configured")

    # Get user email
    sb = require_supabase()
    account = sb.table("auth_accounts").select("email").eq("id", auth_id).execute()
    email = account.data[0]["email"] if account.data and account.data[0].get("email") else None
    if not email:
        raise HTTPException(status_code=400, detail="Email required for payment. Please link your email in Settings first.")

    # Create invoice via Lava.top API
    try:
        resp = requests.post(
            f"{LAVA_TOP_API_URL}/api/v3/invoice",
            headers={"X-Api-Key": LAVA_TOP_API_KEY, "Content-Type": "application/json"},
            json={
                "email": email,
                "offerId": offer_id,
                "currency": currency,
                "periodicity": "MONTHLY",
                "buyerLanguage": "RU",
                "clientUtm": {"auth_id": str(auth_id), "plan": plan},
            },
            timeout=15,
        )
        if resp.status_code == 201:
            data = resp.json()
            return {"paymentUrl": data.get("paymentUrl"), "invoiceId": data.get("id")}
        else:
            print(f"[lava] Create invoice error {resp.status_code}: {resp.text}")
            raise HTTPException(status_code=502, detail="Payment service error")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Payment service timeout")
    except Exception as e:
        print(f"[lava] Create invoice exception: {e}")
        raise HTTPException(status_code=500, detail="Payment error")


@app.post("/api/payments/webhook")
async def payments_webhook(request: Request):
    """Handle Lava.top webhook events."""
    import hashlib, hmac

    body_bytes = await request.body()
    body_str = body_bytes.decode("utf-8")

    # Verify API key — Lava.top sends our secret in X-Api-Key header
    if LAVA_TOP_WEBHOOK_SECRET:
        incoming_key = request.headers.get("X-Api-Key", "")
        if not hmac.compare_digest(incoming_key, LAVA_TOP_WEBHOOK_SECRET):
            print(f"[lava webhook] Invalid API key")
            raise HTTPException(status_code=403, detail="Invalid API key")

    import json as json_mod
    try:
        payload = json_mod.loads(body_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event_type = payload.get("eventType", "")
    buyer_email = payload.get("buyer", {}).get("email", "")
    contract_id = payload.get("contractId", "")
    amount = payload.get("amount")
    currency = payload.get("currency", "USD")
    product_title = payload.get("product", {}).get("title", "")
    client_utm = payload.get("clientUtm", {})

    print(f"[lava webhook] {event_type} | email={buyer_email} | contract={contract_id} | product={product_title}")

    sb = require_supabase()

    # Find auth_account by email
    auth_account = None
    if buyer_email:
        result = sb.table("auth_accounts").select("*").eq("email", buyer_email).execute()
        if result.data:
            auth_account = result.data[0]

    if not auth_account:
        print(f"[lava webhook] No auth_account found for email: {buyer_email}")
        return {"status": "ok", "note": "no matching account"}

    auth_id = auth_account["id"]

    # Determine plan from UTM or product title
    plan = client_utm.get("plan", "")
    if not plan:
        title_lower = product_title.lower()
        if "business" in title_lower:
            plan = "business"
        elif "pro" in title_lower:
            plan = "pro"
        else:
            plan = "pro"  # default

    import datetime

    if event_type == "payment.success":
        # Activate subscription
        sb.table("auth_accounts").update({
            "plan": plan, "updated_at": datetime.datetime.utcnow().isoformat()
        }).eq("id", auth_id).execute()

        # Create/update subscription record
        sb.table("subscriptions").upsert({
            "auth_id": auth_id,
            "plan": plan,
            "status": "active",
            "lava_subscription_id": contract_id,
            "amount": int(amount * 100) if amount else 0,
            "currency": currency,
            "period_start": datetime.datetime.utcnow().isoformat(),
            "period_end": (datetime.datetime.utcnow() + datetime.timedelta(days=30)).isoformat(),
            "updated_at": datetime.datetime.utcnow().isoformat(),
        }, on_conflict="auth_id").execute()

        # Clear auth cache
        _auth_cache.pop(auth_id, None)
        print(f"[lava webhook] Activated {plan} for {buyer_email}")

    elif event_type == "subscription.recurring.payment.success":
        # Renew subscription
        sb.table("auth_accounts").update({
            "plan": plan, "updated_at": datetime.datetime.utcnow().isoformat()
        }).eq("id", auth_id).execute()

        sb.table("subscriptions").update({
            "status": "active",
            "period_start": datetime.datetime.utcnow().isoformat(),
            "period_end": (datetime.datetime.utcnow() + datetime.timedelta(days=30)).isoformat(),
            "updated_at": datetime.datetime.utcnow().isoformat(),
        }).eq("auth_id", auth_id).execute()

        _auth_cache.pop(auth_id, None)
        print(f"[lava webhook] Renewed {plan} for {buyer_email}")

    elif event_type == "subscription.cancelled":
        sb.table("auth_accounts").update({
            "plan": "free", "updated_at": datetime.datetime.utcnow().isoformat()
        }).eq("id", auth_id).execute()

        sb.table("subscriptions").update({
            "status": "cancelled",
            "updated_at": datetime.datetime.utcnow().isoformat(),
        }).eq("auth_id", auth_id).execute()

        _auth_cache.pop(auth_id, None)
        print(f"[lava webhook] Cancelled for {buyer_email}")

    elif event_type == "subscription.recurring.payment.failed":
        sb.table("subscriptions").update({
            "status": "past_due",
            "updated_at": datetime.datetime.utcnow().isoformat(),
        }).eq("auth_id", auth_id).execute()
        print(f"[lava webhook] Payment failed for {buyer_email}")

    return {"status": "ok"}


@app.get("/api/payments/subscription")
async def get_subscription(request: Request):
    """Get user's subscription status and usage."""
    auth_id, _ = await resolve_auth_id(request)
    if not auth_id:
        # Return free plan info for unauthenticated users instead of 401
        anon_limits = PLAN_LIMITS.get("anonymous", {})
        return {
            "plan": "free",
            "effective_plan": "anonymous",
            "is_club_member": False,
            "subscription": None,
            "usage": {
                action: {"used": 0, "limit": anon_limits.get(action, 0)}
                for action in ["ai_chat", "carousel_generate", "competitor_analysis"]
            },
        }

    sb = require_supabase()
    account = sb.table("auth_accounts").select("plan,is_club_member,language").eq("id", auth_id).execute()
    plan = "free"
    is_club = False
    user_language = None
    if account.data:
        plan = account.data[0].get("plan", "free")
        is_club = account.data[0].get("is_club_member", False)
        user_language = account.data[0].get("language")

    # Get subscription info
    sub_data = None
    sub = sb.table("subscriptions").select("*").eq("auth_id", auth_id).execute()
    if sub.data:
        s = sub.data[0]
        sub_data = {
            "status": s.get("status"),
            "plan": s.get("plan"),
            "period_end": s.get("period_end"),
            "amount": s.get("amount"),
            "currency": s.get("currency"),
        }

    # Get today's usage
    import datetime
    today = datetime.date.today().isoformat()
    effective_plan = "business" if is_club else plan
    limits = PLAN_LIMITS.get(effective_plan, {})

    usage = {}
    for action in ["ai_chat", "carousel_generate", "competitor_analysis"]:
        try:
            result = sb.table("usage_tracking").select("count").eq(
                "auth_id", auth_id
            ).eq("action_type", action).eq("date", today).execute()
            used = result.data[0]["count"] if result.data else 0
        except Exception:
            used = 0
        limit = limits.get(action, -1)
        usage[action] = {"used": used, "limit": limit}

    return {
        "plan": plan,
        "effective_plan": effective_plan,
        "is_club_member": is_club,
        "subscription": sub_data,
        "usage": usage,
        "language": user_language,
    }


@app.post("/api/payments/cancel")
async def cancel_subscription(request: Request):
    """Cancel user's subscription."""
    import datetime
    auth_id, _ = await resolve_auth_id(request)
    if not auth_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    sb = require_supabase()

    # Get subscription
    sub = sb.table("subscriptions").select("*").eq("auth_id", auth_id).eq("status", "active").execute()
    if not sub.data:
        raise HTTPException(status_code=404, detail="No active subscription")

    s = sub.data[0]
    contract_id = s.get("lava_subscription_id")

    # Get email
    account = sb.table("auth_accounts").select("email").eq("id", auth_id).execute()
    email = account.data[0]["email"] if account.data else ""

    # Cancel on Lava.top
    if LAVA_TOP_API_KEY and contract_id and email:
        try:
            resp = requests.delete(
                f"{LAVA_TOP_API_URL}/api/v1/subscriptions",
                headers={"X-Api-Key": LAVA_TOP_API_KEY},
                params={"contractId": contract_id, "email": email},
                timeout=15,
            )
            print(f"[lava] Cancel subscription: {resp.status_code}")
        except Exception as e:
            print(f"[lava] Cancel error: {e}")

    # Update local DB
    sb.table("subscriptions").update({
        "status": "cancelled",
        "updated_at": datetime.datetime.utcnow().isoformat(),
    }).eq("auth_id", auth_id).execute()

    sb.table("auth_accounts").update({
        "plan": "free",
        "updated_at": datetime.datetime.utcnow().isoformat(),
    }).eq("id", auth_id).execute()

    _auth_cache.pop(auth_id, None)

    return {"status": "cancelled"}


async def _ensure_webhook(request: Request):
    """Lazily register Telegram webhook on first use."""
    global _webhook_registered
    if _webhook_registered or not TELEGRAM_BOT_TOKEN:
        return
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    proto = request.headers.get("x-forwarded-proto", "https")
    if not host:
        return
    base_url = f"{proto}://{host}"
    webhook_url = f"{base_url}/api/telegram/webhook"
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook",
            json={"url": webhook_url},
            timeout=10
        )
        if resp.status_code == 200:
            _webhook_registered = True
            print(f"✅ Telegram webhook set: {webhook_url}")
        else:
            print(f"⚠️ Webhook setup response: {resp.text}")
    except Exception as e:
        print(f"⚠️ Webhook setup failed: {e}")


@app.post("/api/auth/create-login-token")
async def create_login_token(request: Request):
    """Create a one-time login token for bot deep link auth."""
    # Cleanup expired tokens
    now = time.time()
    expired = [t for t, v in _pending_login_tokens.items() if now - v["created"] > _LOGIN_TOKEN_TTL]
    for t in expired:
        del _pending_login_tokens[t]

    # Ensure webhook is registered
    await _ensure_webhook(request)

    token = secrets.token_urlsafe(16)
    _pending_login_tokens[token] = {"created": now, "user": None}
    return {"token": token, "bot_username": TELEGRAM_BOT_USERNAME}


@app.get("/api/auth/check-login-token/{token}")
async def check_login_token(token: str):
    """Check if a login token has been authenticated via bot."""
    data = _pending_login_tokens.get(token)
    if not data:
        return {"status": "expired"}
    if time.time() - data["created"] > _LOGIN_TOKEN_TTL:
        del _pending_login_tokens[token]
        return {"status": "expired"}
    if data["user"]:
        user = data["user"]
        del _pending_login_tokens[token]
        return {"status": "authenticated", "user": user}
    return {"status": "pending"}


def get_telegram_photo_url(user_id: int) -> str:
    """Fetch user's Telegram profile photo URL via Bot API."""
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUserProfilePhotos",
            params={"user_id": user_id, "limit": 1},
            timeout=5,
        )
        data = r.json()
        photos = data.get("result", {}).get("photos", [])
        if not photos:
            return ""
        file_id = photos[0][0]["file_id"]
        r2 = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile",
            params={"file_id": file_id},
            timeout=5,
        )
        file_path = r2.json().get("result", {}).get("file_path", "")
        if not file_path:
            return ""
        return f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
    except Exception:
        return ""


@app.get("/api/user/avatar/{user_id}")
async def user_avatar(user_id: int):
    """Return a fresh Telegram avatar URL (redirect). Useful when stored URLs expire."""
    from fastapi.responses import RedirectResponse
    # First try DB
    if supabase:
        try:
            row = supabase.table("users").select("photo_url").eq("user_id", str(user_id)).single().execute()
            stored_url = (row.data or {}).get("photo_url", "")
            # CDN URLs (t.me, telegram.org without /file/bot) are stable
            if stored_url and "api.telegram.org/file/bot" not in stored_url and stored_url.startswith("http"):
                return RedirectResponse(stored_url)
        except Exception:
            pass
    # Fallback: fetch fresh from Bot API
    photo_url = get_telegram_photo_url(user_id)
    if not photo_url:
        raise HTTPException(status_code=404, detail="No avatar")
    return RedirectResponse(photo_url)


@app.post("/api/telegram/webhook")
async def telegram_webhook(request: Request):
    """Handle incoming Telegram bot updates (for login deep links)."""
    try:
        update = await request.json()
    except Exception:
        return {"ok": True}

    message = update.get("message", {})
    text = message.get("text", "")
    tg_user = message.get("from", {})
    chat_id = message.get("chat", {}).get("id")

    # Handle /start login_TOKEN
    if text.startswith("/start login_"):
        token = text.replace("/start login_", "").strip()
        pending = _pending_login_tokens.get(token)

        if not pending or time.time() - pending["created"] > _LOGIN_TOKEN_TTL:
            if chat_id:
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                    json={"chat_id": chat_id, "text": "Ссылка устарела. Попробуйте ещё раз на сайте."},
                    timeout=10
                )
            return {"ok": True}

        user_id = tg_user.get("id", 0)
        is_club_member = check_club_membership(user_id)
        is_admin = str(user_id) in get_admin_ids()

        photo_url = get_telegram_photo_url(user_id)
        sb_user = upsert_user_to_supabase(
            user_id, tg_user.get("first_name", ""), tg_user.get("username", ""), is_club_member,
            last_name=tg_user.get("last_name", ""), photo_url=photo_url
        )

        # Ensure auth_account exists
        auth_account = _ensure_auth_account(
            user_id, display_name=tg_user.get("first_name", ""),
            avatar_url=photo_url, is_club_member=is_club_member
        )

        user_response = {
            "id": user_id,
            "auth_id": auth_account["id"] if auth_account else None,
            "first_name": tg_user.get("first_name", ""),
            "last_name": tg_user.get("last_name", ""),
            "username": tg_user.get("username", ""),
            "photo_url": photo_url,
            "is_club_member": is_club_member,
            "is_admin": is_admin,
        }
        if sb_user:
            is_club = auth_account.get("is_club_member", False) if auth_account else False
            raw_plan = auth_account.get("plan", "free") if auth_account else sb_user.get("plan", "free")
            user_response["plan"] = "business" if is_club else raw_plan
            user_response["is_club_member"] = is_club
            user_response["carousels_used"] = sb_user.get("carousels_used", 0)
            user_response["carousels_limit"] = sb_user.get("carousels_limit", 10)
            user_response["profile_summary"] = sb_user.get("profile_summary")

        pending["user"] = user_response

        if chat_id:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": "✅ Вы авторизованы! Вернитесь в браузер."},
                timeout=10
            )

    return {"ok": True}


@app.post("/api/auth/telegram")
async def auth_telegram(data: dict):
    """Authenticate user via Telegram Login Widget."""
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=500, detail="TELEGRAM_BOT_TOKEN not configured")

    # Verify hash
    if not verify_telegram_hash(data):
        raise HTTPException(status_code=401, detail="Invalid Telegram auth data")

    # Check auth_date (not older than 24 hours)
    auth_date = int(data.get("auth_date", 0))
    if time.time() - auth_date > 86400:
        raise HTTPException(status_code=401, detail="Auth data expired")

    user_id = int(data.get("id", 0))
    is_club_member = check_club_membership(user_id)
    is_admin = str(user_id) in get_admin_ids()

    # Upsert user in Supabase
    sb_user = upsert_user_to_supabase(
        user_id, data.get("first_name", ""), data.get("username", ""), is_club_member,
        last_name=data.get("last_name", ""), photo_url=data.get("photo_url", "")
    )

    # Ensure auth_account exists
    auth_account = _ensure_auth_account(
        user_id, display_name=data.get("first_name", ""),
        avatar_url=data.get("photo_url", ""), is_club_member=is_club_member
    )

    user_response = {
        "id": user_id,
        "auth_id": auth_account["id"] if auth_account else None,
        "first_name": data.get("first_name", ""),
        "last_name": data.get("last_name", ""),
        "username": data.get("username", ""),
        "photo_url": data.get("photo_url", ""),
        "is_club_member": is_club_member,
        "is_admin": is_admin,
    }
    if sb_user:
        is_club = auth_account.get("is_club_member", False) if auth_account else False
        raw_plan = auth_account.get("plan", "free") if auth_account else sb_user.get("plan", "free")
        user_response["plan"] = "business" if is_club else raw_plan
        user_response["is_club_member"] = is_club
        user_response["carousels_used"] = sb_user.get("carousels_used", 0)
        user_response["carousels_limit"] = sb_user.get("carousels_limit", 10)
        user_response["profile_summary"] = sb_user.get("profile_summary")

    return {"user": user_response}


def verify_mini_app_data(init_data_raw: str) -> dict:
    """Validate Telegram Mini App initData using HMAC-SHA256."""
    from urllib.parse import unquote
    parsed = dict(
        pair.split("=", 1)
        for pair in unquote(init_data_raw).split("&")
        if "=" in pair
    )
    received_hash = parsed.pop("hash", None)
    if not received_hash:
        raise ValueError("Missing hash")
    # Check freshness (24 hours)
    auth_date = int(parsed.get("auth_date", 0))
    if time.time() - auth_date > 86400:
        raise ValueError("Init data expired")
    # Sort and join
    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(parsed.items()))
    # Secret = HMAC-SHA256("WebAppData", bot_token)
    secret_key = hmac.new(b"WebAppData", TELEGRAM_BOT_TOKEN.encode(), hashlib.sha256).digest()
    computed_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(computed_hash, received_hash):
        raise ValueError("Invalid hash")
    user_data = json.loads(parsed.get("user", "{}"))
    return {"user": user_data, "auth_date": auth_date}


@app.post("/api/auth/miniapp")
async def auth_miniapp(request: Request):
    """Authenticate user via Telegram Mini App initData (auto-login)."""
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=500, detail="TELEGRAM_BOT_TOKEN not configured")
    body = await request.json()
    init_data = body.get("initData", "")
    if not init_data:
        raise HTTPException(status_code=400, detail="Missing initData")
    try:
        data = verify_mini_app_data(init_data)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    user_id = data["user"].get("id", 0)
    is_club_member = check_club_membership(user_id)
    is_admin = str(user_id) in get_admin_ids()

    # Upsert user in Supabase
    sb_user = upsert_user_to_supabase(
        user_id, data["user"].get("first_name", ""), data["user"].get("username", ""), is_club_member,
        last_name=data["user"].get("last_name", ""), photo_url=data["user"].get("photo_url", "")
    )

    # Ensure auth_account exists
    auth_account = _ensure_auth_account(
        user_id, display_name=data["user"].get("first_name", ""),
        avatar_url=data["user"].get("photo_url", ""), is_club_member=is_club_member
    )

    user_response = {
        "id": user_id,
        "auth_id": auth_account["id"] if auth_account else None,
        "first_name": data["user"].get("first_name", ""),
        "last_name": data["user"].get("last_name", ""),
        "username": data["user"].get("username", ""),
        "photo_url": data["user"].get("photo_url", ""),
        "is_club_member": is_club_member,
        "is_admin": is_admin,
    }
    if sb_user:
        is_club = auth_account.get("is_club_member", False) if auth_account else False
        raw_plan = auth_account.get("plan", "free") if auth_account else sb_user.get("plan", "free")
        user_response["plan"] = "business" if is_club else raw_plan
        user_response["is_club_member"] = is_club
        user_response["carousels_used"] = sb_user.get("carousels_used", 0)
        user_response["carousels_limit"] = sb_user.get("carousels_limit", 10)
        user_response["profile_summary"] = sb_user.get("profile_summary")

    return {"user": user_response}


@app.post("/api/send-to-chat")
async def send_images_to_chat(request: Request):
    """Send generated images to user via Telegram bot"""
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=500, detail="TELEGRAM_BOT_TOKEN not configured")

    data = await request.json()
    chat_id = data.get("chat_id")
    images = data.get("images", [])  # [{base64: "data:image/png;base64,...", filename: "slide_1.png"}]

    if not chat_id or not images:
        raise HTTPException(status_code=400, detail="chat_id and images required")

    # Prepare image files
    files = {}
    media = []
    for i, img in enumerate(images):
        b64 = img.get("base64", "")
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        attach_key = f"slide_{i}"
        files[attach_key] = (f"slide_{i+1}.png", img_bytes, "image/png")
        media.append({"type": "photo", "media": f"attach://{attach_key}"})

    # Send as media group (album), max 10 per group
    sent_count = 0
    for chunk_start in range(0, len(media), 10):
        chunk_media = media[chunk_start:chunk_start + 10]
        chunk_files = {k: v for k, v in files.items() if any(m["media"] == f"attach://{k}" for m in chunk_media)}
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMediaGroup",
                data={"chat_id": chat_id, "media": json.dumps(chunk_media)},
                files=chunk_files,
                timeout=60,
            )
            if resp.status_code == 200:
                sent_count += len(chunk_media)
        except Exception:
            pass

    return {"sent": sent_count, "total": len(images)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
