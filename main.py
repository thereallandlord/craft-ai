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
from fastapi.responses import FileResponse
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
import numpy as np
from supabase import create_client, Client as SupabaseClient

app = FastAPI(title="Carousel Studio", version="7.0")

# === Supabase ===
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
supabase: SupabaseClient | None = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print("✅ Supabase connected")
else:
    print("⚠️ Supabase not configured (SUPABASE_URL / SUPABASE_SERVICE_KEY missing)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def telegram_headers(request: Request, call_next):
    response = await call_next(request)
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


@app.get("/templates")
async def list_templates(preview: str = "full", user_id: str = ""):
    """List templates from Supabase. Returns system + personal + published."""
    if not supabase:
        return {"templates": []}

    templates = []

    def format_row(row, ttype: str):
        all_slides = row.get("slides") or []
        if preview == 'light':
            slides_data = [all_slides[0]] if all_slides else []
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
async def get_template(identifier: str):
    """Get single template by template_id from Supabase."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    try:
        result = supabase.table("user_templates").select("*").eq("template_id", identifier).execute()
        if result.data and len(result.data) > 0:
            row = result.data[0]
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

    tid = template.template_id or generate_template_id(template.name)

    try:
        row = {
            "user_id": template.user_id,
            "template_id": tid,
            "name": template.name,
            "slides": template.slides,
            "settings": template.settings,
            "updated_at": datetime.now().isoformat()
        }
        supabase.table("user_templates").upsert(row, on_conflict="user_id,template_id").execute()
        return {"success": True, "name": template.name, "template_id": tid, "type": "personal"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving template: {e}")


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
    if str(user_id) not in ADMIN_TELEGRAM_IDS:
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
        if str(user_id) in ADMIN_TELEGRAM_IDS:
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
    if str(user_id) not in ADMIN_TELEGRAM_IDS:
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
async def generate_carousel(request: GenerateRequest):
    """
    ПРАВИЛЬНАЯ ЛОГИКА v6.2:
    1. Первый слайд из request.slides → рендерится по INTRO template
    2. Остальные слайды из request.slides → рендерятся по CONTENT template
    3. Ending слайды → добавляются в конец (КАК ЕСТЬ из шаблона)

    NEW v7.0: Поиск шаблона по template_id с fallback на template_name
    """
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
        else:
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

SYSTEM_PROMPT_DRAFT = """Ты — AI контент-менеджер для Instagram-каруселей. Ты помогаешь создавать контент на основе идеи пользователя.

Сейчас этап: ГЕНЕРАЦИЯ ЗАГОЛОВКОВ.
На основе идеи пользователя предложи 5-7 цепляющих заголовков для карусели.

Формат ответа:
1. Заголовок 1
2. Заголовок 2
...

Пиши на языке пользователя. Заголовки должны быть короткими (3-7 слов), цепляющими, вызывать любопытство."""

SYSTEM_PROMPT_HEADLINES = """Ты — AI контент-менеджер для Instagram-каруселей. Пользователь выбрал заголовок. Теперь напиши текст для слайдов карусели.

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


def get_system_prompt(status: str, slides_count: int = 7, custom_prompts: dict = None) -> str:
    cp = custom_prompts or {}
    if status == 'draft':
        return cp.get('draft') or SYSTEM_PROMPT_DRAFT
    elif status == 'headlines':
        custom = cp.get('headlines')
        if custom:
            return custom.replace('{slides_count}', str(slides_count))
        return SYSTEM_PROMPT_HEADLINES.replace('{slides_count}', str(slides_count))
    elif status == 'text_ready':
        return cp.get('text_ready') or SYSTEM_PROMPT_TEXT_READY
    return cp.get('draft') or SYSTEM_PROMPT_DRAFT


def get_user_context(user_id: int, query: str = "") -> str:
    """Get user profile and memory for personalization."""
    if not supabase:
        return ""
    context_parts = []
    try:
        user = supabase.table("users").select("profile_summary").eq("user_id", str(user_id)).single().execute()
        if user.data and user.data.get("profile_summary"):
            context_parts.append(f"Профиль пользователя: {user.data['profile_summary']}")
    except Exception:
        pass
    # Vector search for relevant memories
    try:
        memories = get_relevant_memories(user_id, query)
        if memories:
            context_parts.append("Факты о пользователе:\n" + "\n".join(f"- {m}" for m in memories))
    except Exception:
        pass
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
            except Exception:
                pass
    # Fallback: get recent memories without vector search
    try:
        result = supabase.table("user_memory").select("content").eq("user_id", str(user_id)).order("created_at", desc=True).limit(limit).execute()
        return [r["content"] for r in (result.data or [])]
    except Exception:
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

        ai_text = _call_openrouter([
            {"role": "system", "content": prompt_content},
            {"role": "user", "content": conversation}
        ], memory_model)
        print(f"[Memory] AI response: {ai_text[:200]}")

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
        if not prompt or prompt == SYSTEM_PROMPT_DRAFT:
            prompt = f"На основе следующих фактов о пользователе создай краткий профиль (3-5 предложений): его ниша, стиль контента, целевая аудитория, предпочтения.\n\nФакты:\n{memories_text}"
            profile_model = "openai/gpt-4o-mini"

        summary = _call_openrouter([
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Создай профиль пользователя."}
        ], profile_model)

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

    if chat_type == 'headlines':
        prompt_key = 'headlines'
    elif chat_type == 'text':
        prompt_key = 'editing' if has_slides else 'text'
    elif chat_type == 'carousel':
        prompt_key = 'carousel'
    else:
        prompt_key = chat_type  # memory_extract, profile_summarize, etc.

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
        'headlines': SYSTEM_PROMPT_DRAFT,
        'text': SYSTEM_PROMPT_HEADLINES,
        'editing': SYSTEM_PROMPT_TEXT_READY,
        'carousel': "Ты — AI помощник по дизайну каруселей. Помоги пользователю выбрать визуальное оформление.",
        'format_text': open(os.path.join(os.path.dirname(__file__), "prompts", "format_carousel_text.txt"), encoding="utf-8").read() if os.path.exists(os.path.join(os.path.dirname(__file__), "prompts", "format_carousel_text.txt")) else "Parse text into JSON array of slides with TITLE and DESCRIPTION fields."
    }

    prompt_data = _prompt_cache.get(prompt_key)
    if prompt_data:
        template = prompt_data["content"]
        model = prompt_data.get("model", "openai/gpt-4o")
    else:
        template = fallbacks.get(prompt_key, SYSTEM_PROMPT_DRAFT)
        fallback_models = {'format_text': 'openai/gpt-4o-mini', 'memory_extract': 'openai/gpt-4o-mini', 'profile_summarize': 'openai/gpt-4o-mini'}
        model = fallback_models.get(prompt_key, "openai/gpt-4o")

    if variables:
        for key, value in variables.items():
            template = template.replace(f'{{{key}}}', str(value))

    return template, model


def _call_openrouter(messages: list, model: str = "openai/gpt-4o", temperature: float = 0.7) -> str:
    """Call OpenRouter API and return AI text response."""
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
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"OpenRouter error: {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]


def _parse_headlines(text: str) -> list:
    """Parse numbered headlines from AI response."""
    headlines = []
    for line in text.split('\n'):
        line = line.strip()
        match = re.match(r'^\d+[\.\)]\s*(.+)', line)
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
        result = _call_openrouter(messages, model, temperature=0.3)
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
        ai_text = _call_openrouter(ai_messages, body.get("model") or prompt_model)
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
        system_prompt, prompt_model = get_system_prompt_v3("text", variables={
            "slides_count": str(slides_count),
            "selected_headline": selected_headline,
        })
        user_context = get_user_context(int(user_id))
        if user_context:
            system_prompt += f"\n\nКонтекст о пользователе:\n{user_context}"

        try:
            ai_text = _call_openrouter([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Напиши текст карусели на тему: {selected_headline}"}
            ], body.get("model") or prompt_model)

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

    # === v3 mode: sub_chat_id ===
    if sub_chat_id:
        sub_chat = sb.table("sub_chats").select("*").eq("id", sub_chat_id).single().execute()
        if not sub_chat.data:
            raise HTTPException(status_code=404, detail="Sub-chat not found")
        sc = sub_chat.data
        topic_id = sc["topic_id"]
        chat_type = sc["chat_type"]

        # Determine prompt key
        if chat_type == "text" and sc.get("slides_data"):
            prompt_key = "editing"
        else:
            prompt_key = chat_type  # headlines, text, carousel

        # Build system prompt from DB
        variables = {
            "slides_count": str(slides_count),
            "selected_headline": sc.get("selected_headline") or "",
        }
        system_prompt, prompt_model = get_system_prompt_v3(prompt_key, variables=variables)
        user_context = get_user_context(int(user_id), query=user_message)
        if user_context:
            system_prompt += f"\n\nКонтекст о пользователе:\n{user_context}"

        # Get message history for this sub-chat
        messages_result = sb.table("project_messages").select("role,content").eq("sub_chat_id", sub_chat_id).order("created_at").execute()
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
            ai_text = _call_openrouter(ai_messages, body.get("model") or prompt_model)
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
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://app.craftopen.space",
                "X-Title": "Craft AI"
            },
            json={
                "model": body.get("model", "openai/gpt-4o"),
                "messages": ai_messages,
                "temperature": 0.7,
                "max_tokens": 4096
            },
            timeout=120
        )

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"OpenRouter error: {resp.text}")

        ai_response = resp.json()
        ai_text = ai_response["choices"][0]["message"]["content"]

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
async def list_prompts(user_id: int):
    """List all system prompts. Admin only. Seeds defaults if empty."""
    if str(user_id) not in ADMIN_TELEGRAM_IDS:
        raise HTTPException(status_code=403, detail="Admin access required")
    sb = require_supabase()
    result = sb.table("system_prompts").select("*").order("prompt_key").execute()
    # Seed missing default prompts
    existing_keys = {row["prompt_key"] for row in (result.data or [])}
    defaults = [
            {"prompt_key": "headlines", "title": "Генератор заголовков", "description": "Генерирует 5-7 вариантов заголовков для карусели",
             "model": "google/gemini-2.5-flash",
             "content": """# РОЛЬ: Генератор вирусных заголовков для Instagram

Ты — креативный копирайтер, который создаёшь цепляющие заголовки для каруселей.
Твоя задача — создать максимально релевантные заголовки на основе информации о пользователе.

## АЛГОРИТМ РАБОТЫ

### ЭТАП 1: Анализ сообщения пользователя

Пользователь может прислать:
- Просто тему ("инвестиции", "похудение")
- Тему + информацию о себе ("инвестиции, моя аудитория — новички с зарплатой 50-100к")
- Подробный рассказ о себе, продукте, аудитории

### ЭТАП 2: Генерация заголовков

На основе:
1. Темы от пользователя
2. Контекста о пользователе (если есть)
3. Новой информации (если дал)

Сгенерируй 5-7 уникальных заголовков.

### ЭТАП 3: Показ результата

Формат ответа — просто пронумерованный список заголовков:
1. [заголовок]
2. [заголовок]
...

## ТРЕБОВАНИЯ К ЗАГОЛОВКАМ

### Стиль:
- Короткие, ёмкие (макс 60 символов)
- Разговорный язык
- На "ты"
- Конкретика > абстракции

### Триггеры (миксуй разные):

Боль: "Ты тратишь 5 часов на контент, а подписчики не растут"
Контраст: "Они набирают 10К в месяц. Ты — 200 за полгода"
Цифры: "847 подписчиков за 14 дней"
Провокация: "Нейросеть заменит тебя через 6 месяцев"
Секрет: "Алгоритм изменился. 90% не знают"
Срочность: "60 дней, пока ниша не переполнена"
Вопрос: "Почему твой контент не смотрят?"

### Персонализация:
- Если знаешь ЦА — используй их боли и язык
- Если знаешь продукт — упоминай релевантные темы
- Если знаешь нишу — используй специфику

## ЗАПРЕЩЕНО

- "Давайте разберём"
- "В современном мире"
- "Это не просто..."
- "Секрет успеха в том..."
- "Давай я тебе сейчас расскажу"
- "Итак, друзья..."
- Канцелярит и официоз
- Вода без конкретики
- Заголовки длиннее 60 символов
- Повторяющиеся структуры

## ПРИМЕРЫ

Пример 1 — Только тема:
User: "инвестиции"
→ Если есть контекст (ЦА: новички 25-35 лет), учти его
→ "1. Первые 100К на бирже без опыта\n2. Почему твой вклад съедает инфляция\n..."

Пример 2 — Тема + информация:
User: "заголовок про выгорание, моя аудитория — предприниматели с доходом от 500к"
→ "1. Заработал миллион, а кайфа нет\n2. Почему успешные тоже выгорают\n..."

Формат: просто пронумерованный список заголовков, без лишнего текста.""",
             "variables": [{"name": "user_context", "description": "Профиль и стиль пользователя"}]},
            {"prompt_key": "text", "title": "Автор текста слайдов", "description": "Пишет текст слайдов карусели на основе выбранного заголовка",
             "model": "google/gemini-2.5-flash",
             "content": """# РОЛЬ: КОПИРАЙТЕР INSTAGRAM-КАРУСЕЛЕЙ

Ты создаёшь тексты для Instagram-каруселей. Работаешь строго по структуре.

## ДАННЫЕ

- Заголовок карусели: "{selected_headline}"
- Количество слайдов: {slides_count}

## СТРУКТУРА КАРУСЕЛИ

### Слайд 1 — ОБЛОЖКА (INTRO)
- Заголовок: ТОЧНО как "{selected_headline}" (без изменений!)
- Описание: Короткий подзаголовок-интрига (до 80 символов)

### Слайд 2 — ХУК (HOOK)
- Заголовок: Зачем это читать? (2-4 слова)
- Описание: Какую проблему решает карусель? (до 150 символов)

### Слайды 3-8 — КОНТЕНТ (CONTENT)
- Заголовок: Заголовок пункта (короткий, ёмкий)
- Описание: Раскрытие пункта (до 200 символов)

### Предпоследний слайд — ВЫВОД (CONCLUSION)
- Заголовок: "Главное" / "Итог" / "Запомни"
- Описание: Суть всей карусели (до 150 символов)

### Последний слайд — ПРИЗЫВ (CTA)
- Заголовок: "Что дальше?"
- Описание: Призыв к действию (до 120 символов)

## ПРАВИЛА НАПИСАНИЯ

### Стиль:
- Пиши на "ты"
- Короткие предложения (max 15-20 слов)
- Без воды — каждое слово работает
- Как человек, не как робот
- Можно материться (если в тему)

### Персонализация:
- Если есть контекст о ЦА — обращайся к ней
- Если есть контекст о продукте — органично упомяни
- Если есть контекст о стиле — следуй ему
- Если ничего не известно — пиши универсально

### СТОП-СЛОВА (ЗАПРЕЩЕНЫ):
- "Это не просто"
- "Представьте"
- "Совершил революцию"
- "В современном мире"
- "Как известно"
- "Не секрет, что"
- "Уникальный" (без доказательств)
- "Инновационный"

## ФОРМАТ ОТВЕТА СТРОГО

Слайд 1
Заголовок: ...
Описание: ...

Слайд 2
Заголовок: ...
Описание: ...

(и так далее для всех слайдов)

## КРИТИЧЕСКИЕ ПРАВИЛА

1. "{selected_headline}" → TITLE слайда 1 (без изменений!)
2. {slides_count} → точное количество слайдов
3. НЕ спрашивай заголовок — он уже дан
4. НЕ спрашивай количество слайдов — оно уже дано
5. НЕ придумывай свой заголовок
6. Пустой контекст о пользователе — НЕ ошибка, пиши универсально""",
             "variables": [{"name": "slides_count", "description": "Количество слайдов"}, {"name": "selected_headline", "description": "Выбранный заголовок"}, {"name": "user_context", "description": "Профиль пользователя"}]},
            {"prompt_key": "editing", "title": "Редактор текста", "description": "Помогает улучшить и отредактировать существующий текст карусели",
             "model": "openai/gpt-4o",
             "content": """# РОЛЬ: Оркестратор редактирования карусели

Текст карусели уже готов. Помоги пользователю улучшить или отредактировать его.

## АЛГОРИТМ РАБОТЫ

### Обработка запросов на изменение:

Если просят изменить конкретный слайд — перепиши только его.
Если просят изменить стиль — перепиши все слайды в новом стиле.
Если просят добавить/удалить слайд — сделай это и покажи результат.

### Формат ответа:

Всегда сохраняй формат:
Слайд N
Заголовок: ...
Описание: ...

### Если пользователь даёт информацию о себе:

Если пользователь сообщает новую информацию (ЦА, продукт, ниша, стиль):
- Учти её при редактировании
- Примеры: "моя аудитория — мамы в декрете" → учитывай при правках
- "пиши дерзко" → меняй стиль

## ПРАВИЛА

### Обязательно:
- Жди указаний пользователя
- Пиши на "ты", кратко, ёмко
- Показывай обновлённый текст после правок
- Если пользователь говорит "да" / "всё ок" — подтверди что текст финальный

### Запрещено:
- НЕ переписывай всё без запроса
- НЕ добавляй слайды без запроса
- НЕ меняй заголовок (Слайд 1) без прямого запроса
- НЕ генерируй картинки — только текст

## ПРИМЕРЫ ДИАЛОГОВ

Пример 1 — Правка конкретного слайда:
User: "Измени описание на слайде 3, сделай более дерзким"
→ Вноси правки, показывай обновлённый слайд 3

Пример 2 — Общая правка:
User: "Сделай весь текст короче"
→ Перепиши все слайды короче, покажи результат

Пример 3 — Подтверждение:
User: "Всё ок, нравится"
→ "Отлично! Текст готов."
""",
             "variables": [{"name": "user_context", "description": "Профиль пользователя"}]},
            {"prompt_key": "carousel", "title": "Дизайнер каруселей", "description": "Помогает с визуальным оформлением карусели",
             "model": "openai/gpt-4o",
             "content": """# РОЛЬ: Дизайнер каруселей

Помоги пользователю с визуальным оформлением карусели: шаблон, цвета, шрифты, фотографии.

## АЛГОРИТМ

1. Выслушай пожелания пользователя
2. Дай конкретные рекомендации по оформлению
3. Если пользователь спрашивает про шаблон — опиши различия между доступными
4. Если спрашивает про цвета — предложи палитру под нишу/тему

## ПРАВИЛА

- Отвечай кратко и по делу
- Давай конкретные рекомендации, а не общие фразы
- Учитывай нишу и ЦА пользователя (если известны)
- Пиши на "ты"

## ЗАПРЕЩЕНО

- Не генерируй текст слайдов — только визуал
- Не используй канцелярит
- Не давай слишком много вариантов — максимум 3 конкретных""",
             "variables": [{"name": "user_context", "description": "Профиль пользователя"}]},
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
    if changed:
        result = sb.table("system_prompts").select("*").order("prompt_key").execute()
    return result.data or []


@app.put("/api/admin/prompts/{prompt_key}")
async def update_prompt(prompt_key: str, request: Request):
    """Update a system prompt. Admin only."""
    global _prompt_cache, _prompt_cache_time
    sb = require_supabase()
    body = await request.json()
    user_id = body.get("user_id")
    if not user_id or str(user_id) not in ADMIN_TELEGRAM_IDS:
        raise HTTPException(status_code=403, detail="Admin access required")

    allowed_fields = {"title", "description", "content", "variables", "is_active", "model"}
    update_data = {k: v for k, v in body.items() if k in allowed_fields}
    if not update_data:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    update_data["updated_by"] = int(user_id)
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
    global _prompt_cache, _prompt_cache_time
    sb = require_supabase()
    body = await request.json()
    user_id = body.get("user_id")
    if not user_id or str(user_id) not in ADMIN_TELEGRAM_IDS:
        raise HTTPException(status_code=403, detail="Admin access required")

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
    return {"ok": True}


# === Telegram Auth ===
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_BOT_USERNAME = os.getenv("TELEGRAM_BOT_USERNAME", "")
ADMIN_TELEGRAM_IDS = [x.strip() for x in os.getenv("ADMIN_TELEGRAM_IDS", "").split(",") if x.strip()]
CLUB_CHAT_ID = "-1002841247853"


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


def upsert_user_to_supabase(user_id: int, first_name: str = "", username: str = "", is_club_member: bool = False):
    """Upsert user into Supabase users table. Returns full user profile."""
    if not supabase:
        return None
    try:
        user_data = {
            "user_id": str(user_id),
            "first_name": first_name,
            "username": username or "",
            "last_active": "now()",
        }
        result = supabase.table("users").upsert(user_data, on_conflict="user_id").execute()
        if result.data:
            return result.data[0]
    except Exception as e:
        print(f"Supabase upsert error: {e}")
    return None


@app.get("/api/auth/config")
async def auth_config():
    """Return auth config for frontend (bot username)."""
    return {"bot_username": TELEGRAM_BOT_USERNAME}


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
    is_admin = str(user_id) in ADMIN_TELEGRAM_IDS

    # Upsert user in Supabase
    sb_user = upsert_user_to_supabase(
        user_id, data.get("first_name", ""), data.get("username", ""), is_club_member
    )

    user_response = {
        "id": user_id,
        "first_name": data.get("first_name", ""),
        "last_name": data.get("last_name", ""),
        "username": data.get("username", ""),
        "photo_url": data.get("photo_url", ""),
        "is_club_member": is_club_member,
        "is_admin": is_admin,
    }
    if sb_user:
        user_response["plan"] = sb_user.get("plan", "free")
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
    is_admin = str(user_id) in ADMIN_TELEGRAM_IDS

    # Upsert user in Supabase
    sb_user = upsert_user_to_supabase(
        user_id, data["user"].get("first_name", ""), data["user"].get("username", ""), is_club_member
    )

    user_response = {
        "id": user_id,
        "first_name": data["user"].get("first_name", ""),
        "last_name": data["user"].get("last_name", ""),
        "username": data["user"].get("username", ""),
        "photo_url": data["user"].get("photo_url", ""),
        "is_club_member": is_club_member,
        "is_admin": is_admin,
    }
    if sb_user:
        user_response["plan"] = sb_user.get("plan", "free")
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
