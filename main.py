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
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import requests
import base64
import json
import os
import io
import re
import uuid
import numpy as np

app = FastAPI(title="Carousel Studio", version="7.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("fonts", exist_ok=True)
os.makedirs("output", exist_ok=True)

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
    settings: Dict[str, Any] = {}
    slides: List[Dict[str, Any]]
    createdAt: Optional[str] = None


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
                f"uploads/{family}.ttf",  # NEW: Custom fonts
                f"uploads/{family}.otf",  # NEW: Custom fonts
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
                self.font_cache[key] = ImageFont.load_default()

        return self.font_cache[key]

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

    def get_text_width(self, text: str, font, letter_spacing: int, draw) -> int:
        """Calculate text width with letter spacing"""
        if letter_spacing == 0:
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0]
        else:
            # Calculate width char by char with spacing
            total_width = 0
            for char in text:
                bbox = draw.textbbox((0, 0), char, font=font)
                total_width += (bbox[2] - bbox[0]) + letter_spacing
            # Remove last letter spacing
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

        # NEW v7.0: Parse both *bold* and _highlight_
        segments = []
        # Combined pattern: *text* for bold, _text_ for highlight
        pattern = r'\*([^*]+)\*|_([^_]+)_'
        last_end = 0
        for match in re.finditer(pattern, content):
            if match.start() > last_end:
                segments.append({'text': content[last_end:match.start()], 'hl': False, 'bold': False})

            if match.group(1):  # *text* = bold
                segments.append({'text': match.group(1), 'hl': False, 'bold': True})
            else:  # _text_ = highlight
                segments.append({'text': match.group(2), 'hl': True, 'bold': False})

            last_end = match.end()

        if last_end < len(content):
            segments.append({'text': content[last_end:], 'hl': False, 'bold': False})

        if not segments:
            segments = [{'text': content, 'hl': False, 'bold': False}]

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
                    test_width = self.get_text_width(test, font, letter_spacing, draw)
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

        curr_y = y
        for line_segs in lines:
            if not line_segs:
                curr_y += int(font_size * line_height)
                continue

            line_text = ''.join(s['text'] for s in line_segs)
            # Use helper function that accounts for letter spacing
            line_w = self.get_text_width(line_text, font, letter_spacing, draw)

            curr_x = x - line_w if align == 'right' else (x - line_w // 2 if align == 'center' else x)

            for seg in line_segs:
                col = hl_color if seg['hl'] else base_color
                # NEW v7.0: Use bold font for *bold* segments
                seg_font = bold_font if seg.get('bold') else font

                if letter_spacing != 0:
                    # NEW: Посимвольный рендеринг с letter-spacing
                    for char in seg['text']:
                        draw.text((curr_x, curr_y), char, font=seg_font, fill=col)
                        bbox = draw.textbbox((0, 0), char, font=seg_font)
                        curr_x += (bbox[2] - bbox[0]) + letter_spacing
                else:
                    # Обычный рендеринг без letter-spacing
                    draw.text((curr_x, curr_y), seg['text'], font=seg_font, fill=col)
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

        font = self.get_font(font_family, font_size, font_weight)

        # Создаём временный draw для вычисления
        temp_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_img)

        # Разбиваем на сегменты
        # NEW v7.0: Parse both *bold* and _highlight_
        segments = []
        # Combined pattern: *text* for bold, _text_ for highlight
        pattern = r'\*([^*]+)\*|_([^_]+)_'
        last_end = 0
        for match in re.finditer(pattern, content):
            if match.start() > last_end:
                segments.append({'text': content[last_end:match.start()], 'hl': False, 'bold': False})

            if match.group(1):  # *text* = bold
                segments.append({'text': match.group(1), 'hl': False, 'bold': True})
            else:  # _text_ = highlight
                segments.append({'text': match.group(2), 'hl': True, 'bold': False})

            last_end = match.end()

        if last_end < len(content):
            segments.append({'text': content[last_end:], 'hl': False, 'bold': False})

        if not segments:
            segments = [{'text': content, 'hl': False, 'bold': False}]

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
                    test_width = self.get_text_width(test, font, letter_spacing, draw)
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

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "7.0",  # NEW: Updated version
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
    file_path = f"uploads/{safe_filename}"

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
async def list_templates():
    templates = []
    for f in os.listdir("templates"):
        if f.endswith('.json'):
            try:
                with open(f"templates/{f}", 'r', encoding='utf-8') as file:
                    d = json.load(file)
                    templates.append({
                        "name": d.get('name', f.replace('.json', '')),
                        "createdAt": d.get('createdAt', ''),
                        "slidesCount": len(d.get('slides', []))
                    })
            except:
                pass
    return {"templates": templates}


@app.get("/templates/{name}")
async def get_template(name: str):
    safe = re.sub(r'[^a-zA-Z0-9_\-а-яА-ЯёЁ]', '_', name)
    path = f"templates/{safe}.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@app.post("/templates")
async def save_template(template: TemplateData):
    from datetime import datetime
    t = template.dict()
    t['createdAt'] = datetime.now().isoformat()
    safe = re.sub(r'[^a-zA-Z0-9_\-а-яА-ЯёЁ]', '_', template.name)
    with open(f"templates/{safe}.json", 'w', encoding='utf-8') as f:
        json.dump(t, f, ensure_ascii=False, indent=2)
    return {"success": True, "name": template.name}


@app.delete("/templates/{name}")
async def delete_template(name: str):
    safe = re.sub(r'[^a-zA-Z0-9_\-а-яА-ЯёЁ]', '_', name)
    path = f"templates/{safe}.json"
    if os.path.exists(path):
        os.remove(path)
        return {"success": True}
    raise HTTPException(status_code=404, detail="Not found")


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
    """
    template_name = request.template_id or request.template_name
    username = request.username or request.USERNAME or "@username"

    if not template_name:
        raise HTTPException(status_code=400, detail="template_name or template_id required")

    safe = re.sub(r'[^a-zA-Z0-9_\-а-яА-ЯёЁ]', '_', template_name)
    path = f"templates/{safe}.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")

    with open(path, 'r', encoding='utf-8') as f:
        template = json.load(f)

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
    content_template = content_slides[0] if content_slides else slides[0]  # Fallback template

    # NEW: Функция для выбора content template с fallback
    def get_content_template(index: int):
        """Выбрать content{index} template или fallback на content"""
        content_type = f'content{index}'
        if content_type in content_specific:
            return content_specific[content_type]
        else:
            return content_template  # Fallback

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
        with open(f"output/{filename}", 'wb') as f:
            f.write(buf.getvalue())

        rendered.append({
            "slide_number": i + 1,
            "url": f"https://carousel-generator-production.up.railway.app/output/{filename}",
            "filename": filename,
            "base64": f"data:image/png;base64,{b64}"  # оставь на случай, если понадобится
        })

    return {
        "success": True,
        "slides": rendered
    }


@app.get("/output/{filename}")
async def get_output(filename: str):
    path = f"output/{filename}"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
