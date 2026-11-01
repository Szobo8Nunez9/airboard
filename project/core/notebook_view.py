"""A two-page view of original ink and clean, editable text."""
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def layout(width, height, ink_width, ink_height):
    margin, gap = 24, 24
    page_width = (width - margin * 2 - gap) // 2
    page_height = height - 155
    left = (margin, 104, page_width, page_height)
    right = (margin + page_width + gap, 104, page_width, page_height)
    scale = min((page_width - 32) / ink_width, (page_height - 115) / ink_height)
    iw, ih = round(ink_width * scale), round(ink_height * scale)
    ink = (left[0] + (page_width - iw) // 2, left[1] + 65, iw, ih)
    labels = [('Prev', ','), ('Next', '.'), ('New page', 'n'), ('Tidy text', 't'),
              ('Edit text', 'e'), ('Undo text', 'u'), ('PDF', 'f'), ('Board', 'b')]
    button_width = min(128, (width - 32) // len(labels))
    buttons = [(label, key, (16 + i * button_width, 48, button_width - 8, 34))
               for i, (label, key) in enumerate(labels)]
    return left, right, ink, buttons


@lru_cache(maxsize=8)
def font(size):
    for path in ('/System/Library/Fonts/Supplemental/Arial.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def wrap_text(text, width, size=21):
    measure = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    lines = []
    for paragraph in text.split('\n'):
        line = ''
        for word in paragraph.split(' '):
            candidate = (line + ' ' + word).lstrip()
            if measure.textlength(candidate, font=font(size)) <= width:
                line = candidate
            else:
                if line:
                    lines.append(line)
                line = word
                while measure.textlength(line, font=font(size)) > width and len(line) > 1:
                    cut = len(line) - 1
                    while cut > 1 and measure.textlength(line[:cut], font=font(size)) > width:
                        cut -= 1
                    lines.append(line[:cut])
                    line = line[cut:]
        lines.append(line)
    return lines or ['']


def render(book, ink_display, source, notice='', editing=False, scroll=0, fingertip=None, clear_progress=0, erasing=False):
    height, width = ink_display.shape[:2]
    left, right, ink, buttons = layout(width, height, book.width, book.height)
    result = np.full_like(ink_display, (232, 236, 238))
    for x, y, w, h in (left, right):
        cv2.rectangle(result, (x, y), (x+w, y+h), (249, 253, 255), -1)
        cv2.rectangle(result, (x, y), (x+w, y+h), (202, 211, 216), 1)
    x, y, w, h = ink
    result[y:y+h, x:x+w] = cv2.resize(ink_display, (w, h))
    if fingertip is not None:
        point = x + round(fingertip[0] * w / book.width), y + round(fingertip[1] * h / book.height)
        cv2.circle(result, point, max(5, round(45*w/book.width)) if erasing else 5, (130, 95, 20), 2, cv2.LINE_AA)
    for label, key, (bx, by, bw, bh) in buttons:
        cv2.rectangle(result, (bx, by), (bx+bw, by+bh), (242, 247, 249), -1)
        cv2.rectangle(result, (bx, by), (bx+bw, by+bh), (192, 204, 210), 1)
    rgb = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(rgb)
    draw.text((24, 12), 'airboard notebook', font=font(25), fill='#183d4b')
    draw.text((max(290, width // 2 - 65), 17), f'Page {book.index+1} of {len(book.pages)}', font=font(18), fill='#365568')
    draw.text((width-220, 18), source, font=font(16), fill='#365568')
    for label, key, (bx, by, bw, bh) in buttons:
        draw.text((bx+9, by+8), label, font=font(15 if width < 1000 else 17), fill='#183d4b')
    draw.text((left[0]+16, left[1]+16), 'Your handwriting', font=font(23), fill='#183d4b')
    draw.text((right[0]+16, right[1]+16), 'Clean text' + ('  [editing]' if editing else ''), font=font(23), fill='#183d4b')
    page = book.current
    text = page.text or 'Write on the left, then choose Tidy text.\n\nYou can correct any words with Edit text.'
    size = 20 if width >= 1000 else 17
    lines = wrap_text(text, right[2]-40, size)
    line_height = size + 9
    capacity = max(1, (right[3]-130)//line_height)
    start = max(0, min(scroll, max(0, len(lines)-capacity)))
    if editing:
        start = max(0, len(lines)-capacity)
    for number, line in enumerate(lines[start:start+capacity]):
        draw.text((right[0]+20, right[1]+66+number*line_height), line, font=font(size), fill='#213846')
    if editing:
        draw.text((right[0]+20, right[1]+right[3]-49), 'Type to edit. Enter adds a line. Esc finishes.', font=font(14), fill='#416b78')
    elif page.stale:
        draw.text((right[0]+20, right[1]+right[3]-49), 'Ink changed. Tidy again to include the new words.', font=font(14), fill='#965c21')
    elif page.edited:
        draw.text((right[0]+20, right[1]+right[3]-49), 'Your edited wording. Original ink is kept.', font=font(14), fill='#416b78')
    elif page.text:
        draw.text((right[0]+20, right[1]+right[3]-49), 'Check the words against the original handwriting.', font=font(14), fill='#965c21')
    draw.text((left[0]+16, left[1]+left[3]-49), 'Original ink stays on this page.', font=font(14), fill='#416b78')
    if len(lines) > capacity:
        draw.text((right[0]+20, right[1]+right[3]-27), 'j/k scroll text. PDF keeps all the words.', font=font(14), fill='#416b78')
    status = notice or 'Local recognition only. No API key or image upload needed. b returns to the board.'
    draw.text((24, height-37), status[:130], font=font(15), fill='#365568')
    result = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    if clear_progress > 0:
        cv2.rectangle(result, (left[0]+16, left[1]+left[3]-16),
                      (left[0]+16+int((left[2]-32)*clear_progress), left[1]+left[3]-10), (65, 65, 210), -1)
    return result
