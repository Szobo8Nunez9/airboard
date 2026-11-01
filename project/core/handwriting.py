"""Local Apple Vision OCR. Formatting changes whitespace only, never wording."""
from pathlib import Path
from statistics import median
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def tidy_spacing(text):
    """Keep all recognized words, punctuation, spelling, and line breaks."""
    lines = [re.sub(r'[^\S\n]+', ' ', line).strip() for line in text.replace('\r\n', '\n').replace('\r', '\n').split('\n')]
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()


def arrange_lines(lines):
    """Order observations on a simple single-column page; preserve raw candidates."""
    valid = [dict(line) for line in lines if isinstance(line.get('text'), str) and line['text'].strip()]
    valid.sort(key=lambda line: (-float(line.get('y', 0)) - float(line.get('height', .02)) / 2, float(line.get('x', 0))))
    rows = []
    for line in valid:
        center = float(line.get('y', 0)) + float(line.get('height', .02)) / 2
        height = max(.001, float(line.get('height', .02)))
        row = next((r for r in rows if abs(r['center'] - center) < min(r['height'], height) * .6), None)
        if row is None:
            rows.append({'center': center, 'height': height, 'parts': [line]})
        else:
            row['parts'].append(line)
    rows.sort(key=lambda row: -row['center'])
    typical_height = median([r['height'] for r in rows]) if rows else .02
    output = []
    previous = None
    for row in rows:
        if previous is not None and previous - row['center'] > typical_height * 2.4:
            output.append('')
        output.append(' '.join(p['text'] for p in sorted(row['parts'], key=lambda p: float(p.get('x', 0)))))
        previous = row['center']
    return tidy_spacing('\n'.join(output))


def ensure_helper(cache_dir=None):
    if sys.platform != 'darwin':
        raise RuntimeError('Local handwriting recognition currently requires macOS. You can still type with Edit text.')
    source = ROOT / 'native' / 'recognize_text.swift'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache = Path(cache_dir) if cache_dir else ROOT / '.cache' / 'ocr'
    cache.mkdir(parents=True, exist_ok=True)
    helper = cache / f'vision-text-{digest}'
    if helper.is_file() and os.access(helper, os.X_OK):
        return helper
    compiler = shutil.which('swiftc')
    if compiler is None:
        raise RuntimeError('Local recognition needs Apple command-line tools. Use Edit text for now.')
    modules = cache / 'modules'
    modules.mkdir(exist_ok=True)
    candidate = helper.with_suffix('.building')
    process = subprocess.run([compiler, '-O', '-module-cache-path', str(modules), str(source), '-o', str(candidate)],
                             capture_output=True, text=True, timeout=120)
    if process.returncode:
        raise RuntimeError('Could not prepare local text recognition. Use Edit text; your ink is unchanged.')
    candidate.replace(helper)
    return helper


def recognize_handwriting(png_bytes, cache_dir=None):
    if not png_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('Expected a PNG image')
    helper = ensure_helper(cache_dir)
    with tempfile.TemporaryDirectory(prefix='airboard-ocr-') as directory:
        path = Path(directory) / 'page.png'
        path.write_bytes(png_bytes)
        process = subprocess.run([str(helper), str(path)], capture_output=True, text=True, timeout=45)
    if process.returncode:
        raise RuntimeError('Local text recognition could not read this page. The original writing is kept.')
    result = json.loads(process.stdout)
    lines = result.get('lines', [])
    result['text'] = arrange_lines(lines)
    result['needs_review'] = True
    result['low_confidence'] = any(float(line.get('confidence', 0)) < .5 for line in lines)
    if not result['text']:
        result['notice'] = 'No readable words found. Write larger separate letters, or use Edit text.'
    else:
        result['notice'] = 'Review the recognized words beside your handwriting. Spelling was not corrected.'
    return result
