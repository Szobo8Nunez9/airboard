"""Notebook pages retain original ink and independently editable transcription."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import re
import uuid

import cv2
import numpy as np


@dataclass
class NotebookPage:
    ink: np.ndarray
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    text: str = ''
    ocr_lines: list = field(default_factory=list)
    source: str = ''
    transcribed_hash: str = ''
    stale: bool = False
    edited: bool = False
    needs_review: bool = False
    text_history: list = field(default_factory=list)

    def ink_hash(self):
        return hashlib.sha256(self.ink.tobytes()).hexdigest()


class Notebook:
    def __init__(self, width=1280, height=720):
        self.width, self.height = width, height
        self.pages = [self._blank()]
        self.index = 0

    def _blank(self):
        return NotebookPage(np.zeros((self.height, self.width, 3), np.uint8))

    @property
    def current(self):
        return self.pages[self.index]

    def new_page(self):
        self.pages.append(self._blank())
        self.index = len(self.pages) - 1
        return self.current

    def turn(self, step):
        destination = max(0, min(len(self.pages) - 1, self.index + step))
        changed = destination != self.index
        self.index = destination
        return changed

    def find(self, page_id):
        return next((p for p in self.pages if p.id == page_id), None)

    def mark_ink_changed(self):
        if self.current.text:
            self.current.stale = True

    def apply_transcription(self, page_id, ink_hash, result):
        page = self.find(page_id)
        if page is None:
            return False
        text = result.get('text', '')
        if not text.strip():
            return False  # An empty/failed recognition must not erase existing notes.
        if page.text and page.text != text:
            page.text_history = (page.text_history + [page.text])[-10:]
        page.text = text
        page.ocr_lines = result.get('lines', [])
        page.source = result.get('provider', 'local OCR')
        page.transcribed_hash = ink_hash
        page.stale = page.ink_hash() != ink_hash
        page.edited = False
        page.needs_review = True
        return True

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        records = []
        for page in self.pages:
            digest = page.ink_hash()
            name = f'{page.id}-{digest[:16]}.png'
            path = folder / name
            if not path.exists():
                ok, data = cv2.imencode('.png', page.ink)
                if not ok:
                    raise OSError('Could not save notebook ink')
                temporary = path.with_suffix('.tmp')
                temporary.write_bytes(data.tobytes())
                temporary.replace(path)
            records.append({
                'id': page.id, 'image': name, 'text': page.text,
                'ocr_lines': page.ocr_lines, 'source': page.source,
                'transcribed_hash': page.transcribed_hash, 'stale': page.stale,
                'edited': page.edited, 'needs_review': page.needs_review, 'text_history': page.text_history,
            })
        data = {'version': 1, 'width': self.width, 'height': self.height, 'index': self.index, 'pages': records}
        temporary = folder / 'notebook.json.tmp'
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(folder / 'notebook.json')
        return folder / 'notebook.json'

    @classmethod
    def load(cls, folder, width=1280, height=720):
        folder = Path(folder)
        state = folder / 'notebook.json'
        if not state.exists():
            return cls(width, height)
        try:
            data = json.loads(state.read_text(encoding='utf-8'))
            if data.get('version') != 1 or not data.get('pages'):
                raise ValueError('Unrecognized notebook format')
            notebook = cls(width, height)
            notebook.pages = []
            for record in data['pages']:
                if not re.fullmatch(r'[a-f0-9]{32}', record['id']):
                    raise ValueError('Invalid page identifier')
                if not re.fullmatch(r'[a-f0-9]{32}-[a-f0-9]{16}\.png', record['image']):
                    raise ValueError('Invalid page image path')
                ink = cv2.imread(str(folder / record['image']))
                if ink is None:
                    raise ValueError('A notebook ink image is missing')
                if ink.shape[:2] != (height, width):
                    ink = cv2.resize(ink, (width, height))
                notebook.pages.append(NotebookPage(
                    ink=ink, id=record['id'], text=str(record.get('text', '')),
                    ocr_lines=record.get('ocr_lines', []), source=record.get('source', ''),
                    transcribed_hash=record.get('transcribed_hash', ''), stale=record.get('stale', False),
                    edited=record.get('edited', False), needs_review=record.get('needs_review', False),
                    text_history=record.get('text_history', []),
                ))
            notebook.index = max(0, min(len(notebook.pages) - 1, int(data.get('index', 0))))
            return notebook
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise ValueError('Could not open the saved notebook. Its files were left unchanged. Use another --notebook-dir to start fresh.') from error
