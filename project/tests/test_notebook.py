"""Notebook regression checks. No camera, API, or native OCR is invoked here."""
from concurrent.futures import Future
import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from airboard import AirboardApp
from core.handwriting import tidy_spacing, arrange_lines, ensure_helper
from core.notebook import Notebook
from core.notebook_export import export_notebook_pdf
from core import notebook_view


@pytest.fixture
def app(tmp_path):
    instance = AirboardApp(mouse=True)
    instance.notebook_folder = tmp_path / 'book'
    instance.snapshots_dir = tmp_path / 'snapshots'
    yield instance
    instance.executor.shutdown(wait=True, cancel_futures=True)


def result(text):
    return {'text': text, 'provider': 'test fixture', 'lines': [], 'needs_review': True}


def test_spacing_preserves_wording_and_typos():
    assert tidy_spacing('  teh   book\t stays! \n\n\n my  words  ') == 'teh book stays!\n\nmy words'


def test_text_lines_follow_reading_order():
    lines = [
        {'text': 'second line', 'x': .1, 'y': .4, 'height': .05},
        {'text': 'world', 'x': .5, 'y': .8, 'height': .05},
        {'text': 'Hello', 'x': .1, 'y': .8, 'height': .05},
    ]
    text = arrange_lines(lines)
    assert text.startswith('Hello world')
    assert text.endswith('second line')


def test_recognition_does_not_modify_ink():
    book = Notebook(50, 40)
    book.current.ink[10:20, 10:20] = (230, 110, 20)
    original = book.current.ink.copy()
    book.apply_transcription(book.current.id, book.current.ink_hash(), result('My own words'))
    assert np.array_equal(book.current.ink, original)
    assert book.current.text == 'My own words'
    assert book.current.needs_review


def test_delayed_recognition_attaches_to_original_page():
    book = Notebook(30, 20)
    first_id, first_hash = book.current.id, book.current.ink_hash()
    book.new_page()
    assert book.apply_transcription(first_id, first_hash, result('Page one'))
    assert book.current.text == ''
    assert book.pages[0].text == 'Page one'


def test_changed_ink_marks_transcription_stale():
    book = Notebook(30, 20)
    before = book.current.ink_hash()
    book.current.ink[5, 5] = 255
    book.apply_transcription(book.current.id, before, result('Old snapshot'))
    assert book.current.stale


def test_empty_recognition_preserves_existing_text():
    book = Notebook()
    book.current.text = 'Carefully edited words'
    assert not book.apply_transcription(book.current.id, book.current.ink_hash(), result(''))
    assert book.current.text == 'Carefully edited words'


def test_previous_text_is_kept_for_undo():
    book = Notebook()
    book.current.text = 'Original words'
    book.apply_transcription(book.current.id, book.current.ink_hash(), result('New recognition'))
    assert book.current.text_history == ['Original words']


def test_pages_round_trip_with_ink_and_text(tmp_path):
    book = Notebook(50, 40)
    first_id = book.current.id
    book.current.ink[12, 15] = (20, 40, 230)
    book.current.text = 'First page & punctuation < >'
    book.current.edited = True
    book.new_page()
    book.current.text = 'Second page'
    book.save(tmp_path)
    loaded = Notebook.load(tmp_path, 50, 40)
    assert len(loaded.pages) == 2 and loaded.index == 1
    assert loaded.pages[0].id == first_id
    assert tuple(loaded.pages[0].ink[12, 15]) == (20, 40, 230)
    assert loaded.pages[0].text == 'First page & punctuation < >'
    assert loaded.current.text == 'Second page'


def test_corrupt_notebook_is_not_overwritten(tmp_path):
    path = tmp_path / 'notebook.json'
    path.write_text('{bad json')
    with pytest.raises(ValueError, match='left unchanged'):
        Notebook.load(tmp_path)
    assert path.read_text() == '{bad json'


def test_saved_state_cannot_read_arbitrary_image_paths(tmp_path):
    book = Notebook()
    book.save(tmp_path)
    path = tmp_path / 'notebook.json'
    data = json.loads(path.read_text())
    data['pages'][0]['image'] = '../private.png'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        Notebook.load(tmp_path)


def test_pdf_export_keeps_model_text_and_original_ink(tmp_path):
    book = Notebook(100, 60)
    book.current.text = 'My exact words & <tags>\nKeep teh spelling.'
    book.current.ink[20:25, 20:25] = (200, 60, 30)
    before = book.current.ink.copy()
    path = export_notebook_pdf(book, tmp_path / 'notes.pdf')
    assert path.read_bytes().startswith(b'%PDF')
    assert path.stat().st_size > 1000
    assert np.array_equal(book.current.ink, before)
    assert book.current.text == 'My exact words & <tags>\nKeep teh spelling.'


def test_non_mac_has_clear_manual_edit_fallback(monkeypatch):
    monkeypatch.setattr('core.handwriting.sys.platform', 'linux')
    with pytest.raises(RuntimeError, match='Edit text'):
        ensure_helper()


def test_book_pointer_maps_to_the_ink_page(app):
    app.notebook_view = True
    _, _, (x, y, w, h), _ = notebook_view.layout(app.width, app.height, app.width, app.height)
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, x+w//2, y+h//2, 0)
    assert abs(app.pointer[0]-app.width/2) < 3
    assert abs(app.pointer[1]-app.height/2) < 3
    assert app.pointer_draw
    app.on_mouse(cv2.EVENT_LBUTTONUP, 0, 0, 0)
    assert not app.pointer_draw


def test_new_page_preserves_previous_ink(app):
    app.canvas.draw_segment((200, 200), (300, 200))
    original = app.canvas.canvas.copy()
    app.handle_key(ord('n'))
    assert len(app.book.pages) == 2 and app.canvas.is_empty()
    app.handle_key(ord(','))
    assert np.array_equal(app.canvas.canvas, original)


def test_late_ocr_result_after_page_turn_uses_saved_page(app):
    page_id, digest = app.book.current.id, app.book.current.ink_hash()
    app.notebook_future = Future()
    app.notebook_job = page_id, digest
    app.handle_key(ord('n'))
    app.notebook_future.set_result(result('Words on the first page'))
    app.poll_notebook()
    assert app.book.pages[0].text == 'Words on the first page'
    assert app.book.current.text == ''
    assert app.notebook_future is None


def test_editor_does_not_treat_words_as_app_shortcuts(app):
    app.handle_key(ord('e'))
    for char in 'a quiet book':
        assert app.handle_key(ord(char)) is True
    assert app.book.current.text == 'a quiet book'
    assert app.ai_future is None
    app.handle_key(27)
    assert not app.notebook_editing
    assert (app.notebook_folder / 'notebook.json').exists()


def test_clear_ink_retains_text_for_comparison(app):
    app.book.current.text = 'Keep these words'
    app.canvas.draw_segment((200, 200), (300, 200))
    app.handle_key(ord('c'))
    assert app.canvas.is_empty()
    assert app.book.current.text == 'Keep these words'


def test_empty_tidy_is_explained_without_native_ocr(app, monkeypatch):
    monkeypatch.setattr('airboard.recognize_handwriting', lambda image: pytest.fail('should not run OCR'))
    app.trigger_tidy_text()
    assert app.notebook_future is None
    assert 'Write some words' in app.notebook_notice


def test_empty_ocr_result_does_not_clear_notes(app):
    app.book.current.text = 'Keep this'
    app.notebook_future = Future()
    app.notebook_job = app.book.current.id, app.book.current.ink_hash()
    app.notebook_future.set_result(result(''))
    app.poll_notebook()
    assert app.book.current.text == 'Keep this'


def test_book_view_renders_readable_unicode_and_empty_state(app):
    app.book.current.text = 'Café and my exact words'
    frame = app.canvas.render_view(np.full((720, 1280, 3), 235, np.uint8))
    rendered = notebook_view.render(app.book, frame, 'MOUSE DEMO')
    assert rendered.shape == (720, 1280, 3)
    assert np.any(rendered != frame)
