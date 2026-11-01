"""Application state tests use scripted input, never camera or GUI windows."""
from concurrent.futures import Future
from unittest.mock import Mock
import json

import cv2
import numpy as np
import pytest

from airboard import AirboardApp
from agents import AgentResult
from core.gestures import GestureMode
from evaluation.test_gestures import generate_synthetic_hand_landmarks


@pytest.fixture
def app(tmp_path):
    instance = AirboardApp(mouse=True)
    instance.snapshots_dir = tmp_path
    yield instance
    instance.executor.shutdown(wait=True, cancel_futures=True)


def test_tracking_loss_restarts_fist_hold_in_app(app):
    fist = generate_synthetic_hand_landmarks('CLEAR')
    assert app.process_landmarks(fist, 100.0)[0] == GestureMode.HOVER
    assert app.process_landmarks([], 100.5)[0] == GestureMode.IDLE
    assert app.process_landmarks(fist, 101.2)[0] == GestureMode.HOVER
    assert app.process_landmarks(fist, 102.25)[0] == GestureMode.CLEAR
    assert app.process_landmarks(fist, 102.5)[0] == GestureMode.IDLE


def test_stroke_does_not_bridge_a_tracking_gap(app):
    draw = generate_synthetic_hand_landmarks('DRAW')
    app.process_landmarks(draw, 0)
    first = app.canvas.prev_point
    app.process_landmarks([], 0.1)
    moved = [(x + 300, y) for x, y in draw]
    app.process_landmarks(moved, 0.2)
    assert app.canvas.prev_point == (first[0] + 300, first[1])
    assert not app.canvas.canvas[first[1], first[0] + 150].any()


def test_mouse_demo_draws_erases_and_clears(app):
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, 200, 250, 0)
    assert app.process_pointer(0)[0] == GestureMode.DRAW
    app.on_mouse(cv2.EVENT_MOUSEMOVE, 300, 250, 0)
    app.process_pointer(0.1)
    assert not app.canvas.is_empty()
    app.on_mouse(cv2.EVENT_LBUTTONUP, 300, 250, 0)
    app.on_mouse(cv2.EVENT_RBUTTONDOWN, 200, 250, 0)
    assert app.process_pointer(0.2)[0] == GestureMode.ERASE
    assert not app.canvas.canvas[250, 200].any()
    app.handle_key(ord('c'))
    assert app.canvas.is_empty()
    assert app.smoother.current is None


def test_ai_panel_shows_each_error(app, monkeypatch):
    app.ai_panel_visible = True
    app.ai_results = (AgentResult('Claude', '', 0, False, 'Claude timeout'),
                      AgentResult('ChatGPT', '', 0, False, 'Missing API key'), None)
    rendered = []
    monkeypatch.setattr(app, '_wrap_text', lambda image, text, *args: rendered.append(text))
    app.render_ai_panel(np.zeros((720, 1280, 3), np.uint8))
    assert 'Claude timeout' in rendered[0]
    assert 'Missing API key' in rendered[1]


def test_ai_poll_is_nonblocking_and_preserves_success(app):
    app.ai_loading = True
    app.ai_future = Future()
    app.poll_ai_results()
    assert app.ai_loading is True
    live = AgentResult('Claude', json.dumps({'final_answer': 'x=6'}), .1, True)
    failure = AgentResult('ChatGPT', '', .1, False, 'No API key')
    app.ai_future.set_result((live, failure, None))
    app.poll_ai_results()
    assert not app.ai_loading
    assert app.ai_results[0].success
    text = (app.snapshots_dir / 'last_ai_answers.txt').read_text()
    assert 'x=6' in text and 'No API key' in text


def test_ai_panel_pauses_drawing_and_timer(app):
    fist = generate_synthetic_hand_landmarks('CLEAR')
    app.process_landmarks(fist, 0)
    app.ai_panel_visible = True
    assert app.process_landmarks(fist, 2)[0] == GestureMode.IDLE
    app.ai_panel_visible = False
    assert app.process_landmarks(fist, 3)[0] == GestureMode.HOVER


def test_mouse_mode_does_not_open_camera(app, monkeypatch):
    open_camera = Mock(side_effect=AssertionError('camera must not open'))
    monkeypatch.setattr(cv2, 'VideoCapture', open_camera)
    assert app.start_camera() is False
    open_camera.assert_not_called()


def test_failed_snapshot_does_not_claim_success(app, monkeypatch):
    monkeypatch.setattr(cv2, 'imwrite', lambda *args: False)
    with pytest.raises(OSError):
        app.canvas.save_snapshot(str(app.snapshots_dir / 'bad.png'))


def test_unavailable_graphics_falls_back_to_mouse(app, monkeypatch):
    import airboard
    camera = Mock()
    camera.isOpened.return_value = True
    monkeypatch.setattr(cv2, 'VideoCapture', lambda index, backend: camera)
    monkeypatch.setattr(airboard, 'HandTracker', Mock(side_effect=RuntimeError('Graphics unavailable')))
    app.mouse_control = False
    assert app.start_camera() is False
    assert app.mouse_control is True
    assert app.cap is None
    camera.release.assert_called_once()


def test_failed_camera_switch_preserves_current_camera(app, monkeypatch):
    current = Mock()
    app.cap = current
    app.mouse_control = False
    monkeypatch.setattr(app, '_open_camera', lambda index: None)
    assert app.switch_camera() is False
    assert app.cap is current
    current.release.assert_not_called()


def test_successful_camera_switch_resets_stroke_and_timer(app, monkeypatch):
    current, candidate = Mock(), Mock()
    app.cap = current
    app.mouse_control = False
    app.tracker = Mock()
    app.canvas.prev_point = (10, 20)
    app.classifier.fist_start_time = 1.0
    monkeypatch.setattr(app, '_open_camera', lambda index: candidate)
    assert app.switch_camera() is True
    assert app.cap is candidate and app.camera_id == 1
    assert app.canvas.prev_point is None
    assert app.classifier.fist_start_time is None
    current.release.assert_called_once()


def test_smoothing_controls_reach_raw_alpha(app):
    for _ in range(20):
        app.handle_key(ord(']'))
    assert app.smoother.alpha == 1.0
    for _ in range(30):
        app.handle_key(ord('['))
    assert app.smoother.alpha == 0.1


def test_two_finger_eraser_tracks_between_fingertips(app):
    landmarks = generate_synthetic_hand_landmarks('ERASE')
    mode, _, tip = app.process_landmarks(landmarks, 0)
    assert mode == GestureMode.ERASE
    assert tip == ((landmarks[8][0]+landmarks[12][0])//2, (landmarks[8][1]+landmarks[12][1])//2)


def test_open_hand_hovers_without_erasing(app):
    landmarks = generate_synthetic_hand_landmarks('HOVER')
    app.canvas.draw_segment((100, 100), (200, 100))
    before = app.canvas.canvas.copy()
    mode, _, _ = app.process_landmarks(landmarks, 0)
    assert mode == GestureMode.HOVER
    assert np.array_equal(app.canvas.canvas, before)
