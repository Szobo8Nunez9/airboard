"""
Unit tests for core airboard modules: smoothing, gestures, and canvas.
"""

import pytest
import numpy as np
from core.smoothing import ExponentialSmoother
from core.gestures import GestureClassifier, GestureMode
from core.canvas import AirCanvas
from evaluation.test_gestures import generate_synthetic_hand_landmarks


def test_smoothing_bounds():
    with pytest.raises(ValueError):
        ExponentialSmoother(alpha=0.0)
    with pytest.raises(ValueError):
        ExponentialSmoother(alpha=1.5)


def test_smoothing_convergence():
    smoother = ExponentialSmoother(alpha=0.5)
    # First point seeds immediately
    pt1 = smoother.smooth(100, 100)
    assert pt1 == (100, 100)

    # Step to 200: 0.5 * 200 + 0.5 * 100 = 150
    pt2 = smoother.smooth(200, 200)
    assert pt2 == (150, 150)

    # Step to 200 again: 0.5 * 200 + 0.5 * 150 = 175
    pt3 = smoother.smooth(200, 200)
    assert pt3 == (175, 175)


def test_smoothing_reset():
    smoother = ExponentialSmoother(alpha=0.5)
    smoother.smooth(100, 100)
    smoother.smooth(200, 200)
    assert smoother.current is not None

    smoother.reset()
    assert smoother.current is None
    # Next point acts as new seed without rubber banding
    pt_new = smoother.smooth(500, 500)
    assert pt_new == (500, 500)


def test_gesture_classification_draw():
    classifier = GestureClassifier()
    lms = generate_synthetic_hand_landmarks("DRAW", "medium", "bright", seed=42)
    mode, progress, diag = classifier.classify(lms, current_time=1.0)
    assert mode == GestureMode.DRAW
    assert diag["fingers"]["index"] is True
    assert diag["fingers"]["middle"] is False


def test_gesture_classification_hover():
    classifier = GestureClassifier()
    lms = generate_synthetic_hand_landmarks("HOVER", "medium", "bright", seed=42)
    mode, progress, diag = classifier.classify(lms, current_time=1.0)
    assert mode == GestureMode.HOVER
    assert diag["fingers"]["index"] is True
    assert diag["fingers"]["middle"] is True
    assert diag["fingers"]["ring"] is True
    assert diag["fingers"]["pinky"] is True


def test_gesture_classification_erase():
    classifier = GestureClassifier()
    lms = generate_synthetic_hand_landmarks("ERASE", "medium", "bright", seed=42)
    mode, progress, diag = classifier.classify(lms, current_time=1.0)
    assert mode == GestureMode.ERASE
    assert diag["fingers"]["index"] is True
    assert diag["fingers"]["middle"] is True
    assert diag["fingers"]["ring"] is False
    assert diag["fingers"]["pinky"] is False


def test_gesture_clear_hold_timer():
    classifier = GestureClassifier(clear_hold_seconds=1.0)
    lms = generate_synthetic_hand_landmarks("CLEAR", "medium", "bright", seed=42)

    # At t=0.0: hold starts, progress 0.0
    mode, progress, _ = classifier.classify(lms, current_time=0.0)
    assert mode == GestureMode.HOVER  # Pending countdown
    assert progress == 0.0

    # At t=0.5: half way
    mode, progress, _ = classifier.classify(lms, current_time=0.5)
    assert 0.45 <= progress <= 0.55

    # At t=1.05: hold duration exceeded, triggers CLEAR
    mode, progress, _ = classifier.classify(lms, current_time=1.05)
    assert mode == GestureMode.CLEAR
    assert progress == 1.0


def test_canvas_draw_and_clear():
    canvas = AirCanvas(width=640, height=480)
    assert canvas.is_empty()

    canvas.draw_segment((100, 100), (200, 200))
    assert not canvas.is_empty()

    png_bytes = canvas.get_png_bytes()
    assert len(png_bytes) > 100
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    canvas.clear()
    assert canvas.is_empty()


def test_canvas_whiteboard_mode():
    canvas = AirCanvas(width=640, height=480)
    dummy_frame = np.full((480, 640, 3), 50, dtype=np.uint8)

    # Standard overlay view
    canvas.whiteboard_mode = False
    view1 = canvas.render_view(dummy_frame)
    assert view1.shape == (480, 640, 3)

    # Whiteboard view (white backdrop)
    canvas.whiteboard_mode = True
    view2 = canvas.render_view(dummy_frame)
    assert np.mean(view2) > 240  # predominantly pure white


def test_alpha_one_is_raw_even_with_adaptation():
    smoother = ExponentialSmoother(alpha=1.0, adaptive=True)
    samples = [(100, 100), (101, 99), (600, 400), (5, 25)]
    assert [smoother.smooth(*point) for point in samples] == samples
    smoother.set_alpha(0.5)
    smoother.smooth(800, 700)
    smoother.set_alpha(1.0)
    assert smoother.smooth(12, 14) == (12, 14)


def test_adaptive_smoothing_stays_finite_and_between_input_and_previous():
    smoother = ExponentialSmoother(alpha=0.35, adaptive=True)
    points = [(0, 0), (4, 5), (900, 600), (0, 0), (200, 150)]
    for x, y in points:
        previous = smoother.current
        current = smoother.smooth(x, y)
        assert all(np.isfinite(value) for value in current)
        if previous is not None:
            assert min(previous[0], x) <= current[0] <= max(previous[0], x)
            assert min(previous[1], y) <= current[1] <= max(previous[1], y)


def test_history_is_bounded_and_reset_reseeds_adaptive_filter():
    smoother = ExponentialSmoother(alpha=0.5, adaptive=True)
    for point in range(650):
        smoother.smooth(point, point)
    assert len(smoother.history) == 600
    smoother.reset()
    assert smoother.smooth(20, 40) == (20, 40)


def test_smoothing_rejects_nonfinite_coordinates():
    smoother = ExponentialSmoother(adaptive=True)
    with pytest.raises(ValueError, match='finite'):
        smoother.smooth(float('inf'), 0)
    with pytest.raises(ValueError, match='finite'):
        smoother.smooth(0, float('nan'))


def test_distance_extension_recognizes_rotated_fixture():
    classifier = GestureClassifier()
    landmarks = generate_synthetic_hand_landmarks('DRAW', seed=42)
    wrist = landmarks[0]
    # Rotate this fixture 90 degrees so a pure vertical comparison is insufficient.
    rotated = [(wrist[0] - (y - wrist[1]), wrist[1] + (x - wrist[0])) for x, y in landmarks]
    assert classifier.get_finger_states(rotated)['index'] is True


def test_draw_grace_is_limited_and_missing_hand_resets_it(monkeypatch):
    classifier = GestureClassifier()
    landmarks = generate_synthetic_hand_landmarks('DRAW', seed=42)
    assert classifier.draw_grace_counter == 0
    assert classifier.classify(landmarks, current_time=0.0)[0] == GestureMode.DRAW
    ambiguous = {'thumb': False, 'index': True, 'middle': False, 'ring': True, 'pinky': False}
    monkeypatch.setattr(classifier, 'get_finger_states', lambda _: ambiguous)
    assert classifier.classify(landmarks, current_time=0.1)[0] == GestureMode.DRAW
    assert classifier.classify(landmarks, current_time=0.2)[0] == GestureMode.DRAW
    assert classifier.classify(landmarks, current_time=0.3)[0] == GestureMode.IDLE
    # Establish grace again, then lose tracking before the ambiguous pose returns.
    classifier.last_mode = GestureMode.DRAW
    classifier.draw_grace_counter = 2
    assert classifier.classify([], current_time=0.4)[0] == GestureMode.IDLE
    assert classifier.draw_grace_counter == 0
    assert classifier.classify(landmarks, current_time=0.5)[0] == GestureMode.IDLE


@pytest.mark.parametrize('next_gesture, expected', [('HOVER', GestureMode.HOVER), ('ERASE', GestureMode.ERASE)])
def test_draw_grace_never_overrides_an_explicit_mode(next_gesture, expected):
    classifier = GestureClassifier()
    classifier.classify(generate_synthetic_hand_landmarks('DRAW', seed=42), current_time=0)
    mode, _, _ = classifier.classify(generate_synthetic_hand_landmarks(next_gesture, seed=42), current_time=0.1)
    assert mode == expected
    assert classifier.draw_grace_counter == 0


def test_hold_at_time_zero_has_correct_diagnostics_and_resets_on_missing_hand():
    classifier = GestureClassifier()
    fist = generate_synthetic_hand_landmarks('CLEAR', seed=42)
    classifier.classify(fist, current_time=0.0)
    _, _, diagnostic = classifier.classify(fist, current_time=0.5)
    assert diagnostic['fist_hold_elapsed'] == 0.5
    classifier.classify([], current_time=0.6)
    mode, progress, _ = classifier.classify(fist, current_time=1.5)
    assert mode == GestureMode.HOVER and progress == 0.0


def test_round_cap_stroke_is_continuous_and_png_preserves_color():
    import io
    from PIL import Image
    canvas = AirCanvas(width=140, height=80)
    canvas.draw_segment((20, 40), (120, 40))
    assert np.all(np.any(canvas.canvas[40, 20:121] > 10, axis=1))
    assert np.any(canvas.canvas[40, 17] > 10)
    assert np.any(canvas.canvas[40, 123] > 10)
    image = Image.open(io.BytesIO(canvas.get_png_bytes()))
    assert image.getpixel((70, 40)) == tuple(reversed(canvas.current_color))


def test_snapshot_write_failure_is_reported(monkeypatch, tmp_path):
    import cv2
    monkeypatch.setattr(cv2, 'imwrite', lambda *args, **kwargs: False)
    with pytest.raises(OSError, match='Could not save'):
        AirCanvas(20, 20).save_snapshot(str(tmp_path / 'board.png'))
