"""Synthetic classifier checks. These do not measure webcam accuracy."""
from pathlib import Path
import sys
import numpy as np
from typing import List, Tuple
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.gestures import GestureClassifier
from evaluation.metrics import GESTURE_CLASSES, write_gesture_outputs


def generate_synthetic_hand_landmarks(
    gesture: str,
    distance_mode: str = "medium",  # "close", "medium", "far"
    lighting_mode: str = "bright",  # "bright", "dim"
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """
    Creates rough hand-shaped coordinate fixtures for classifier regression checks.
    Scale and noise labels are simulations, not measured distance or lighting.
    """
    rng = np.random.default_rng(seed)

    # Scale base on distance
    scale_map = {"close": 2.2, "medium": 1.4, "far": 0.75}
    scale = scale_map.get(distance_mode, 1.4)

    # Noise variance based on lighting
    noise_sigma = 1.0 if lighting_mode == "bright" else 3.8

    # Base palm center and wrist
    wrist = np.array([640.0, 520.0])

    # Finger configurations: [thumb, index, middle, ring, pinky]
    # In screen coordinates, negative dy is UP (extended)
    finger_extended = {
        "DRAW":  [False, True, False, False, False],
        "HOVER": [True, True, True, True, True],
        "ERASE": [False, True, True, False, False],
        "CLEAR": [False, False, False, False, False],
    }.get(gesture, [False, False, False, False, False])

    landmarks: List[Tuple[int, int]] = []
    # 0: Wrist
    w_noisy = wrist + rng.normal(0, noise_sigma, 2)
    landmarks.append((int(w_noisy[0]), int(w_noisy[1])))

    # Base x-offsets for MCP joints: thumb(-45), index(-25), middle(0), ring(25), pinky(45)
    mcp_offsets = [-45.0, -25.0, 0.0, 25.0, 45.0]

    # Generate 4 joints per finger
    for f_idx in range(5):
        ext = finger_extended[f_idx]
        base_x = mcp_offsets[f_idx] * scale

        if f_idx == 0:
            # Thumb geometry
            if ext:
                tip_offset = np.array([-70.0, -40.0]) * scale
                ip_offset = np.array([-50.0, -10.0]) * scale
            else:
                tip_offset = np.array([-25.0, -25.0]) * scale
                ip_offset = np.array([-35.0, -15.0]) * scale
            mcp = wrist + np.array([-30.0, -35.0]) * scale
            cmc = wrist + np.array([-15.0, -20.0]) * scale
            joints = [cmc, mcp, wrist + ip_offset, wrist + tip_offset]
        else:
            # Main fingers
            mcp_y = -70.0 * scale
            mcp = wrist + np.array([base_x, mcp_y])
            if ext:
                # Upwards extension
                pip = mcp + np.array([0.0, -35.0 * scale])
                dip = pip + np.array([0.0, -30.0 * scale])
                tip = dip + np.array([0.0, -25.0 * scale])
            else:
                # Curled downwards into palm
                pip = mcp + np.array([0.0, -15.0 * scale])
                dip = pip + np.array([0.0, 15.0 * scale])
                tip = dip + np.array([0.0, 20.0 * scale])
            joints = [mcp, pip, dip, tip]

        for pt in joints:
            p_noisy = pt + rng.normal(0, noise_sigma, 2)
            landmarks.append((int(p_noisy[0]), int(p_noisy[1])))

    return landmarks


def run_gesture_evaluation(output_dir='evaluation/results'):
    records = []
    conditions = [(light, distance) for light in ('bright', 'dim') for distance in ('close', 'medium', 'far')]
    for expected in GESTURE_CLASSES:
        for index in range(25):
            light, distance = conditions[index % len(conditions)]
            trial = len(records) + 1
            classifier = GestureClassifier(clear_hold_seconds=1.0)
            landmarks = generate_synthetic_hand_landmarks(expected, distance, light, 1000 + trial * 37)
            if expected == 'CLEAR':
                classifier.classify(landmarks, current_time=0.0)
                mode, _, _ = classifier.classify(landmarks, current_time=1.05)
            else:
                mode, _, _ = classifier.classify(landmarks, current_time=1.0)
            records.append({
                'trial': trial, 'ground_truth': expected, 'predicted': mode.value,
                'distance': distance, 'lighting': light, 'source': 'synthetic_landmarks',
            })
    summary = write_gesture_outputs(records, output_dir, 'synthetic_landmarks')
    summary['camera_used'] = False
    # Re-save after explicitly recording the scope of the simulation.
    import json
    (Path(output_dir) / 'gesture_accuracy_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f"Synthetic classifier checks: {summary['total_trials']} fixtures; not webcam measurements.")
    return summary


if __name__ == '__main__':
    run_gesture_evaluation()
