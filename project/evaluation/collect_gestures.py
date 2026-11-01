"""Record 25 observed webcam trials per gesture across six labeled conditions."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.gestures import GestureClassifier, GestureMode
from core.hand_tracking import HandTracker
from evaluation.metrics import GESTURE_CLASSES

POSES = {'DRAW':'index only', 'HOVER':'open hand', 'ERASE':'index and middle', 'CLEAR':'fist 1 second'}

FIELDS = ['trial', 'ground_truth', 'predicted', 'distance', 'lighting', 'source',
          'frames', 'detected_frames', 'hold_seconds', 'timestamp']


def trial_plan():
    plan = []
    for condition, (light, distance) in enumerate((light, distance) for light in ('bright', 'dim')
                                                  for distance in ('close', 'medium', 'far')):
        repetitions = 5 if condition == 0 else 4
        for expected in GESTURE_CLASSES:
            for _ in range(repetitions):
                plan.append({'ground_truth': expected, 'lighting': light, 'distance': distance})
    return plan


def observed_prediction(expected, modes):
    """Retain missed frames; CLEAR is a one-time event rather than a held mode."""
    if not modes:
        return 'NO_HAND'
    if 'CLEAR' in modes:
        return 'CLEAR'
    return Counter(modes).most_common(1)[0][0]


def main():
    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--output', default='evaluation/real_gesture_trials.csv')
    parser.add_argument('--hold-seconds', type=float, default=1.5)
    args = parser.parse_args()
    if args.hold_seconds < 1.1:
        parser.error('Use at least 1.1 seconds so CLEAR can complete its one-second hold')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f'{output} already exists. Use another --output path to preserve recorded observations.')
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        raise SystemExit('Camera unavailable. Check camera permissions and device index.')
    tracker = None
    try:
        tracker = HandTracker()
        plan = trial_plan()
        index = 0
        started = None
        classifier = GestureClassifier()
        modes = []
        detected = 0
        with output.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            file.flush()
            while index < len(plan):
                ok, frame = camera.read()
                if not ok:
                    print('Camera stopped. Completed trials have been saved.')
                    break
                frame = cv2.flip(frame, 1)
                tracker.find_hands(frame, draw=True)
                landmarks, _, _ = tracker.get_landmark_positions(frame)
                trial = plan[index]
                now = time.monotonic()
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 105), (20, 20, 20), -1)
                cv2.putText(frame, f"Trial {index + 1}/100  {trial['ground_truth']}: {POSES[trial['ground_truth']]}", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Arrange {trial['lighting']} light, {trial['distance']} distance", (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                if started is None:
                    instruction = 'SPACE starts a trial. Q stops and saves completed trials.'
                else:
                    mode, _, _ = classifier.classify(landmarks, current_time=now)
                    modes.append(mode.value if landmarks else 'NO_HAND')
                    detected += bool(landmarks)
                    elapsed = now - started
                    instruction = f'Recording: {max(0, args.hold_seconds - elapsed):.1f}s remaining'
                    if elapsed >= args.hold_seconds:
                        writer.writerow({
                            'trial': index + 1, **trial, 'predicted': observed_prediction(trial['ground_truth'], modes),
                            'source': 'webcam_observation', 'frames': len(modes), 'detected_frames': detected,
                            'hold_seconds': round(elapsed, 3), 'timestamp': datetime.now(timezone.utc).isoformat(),
                        })
                        file.flush()
                        index += 1
                        started = None
                cv2.putText(frame, instruction, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 240, 190), 1)
                cv2.imshow('airboard observed gesture trials', frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == 32 and started is None:
                    classifier = GestureClassifier()
                    modes, detected = [], 0
                    started = time.monotonic()
        print(f'Saved {index} completed webcam trials to {output}')
    finally:
        camera.release()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
