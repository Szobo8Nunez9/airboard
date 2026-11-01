"""Reproduce offline simulations and explicitly mark live AI quality unmeasured."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.test_gestures import run_gesture_evaluation
from evaluation.test_smoothing import run_smoothing_evaluation
from evaluation.test_ai_boards import run_ai_boards_benchmark


def run_evaluation(output_dir='evaluation/results', *, live=False):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gesture = run_gesture_evaluation(output)
    smoothing = run_smoothing_evaluation(str(output))
    ai = run_ai_boards_benchmark(output, live=live)
    result = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'scope': 'Synthetic software checks; observed webcam accuracy and live AI quality require separate measurements',
        'gesture_evaluation': gesture, 'smoothing_evaluation': smoothing, 'ai_benchmark': ai,
    }
    (output / 'evaluation_master.json').write_text(json.dumps(result, indent=2) + '\n')
    percent = lambda value: f'{value:.1%}' if value is not None else 'Not measured'
    lines = [
        '# airboard evaluation evidence', '',
        '## Synthetic gesture classifier checks', '',
        f"{gesture['total_trials']} generated landmark fixtures. Fixture accuracy: {percent(gesture['overall_accuracy'])}.",
        'The classifier receives the coordinates directly. No webcam or hand detector participates.',
        'Lighting and distance labels select noise and scale in a simulation, not real environmental conditions.',
        'IDLE and NO_HAND remain separate prediction outcomes. Raw trial rows and a confusion matrix are included.', '',
        '## Synthetic smoothing comparison', '',
        f"The same {smoothing['sample_count']} generated samples were replayed at alpha 1.0 and 0.5.",
        f"Reduction in the chosen second-difference jitter variance: {smoothing['overall_jitter_reduction_percent']:.1f}%.",
        'This number describes the generated trajectories only. It is not a measured improvement in human handwriting.',
        'Geometric RMSE records the tradeoff between smoothing and following the intended trajectory.',
        'The filter has a theoretical low-frequency delay of one sample at alpha 0.5. End-to-end camera latency is not measured.', '',
        '| Synthetic path | Raw RMSE px | EMA RMSE px |', '| --- | ---: | ---: |',
    ]
    for shape, metrics in smoothing['trajectories'].items():
        lines.append(f"| {shape} | {metrics['raw_rmse_to_ground_truth_px']} | {metrics['ema_rmse_to_ground_truth_px']} |")
    lines += ['', '## Optional AI board evaluation', '',
              f"Status: {ai['status']}. {ai['reason']}",
              'Model correctness is not automatically inferred from incidental numbers or matching model answers.',
              f"Claude accuracy: {percent(ai['claude_alone_accuracy'])}. ChatGPT accuracy: {percent(ai['chatgpt_alone_accuracy'])}.",
              'The generated sample set contains 5 equations, 5 sketches, and 5 messy boards. Human grading instructions are in evaluation/GRADING.md.', '',
              '## Remaining checks', '',
              'Run the real webcam trial logger, assess the resulting conditions, verify a live provider request if using the AI feature, and rehearse the classroom demo.']
    (output / 'evaluation_report.md').write_text('\n'.join(lines) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='evaluation/results')
    parser.add_argument('--live', action='store_true', help='Explicitly request live model replies for manual grading')
    args = parser.parse_args()
    result = run_evaluation(args.output_dir, live=args.live)
    print(f"Saved evidence to {args.output_dir}. AI status: {result['ai_benchmark']['status']}")


if __name__ == '__main__':
    main()
