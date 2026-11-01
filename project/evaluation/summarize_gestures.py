"""Summarize an observed trial CSV without mixing it with synthetic fixtures."""
from pathlib import Path
import argparse
import csv
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import write_gesture_outputs


def summarize_recorded_csv(path, output_dir='evaluation/observed_results'):
    with Path(path).open(newline='') as file:
        records = list(csv.DictReader(file))
    if not records:
        raise ValueError('No completed observations in this file')
    if any(row.get('source') != 'webcam_observation' for row in records):
        raise ValueError('Only webcam_observation rows belong in an observed-trial summary')
    return write_gesture_outputs(records, output_dir, 'webcam_observations', prefix='observed_gesture')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path')
    parser.add_argument('--output-dir', default='evaluation/observed_results')
    args = parser.parse_args()
    result = summarize_recorded_csv(args.csv_path, args.output_dir)
    print(f"Observed trials: {result['total_trials']}; accuracy: {result['overall_accuracy']:.1%}")


if __name__ == '__main__':
    main()
