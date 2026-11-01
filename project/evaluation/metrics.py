"""Confusion matrices retain missed or unrecognized predictions as failures."""
from pathlib import Path
import csv
import json

GESTURE_CLASSES = ['DRAW', 'HOVER', 'ERASE', 'CLEAR']
PREDICTION_CLASSES = GESTURE_CLASSES + ['IDLE', 'NO_HAND']


def summarize_trials(records, data_source):
    matrix = {label: {pred: 0 for pred in PREDICTION_CLASSES} for label in GESTURE_CLASSES}
    for record in records:
        expected, predicted = record['ground_truth'], record['predicted']
        if expected not in GESTURE_CLASSES or predicted not in PREDICTION_CLASSES:
            raise ValueError(f'Unknown trial label: {expected!r}, {predicted!r}')
        matrix[expected][predicted] += 1
    metrics = {}
    correct = sum(matrix[label][label] for label in GESTURE_CLASSES)
    for label in GESTURE_CLASSES:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted_count = sum(matrix[actual][label] for actual in GESTURE_CLASSES)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        metrics[label] = {
            'precision': round(precision, 4), 'recall': round(recall, 4),
            'f1_score': round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
            'support': support,
        }
    return {
        'data_source': data_source, 'total_trials': len(records),
        'overall_accuracy': correct / len(records) if records else None,
        'confusion_matrix': matrix, 'class_metrics': metrics,
        'prediction_classes': PREDICTION_CLASSES,
    }


def write_gesture_outputs(records, output_dir, data_source, prefix='gesture'):
    import matplotlib.pyplot as plt
    import numpy as np
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_trials(records, data_source)
    csv_path = output / f'{prefix}_trials.csv'
    fieldnames = ['trial', 'ground_truth', 'predicted', 'distance', 'lighting', 'source',
                  'frames', 'detected_frames', 'hold_seconds', 'timestamp']
    with csv_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key, '') for key in fieldnames})
    matrix_path = output / f'{prefix}_confusion_matrix.csv'
    with matrix_path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['GroundTruth'] + PREDICTION_CLASSES)
        for label in GESTURE_CLASSES:
            writer.writerow([label] + [summary['confusion_matrix'][label][pred] for pred in PREDICTION_CLASSES])
    values = np.array([[summary['confusion_matrix'][a][p] for p in PREDICTION_CLASSES] for a in GESTURE_CLASSES])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(values, cmap='Blues', vmin=0)
    ax.set_xticks(range(len(PREDICTION_CLASSES)), PREDICTION_CLASSES)
    ax.set_yticks(range(len(GESTURE_CLASSES)), GESTURE_CLASSES)
    ax.set_xlabel('Predicted mode, including missed detections')
    ax.set_ylabel('Requested gesture')
    label = 'Synthetic landmark simulation' if data_source == 'synthetic_landmarks' else 'Recorded webcam trials'
    ax.set_title(label)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, str(values[i, j]), ha='center', va='center', color='black')
    fig.tight_layout()
    chart_path = output / f'{prefix}_confusion_matrix.png'
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    summary.update({'trials_csv': csv_path.as_posix(), 'csv_path': matrix_path.as_posix(), 'chart_path': chart_path.as_posix()})
    (output / f'{prefix}_accuracy_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary
