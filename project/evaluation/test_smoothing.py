"""
Synthetic smoothing simulation: compares generated coordinate sequences (ALPHA = 1.0)
against Exponential Moving Average filtering (ALPHA = 0.5).
Reports jitter and geometric error on simulated input, plus theoretical filter lag.
No camera frame rate or user latency is measured.
"""

import json
import os
import sys
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.smoothing import ExponentialSmoother


def generate_benchmark_trajectories() -> Dict[str, np.ndarray]:
    """Generates ground truth geometric stroke trajectories."""
    # 1. Straight horizontal line (N=120)
    t = np.linspace(200, 1000, 120)
    line = np.column_stack([t, np.full_like(t, 360)])

    # 2. Smooth circular arc (N=120)
    theta = np.linspace(0, 2 * np.pi, 120)
    circle = np.column_stack([640 + 180 * np.cos(theta), 360 + 180 * np.sin(theta)])

    # 3. Corner turn (N=120)
    leg1 = np.column_stack([np.linspace(300, 640, 60), np.full(60, 480)])
    leg2 = np.column_stack([np.full(60, 640), np.linspace(480, 200, 60)])
    corner = np.vstack([leg1, leg2])

    return {"line": line, "circle": circle, "corner": corner}


def add_physiological_hand_tremor(trajectory: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Adds a chosen synthetic perturbation:
    Combines a sinusoid and Gaussian coordinate noise; this is not a measured physiological model.
    """
    np.random.seed(seed)
    n = len(trajectory)
    t = np.linspace(0, 4.0, n)  # 4 seconds duration
    # 10 Hz sinusoidal tremor
    tremor_x = 3.5 * np.sin(2 * np.pi * 9.8 * t) + np.random.normal(0, 2.2, n)
    tremor_y = 3.5 * np.cos(2 * np.pi * 10.2 * t) + np.random.normal(0, 2.2, n)
    noisy_traj = trajectory.copy()
    noisy_traj[:, 0] += tremor_x
    noisy_traj[:, 1] += tremor_y
    return noisy_traj


def compute_jitter_variance(points: np.ndarray) -> float:
    """
    Calculates variance of second differences: Var(P_{t+1} - 2P_t + P_{t-1}).
    Quantifies perceived jitter and stroke jaggedness.
    """
    if len(points) < 3:
        return 0.0
    second_diff = points[2:] - 2 * points[1:-1] + points[:-2]
    magnitudes = np.linalg.norm(second_diff, axis=1)
    return float(np.var(magnitudes))


def compute_path_length(points: np.ndarray) -> float:
    """Calculates total Euclidean stroke path length."""
    if len(points) < 2:
        return 0.0
    diffs = points[1:] - points[:-1]
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def run_smoothing_evaluation(output_dir: str = "evaluation/results") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    trajectories = generate_benchmark_trajectories()

    results_by_shape = {}
    sample_rows = []
    aggregate_raw_jitter = []
    aggregate_smooth_jitter = []

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))

    for idx, (shape_name, gt_path) in enumerate(trajectories.items()):
        raw_path = add_physiological_hand_tremor(gt_path, seed=100 + idx * 23)

        # Run ALPHA = 1.0 (Raw / No smoothing)
        smoother_raw = ExponentialSmoother(alpha=1.0)
        filtered_raw = np.array([smoother_raw.smooth(p[0], p[1]) for p in raw_path])

        # Run ALPHA = 0.5 (EMA Filtered)
        smoother_ema = ExponentialSmoother(alpha=0.5)
        filtered_ema = np.array([smoother_ema.smooth(p[0], p[1]) for p in raw_path])

        for frame, (gt, raw, ema) in enumerate(zip(gt_path, filtered_raw, filtered_ema)):
            sample_rows.append({'trajectory': shape_name, 'frame': frame, 'ground_truth_x': gt[0], 'ground_truth_y': gt[1],
                                'raw_x': raw[0], 'raw_y': raw[1], 'ema_x': ema[0], 'ema_y': ema[1]})
        gt_len = compute_path_length(gt_path)
        raw_len = compute_path_length(filtered_raw)
        ema_len = compute_path_length(filtered_ema)

        raw_jitter = compute_jitter_variance(filtered_raw)
        ema_jitter = compute_jitter_variance(filtered_ema)
        jitter_reduction_pct = ((raw_jitter - ema_jitter) / max(1e-4, raw_jitter)) * 100.0

        aggregate_raw_jitter.append(raw_jitter)
        aggregate_smooth_jitter.append(ema_jitter)

        results_by_shape[shape_name] = {
            "ground_truth_length_px": round(gt_len, 2),
            "raw_rmse_to_ground_truth_px": round(float(np.sqrt(np.mean(np.sum((filtered_raw - gt_path)**2, axis=1)))), 3),
            "ema_rmse_to_ground_truth_px": round(float(np.sqrt(np.mean(np.sum((filtered_ema - gt_path)**2, axis=1)))), 3),
            "raw_length_px": round(raw_len, 2),
            "ema_length_px": round(ema_len, 2),
            "raw_jitter_variance": round(raw_jitter, 3),
            "ema_jitter_variance": round(ema_jitter, 3),
            "jitter_reduction_percent": round(jitter_reduction_pct, 1),
        }

        # Plot 2D trajectories
        ax_traj = axes[idx, 0]
        ax_traj.plot(gt_path[:, 0], gt_path[:, 1], "k--", label="Ground Truth", alpha=0.5, linewidth=2)
        ax_traj.plot(filtered_raw[:, 0], filtered_raw[:, 1], "r-", label="Raw (ALPHA=1.0)", alpha=0.6, linewidth=1.5)
        ax_traj.plot(filtered_ema[:, 0], filtered_ema[:, 1], "b-", label="Smoothed (ALPHA=0.5)", linewidth=2.5)
        ax_traj.set_title(f"Stroke Trajectory: {shape_name.capitalize()}", fontweight="bold", fontsize=11)
        ax_traj.set_xlabel("X (px)")
        ax_traj.set_ylabel("Y (px)")
        ax_traj.legend(loc="upper right", fontsize=9)
        ax_traj.grid(True, linestyle=":", alpha=0.6)

        # Plot Acceleration / Jitter Curves
        ax_jit = axes[idx, 1]
        raw_sec = np.linalg.norm(filtered_raw[2:] - 2 * filtered_raw[1:-1] + filtered_raw[:-2], axis=1)
        ema_sec = np.linalg.norm(filtered_ema[2:] - 2 * filtered_ema[1:-1] + filtered_ema[:-2], axis=1)
        ax_jit.plot(raw_sec, "r-", label=f"Raw Jitter (Var: {raw_jitter:.1f})", alpha=0.6)
        ax_jit.plot(ema_sec, "b-", label=f"Smoothed Jitter (Var: {ema_jitter:.1f})", linewidth=2)
        ax_jit.set_title(f"Jitter Acceleration Profile: {jitter_reduction_pct:.1f}% Noise Reduction", fontweight="bold", fontsize=11)
        ax_jit.set_xlabel("Frame Step (t)")
        ax_jit.set_ylabel("||Δ² P_t|| (px/frame²)")
        ax_jit.legend(loc="upper right", fontsize=9)
        ax_jit.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Synthetic smoothing simulation: ALPHA 1.0 vs ALPHA 0.5", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "smoothing_comparison.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()

    mean_raw_j = float(np.mean(aggregate_raw_jitter))
    mean_smooth_j = float(np.mean(aggregate_smooth_jitter))
    overall_reduction = ((mean_raw_j - mean_smooth_j) / mean_raw_j) * 100.0

    import csv
    with open(os.path.join(output_dir, 'smoothing_samples.csv'), 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    summary = {
        "data_source": "synthetic_trajectories",
        "camera_used": False,
        "sample_count": len(sample_rows),
        "alpha_raw": 1.0,
        "alpha_ema": 0.5,
        "theoretical_low_frequency_lag_frames": 1.0,
        "lag_note": "EMA (1-alpha)/alpha at low frequencies; not measured webcam latency",
        "jitter_metric": "Variance of the second-difference magnitude, in squared coordinate units per frame to the fourth power",
        "mean_raw_jitter_variance": round(mean_raw_j, 3),
        "mean_smoothed_jitter_variance": round(mean_smooth_j, 3),
        "overall_jitter_reduction_percent": round(overall_reduction, 1),
        "trajectories": results_by_shape,
        "chart_path": chart_path,
    }

    json_path = os.path.join(output_dir, "smoothing_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[test_smoothing] Completed synthetic smoothing simulation; no webcam used.")
    print(f"[test_smoothing] Synthetic jitter metric reduced by {overall_reduction:.1f}%. Artifacts saved in '{output_dir}'.")
    return summary


if __name__ == "__main__":
    run_smoothing_evaluation()
