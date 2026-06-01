"""
Calculate mae_teacheronly metrics from existing best_model data.
Teacher-only uses only reconstruction_loss as anomaly score.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, f1_score, roc_curve
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.evaluator import (
    PointLevelEvaluator,
    compute_best_f1_t,
    get_anomaly_segments,
    find_f1_optimal_threshold,
    compute_pa_k_roc_auc,
    compute_pa_k_adjusted_predictions,
)


def main():
    # Load detailed results
    csv_path = Path("/home/ykio/notebooks/claude/results/experiments/20260131_225102_phase2/000_best_model/best_model_detailed.csv")
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Extract teacher reconstruction error (this is the anomaly score for teacher-only)
    scores = df['reconstruction_loss'].values.astype(np.float32)
    labels = df['label'].values.astype(np.int32)

    print(f"  Total samples: {len(df)}")
    print(f"  Anomaly samples: {labels.sum()}")
    print(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")

    # Get anomaly segments
    print("\nExtracting anomaly segments...")
    segments = get_anomaly_segments(labels)
    print(f"  Found {len(segments)} anomaly segments")

    # Initialize evaluator
    evaluator = PointLevelEvaluator(labels, segments)

    # Calculate all metrics (this is the primary method)
    print("\nCalculating all metrics...")
    metrics = evaluator.evaluate(scores, "mae_teacheronly")

    # Extract point-level metrics
    point_level = {
        "prc_auc": metrics['prc_auc'],
        "roc_auc": metrics['roc_auc'],
        "f1_score": metrics['f1_score'],
        "recall": metrics['recall'],
        "precision": metrics['precision'],
    }

    # Extract F1_T
    f1_t_metrics = {
        "f1_t": metrics['f1_t'],
        "precision_t": metrics['precision_t'],
        "recall_t": metrics['recall_t'],
    }

    # Extract PA%K metrics
    pa_k_metrics = {}
    for k in [10, 20, 50, 80, 100]:
        pa_k_metrics[f"pa_{k}"] = {
            "prc_auc": metrics[f'pa_{k}_prc_auc'],
            "roc_auc": metrics[f'pa_{k}_roc_auc'],
            "f1_score": metrics[f'pa_{k}_f1_score'],
            "recall": 0.0,  # Not computed separately
            "precision": 0.0,  # Not computed separately
        }

    # Build result structure (matching the format in results.json)
    result = {
        "point_level": point_level,
        "f1_t": f1_t_metrics,
        "pa_k": pa_k_metrics,
        "timing": {
            "train_time": 0,  # Pre-trained model
            "inference_time": 527.2,  # Same as adaptive (same model)
        },
        "source": "results/experiments/20260131_225102_phase2/000_best_model",
        "note": "Teacher reconstruction error only (no discrepancy)"
    }

    # Print as JSON
    print("\n" + "="*60)
    print("mae_teacheronly result structure:")
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    result = main()
