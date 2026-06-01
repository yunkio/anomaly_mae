#!/usr/bin/env python
"""
Run Comparison Experiments

Runs baseline methods on WaDi A1 dataset and saves results.

Methods (from QuoVadisTAD, arXiv:2405.02678):
- Simple: Random, SensorRangeDeviation, PCA Error, L2-Norm, 1-NN Distance
- Neural: 1-Layer MLP, Single Block MLPMixer, Single Transformer Block, 1-Layer GCN-LSTM
- Anomaly Transformer (ICLR 2022, arXiv:2110.02642)

Results Directory Structure:
    comparison/results/{experiment_name}/{model_name}/
    comparison/results/{experiment_name}/results.json

Usage:
    conda activate dc_vis
    python comparison/run_comparison.py
    python comparison/run_comparison.py --only-simple    # Only simple baselines
    python comparison/run_comparison.py --only-neural    # Only neural baselines
    python comparison/run_comparison.py --skip-at        # Skip Anomaly Transformer

To add MAE model results separately:
    python comparison/add_mae_results.py --experiment WaDi_A1 --mae-dir <dir> --name <name>
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from comparison.data.wadi_loader import WaDiLoader
from comparison.baselines import (
    RandomBaseline,
    SensorRangeDeviation,
    PCAError,
    L2Norm,
    NNDistance,
    MLPBaseline,
    MLPMixerBaseline,
    TransformerBaseline,
    GCNLSTMBaseline,
    AnomalyTransformerBaseline,
    PointLevelEvaluator,
    compute_best_f1_t,
    save_results,
    get_anomaly_segments,
)

# Experiment configuration
EXPERIMENT_NAME = "WaDi_A1"
RESULTS_DIR = PROJECT_ROOT / "comparison/results" / EXPERIMENT_NAME

# PA%K values to evaluate
PA_K_VALUES = [10, 20, 50, 80, 100]


def compute_metrics_for_results_json(scores: np.ndarray, test_y: np.ndarray, segments) -> dict:
    """
    Compute all metrics required for results.json.

    Metrics:
    - Point-level: PRC_AUC, ROC_AUC, F1, recall, precision
    - F1_T (time-series F1)
    - PA%K for K in [10, 20, 50, 80, 100]: PRC_AUC, ROC_AUC, F1, recall, precision
    """
    evaluator = PointLevelEvaluator(test_y, segments)

    # Get base metrics
    base_metrics = evaluator.evaluate(scores, "model")

    # Compute F1_T
    f1_t, prec_t, rec_t, _ = compute_best_f1_t(test_y, scores)

    # Build results dict
    result = {
        "point_level": {
            "prc_auc": base_metrics["prc_auc"],
            "roc_auc": base_metrics["roc_auc"],
            "f1_score": base_metrics["f1_score"],
            "recall": base_metrics["recall"],
            "precision": base_metrics["precision"],
        },
        "f1_t": {
            "f1_t": f1_t,
            "precision_t": prec_t,
            "recall_t": rec_t,
        },
        "pa_k": {}
    }

    # PA%K metrics
    for k in PA_K_VALUES:
        pa_key = f"pa_{k}"
        result["pa_k"][pa_key] = {
            "prc_auc": base_metrics.get(f"pa_{k}_prc_auc", 0.0),
            "roc_auc": base_metrics.get(f"pa_{k}_roc_auc", 0.0),
            "f1_score": base_metrics.get(f"pa_{k}_f1_score", 0.0),
            "recall": base_metrics.get(f"pa_{k}_recall", 0.0),
            "precision": base_metrics.get(f"pa_{k}_precision", 0.0),
        }

    return result


def run_baseline(
    model,
    train_X: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    segments,
    model_name: str,
    save_model: bool = True,
) -> dict:
    """
    Run a baseline model and save results.

    Args:
        model: Baseline model instance
        train_X: Training features
        test_X: Test features
        test_y: Test labels
        segments: Anomaly segments
        model_name: Name of the model (used for directory)
        save_model: Whether to save trained model weights

    Returns:
        Metrics dictionary for results.json
    """
    output_dir = RESULTS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running {model_name}")
    print(f"{'='*60}")

    # Fit and predict
    start_time = time.time()
    model.fit(train_X)
    train_time = time.time() - start_time

    # Save model if it has save method (neural baselines)
    if save_model and hasattr(model, 'save'):
        model_dir = output_dir / "model"
        model.save(model_dir)

    start_time = time.time()
    scores = model.predict(test_X)
    inference_time = time.time() - start_time

    print(f"\n  Train time: {train_time:.2f}s")
    print(f"  Inference time: {inference_time:.2f}s")

    # Save scores for later analysis
    np.save(output_dir / "scores.npy", scores)

    # Compute metrics
    metrics = compute_metrics_for_results_json(scores, test_y, segments)
    metrics["timing"] = {
        "train_time": train_time,
        "inference_time": inference_time,
    }

    # Print summary
    print(f"  ROC-AUC: {metrics['point_level']['roc_auc']:.4f}")
    print(f"  PRC-AUC: {metrics['point_level']['prc_auc']:.4f}")
    print(f"  F1: {metrics['point_level']['f1_score']:.4f}")
    print(f"  F1_T: {metrics['f1_t']['f1_t']:.4f}")

    # Save individual model metadata
    metadata = {
        "model_name": model_name,
        "experiment": EXPERIMENT_NAME,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return metrics


def main():
    """Run all baseline comparisons."""
    parser = argparse.ArgumentParser(description='Run comparison experiments')
    parser.add_argument('--only-simple', action='store_true',
                       help='Only run simple baselines')
    parser.add_argument('--only-neural', action='store_true',
                       help='Only run neural baselines')
    parser.add_argument('--skip-at', action='store_true',
                       help='Skip Anomaly Transformer')
    parser.add_argument('--at-epochs', type=int, default=10,
                       help='Epochs for Anomaly Transformer')
    parser.add_argument('--at-batch-size', type=int, default=32,
                       help='Batch size for Anomaly Transformer')
    parser.add_argument('--neural-epochs', type=int, default=10,
                       help='Epochs for neural baselines')
    parser.add_argument('--nn-subsample', type=int, default=10000,
                       help='Subsample size for 1-NN Distance')
    args = parser.parse_args()

    print("=" * 70)
    print(f"Comparison Experiment: {EXPERIMENT_NAME}")
    print("Methods: QuoVadisTAD baselines + Anomaly Transformer + MAE (Ours)")
    print("=" * 70)

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = PROJECT_ROOT / "dataset/WaDi/WADI.A1_9 Oct 2017/WADI_attackdata_preprocessed.csv"
    loader = WaDiLoader(data_path=str(data_path), train_ratio=0.5)
    loader.load()

    train_X, train_y = loader.get_train_data()
    test_X, test_y = loader.get_test_data()
    segments = loader.get_anomaly_segments()

    print(f"\nDataset: WaDi A1")
    print(f"  Train samples: {len(train_X):,}")
    print(f"  Test samples: {len(test_X):,}")
    print(f"  Features: {train_X.shape[1]}")
    print(f"  Train:Test ratio: 50:50")
    print(f"  Anomaly segments: {len(segments)}")

    # Results storage
    all_results = {}

    run_simple = not args.only_neural
    run_neural = not args.only_simple

    # ========== SIMPLE BASELINES ==========
    if run_simple:
        print("\n" + "=" * 70)
        print("SIMPLE BASELINES")
        print("=" * 70)

        all_results["random"] = run_baseline(
            RandomBaseline(seed=42),
            train_X, test_X, test_y, segments, "random"
        )

        all_results["sensor_range"] = run_baseline(
            SensorRangeDeviation(count_sensors=False),
            train_X, test_X, test_y, segments, "sensor_range"
        )

        all_results["pca_error"] = run_baseline(
            PCAError(n_components='auto'),
            train_X, test_X, test_y, segments, "pca_error"
        )

        all_results["l2_norm"] = run_baseline(
            L2Norm(ord=2, normalize=True),
            train_X, test_X, test_y, segments, "l2_norm"
        )

        all_results["nn_distance"] = run_baseline(
            NNDistance(distance='euclidean', subsample=args.nn_subsample),
            train_X, test_X, test_y, segments, "nn_distance"
        )

    # ========== NEURAL BASELINES ==========
    if run_neural:
        print("\n" + "=" * 70)
        print("NEURAL BASELINES (QuoVadisTAD)")
        print("=" * 70)

        all_results["mlp"] = run_baseline(
            MLPBaseline(seq_len=5, embedding_dim=32, epochs=args.neural_epochs, verbose=True),
            train_X, test_X, test_y, segments, "mlp"
        )

        all_results["mlpmixer"] = run_baseline(
            MLPMixerBaseline(seq_len=5, embedding_dim=128, lr=0.0002, epochs=args.neural_epochs, verbose=True),
            train_X, test_X, test_y, segments, "mlpmixer"
        )

        all_results["transformer"] = run_baseline(
            TransformerBaseline(seq_len=5, embedding_dim=128, num_heads=1, epochs=args.neural_epochs, verbose=True),
            train_X, test_X, test_y, segments, "transformer"
        )

        all_results["gcn_lstm"] = run_baseline(
            GCNLSTMBaseline(seq_len=5, gcn_out_dim=10, lstm_units=64, batch_size=100, epochs=args.neural_epochs, verbose=True),
            train_X, test_X, test_y, segments, "gcn_lstm"
        )

    # ========== ANOMALY TRANSFORMER ==========
    if not args.skip_at:
        print("\n" + "=" * 70)
        print("ANOMALY TRANSFORMER (ICLR 2022)")
        print("=" * 70)

        all_results["anomaly_transformer"] = run_baseline(
            AnomalyTransformerBaseline(
                win_size=100, d_model=512, n_heads=8, e_layers=3,
                epochs=args.at_epochs, batch_size=args.at_batch_size, verbose=True
            ),
            train_X, test_X, test_y, segments, "anomaly_transformer"
        )

    # ========== SAVE RESULTS.JSON ==========
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results_json = {
        "experiment": EXPERIMENT_NAME,
        "dataset": {
            "name": "WaDi A1",
            "train_samples": len(train_X),
            "test_samples": len(test_X),
            "features": train_X.shape[1],
            "train_test_ratio": "50:50",
            "anomaly_segments": len(segments),
        },
        "timestamp": datetime.now().isoformat(),
        "models": all_results,
    }

    results_path = RESULTS_DIR / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved: {results_path}")

    # ========== SUMMARY TABLE ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<25} {'ROC-AUC':>10} {'PRC-AUC':>10} {'F1':>10} {'F1_T':>10}")
    print("-" * 70)

    # Sort by PRC-AUC descending
    sorted_models = sorted(all_results.items(),
                          key=lambda x: x[1]['point_level']['prc_auc'],
                          reverse=True)

    for model_name, metrics in sorted_models:
        pl = metrics['point_level']
        f1t = metrics['f1_t']['f1_t']
        print(f"{model_name:<25} {pl['roc_auc']:>10.4f} {pl['prc_auc']:>10.4f} "
              f"{pl['f1_score']:>10.4f} {f1t:>10.4f}")

    print("-" * 70)
    print("\nDone!")


if __name__ == "__main__":
    main()
