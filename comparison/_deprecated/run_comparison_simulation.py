#!/usr/bin/env python
"""
Run Comparison Experiments on Simulation Dataset

Runs baseline methods on simulation dataset matching phase2 conditions.

Methods (from QuoVadisTAD, arXiv:2405.02678):
- Simple: Random, SensorRangeDeviation, PCA Error, L2-Norm, 1-NN Distance
- Neural: 1-Layer MLP, Single Block MLPMixer, Single Transformer Block, 1-Layer GCN-LSTM
- Anomaly Transformer (ICLR 2022, arXiv:2110.02642)

Results Directory Structure:
    comparison/results/simulation/{model_name}/
    comparison/results/simulation/results.json

Usage:
    conda activate dc_vis
    python comparison/run_comparison_simulation.py
    python comparison/run_comparison_simulation.py --only-simple    # Only simple baselines
    python comparison/run_comparison_simulation.py --only-neural    # Only neural baselines
    python comparison/run_comparison_simulation.py --skip-at        # Skip Anomaly Transformer
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

from comparison.data.simulation_loader import SimulationLoader
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
)

# Experiment configuration
EXPERIMENT_NAME = "simulation"
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


def update_results_json(model_name: str, metrics: dict):
    """Update results.json with new model results."""
    results_path = RESULTS_DIR / "results.json"

    # Load existing results
    with open(results_path, 'r') as f:
        results_json = json.load(f)

    # Update
    results_json["models"][model_name] = metrics
    results_json["timestamp"] = datetime.now().isoformat()

    # Save
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)


def print_current_status(results_json: dict, current_model: str = None):
    """Print current experiment status as a table."""
    all_models = [
        "mae_adaptive",
        "random", "sensor_range", "pca_error", "l2_norm", "nn_distance",
        "mlp", "mlpmixer", "transformer", "gcn_lstm",
        "anomaly_transformer"
    ]

    print("\n" + "=" * 120)
    print("CURRENT STATUS")
    print("=" * 120)

    header = f"{'#':<3} {'Model':<22} {'Status':<15} {'ROC':>7} {'PRC':>7} {'F1':>7} {'F1_T':>7} {'PA20_F1':>8} {'PA20_PRC':>9} {'PA80_F1':>8} {'PA80_PRC':>9}"
    print(header)
    print("-" * 120)

    for i, model_name in enumerate(all_models, 1):
        if model_name in results_json.get("models", {}):
            m = results_json["models"][model_name]
            pl = m["point_level"]
            f1t = m["f1_t"]["f1_t"]
            pa20_f1 = m["pa_k"]["pa_20"]["f1_score"]
            pa20_prc = m["pa_k"]["pa_20"]["prc_auc"]
            pa80_f1 = m["pa_k"]["pa_80"]["f1_score"]
            pa80_prc = m["pa_k"]["pa_80"]["prc_auc"]
            status = "✅"
            print(f"{i:<3} {model_name:<22} {status:<15} {pl['roc_auc']:>7.3f} {pl['prc_auc']:>7.3f} {pl['f1_score']:>7.3f} {f1t:>7.3f} {pa20_f1:>8.3f} {pa20_prc:>9.3f} {pa80_f1:>8.3f} {pa80_prc:>9.3f}")
        elif model_name == current_model:
            status = "🔄 Running"
            print(f"{i:<3} {model_name:<22} {status:<15} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>8} {'-':>9} {'-':>8} {'-':>9}")
        else:
            status = "-"
            print(f"{i:<3} {model_name:<22} {status:<15} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>8} {'-':>9} {'-':>8} {'-':>9}")

    print("-" * 120)


def main():
    """Run all baseline comparisons."""
    parser = argparse.ArgumentParser(description='Run comparison experiments on simulation data')
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
                       help='Epochs for neural baselines (MLP uses 200)')
    parser.add_argument('--nn-subsample', type=int, default=10000,
                       help='Subsample size for 1-NN Distance')
    args = parser.parse_args()

    print("=" * 70)
    print(f"Comparison Experiment: {EXPERIMENT_NAME}")
    print("Dataset: Simulation (Phase2 conditions)")
    print("Methods: QuoVadisTAD baselines + Anomaly Transformer + MAE (Ours)")
    print("=" * 70)

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results.json
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, 'r') as f:
        results_json = json.load(f)

    # Load data
    print("\n[1/2] Loading simulation data...")
    loader = SimulationLoader(
        total_length=275000,
        train_ratio=0.8,
        num_features=8,
        interval_scale=0.75,
        seed=42
    )
    loader.load()

    train_X, train_y = loader.get_train_data()
    test_X, test_y = loader.get_test_data()
    segments = loader.get_anomaly_segments()

    print(f"\nDataset: Simulation")
    print(f"  Train samples: {len(train_X):,}")
    print(f"  Test samples: {len(test_X):,}")
    print(f"  Features: {train_X.shape[1]}")
    print(f"  Train:Test ratio: 80:20")
    print(f"  Anomaly segments: {len(segments)}")

    # Show initial status
    print_current_status(results_json)

    run_simple = not args.only_neural
    run_neural = not args.only_simple

    # ========== SIMPLE BASELINES ==========
    if run_simple:
        print("\n" + "=" * 70)
        print("SIMPLE BASELINES")
        print("=" * 70)

        # Random
        print_current_status(results_json, "random")
        metrics = run_baseline(
            RandomBaseline(seed=42),
            train_X, test_X, test_y, segments, "random"
        )
        update_results_json("random", metrics)
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        print_current_status(results_json)

        # Sensor Range
        print_current_status(results_json, "sensor_range")
        metrics = run_baseline(
            SensorRangeDeviation(count_sensors=False),
            train_X, test_X, test_y, segments, "sensor_range"
        )
        update_results_json("sensor_range", metrics)
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        print_current_status(results_json)

        # PCA Error
        print_current_status(results_json, "pca_error")
        metrics = run_baseline(
            PCAError(n_components='auto'),
            train_X, test_X, test_y, segments, "pca_error"
        )
        update_results_json("pca_error", metrics)
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        print_current_status(results_json)

        # L2 Norm
        print_current_status(results_json, "l2_norm")
        metrics = run_baseline(
            L2Norm(ord=2, normalize=True),
            train_X, test_X, test_y, segments, "l2_norm"
        )
        update_results_json("l2_norm", metrics)
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        print_current_status(results_json)

        # 1-NN Distance
        print_current_status(results_json, "nn_distance")
        metrics = run_baseline(
            NNDistance(distance='euclidean', subsample=args.nn_subsample),
            train_X, test_X, test_y, segments, "nn_distance"
        )
        update_results_json("nn_distance", metrics)
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        print_current_status(results_json)

    # ========== NEURAL BASELINES ==========
    if run_neural:
        print("\n" + "=" * 70)
        print("NEURAL BASELINES (QuoVadisTAD)")
        print("=" * 70)

        # MLP
        print_current_status(results_json, "mlp")
        metrics = run_baseline(
            MLPBaseline(seq_len=5, embedding_dim=32, epochs=args.neural_epochs, verbose=True),
            train_X, test_X, test_y, segments, "mlp"
        )
        update_results_json("mlp", metrics)
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        print_current_status(results_json)

        # MLPMixer
        print_current_status(results_json, "mlpmixer")
        metrics = run_baseline(
            MLPMixerBaseline(seq_len=5, embedding_dim=128, lr=0.0002, epochs=args.neural_epochs, verbose=True),
            train_X, test_X, test_y, segments, "mlpmixer"
        )
        update_results_json("mlpmixer", metrics)
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        print_current_status(results_json)

        # Transformer
        print_current_status(results_json, "transformer")
        metrics = run_baseline(
            TransformerBaseline(seq_len=5, embedding_dim=128, num_heads=1, epochs=args.neural_epochs, verbose=True),
            train_X, test_X, test_y, segments, "transformer"
        )
        update_results_json("transformer", metrics)
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        print_current_status(results_json)

        # GCN-LSTM
        print_current_status(results_json, "gcn_lstm")
        metrics = run_baseline(
            GCNLSTMBaseline(seq_len=5, gcn_out_dim=10, lstm_units=64, batch_size=100, epochs=args.neural_epochs, verbose=True),
            train_X, test_X, test_y, segments, "gcn_lstm"
        )
        update_results_json("gcn_lstm", metrics)
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        print_current_status(results_json)

    # ========== ANOMALY TRANSFORMER ==========
    if not args.skip_at:
        print("\n" + "=" * 70)
        print("ANOMALY TRANSFORMER (ICLR 2022)")
        print("=" * 70)

        print_current_status(results_json, "anomaly_transformer")
        metrics = run_baseline(
            AnomalyTransformerBaseline(
                win_size=100, d_model=512, n_heads=8, e_layers=3,
                epochs=args.at_epochs, batch_size=args.at_batch_size, verbose=True
            ),
            train_X, test_X, test_y, segments, "anomaly_transformer"
        )
        update_results_json("anomaly_transformer", metrics)
        with open(results_path, 'r') as f:
            results_json = json.load(f)

    # ========== FINAL SUMMARY ==========
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print_current_status(results_json)

    print(f"\nResults saved to: {results_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
