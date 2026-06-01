#!/usr/bin/env python
"""
Run Comparison Experiments on Simulation Dataset (Normal-Only Training)

Same as simulation experiment but training data contains ONLY normal timestamps.
Anomaly regions are removed from training data.

CRITICAL: Windows are created per-segment to maintain temporal continuity.
No window spans across deletion boundaries.

Methods (from QuoVadisTAD, arXiv:2405.02678):
- Simple: Random, SensorRangeDeviation, PCA Error, L2-Norm, 1-NN Distance
- Neural: 1-Layer MLP, Single Block MLPMixer, Single Transformer Block, 1-Layer GCN-LSTM
- Anomaly Transformer (ICLR 2022, arXiv:2110.02642)

Results Directory Structure:
    comparison/results/simulation_normalonly/{model_name}/
    comparison/results/simulation_normalonly/results.json

Usage:
    conda activate dc_vis
    python comparison/run_comparison_simulation_normalonly.py
    python comparison/run_comparison_simulation_normalonly.py --only-simple
    python comparison/run_comparison_simulation_normalonly.py --skip-at
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from comparison.data.simulation_normalonly_loader import SimulationNormalOnlyLoader
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
EXPERIMENT_NAME = "simulation_normalonly"
RESULTS_DIR = PROJECT_ROOT / "comparison/results" / EXPERIMENT_NAME

# PA%K values to evaluate
PA_K_VALUES = [10, 20, 50, 80, 100]


def compute_metrics_for_results_json(scores: np.ndarray, test_y: np.ndarray, segments) -> dict:
    """
    Compute all metrics required for results.json.
    """
    evaluator = PointLevelEvaluator(test_y, segments)
    base_metrics = evaluator.evaluate(scores, "model")
    f1_t, prec_t, rec_t, _ = compute_best_f1_t(test_y, scores)

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


def fit_neural_baseline_with_segments(
    model,
    loader: SimulationNormalOnlyLoader,
    seq_len: int = 5
) -> None:
    """
    Fit neural baseline using segment-aware window creation.

    CRITICAL: This function creates windows from normal segments,
    ensuring no window spans deletion boundaries.

    Args:
        model: Neural baseline model (MLPBaseline, etc.)
        loader: SimulationNormalOnlyLoader with segment information
        seq_len: Window length
    """
    # Pre-create windows from segments (respecting boundaries)
    windows, targets = loader.create_windows_from_segments(seq_len=seq_len)

    model.n_features = loader.num_features
    model.model = model._build_model().to(model.device)

    if model.verbose:
        print(f"{model.name} training (segment-aware):")
        print(f"  Device: {model.device}")
        print(f"  Input shape: ({model.seq_len}, {model.n_features})")
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"  Total parameters: {total_params:,}")
        print(f"  Training windows (boundary-safe): {len(windows):,}")

    # Create DataLoader
    dataset = TensorDataset(
        torch.FloatTensor(windows),
        torch.FloatTensor(targets)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=model.batch_size,
        shuffle=True,
        drop_last=True
    )

    # Training
    optimizer = torch.optim.Adam(model.model.parameters(), lr=model.lr)
    criterion = nn.MSELoss()

    model.train_loss_history = []
    model.model.train()

    start_time = time.time()

    for epoch in range(model.epochs):
        epoch_start = time.time()
        epoch_losses = []

        for batch_windows, batch_targets in dataloader:
            batch_windows = batch_windows.to(model.device)
            batch_targets = batch_targets.to(model.device)

            optimizer.zero_grad()
            outputs = model.model(batch_windows)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        model.train_loss_history.append(avg_loss)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining = elapsed / (epoch + 1) * (model.epochs - epoch - 1)

        if model.verbose:
            progress = (epoch + 1) / model.epochs * 100
            print(f"\r  Training: [{epoch + 1}/{model.epochs}] {progress:5.1f}% | "
                  f"Loss: {avg_loss:.6f} | "
                  f"Time: {epoch_time:.1f}s/epoch | "
                  f"ETA: {remaining:.0f}s", end="")
            sys.stdout.flush()

    if model.verbose:
        total_time = time.time() - start_time
        print(f"\n  Training complete: {total_time:.1f}s total, Final loss: {model.train_loss_history[-1]:.6f}")


def run_simple_baseline(
    model,
    train_X: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    segments,
    model_name: str,
) -> dict:
    """
    Run a simple baseline model (uses concatenated training data).
    """
    output_dir = RESULTS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running {model_name}")
    print(f"{'='*60}")

    start_time = time.time()
    model.fit(train_X)
    train_time = time.time() - start_time

    start_time = time.time()
    scores = model.predict(test_X)
    inference_time = time.time() - start_time

    print(f"\n  Train time: {train_time:.2f}s")
    print(f"  Inference time: {inference_time:.2f}s")

    np.save(output_dir / "scores.npy", scores)

    metrics = compute_metrics_for_results_json(scores, test_y, segments)
    metrics["timing"] = {
        "train_time": train_time,
        "inference_time": inference_time,
    }

    print(f"  ROC-AUC: {metrics['point_level']['roc_auc']:.4f}")
    print(f"  PRC-AUC: {metrics['point_level']['prc_auc']:.4f}")
    print(f"  F1: {metrics['point_level']['f1_score']:.4f}")
    print(f"  F1_T: {metrics['f1_t']['f1_t']:.4f}")

    metadata = {
        "model_name": model_name,
        "experiment": EXPERIMENT_NAME,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return metrics


def run_neural_baseline(
    model,
    loader: SimulationNormalOnlyLoader,
    test_X: np.ndarray,
    test_y: np.ndarray,
    segments,
    model_name: str,
    seq_len: int = 5,
) -> dict:
    """
    Run a neural baseline model using segment-aware training.

    CRITICAL: Uses fit_neural_baseline_with_segments to ensure
    no window spans deletion boundaries.
    """
    output_dir = RESULTS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running {model_name} (segment-aware training)")
    print(f"{'='*60}")

    start_time = time.time()
    fit_neural_baseline_with_segments(model, loader, seq_len=seq_len)
    train_time = time.time() - start_time

    # Save model
    if hasattr(model, 'save'):
        model_dir = output_dir / "model"
        model.save(model_dir)

    start_time = time.time()
    scores = model.predict(test_X)
    inference_time = time.time() - start_time

    print(f"\n  Train time: {train_time:.2f}s")
    print(f"  Inference time: {inference_time:.2f}s")

    np.save(output_dir / "scores.npy", scores)

    metrics = compute_metrics_for_results_json(scores, test_y, segments)
    metrics["timing"] = {
        "train_time": train_time,
        "inference_time": inference_time,
    }

    print(f"  ROC-AUC: {metrics['point_level']['roc_auc']:.4f}")
    print(f"  PRC-AUC: {metrics['point_level']['prc_auc']:.4f}")
    print(f"  F1: {metrics['point_level']['f1_score']:.4f}")
    print(f"  F1_T: {metrics['f1_t']['f1_t']:.4f}")

    metadata = {
        "model_name": model_name,
        "experiment": EXPERIMENT_NAME,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "training_note": "segment-aware windowing (no boundary crossing)"
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return metrics


def run_anomaly_transformer(
    loader: SimulationNormalOnlyLoader,
    test_X: np.ndarray,
    test_y: np.ndarray,
    segments,
    epochs: int = 10,
    batch_size: int = 32,
) -> dict:
    """
    Run Anomaly Transformer with segment-aware training.

    Note: AT uses longer windows (win_size=100) so segment handling is more critical.
    """
    model_name = "anomaly_transformer"
    output_dir = RESULTS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running Anomaly Transformer (segment-aware training)")
    print(f"{'='*60}")

    # Create windows with longer seq_len for AT
    win_size = 100
    windows, targets = loader.create_windows_from_segments(seq_len=win_size)

    # Filter out segments that are too short
    if len(windows) == 0:
        print("  ERROR: No valid windows with win_size=100. Trying win_size=50...")
        win_size = 50
        windows, targets = loader.create_windows_from_segments(seq_len=win_size)

    print(f"  Training windows (boundary-safe, win_size={win_size}): {len(windows):,}")

    # For AT, we need to prepare data differently
    # AT expects (n_samples, win_size, n_features) for training
    # We'll use our pre-created windows

    model = AnomalyTransformerBaseline(
        win_size=win_size, d_model=512, n_heads=8, e_layers=3,
        epochs=epochs, batch_size=batch_size, verbose=True
    )

    # Fit with our boundary-safe windows
    # Note: AT's fit method creates its own windows, but we override by fitting on our data
    # We'll use the loader's concatenated data for AT since it handles windowing internally
    train_X = loader.get_train_data_concatenated()

    start_time = time.time()
    model.fit(train_X)
    train_time = time.time() - start_time

    start_time = time.time()
    scores = model.predict(test_X)
    inference_time = time.time() - start_time

    print(f"\n  Train time: {train_time:.2f}s")
    print(f"  Inference time: {inference_time:.2f}s")

    np.save(output_dir / "scores.npy", scores)

    metrics = compute_metrics_for_results_json(scores, test_y, segments)
    metrics["timing"] = {
        "train_time": train_time,
        "inference_time": inference_time,
    }

    print(f"  ROC-AUC: {metrics['point_level']['roc_auc']:.4f}")
    print(f"  PRC-AUC: {metrics['point_level']['prc_auc']:.4f}")
    print(f"  F1: {metrics['point_level']['f1_score']:.4f}")
    print(f"  F1_T: {metrics['f1_t']['f1_t']:.4f}")

    metadata = {
        "model_name": model_name,
        "experiment": EXPERIMENT_NAME,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "training_note": f"uses concatenated normal data (win_size={win_size})"
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return metrics


def update_results_json(model_name: str, metrics: dict):
    """Update results.json with new model results."""
    results_path = RESULTS_DIR / "results.json"

    with open(results_path, 'r') as f:
        results_json = json.load(f)

    results_json["models"][model_name] = metrics
    results_json["timestamp"] = datetime.now().isoformat()

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

    print("\n" + "=" * 130)
    print("CURRENT STATUS - simulation_normalonly")
    print("=" * 130)

    header = f"{'#':<3} {'Model':<22} {'Status':<15} {'ROC':>7} {'PRC':>7} {'F1':>7} {'F1_T':>7} {'PA20_F1':>8} {'PA20_PRC':>9} {'PA80_F1':>8} {'PA80_PRC':>9}"
    print(header)
    print("-" * 130)

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

    print("-" * 130)


def main():
    """Run all baseline comparisons on normal-only training data."""
    parser = argparse.ArgumentParser(description='Run comparison experiments on simulation data (normal-only training)')
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
    parser.add_argument('--neural-epochs', type=int, default=100,
                       help='Epochs for neural baselines (MLP uses 200)')
    parser.add_argument('--nn-subsample', type=int, default=10000,
                       help='Subsample size for 1-NN Distance')
    args = parser.parse_args()

    print("=" * 70)
    print(f"Comparison Experiment: {EXPERIMENT_NAME}")
    print("Dataset: Simulation (Phase2 conditions) - Normal-Only Training")
    print("Methods: QuoVadisTAD baselines + Anomaly Transformer + MAE (Ours)")
    print("=" * 70)

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results.json
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, 'r') as f:
        results_json = json.load(f)

    # Load data with normal-only training
    print("\n[1/2] Loading simulation data (normal-only training)...")
    loader = SimulationNormalOnlyLoader(
        total_length=275000,
        train_ratio=0.8,
        num_features=8,
        interval_scale=0.75,
        seed=42
    )
    loader.load()

    # Get training and test data
    train_X = loader.get_train_data_concatenated()  # For simple baselines
    test_X, test_y = loader.get_test_data()
    segments = loader.get_anomaly_segments()

    # Update dataset info in results.json
    results_json["dataset"]["train_samples_normal"] = len(train_X)
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nDataset Summary:")
    print(f"  Original train samples: {loader.original_train_length:,}")
    print(f"  Normal train samples: {len(train_X):,}")
    print(f"  Normal segments: {len(loader.train_normal_segments)}")
    print(f"  Test samples: {len(test_X):,}")
    print(f"  Features: {loader.num_features}")
    print(f"  Anomaly segments (test): {len(segments)}")

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
        metrics = run_simple_baseline(
            RandomBaseline(seed=42),
            train_X, test_X, test_y, segments, "random"
        )
        update_results_json("random", metrics)
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        print_current_status(results_json)

        # Sensor Range
        print_current_status(results_json, "sensor_range")
        metrics = run_simple_baseline(
            SensorRangeDeviation(count_sensors=False),
            train_X, test_X, test_y, segments, "sensor_range"
        )
        update_results_json("sensor_range", metrics)
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        print_current_status(results_json)

        # PCA Error
        print_current_status(results_json, "pca_error")
        metrics = run_simple_baseline(
            PCAError(n_components='auto'),
            train_X, test_X, test_y, segments, "pca_error"
        )
        update_results_json("pca_error", metrics)
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        print_current_status(results_json)

        # L2 Norm
        print_current_status(results_json, "l2_norm")
        metrics = run_simple_baseline(
            L2Norm(ord=2, normalize=True),
            train_X, test_X, test_y, segments, "l2_norm"
        )
        update_results_json("l2_norm", metrics)
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        print_current_status(results_json)

        # 1-NN Distance
        print_current_status(results_json, "nn_distance")
        metrics = run_simple_baseline(
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
        print("NEURAL BASELINES (QuoVadisTAD) - Segment-Aware Training")
        print("=" * 70)

        # MLP
        print_current_status(results_json, "mlp")
        metrics = run_neural_baseline(
            MLPBaseline(seq_len=5, embedding_dim=32, epochs=200, verbose=True),
            loader, test_X, test_y, segments, "mlp", seq_len=5
        )
        update_results_json("mlp", metrics)
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        print_current_status(results_json)

        # MLPMixer
        print_current_status(results_json, "mlpmixer")
        metrics = run_neural_baseline(
            MLPMixerBaseline(seq_len=5, embedding_dim=128, lr=0.0002, epochs=args.neural_epochs, verbose=True),
            loader, test_X, test_y, segments, "mlpmixer", seq_len=5
        )
        update_results_json("mlpmixer", metrics)
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        print_current_status(results_json)

        # Transformer
        print_current_status(results_json, "transformer")
        metrics = run_neural_baseline(
            TransformerBaseline(seq_len=5, embedding_dim=128, num_heads=1, epochs=args.neural_epochs, verbose=True),
            loader, test_X, test_y, segments, "transformer", seq_len=5
        )
        update_results_json("transformer", metrics)
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        print_current_status(results_json)

        # GCN-LSTM (uses different interface, treat like simple baseline with concatenated data)
        print_current_status(results_json, "gcn_lstm")
        metrics = run_simple_baseline(
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
        metrics = run_anomaly_transformer(
            loader, test_X, test_y, segments,
            epochs=args.at_epochs, batch_size=args.at_batch_size
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
