#!/usr/bin/env python
"""
Run Comparison Experiments on SWaT Dataset (A1A2 SWAP Normal-Only Training)

Same test data as run_swat_A1A2_swap.py, but training data has anomaly regions removed.
Baselines train only on normal data points.

    Train: A1 entire + BACK 50% of A2 (anomaly regions removed)
    Test: FRONT 50% of A2 (unchanged)

MAE results are reused from A1A2_swap (MAE trains with anomalies present).
Region #21 is in training set (back 50% of A2) → no excl-R#21 metrics.

Methods (from QuoVadisTAD, arXiv:2405.02678):
- Simple: Random, SensorRangeDeviation, PCA Error, L2-Norm, 1-NN Distance
- Neural: 1-Layer MLP, Single Block MLPMixer, Single Transformer Block, 1-Layer GCN-LSTM
- SOTA: Anomaly Transformer, TranAD, USAD, DAGMM, GDN, OmniAnomaly

Usage:
    conda activate dc_vis
    python comparison/run_swat_A1A2_swap_normalonly.py
    python comparison/run_swat_A1A2_swap_normalonly.py --only-simple
    python comparison/run_swat_A1A2_swap_normalonly.py --skip-sota
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

from comparison.data.swat_normalonly_loader import SWaTCombinedNormalOnlyLoader
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
    get_anomaly_segments,
)

# Try importing additional SOTA models
try:
    from comparison.baselines import TranADBaseline
    HAS_TRANAD = True
except ImportError:
    HAS_TRANAD = False

try:
    from comparison.baselines import USADBaseline
    HAS_USAD = True
except ImportError:
    HAS_USAD = False

try:
    from comparison.baselines import DAGMMBaseline
    HAS_DAGMM = True
except ImportError:
    HAS_DAGMM = False

try:
    from comparison.baselines import GDNBaseline
    HAS_GDN = True
except ImportError:
    HAS_GDN = False

try:
    from comparison.baselines import OmniAnomalyBaseline
    HAS_OMNIANOMALY = True
except ImportError:
    HAS_OMNIANOMALY = False

# Training stride (matches MAE ablation)
TRAIN_STRIDE = 3

# PA%K values to evaluate
PA_K_VALUES = [10, 20, 50, 80, 100]


def compute_metrics_for_results_json(scores: np.ndarray, test_y: np.ndarray, segments) -> dict:
    """Compute all metrics required for results.json."""
    evaluator = PointLevelEvaluator(test_y, segments)
    base_metrics = evaluator.evaluate(scores, "model")

    # Compute F1_T
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


def run_baseline(
    model,
    train_X: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    segments,
    model_name: str,
    results_dir: Path,
    save_model: bool = True,
) -> dict:
    """Run a baseline model and save results (full metrics only, no excl-R#21)."""
    output_dir = results_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running {model_name}")
    print(f"{'='*60}")

    # Fit
    start_time = time.time()
    model.fit(train_X)
    train_time = time.time() - start_time

    # Save model if supported
    if save_model and hasattr(model, 'save'):
        model_dir = output_dir / "model"
        model.save(model_dir)

    # Predict
    start_time = time.time()
    scores = model.predict(test_X)
    inference_time = time.time() - start_time

    print(f"\n  Train time: {train_time:.2f}s")
    print(f"  Inference time: {inference_time:.2f}s")

    # Save scores
    np.save(output_dir / "scores.npy", scores)

    # ===== Full metrics =====
    metrics = compute_metrics_for_results_json(scores, test_y, segments)
    metrics["timing"] = {
        "train_time": train_time,
        "inference_time": inference_time,
    }

    print(f"\n  ROC-AUC: {metrics['point_level']['roc_auc']:.4f}")
    print(f"  PRC-AUC: {metrics['point_level']['prc_auc']:.4f}")
    print(f"  F1: {metrics['point_level']['f1_score']:.4f}")
    print(f"  F1_T: {metrics['f1_t']['f1_t']:.4f}")

    # Save metadata
    metadata = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return metrics


def main():
    """Run all baseline comparisons on SWaT dataset (swap + normal-only training)."""
    parser = argparse.ArgumentParser(description='Run SWaT A1A2_swap normalonly comparison experiments')
    parser.add_argument('--only-simple', action='store_true',
                       help='Only run simple baselines')
    parser.add_argument('--only-neural', action='store_true',
                       help='Only run neural baselines')
    parser.add_argument('--skip-at', action='store_true',
                       help='Skip Anomaly Transformer')
    parser.add_argument('--skip-sota', action='store_true',
                       help='Skip all SOTA models (AT, TranAD, USAD, etc.)')
    parser.add_argument('--at-epochs', type=int, default=10,
                       help='Epochs for Anomaly Transformer')
    parser.add_argument('--neural-epochs', type=int, default=10,
                       help='Epochs for neural baselines')
    parser.add_argument('--sota-epochs', type=int, default=10,
                       help='Epochs for SOTA models')
    parser.add_argument('--nn-subsample', type=int, default=10000,
                       help='Subsample size for 1-NN Distance')
    args = parser.parse_args()

    EXPERIMENT_NAME = "SWaT_A1A2_swap_normalonly"
    RESULTS_DIR = PROJECT_ROOT / "comparison/results" / EXPERIMENT_NAME

    print("=" * 70)
    print(f"Comparison Experiment: {EXPERIMENT_NAME}")
    print("Dataset: SWaT A1 (normal) + A2 (attack) [SWAP + NORMAL-ONLY TRAINING]")
    print("Train: A1 entire + BACK 50% A2 (anomaly regions removed)")
    print("Test: FRONT 50% A2 (unchanged)")
    print("Note: R#21 in training set (no excl-R#21 metrics)")
    print("Methods: QuoVadisTAD baselines + SOTA")
    print("=" * 70)

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data with swap + normal-only training
    loader = SWaTCombinedNormalOnlyLoader(swap=True)
    loader.load()

    train_X, train_y = loader.get_train_data()
    test_X, test_y = loader.get_test_data()
    segments = loader.get_anomaly_segments()
    dataset_info = loader.get_dataset_info()

    print(f"\nDataset: SWaT A1+A2 [SWAP + NORMAL-ONLY]")
    print(f"  Train samples (normal-only): {len(train_X):,} "
          f"(removed {dataset_info['train_anomaly_removed']:,} anomaly points)")
    print(f"  Train samples (full): {dataset_info['train_length_full']:,}")
    print(f"  Test samples: {len(test_X):,} (anomaly ratio: {test_y.mean():.2%})")
    print(f"  Features: {loader.n_features}")
    print(f"  Normal segments: {dataset_info['normal_segments']}")
    print(f"  Anomaly segments (test): {len(segments)}")
    print(f"  R#21 in test: {dataset_info['r21_in_test']}")

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
            train_X, test_X, test_y, segments, "random", RESULTS_DIR
        )

        all_results["sensor_range"] = run_baseline(
            SensorRangeDeviation(count_sensors=False),
            train_X, test_X, test_y, segments, "sensor_range", RESULTS_DIR
        )

        all_results["pca_error"] = run_baseline(
            PCAError(n_components='auto'),
            train_X, test_X, test_y, segments, "pca_error", RESULTS_DIR
        )

        all_results["l2_norm"] = run_baseline(
            L2Norm(ord=2, normalize=True),
            train_X, test_X, test_y, segments, "l2_norm", RESULTS_DIR
        )

        all_results["nn_distance"] = run_baseline(
            NNDistance(distance='euclidean', subsample=args.nn_subsample),
            train_X, test_X, test_y, segments, "nn_distance", RESULTS_DIR
        )

    # ========== NEURAL BASELINES ==========
    if run_neural:
        print("\n" + "=" * 70)
        print("NEURAL BASELINES (QuoVadisTAD)")
        print("=" * 70)

        all_results["mlp"] = run_baseline(
            MLPBaseline(seq_len=5, embedding_dim=32, epochs=args.neural_epochs,
                        train_stride=TRAIN_STRIDE, verbose=True),
            train_X, test_X, test_y, segments, "mlp", RESULTS_DIR
        )

        all_results["mlpmixer"] = run_baseline(
            MLPMixerBaseline(seq_len=5, embedding_dim=128, lr=0.0002,
                            epochs=args.neural_epochs, train_stride=TRAIN_STRIDE,
                            verbose=True),
            train_X, test_X, test_y, segments, "mlpmixer", RESULTS_DIR
        )

        all_results["transformer"] = run_baseline(
            TransformerBaseline(seq_len=5, embedding_dim=128, num_heads=1,
                               epochs=args.neural_epochs, train_stride=TRAIN_STRIDE,
                               verbose=True),
            train_X, test_X, test_y, segments, "transformer", RESULTS_DIR
        )

        all_results["gcn_lstm"] = run_baseline(
            GCNLSTMBaseline(seq_len=5, gcn_out_dim=10, lstm_units=64,
                           batch_size=100, epochs=args.neural_epochs,
                           train_stride=TRAIN_STRIDE, verbose=True),
            train_X, test_X, test_y, segments, "gcn_lstm", RESULTS_DIR
        )

    # ========== SOTA MODELS ==========
    if not args.skip_sota and run_neural:
        print("\n" + "=" * 70)
        print("SOTA MODELS")
        print("=" * 70)

        # Anomaly Transformer
        if not args.skip_at:
            all_results["anomaly_transformer"] = run_baseline(
                AnomalyTransformerBaseline(
                    win_size=100, d_model=512, n_heads=8, e_layers=3,
                    epochs=args.at_epochs, batch_size=32,
                    train_stride=TRAIN_STRIDE, verbose=True
                ),
                train_X, test_X, test_y, segments, "anomaly_transformer",
                RESULTS_DIR
            )

        # TranAD
        if HAS_TRANAD:
            all_results["tranad"] = run_baseline(
                TranADBaseline(seq_len=10, epochs=args.sota_epochs,
                              train_stride=TRAIN_STRIDE, verbose=True),
                train_X, test_X, test_y, segments, "tranad", RESULTS_DIR
            )

        # USAD
        if HAS_USAD:
            all_results["usad"] = run_baseline(
                USADBaseline(seq_len=5, epochs=args.sota_epochs,
                            train_stride=TRAIN_STRIDE, verbose=True),
                train_X, test_X, test_y, segments, "usad", RESULTS_DIR
            )

        # DAGMM
        if HAS_DAGMM:
            all_results["dagmm"] = run_baseline(
                DAGMMBaseline(seq_len=5, epochs=args.sota_epochs,
                             train_stride=TRAIN_STRIDE, verbose=True),
                train_X, test_X, test_y, segments, "dagmm", RESULTS_DIR
            )

        # GDN
        if HAS_GDN:
            all_results["gdn"] = run_baseline(
                GDNBaseline(seq_len=5, epochs=args.sota_epochs,
                           train_stride=TRAIN_STRIDE, verbose=True),
                train_X, test_X, test_y, segments, "gdn", RESULTS_DIR
            )

        # OmniAnomaly
        if HAS_OMNIANOMALY:
            all_results["omnianomaly"] = run_baseline(
                OmniAnomalyBaseline(seq_len=100, epochs=args.sota_epochs,
                                   train_stride=TRAIN_STRIDE, verbose=True),
                train_X, test_X, test_y, segments, "omnianomaly", RESULTS_DIR
            )

    # ========== SAVE RESULTS.JSON ==========
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results_json = {
        "experiment": EXPERIMENT_NAME,
        "dataset": {
            "name": "SWaT A1+A2 [SWAP + Normal-Only Training]",
            "split": "Train=A1+back50%A2 (normal only), Test=front50%A2",
            "n_a1_samples": dataset_info['n_a1_samples'],
            "n_a2_samples": dataset_info['n_a2_samples'],
            "train_samples_full": dataset_info['train_length_full'],
            "train_samples_normal": len(train_X),
            "train_anomaly_removed": dataset_info['train_anomaly_removed'],
            "normal_segments": dataset_info['normal_segments'],
            "test_samples": len(test_X),
            "features": loader.n_features,
            "test_anomaly_ratio": float(test_y.mean()),
            "anomaly_segments": len(segments),
            "r21_in_test": False,
        },
        "timestamp": datetime.now().isoformat(),
        "models": all_results,
    }

    results_path = RESULTS_DIR / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved: {results_path}")

    # ========== SUMMARY TABLE ==========
    print("\n" + "=" * 90)
    print("SUMMARY: Full Metrics (A1A2_swap_normalonly)")
    print("=" * 90)

    print(f"\n{'Model':<25} {'ROC-AUC':>10} {'PRC-AUC':>10} {'F1':>10} {'F1_T':>10}")
    print("-" * 70)

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
