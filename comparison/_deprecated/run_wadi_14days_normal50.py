#!/usr/bin/env python
"""
Run Comparison Experiments on WaDi 14days Normal50 Dataset

50% of training anomaly regions removed (known anomalies excluded).
Remaining 50% relabeled as normal (unknown contamination).
Test data unchanged.

Usage:
    conda activate dc_vis
    python comparison/run_wadi_14days_normal50.py --scenario A1
    python comparison/run_wadi_14days_normal50.py --scenario A2
    python comparison/run_wadi_14days_normal50.py --scenario A1 --only-simple
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from comparison.data.wadi_14days_normal50_loader import WaDi14daysNormal50Loader
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

PA_K_VALUES = [10, 20, 50, 80, 100]


def compute_metrics_for_results_json(scores: np.ndarray, test_y: np.ndarray, segments) -> dict:
    """Compute all metrics required for results.json."""
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


def run_baseline(model, train_X, test_X, test_y, segments, model_name, results_dir) -> dict:
    """Run a baseline model and save results."""
    output_dir = results_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running {model_name}")
    print(f"{'='*60}")

    start_time = time.time()
    model.fit(train_X)
    train_time = time.time() - start_time

    if hasattr(model, 'save'):
        model_dir = output_dir / "model"
        try:
            model.save(model_dir)
        except Exception:
            pass

    start_time = time.time()
    scores = model.predict(test_X)
    inference_time = time.time() - start_time

    print(f"  Train time: {train_time:.2f}s")
    print(f"  Inference time: {inference_time:.2f}s")

    np.save(output_dir / "scores.npy", scores)

    metrics = compute_metrics_for_results_json(scores, test_y, segments)
    metrics["timing"] = {"train_time": train_time, "inference_time": inference_time}

    print(f"  ROC-AUC: {metrics['point_level']['roc_auc']:.4f}")
    print(f"  PRC-AUC: {metrics['point_level']['prc_auc']:.4f}")
    print(f"  F1: {metrics['point_level']['f1_score']:.4f}")
    print(f"  F1_T: {metrics['f1_t']['f1_t']:.4f}")

    metadata = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run WaDi 14days Normal50 baselines')
    parser.add_argument('--scenario', type=str, default='A1', choices=['A1', 'A2'])
    parser.add_argument('--only-simple', action='store_true')
    parser.add_argument('--only-neural', action='store_true')
    parser.add_argument('--skip-at', action='store_true')
    parser.add_argument('--skip-sota', action='store_true')
    parser.add_argument('--neural-epochs', type=int, default=10)
    parser.add_argument('--sota-epochs', type=int, default=10)
    parser.add_argument('--nn-subsample', type=int, default=10000)
    args = parser.parse_args()

    EXPERIMENT_NAME = f"WaDi_{args.scenario}_14days_normal50"
    RESULTS_DIR = PROJECT_ROOT / "comparison/results" / EXPERIMENT_NAME

    print("=" * 70)
    print(f"Comparison Experiment: {EXPERIMENT_NAME}")
    print(f"Dataset: WaDi {args.scenario} 14days with 50% training anomaly label noise")
    print("Baselines train on data with known anomalies removed")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data in baseline mode (remove known 50%)
    loader = WaDi14daysNormal50Loader(scenario=args.scenario, remove_unlabeled=True)
    loader.load()

    train_X, train_y = loader.get_train_data()
    test_X, test_y = loader.get_test_data()
    segments = loader.get_anomaly_segments()
    dataset_info = loader.get_dataset_info()

    print(f"\nDataset: WaDi {args.scenario} 14days Normal50 (Baseline mode)")
    print(f"  Train samples: {len(train_X):,} (anomaly ratio: {train_y.mean():.2%})")
    print(f"  Test samples: {len(test_X):,} (anomaly ratio: {test_y.mean():.2%})")
    print(f"  Features: {train_X.shape[1]}")
    print(f"  Anomaly segments (test): {len(segments)}")

    all_results = {}
    run_simple = not args.only_neural
    run_neural = not args.only_simple

    # ========== SIMPLE BASELINES ==========
    if run_simple:
        print("\n" + "=" * 70)
        print("SIMPLE BASELINES")
        print("=" * 70)

        all_results["random"] = run_baseline(
            RandomBaseline(seed=42), train_X, test_X, test_y, segments, "random", RESULTS_DIR)
        all_results["sensor_range"] = run_baseline(
            SensorRangeDeviation(count_sensors=False), train_X, test_X, test_y, segments, "sensor_range", RESULTS_DIR)
        all_results["pca_error"] = run_baseline(
            PCAError(n_components='auto'), train_X, test_X, test_y, segments, "pca_error", RESULTS_DIR)
        all_results["l2_norm"] = run_baseline(
            L2Norm(ord=2, normalize=True), train_X, test_X, test_y, segments, "l2_norm", RESULTS_DIR)
        all_results["nn_distance"] = run_baseline(
            NNDistance(distance='euclidean', subsample=args.nn_subsample),
            train_X, test_X, test_y, segments, "nn_distance", RESULTS_DIR)

    # ========== NEURAL BASELINES ==========
    if run_neural:
        print("\n" + "=" * 70)
        print("NEURAL BASELINES")
        print("=" * 70)

        all_results["mlp"] = run_baseline(
            MLPBaseline(seq_len=5, embedding_dim=32, epochs=args.neural_epochs, verbose=True),
            train_X, test_X, test_y, segments, "mlp", RESULTS_DIR)
        all_results["mlpmixer"] = run_baseline(
            MLPMixerBaseline(seq_len=5, embedding_dim=128, lr=0.0002, epochs=args.neural_epochs, verbose=True),
            train_X, test_X, test_y, segments, "mlpmixer", RESULTS_DIR)
        all_results["transformer"] = run_baseline(
            TransformerBaseline(seq_len=5, embedding_dim=128, num_heads=1, epochs=args.neural_epochs, verbose=True),
            train_X, test_X, test_y, segments, "transformer", RESULTS_DIR)
        all_results["gcn_lstm"] = run_baseline(
            GCNLSTMBaseline(seq_len=5, gcn_out_dim=10, lstm_units=64, batch_size=100,
                           epochs=args.neural_epochs, verbose=True),
            train_X, test_X, test_y, segments, "gcn_lstm", RESULTS_DIR)

    # ========== SOTA MODELS ==========
    if not args.skip_sota and run_neural:
        print("\n" + "=" * 70)
        print("SOTA MODELS")
        print("=" * 70)

        if not args.skip_at:
            all_results["anomaly_transformer"] = run_baseline(
                AnomalyTransformerBaseline(win_size=100, d_model=512, n_heads=8, e_layers=3,
                                          epochs=args.sota_epochs, batch_size=32, verbose=True),
                train_X, test_X, test_y, segments, "anomaly_transformer", RESULTS_DIR)

        if HAS_TRANAD:
            all_results["tranad"] = run_baseline(
                TranADBaseline(seq_len=10, epochs=args.sota_epochs, verbose=True),
                train_X, test_X, test_y, segments, "tranad", RESULTS_DIR)
        if HAS_USAD:
            all_results["usad"] = run_baseline(
                USADBaseline(seq_len=5, epochs=args.sota_epochs, verbose=True),
                train_X, test_X, test_y, segments, "usad", RESULTS_DIR)
        if HAS_DAGMM:
            all_results["dagmm"] = run_baseline(
                DAGMMBaseline(seq_len=5, epochs=args.sota_epochs, verbose=True),
                train_X, test_X, test_y, segments, "dagmm", RESULTS_DIR)
        if HAS_GDN:
            all_results["gdn"] = run_baseline(
                GDNBaseline(seq_len=5, epochs=args.sota_epochs, verbose=True),
                train_X, test_X, test_y, segments, "gdn", RESULTS_DIR)
        if HAS_OMNIANOMALY:
            all_results["omnianomaly"] = run_baseline(
                OmniAnomalyBaseline(seq_len=100, epochs=args.sota_epochs, verbose=True),
                train_X, test_X, test_y, segments, "omnianomaly", RESULTS_DIR)

    # ========== SAVE RESULTS.JSON ==========
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results_json = {
        "experiment": EXPERIMENT_NAME,
        "dataset": {
            "name": f"WaDi {args.scenario} 14days Normal50",
            "scenario": args.scenario,
            "description": "50% of training anomaly regions removed, 50% relabeled as normal",
            "n_14days_samples": dataset_info.get('n_14days_samples', 0),
            "n_attack_samples": dataset_info.get('n_attack_samples', 0),
            "train_samples": len(train_X),
            "test_samples": len(test_X),
            "features": train_X.shape[1],
            "train_anomaly_ratio": float(train_y.mean()),
            "test_anomaly_ratio": float(test_y.mean()),
            "anomaly_segments": len(segments),
            "noise_seed": 123,
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

    sorted_models = sorted(all_results.items(),
                          key=lambda x: x[1]['point_level']['prc_auc'], reverse=True)

    for model_name, metrics in sorted_models:
        pl = metrics['point_level']
        f1t = metrics['f1_t']['f1_t']
        print(f"{model_name:<25} {pl['roc_auc']:>10.4f} {pl['prc_auc']:>10.4f} "
              f"{pl['f1_score']:>10.4f} {f1t:>10.4f}")

    print("-" * 70)
    print(f"\nTo add MAE results, run:")
    print(f"  python comparison/add_mae_wadi_14days_normal50_results.py --scenario {args.scenario}")


if __name__ == "__main__":
    main()
