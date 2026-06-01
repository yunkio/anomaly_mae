#!/usr/bin/env python
"""
Run baseline comparison experiments on simulation_complex dataset (complexity=True).

Matches the data used in results/base/simulation/simulation/ (MAE base experiment).

Usage:
    conda activate dc_vis
    python comparison/run_comparison_simulation_complex.py --model all
    python comparison/run_comparison_simulation_complex.py --model random
    python comparison/run_comparison_simulation_complex.py --list
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
import torch

from comparison.data.simulation_complex_loader import SimulationComplexLoader
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
    TranADBaseline,
    USADBaseline,
    DAGMMBaseline,
    GDNBaseline,
    OmniAnomalyBaseline,
    PointLevelEvaluator,
    compute_best_f1_t,
)

PA_K_VALUES = [10, 20, 50, 80, 100]
EXPERIMENT_NAME = "simulation_complex"
RESULTS_DIR = PROJECT_ROOT / "comparison/results" / EXPERIMENT_NAME

ALL_MODELS = [
    'random', 'sensor_range', 'pca_error', 'l2_norm', 'nn_distance',
    'mlp', 'mlpmixer', 'transformer', 'gcn_lstm',
    'anomaly_transformer', 'dagmm', 'usad', 'gdn', 'tranad', 'omnianomaly',
]


def compute_metrics_for_results_json(scores: np.ndarray, test_y: np.ndarray, segments) -> dict:
    evaluator = PointLevelEvaluator(test_y, segments)
    print("\nEvaluating model...")
    base_metrics = evaluator.evaluate(scores, "model")
    f1_t, prec_t, rec_t, _ = compute_best_f1_t(test_y, scores)

    print(f"  ROC-AUC: {base_metrics['roc_auc']:.4f}")
    print(f"  PRC-AUC: {base_metrics['prc_auc']:.4f}")
    print(f"  F1: {base_metrics['f1_score']:.4f} (P={base_metrics['precision']:.4f}, R={base_metrics['recall']:.4f})")
    print(f"  F1_T: {f1_t:.4f} (P_T={prec_t:.4f}, R_T={rec_t:.4f})")
    print(f"  PA%20 ROC-AUC: {base_metrics.get('pa_20_roc_auc', 0):.4f}")
    print(f"  PA%20 F1: {base_metrics.get('pa_20_f1_score', 0):.4f}")

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


def run_baseline(model, train_X, test_X, test_y, segments, model_name, results_dir):
    output_dir = results_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running {model_name}")
    print(f"{'='*60}")
    print(f"  Train shape: {train_X.shape}")
    print(f"  Test shape: {test_X.shape}")

    start_time = time.time()
    model.fit(train_X)
    train_time = time.time() - start_time
    print(f"\n  Train time: {train_time:.2f}s")

    if hasattr(model, 'save'):
        model_dir = output_dir / "model"
        try:
            model.save(model_dir)
        except Exception as e:
            print(f"  Warning: Could not save model: {e}")

    start_time = time.time()
    scores = model.predict(test_X)
    inference_time = time.time() - start_time
    print(f"  Inference time: {inference_time:.2f}s")

    np.save(output_dir / "scores.npy", scores)

    metrics = compute_metrics_for_results_json(scores, test_y, segments)
    metrics["timing"] = {"train_time": train_time, "inference_time": inference_time}

    metadata = {
        "model_name": model_name,
        "experiment": EXPERIMENT_NAME,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return metrics


def update_results_json(model_name, metrics):
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, 'r') as f:
        results_json = json.load(f)
    results_json["models"][model_name] = metrics
    results_json["timestamp"] = datetime.now().isoformat()
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  Updated: {results_path}")


def print_status():
    results_path = RESULTS_DIR / "results.json"
    if not results_path.exists():
        print("No results.json found")
        return

    with open(results_path, 'r') as f:
        results_json = json.load(f)

    print(f"\n{'='*130}")
    print(f"CURRENT STATUS - {EXPERIMENT_NAME}")
    print(f"{'='*130}")

    header = f"{'#':<3} {'Model':<22} {'Status':<12} {'ROC':>7} {'PRC':>7} {'F1':>7} {'F1_T':>7} {'PA20_F1':>8} {'PA20_PRC':>9} {'PA80_F1':>8} {'PA80_PRC':>9}"
    print(header)
    print("-" * 130)

    for i, model_name in enumerate(ALL_MODELS, 1):
        if model_name in results_json.get("models", {}):
            m = results_json["models"][model_name]
            pl = m["point_level"]
            f1t = m["f1_t"]["f1_t"]
            pa20_f1 = m["pa_k"]["pa_20"]["f1_score"]
            pa20_prc = m["pa_k"]["pa_20"]["prc_auc"]
            pa80_f1 = m["pa_k"]["pa_80"]["f1_score"]
            pa80_prc = m["pa_k"]["pa_80"]["prc_auc"]
            print(f"{i:<3} {model_name:<22} {'OK':<12} {pl['roc_auc']:>7.3f} {pl['prc_auc']:>7.3f} {pl['f1_score']:>7.3f} {f1t:>7.3f} {pa20_f1:>8.3f} {pa20_prc:>9.3f} {pa80_f1:>8.3f} {pa80_prc:>9.3f}")
        else:
            print(f"{i:<3} {model_name:<22} {'-':<12} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>8} {'-':>9} {'-':>8} {'-':>9}")

    print("-" * 130)


def get_model(model_name, n_features=8):
    if model_name == 'random':
        return RandomBaseline()
    elif model_name == 'sensor_range':
        return SensorRangeDeviation()
    elif model_name == 'pca_error':
        return PCAError()
    elif model_name == 'l2_norm':
        return L2Norm()
    elif model_name == 'nn_distance':
        return NNDistance()
    elif model_name == 'mlp':
        return MLPBaseline(seq_len=100, embedding_dim=128, epochs=10, batch_size=256, verbose=True)
    elif model_name == 'mlpmixer':
        return MLPMixerBaseline(seq_len=100, embedding_dim=128, epochs=10, batch_size=256, verbose=True)
    elif model_name == 'transformer':
        return TransformerBaseline(seq_len=100, embedding_dim=128, epochs=10, batch_size=256, verbose=True)
    elif model_name == 'gcn_lstm':
        return GCNLSTMBaseline(seq_len=10, gcn_out_dim=10, lstm_units=64, epochs=10, batch_size=256, verbose=True)
    elif model_name == 'anomaly_transformer':
        return AnomalyTransformerBaseline(win_size=100, d_model=512, n_heads=8, epochs=10, batch_size=256, verbose=True)
    elif model_name == 'dagmm':
        return DAGMMBaseline(seq_len=5, latent_dim=1, n_gmm=2, epochs=20, batch_size=256, verbose=True)
    elif model_name == 'usad':
        return USADBaseline(seq_len=5, latent_dim=32, epochs=20, batch_size=256, verbose=True)
    elif model_name == 'gdn':
        top_k = min(5, n_features - 1)
        return GDNBaseline(seq_len=5, embed_dim=64, top_k=top_k, epochs=20, batch_size=256, verbose=True)
    elif model_name == 'tranad':
        return TranADBaseline(seq_len=10, d_ff=16, epochs=20, batch_size=128, verbose=True)
    elif model_name == 'omnianomaly':
        return OmniAnomalyBaseline(seq_len=100, hidden_dim=100, z_dim=3, epochs=20, batch_size=256, verbose=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def init_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "results.json"
    if not results_path.exists():
        results_json = {
            "experiment": EXPERIMENT_NAME,
            "dataset": {
                "name": "Simulation Complex (complexity=True)",
                "description": "Simulation with realistic normal data complexity (regime switching, drift, bumps, etc.)",
                "total_length": 275000,
                "features": 8,
                "train_test_ratio": "80:20",
                "anomaly_interval_scale": 0.75,
                "complexity": True,
            },
            "timestamp": datetime.now().isoformat(),
            "models": {}
        }
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"Created: {results_path}")


def main():
    parser = argparse.ArgumentParser(description=f'Run {EXPERIMENT_NAME} experiments')
    parser.add_argument('--model', type=str, default=None,
                       help=f'Model to run: {", ".join(ALL_MODELS)} or "all"')
    parser.add_argument('--list', action='store_true', help='List current status')
    args = parser.parse_args()

    init_results_dir()

    if args.list:
        print_status()
        return

    if args.model is None:
        print(f"Usage: python {Path(__file__).name} --model <model_name|all>")
        print(f"Available: {', '.join(ALL_MODELS)}")
        print_status()
        return

    # Load data
    print("=" * 70)
    print(f"Loading {EXPERIMENT_NAME} data (complexity=True)...")
    print("=" * 70)

    loader = SimulationComplexLoader(seed=42)
    loader.load()

    train_X, train_y = loader.get_train_data()
    test_X, test_y = loader.get_test_data()
    segments = loader.get_anomaly_segments()

    print(f"  Train samples: {len(train_X):,}")
    print(f"  Test samples: {len(test_X):,}")
    print(f"  Features: {loader.num_features}")
    print(f"  Anomaly segments: {len(segments)}")

    print_status()

    models_to_run = ALL_MODELS if args.model == 'all' else [args.model]

    results_path = RESULTS_DIR / "results.json"
    with open(results_path, 'r') as f:
        results_json = json.load(f)

    for model_name in models_to_run:
        if model_name in results_json.get("models", {}):
            print(f"\n[SKIP] {model_name} already has results")
            continue

        try:
            model = get_model(model_name, loader.num_features)
            metrics = run_baseline(model, train_X, test_X, test_y, segments, model_name, RESULTS_DIR)
            update_results_json(model_name, metrics)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[ERROR] {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        with open(results_path, 'r') as f:
            results_json = json.load(f)
        print_status()

    print("\n" + "=" * 70)
    print("FINAL STATUS")
    print("=" * 70)
    print_status()


if __name__ == "__main__":
    main()
