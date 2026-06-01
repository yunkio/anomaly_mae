#!/usr/bin/env python
"""
Run comparison experiments on WaDi A2 dataset.

Training data includes anomalies (same as results/WaDi/A2 setting).
Test data is the second 50% of the dataset.

Usage:
    conda activate dc_vis
    python comparison/run_wadi_a2.py --list           # Show status
    python comparison/run_wadi_a2.py --model random   # Run single model
    python comparison/run_wadi_a2.py --model all      # Run all models
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

from comparison.data.wadi_a2_loader import WaDiA2Loader
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
RESULTS_DIR = PROJECT_ROOT / "comparison/results/WaDi_A2"

# Model order (consistent with other experiments)
ALL_MODELS = [
    'mae_teacher',
    'mae_tuned',
    'random',
    'sensor_range',
    'pca_error',
    'l2_norm',
    'nn_distance',
    'mlp',
    'mlpmixer',
    'transformer',
    'gcn_lstm',
    'anomaly_transformer',
    'dagmm',
    'usad',
    'gdn',
    'tranad',
    'omnianomaly',
]

# Baseline models to run (excluding mae which needs separate training)
BASELINE_MODELS = [m for m in ALL_MODELS if not m.startswith('mae_')]


def compute_metrics_for_results_json(scores: np.ndarray, test_y: np.ndarray, segments) -> dict:
    """Compute all metrics required for results.json."""
    evaluator = PointLevelEvaluator(test_y, segments)
    print("\n  Evaluating model...")
    base_metrics = evaluator.evaluate(scores, "model")
    f1_t, prec_t, rec_t, _ = compute_best_f1_t(test_y, scores)

    print(f"    ROC-AUC: {base_metrics['roc_auc']:.4f}")
    print(f"    PRC-AUC: {base_metrics['prc_auc']:.4f}")
    print(f"    F1: {base_metrics['f1_score']:.4f} (P={base_metrics['precision']:.4f}, R={base_metrics['recall']:.4f})")
    print(f"    F1_T: {f1_t:.4f}")
    print(f"    PA%20 F1: {base_metrics.get('pa_20_f1_score', 0):.4f}")
    print(f"    PA%80 F1: {base_metrics.get('pa_80_f1_score', 0):.4f}")

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
) -> dict:
    """Run a baseline model."""
    output_dir = results_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Running {model_name}")
    print(f"{'='*70}")
    print(f"  Device: {getattr(model, 'device', 'cpu')}")
    print(f"  Train shape: {train_X.shape}")
    print(f"  Test shape: {test_X.shape}")

    start_time = time.time()
    model.fit(train_X)
    train_time = time.time() - start_time

    print(f"\n  Train time: {train_time:.2f}s")

    # Save model
    if hasattr(model, 'save'):
        model_dir = output_dir / "model"
        try:
            model.save(model_dir)
            print(f"  Model saved to: {model_dir}")
        except Exception as e:
            print(f"  Warning: Could not save model: {e}")

    start_time = time.time()
    scores = model.predict(test_X)
    inference_time = time.time() - start_time

    print(f"  Inference time: {inference_time:.2f}s")

    np.save(output_dir / "scores.npy", scores)

    metrics = compute_metrics_for_results_json(scores, test_y, segments)
    metrics["timing"] = {
        "train_time": train_time,
        "inference_time": inference_time,
    }

    return metrics


def update_results_json(model_name: str, metrics: dict):
    """Update results.json with new model results."""
    results_path = RESULTS_DIR / "results.json"

    if results_path.exists():
        with open(results_path, 'r') as f:
            results_json = json.load(f)
    else:
        results_json = {
            "experiment": "WaDi_A2",
            "dataset": {
                "name": "WaDi A2",
                "train_test_ratio": "50:50",
            },
            "models": {}
        }

    results_json["models"][model_name] = metrics
    results_json["timestamp"] = datetime.now().isoformat()

    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"  Updated: {results_path}")


def print_status():
    """Print current status of all models."""
    results_path = RESULTS_DIR / "results.json"

    if not results_path.exists():
        print("No results.json found")
        return

    with open(results_path, 'r') as f:
        results_json = json.load(f)

    print(f"\n{'='*140}")
    print(f"CURRENT STATUS - WaDi_A2")
    print(f"{'='*140}")

    header = f"{'#':<3} {'Model':<22} {'Status':<12} {'ROC':>7} {'PRC':>7} {'F1':>7} {'F1_T':>7} {'PA20_F1':>8} {'PA20_PRC':>9} {'PA80_F1':>8} {'PA80_PRC':>9}"
    print(header)
    print("-" * 140)

    for i, model_name in enumerate(ALL_MODELS, 1):
        if model_name in results_json.get("models", {}):
            m = results_json["models"][model_name]
            pl = m["point_level"]
            f1t = m["f1_t"]["f1_t"]
            pa20_f1 = m["pa_k"]["pa_20"]["f1_score"]
            pa20_prc = m["pa_k"]["pa_20"]["prc_auc"]
            pa80_f1 = m["pa_k"]["pa_80"]["f1_score"]
            pa80_prc = m["pa_k"]["pa_80"]["prc_auc"]
            status = "✅"
            print(f"{i:<3} {model_name:<22} {status:<12} {pl['roc_auc']:>7.3f} {pl['prc_auc']:>7.3f} {pl['f1_score']:>7.3f} {f1t:>7.3f} {pa20_f1:>8.3f} {pa20_prc:>9.3f} {pa80_f1:>8.3f} {pa80_prc:>9.3f}")
        else:
            status = "-"
            print(f"{i:<3} {model_name:<22} {status:<12} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>8} {'-':>9} {'-':>8} {'-':>9}")

    print("-" * 140)

    missing = [m for m in ALL_MODELS if m not in results_json.get("models", {})]
    if missing:
        print(f"\nMissing models: {', '.join(missing)}")
    else:
        print(f"\nAll models completed!")


def get_model(model_name: str, n_features: int = 96):
    """Get model instance by name."""
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


def init_results_json():
    """Initialize results.json with dataset info."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "results.json"

    loader = WaDiA2Loader()
    loader.load()

    results_json = {
        "experiment": "WaDi_A2",
        "dataset": {
            "name": "WaDi A2",
            "train_samples": len(loader.train_labels),
            "test_samples": len(loader.test_labels),
            "features": loader.n_features,
            "train_test_ratio": "50:50",
            "train_anomaly_ratio": float(loader.train_labels.mean()),
            "test_anomaly_ratio": float(loader.test_labels.mean()),
            "anomaly_segments": len(loader.get_anomaly_segments()),
        },
        "timestamp": datetime.now().isoformat(),
        "models": {}
    }

    # If file exists, preserve existing models
    if results_path.exists():
        with open(results_path, 'r') as f:
            existing = json.load(f)
            results_json["models"] = existing.get("models", {})

    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    return loader


def main():
    parser = argparse.ArgumentParser(description='Run WaDi_A2 comparison experiments')
    parser.add_argument('--model', type=str, default=None,
                       help=f'Model to run: {", ".join(BASELINE_MODELS)} or "all"')
    parser.add_argument('--list', action='store_true',
                       help='List current status and exit')
    args = parser.parse_args()

    if args.list:
        print_status()
        return

    if args.model is None:
        print("Usage: python comparison/run_wadi_a2.py --model <model_name|all>")
        print(f"Available baseline models: {', '.join(BASELINE_MODELS)}")
        print("\nNote: mae_teacher and mae_tuned need separate training via run_wadi_ablation.py")
        print("\nCurrent status:")
        print_status()
        return

    # Initialize and load data
    print("=" * 70)
    print("Loading WaDi_A2 data...")
    print("=" * 70)

    loader = init_results_json()

    train_X, train_y = loader.get_train_data()
    test_X, test_y = loader.get_test_data()
    segments = loader.get_anomaly_segments()

    print(f"\n  Train samples: {len(train_X):,} (anomaly: {train_y.mean():.2%})")
    print(f"  Test samples: {len(test_X):,} (anomaly: {test_y.mean():.2%})")
    print(f"  Features: {loader.n_features}")
    print(f"  Test anomaly segments: {len(segments)}")

    print_status()

    # Determine models to run
    if args.model == 'all':
        models_to_run = BASELINE_MODELS
    else:
        if args.model.startswith('mae_'):
            print(f"\n[ERROR] {args.model} requires separate training.")
            print("Use: python scripts/run_wadi_ablation.py --scenario A2 --only <config>")
            return
        models_to_run = [args.model]

    # Load existing results
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, 'r') as f:
        results_json = json.load(f)

    for model_name in models_to_run:
        if model_name in results_json.get("models", {}):
            print(f"\n[SKIP] {model_name} already has results")
            continue

        try:
            model = get_model(model_name, loader.n_features)
            metrics = run_baseline(model, train_X, test_X, test_y, segments, model_name, RESULTS_DIR)
            update_results_json(model_name, metrics)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[ERROR] {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Reload and show status
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        print_status()

    print("\n" + "=" * 70)
    print("FINAL STATUS")
    print("=" * 70)
    print_status()


if __name__ == "__main__":
    main()
