#!/usr/bin/env python
"""
Run new deep learning baseline models on simulation_normalonly experiment.

Models to run:
1. DAGMM - Deep Autoencoding GMM
2. USAD - UnSupervised Anomaly Detection
3. GDN - Graph Deviation Network
4. TranAD - Transformer-based AD
5. OmniAnomaly - Stochastic RNN VAE

Usage:
    conda activate dc_vis
    python comparison/run_new_models.py --model dagmm
    python comparison/run_new_models.py --model all
    python comparison/run_new_models.py --list  # Show status
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

from comparison.data.simulation_normalonly_loader import SimulationNormalOnlyLoader
from comparison.baselines import (
    PointLevelEvaluator,
    compute_best_f1_t,
)
from comparison.baselines.tranad import TranADBaseline
from comparison.baselines.usad import USADBaseline
from comparison.baselines.dagmm import DAGMMBaseline
from comparison.baselines.gdn import GDNBaseline
from comparison.baselines.omnianomaly import OmniAnomalyBaseline

EXPERIMENT_NAME = "simulation_normalonly"
RESULTS_DIR = PROJECT_ROOT / "comparison/results" / EXPERIMENT_NAME
PA_K_VALUES = [10, 20, 50, 80, 100]

# Models to run (order by expected runtime)
NEW_MODELS = ['dagmm', 'usad', 'gdn', 'tranad', 'omnianomaly']


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


def run_baseline(
    model,
    train_X: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    segments,
    model_name: str,
) -> dict:
    """Run a baseline model."""
    output_dir = RESULTS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running {model_name}")
    print(f"{'='*60}")
    print(f"  Device: {getattr(model, 'device', 'cpu')}")

    start_time = time.time()
    model.fit(train_X)
    train_time = time.time() - start_time

    print(f"\n  Train time: {train_time:.2f}s")

    # Save model if possible
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

    print(f"\n  Results:")
    print(f"    ROC-AUC: {metrics['point_level']['roc_auc']:.4f}")
    print(f"    PRC-AUC: {metrics['point_level']['prc_auc']:.4f}")
    print(f"    F1: {metrics['point_level']['f1_score']:.4f}")
    print(f"    F1_T: {metrics['f1_t']['f1_t']:.4f}")
    print(f"    PA20_ROC: {metrics['pa_k']['pa_20']['roc_auc']:.4f}")
    print(f"    PA20_F1: {metrics['pa_k']['pa_20']['f1_score']:.4f}")

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

    with open(results_path, 'r') as f:
        results_json = json.load(f)

    results_json["models"][model_name] = metrics
    results_json["timestamp"] = datetime.now().isoformat()

    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"  Updated: {results_path}")


def print_status():
    """Print current status of all models."""
    results_path = RESULTS_DIR / "results.json"

    with open(results_path, 'r') as f:
        results_json = json.load(f)

    all_models = [
        "mae_adaptive", "mae_teacheronly",
        "random", "sensor_range", "pca_error", "l2_norm", "nn_distance",
        "mlp", "mlpmixer", "transformer", "gcn_lstm",
        "anomaly_transformer",
        "dagmm", "usad", "gdn", "tranad", "omnianomaly"
    ]

    print("\n" + "=" * 130)
    print("CURRENT STATUS - simulation_normalonly")
    print("=" * 130)

    header = f"{'#':<3} {'Model':<22} {'Status':<8} {'ROC':>7} {'PRC':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'F1_T':>7} {'PA20_ROC':>9} {'PA20_F1':>8}"
    print(header)
    print("-" * 130)

    for i, model_name in enumerate(all_models, 1):
        if model_name in results_json.get("models", {}):
            m = results_json["models"][model_name]
            pl = m["point_level"]
            f1t = m["f1_t"]["f1_t"]
            pa20_roc = m["pa_k"]["pa_20"]["roc_auc"]
            pa20_f1 = m["pa_k"]["pa_20"]["f1_score"]
            status = "✅"
            print(f"{i:<3} {model_name:<22} {status:<8} {pl['roc_auc']:>7.3f} {pl['prc_auc']:>7.3f} {pl['f1_score']:>7.3f} {pl['precision']:>7.3f} {pl['recall']:>7.3f} {f1t:>7.3f} {pa20_roc:>9.3f} {pa20_f1:>8.3f}")
        else:
            status = "-"
            print(f"{i:<3} {model_name:<22} {status:<8} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>9} {'-':>8}")

    print("-" * 130)

    # Show which models need to run
    missing = [m for m in NEW_MODELS if m not in results_json.get("models", {})]
    if missing:
        print(f"\nModels to run: {', '.join(missing)}")
    else:
        print(f"\nAll new models completed!")


def get_model(model_name: str):
    """Get model instance by name."""
    if model_name == 'dagmm':
        return DAGMMBaseline(seq_len=5, latent_dim=1, n_gmm=2, epochs=20, batch_size=256, verbose=True)
    elif model_name == 'usad':
        return USADBaseline(seq_len=5, latent_dim=32, epochs=20, batch_size=256, verbose=True)
    elif model_name == 'gdn':
        return GDNBaseline(seq_len=5, embed_dim=64, top_k=5, epochs=20, batch_size=256, verbose=True)
    elif model_name == 'tranad':
        return TranADBaseline(seq_len=10, d_ff=16, epochs=20, batch_size=128, verbose=True)
    elif model_name == 'omnianomaly':
        return OmniAnomalyBaseline(seq_len=100, hidden_dim=100, z_dim=3, epochs=20, batch_size=256, verbose=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description='Run new baseline models on simulation_normalonly')
    parser.add_argument('--model', type=str, default=None,
                       help=f'Model to run: {", ".join(NEW_MODELS)} or "all"')
    parser.add_argument('--list', action='store_true',
                       help='List current status and exit')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Override epochs for neural models')
    args = parser.parse_args()

    if args.list:
        print_status()
        return

    if args.model is None:
        print("Usage: python run_new_models.py --model <model_name>")
        print(f"Available models: {', '.join(NEW_MODELS)}, or 'all'")
        print("\nCurrent status:")
        print_status()
        return

    # Load data
    print("=" * 70)
    print("Loading simulation data (normal-only training)...")
    print("=" * 70)

    loader = SimulationNormalOnlyLoader(
        total_length=275000,
        train_ratio=0.8,
        num_features=8,
        interval_scale=0.75,
        seed=42
    )
    loader.load()

    train_X = loader.get_train_data_concatenated()
    test_X, test_y = loader.get_test_data()
    segments = loader.get_anomaly_segments()

    print(f"  Train samples: {len(train_X):,}")
    print(f"  Test samples: {len(test_X):,}")
    print(f"  Features: {loader.num_features}")
    print(f"  Anomaly segments: {len(segments)}")

    # Show initial status
    print_status()

    # Determine which models to run
    models_to_run = NEW_MODELS if args.model == 'all' else [args.model]

    # Check which models already have results
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, 'r') as f:
        results_json = json.load(f)

    for model_name in models_to_run:
        if model_name in results_json.get("models", {}):
            print(f"\n[SKIP] {model_name} already has results")
            continue

        # Run model
        try:
            model = get_model(model_name)
            metrics = run_baseline(model, train_X, test_X, test_y, segments, model_name)
            update_results_json(model_name, metrics)

            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[ERROR] {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Show updated status
        print_status()

    print("\n" + "=" * 70)
    print("FINAL STATUS")
    print("=" * 70)
    print_status()


if __name__ == "__main__":
    main()
