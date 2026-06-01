#!/usr/bin/env python
"""
Run experiments on simulation_normal_50 dataset.

This dataset has 50% of training anomalies labeled as 'normal' (noise).
Test data is unchanged from original simulation.

Usage:
    conda activate dc_vis
    python comparison/run_simulation_normal_50.py --model random
    python comparison/run_simulation_normal_50.py --model mae
    python comparison/run_simulation_normal_50.py --model all
    python comparison/run_simulation_normal_50.py --list
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

from comparison.data.simulation_normal50_loader import SimulationNormal50Loader
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
RESULTS_DIR = PROJECT_ROOT / "comparison/results/simulation_normal_50"

# Model order (same as simulation experiment)
ALL_MODELS = [
    'mae_adaptive',
    'mae_teacheronly',
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


def compute_metrics_for_results_json(scores: np.ndarray, test_y: np.ndarray, segments) -> dict:
    """Compute all metrics required for results.json."""
    evaluator = PointLevelEvaluator(test_y, segments)
    print("\nEvaluating model...")
    base_metrics = evaluator.evaluate(scores, "model")
    f1_t, prec_t, rec_t, _ = compute_best_f1_t(test_y, scores)

    print(f"  ROC-AUC: {base_metrics['roc_auc']:.4f}")
    print(f"  PRC-AUC: {base_metrics['prc_auc']:.4f}")
    print(f"  F1: {base_metrics['f1_score']:.4f} (P={base_metrics['precision']:.4f}, R={base_metrics['recall']:.4f})")
    print(f"  F1_T: {f1_t:.4f} (P_T={prec_t:.4f}, R_T={rec_t:.4f})")
    print(f"  Computing PA%K metrics...")
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


def run_mae_training(train_X: np.ndarray, train_y: np.ndarray,
                     test_X: np.ndarray, test_y: np.ndarray,
                     segments, results_dir: Path) -> dict:
    """Train MAE model and compute metrics."""
    from mae_anomaly.config import Config
    from mae_anomaly.model import SelfDistilledMAEMultivariate
    from mae_anomaly.loss import SelfDistillationLoss
    from mae_anomaly.dataset_sliding import SlidingWindowTimeSeriesGenerator, NormalDataComplexity
    import torch.nn as nn
    import torch.optim as optim
    from torch.amp import GradScaler, autocast

    print("\n" + "=" * 60)
    print("Training MAE Model")
    print("=" * 60)

    # Create config matching 000_best_model
    config = Config(
        seq_length=500,
        num_features=8,
        d_model=128,
        nhead=8,
        num_encoder_layers=2,  # enc2 (optimal from ablation)
        num_teacher_decoder_layers=4,  # td4 (optimal from ablation)
        num_student_decoder_layers=1,  # sd1 (shallow for better discrepancy)
        num_shared_decoder_layers=0,
        dim_feedforward=512,
        dropout=0.15,
        masking_ratio=0.15,
        num_patches=100,
        patch_size=5,
        patchify_mode='patch_cnn',
        mask_after_encoder=True,
        shared_mask_token=False,
        cnn_channels=[64, 128],
        margin=0.5,
        lambda_disc=2.0,
        margin_type='dynamic',
        dynamic_margin_k=2.0,
        patch_level_loss=True,
        anomaly_loss_weight=2.0,
        anomaly_score_mode='adaptive',
        batch_size=256,
        num_epochs=50,
        learning_rate=0.002,
        weight_decay=1e-5,
        warmup_epochs=10,
        teacher_only_warmup_epochs=3,
        use_amp=True,
        point_aggregation_method='voting',
        use_discrepancy_loss=True,
        use_teacher=True,
        use_student=True,
        use_masking=True,
        force_mask_anomaly=True,
        random_seed=42,
        device='cuda',
        # Dataset params
        use_sliding_window_dataset=True,
        sliding_window_total_length=275000,
        sliding_window_stride=3,
        sliding_window_test_stride=1,
        sliding_window_train_ratio=0.8,
        anomaly_interval_scale=0.75,
    )

    # Save config
    output_dir = results_dir / "mae_adaptive"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    print(f"  Device: {config.device}")
    print(f"  d_model: {config.d_model}, nhead: {config.nhead}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")

    # Load data with MAE mode: keep all data, relabel 50% anomalies
    loader = SimulationNormal50Loader(seed=42, noise_seed=123, remove_unlabeled=False)
    loader.load()

    # Create sliding windows for training
    print("\n  Creating training windows from noisy data...")
    train_windows = []
    train_window_labels = []

    stride = config.sliding_window_stride
    noisy_train_X, noisy_train_y = loader.get_train_data()

    for i in range(0, len(noisy_train_X) - config.seq_length + 1, stride):
        window = noisy_train_X[i:i + config.seq_length]
        window_label = noisy_train_y[i:i + config.seq_length]
        train_windows.append(window)
        train_window_labels.append(window_label)

    train_windows = np.array(train_windows, dtype=np.float32)
    train_window_labels = np.array(train_window_labels, dtype=np.float32)

    print(f"  Training windows: {len(train_windows)}")
    print(f"  Window anomaly ratio: {train_window_labels.mean():.2%}")

    # Create test windows (from original test data - unchanged)
    test_windows = []
    test_window_labels = []

    for i in range(0, len(test_X) - config.seq_length + 1, 1):  # stride=1 for test
        window = test_X[i:i + config.seq_length]
        window_label = test_y[i:i + config.seq_length]
        test_windows.append(window)
        test_window_labels.append(window_label)

    test_windows = np.array(test_windows, dtype=np.float32)
    test_window_labels = np.array(test_window_labels, dtype=np.float32)

    print(f"  Test windows: {len(test_windows)}")

    # Create model
    model = SelfDistilledMAEMultivariate(config)
    model = model.to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Training setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / config.warmup_epochs
        return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda') if config.use_amp else None
    criterion = SelfDistillationLoss(config)

    # Convert to tensors
    train_tensor = torch.from_numpy(train_windows).to(config.device)
    train_labels_tensor = torch.from_numpy(train_window_labels).to(config.device)

    # Training loop
    start_time = time.time()
    model.train()

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle
        perm = torch.randperm(len(train_tensor))
        train_tensor = train_tensor[perm]
        train_labels_tensor = train_labels_tensor[perm]

        # Compute warmup factor
        warmup_factor = min(1.0, (epoch + 1) / config.warmup_epochs) if epoch < config.warmup_epochs else 1.0
        teacher_only = (epoch < config.teacher_only_warmup_epochs)

        for i in range(0, len(train_tensor), config.batch_size):
            batch_x = train_tensor[i:i + config.batch_size]
            batch_labels = train_labels_tensor[i:i + config.batch_size]

            optimizer.zero_grad()

            with autocast('cuda', enabled=config.use_amp):
                teacher_output, student_output, mask = model(batch_x, point_labels=batch_labels)
                loss, loss_dict = criterion(
                    teacher_output, student_output, batch_x, mask, batch_labels,
                    warmup_factor, teacher_only=teacher_only
                )

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        # Progress
        elapsed = time.time() - start_time
        eta = elapsed / (epoch + 1) * (config.num_epochs - epoch - 1)
        print(f"\r  Training: [{epoch+1}/{config.num_epochs}] {100*(epoch+1)/config.num_epochs:5.1f}% | "
              f"Loss: {avg_loss:.6f} | Time: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min", end="")

    print(f"\n  Training complete: {(time.time() - start_time)/60:.1f} minutes")

    train_time = time.time() - start_time

    # Save model
    torch.save(model.state_dict(), output_dir / "model.pt")
    print(f"  Model saved to: {output_dir / 'model.pt'}")

    # Inference
    print("\n  Running inference...")
    model.eval()
    start_time = time.time()

    test_tensor = torch.from_numpy(test_windows).to(config.device)
    all_scores = []

    with torch.no_grad():
        for i in range(0, len(test_tensor), config.batch_size):
            batch_x = test_tensor[i:i + config.batch_size]

            with autocast('cuda', enabled=config.use_amp):
                teacher_output, student_output, mask = model(batch_x)
                # Compute anomaly scores: recon error + adaptive discrepancy
                recon_error = ((teacher_output - batch_x) ** 2).mean(dim=2)  # (batch, seq_len)
                discrepancy = ((teacher_output - student_output) ** 2).mean(dim=2)
                # Adaptive lambda
                adaptive_lambda = recon_error.mean() / (discrepancy.mean() + 1e-6)
                scores = recon_error + adaptive_lambda * discrepancy

            all_scores.append(scores.cpu().numpy())

    window_scores = np.concatenate(all_scores, axis=0)
    inference_time = time.time() - start_time
    print(f"  Inference complete: {inference_time:.1f}s")

    # Aggregate to point-level scores using voting
    print("  Aggregating to point-level scores...")
    point_scores = np.zeros(len(test_y))
    point_counts = np.zeros(len(test_y))

    for i, score in enumerate(window_scores):
        if isinstance(score, np.ndarray) and len(score) == config.seq_length:
            point_scores[i:i + config.seq_length] += score
            point_counts[i:i + config.seq_length] += 1
        else:
            point_scores[i:i + config.seq_length] += score
            point_counts[i:i + config.seq_length] += 1

    point_counts = np.maximum(point_counts, 1)
    point_scores = point_scores / point_counts

    # Save scores
    np.save(output_dir / "scores.npy", point_scores)

    # Compute metrics
    metrics = compute_metrics_for_results_json(point_scores, test_y, segments)
    metrics["timing"] = {
        "train_time": train_time,
        "inference_time": inference_time,
    }

    # Also compute teacher-only scores
    print("\n  Computing teacher-only scores...")
    teacher_dir = results_dir / "mae_teacheronly"
    teacher_dir.mkdir(parents=True, exist_ok=True)

    all_teacher_scores = []
    with torch.no_grad():
        for i in range(0, len(test_tensor), config.batch_size):
            batch_x = test_tensor[i:i + config.batch_size]

            with autocast('cuda', enabled=config.use_amp):
                teacher_output, student_output, mask = model(batch_x)
                # Teacher reconstruction error only
                teacher_scores = ((teacher_output - batch_x) ** 2).mean(dim=2)

            all_teacher_scores.append(teacher_scores.cpu().numpy())

    teacher_window_scores = np.concatenate(all_teacher_scores, axis=0)

    # Aggregate teacher scores
    teacher_point_scores = np.zeros(len(test_y))
    teacher_point_counts = np.zeros(len(test_y))

    for i, score in enumerate(teacher_window_scores):
        if isinstance(score, np.ndarray) and len(score) == config.seq_length:
            teacher_point_scores[i:i + config.seq_length] += score
            teacher_point_counts[i:i + config.seq_length] += 1
        else:
            teacher_point_scores[i:i + config.seq_length] += score
            teacher_point_counts[i:i + config.seq_length] += 1

    teacher_point_counts = np.maximum(teacher_point_counts, 1)
    teacher_point_scores = teacher_point_scores / teacher_point_counts

    np.save(teacher_dir / "scores.npy", teacher_point_scores)

    teacher_metrics = compute_metrics_for_results_json(teacher_point_scores, test_y, segments)
    teacher_metrics["timing"] = {
        "train_time": 0,
        "inference_time": inference_time,
    }
    teacher_metrics["note"] = "Teacher reconstruction error only (no discrepancy)"

    return metrics, teacher_metrics


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

    print(f"\n{'='*60}")
    print(f"Running {model_name}")
    print(f"{'='*60}")
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

    print(f"\n  Results:")
    print(f"    ROC-AUC: {metrics['point_level']['roc_auc']:.4f}")
    print(f"    PRC-AUC: {metrics['point_level']['prc_auc']:.4f}")
    print(f"    F1: {metrics['point_level']['f1_score']:.4f}")
    print(f"    F1_T: {metrics['f1_t']['f1_t']:.4f}")
    print(f"    PA20_ROC: {metrics['pa_k']['pa_20']['roc_auc']:.4f}")
    print(f"    PA20_F1: {metrics['pa_k']['pa_20']['f1_score']:.4f}")

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

    if not results_path.exists():
        print("No results.json found")
        return

    with open(results_path, 'r') as f:
        results_json = json.load(f)

    print(f"\n{'='*130}")
    print(f"CURRENT STATUS - simulation_normal_50")
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
            status = "OK"
            print(f"{i:<3} {model_name:<22} {status:<12} {pl['roc_auc']:>7.3f} {pl['prc_auc']:>7.3f} {pl['f1_score']:>7.3f} {f1t:>7.3f} {pa20_f1:>8.3f} {pa20_prc:>9.3f} {pa80_f1:>8.3f} {pa80_prc:>9.3f}")
        else:
            status = "-"
            print(f"{i:<3} {model_name:<22} {status:<12} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>8} {'-':>9} {'-':>8} {'-':>9}")

    print("-" * 130)

    missing = [m for m in ALL_MODELS if m not in results_json.get("models", {})]
    if missing:
        print(f"\nMissing models: {', '.join(missing)}")
    else:
        print(f"\nAll models completed!")


def get_model(model_name: str, n_features: int = 8):
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


def init_results_dir():
    """Initialize results directory and results.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = RESULTS_DIR / "results.json"
    if not results_path.exists():
        results_json = {
            "experiment": "simulation_normal_50",
            "dataset": {
                "name": "Simulation Normal50 (50% anomaly noise in training)",
                "description": "50% of training anomalies labeled as normal, 50% removed",
                "total_length": 275000,
                "features": 8,
                "train_test_ratio": "80:20",
                "anomaly_interval_scale": 0.75,
            },
            "timestamp": datetime.now().isoformat(),
            "models": {}
        }
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"Created: {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Run simulation_normal_50 experiments')
    parser.add_argument('--model', type=str, default=None,
                       help=f'Model to run: {", ".join(ALL_MODELS)} or "all" or "mae"')
    parser.add_argument('--list', action='store_true',
                       help='List current status and exit')
    args = parser.parse_args()

    init_results_dir()

    if args.list:
        print_status()
        return

    if args.model is None:
        print("Usage: python run_simulation_normal_50.py --model <model_name|all|mae>")
        print(f"Available models: {', '.join(ALL_MODELS)}")
        print("\nCurrent status:")
        print_status()
        return

    # Load data
    print("=" * 70)
    print("Loading simulation_normal_50 data...")
    print("=" * 70)

    loader = SimulationNormal50Loader(seed=42, noise_seed=123)
    loader.load()

    train_X, train_y = loader.get_train_data()
    test_X, test_y = loader.get_test_data()
    segments = loader.get_anomaly_segments()

    print(f"  Train samples: {len(train_X):,}")
    print(f"  Test samples: {len(test_X):,}")
    print(f"  Features: {loader.num_features}")
    print(f"  Anomaly segments: {len(segments)}")

    print_status()

    # Determine models to run
    if args.model == 'all':
        models_to_run = ALL_MODELS
    elif args.model == 'mae':
        models_to_run = ['mae_adaptive', 'mae_teacheronly']
    else:
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
            if model_name in ['mae_adaptive', 'mae_teacheronly']:
                if 'mae_adaptive' not in results_json.get("models", {}):
                    mae_metrics, teacher_metrics = run_mae_training(
                        train_X, train_y, test_X, test_y, segments, RESULTS_DIR
                    )
                    update_results_json('mae_adaptive', mae_metrics)
                    update_results_json('mae_teacheronly', teacher_metrics)
                continue

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
