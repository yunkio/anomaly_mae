"""
Base Model Experiments: Default config on ALL datasets with epoch-wise monitoring.

Runs the default model on every dataset with epoch callbacks (lightweight test metrics
every EVAL_INTERVAL epochs). After training, runs full all-patches evaluation + visualization
in a background CPU process.

Pipeline: GPU train (with epoch callbacks) → GPU inference → free GPU → background CPU eval+viz.

Output directory structure:
    results/experiments/{timestamp}_{suffix}/
    ├── {DatasetGroup}/{Scenario}/
    │   ├── best_model.pt, best_config.json, training_histories.json
    │   ├── epoch_metrics.json, experiment_metadata.json
    │   ├── best_model_detailed.csv, anomaly_type_metrics.json
    │   ├── checkpoints/
    │   │   ├── best_checkpoint.pt   (best PRC-AUC epoch)
    │   │   └── latest_checkpoint.pt (last evaluated epoch)
    │   ├── epoch_scores/
    │   │   └── epoch_{NNN}_scores.npz  (point-level: adaptive, teacher_recon, discrepancy)
    │   └── visualization/
    │       ├── best_model/  (15+ PNGs via BestModelVisualizer.generate_all)
    │       └── epoch_metrics/  (4 PNGs: prc_auc, f1_t, pa_k_f1, dashboard)
    └── summary.json

Usage:
    conda activate dc_vis
    python scripts/run_base_experiments.py --set A                     # All 33 datasets (5 base + 28 SMD), Set A
    python scripts/run_base_experiments.py --set B                     # All 33 datasets, Set B
    python scripts/run_base_experiments.py --set C                     # All 33 datasets, Set C (dynamic d_model, linear)
    python scripts/run_base_experiments.py --set A --dataset SWaT_A1A2 # Specific dataset (also: PSM, smd_machine-1-1, etc.)
    python scripts/run_base_experiments.py --set A --start-from 5      # Resume from index 5
    python scripts/run_base_experiments.py --set A --list              # List all datasets
"""

import os
import sys
import gc
import time
import json
import shutil
import argparse
import warnings
import threading
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from dataclasses import asdict
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# === Imports from active codebase ===
from mae_anomaly import Config, SelfDistilledMAEMultivariate, set_seed
from mae_anomaly.dataset_sliding import SlidingWindowDataset, AnomalyRegion
from mae_anomaly.evaluator import Evaluator
from mae_anomaly.trainer import Trainer
from mae_anomaly.visualization import setup_style, BestModelVisualizer
from mae_anomaly.visualization.base import derive_pred_data
from mae_anomaly.datasets import (
    get_dataset_loader, NoisyLabelSlidingWindowDataset,
    load_swat_combined, load_swat_combined_swap,
    load_wadi_14days_combined,
)
from mae_anomaly.datasets.loaders import SMD_MACHINE_NAMES, EXATHLON_APP_IDS
from mae_anomaly.utils import make_config, free_gpu, mem_status
from mae_anomaly.utils.experiment import make_numbered_experiment_dir, resolve_dynamic_d_model
from scripts.ablation.run_ablation import compute_loss_statistics

# Background process tracking
_background_processes = []

# Epoch eval interval
EVAL_INTERVAL = 5


# =============================================================================
# Dynamic Suffix Generation
# =============================================================================

def make_dynamic_suffix(overrides: dict) -> str:
    """Generate experiment directory suffix from actual config values.

    Format: w{seq}p{patch}e{enc}t{td}d{sd}[_dynamic][_linear|_cnn][_minmax][_k{val}][_dm{val}]
    Examples:
        w500p10e2t4d1_dynamic_linear       (Set C baseline, linear patchify)
        w500p10e2t4d1_dynamic_cnn          (Set C with patch_cnn patchify)
        w500p10e2t4d1_dynamic_linear_minmax (Set C with min-max normalization)
        w500p10e6t5d1_dynamic_linear       (enc=6, td=5)
        w500p10e2t4d1_dynamic_linear_k6    (dynamic_margin_k=6)
        w500p5e2t4d1                       (Set A: patch_cnn, hinge margin)
        w500p20e2t4d1_d256k5               (Set B: d_model=256, kernel=5)
    """
    w = overrides.get('seq_length', 500)
    p = overrides.get('patch_size', 10)
    e = overrides.get('num_encoder_layers', 2)
    t = overrides.get('num_teacher_decoder_layers', 4)
    s = overrides.get('num_student_decoder_layers', 1)
    suffix = f'w{w}p{p}e{e}t{t}d{s}'

    margin = overrides.get('margin_type', 'dynamic')
    patchify = overrides.get('patchify_mode', 'patch_cnn')

    # d_model / kernel modifiers (Set B style)
    d_model = overrides.get('d_model', 128)
    kernel = overrides.get('cnn_kernel_size', 3)
    if patchify == 'patch_cnn':
        extras = ''
        if isinstance(d_model, int) and d_model != 128:
            extras += f'_d{d_model}'
        if kernel != 3:
            extras += f'k{kernel}'
        suffix += extras

    # Margin / patchify mode
    if margin == 'dynamic':
        suffix += '_dynamic'
    elif margin == 'none':
        suffix += '_nomargin'
    if patchify == 'linear':
        suffix += '_linear'
    elif patchify == 'patch_cnn':
        suffix += '_cnn'

    # Normalization mode
    normalize = overrides.get('normalize_mode', 'zscore')
    if normalize != 'zscore':
        suffix += f'_{normalize}'

    # Non-default dynamic_margin_k
    k = overrides.get('dynamic_margin_k', 2.0)
    if k != 2.0:
        k_str = str(int(k)) if k == int(k) else str(k)
        suffix += f'_k{k_str}'

    # Discriminator params
    if overrides.get('use_discriminator', False):
        grad = overrides.get('d_grad_student_layers', 'all')
        adv_w = overrides.get('adv_loss_weight', 1.0)
        ch = overrides.get('disc_channels', (64, 32))
        # Format: _D{grad}a{weight}c{c1}-{c2}
        grad_tag = 'A' if grad == 'all' else 'L'
        adv_str = str(adv_w).replace('.', '')
        if isinstance(ch, (tuple, list)):
            ch_str = f'{ch[0]}-{ch[1]}'
        else:
            ch_str = str(ch)
        suffix += f'_D{grad_tag}a{adv_str}c{ch_str}'

    return suffix


# =============================================================================
# Config Presets (Set A / Set B)
# =============================================================================

CONFIG_PRESETS = {
    'A': {
        'suffix': 'w500p5e2t4d1',
        'description': 'w500, p5, enc2, td4, sd1, d128, k3',
        'overrides': {
            'seq_length': 500,
            'patch_size': 5,
            'num_patches': 100,
            'd_model': 128,
            'nhead': 8,
            'dim_feedforward': 512,
            'num_encoder_layers': 2,
            'num_teacher_decoder_layers': 4,
            'num_student_decoder_layers': 1,
            'num_epochs': 50,
            'learning_rate': 0.001,
            'batch_size': 512,
            'cnn_kernel_size': 3,
            'cnn_channels': None,  # Auto-scale (d_model//2, d_model) = (64, 128)
            'patchify_mode': 'patch_cnn',
            'mask_after_encoder': True,
            'anomaly_score_mode': 'adaptive',
        },
    },
    'B': {
        'suffix': 'w500p20e2t4d1_d256k5',
        'description': 'w500, p20, enc2, td4, sd1, d256, k5',
        'overrides': {
            'seq_length': 500,
            'patch_size': 20,
            'num_patches': 25,
            'd_model': 256,
            'nhead': 8,
            'dim_feedforward': 1024,
            'num_encoder_layers': 2,
            'num_teacher_decoder_layers': 4,
            'num_student_decoder_layers': 1,
            'num_epochs': 50,
            'learning_rate': 0.001,
            'batch_size': 512,
            'cnn_kernel_size': 5,
            'cnn_channels': None,  # Auto-scale (d_model//2, d_model) = (128, 256)
            'patchify_mode': 'patch_cnn',
            'mask_after_encoder': True,
            'anomaly_score_mode': 'adaptive',
        },
    },
    'C': {
        'suffix': 'w500p10e2t4d1_dynamic_linear',
        'description': 'w500, p10, enc2, td4, sd1, dynamic d_model, linear embed',
        'overrides': {
            'seq_length': 500,
            'patch_size': 10,
            'num_patches': 50,
            'd_model': 'dynamic',  # Resolved per-dataset: smallest in [128,192,256,384,512] >= 10*num_features
            # dim_feedforward: auto-computed as 4 * d_model (not in overrides → auto)
            'nhead': 8,
            'num_encoder_layers': 2,
            'num_teacher_decoder_layers': 4,
            'num_student_decoder_layers': 1,
            'num_epochs': 50,
            'learning_rate': 0.001,
            'batch_size': 512,
            'patchify_mode': 'linear',  # Linear embedding (no CNN)
            'mask_after_encoder': True,
            'anomaly_score_mode': 'adaptive',
        },
    },
}


# =============================================================================
# Dataset Definitions (5 active base datasets — simulation, SWaT, WaDi×2, PSM)
# Plus 28 dynamically-generated SMD datasets below (total 33 datasets).
# SWaT는 단일 학습 + dual eval (full + excl_region22).
# =============================================================================

DATASETS = [
    # Simulation
    {
        'key': 'simulation',
        'loader': 'simulation',
        'train_stride': 21,
        'normal50': False,
        'results_subdir': 'simulation/simulation',
    },
    # SWaT — single training run, dual evaluation (full + excl_region22)
    {
        'key': 'SWaT_A1A2',
        'loader': 'swat_A1A2',
        'train_stride': 21,
        'normal50': False,
        'results_subdir': 'SWaT/A1A2',
    },
    # WaDi (14days normal + attack raw, A1 & A2)
    {
        'key': 'WaDi_A1',
        'loader': 'WaDi_14days_A1',
        'train_stride': 21,
        'normal50': False,
        'results_subdir': 'WaDi/A1',
    },
    {
        'key': 'WaDi_A2',
        'loader': 'WaDi_14days_A2',
        'train_stride': 21,
        'normal50': False,
        'results_subdir': 'WaDi/A2',
    },
    # PSM (Pooled Server Metrics, eBay) — single contiguous stream
    # 220,322 timesteps × 25 features, train_ratio=0.8007, 72 anomaly regions
    {
        'key': 'PSM',
        'loader': 'PSM',
        'train_stride': 21,
        'normal50': False,
        'results_subdir': 'PSM',
    },
]
# NOTE: 비활성 variant들 (normal50, simulation_complex, SWaT_A1A2_swap)은 2026-05-17 자로
# default DATASETS에서 제외. loader 자체는 mae_anomaly/datasets/loaders.py에 유지되어
# 명시적 --dataset 호출 또는 과거 결과 로드는 가능. 활성 데이터셋은 위 5개 + 28 SMD + 6 Exathlon = 39개.

# SMD Simple Split datasets (28 machines, train+front50%test / back50%test)
# Dynamically generated from SMD_MACHINE_NAMES (mae_anomaly.datasets.loaders)
SMD_DATASETS = []
for _machine in SMD_MACHINE_NAMES:
    SMD_DATASETS.append({
        'key': f'smd_{_machine}',
        'loader': f'smd_simple_{_machine}',
        'train_stride': 21,
        'normal50': False,
        'results_subdir': f'SMD/{_machine}',
    })
del _machine  # Clean up loop variable

# Exathlon Per-App datasets (6 apps × per-app, TimeSeAD 6-app convention)
# Apps {1, 2, 4, 5, 6, 9}. Apps 7/8 excluded (structural deficiency).
# Each app: all undisturbed traces + first floor(N_dist/2) disturbed → train, rest → test
EXATHLON_DATASETS = []
for _app in EXATHLON_APP_IDS:
    EXATHLON_DATASETS.append({
        'key': f'exathlon_app{_app}',
        'loader': f'exathlon_app{_app}',
        'train_stride': 21,
        'normal50': False,
        'results_subdir': f'Exathlon/app{_app}',
    })
del _app  # Clean up loop variable


# =============================================================================
# Normal50 Noise Application
# =============================================================================

def apply_normal50_noise(point_labels, anomaly_regions, train_ratio, noise_seed=123):
    """Apply 50% label noise to training anomaly regions."""
    total_length = len(point_labels)
    train_end = int(total_length * train_ratio)

    train_regions = [r for r in anomaly_regions if r.start < train_end]

    np.random.seed(noise_seed)
    n_to_relabel = len(train_regions) // 2
    indices_to_relabel = set(np.random.choice(
        len(train_regions), n_to_relabel, replace=False
    ))

    noisy_labels = point_labels.copy()
    for idx, region in enumerate(train_regions):
        if idx in indices_to_relabel:
            end_idx = min(region.end, train_end)
            noisy_labels[region.start:end_idx] = 0

    return noisy_labels


# =============================================================================
# Batch Profiling: Save first-N-batch per-component timing from Trainer
# =============================================================================

PROFILE_N_BATCHES = 10  # Number of batches to profile in epoch 1


def save_batch_profiling(batch_profiles, config, n_batches_per_epoch, exp_dir):
    """Save per-batch component timing and print profiler-like summary table.

    Args:
        batch_profiles: List of dicts from Trainer (one per profiled batch).
        config: Model config (for batch_size, num_epochs).
        n_batches_per_epoch: Total batches per epoch.
        exp_dir: Directory to save files.
    """
    if not batch_profiles:
        return

    n = len(batch_profiles)
    components = ['data_to_gpu_ms', 'model_forward_ms', 'loss_compute_ms',
                  'backward_ms', 'optimizer_step_ms']
    comp_labels = ['Data → GPU', 'Model Forward', 'Loss Compute',
                   'Backward', 'Optimizer Step']

    # Layer-level components (nested inside model_forward)
    layer_components = ['embed_input_ms', 'masking_ms', 'encoder_ms',
                        'teacher_decoder_ms', 'student_decoder_ms']
    layer_labels = ['Embed (Patchify+CNN)', 'Masking', 'Encoder',
                    'Teacher Decoder', 'Student Decoder']
    has_layers = batch_profiles[0].get('layer_timing') is not None

    # Compute summary statistics
    summary_rows = []
    layer_summary_rows = []
    total_sum = 0.0
    for comp, label in zip(components, comp_labels):
        values = [bp[comp] for bp in batch_profiles]
        total = sum(values)
        avg = total / n
        total_sum += total
        summary_rows.append({
            'component': label,
            'total_ms': total,
            'avg_ms': avg,
            'min_ms': min(values),
            'max_ms': max(values),
            'calls': n,
        })

    if has_layers:
        for lcomp, llabel in zip(layer_components, layer_labels):
            lvals = [bp['layer_timing'][lcomp] for bp in batch_profiles]
            ltotal = sum(lvals)
            layer_summary_rows.append({
                'component': llabel,
                'total_ms': ltotal,
                'avg_ms': ltotal / n,
                'min_ms': min(lvals),
                'max_ms': max(lvals),
                'calls': n,
            })

    avg_batch_ms = total_sum / n
    est_epoch_s = avg_batch_ms * n_batches_per_epoch / 1000
    est_total_s = est_epoch_s * config.num_epochs

    # Print profiler-like table
    header = f"{'Component':<24} {'Total (ms)':>12} {'Avg (ms)':>10} {'Min':>10} {'Max':>10} {'Calls':>6}"
    sep = '-' * len(header)
    lines = [
        f"Batch Profiling: {n} batches (batch_size={config.batch_size}, batch 0 skipped)",
        f"{'=' * len(header)}",
        header, sep,
    ]
    for i, row in enumerate(summary_rows):
        lines.append(f"{row['component']:<24} {row['total_ms']:>12.2f} {row['avg_ms']:>10.2f} "
                     f"{row['min_ms']:>10.2f} {row['max_ms']:>10.2f} {row['calls']:>6}")
        # Layer breakdown after Model Forward
        if i == 1 and layer_summary_rows:
            for j, lr in enumerate(layer_summary_rows):
                prefix = '  └─ ' if j == len(layer_summary_rows) - 1 else '  ├─ '
                lines.append(f"{prefix}{lr['component']:<20} {lr['total_ms']:>12.2f} {lr['avg_ms']:>10.2f} "
                             f"{lr['min_ms']:>10.2f} {lr['max_ms']:>10.2f} {lr['calls']:>6}")
    lines.append(sep)
    lines.append(f"{'TOTAL':<24} {total_sum:>12.2f} {avg_batch_ms:>10.2f} "
                 f"{'':>10} {'':>10} {n:>6}")
    lines.append(sep)
    lines.append(f"Estimated: {avg_batch_ms:.1f}ms/batch, {est_epoch_s:.1f}s/epoch, "
                 f"{est_total_s:.0f}s total ({config.num_epochs} epochs, {n_batches_per_epoch} batches/epoch)")
    table_str = '\n'.join(lines)
    print(f"\n{table_str}\n")

    # Save detail text
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'batch_profiling.txt'), 'w') as f:
        f.write(table_str)
        f.write('\n\n--- Per-batch detail ---\n')
        for bp in batch_profiles:
            line = (f"  Batch {bp['batch']:>3}: "
                    f"data={bp['data_to_gpu_ms']:.2f} fwd={bp['model_forward_ms']:.2f} "
                    f"loss={bp['loss_compute_ms']:.2f} bwd={bp['backward_ms']:.2f} "
                    f"optim={bp['optimizer_step_ms']:.2f} total={bp['total_ms']:.2f} ms")
            if bp.get('layer_timing'):
                lt = bp['layer_timing']
                line += (f"\n         layers: embed={lt['embed_input_ms']:.2f} "
                         f"mask={lt['masking_ms']:.2f} enc={lt['encoder_ms']:.2f} "
                         f"t_dec={lt['teacher_decoder_ms']:.2f} s_dec={lt['student_decoder_ms']:.2f}")
            f.write(line + '\n')

    # Save JSON summary
    profiling_json = {
        'batch_size': config.batch_size,
        'n_batches_per_epoch': n_batches_per_epoch,
        'n_profiled_batches': n,
        'batch_0_skipped': True,
        'avg_batch_time_ms': avg_batch_ms,
        'estimated_epoch_time_s': est_epoch_s,
        'estimated_total_train_time_s': est_total_s,
        'num_epochs': config.num_epochs,
        'components': summary_rows,
        'layer_components': layer_summary_rows if has_layers else [],
        'per_batch': batch_profiles,
    }
    with open(os.path.join(exp_dir, 'batch_profiling.json'), 'w') as f:
        json.dump(profiling_json, f, indent=2)


# =============================================================================
# Epoch-wise Test Evaluation (All-Patches)
# Uses Evaluator._compute_patch_scores_all_patches() for correct inference.
# =============================================================================

def compute_epoch_test_inference(model, test_loader, config, test_dataset=None):
    """GPU phase: all-patches inference. Returns numpy patch scores for CPU eval.

    This runs synchronously on GPU and returns data that can be evaluated
    asynchronously on CPU while GPU continues training.
    """
    t_infer = time.time()
    evaluator = Evaluator(model, config, test_loader, test_dataset=test_dataset)
    recon_patches, disc_patches, student_recon_patches, labels, sample_types, anomaly_types = \
        evaluator._compute_patch_scores_all_patches()
    torch.cuda.synchronize()
    inference_time = time.time() - t_infer
    disc_per_feature = evaluator.disc_per_feature  # (n_windows, F) or None
    fm_patches = getattr(evaluator, 'fm_patches', None)  # (n_windows, num_patches) or None
    del evaluator

    return {
        'recon_patches': recon_patches,
        'disc_patches': disc_patches,
        'student_recon_patches': student_recon_patches,
        'labels': labels,
        'sample_types': sample_types,
        'anomaly_types': anomaly_types,
        'inference_time': inference_time,
        'disc_per_feature': disc_per_feature,
        'fm_patches': fm_patches,
    }


def compute_epoch_test_eval(eval_data, config, test_loader, test_dataset=None):
    """CPU phase: point-level evaluation from precomputed patch scores.

    Pure numpy/sklearn operations — no GPU needed. Safe to run in a background thread.
    """
    t_eval = time.time()
    evaluator = Evaluator(None, config, test_loader, test_dataset=test_dataset)
    evaluator.set_precomputed_patch_scores(
        eval_data['recon_patches'], eval_data['disc_patches'],
        eval_data['student_recon_patches'],
        eval_data['labels'], eval_data['sample_types'], eval_data['anomaly_types'],
        fm_patches=eval_data.get('fm_patches'),
    )
    metrics = evaluator.evaluate()

    disc_metrics = evaluator.evaluate_by_score_type('disc')
    teacher_recon_metrics = evaluator.evaluate_by_score_type('teacher_recon')
    eval_time = time.time() - t_eval

    # Compute disc_SNR from patch-level discrepancy scores
    detailed_losses = evaluator.compute_detailed_losses()
    loss_stats = compute_loss_statistics(detailed_losses)
    metrics['disc_snr'] = loss_stats.get('disc_SNR', 0)

    metrics['teacher_prc_auc'] = teacher_recon_metrics.get('prc_auc', 0)
    metrics['teacher_f1_t'] = teacher_recon_metrics.get('f1_t', 0)
    metrics['teacher_pa_20_f1'] = teacher_recon_metrics.get('pa_20_f1', 0)

    # PA%K AUC: teacher-only versions (adaptive versions already in metrics)
    for m in ['prc_auc', 'roc_auc', 'f1', 'f1_t', 'precision', 'recall',
              'f1_raw', 'f1_t_raw', 'precision_raw', 'recall_raw']:
        metrics[f'teacher_pak_auc_{m}'] = teacher_recon_metrics.get(f'pak_auc_{m}', 0)

    metrics['_inference_time'] = eval_data['inference_time']
    metrics['_eval_time'] = eval_time

    del evaluator
    return metrics


def compute_epoch_test_metrics(model, test_loader, config, test_dataset=None):
    """All-patches test evaluation with point-level metrics (synchronous).

    GPU all-patches inference → Evaluator point-level evaluation (PRC, F1_T, PA%K).
    Returns point-level metrics dict (with timing) + raw patch_scores.
    Used by post-training full evaluation. For epoch callbacks, use the async variant.
    """
    eval_data = compute_epoch_test_inference(model, test_loader, config, test_dataset)
    metrics = compute_epoch_test_eval(eval_data, config, test_loader, test_dataset)

    patch_scores_dict = {
        'recon': eval_data['recon_patches'],
        'disc': eval_data['disc_patches'],
        'student_recon': eval_data['student_recon_patches'],
        'labels': eval_data['labels'],
        'sample_types': eval_data['sample_types'],
        'anomaly_types': eval_data['anomaly_types'],
        'disc_per_feature': eval_data.get('disc_per_feature'),
    }

    return metrics, patch_scores_dict


def compute_contrib_from_eval_data(eval_data, config):
    """Compute contribution ratios from pre-computed patch scores (pure numpy, no GPU).

    Uses the same eval_data already produced by compute_epoch_test_inference(),
    so no additional model inference is needed.
    """
    from mae_anomaly import SLIDING_ANOMALY_TYPE_NAMES

    recon_patches = eval_data['recon_patches']
    disc_patches = eval_data['disc_patches']
    sample_types_all = eval_data['sample_types']
    anomaly_types_all = eval_data['anomaly_types']
    fm_patches = eval_data.get('fm_patches')  # None if FM not used

    # Window-level scores (mean over patches)
    recon_all = recon_patches.mean(axis=1)
    disc_all = disc_patches.mean(axis=1)
    fm_all = fm_patches.mean(axis=1) if fm_patches is not None else None

    # Weighted contributions based on scoring mode
    # student_error = (w_disc * scaled_disc + w_fm * scaled_fm) / (w_disc + w_fm)
    # score = recon + student_error  (teacher:student = 1:1)
    if config.anomaly_score_mode == 'adaptive':
        recon_mean = recon_all.mean() + 1e-4
        scaled_disc = disc_all * (recon_mean / (disc_all.mean() + 1e-4))

        use_fm = getattr(config, 'use_feature_matching', False) and fm_all is not None
        od_enabled = getattr(config, 'use_output_discrepancy', True)

        w_disc = getattr(config, 'eval_disc_weight', -1.0)
        w_fm = getattr(config, 'eval_fm_weight', -1.0)
        if w_disc < 0:
            w_disc = 1.0
        if w_fm < 0:
            w_fm = getattr(config, 'fm_loss_weight', 1.0)
        if not od_enabled:
            w_disc = 0.0

        if use_fm:
            scaled_fm = fm_all * (recon_mean / (fm_all.mean() + 1e-4))
            if w_disc + w_fm > 0:
                student_contrib_all = (w_disc * scaled_disc + w_fm * scaled_fm) / (w_disc + w_fm)
            else:
                student_contrib_all = np.zeros_like(recon_all)
        elif w_disc > 0:
            student_contrib_all = scaled_disc
        else:
            student_contrib_all = np.zeros_like(recon_all)

        recon_contrib_all = recon_all
        disc_contrib_all = student_contrib_all  # "disc" = entire student error (disc + fm)
    else:
        recon_contrib_all = recon_all
        disc_contrib_all = config.lambda_disc * disc_all
    total = recon_contrib_all + disc_contrib_all + 1e-4
    recon_ratio_all = recon_contrib_all / total
    disc_ratio_all = disc_contrib_all / total

    # Per sample-type means (0=pure_normal, 1=disturbing_normal, 2=anomaly)
    results = {}
    for type_idx, type_name in [(0, 'normal'), (1, 'disturbing'), (2, 'anomaly')]:
        mask = (sample_types_all == type_idx)
        if mask.sum() > 0:
            results[f'recon_ratio_{type_name}'] = float(recon_ratio_all[mask].mean() * 100)
            results[f'disc_ratio_{type_name}'] = float(disc_ratio_all[mask].mean() * 100)
            results[f'recon_score_{type_name}'] = float(recon_contrib_all[mask].mean())
            results[f'disc_score_{type_name}'] = float(disc_contrib_all[mask].mean())
            results[f'raw_recon_{type_name}'] = float(recon_all[mask].mean())
            results[f'raw_disc_{type_name}'] = float(disc_all[mask].mean())
        else:
            for prefix in ['recon_ratio_', 'disc_ratio_', 'recon_score_', 'disc_score_',
                           'raw_recon_', 'raw_disc_']:
                results[f'{prefix}{type_name}'] = 0.0

    # Per anomaly-type scores
    anomaly_type_scores = {}
    anomaly_mask = (sample_types_all == 2)
    unique_atypes = sorted(set(int(x) for x in np.unique(anomaly_types_all[anomaly_mask]))) if anomaly_mask.any() else []
    for atype_idx in unique_atypes:
        atype_name = SLIDING_ANOMALY_TYPE_NAMES[atype_idx] if atype_idx < len(SLIDING_ANOMALY_TYPE_NAMES) else f'fault_{atype_idx}'
        atype_mask = anomaly_mask & (anomaly_types_all == atype_idx)
        if atype_mask.sum() > 0:
            anomaly_type_scores[atype_name] = {
                'recon_score': float(recon_contrib_all[atype_mask].mean()),
                'disc_score': float(disc_contrib_all[atype_mask].mean()),
                'recon_ratio': float(recon_ratio_all[atype_mask].mean() * 100),
                'disc_ratio': float(disc_ratio_all[atype_mask].mean() * 100),
                'count': int(atype_mask.sum()),
            }
    normal_mask = (sample_types_all == 0)
    if normal_mask.sum() > 0:
        anomaly_type_scores['normal'] = {
            'recon_score': float(recon_contrib_all[normal_mask].mean()),
            'disc_score': float(disc_contrib_all[normal_mask].mean()),
            'recon_ratio': float(recon_ratio_all[normal_mask].mean() * 100),
            'disc_ratio': float(disc_ratio_all[normal_mask].mean() * 100),
            'count': int(normal_mask.sum()),
        }
    results['anomaly_type_scores'] = anomaly_type_scores
    return results


# =============================================================================
# Epoch-wise Visualization
# (Self-contained: uses matplotlib + numpy only)
# =============================================================================

def plot_epoch_metrics(epoch_metrics_list, output_dir):
    """Generate epoch-wise point-level metric trend plots (4 PNGs)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    setup_style()

    os.makedirs(output_dir, exist_ok=True)

    epochs = [m['epoch'] for m in epoch_metrics_list]
    if len(epochs) < 2:
        return

    # 1. PRC-AUC Evolution (adaptive + teacher)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [m.get('prc_auc', 0) for m in epoch_metrics_list],
            color='#e74c3c', label='Adaptive', marker='o', markersize=4)
    ax.plot(epochs, [m.get('teacher_prc_auc', 0) for m in epoch_metrics_list],
            color='#3498db', label='Teacher Recon', marker='s', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PRC-AUC')
    ax.set_title('Point-Level PRC-AUC Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'epoch_prc_auc.png'), dpi=150)
    plt.close(fig)

    # 2. F1_T Evolution (adaptive + teacher)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [m.get('f1_t', 0) for m in epoch_metrics_list],
            color='#e74c3c', label='Adaptive F1_T', marker='o', markersize=4)
    ax.plot(epochs, [m.get('teacher_f1_t', 0) for m in epoch_metrics_list],
            color='#3498db', label='Teacher F1_T', marker='s', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1_T')
    ax.set_title('Point-Level F1_T Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'epoch_f1_t.png'), dpi=150)
    plt.close(fig)

    # 3. PA%K F1 Evolution (PA0, PA20, PA50, PA100) — fixed-threshold
    pa_ks = [0, 20, 50, 100]
    pa_colors = ['#95a5a6', '#e74c3c', '#f39c12', '#2ecc71']
    fig, ax = plt.subplots(figsize=(10, 6))
    for k, c in zip(pa_ks, pa_colors):
        ax.plot(epochs, [m.get(f'pa_{k}_f1', 0) for m in epoch_metrics_list],
                color=c, label=f'PA{k}% F1', marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PA%K F1')
    ax.set_title('Point-Level PA%K F1 Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'epoch_pa_k_f1.png'), dpi=150)
    plt.close(fig)

    # 4. PA%K AUC Evolution (F1 best/raw + PRC-AUC) — dedicated
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [m.get('pak_auc_f1', 0) for m in epoch_metrics_list],
            color='#8e44ad', label='PAK AUC F1 (best)', marker='D', markersize=5, linewidth=2)
    ax.plot(epochs, [m.get('pak_auc_f1_raw', 0) for m in epoch_metrics_list],
            color='#8e44ad', label='PAK AUC F1 (raw)', marker='D', markersize=4,
            linewidth=1.5, linestyle='--', alpha=0.6)
    ax.plot(epochs, [m.get('pak_auc_prc_auc', 0) for m in epoch_metrics_list],
            color='#e67e22', label='PAK AUC PRC', marker='^', markersize=5, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PAK AUC')
    ax.set_title('Point-Level PA%K AUC Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'epoch_pak_auc.png'), dpi=150)
    plt.close(fig)

    # 5. Combined summary dashboard (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0][0]
    ax.plot(epochs, [m.get('prc_auc', 0) for m in epoch_metrics_list],
            color='#e74c3c', label='Adaptive', linewidth=1.5)
    ax.plot(epochs, [m.get('teacher_prc_auc', 0) for m in epoch_metrics_list],
            color='#3498db', label='Teacher', linewidth=1.5)
    ax.set_title('PRC-AUC')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0][1]
    ax.plot(epochs, [m.get('f1_t', 0) for m in epoch_metrics_list],
            color='#e74c3c', label='Adaptive', linewidth=1.5)
    ax.plot(epochs, [m.get('teacher_f1_t', 0) for m in epoch_metrics_list],
            color='#3498db', label='Teacher', linewidth=1.5)
    ax.set_title('F1_T')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0][2]
    for k, c in zip(pa_ks, pa_colors):
        ax.plot(epochs, [m.get(f'pa_{k}_f1', 0) for m in epoch_metrics_list],
                color=c, label=f'PA{k}%', linewidth=1.5)
    ax.set_title('PA%K F1')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1][0]
    ax.plot(epochs, [m.get('pak_auc_f1', 0) for m in epoch_metrics_list],
            color='#8e44ad', label='PAK F1', linewidth=2)
    ax.plot(epochs, [m.get('pak_auc_prc_auc', 0) for m in epoch_metrics_list],
            color='#e67e22', label='PAK PRC', linewidth=2)
    ax.set_title('PAK AUC')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1][1]
    ax.plot(epochs, [m.get('disc_snr', 0) for m in epoch_metrics_list],
            color='#9b59b6', marker='o', markersize=4, linewidth=1.5)
    ax.set_title('Disc SNR')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    ax = axes[1][2]
    # Teacher PA%K AUC metrics
    ax.plot(epochs, [m.get('teacher_pak_auc_prc_auc', 0) for m in epoch_metrics_list],
            color='#3498db', label='Teacher PAK PRC', linewidth=2)
    ax.plot(epochs, [m.get('teacher_pak_auc_f1', 0) for m in epoch_metrics_list],
            color='#3498db', label='Teacher PAK F1', linewidth=1.5, linestyle='--')
    ax.set_title('Teacher PAK AUC')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('Epoch', fontsize=9)

    fig.suptitle('Point-Level Training Dynamics', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'epoch_dashboard.png'), dpi=150)
    plt.close(fig)

    # 5. Discriminator metrics (only when D metrics present)
    has_d = any(m.get('d_loss') is not None for m in epoch_metrics_list)
    if has_d:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # D Loss + Accuracy (dual y-axis)
        ax = axes[0]
        ax.plot(epochs, [m.get('d_loss', 0) for m in epoch_metrics_list],
                color='#E65100', lw=2, label='D Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('D Loss', color='#E65100')
        ax.tick_params(axis='y', labelcolor='#E65100')
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(epochs, [m.get('d_real_acc', 0) for m in epoch_metrics_list],
                color='#1565C0', ls='--', lw=1.5, label='Real Acc')
        ax2.plot(epochs, [m.get('d_fake_acc', 0) for m in epoch_metrics_list],
                color='#C62828', ls='--', lw=1.5, label='Fake Acc')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1.05])
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        ax.set_title('D Loss & Accuracy')

        # Adversarial Loss
        ax = axes[1]
        ax.plot(epochs, [m.get('adv_loss', 0) for m in epoch_metrics_list],
                color='#7B1FA2', lw=2, marker='D', ms=3, label='Adv Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Adversarial Loss')
        ax.set_title('Student Adversarial Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Adaptive Lambda
        ax = axes[2]
        ax.plot(epochs, [m.get('adaptive_lambda', 0) for m in epoch_metrics_list],
                color='#00695C', lw=2, marker='*', ms=4, label='lambda_adv')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Adaptive lambda')
        ax.set_title('Adaptive Lambda')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.suptitle('Discriminator Metrics Over Training', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'epoch_discriminator.png'), dpi=150)
        plt.close(fig)

    # 6. Feature Matching loss (only when FM data present)
    has_fm = any(m.get('fm_loss') is not None and m.get('fm_loss', 0) > 0
                 for m in epoch_metrics_list)
    if has_fm:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(epochs, [m.get('fm_loss', 0) for m in epoch_metrics_list],
                color='#00897B', lw=2, marker='o', ms=3, label='FM Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Feature Matching Loss')
        ax.set_title('Feature Matching Loss (cosine distance)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'epoch_fm_loss.png'), dpi=150)
        plt.close(fig)

    # 7. GRL metrics (only when GRL data present)
    has_grl = any(m.get('grl_cls_loss') is not None for m in epoch_metrics_list)
    if has_grl:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # GRL Classifier Loss
        ax = axes[0]
        ax.plot(epochs, [m.get('grl_cls_loss', 0) for m in epoch_metrics_list],
                color='#AD1457', lw=2, label='GRL Cls Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Classifier Loss')
        ax.set_title('GRL Classifier Loss (Focal)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Accuracy breakdown (balanced + anomaly + normal)
        ax = axes[1]
        ax.plot(epochs, [m.get('grl_balanced_acc', 0) for m in epoch_metrics_list],
                color='#1565C0', lw=2, marker='s', ms=3, label='Balanced Acc')
        ax.plot(epochs, [m.get('grl_anomaly_acc', 0) for m in epoch_metrics_list],
                color='#C62828', lw=1.5, ls='--', marker='^', ms=3, label='Anomaly Acc (TPR)')
        ax.plot(epochs, [m.get('grl_normal_acc', 0) for m in epoch_metrics_list],
                color='#2E7D32', lw=1.5, ls='--', marker='v', ms=3, label='Normal Acc (TNR)')
        ax.axhline(y=0.50, color='gray', ls=':', lw=1, alpha=0.5, label='Random (0.50)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0.0, 1.05])
        ax.set_title('GRL Classification Accuracy')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Effective Weight (actual multiplier on GRL loss)
        ax = axes[2]
        ax.plot(epochs, [m.get('grl_effective_weight', m.get('grl_lambda', 0)) for m in epoch_metrics_list],
                color='#6A1B9A', lw=2, marker='D', ms=3, label='Effective Weight')
        ax.plot(epochs, [m.get('grl_lambda', 0) for m in epoch_metrics_list],
                color='#00695C', lw=1.5, ls='--', marker='*', ms=3, alpha=0.6, label='Adaptive Lambda')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight')
        ax.set_title('GRL Loss Weight (actual multiplier)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.suptitle('GRL Metrics Over Training', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'epoch_grl.png'), dpi=150)
        plt.close(fig)

    # 8. SCAD metrics (only when SCAD data present) — mirrors GRL block structure
    has_scad = any(m.get('scad_loss') is not None for m in epoch_metrics_list)
    if has_scad:
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))

        # (A) SCAD Loss
        ax = axes[0]
        ax.plot(epochs, [m.get('scad_loss', 0) for m in epoch_metrics_list],
                color='#D32F2F', lw=2, marker='o', ms=3, label='SCAD Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('SCAD Loss')
        ax.set_title('(A) SCAD Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (B) Cluster Separation (primary learning signal)
        ax = axes[1]
        ax.plot(epochs, [m.get('scad_z_separation', 0) for m in epoch_metrics_list],
                color='#388E3C', lw=2, marker='s', ms=3, label='||z_anom - z_norm||')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Separation')
        ax.set_title('(B) Anom-Norm Separation (higher better)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (C) Cluster Variances (collapse detection)
        ax = axes[2]
        ax.plot(epochs, [m.get('scad_z_anom_var', 0) for m in epoch_metrics_list],
                color='#C62828', lw=1.5, ls='--', marker='^', ms=3, label='Anom Var')
        ax.plot(epochs, [m.get('scad_z_norm_var', 0) for m in epoch_metrics_list],
                color='#1565C0', lw=1.5, ls='--', marker='v', ms=3, label='Norm Var')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cluster Variance')
        ax.set_title('(C) Cluster Var (zero = collapse warning)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (D) Effective Weight (= adaptive λ × ramp × scad_loss_weight)
        ax = axes[3]
        ax.plot(epochs, [m.get('scad_effective_weight', 0) for m in epoch_metrics_list],
                color='#7B1FA2', lw=2, marker='D', ms=3, label='Effective Weight')
        ax.plot(epochs, [m.get('scad_adaptive_lambda', 0) for m in epoch_metrics_list],
                color='#00695C', lw=1.5, ls=':', marker='*', ms=3, alpha=0.7, label='Adaptive λ')
        ax.plot(epochs, [m.get('scad_ramp', 0) for m in epoch_metrics_list],
                color='#F57C00', lw=1.5, ls='-.', marker='x', ms=3, alpha=0.7, label='Ramp')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight')
        ax.set_title('(D) SCAD Effective Weight')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        fig.suptitle('SCAD Metrics Over Training', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'epoch_scad.png'), dpi=150)
        plt.close(fig)


def plot_epoch_feature_stats(epoch_metrics_list, output_dir):
    """Generate per-feature loss evolution plots from epoch_metrics data.

    Produces epoch_feature_disc.png and epoch_feature_recon.png showing
    per-feature mean/max discrepancy and reconstruction loss over training.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    setup_style()

    os.makedirs(output_dir, exist_ok=True)
    epochs = [m['epoch'] for m in epoch_metrics_list]
    if len(epochs) < 2:
        return

    # Check for feature stats — scan all epochs because early teacher_only epochs
    # may have None for _train_feature_disc_mean (discrepancy not computed yet)
    has_train = any(m.get('_train_feature_disc_mean') is not None for m in epoch_metrics_list)
    has_infer = any(m.get('_infer_feature_disc_mean') is not None for m in epoch_metrics_list)
    if not has_train and not has_infer:
        return

    # Determine number of features from first available entry
    for m in epoch_metrics_list:
        ref = m.get('_train_feature_disc_mean') or m.get('_infer_feature_disc_mean')
        if ref is not None:
            n_features = len(ref)
            break
    else:
        return

    cmap = plt.cm.tab10 if n_features <= 10 else plt.cm.tab20

    def _extract_matrix(key):
        """Extract (n_epochs, n_features) matrix from epoch_metrics, filling missing with NaN."""
        mat = np.full((len(epochs), n_features), np.nan)
        for i, m in enumerate(epoch_metrics_list):
            val = m.get(key)
            if val is not None:
                mat[i] = val
        return mat

    # --- Discrepancy feature plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if has_train:
        ax = axes[0]
        train_disc_mean = _extract_matrix('_train_feature_disc_mean')
        for f in range(n_features):
            ax.plot(epochs, train_disc_mean[:, f], color=cmap(f % 20),
                    label=f'F{f}', linewidth=1.2, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Discrepancy')
        ax.set_title('Training: Per-Feature Disc Mean', fontweight='bold')
        ax.legend(fontsize=6, ncol=max(1, n_features // 4), loc='upper right')
        ax.grid(True, alpha=0.3)
    else:
        axes[0].set_visible(False)

    if has_infer:
        ax = axes[1]
        infer_disc_mean = _extract_matrix('_infer_feature_disc_mean')
        for f in range(n_features):
            ax.plot(epochs, infer_disc_mean[:, f], color=cmap(f % 20),
                    label=f'F{f}', linewidth=1.2, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Discrepancy')
        ax.set_title('Inference: Per-Feature Disc Mean', fontweight='bold')
        ax.legend(fontsize=6, ncol=max(1, n_features // 4), loc='upper right')
        ax.grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)

    fig.suptitle('Per-Feature Discrepancy Over Training', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'epoch_feature_disc.png'), dpi=150)
    plt.close(fig)

    # --- Reconstruction feature plot ---
    if has_train:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax = axes[0]
        train_recon_mean = _extract_matrix('_train_feature_recon_mean')
        for f in range(n_features):
            ax.plot(epochs, train_recon_mean[:, f], color=cmap(f % 20),
                    label=f'F{f}', linewidth=1.2, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Recon Error')
        ax.set_title('Training: Per-Feature Recon Mean', fontweight='bold')
        ax.legend(fontsize=6, ncol=max(1, n_features // 4), loc='upper right')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        train_recon_max = _extract_matrix('_train_feature_recon_max')
        for f in range(n_features):
            ax.plot(epochs, train_recon_max[:, f], color=cmap(f % 20),
                    label=f'F{f}', linewidth=1.2, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Max Recon Error')
        ax.set_title('Training: Per-Feature Recon Max', fontweight='bold')
        ax.legend(fontsize=6, ncol=max(1, n_features // 4), loc='upper right')
        ax.grid(True, alpha=0.3)

        fig.suptitle('Per-Feature Reconstruction Error Over Training', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'epoch_feature_recon.png'), dpi=150)
        plt.close(fig)


# =============================================================================
# excl22 viz data filtering (called before spawning excl22 worker)
# =============================================================================

def _filter_excl22_viz_data(pred_data, detailed_data, config, region_start, region_end):
    """Filter pred_data/detailed_data to exclude region 22 for SWaT excl22 visualization.

    Removes windows overlapping [region_start, region_end), re-aggregates point-level
    scores, and removes region 22 from point-level arrays. Returns deep-filtered copies.
    """
    import copy
    from mae_anomaly.evaluator import aggregate_patch_scores_to_point_level

    pred = copy.copy(pred_data)  # shallow copy, then replace arrays
    det = copy.copy(detailed_data) if detailed_data is not None else None

    seq_len = config.seq_length
    num_patches = pred.get('num_patches', config.num_patches)
    patch_size = pred.get('patch_size', config.patch_size)

    viz_ws = pred.get('window_start_indices')
    if viz_ws is None:
        return pred, det

    # Windows that DON'T overlap region 22
    keep_win = ~((viz_ws < region_end) & (viz_ws + seq_len > region_start))
    keep_patch = np.repeat(keep_win, num_patches)
    new_n_windows = int(keep_win.sum())

    # Filter patch-level (flattened: n_windows * num_patches)
    for key in ['patch_scores', 'patch_labels', 'recon_errors',
                'student_errors', 'discrepancies', 'sample_types']:
        if key in pred:
            pred[key] = pred[key][keep_patch]

    # Filter 2D patch arrays
    for key in ['patch_recon_2d', 'patch_disc_2d']:
        if key in pred:
            pred[key] = pred[key][keep_win]

    pred['window_start_indices'] = viz_ws[keep_win]
    pred['n_windows'] = new_n_windows

    # Re-aggregate point-level scores (excluding region 22 windows)
    total_len = pred.get('total_length', len(pred.get('point_labels', [])))
    for score_key, point_key in [
        ('patch_scores', 'point_scores'), ('recon_errors', 'point_recon'),
        ('discrepancies', 'point_disc'), ('student_errors', 'point_student'),
    ]:
        if score_key in pred and point_key in pred:
            scores_2d = pred[score_key].reshape(new_n_windows, num_patches)
            pt_agg, _ = aggregate_patch_scores_to_point_level(
                scores_2d, pred['window_start_indices'], seq_len,
                patch_size, num_patches, total_len, method='mean'
            )
            pred[point_key] = np.nan_to_num(pt_agg, nan=0.0)

    # Remove region 22 range from point-level arrays
    keep_pts = np.ones(total_len, dtype=bool)
    keep_pts[region_start:region_end] = False
    for key in ['point_scores', 'point_labels', 'point_recon',
                'point_disc', 'point_student']:
        if key in pred:
            pred[key] = pred[key][keep_pts]
    pred['scores'] = pred.get('point_scores', pred.get('scores'))
    pred['labels'] = pred.get('point_labels', pred.get('labels'))
    pred['total_length'] = int(keep_pts.sum())

    # Filter detailed_data (window-level)
    if det is not None:
        for key in ['discrepancies', 'labels', 'point_labels', 'originals',
                    'teacher_recons', 'student_recons', 'sample_types']:
            if key in det:
                det[key] = det[key][keep_win]

    n_removed = int((~keep_win).sum())
    print(f"  excl22 filter: removed {n_removed}/{len(keep_win)} viz windows "
          f"(region [{region_start}, {region_end})), {pred['total_length']} points remain")

    return pred, det, keep_win


# =============================================================================
# Background CPU Eval+Viz Worker
# Uses: Evaluator, compute_loss_statistics (from run_ablation), BestModelVisualizer
# =============================================================================

def _cpu_eval_viz_worker(exp_name, exp_dir, config_dict, signals, point_labels,
                         anomaly_regions_ser, history, train_ratio, timing,
                         patch_scores, pred_data, detailed_data,
                         progress_info="", window_start_indices=None,
                         swat_eval_mode=None):
    """Background worker: CPU-only eval metrics + viz using precomputed GPU data.

    Combines evaluation (via Evaluator.set_precomputed_patch_scores) and
    visualization (via BestModelVisualizer.generate_all) into a single
    background process.

    Args:
        swat_eval_mode: None for full eval, 'excl22' for SWaT excl-region22 eval.

    Args:
        timing: dict with keys train_time, train_per_epoch, num_epochs,
                inference_time, gpu_total
        window_start_indices: actual test dataset window start indices (avoids
                recomputation mismatch)
    """
    # Lazy imports for subprocess
    import sys
    sys.path.insert(0, PROJECT_ROOT)
    from mae_anomaly import Config, set_seed
    from mae_anomaly.dataset_sliding import SlidingWindowDataset, AnomalyRegion
    from mae_anomaly.evaluator import Evaluator, DatasetMetadata
    from mae_anomaly.visualization import setup_style, BestModelVisualizer
    from scripts.ablation.run_ablation import compute_loss_statistics

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # CPU throttling
    os.nice(19)
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    os.environ['NUMEXPR_NUM_THREADS'] = '4'
    torch.set_num_threads(4)
    try:
        os.sched_setaffinity(0, set(range(16, 24)))
    except Exception:
        pass
    warnings.filterwarnings('ignore')

    print(f"  [{exp_name}] Background CPU eval+viz started (PID={os.getpid()})", flush=True)

    # Reconstruct config
    config = Config()
    for k, v in config_dict.items():
        if hasattr(config, k):
            setattr(config, k, v)
    config.device = 'cpu'

    # Reconstruct anomaly_regions (global coordinates)
    anomaly_regions_global = [
        AnomalyRegion(start=r['start'], end=r['end'], anomaly_type=r['anomaly_type'])
        for r in anomaly_regions_ser
    ]

    # Create lightweight DatasetMetadata (avoid ~2-3 GB memory waste from full dataset)
    total_length = len(signals)
    train_end = int(total_length * train_ratio)
    test_point_labels = point_labels[train_end:]

    # Adjust anomaly_regions to LOCAL test coordinates (matching SlidingWindowDataset behavior)
    test_anomaly_regions = [
        AnomalyRegion(start=r.start - train_end, end=r.end - train_end, anomaly_type=r.anomaly_type)
        for r in anomaly_regions_global
        if r.start >= train_end
    ]

    # Use actual window_start_indices from test dataset (avoids off-by-one from manual computation)
    if window_start_indices is None:
        test_length = len(test_point_labels)
        stride = config.sliding_window_test_stride
        window_size = config.seq_length
        n_windows = max(0, (test_length - window_size) // stride + 1)
        window_start_indices = [i * stride for i in range(n_windows)]

    dataset_meta = DatasetMetadata(
        point_labels=test_point_labels,
        window_start_indices=window_start_indices,
        anomaly_regions=test_anomaly_regions,
    )

    # Eval using pre-cached GPU scores
    eval_start = time.time()
    dummy_model = SelfDistilledMAEMultivariate(config)
    dummy_model.eval()
    evaluator = Evaluator(dummy_model, config, test_loader=None, test_dataset=dataset_meta)
    evaluator.set_precomputed_patch_scores(
        recon_patches=patch_scores['recon'],
        disc_patches=patch_scores['disc'],
        student_recon_patches=patch_scores['student_recon'],
        labels=patch_scores['labels'],
        sample_types=patch_scores['sample_types'],
        anomaly_types=patch_scores['anomaly_types'],
    )
    print(f"  [{exp_name}] Computing metrics...", flush=True)
    metrics = evaluator.evaluate()
    eval_time = time.time() - eval_start

    # Per-score-type metrics
    disc_metrics = evaluator.evaluate_by_score_type('disc')
    teacher_recon_metrics = evaluator.evaluate_by_score_type('teacher_recon')
    student_recon_metrics = evaluator.evaluate_by_score_type('student_recon')

    print(f"  [{exp_name}] {progress_info} Eval done ({eval_time:.0f}s): "
          f"PRC={metrics.get('prc_auc',0):.4f} "
          f"PAK_AUC_F1={metrics.get('pak_auc_f1',0):.4f} PAK_AUC_PRC={metrics.get('pak_auc_prc_auc',0):.4f} "
          f"F1_T={metrics.get('f1_t',0):.4f}", flush=True)

    # d_SNR will be available after compute_loss_statistics (printed in COMPLETE)

    # Save config + history
    with open(os.path.join(exp_dir, 'best_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    with open(os.path.join(exp_dir, 'training_histories.json'), 'w') as f:
        json.dump({'0': history}, f, indent=2)

    # Detailed losses + CSV
    detailed_losses = evaluator.compute_detailed_losses()
    atype_names = np.where(
        detailed_losses['anomaly_types'].astype(int) == 0, 'normal', 'attack'
    )
    # Subsample detailed data to 10K per anomaly type for efficiency
    from mae_anomaly.utils import subsample_by_category
    detailed_df = pd.DataFrame({
        'reconstruction_loss': detailed_losses['reconstruction_loss'],
        'discrepancy_loss': detailed_losses['discrepancy_loss'],
        'total_loss': detailed_losses['total_loss'],
        'label': detailed_losses['labels'],
        'sample_type': detailed_losses['sample_types'],
        'anomaly_type': detailed_losses['anomaly_types'],
        'anomaly_type_name': atype_names
    })
    detailed_df_sampled = subsample_by_category(detailed_df, max_samples_per_category=10000)
    detailed_df_sampled.to_csv(os.path.join(exp_dir, 'best_model_detailed.csv'), index=False)

    loss_stats = compute_loss_statistics(detailed_losses)
    anomaly_type_metrics = evaluator.get_performance_by_anomaly_type()
    with open(os.path.join(exp_dir, 'anomaly_type_metrics.json'), 'w') as f:
        json.dump(anomaly_type_metrics, f, indent=2)

    # SWaT excl22: compute excl_region22 metrics
    excl22_metrics = None
    excl22_teacher_metrics = None
    excl22_info = None
    is_swat = 'SWaT' in exp_name
    if is_swat and test_anomaly_regions:
        try:
            from mae_anomaly.evaluator import (
                find_swat_largest_region, compute_metrics_with_exclusion
            )
            largest_region = find_swat_largest_region(test_anomaly_regions)
            if largest_region is not None:
                epoch_scores_dir = os.path.join(exp_dir, 'epoch_scores')

                if swat_eval_mode == 'excl22':
                    # excl22 worker: scan ALL epochs for excl22-specific metrics + best epoch
                    import glob as glob_mod
                    score_files = sorted(glob_mod.glob(
                        os.path.join(epoch_scores_dir, 'epoch_*_scores.npz')
                    ))
                    best_excl22_epoch = timing.get('best_epoch', 0)
                    best_excl22_pak_f1 = -1.0
                    best_excl22_em = None
                    best_excl22_teacher_em = None
                    excl22_epoch_metrics_list = []

                    # Load copied epoch_metrics for disc_snr and D metrics (non-exclusion-dependent)
                    em_path = os.path.join(exp_dir, 'epoch_metrics.json')
                    full_epoch_data = {}
                    eval_interval = 5
                    if os.path.exists(em_path):
                        with open(em_path) as f:
                            em_json = json.load(f)
                        eval_interval = em_json.get('eval_interval', 5)
                        for fem in em_json.get('epochs', []):
                            full_epoch_data[fem['epoch']] = fem

                    for sf in score_files:
                        try:
                            ep_num = int(os.path.basename(sf).split('_')[1])
                            sd = np.load(sf)
                            a_scores = sd['adaptive_score']
                            t_scores = sd['teacher_recon_error']
                            ml = min(len(a_scores), len(test_point_labels))
                            em = compute_metrics_with_exclusion(
                                a_scores[:ml], test_point_labels[:ml],
                                test_anomaly_regions, largest_region
                            )
                            teacher_em = compute_metrics_with_exclusion(
                                t_scores[:ml], test_point_labels[:ml],
                                test_anomaly_regions, largest_region
                            )
                            pak_f1 = em.get('pak_auc_f1', 0)
                            if pak_f1 > best_excl22_pak_f1:
                                best_excl22_pak_f1 = pak_f1
                                best_excl22_epoch = ep_num
                                best_excl22_em = em
                                best_excl22_teacher_em = teacher_em

                            entry = {'epoch': ep_num}
                            entry.update(em)
                            entry['teacher_prc_auc'] = teacher_em.get('prc_auc', 0)
                            entry['teacher_f1_t'] = teacher_em.get('f1_t', 0)
                            # Teacher PA%K AUC metrics for excl22
                            for m in ['prc_auc', 'roc_auc', 'f1', 'f1_t', 'precision', 'recall',
                                      'f1_raw', 'f1_t_raw', 'precision_raw', 'recall_raw']:
                                entry[f'teacher_pak_auc_{m}'] = teacher_em.get(f'pak_auc_{m}', 0)
                            entry['teacher_pak_auc_prc_auc'] = teacher_em.get('pak_auc_prc_auc', 0)
                            if ep_num in full_epoch_data:
                                for key in ['disc_snr', 'd_loss', 'd_real_acc', 'd_fake_acc',
                                            'adv_loss', 'adaptive_lambda',
                                            'grl_cls_loss', 'grl_balanced_acc', 'grl_anomaly_acc',
                                            'grl_normal_acc', 'grl_lambda', 'grl_effective_weight']:
                                    if key in full_epoch_data[ep_num]:
                                        entry[key] = full_epoch_data[ep_num][key]
                            excl22_epoch_metrics_list.append(entry)
                        except Exception:
                            continue

                    excl22_epoch_metrics_list.sort(key=lambda x: x['epoch'])

                    # Override best_epoch in timing
                    timing['best_epoch'] = best_excl22_epoch
                    timing['best_epoch_metric'] = 'excl22_pak_auc_f1'
                    timing['best_epoch_score'] = best_excl22_pak_f1
                    print(f"  [{exp_name}] excl22 best epoch: {best_excl22_epoch} "
                          f"(pak_f1={best_excl22_pak_f1:.4f}, scanned {len(score_files)} epochs)",
                          flush=True)

                    # Save excl22-specific epoch_metrics.json (overwrite copied full)
                    with open(em_path, 'w') as f:
                        json.dump({'eval_interval': eval_interval,
                                   'epochs': excl22_epoch_metrics_list}, f, indent=2)
                    excl22_epoch_viz_dir = os.path.join(exp_dir, 'visualization', 'epoch_metrics')
                    plot_epoch_metrics(excl22_epoch_metrics_list, excl22_epoch_viz_dir)
                    print(f"  [{exp_name}] excl22 epoch_metrics viz saved: {excl22_epoch_viz_dir}",
                          flush=True)

                    # Use scan results directly (no redundant re-computation)
                    excl22_metrics = best_excl22_em
                    excl22_teacher_metrics = best_excl22_teacher_em
                    if best_excl22_em is not None:
                        excl_len = largest_region.end - largest_region.start
                        excl22_info = {
                            'region_start': largest_region.start,
                            'region_end': largest_region.end,
                            'region_length': excl_len,
                            'test_length': len(test_point_labels),
                            'region_pct_of_test': excl_len / len(test_point_labels) * 100,
                        }
                        print(f"  [{exp_name}] excl_region22: "
                              f"PRC={excl22_metrics.get('prc_auc', 0):.4f} "
                              f"PAK_F1={excl22_metrics.get('pak_auc_f1', 0):.4f} "
                              f"F1_T={excl22_metrics.get('f1_t', 0):.4f}", flush=True)

                else:
                    # Full worker: compute excl22 reference metrics for best epoch only
                    best_epoch = timing.get('best_epoch', 0)
                    scores_path = os.path.join(
                        epoch_scores_dir, f'epoch_{best_epoch:03d}_scores.npz'
                    )
                    if os.path.exists(scores_path):
                        scores_data = np.load(scores_path)
                        adaptive_scores = scores_data['adaptive_score']
                        teacher_scores = scores_data['teacher_recon_error']

                        min_len = min(len(adaptive_scores), len(test_point_labels))
                        excl22_metrics = compute_metrics_with_exclusion(
                            adaptive_scores[:min_len], test_point_labels[:min_len],
                            test_anomaly_regions, largest_region
                        )
                        excl22_teacher_metrics = compute_metrics_with_exclusion(
                            teacher_scores[:min_len], test_point_labels[:min_len],
                            test_anomaly_regions, largest_region
                        )
                        excl_len = largest_region.end - largest_region.start
                        excl22_info = {
                            'region_start': largest_region.start,
                            'region_end': largest_region.end,
                            'region_length': excl_len,
                            'test_length': len(test_point_labels),
                            'region_pct_of_test': excl_len / len(test_point_labels) * 100,
                        }
                        print(f"  [{exp_name}] excl_region22: "
                              f"PRC={excl22_metrics.get('prc_auc', 0):.4f} "
                              f"PAK_F1={excl22_metrics.get('pak_auc_f1', 0):.4f} "
                              f"F1_T={excl22_metrics.get('f1_t', 0):.4f}", flush=True)
                    else:
                        print(f"  [{exp_name}] excl_region22: scores file not found for ep{best_epoch}")
        except Exception as e:
            print(f"  [{exp_name}] excl_region22 error: {e}")

    # Experiment metadata (timing includes all phases)
    # For excl22 worker: excl22 metrics become primary, full metrics stored as reference
    if swat_eval_mode == 'excl22' and excl22_metrics is not None:
        primary_metrics = excl22_metrics
    else:
        primary_metrics = metrics

    # For excl22 worker: teacher_recon_metrics should also use excl22 version
    # (otherwise pa_k_auc_summary.png shows full teacher metrics instead of excl22)
    if swat_eval_mode == 'excl22' and excl22_teacher_metrics is not None:
        primary_teacher_metrics = excl22_teacher_metrics
    else:
        primary_teacher_metrics = teacher_recon_metrics

    metadata = {
        'experiment_name': exp_name,
        'scoring_mode': 'adaptive',
        'swat_eval_mode': swat_eval_mode,
        'timing': {
            **timing,
            'cpu_eval_time': eval_time,
        },
        'metrics': primary_metrics,
        'disc_metrics': disc_metrics,
        'teacher_recon_metrics': primary_teacher_metrics,
        'student_recon_metrics': student_recon_metrics,
        'config': config_dict,
        'loss_stats': loss_stats,
    }
    # Always include both full and excl22 metrics when available
    if excl22_metrics is not None:
        metadata['metrics_excl_region22'] = excl22_metrics
        metadata['teacher_recon_metrics_excl_region22'] = excl22_teacher_metrics
        metadata['excl_region22_info'] = excl22_info
    if swat_eval_mode == 'excl22':
        metadata['metrics_full'] = metrics  # Store full metrics as reference
        metadata['teacher_recon_metrics_full'] = teacher_recon_metrics

    with open(os.path.join(exp_dir, 'experiment_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=float)

    # Visualization via BestModelVisualizer.generate_all()
    viz_start = time.time()
    setup_style()
    viz_dir = os.path.join(exp_dir, 'visualization', 'best_model')
    os.makedirs(viz_dir, exist_ok=True)

    vis = BestModelVisualizer(
        pred_data=pred_data,
        detailed_data=detailed_data,
        config=config,
        output_dir=viz_dir,
    )
    # Attach per-feature discrepancy data for feature visualizations
    vis.disc_per_feature = patch_scores.get('disc_per_feature')  # (n_windows, F) or None
    vis.generate_all(experiment_dir=exp_dir, history=history)
    plt.close('all')

    viz_time = time.time() - viz_start

    # Update metadata with viz_time and total
    total_time = timing.get('gpu_total', 0) + eval_time + viz_time
    metadata['timing']['cpu_viz_time'] = viz_time
    metadata['timing']['total_time'] = total_time
    with open(os.path.join(exp_dir, 'experiment_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=float)

    pt = timing.get('pure_train_time', timing.get('train_time', 0))
    tpe = timing.get('train_per_epoch', 0)
    be = timing.get('best_epoch', '?')
    # Training losses from history (last epoch)
    t_loss = history['train_rec_loss'][-1] if history and history.get('train_rec_loss') else 0
    s_loss = history['train_disc_loss'][-1] if history and history.get('train_disc_loss') else 0
    d_loss_str = ""
    if history and history.get('train_d_loss'):
        d_loss_str = f" d_loss={history['train_d_loss'][-1]:.4f}"
    print(f"  [{exp_name}] {progress_info} COMPLETE ({total_time:.0f}s): "
          f"PRC={primary_metrics.get('prc_auc',0):.4f} "
          f"PAK_AUC_F1={primary_metrics.get('pak_auc_f1',0):.4f} PAK_AUC_PRC={primary_metrics.get('pak_auc_prc_auc',0):.4f} "
          f"F1_T={primary_metrics.get('f1_t',0):.4f} | "
          f"d_SNR={loss_stats.get('disc_SNR',0):.4f} t_loss={t_loss:.4f} s_loss={s_loss:.4f}{d_loss_str} | "
          f"best_ep={be} | "
          f"Time: train={pt:.0f}s({tpe:.1f}s/ep) "
          f"infer={timing.get('inference_time',0):.0f}s eval={eval_time:.0f}s viz={viz_time:.0f}s", flush=True)


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_base_experiment(dataset_def, config_preset, results_base, progress_info="",
                        save_weights=False):
    """Run a single base experiment with epoch-wise monitoring."""
    key = dataset_def['key']
    train_stride = dataset_def['train_stride']
    is_normal50 = dataset_def['normal50']
    results_dir = os.path.join(results_base, dataset_def['results_subdir'])

    print(f"\n{'='*80}")
    print(f"Base Experiment: {key}")
    print(f"  Train stride: {train_stride}")
    print(f"  Normal50: {is_normal50}")
    print(f"  Output: {results_dir}")
    print(f"  Memory: {mem_status()}")
    print(f"{'='*80}")

    # Load data via DATASET_LOADERS registry
    print("Loading data...")
    loader_fn = get_dataset_loader(dataset_def['loader'])
    data = loader_fn()

    # Handle different return signatures
    data_info = {}
    if len(data) == 6:
        signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info = data
    elif len(data) == 5:
        signals, point_labels, anomaly_regions, feature_names, train_ratio = data
    else:
        signals, point_labels, anomaly_regions, feature_names = data
        train_ratio = 0.5

    # Extract run_boundaries for datasets with non-contiguous blocks (e.g., SMD block split)
    run_boundaries = data_info.get('run_boundaries') if data_info else None

    print(f"  Signals: {signals.shape}")
    print(f"  Labels: normal={np.sum(point_labels==0):,}, anomaly={np.sum(point_labels==1):,}")
    print(f"  Train ratio: {train_ratio:.4f}")
    if run_boundaries:
        print(f"  Run boundaries: {len(run_boundaries)} (windows will not cross block boundaries)")

    # Apply normal50 noise if needed
    noisy_labels = None
    if is_normal50:
        noisy_labels = apply_normal50_noise(point_labels, anomaly_regions, train_ratio)
        noisy_train_end = int(len(point_labels) * train_ratio)
        print(f"  Normal50 noise applied: train anomaly ratio {noisy_labels[:noisy_train_end].mean():.2%}")

    # Create config from preset + dataset-specific overrides
    overrides = dict(config_preset['overrides'])
    overrides['sliding_window_stride'] = train_stride
    overrides['sliding_window_train_ratio'] = train_ratio

    # Resolve dynamic d_model: select from [128,192,256,384,512] based on num_features
    num_features = signals.shape[1]
    if overrides.get('d_model') == 'dynamic':
        overrides['d_model'] = resolve_dynamic_d_model(num_features, overrides['patch_size'])
        print(f"  Dynamic d_model: {overrides['d_model']} (raw={overrides['patch_size']*num_features}, "
              f"dim_ff={overrides['d_model']*4})")

    config = make_config(overrides)
    config.num_features = num_features
    config.device = 'cuda'

    # Experiment directory = results_subdir directly (no timestamp subdirectory)
    # SWaT dual-eval: split into _full (training + full eval) and _excl22 (excl region22 eval)
    is_swat_dual = 'SWaT' in key and 'swap' not in key
    if is_swat_dual:
        exp_dir = results_dir + '_full'
        exp_dir_excl22 = results_dir + '_excl22'
        print(f"  SWaT dual-eval: {exp_dir} + {exp_dir_excl22}")
    else:
        exp_dir = results_dir
        exp_dir_excl22 = None
    os.makedirs(exp_dir, exist_ok=True)
    print(f"  Experiment dir: {exp_dir}")

    # Create datasets
    print("Creating datasets...")
    set_seed(config.random_seed)

    test_stride = config.sliding_window_test_stride
    test_dataset = SlidingWindowDataset(
        signals=signals, point_labels=point_labels, anomaly_regions=anomaly_regions,
        window_size=config.seq_length, stride=test_stride, mask_last_n=config.patch_size,
        split='test', train_ratio=train_ratio, seed=config.random_seed,
        run_boundaries=run_boundaries,
        normalize_mode=config.normalize_mode,
        minmax_range=getattr(config, 'minmax_range', '0_1'),
        minmax_clamp_min=getattr(config, 'minmax_clamp_min', None),
        minmax_clamp_max=getattr(config, 'minmax_clamp_max', None),
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    print(f"  Test: {len(test_dataset)} windows (stride={test_stride})")

    if is_normal50:
        train_dataset = NoisyLabelSlidingWindowDataset(
            signals=signals,
            point_labels=point_labels,
            noisy_point_labels=noisy_labels,
            anomaly_regions=anomaly_regions,
            window_size=config.seq_length,
            stride=train_stride,
            mask_last_n=config.patch_size,
            split='train',
            train_ratio=train_ratio,
            seed=config.random_seed,
            run_boundaries=run_boundaries,
            normalize_mode=config.normalize_mode,
            minmax_range=getattr(config, 'minmax_range', '0_1'),
            minmax_clamp_min=getattr(config, 'minmax_clamp_min', None),
            minmax_clamp_max=getattr(config, 'minmax_clamp_max', None),
        )
    else:
        train_dataset = SlidingWindowDataset(
            signals=signals, point_labels=point_labels, anomaly_regions=anomaly_regions,
            window_size=config.seq_length, stride=train_stride,
            mask_last_n=config.patch_size, split='train', train_ratio=train_ratio,
            seed=config.random_seed,
            run_boundaries=run_boundaries,
            normalize_mode=config.normalize_mode,
            minmax_range=getattr(config, 'minmax_range', '0_1'),
            minmax_clamp_min=getattr(config, 'minmax_clamp_min', None),
            minmax_clamp_max=getattr(config, 'minmax_clamp_max', None),
        )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    # Compute GRL pos_weight from actual dataset anomaly ratio (patch-level)
    # GRL classifier uses patch_has_anomaly (1 if any anomaly point in patch), not point-level labels.
    # Compute patch-level anomaly ratio for accurate pos_weight.
    _total_patches = 0
    _anomaly_patches = 0
    _total_points = 0
    _anomaly_points = 0
    _num_patches = config.seq_length // config.patch_size
    for _i in range(len(train_dataset)):
        _, _, _pl = train_dataset[_i][:3]
        _total_points += _pl.numel()
        _anomaly_points += _pl.sum().item()
        # Patch-level: reshape to (num_patches, patch_size), check if any anomaly per patch
        _pl_patches = _pl.reshape(_num_patches, config.patch_size)
        _patch_has_anomaly = (_pl_patches.sum(dim=1) > 0).float()
        _total_patches += _num_patches
        _anomaly_patches += _patch_has_anomaly.sum().item()
    _point_ratio = _anomaly_points / (_total_points + 1e-8)
    _patch_ratio = _anomaly_patches / (_total_patches + 1e-8)
    _patch_ratio = max(_patch_ratio, 0.001)  # clamp to avoid div-by-zero
    config.grl_pos_weight = (1.0 - _patch_ratio) / _patch_ratio
    print(f"  Train: {len(train_dataset)} windows (stride={train_stride}), "
          f"point_anomaly_ratio={_point_ratio:.4f}, patch_anomaly_ratio={_patch_ratio:.4f}, "
          f"grl_pos_weight={config.grl_pos_weight:.1f}")

    # ========== GPU Training with Epoch Callbacks ==========
    model = SelfDistilledMAEMultivariate(config)
    print(f"Training on GPU... ({mem_status()})")

    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    epoch_scores_dir = os.path.join(exp_dir, 'epoch_scores')
    os.makedirs(epoch_scores_dir, exist_ok=True)

    epoch_metrics_list = []
    callback_infer_time = 0.0  # GPU inference time (blocks training)
    import queue as _queue_mod
    _eval_queue = _queue_mod.Queue()   # thread-safe result queue
    _async_eval_thread = [None]        # [thread] — for join before checkpoint ops
    _best_epoch_metric_key = config.best_epoch_metric  # e.g. 'pak_auc_f1'
    _best_ckpt_score = [0.0]  # best score seen so far (mutable for nonlocal)

    def _process_eval_result(cb_metrics):
        """Process a completed eval result: update metrics list, best checkpoint, print."""
        epoch_metrics_list.append(cb_metrics)
        prc = cb_metrics.get('prc_auc', 0)
        f1t = cb_metrics.get('f1_t', 0)
        d_snr = cb_metrics.get('disc_snr', 0)
        pak_f1 = cb_metrics.get('pak_auc_f1', 0)
        pak_prc = cb_metrics.get('pak_auc_prc_auc', 0)
        i_t = cb_metrics.get('_inference_time', 0)
        e_t = cb_metrics.get('_eval_time', 0)
        ep = cb_metrics.get('epoch', 0)
        # Update best checkpoint if best_epoch_metric improved
        epoch_score = cb_metrics.get(_best_epoch_metric_key, 0)
        is_best = epoch_score > _best_ckpt_score[0]
        best_marker = ""
        if is_best:
            _best_ckpt_score[0] = epoch_score
            latest_path = os.path.join(checkpoints_dir, 'latest_checkpoint.pt')
            best_path = os.path.join(checkpoints_dir, 'best_checkpoint.pt')
            if os.path.exists(latest_path):
                shutil.copy2(latest_path, best_path)
            best_marker = " ★"

        # Training losses from history
        h = _trainer_ref[0].history if _trainer_ref[0] is not None else {}
        idx = ep - 1  # epoch 1-indexed, history 0-indexed
        t_loss = h['train_rec_loss'][idx] if idx < len(h.get('train_rec_loss', [])) else 0
        s_loss = h['train_disc_loss'][idx] if idx < len(h.get('train_disc_loss', [])) else 0
        d_loss_str = ""
        if idx < len(h.get('train_d_loss', [])):
            d_loss_str = f" d_loss={h['train_d_loss'][idx]:.4f}"
        tqdm.write(f"  [Epoch {ep:>2}] PRC={prc:.4f} PAK_F1={pak_f1:.4f} PAK_PRC={pak_prc:.4f} F1_T={f1t:.4f} "
                   f"d_SNR={d_snr:.4f} | t_loss={t_loss:.4f} s_loss={s_loss:.4f}{d_loss_str} "
                   f"(infer={i_t:.0f}s eval={e_t:.0f}s) [async]{best_marker}")

    def _collect_async_eval(blocking=False):
        """Drain eval queue. If blocking=True, wait for pending thread first."""
        if blocking and _async_eval_thread[0] is not None:
            _async_eval_thread[0].join()
            _async_eval_thread[0] = None
        # Non-blocking drain: process all available results
        while True:
            try:
                cb_metrics = _eval_queue.get_nowait()
                _process_eval_result(cb_metrics)
            except _queue_mod.Empty:
                break

    def _run_bg_all(eval_data, ep, ckpt_data, prev_thread):
        """Background thread: join prev → drain prev results → save checkpoint → CPU eval.

        Thread chaining ensures serial execution: each thread joins the previous one
        before processing, so checkpoint file order and epoch_metrics_list order are guaranteed.
        GPU is free to train while this runs on CPU.
        """
        try:
            # A. Wait for previous bg thread to finish (CPU waits, GPU trains)
            if prev_thread is not None:
                prev_thread.join()

            # B. Drain previous eval results from queue
            #    _process_eval_result may copy latest→best checkpoint, so this MUST
            #    complete before we overwrite latest_checkpoint.pt in step C.
            while True:
                try:
                    cb_metrics = _eval_queue.get_nowait()
                    _process_eval_result(cb_metrics)
                except _queue_mod.Empty:
                    break

            # C. Save this epoch's checkpoint (safe: prev drain done)
            torch.save(ckpt_data, os.path.join(checkpoints_dir, 'latest_checkpoint.pt'))

            # D. Run CPU eval (puts result into _eval_queue)
            _run_cpu_eval(eval_data, ep)

        except Exception as e:
            tqdm.write(f"  [Epoch {ep:>2}] BG ALL ERROR: {e}")

    def _run_cpu_eval(eval_data, ep):
        """Background thread target: CPU-only eval + store result + save epoch scores."""
        try:
            cb_metrics = compute_epoch_test_eval(
                eval_data, config, test_loader, test_dataset=test_dataset
            )
            cb_metrics['epoch'] = ep

            # Per-feature inference stats (compact: mean/max over windows → (F,))
            dpf = eval_data.get('disc_per_feature')
            if dpf is not None:
                cb_metrics['_infer_feature_disc_mean'] = dpf.mean(axis=0).tolist()
                cb_metrics['_infer_feature_disc_max'] = dpf.max(axis=0).tolist()

            # Attach D metrics from trainer history (if discriminator active)
            trainer = _trainer_ref[0]
            if trainer is not None and trainer.use_discriminator:
                h = trainer.history
                idx = ep - 1  # epoch is 1-indexed, history is 0-indexed
                if idx < len(h.get('train_d_loss', [])):
                    cb_metrics['d_loss'] = h['train_d_loss'][idx]
                    cb_metrics['d_real_acc'] = h['train_d_real_acc'][idx]
                    cb_metrics['d_fake_acc'] = h['train_d_fake_acc'][idx]
                    cb_metrics['adv_loss'] = h['train_adv_loss'][idx]
                    cb_metrics['adaptive_lambda'] = h['train_adaptive_lambda'][idx]

            # Attach FM loss from trainer history
            if trainer is not None:
                h = trainer.history
                idx = ep - 1
                if idx < len(h.get('train_fm_loss', [])):
                    cb_metrics['fm_loss'] = h['train_fm_loss'][idx]

            # Attach GRL metrics from trainer history (if GRL active)
            if trainer is not None and getattr(trainer.config, 'use_grl', False):
                h = trainer.history
                idx = ep - 1
                if idx < len(h.get('train_grl_cls_loss', [])):
                    cb_metrics['grl_cls_loss'] = h['train_grl_cls_loss'][idx]
                    cb_metrics['grl_balanced_acc'] = h['train_grl_balanced_acc'][idx]
                    cb_metrics['grl_anomaly_acc'] = h['train_grl_anomaly_acc'][idx] if idx < len(h.get('train_grl_anomaly_acc', [])) else 0.0
                    cb_metrics['grl_normal_acc'] = h['train_grl_normal_acc'][idx] if idx < len(h.get('train_grl_normal_acc', [])) else 0.0
                    cb_metrics['grl_lambda'] = h['train_grl_lambda'][idx]
                    cb_metrics['grl_effective_weight'] = h['train_grl_effective_weight'][idx] if idx < len(h.get('train_grl_effective_weight', [])) else 0.0

            # Attach SCAD metrics from trainer history (if SCAD active) — mirrors GRL pattern
            if trainer is not None and getattr(trainer.config, 'use_scad', False):
                h = trainer.history
                idx = ep - 1
                if idx < len(h.get('train_scad_loss', [])):
                    cb_metrics['scad_loss']             = h['train_scad_loss'][idx]
                    cb_metrics['scad_n_anom']           = h['train_scad_n_anom'][idx] if idx < len(h.get('train_scad_n_anom', [])) else 0
                    cb_metrics['scad_n_norm']           = h['train_scad_n_norm'][idx] if idx < len(h.get('train_scad_n_norm', [])) else 0
                    cb_metrics['scad_z_separation']     = h['train_scad_z_separation'][idx] if idx < len(h.get('train_scad_z_separation', [])) else 0.0
                    cb_metrics['scad_z_anom_var']       = h['train_scad_z_anom_var'][idx] if idx < len(h.get('train_scad_z_anom_var', [])) else 0.0
                    cb_metrics['scad_z_norm_var']       = h['train_scad_z_norm_var'][idx] if idx < len(h.get('train_scad_z_norm_var', [])) else 0.0
                    cb_metrics['scad_lambda']           = h['train_scad_lambda'][idx] if idx < len(h.get('train_scad_lambda', [])) else 0.0
                    cb_metrics['scad_adaptive_lambda']  = h['train_scad_adaptive_lambda'][idx] if idx < len(h.get('train_scad_adaptive_lambda', [])) else 0.0
                    cb_metrics['scad_ramp']             = h['train_scad_ramp'][idx] if idx < len(h.get('train_scad_ramp', [])) else 0.0
                    cb_metrics['scad_effective_weight'] = h['train_scad_effective_weight'][idx] if idx < len(h.get('train_scad_effective_weight', [])) else 0.0
                    cb_metrics['scad_grad_norm']        = h['train_scad_grad_norm'][idx] if idx < len(h.get('train_scad_grad_norm', [])) else 0.0
                    cb_metrics['scad_main_grad_norm']   = h['train_scad_main_grad_norm'][idx] if idx < len(h.get('train_scad_main_grad_norm', [])) else 0.0

            # Attach training feature stats from history
            if trainer is not None:
                h = trainer.history
                idx = ep - 1
                for fk in ['train_feature_disc_mean', 'train_feature_disc_max',
                            'train_feature_recon_mean', 'train_feature_recon_max']:
                    if idx < len(h.get(fk, [])):
                        val = h[fk][idx]
                        cb_metrics[f'_{fk}'] = val.tolist() if hasattr(val, 'tolist') else val

            _eval_queue.put(cb_metrics)

            # Save point-level scores (mean-aggregated) as lightweight npz
            try:
                from mae_anomaly.evaluator import _build_aggregation_map, _aggregate_with_map
                pt_labels = np.array(test_dataset.point_labels)
                total_len = len(pt_labels)
                ws_indices = np.array(test_dataset.window_start_indices)
                flat_t, flat_wp, coverage, covered = _build_aggregation_map(
                    ws_indices, config.patch_size, config.num_patches, total_len
                )

                recon_p = eval_data['recon_patches']
                disc_p = eval_data['disc_patches']
                student_p = eval_data['student_recon_patches']
                fm_p = eval_data.get('fm_patches')  # None if FM not used

                # Adaptive score (patch-level → point-level mean).
                # Matches evaluator._apply_scoring_formula:
                #   scaled_disc = disc * (recon.mean / disc.mean)
                #   scaled_fm   = fm   * (recon.mean / fm.mean)
                #   student_error = (w_disc*scaled_disc + w_fm*scaled_fm) / (w_disc + w_fm)
                #   score = recon + student_error
                score_mode = config.anomaly_score_mode
                use_fm_eval = (fm_p is not None) and getattr(config, 'use_feature_matching', False)
                if score_mode == 'adaptive':
                    recon_mean = recon_p.mean() + 1e-4
                    w_disc = getattr(config, 'eval_disc_weight', -1.0)
                    w_fm = getattr(config, 'eval_fm_weight', -1.0)
                    if w_disc < 0:
                        w_disc = 1.0
                    if w_fm < 0:
                        w_fm = getattr(config, 'fm_loss_weight', 1.0)
                    if not getattr(config, 'use_output_discrepancy', True):
                        w_disc = 0.0
                    scaled_disc = disc_p * (recon_mean / (disc_p.mean() + 1e-4))
                    if use_fm_eval:
                        scaled_fm = fm_p * (recon_mean / (fm_p.mean() + 1e-4))
                        denom = max(w_disc + w_fm, 1e-6)
                        student_error = (w_disc * scaled_disc + w_fm * scaled_fm) / denom
                    elif w_disc > 0:
                        student_error = scaled_disc
                    else:
                        student_error = np.zeros_like(recon_p)
                    adaptive_patch = recon_p + student_error
                else:
                    adaptive_patch = recon_p + config.lambda_disc * disc_p

                adaptive_scores = _aggregate_with_map(
                    adaptive_patch.ravel(), flat_t, flat_wp, coverage, covered, total_len, method='mean'
                )
                teacher_recon_scores = _aggregate_with_map(
                    recon_p.ravel(), flat_t, flat_wp, coverage, covered, total_len, method='mean'
                )
                disc_scores = _aggregate_with_map(
                    disc_p.ravel(), flat_t, flat_wp, coverage, covered, total_len, method='mean'
                )
                fm_scores = None
                if use_fm_eval:
                    # fm_p shape is typically (n_windows, num_patches) — same as disc_p
                    fm_scores = _aggregate_with_map(
                        fm_p.ravel(), flat_t, flat_wp, coverage, covered, total_len, method='mean'
                    )

                save_dict = {
                    'adaptive_score': np.nan_to_num(adaptive_scores, nan=0.0).astype(np.float32),
                    'teacher_recon_error': np.nan_to_num(teacher_recon_scores, nan=0.0).astype(np.float32),
                    'discrepancy_error': np.nan_to_num(disc_scores, nan=0.0).astype(np.float32),
                    'point_labels': np.asarray(pt_labels, dtype=np.int8),
                }
                if fm_scores is not None:
                    save_dict['fm_error'] = np.nan_to_num(fm_scores, nan=0.0).astype(np.float32)
                np.savez_compressed(
                    os.path.join(epoch_scores_dir, f'epoch_{ep:03d}_scores.npz'),
                    **save_dict,
                )

            except Exception as e_scores:
                tqdm.write(f"  [Epoch {ep:>2}] SCORE SAVE ERROR: {e_scores}")
        except Exception as e:
            tqdm.write(f"  [Epoch {ep:>2}] ASYNC EVAL ERROR: {e}")

    def epoch_eval_callback(epoch, model, history):
        nonlocal callback_infer_time
        ep = epoch + 1
        if ep % EVAL_INTERVAL != 0 and ep != config.num_epochs:
            return

        # GPU inference (synchronous — must block while model is available)
        cb_start = time.time()
        try:
            eval_data = compute_epoch_test_inference(
                model, test_loader, config, test_dataset=test_dataset
            )

            # Compute contribution ratios from eval_data (pure numpy, ~1ms)
            contrib = compute_contrib_from_eval_data(eval_data, config)
            _trainer_ref[0]._pending_contrib = contrib

            # Capture state_dict NOW (GPU→CPU copy, ~100-500ms; must happen before
            # next training step mutates weights)
            ckpt_data = {
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'config': asdict(config),
            }
            if _trainer_ref[0] is not None and _trainer_ref[0].discriminator is not None:
                ckpt_data['discriminator_state_dict'] = _trainer_ref[0].discriminator.state_dict()

            cb_time = time.time() - cb_start
            callback_infer_time += cb_time

            # Launch background thread: join(prev) → drain → save ckpt → CPU eval
            # GPU returns to training immediately after this point.
            prev_thread = _async_eval_thread[0]
            t = threading.Thread(
                target=_run_bg_all,
                args=(eval_data, ep, ckpt_data, prev_thread),
                daemon=True,
            )
            t.start()
            _async_eval_thread[0] = t
        except Exception as e:
            cb_time = time.time() - cb_start
            callback_infer_time += cb_time
            print(f"  [Epoch {ep:>2}] EVAL ERROR: {e} ({cb_time:.0f}s)")

    wall_start = time.time()
    _trainer_ref = [None]  # Shared reference for callback to access discriminator
    trainer = Trainer(model, config, train_loader, test_loader, verbose=True)
    _trainer_ref[0] = trainer
    trainer.train(epoch_callback=epoch_eval_callback, profile_n_batches=PROFILE_N_BATCHES)
    # Collect last async eval result (blocking: wait for final thread)
    _collect_async_eval(blocking=True)
    wall_time = time.time() - wall_start
    train_time = wall_time - callback_infer_time  # Pure training time (no inference callback)
    history = trainer.history
    epochs_done = config.num_epochs
    per_epoch = train_time / max(epochs_done, 1)
    n_evals = len(epoch_metrics_list)
    print(f"Training complete: wall={wall_time:.0f}s, pure_train={train_time:.0f}s "
          f"({per_epoch:.1f}s/ep), gpu_infer_callback={callback_infer_time:.0f}s ({n_evals} evals) | {mem_status()}")

    # Save batch profiling from epoch 1 (first N batches with per-component sync timing)
    batch_profiling = history.get('batch_profiling', [])
    if batch_profiling:
        save_batch_profiling(batch_profiling, config, len(train_loader), exp_dir)

    # Save epoch metrics
    epoch_metrics_path = os.path.join(exp_dir, 'epoch_metrics.json')
    with open(epoch_metrics_path, 'w') as f:
        json.dump({'eval_interval': EVAL_INTERVAL, 'epochs': epoch_metrics_list}, f, indent=2)
    print(f"  Epoch metrics saved: {epoch_metrics_path}")

    # Generate epoch-wise visualizations
    epoch_viz_dir = os.path.join(exp_dir, 'visualization', 'epoch_metrics')
    plot_epoch_metrics(epoch_metrics_list, epoch_viz_dir)
    plot_epoch_feature_stats(epoch_metrics_list, epoch_viz_dir)
    print(f"  Epoch visualizations saved: {epoch_viz_dir}")

    # ========== Find Best Epoch ==========
    best_epoch = config.num_epochs
    best_score = -1.0
    best_prc = -1.0
    for em in epoch_metrics_list:
        score = em.get(_best_epoch_metric_key, 0)
        if score > best_score:
            best_score = score
            best_epoch = em.get('epoch', config.num_epochs)
            best_prc = em.get('prc_auc', 0)

    best_ckpt_path = os.path.join(checkpoints_dir, 'best_checkpoint.pt')
    if best_epoch != config.num_epochs and os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=config.device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        model.eval()
        print(f"  Best model: epoch {best_ckpt.get('epoch', best_epoch)} "
              f"({_best_epoch_metric_key}={best_score:.4f}, PRC={best_prc:.4f}), loaded from best_checkpoint.pt")
    else:
        print(f"  Best model: last epoch {best_epoch} ({_best_epoch_metric_key}={best_score:.4f}, PRC={best_prc:.4f})")

    # ========== Save Model ==========
    save_data = {
        'model_state_dict': model.state_dict(),
        'config': asdict(config),
        'best_epoch': best_epoch,
        'best_epoch_metric': _best_epoch_metric_key,
        'best_epoch_score': best_score,
        'best_prc_auc': best_prc,
    }
    if trainer.discriminator is not None:
        save_data['discriminator_state_dict'] = trainer.discriminator.state_dict()
    torch.save(save_data, os.path.join(exp_dir, 'best_model.pt'))

    # ========== Best-Epoch Train Data Scoring ==========
    # After training: run the same inference+scoring pipeline on TRAIN data
    # with the loaded best-epoch model. Save to best_epoch_train_scores.npz.
    # Reason: enables visualization / analysis of train-time anomaly score
    # distribution as a reference for test-time scoring.
    try:
        from mae_anomaly.evaluator import _build_aggregation_map, _aggregate_with_map
        print(f"  Best-epoch TRAIN inference (stride={test_stride})...")
        # New train loader for inference: same stride as test, no shuffle, no epoch_offset
        train_infer_dataset = SlidingWindowDataset(
            signals=signals, point_labels=point_labels, anomaly_regions=anomaly_regions,
            window_size=config.seq_length, stride=test_stride, mask_last_n=config.patch_size,
            split='train', train_ratio=train_ratio, seed=config.random_seed,
            run_boundaries=run_boundaries,
            normalize_mode=config.normalize_mode,
            minmax_range=getattr(config, 'minmax_range', '0_1'),
            minmax_clamp_min=getattr(config, 'minmax_clamp_min', None),
            minmax_clamp_max=getattr(config, 'minmax_clamp_max', None),
        )
        train_infer_loader = DataLoader(train_infer_dataset, batch_size=config.batch_size, shuffle=False)
        t_train_infer = time.time()
        train_eval_data = compute_epoch_test_inference(model, train_infer_loader, config, test_dataset=train_infer_dataset)
        train_recon_p = train_eval_data['recon_patches']
        train_disc_p = train_eval_data['disc_patches']
        train_fm_p = train_eval_data.get('fm_patches')

        # Aggregation map for train data
        train_pt_labels = np.array(train_infer_dataset.point_labels)
        train_total_len = len(train_pt_labels)
        train_ws_indices = np.array(train_infer_dataset.window_start_indices)
        t_flat_t, t_flat_wp, t_coverage, t_covered = _build_aggregation_map(
            train_ws_indices, config.patch_size, config.num_patches, train_total_len
        )

        # Scoring (same formula as test: evaluator._apply_scoring_formula adaptive)
        score_mode = config.anomaly_score_mode
        use_fm_eval = (train_fm_p is not None) and getattr(config, 'use_feature_matching', False)
        if score_mode == 'adaptive':
            t_recon_mean = train_recon_p.mean() + 1e-4
            w_disc = getattr(config, 'eval_disc_weight', -1.0)
            w_fm = getattr(config, 'eval_fm_weight', -1.0)
            if w_disc < 0:
                w_disc = 1.0
            if w_fm < 0:
                w_fm = getattr(config, 'fm_loss_weight', 1.0)
            if not getattr(config, 'use_output_discrepancy', True):
                w_disc = 0.0
            t_scaled_disc = train_disc_p * (t_recon_mean / (train_disc_p.mean() + 1e-4))
            if use_fm_eval:
                t_scaled_fm = train_fm_p * (t_recon_mean / (train_fm_p.mean() + 1e-4))
                denom = max(w_disc + w_fm, 1e-6)
                t_student_err = (w_disc * t_scaled_disc + w_fm * t_scaled_fm) / denom
            elif w_disc > 0:
                t_student_err = t_scaled_disc
            else:
                t_student_err = np.zeros_like(train_recon_p)
            t_adaptive_patch = train_recon_p + t_student_err
        else:
            t_adaptive_patch = train_recon_p + config.lambda_disc * train_disc_p

        t_adaptive_scores = _aggregate_with_map(
            t_adaptive_patch.ravel(), t_flat_t, t_flat_wp, t_coverage, t_covered,
            train_total_len, method='mean')
        t_recon_scores = _aggregate_with_map(
            train_recon_p.ravel(), t_flat_t, t_flat_wp, t_coverage, t_covered,
            train_total_len, method='mean')
        t_disc_scores = _aggregate_with_map(
            train_disc_p.ravel(), t_flat_t, t_flat_wp, t_coverage, t_covered,
            train_total_len, method='mean')

        train_save_dict = {
            'adaptive_score': np.nan_to_num(t_adaptive_scores, nan=0.0).astype(np.float32),
            'teacher_recon_error': np.nan_to_num(t_recon_scores, nan=0.0).astype(np.float32),
            'discrepancy_error': np.nan_to_num(t_disc_scores, nan=0.0).astype(np.float32),
            'point_labels': np.asarray(train_pt_labels, dtype=np.int8),
        }
        if use_fm_eval:
            t_fm_scores = _aggregate_with_map(
                train_fm_p.ravel(), t_flat_t, t_flat_wp, t_coverage, t_covered,
                train_total_len, method='mean')
            train_save_dict['fm_error'] = np.nan_to_num(t_fm_scores, nan=0.0).astype(np.float32)
        train_scores_path = os.path.join(exp_dir, 'best_epoch_train_scores.npz')
        np.savez_compressed(train_scores_path, **train_save_dict)
        print(f"    saved {train_scores_path} (took {time.time() - t_train_infer:.1f}s)")
        # Free memory
        del train_eval_data, train_recon_p, train_disc_p, train_fm_p
        del t_adaptive_patch, t_adaptive_scores, t_recon_scores, t_disc_scores
        del train_infer_loader, train_infer_dataset

        # Clean up the temporary best-epoch checkpoint (params now consolidated
        # in best_model.pt). Per user spec: "train inference 후 파라미터 파일 삭제".
        # KEEP_BEST_CKPT=1 환경변수 설정 시 보존 (274 재실험용)
        if os.environ.get('KEEP_BEST_CKPT') == '1':
            print(f"    KEEP_BEST_CKPT=1 — preserving {best_ckpt_path}")
        elif os.path.exists(best_ckpt_path):
            os.remove(best_ckpt_path)
            print(f"    removed temporary {best_ckpt_path}")
    except Exception as _e_train_infer:
        print(f"    [WARN] best-epoch train inference failed: {_e_train_infer}")
        import traceback; traceback.print_exc()

    # ========== GPU Inference (single unified pass) ==========
    print(f"  GPU inference for eval+viz... ({mem_status()})")
    model.eval()

    t_infer = time.time()
    evaluator = Evaluator(model, config, test_loader, test_dataset=test_dataset)
    recon_p, disc_p, student_p, labels, sample_types, anomaly_types = \
        evaluator._compute_patch_scores_all_patches(collect_detail=True)
    torch.cuda.synchronize()
    inference_time = time.time() - t_infer
    patch_scores = {
        'recon': recon_p, 'disc': disc_p, 'student_recon': student_p,
        'labels': labels, 'sample_types': sample_types, 'anomaly_types': anomaly_types,
        'disc_per_feature': evaluator.disc_per_feature,  # (n_windows, F)
    }
    full_detail = evaluator.detail_results  # (N_all, seq_length) reconstruction data
    del evaluator
    torch.cuda.empty_cache()
    print(f"    Patch scores: {recon_p.shape} + detail ({inference_time:.1f}s) | {mem_status()}")

    # Subsample for visualization (pred_data + detailed_data)
    VIZ_MAX_SAMPLES = 10000
    total_test = len(test_dataset)
    if total_test > VIZ_MAX_SAMPLES:
        viz_indices = np.linspace(0, total_test - 1, VIZ_MAX_SAMPLES, dtype=int)
    else:
        viz_indices = np.arange(total_test)

    pred_data = derive_pred_data(
        recon_p, disc_p, student_p, labels, sample_types,
        config, test_dataset, subset_indices=viz_indices,
    )
    detailed_data = {k: v[viz_indices] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == total_test else v
                     for k, v in full_detail.items()}
    del full_detail

    # Subsample disc_per_feature to match viz subset (prevents shape mismatch in feature plots)
    if patch_scores.get('disc_per_feature') is not None:
        patch_scores['disc_per_feature'] = patch_scores['disc_per_feature'][viz_indices]

    # Capture window_start_indices before deleting test_dataset
    test_window_start_indices = test_dataset.window_start_indices.copy()

    # ========== Free GPU ==========
    del model, trainer, train_loader, test_loader, train_dataset, test_dataset
    gc.collect()
    free_gpu()
    print(f"  GPU freed | {mem_status()}")

    # ========== Spawn Background CPU eval+viz ==========
    # Serialize anomaly_regions for multiprocessing
    anomaly_regions_ser = [
        {'start': r.start, 'end': r.end, 'anomaly_type': r.anomaly_type}
        for r in anomaly_regions
    ]

    # Timing summary for GPU phase
    gpu_total = train_time + callback_infer_time + inference_time
    print(f"  GPU phase: train={train_time:.0f}s + epoch_eval={callback_infer_time:.0f}s "
          f"+ inference={inference_time:.0f}s = {gpu_total:.0f}s")

    print(f"  Spawning background eval+viz for {key}...")
    timing = {
        'wall_time': wall_time,
        'pure_train_time': train_time,
        'train_per_epoch': train_time / max(epochs_done, 1),
        'epoch_eval_time': callback_infer_time,
        'num_epochs': epochs_done,
        'num_evals': n_evals,
        'inference_time': inference_time,
        'gpu_total': gpu_total,
        'best_epoch': best_epoch,
        'best_epoch_metric': _best_epoch_metric_key,
        'best_epoch_score': best_score,
        'best_prc_auc': best_prc,
    }

    ctx = mp.get_context('spawn')
    p = ctx.Process(target=_cpu_eval_viz_worker, args=(
        f"base_{key}", exp_dir, asdict(config), signals, point_labels,
        anomaly_regions_ser, history, train_ratio, timing,
        patch_scores, pred_data, detailed_data,
        progress_info, test_window_start_indices,
        None,  # swat_eval_mode=None for full eval
    ))
    p.start()
    _background_processes.append((key, p))

    # SWaT dual-eval: spawn second worker for excl22 evaluation
    if is_swat_dual:
        # Copy shared training files to excl22 dir (worker saves best_config/history itself)
        os.makedirs(exp_dir_excl22, exist_ok=True)
        for shared_file in ['epoch_metrics.json', 'batch_profiling.json', 'batch_profiling.txt']:
            src = os.path.join(exp_dir, shared_file)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(exp_dir_excl22, shared_file))

        # Symlink shared directories (checkpoints, epoch_scores)
        for shared_dir in ['checkpoints', 'epoch_scores']:
            src = os.path.join(exp_dir, shared_dir)
            dst = os.path.join(exp_dir_excl22, shared_dir)
            if os.path.isdir(src) and not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)

        # Copy full best_model.pt as baseline (worker may override if excl22 best epoch differs)
        src_model = os.path.join(exp_dir, 'best_model.pt')
        if os.path.exists(src_model):
            shutil.copy2(src_model, os.path.join(exp_dir_excl22, 'best_model.pt'))

        # Pre-filter viz data for excl22 (region 22 excluded from visualizations)
        from mae_anomaly.evaluator import find_swat_largest_region
        from mae_anomaly.dataset_sliding import AnomalyRegion
        train_end = int(len(signals) * train_ratio)
        test_anomaly_regions = [
            AnomalyRegion(start=r['start'] - train_end, end=r['end'] - train_end,
                          anomaly_type=r['anomaly_type'])
            for r in anomaly_regions_ser if r['start'] >= train_end
        ]
        largest_region = find_swat_largest_region(test_anomaly_regions)
        if largest_region is not None:
            pred_data_excl22, detailed_data_excl22, excl22_keep_win = _filter_excl22_viz_data(
                pred_data, detailed_data, config, largest_region.start, largest_region.end)
        else:
            pred_data_excl22, detailed_data_excl22 = pred_data, detailed_data
            excl22_keep_win = None

        # Filter disc_per_feature for excl22 (same window mask as pred_data filtering)
        import copy as _copy
        patch_scores_excl22 = _copy.copy(patch_scores)
        if excl22_keep_win is not None and patch_scores_excl22.get('disc_per_feature') is not None:
            patch_scores_excl22['disc_per_feature'] = patch_scores_excl22['disc_per_feature'][excl22_keep_win]

        # excl22 worker determines its own best epoch by scanning epoch_scores
        print(f"  Spawning background eval+viz for {key}_excl22...")
        p2 = ctx.Process(target=_cpu_eval_viz_worker, args=(
            f"base_{key}_excl22", exp_dir_excl22, asdict(config), signals, point_labels,
            anomaly_regions_ser, history, train_ratio, dict(timing),
            patch_scores_excl22, pred_data_excl22, detailed_data_excl22,
            progress_info, test_window_start_indices,
            'excl22',  # swat_eval_mode='excl22'
        ))
        p2.start()
        _background_processes.append((f"{key}_excl22", p2))

    del patch_scores, pred_data, detailed_data, signals, point_labels
    if is_swat_dual:
        del pred_data_excl22, detailed_data_excl22, patch_scores_excl22

    # save_weights=False: delete weight files after inference/viz data is dispatched.
    # Background workers already have all data they need (patch_scores, pred_data).
    # KEEP_BEST_CKPT=1 환경변수 설정 시 best_checkpoint.pt 보존 (274 재실험용)
    if not save_weights:
        keep_best = os.environ.get('KEEP_BEST_CKPT') == '1'
        cleanup_list = [os.path.join(exp_dir, 'best_model.pt'),
                        os.path.join(checkpoints_dir, 'latest_checkpoint.pt')]
        if not keep_best:
            cleanup_list.append(os.path.join(checkpoints_dir, 'best_checkpoint.pt'))
        else:
            print(f"  KEEP_BEST_CKPT=1 — preserving {os.path.join(checkpoints_dir, 'best_checkpoint.pt')}")
        for wf in cleanup_list:
            if os.path.exists(wf):
                os.remove(wf)
        if os.path.isdir(checkpoints_dir) and not os.listdir(checkpoints_dir):
            os.rmdir(checkpoints_dir)
        # SWaT excl22 checkpoints are symlinked — remove symlink
        if is_swat_dual and exp_dir_excl22:
            excl22_ckpt = os.path.join(exp_dir_excl22, 'checkpoints')
            if os.path.islink(excl22_ckpt):
                os.unlink(excl22_ckpt)
            excl22_model = os.path.join(exp_dir_excl22, 'best_model.pt')
            if os.path.exists(excl22_model):
                os.remove(excl22_model)

    return {
        'key': key,
        'dir': exp_dir,
        'pure_train_time': train_time,
        'epoch_eval_time': callback_infer_time,
        'inference_time': inference_time,
        'per_epoch': train_time / max(epochs_done, 1),
        'best_epoch': best_epoch,
        'best_score': best_score,
        'best_prc': best_prc if best_prc >= 0 else 0,
        'final_prc': epoch_metrics_list[-1].get('prc_auc', 0) if epoch_metrics_list else 0,
        'final_f1': epoch_metrics_list[-1].get('f1_t', 0) if epoch_metrics_list else 0,
    }


# =============================================================================
# SMD Results Aggregation (28 machines × 2 parities → single average)
# =============================================================================

SKIP_AGG_KEYS = {'epoch', '_inference_time', '_eval_time', '_train_time',
                 'callback_time', '_infer_feature_disc_mean', '_infer_feature_disc_max',
                 '_train_feature_disc_mean', '_train_feature_disc_max',
                 '_train_feature_recon_mean', '_train_feature_recon_max'}


def aggregate_smd_results(experiment_dir):
    """Aggregate SMD results across 28 machines.

    Reads epoch_metrics.json from SMD/{machine}/, selects best epoch per machine,
    then averages across machines. Saves to SMD/results/results.csv.

    Args:
        experiment_dir: Path to experiment directory (e.g., results/experiments/102_...)

    Returns:
        dict with global average metrics, or None if no results found.
    """
    import csv

    smd_base = os.path.join(experiment_dir, 'SMD')
    if not os.path.isdir(smd_base):
        print(f"  No SMD directory found in {experiment_dir}")
        return None

    print(f"\n{'='*60}")
    print(f"Aggregating SMD results: {experiment_dir}")
    print(f"{'='*60}")

    machine_metrics = []  # List of (machine_name, best_metrics_dict)

    for machine in SMD_MACHINE_NAMES:
        em_path = os.path.join(smd_base, machine, 'epoch_metrics.json')
        if not os.path.exists(em_path):
            print(f"  {machine}: MISSING")
            continue

        try:
            with open(em_path) as f:
                data = json.load(f)
            epochs = data.get('epochs', [])
            if not epochs:
                print(f"  {machine}: EMPTY")
                continue
            best = max(epochs, key=lambda e: e.get('pak_auc_f1', 0) or 0)
        except Exception as e:
            print(f"  {machine}: ERROR ({e})")
            continue

        machine_metrics.append((machine, best))
        print(f"  {machine}: OK (pak_f1={best.get('pak_auc_f1', 0):.4f})")

    if not machine_metrics:
        print("  NO RESULTS found")
        return None

    # Average across machines
    all_keys = set()
    for _, mm in machine_metrics:
        all_keys.update(mm.keys())

    global_avg = {}
    for key in sorted(all_keys):
        if key in SKIP_AGG_KEYS:
            continue
        vals = [mm.get(key) for _, mm in machine_metrics if mm.get(key) is not None]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            global_avg[key] = sum(vals) / len(vals)

    # Save results.csv
    results_dir = os.path.join(smd_base, 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'results.csv')

    metric_keys = sorted(global_avg.keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['machine'] + metric_keys)
        for machine_name, mm in machine_metrics:
            row = [machine_name]
            for key in metric_keys:
                val = mm.get(key)
                if val is not None:
                    row.append(f"{val:.6f}" if isinstance(val, float) else str(val))
                else:
                    row.append("")
            writer.writerow(row)
        # Average row
        avg_row = ['AVERAGE']
        for key in metric_keys:
            val = global_avg.get(key)
            if val is not None:
                avg_row.append(f"{val:.6f}" if isinstance(val, float) else str(val))
            else:
                avg_row.append("")
        writer.writerow(avg_row)

    print(f"\n  Saved: {csv_path}")
    print(f"  Machines: {len(machine_metrics)}/{len(SMD_MACHINE_NAMES)}")
    print(f"  AVERAGE pak_auc_f1: {global_avg.get('pak_auc_f1', 0):.4f}")
    print(f"  AVERAGE pak_auc_prc_auc: {global_avg.get('pak_auc_prc_auc', 0):.4f}")
    print(f"  AVERAGE prc_auc: {global_avg.get('prc_auc', 0):.4f}")
    print(f"  AVERAGE f1_t: {global_avg.get('f1_t', 0):.4f}")

    return global_avg


def aggregate_exathlon_results(experiment_dir):
    """Aggregate Exathlon results across 6 apps.

    Reads epoch_metrics.json from Exathlon/app{N}/, selects best epoch per app
    (by pak_auc_f1), then averages across apps. Saves to Exathlon/results/results.csv.

    Mirrors aggregate_smd_results structure for SMD pattern consistency.

    Args:
        experiment_dir: Path to experiment directory (e.g., results/experiments/274_...)

    Returns:
        dict with global average metrics, or None if no results found.
    """
    import csv

    exa_base = os.path.join(experiment_dir, 'Exathlon')
    if not os.path.isdir(exa_base):
        print(f"  No Exathlon directory found in {experiment_dir}")
        return None

    print(f"\n{'='*60}")
    print(f"Aggregating Exathlon results: {experiment_dir}")
    print(f"{'='*60}")

    app_metrics = []  # List of (app_name, best_metrics_dict)

    for app in EXATHLON_APP_IDS:
        app_name = f'app{app}'
        em_path = os.path.join(exa_base, app_name, 'epoch_metrics.json')
        if not os.path.exists(em_path):
            print(f"  {app_name}: MISSING")
            continue

        try:
            with open(em_path) as f:
                data = json.load(f)
            epochs = data.get('epochs', [])
            if not epochs:
                print(f"  {app_name}: EMPTY")
                continue
            best = max(epochs, key=lambda e: e.get('pak_auc_f1', 0) or 0)
        except Exception as e:
            print(f"  {app_name}: ERROR ({e})")
            continue

        app_metrics.append((app_name, best))
        print(f"  {app_name}: OK (pak_f1={best.get('pak_auc_f1', 0):.4f})")

    if not app_metrics:
        print("  NO RESULTS found")
        return None

    # Average across apps
    all_keys = set()
    for _, mm in app_metrics:
        all_keys.update(mm.keys())

    global_avg = {}
    for key in sorted(all_keys):
        if key in SKIP_AGG_KEYS:
            continue
        vals = [mm.get(key) for _, mm in app_metrics if mm.get(key) is not None]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            global_avg[key] = sum(vals) / len(vals)

    # Save results.csv
    results_dir = os.path.join(exa_base, 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'results.csv')

    metric_keys = sorted(global_avg.keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['app'] + metric_keys)
        for app_name, mm in app_metrics:
            row = [app_name]
            for key in metric_keys:
                val = mm.get(key)
                if val is not None:
                    row.append(f"{val:.6f}" if isinstance(val, float) else str(val))
                else:
                    row.append("")
            writer.writerow(row)
        # Average row
        avg_row = ['AVERAGE']
        for key in metric_keys:
            val = global_avg.get(key)
            if val is not None:
                avg_row.append(f"{val:.6f}" if isinstance(val, float) else str(val))
            else:
                avg_row.append("")
        writer.writerow(avg_row)

    print(f"\n  Saved: {csv_path}")
    print(f"  Apps: {len(app_metrics)}/{len(EXATHLON_APP_IDS)}")
    print(f"  AVERAGE pak_auc_f1: {global_avg.get('pak_auc_f1', 0):.4f}")
    print(f"  AVERAGE pak_auc_prc_auc: {global_avg.get('pak_auc_prc_auc', 0):.4f}")
    print(f"  AVERAGE prc_auc: {global_avg.get('prc_auc', 0):.4f}")
    print(f"  AVERAGE f1_t: {global_avg.get('f1_t', 0):.4f}")

    return global_avg


def main():
    parser = argparse.ArgumentParser(description='Run base model on all datasets')
    parser.add_argument('--set', type=str, required=True, choices=['A', 'B', 'C'],
                        help='Config preset (A=d128/p5/k3, B=d256/p20/k5, C=dynamic/p10/linear)')
    parser.add_argument('--dataset', type=str, nargs='+', default=None,
                        help='Run specific dataset key(s) (e.g., SWaT_A1A2 WaDi_A1)')
    parser.add_argument('--start-from', type=int, default=0,
                        help='Start from dataset index (0-based)')
    parser.add_argument('--list', action='store_true',
                        help='List all datasets and exit')
    parser.add_argument('--output-base', type=str, default=None,
                        help='Override output base directory')
    parser.add_argument('--config-override', type=str, nargs='+', default=None,
                        help='Override config values (e.g., force_mask_anomaly=False num_epochs=50)')
    parser.add_argument('--no-wait', action='store_true',
                        help='Skip waiting for background CPU processes (eval+viz). '
                             'Used by run_queue.py to pipeline experiments.')
    parser.add_argument('--save-weights', action='store_true', default=False,
                        help='Save model weights (checkpoints, best_model.pt). '
                             'Default: off. Epoch metrics and scores are always saved.')
    args = parser.parse_args()

    config_preset = CONFIG_PRESETS[args.set]

    all_datasets = DATASETS + SMD_DATASETS + EXATHLON_DATASETS

    if args.list:
        print(f"\nSet {args.set}: {config_preset['description']}")
        print(f"{'#':>3} {'Key':<30} {'Loader':<30} {'Stride':>7} {'N50':>5} {'Subdir':<35}")
        print("-" * 115)
        for i, d in enumerate(all_datasets):
            print(f"{i:>3} {d['key']:<30} {d['loader']:<30} {d['train_stride']:>7} "
                  f"{'Yes' if d['normal50'] else 'No':>5} {d['results_subdir']:<35}")
        print(f"\nTotal: {len(DATASETS)} base + {len(SMD_DATASETS)} SMD + {len(EXATHLON_DATASETS)} Exathlon = {len(all_datasets)}")
        return

    # Apply config overrides BEFORE creating output dir (so suffix reflects overrides)
    if args.config_override:
        # Flatten: each item may contain multiple space-separated key=value pairs
        flat_kvs = []
        for item in args.config_override:
            flat_kvs.extend(item.split())
        for kv in flat_kvs:
            key, val = kv.split('=', 1)
            # Auto-cast value types
            if val.lower() == 'true':
                val = True
            elif val.lower() == 'false':
                val = False
            elif val.lower() == 'none':
                val = None
            elif val.startswith('(') and val.endswith(')'):
                # Parse tuple: "(64,32)" -> (64, 32)
                inner = val[1:-1]
                parts = [p.strip() for p in inner.split(',') if p.strip()]
                val = tuple(float(p) if '.' in p else int(p) for p in parts)
            elif val.replace('.', '', 1).replace('-', '', 1).isdigit():
                val = float(val) if '.' in val else int(val)
            config_preset['overrides'][key] = val
        print(f"  Config overrides: {args.config_override}")

    # Determine output base (after --list check and overrides applied)
    if args.output_base:
        results_base = args.output_base
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiments_dir = os.path.join(PROJECT_ROOT, 'results', 'experiments')
        suffix = f'{timestamp}_{make_dynamic_suffix(config_preset["overrides"])}'
        results_base = make_numbered_experiment_dir(experiments_dir, suffix)

    # Determine which datasets to run
    if args.dataset:
        dataset_keys = set(args.dataset)
        # Support "smd_all" shortcut for all SMD datasets
        if 'smd_all' in dataset_keys:
            dataset_keys.discard('smd_all')
            dataset_keys.update(d['key'] for d in SMD_DATASETS)
        datasets = [d for d in all_datasets if d['key'] in dataset_keys]
        if not datasets:
            print(f"ERROR: No matching datasets for {args.dataset}. Use --list to see available.")
            return
    else:
        datasets = DATASETS[args.start_from:]

    os.makedirs(results_base, exist_ok=True)

    print(f"{'='*80}")
    print(f"Base Model Experiments: {len(datasets)} datasets")
    print(f"  Set: {args.set} ({config_preset['description']})")
    print(f"  Eval interval: every {EVAL_INTERVAL} epochs")
    print(f"  Results: {results_base}")
    print(f"{'='*80}")

    results = []
    total_start = time.time()

    for i, dataset_def in enumerate(datasets):
        idx = args.start_from + i
        print(f"\n{'#'*80}")
        print(f"# [{i+1}/{len(datasets)}] {dataset_def['key']}")
        print(f"{'#'*80}")

        # Wait if too many background processes
        while len([p for _, p in _background_processes if p.is_alive()]) >= 10:
            print(f"  Waiting for background processes... ({mem_status()})")
            time.sleep(10)

        try:
            progress_info = f"[{i+1}/{len(datasets)}]"
            result = run_base_experiment(
                dataset_def, config_preset, results_base, progress_info,
                save_weights=args.save_weights,
            )
            results.append(result)
            pt = result['pure_train_time']
            ee = result['epoch_eval_time']
            print(f"  Completed: {result['key']} (train={pt:.0f}s eval={ee:.0f}s, "
                  f"bestScore={result['best_score']:.4f}@ep{result['best_epoch']}, "
                  f"bestPRC={result['best_prc']:.4f}, finalPRC={result['final_prc']:.4f}, F1={result['final_f1']:.4f})")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Wait for background processes (unless --no-wait for pipelining)
    if args.no_wait:
        bg_pids = [p.pid for _, p in _background_processes if p.is_alive()]
        print(f"\n{'='*80}")
        print(f"--no-wait: skipping background join ({len(bg_pids)} processes still running)")
        print(f"  Background PIDs: {bg_pids}")
    else:
        print(f"\n{'='*80}")
        print(f"Waiting for {len(_background_processes)} background processes...")
        for name, p in _background_processes:
            p.join(timeout=600)
            if p.is_alive():
                print(f"  {name}: still running (timeout)")
                p.terminate()
            else:
                print(f"  {name}: completed")

    # Summary
    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETE ({total_time/60:.1f} min)")
    print(f"{'='*80}")

    if results:
        print(f"\n{'Dataset':<25} {'Train(s)':>8} {'BestEp':>6} {'BestScore':>10} {'BestPRC':>8} {'FinalPRC':>9} {'F1':>8}")
        print("-" * 80)
        for r in results:
            print(f"{r['key']:<25} {r['pure_train_time']:>8.0f} {r['best_epoch']:>6} "
                  f"{r['best_score']:>10.4f} {r['best_prc']:>8.4f} {r['final_prc']:>9.4f} {r['final_f1']:>8.4f}")

    # Save summary
    summary_path = os.path.join(results_base, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'total_time': total_time,
            'config_set': args.set,
            'description': config_preset['description'],
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
