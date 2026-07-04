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

# === OMP / BLAS thread cap (2026-05-29) ===
# Must precede numpy/torch/sklearn import. OpenBLAS reads OMP_NUM_THREADS at
# C-level pool init time (first BLAS call after numpy load) and keeps that
# pool size for the whole process lifetime. Setting it later — inside a
# function body or after numpy import — is silently ignored because the pool
# is already allocated at cpu_count threads (=16 here). Without this cap,
# bg-workers spawn ~35 threads/process, then multiplex on 4 affinity cores
# with heavy context-switch overhead. setdefault honors launcher overrides.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')

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
from mae_anomaly.config import set_seed_official, official_worker_init_fn, CANON_271
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
from mae_anomaly.datasets.loaders import (SMD_MACHINE_NAMES, EXATHLON_APP_IDS,
                                          SMAP_CHANNEL_NAMES, MSL_CHANNEL_NAMES)
from mae_anomaly.utils import make_config, free_gpu, mem_status
from mae_anomaly.utils.experiment import (make_numbered_experiment_dir, resolve_dynamic_d_model,
                                          resolve_warmup_boundary, select_best_epoch)
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
    # SMAP / MSL (NASA Telemanom) — 'concat': all channels concatenated into one
    # multivariate stream (per-channel test safe-cut front→train/back→test;
    # run_boundaries segment-aware windowing). SMAP=54ch×25feat, MSL=27ch×55feat.
    # ('simple' per-channel variants generated dynamically below, like SMD.)
    {
        'key': 'SMAP_concat',
        'loader': 'SMAP_concat',
        'train_stride': 21,
        'normal50': False,
        'results_subdir': 'SMAP/concat',
    },
    {
        'key': 'MSL_concat',
        'loader': 'MSL_concat',
        'train_stride': 21,
        'normal50': False,
        'results_subdir': 'MSL/concat',
    },
    # SMD / Exathlon 'concat': all machines / all apps combined into one stream
    # (per-segment test-cut for SMD; per-app trace split merged for Exathlon).
    # 'simple' per-segment variants = SMD_simple_<machine> / Exathlon_simple_app<N>.
    {
        'key': 'SMD_concat',
        'loader': 'SMD_concat',
        'train_stride': 21,
        'normal50': False,
        'results_subdir': 'SMD/concat',
    },
    {
        'key': 'Exathlon_concat',
        'loader': 'Exathlon_concat',
        'train_stride': 21,
        'normal50': False,
        'results_subdir': 'Exathlon/concat',
    },
]
# NOTE: 비활성 variant들 (normal50, simulation_complex, SWaT_A1A2_swap)은 2026-05-17 자로
# default DATASETS에서 제외. loader 자체는 mae_anomaly/datasets/loaders.py에 유지되어
# 명시적 --dataset 호출 또는 과거 결과 로드는 가능. 활성 데이터셋은 위 5개 + 28 SMD + 6 Exathlon = 39개.

# Phase 4 (2026-05-29): the four "main comparison" datasets always preserve their
# best+latest checkpoints after training so that the corresponding model weights
# can be reloaded for ad-hoc inference and downstream analysis. For SWaT the
# excl22 best-epoch checkpoint is preserved alongside the full best-epoch
# checkpoint (the dual-eval path naturally produces both). Other datasets keep
# the legacy delete-after-inference behaviour to bound disk usage across the
# 28 SMD machines and 6 Exathlon apps. The ``KEEP_BEST_CKPT=1`` environment
# variable still takes precedence and applies to all datasets.
KEEP_CHECKPOINT_DATASETS = frozenset({
    'SWaT_A1A2',
    'WaDi_A1',
    'WaDi_A2',
    'PSM',
})

# SMD Simple Split datasets (28 machines, train+front50%test / back50%test)
# Dynamically generated from SMD_MACHINE_NAMES (mae_anomaly.datasets.loaders)
SMD_DATASETS = []
for _machine in SMD_MACHINE_NAMES:
    SMD_DATASETS.append({
        'key': f'SMD_simple_{_machine}',
        'loader': f'SMD_simple_{_machine}',
        'train_stride': 21,
        'normal50': False,
        'results_subdir': f'SMD/{_machine}',  # unchanged → aggregate_smd_results + existing dirs OK
    })
del _machine  # Clean up loop variable

# Exathlon Per-App datasets (6 apps × per-app, TimeSeAD 6-app convention)
# Apps {1, 2, 4, 5, 6, 9}. Apps 7/8 excluded (structural deficiency).
# Each app: all undisturbed traces + first floor(N_dist/2) disturbed → train, rest → test
EXATHLON_DATASETS = []
for _app in EXATHLON_APP_IDS:
    EXATHLON_DATASETS.append({
        'key': f'Exathlon_simple_app{_app}',
        'loader': f'Exathlon_simple_app{_app}',
        'train_stride': 21,
        'normal50': False,
        'results_subdir': f'Exathlon/app{_app}',  # unchanged → aggregate_exathlon_results + existing dirs OK
    })
del _app  # Clean up loop variable

# SMAP / MSL 'simple' per-channel datasets (one channel = one dataset, SMD-style).
# 'concat' (all channels in one stream) is the SMAP_concat / MSL_concat entry in DATASETS.
SMAP_MSL_SIMPLE_DATASETS = []
for _ch in SMAP_CHANNEL_NAMES:
    SMAP_MSL_SIMPLE_DATASETS.append({
        'key': f'SMAP_simple_{_ch}', 'loader': f'SMAP_simple_{_ch}',
        'train_stride': 21, 'normal50': False, 'results_subdir': f'SMAP/{_ch}',
    })
for _ch in MSL_CHANNEL_NAMES:
    SMAP_MSL_SIMPLE_DATASETS.append({
        'key': f'MSL_simple_{_ch}', 'loader': f'MSL_simple_{_ch}',
        'train_stride': 21, 'normal50': False, 'results_subdir': f'MSL/{_ch}',
    })
del _ch  # Clean up loop variable

# TEP type-disjoint generalization datasets (frozen streams in scripts/TEP/data/).
# Kept OUT of DATASETS so the default 5-base sweep is untouched — reachable only via
# explicit `--dataset TEP_typegen_<fold>`. train_stride is set to 1 (official forces
# stride=1 anyway). ffonly = B0 clean reference; the 4 fault folds share one frozen test.
TEP_TYPEGEN_DATASETS = [
    {'key': f'TEP_typegen_{_f}', 'loader': f'tep_typegen_{_f}',
     'train_stride': 1, 'normal50': False, 'results_subdir': f'TEP/typegen_{_f}'}
    for _f in ('ffonly', 'fstep', 'frand', 'fds', 'funk')
]
# Noisy-label (partial-label) variants: tag = LABELED % of seen-family faulty runs
# (lab80/lab50/lab25/lab10). Additional experiment SWEEP between A (100% labeled) and
# B (0% labeled) — both of which are the Phase-2 main matrix, so NOT re-run here.
# Full LASAD config — labels used on the labeled portion.
TEP_TYPEGEN_DATASETS += [
    {'key': f'TEP_typegen_{_f}_{_t}', 'loader': f'tep_typegen_{_f}_{_t}',
     'train_stride': 1, 'normal50': False, 'results_subdir': f'TEP/typegen_{_f}_{_t}'}
    for _f in ('fstep', 'frand', 'fds', 'funk') for _t in ('lab80', 'lab50', 'lab25', 'lab10')
]
# LOFO (leave-one-family-out) additional protocol: 3 seen families, 1 held out as unseen.
# _<ho> = held-out family excluded from train; _<ho>_cont = held-out also in train, unlabeled.
TEP_TYPEGEN_DATASETS += [
    {'key': f'TEP_typegen_lofo_{_h}{_c}', 'loader': f'tep_typegen_lofo_{_h}{_c}',
     'train_stride': 1, 'normal50': False, 'results_subdir': f'TEP/typegen_lofo_{_h}{_c}'}
    for _h in ('step', 'rand', 'ds', 'unk') for _c in ('', '_cont')
]


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


def _official_keep_ckpt_for(config, dataset_key):
    """[official] Resolve whether to keep this dataset's checkpoints. Global default
    = official_keep_checkpoints; per-dataset overrides come from official_ckpt_overrides
    ('key1:false,key2:true'). Returns True for the non-official path (unused there)."""
    keep = bool(getattr(config, 'official_keep_checkpoints', True))
    ov = getattr(config, 'official_ckpt_overrides', '') or ''
    for pair in str(ov).split(','):
        pair = pair.strip()
        if ':' in pair:
            dk, dv = pair.split(':', 1)
            if dk.strip() == dataset_key:
                keep = dv.strip().lower() in ('true', '1', 'yes', 'keep')
    return keep


def _official_train_seed(model, train_infer_loader, train_infer_dataset, config):
    """[official] Run an eval-style forward over the TRAIN set at the current
    weights and return the train-normal causal seed (R_tr, D_tr) = sum over
    label==0 of the point-level RAW teacher-recon / output-discrepancy. Mirrors
    the post-training train inference (same loader stride=test_stride, same
    aggregation), but produces only the two scalar sums."""
    from mae_anomaly.evaluator import _build_aggregation_map, _aggregate_with_map
    from mae_anomaly.scoring import compute_train_normal_seed
    ted = compute_epoch_test_inference(model, train_infer_loader, config,
                                       test_dataset=train_infer_dataset)
    tr_labels = np.array(train_infer_dataset.point_labels)
    tlen = len(tr_labels)
    tws = np.array(train_infer_dataset.window_start_indices)
    ft, fw, cov, cvd = _build_aggregation_map(tws, config.patch_size, config.num_patches, tlen)
    recon_pts = _aggregate_with_map(ted['recon_patches'].ravel(), ft, fw, cov, cvd, tlen, method='mean')
    disc_pts = _aggregate_with_map(ted['disc_patches'].ravel(), ft, fw, cov, cvd, tlen, method='mean')
    return compute_train_normal_seed(recon_pts, disc_pts, tr_labels)


def _evaluate_all_parallel(evaluator, executor, also_excl22: bool = False, official_seed=None):
    """Dispatch the 3-4 compute_full_metric_set calls (adaptive / disc / teacher_recon /
    optional adaptive-excl22) in parallel via the executor.

    Returns (metrics, disc_metrics, teacher_recon_metrics) where `metrics` carries
    the adaptive-full result, the excl22 keys merged in (when also_excl22), and the
    disturbing-normal stats (computed sequentially after the parallel work).

    Architecture: the heavy work (compute_full_metric_set, ~5-7 s/call) is module-
    level and picklable. We compute point-level scores for each score type inside
    the evaluator (cheap, ~100 ms each, single-process), then submit the
    independent metric computations to the pool.

    Important: this function is only valid inside a single Python interpreter
    that imported mae_anomaly.evaluator. Worker processes import the same module
    on Pool startup (spawn context).
    """
    from mae_anomaly.evaluator import (
        compute_full_metric_set, compute_metrics_with_exclusion,
        find_swat_region_22, _zero_metric_set, _aggregate_with_map,
    )
    score_mode = evaluator.config.anomaly_score_mode
    cached = evaluator._get_cached_scores()
    sample_types = cached['sample_types']
    _fm_p = evaluator.fm_patches if hasattr(evaluator, 'fm_patches') else None

    if not (evaluator.can_compute_point_level_pa_k
            and hasattr(evaluator.test_dataset, 'anomaly_regions')):
        # Degenerate dataset — match sequential fallback
        z = _zero_metric_set()
        return z, z, z

    pt_labels = np.array(evaluator.test_dataset.point_labels)
    total_len = len(pt_labels)
    anomaly_regions = evaluator.test_dataset.anomaly_regions
    eval_mask = np.ones(total_len, dtype=bool)
    flat_t, flat_wp, coverage, covered = evaluator._get_aggregation_map()

    def _agg(patch_scores_):
        return np.nan_to_num(
            _aggregate_with_map(patch_scores_.ravel(), flat_t, flat_wp, coverage, covered, total_len, method='mean'),
            nan=0.0,
        )

    # Point scores per score type (cheap, sequential)
    adaptive_patch_scores = evaluator._apply_scoring_formula(
        cached['patch_recon'], cached['patch_disc'], score_mode, fm=_fm_p
    )
    adaptive_pts = _agg(adaptive_patch_scores)
    disc_pts = _agg(cached['patch_disc'])
    teacher_pts = _agg(cached['patch_recon'])

    # === (2026-06-06) float32 best-epoch-selection parity with offline recompute ===
    # Option (III): the per-epoch best-epoch-selection metric MUST be computed on the
    # SAME score precision as the saved epoch_scores npz (float32), so that offline
    # recompute (which reads the float32 npz) and the inline training-time best-epoch
    # selection agree bit-for-bit. Without this, float32 truncation of the npz vs the
    # float64 inline `adaptive_pts` shifts the PA%K-AUC PRC threshold sampling
    # (tadpak INTERVAL=10 over *unique* score values) by ~1e-3, which flips the
    # selected best epoch (e.g. 445 vs 460). The npz saves
    # `nan_to_num(score).astype(float32)`; `_agg` already applies nan_to_num, so the
    # `.astype(float32)` below is bit-identical to the stored npz score.
    adaptive_pts = adaptive_pts.astype(np.float32)
    disc_pts = disc_pts.astype(np.float32)
    teacher_pts = teacher_pts.astype(np.float32)

    # [official] Replace the adaptive point score with the causal/online score
    # score_t = recon_test[t] + 0.25*disc_test[t]*s_t, where teacher_pts/disc_pts
    # are the point-level RAW recon/disc just computed and (R_tr, D_tr) come from a
    # per-epoch train-normal inference. The downstream f_adaptive/f_excl22 metrics
    # — and therefore best-epoch selection (pak_auc_f1) — become causal-based.
    # f_disc/f_teacher stay raw (diagnostics). No-op when official_seed is None.
    if official_seed is not None and getattr(evaluator.config, 'official', False):
        from mae_anomaly.scoring import compute_official_causal_score
        _R_tr, _D_tr = official_seed
        adaptive_pts = compute_official_causal_score(
            teacher_pts, disc_pts, R_tr=_R_tr, D_tr=_D_tr,
            force_recon_only=evaluator._force_recon_only).astype(np.float32)

    # Dispatch parallel compute_full_metric_set calls.
    # Phase 3 (2026-05-29): compute_full_metric_set requires lite as kw-only.
    # The previous positional True was silent — callers could swap n_thresholds
    # and lite without noticing. Explicit kwarg removes that risk.
    f_adaptive = executor.submit(compute_full_metric_set,
                                 adaptive_pts, pt_labels, anomaly_regions, eval_mask,
                                 200, 100, lite=True)
    f_disc = executor.submit(compute_full_metric_set,
                             disc_pts, pt_labels, anomaly_regions, eval_mask,
                             200, 100, lite=True)
    f_teacher = executor.submit(compute_full_metric_set,
                                teacher_pts, pt_labels, anomaly_regions, eval_mask,
                                200, 100, lite=True)
    f_excl22 = None
    excl22_region = None
    if also_excl22:
        excl22_region = find_swat_region_22(anomaly_regions)
        if excl22_region is not None:
            f_excl22 = executor.submit(compute_metrics_with_exclusion,
                                       adaptive_pts, pt_labels, anomaly_regions,
                                       excl22_region, lite=True)

    metrics = f_adaptive.result()
    disc_metrics = f_disc.result()
    teacher_recon_metrics = f_teacher.result()

    if f_excl22 is not None:
        excl22_results = f_excl22.result()
        if excl22_results:
            for k, v in excl22_results.items():
                if not k.startswith('_'):
                    metrics[f'excl22_{k}'] = v
            metrics['excl22_region_start'] = int(excl22_region.start)
            metrics['excl22_region_end'] = int(excl22_region.end)
            metrics['excl22_region_length'] = int(excl22_region.end - excl22_region.start)

    # Disturbing normal stats (sequential, uses window-level scores — not in compute_full_metric_set)
    if 'optimal_threshold' in metrics:
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        _fm_w = evaluator._get_cached_fm_scores()
        window_scores = evaluator._apply_scoring_formula(
            cached['window_recon'], cached['window_disc'], score_mode, fm=_fm_w
        )
        window_scores = np.nan_to_num(window_scores, nan=0.0)
        disturbing_mask = (sample_types == 0) | (sample_types == 1)
        if disturbing_mask.sum() > 0:
            disturbing_scores = window_scores[disturbing_mask]
            disturbing_labels = sample_types[disturbing_mask]
            if len(np.unique(disturbing_labels)) > 1:
                d_predictions = (disturbing_scores > metrics['optimal_threshold']).astype(int)
                metrics['disturbing_roc_auc'] = float(roc_auc_score(disturbing_labels, disturbing_scores))
                metrics['disturbing_precision'] = float(precision_score(disturbing_labels, d_predictions, zero_division=0))
                metrics['disturbing_recall'] = float(recall_score(disturbing_labels, d_predictions, zero_division=0))
                metrics['disturbing_f1'] = float(f1_score(disturbing_labels, d_predictions, zero_division=0))
                metrics['n_pure_normal'] = int((sample_types == 0).sum())
                metrics['n_disturbing_normal'] = int((sample_types == 1).sum())
                metrics['n_anomaly'] = int((sample_types == 2).sum())

    return metrics, disc_metrics, teacher_recon_metrics


def _read_best_epoch_metric_set(exp_dir, best_epoch):
    """(2026-06-23) Read the best epoch's already-computed metric set from
    epoch_metrics.json — the per-epoch eval computed it (it SELECTED best_epoch), so the
    final bg-worker reads it instead of recomputing identical numbers. Returns
    (adaptive_metrics, teacher_recon_metrics): adaptive = plain keys, teacher = the
    'teacher_'-prefixed keys with the prefix stripped. (None, None) if unavailable."""
    import json as _json
    try:
        with open(os.path.join(exp_dir, 'epoch_metrics.json')) as _f:
            em = _json.load(_f)
        rows = em.get('epochs') if isinstance(em, dict) else em
        row = next((r for r in rows if isinstance(r, dict) and r.get('epoch') == best_epoch), None)
        if row is None:
            return None, None
        adaptive, teacher = {}, {}
        for k, v in row.items():
            if k.startswith('_'):
                continue
            if k.startswith('teacher_'):
                teacher[k[len('teacher_'):]] = v
            else:
                adaptive[k] = v
        return adaptive, teacher
    except Exception as _e:  # noqa: BLE001
        print(f"  [epoch_metrics read warn] {type(_e).__name__}: {_e}", flush=True)
        return None, None


def _score_type_metrics_parallel(evaluator, score_types, exp_name):
    """(2026-06-23) Compute per-score-type metric sets (disc / student_recon — the ones
    NOT persisted per-epoch) in parallel. Mirrors evaluate_by_score_type (same
    mean-aggregation + compute_full_metric_set) but dispatches the GIL-bound PA%K sweeps
    to a ProcessPool (threads don't help — pure-Python threshold loop). bg-worker is
    non-daemon + spawn, so a child pool is safe and inherits MAE_SKIP_VUS. Falls back to
    serial on any pool error."""
    from mae_anomaly.evaluator import (
        compute_full_metric_set as _cfms, _aggregate_with_map as _agg_map, _zero_metric_set,
    )
    cached = evaluator._get_cached_scores()
    if not (evaluator.can_compute_point_level_pa_k
            and hasattr(evaluator.test_dataset, 'anomaly_regions')):
        return [_zero_metric_set() for _ in score_types]
    pt_labels = np.array(evaluator.test_dataset.point_labels)
    total_len = len(pt_labels)
    regions = evaluator.test_dataset.anomaly_regions
    mask = np.ones(total_len, dtype=bool)
    ft, fwp, cov, covd = evaluator._get_aggregation_map()
    kmap = {'disc': 'patch_disc', 'teacher_recon': 'patch_recon', 'student_recon': 'patch_student_recon'}
    def _pts(st):
        ps = cached[kmap[st]]
        return np.nan_to_num(
            _agg_map(ps.ravel(), ft, fwp, cov, covd, total_len, method='mean'), nan=0.0
        ).astype(np.float32)
    arrays = [_pts(st) for st in score_types]
    try:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as _mp
        try:
            nw = len(os.sched_getaffinity(0))
        except Exception:
            nw = os.cpu_count() or 4
        with ProcessPoolExecutor(max_workers=max(1, min(len(score_types), nw)),
                                 mp_context=_mp.get_context('spawn')) as pool:
            futs = [pool.submit(_cfms, a, pt_labels, regions, mask, 200, 100, lite=False)
                    for a in arrays]
            out = []
            for st, f in zip(score_types, futs):
                out.append(f.result())
                print(f"  [{exp_name}] eval: {st} done", flush=True)
            return out
    except Exception as _e:  # noqa: BLE001
        print(f"  [{exp_name}] WARN parallel score-type failed "
              f"({type(_e).__name__}: {_e}) → serial fallback", flush=True)
        return [_cfms(a, pt_labels, regions, mask, 200, 100, lite=False) for a in arrays]


def compute_epoch_test_eval(eval_data, config, test_loader, test_dataset=None,
                            dataset_key=None, executor=None, *, epoch=None):
    """CPU phase: point-level evaluation from precomputed patch scores.

    Pure numpy/sklearn operations — no GPU needed. Safe to run in a background thread.

    Args:
        dataset_key: dataset key (e.g. 'SWaT_A1A2'). When 'SWaT' in key, eval also
            computes excl22-masked metrics in the same pass (Evaluator.evaluate's
            also_excl22=True). Result dict gains `excl22_*` keys.
            Added 2026-05-27 for per-epoch SWaT dual-eval monitoring.
        executor: optional concurrent.futures.ProcessPoolExecutor. When provided,
            the heavy `compute_full_metric_set` calls (3 score types + 1 excl22
            for SWaT = up to 4 calls) are dispatched in parallel across worker
            processes. Each call sends ~2 MB of pickled data and runs ~5-7 s of
            CPU work, so IPC overhead is negligible relative to work (~0.5 %).
            Expected wall-time: ~7 s instead of ~26 s on SWaT, ~5 s instead of
            ~20 s elsewhere. None = sequential (original behavior).
            Added 2026-05-28 (Option A+C, outer-level parallelism).
        epoch: 1-indexed eval epoch (``ep = trained_epoch + 1``). When inside
            the teacher-only warmup window (``0 < epoch <=
            teacher_only_warmup_epochs``) the adaptive anomaly score drops the
            frozen-student disc/FM terms (recon-only) via
            ``evaluator.set_eval_context``. ``None`` (default, post-hoc helper
            callers) → legacy full scoring. Added 2026-06-01 (pre-warmup
            recon-only fix). Keyword-only so existing positional callers cannot
            silently mis-pass it.
    """
    t_eval = time.time()
    evaluator = Evaluator(None, config, test_loader, test_dataset=test_dataset)
    from mae_anomaly.types import PatchScoresBundle
    evaluator.set_precomputed_patch_scores(
        PatchScoresBundle.from_eval_data(eval_data)
    )
    # Pre-warmup gate: mark whether this eval epoch is teacher-only so the
    # adaptive score reduces to teacher reconstruction (the random-init student
    # disc/FM must not leak into the score). No-op (full scoring) when epoch is
    # None or post-warmup. Affects BOTH the executor and sequential branches
    # below because they read evaluator._force_recon_only in-process.
    evaluator.set_eval_context(epoch=epoch)
    # Enable dual eval (full + excl22) for SWaT datasets only
    also_excl22 = bool(dataset_key and 'SWaT' in dataset_key)
    # [official] Causal-score seed (R_tr, D_tr) staged by epoch_eval_callback on
    # eval_data. None ⇒ adaptive scoring (non-official path unchanged).
    _official_seed = None
    if getattr(config, 'official', False) and 'official_R_tr' in eval_data:
        _official_seed = (eval_data['official_R_tr'], eval_data['official_D_tr'])
    if executor is not None:
        # Fast path: dispatch the 3-4 compute_full_metric_set calls in parallel.
        # Each call is a CPU-bound chunk of ~5-7 s; sending 4 in parallel to 4
        # worker processes gives ~4x speedup. Heavy per-K loops stay sequential
        # inside each call (IPC too coarse for inner-level parallelism — verified
        # 2026-05-28 to be IPC-bound when split per-K).
        metrics, disc_metrics, teacher_recon_metrics = _evaluate_all_parallel(
            evaluator, executor, also_excl22=also_excl22, official_seed=_official_seed,
        )
    else:
        # Per-epoch fallback path (no executor) — lite=True for speed (VUS skip).
        # Phase 3 (2026-05-29): lite is kw-only required to prevent path-dependent
        # silent defaults.
        metrics = evaluator.evaluate(also_excl22=also_excl22, lite=True)
        disc_metrics = evaluator.evaluate_by_score_type('disc', lite=True)
        teacher_recon_metrics = evaluator.evaluate_by_score_type('teacher_recon', lite=True)
    eval_time = time.time() - t_eval

    # Compute disc_SNR + recon_SNR from patch-level scores.
    # disc_snr  = (disc_anomaly  - disc_normal)  / (σ_disc_a + σ_disc_n + ε)  — student-teacher discrepancy separation
    # recon_snr = (recon_anomaly - recon_normal) / (σ_recon_a + σ_recon_n + ε) — teacher-only recon separation
    # Both are Cohen's-d-style effect sizes. Positive = anomaly higher than normal (expected).
    detailed_losses = evaluator.compute_detailed_losses()
    loss_stats = compute_loss_statistics(detailed_losses)
    metrics['disc_snr'] = loss_stats.get('disc_SNR', 0)
    metrics['recon_snr'] = loss_stats.get('recon_SNR', 0)

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
        # 'fm' is required so downstream PatchScoresBundle.from_patch_scores_dict
        # can route FM through; missing key would silently drop FM, which was
        # the root cause of the 2026-05-28 FM-omission bug.
        'fm': eval_data.get('fm_patches'),
    }

    return metrics, patch_scores_dict


def compute_contrib_from_eval_data(eval_data, config, *, epoch=None):
    """Compute contribution ratios from pre-computed patch scores (pure numpy, no GPU).

    Uses the same eval_data already produced by compute_epoch_test_inference(),
    so no additional model inference is needed.

    ``epoch`` (1-indexed, keyword-only): during the teacher-only warmup window
    the contribution chart must reflect the recon-only score (student disc/FM
    contribution = 0), matching the gated anomaly score. ``None`` → legacy full
    contribution. Added 2026-06-01 (pre-warmup recon-only fix).
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

    # Weighted contributions based on scoring mode.
    # SINGLE SOURCE: mae_anomaly.scoring.compute_adaptive_components.
    if config.anomaly_score_mode == 'adaptive':
        from mae_anomaly.scoring import compute_adaptive_components, is_prewarmup_epoch
        comps = compute_adaptive_components(
            recon_all, disc_all, fm_all, config,
            force_recon_only=is_prewarmup_epoch(config, epoch),
        )
        recon_contrib_all = recon_all
        disc_contrib_all = comps['student_error']  # entire student error (disc + fm)
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
# Post-training VUS sweep on saved NPZ files
# (Self-contained helper used by the bg-worker so that the epoch_dashboard can
#  show vus_pr / vus_roc / affiliation_f1 trends without inflating per-epoch
#  training-time evaluation. Per-epoch eval keeps lite=True (skip VUS); the
#  bg-worker runs this sweep once after training finishes, in parallel
#  process-pool with max_workers=4. See REFACTOR_PLAN_v3 Phase 5.)
# =============================================================================

def _compute_vus_for_npz_file(npz_path: str, region_starts, region_ends,
                              excl_start=None, excl_end=None) -> dict:
    """Compute lite=False metrics (vus_pr, vus_roc, affiliation_f1, r_based_f1)
    for one saved epoch_NNN_scores.npz file.

    Picklable: keeps anomaly_regions as plain start/end arrays so the worker
    process does not need to re-import the AnomalyRegion class. Returns only
    the four "extra" metric keys; the bg-worker merges them into the existing
    epoch_metrics.json entry for that epoch.

    Args:
        excl_start, excl_end: optional region to drop from the score/label
            arrays before metric computation (excl22 worker case). Slicing is
            mathematically equivalent to region exclusion because the four
            "extra" metrics consume only 1-D `point_scores` and `point_labels`
            arrays (see compute_extra_metrics signature in evaluator.py L669).
    """
    # All imports inside the function body so this is fully self-contained
    # for ProcessPoolExecutor (spawn-friendly).
    import numpy as np
    from types import SimpleNamespace
    from mae_anomaly.evaluator import compute_full_metric_set

    d = np.load(npz_path)
    # [official] Use the causal score when present. Only official runs save the
    # 'official_score' key, so non-official npz files transparently fall back to
    # 'adaptive_score' (this function has no config, so npz-presence is the gate).
    score = d['official_score'] if 'official_score' in d.files else d['adaptive_score']
    labels = d['point_labels'].astype(np.int8)

    # excl22 worker: physically remove region22 timesteps so VUS / Aff / R-F1
    # see the same data the per-epoch excl22 sweep saw.
    if excl_start is not None and excl_end is not None:
        es, ee = int(excl_start), int(excl_end)
        keep = np.ones(len(labels), dtype=bool)
        keep[es:ee] = False
        score = score[keep]
        labels = labels[keep]
        # Filter anomaly_regions: drop region22, shift trailing regions left
        # by the cut length so indices stay consistent with the new arrays.
        # compute_extra_metrics doesn't read `regions`, but
        # compute_full_metric_set's PA%K path does, so we keep them correct.
        cut_len = ee - es
        regions = []
        for s, e in zip(region_starts, region_ends):
            si, ei = int(s), int(e)
            if si == es and ei == ee:
                continue  # this is region22 itself
            if si >= ee:
                si -= cut_len
                ei -= cut_len
            regions.append(SimpleNamespace(start=si, end=ei))
    else:
        regions = [SimpleNamespace(start=int(s), end=int(e))
                   for s, e in zip(region_starts, region_ends)]

    m = compute_full_metric_set(
        score, labels, regions,
        eval_mask=None, lite=False,  # lite=False so VUS is actually computed
    )
    return {
        'vus_pr': float(m.get('vus_pr', 0.0)),
        'vus_roc': float(m.get('vus_roc', 0.0)),
        'affiliation_f1': float(m.get('affiliation_f1', 0.0)),
        'r_based_f1': float(m.get('r_based_f1', 0.0)),
    }


def _run_vus_sweep_on_saved_npz(
    exp_dir: str,
    anomaly_regions,
    *,
    max_workers: int = 4,
    log_prefix: str = "",
    excl_region=None,
) -> dict:
    """Sweep VUS-PR / VUS-ROC / Affiliation-F1 / R-based-F1 over every saved
    epoch_NNN_scores.npz under ``exp_dir/epoch_scores/`` in parallel.

    Per-epoch evaluation during training runs lite=True for speed (VUS skipped).
    This sweep fills in the missing values **once** after training finishes so
    the epoch_dashboard can plot complete trends for these metrics. The work
    runs in the bg-worker (CPU-only), so it does not contend with the main
    process's GPU training of the next dataset.

    Args:
        exp_dir: experiment directory containing ``epoch_scores/``.
        anomaly_regions: list of region objects with ``.start`` / ``.end``;
            the same regions used for training-time evaluation. They are
            converted to plain int arrays before dispatch (picklable).
        max_workers: ProcessPoolExecutor pool size for parallel VUS calls.
        log_prefix: prefix for progress prints, typically ``"[<exp_name>] "``.

    Returns:
        Dict mapping epoch number (int) to a dict with keys
        ``{'vus_pr','vus_roc','affiliation_f1','r_based_f1'}``. Epochs whose
        sweep crashed are skipped silently (a warning is printed).
    """
    import os
    import glob
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    score_dir = os.path.join(exp_dir, 'epoch_scores')
    if not os.path.isdir(score_dir):
        return {}
    npz_files = sorted(glob.glob(os.path.join(score_dir, 'epoch_*_scores.npz')))
    if not npz_files:
        return {}

    # Flatten regions to int arrays (picklable)
    r_starts = [int(r.start) for r in anomaly_regions]
    r_ends = [int(r.end) for r in anomaly_regions]
    # Optional excl_region (excl22 worker only) → flatten for picklable dispatch
    excl_s = int(excl_region.start) if excl_region is not None else None
    excl_e = int(excl_region.end) if excl_region is not None else None
    _mode_tag = "excl22-sliced" if excl_region is not None else "full"

    print(f"  {log_prefix}VUS sweep starting ({_mode_tag}): {len(npz_files)} epochs × max_workers={max_workers}", flush=True)
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for npz_path in npz_files:
            base = os.path.basename(npz_path)
            # 'epoch_005_scores.npz' → 5
            try:
                ep = int(base.split('_')[1])
            except (IndexError, ValueError):
                continue
            futures[executor.submit(_compute_vus_for_npz_file, npz_path, r_starts, r_ends, excl_s, excl_e)] = ep
        done = 0
        for fut in as_completed(futures):
            ep = futures[fut]
            try:
                results[ep] = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  {log_prefix}VUS sweep ep{ep:>3} FAILED: {type(exc).__name__}: {exc}", flush=True)
                results[ep] = {}
            done += 1
            if done % 20 == 0:
                elapsed = time.time() - t0
                print(f"  {log_prefix}VUS sweep progress: {done}/{len(futures)} ({elapsed:.0f}s)", flush=True)
    print(f"  {log_prefix}VUS sweep done: {len(results)} epochs in {time.time() - t0:.0f}s", flush=True)
    return results


def _apply_vus_sweep_results(epoch_metrics_list: list, vus_results: dict) -> list:
    """Merge VUS sweep results into an epoch_metrics list (in place)."""
    for entry in epoch_metrics_list:
        ep = entry.get('epoch')
        if ep in vus_results and vus_results[ep]:
            entry.update(vus_results[ep])
    return epoch_metrics_list


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

    # 5. Combined summary dashboard (3x3).
    # Phase 5 (2026-05-29): row 2 added (vus_pr / vus_roc / affiliation_f1).
    # Per-epoch eval runs lite=True so VUS values default to 0 until the
    # bg-worker's post-training VUS sweep populates them via
    # _run_vus_sweep_on_saved_npz. The bg-worker re-renders this dashboard
    # after the sweep so the final user-facing PNG includes the curves.
    # The initial dashboard rendered by the main process shows zeros for the
    # bottom row; this is intentional and clearly marked in the panel titles.
    def _any_nonzero(key):
        return any(float(m.get(key, 0)) != 0 for m in epoch_metrics_list)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

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

    # Row 2 — VUS / Affiliation (populated by post-training VUS sweep)
    _vus_pr_filled = _any_nonzero('vus_pr')
    _vus_roc_filled = _any_nonzero('vus_roc')
    _aff_filled = _any_nonzero('affiliation_f1')

    ax = axes[2][0]
    ax.plot(epochs, [m.get('vus_pr', 0) for m in epoch_metrics_list],
            color='#16a085', marker='s', markersize=4, linewidth=1.5)
    ax.set_title('VUS-PR' if _vus_pr_filled else 'VUS-PR  (pending bg-worker sweep)')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[2][1]
    ax.plot(epochs, [m.get('vus_roc', 0) for m in epoch_metrics_list],
            color='#2980b9', marker='s', markersize=4, linewidth=1.5)
    ax.set_title('VUS-ROC' if _vus_roc_filled else 'VUS-ROC  (pending bg-worker sweep)')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[2][2]
    ax.plot(epochs, [m.get('affiliation_f1', 0) for m in epoch_metrics_list],
            color='#c0392b', label='Affiliation F1', marker='o', markersize=4, linewidth=1.5)
    ax.plot(epochs, [m.get('r_based_f1', 0) for m in epoch_metrics_list],
            color='#e67e22', label='R-based F1', marker='^', markersize=4, linewidth=1.5, alpha=0.8)
    ax.set_title('Range-based metrics' if _aff_filled else 'Range-based metrics  (pending sweep)')
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
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

        # Accuracy breakdown (balanced + anomaly + normal) + degeneracy gap.
        # balanced_acc=0.5 is DECEPTIVE: it coexists with full degeneracy
        # (normal=0, anomaly=1). The |normal-anomaly| gap disambiguates —
        # gap≈0 with balanced high = genuine; gap≈1 = degenerate (one class only).
        ax = axes[1]
        _aac = [m.get('grl_anomaly_acc', 0) for m in epoch_metrics_list]
        _nac = [m.get('grl_normal_acc', 0) for m in epoch_metrics_list]
        # prefer stored grl_acc_gap; fall back to deriving from the components
        _gap = [m.get('grl_acc_gap') if m.get('grl_acc_gap') is not None
                else abs((m.get('grl_normal_acc') or 0) - (m.get('grl_anomaly_acc') or 0))
                for m in epoch_metrics_list]
        ax.plot(epochs, [m.get('grl_balanced_acc', 0) for m in epoch_metrics_list],
                color='#1565C0', lw=2, marker='s', ms=3, label='Balanced Acc')
        ax.plot(epochs, _aac, color='#C62828', lw=1.5, ls='--', marker='^', ms=3, label='Anomaly Acc (TPR)')
        ax.plot(epochs, _nac, color='#2E7D32', lw=1.5, ls='--', marker='v', ms=3, label='Normal Acc (TNR)')
        ax.plot(epochs, _gap, color='#E65100', lw=2.5, marker='o', ms=3,
                label='|Normal−Anomaly| Gap (degeneracy)')
        # shade epochs where the gap signals degeneracy (one class fully ignored)
        ax.fill_between(epochs, 0, 1.05, where=[g >= 0.8 for g in _gap],
                        color='#E65100', alpha=0.10, label='Degenerate (gap≥0.8)')
        ax.axhline(y=0.50, color='gray', ls=':', lw=1, alpha=0.5, label='Random (0.50)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy / Gap')
        ax.set_ylim([0.0, 1.05])
        ax.set_title('GRL Accuracy + Degeneracy Gap')
        ax.legend(fontsize=6)
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
    # CPU throttling — MUST be set BEFORE numpy/sklearn/matplotlib import so
    # OpenBLAS / MKL / OMP read the limit at library init time (not after the
    # thread pool is already created). 2026-05-29 fix: env was previously set
    # AFTER matplotlib import (which loads numpy), so the limit was ignored and
    # each bg-worker used ~7 cores via OpenBLAS, saturating CPU and starving
    # main process dataloader during the next dataset's training. Reducing
    # value from 4 → 2 also tightens the per-process budget so 2 bg-workers
    # (SWaT dual eval) × 2 internal pool workers × 2 OMP threads = 8 cores
    # total max, leaving the rest for main GPU training.
    import sys
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['NUMEXPR_NUM_THREADS'] = '2'

    sys.path.insert(0, PROJECT_ROOT)
    from mae_anomaly import Config, set_seed
    from mae_anomaly.dataset_sliding import SlidingWindowDataset, AnomalyRegion
    from mae_anomaly.evaluator import Evaluator, DatasetMetadata
    from mae_anomaly.visualization import setup_style, BestModelVisualizer
    from scripts.ablation.run_ablation import compute_loss_statistics

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # CPU process priority + torch thread limit (post-import, OK)
    os.nice(19)
    torch.set_num_threads(2)
    # Dynamic CPU affinity — pin bg-worker to the LAST N cores so main process
    # GPU training (using cores 0..M-N-1 by default) is fully isolated from
    # bg-worker CPU saturation. 2026-05-29 fix: previous hardcode `range(16,24)`
    # required ≥24 cores; on 16-core systems the call raised silently and no
    # affinity was applied → bg-worker grabbed all cores. New formula:
    # n_bg = max(2, cpu_count // 4) — 16c → 4, 24c → 6, 32c → 8. Guarantees
    # at least 2 cores for bg-worker and at most 25% of system.
    try:
        _total_cores = os.cpu_count() or 16
        _n_bg = max(2, _total_cores // 4)
        _bg_cores = set(range(_total_cores - _n_bg, _total_cores))
        os.sched_setaffinity(0, _bg_cores)
        # === Patch 3 (2026-05-29): All-threads taskset enforcement ===
        # os.sched_setaffinity(0, ...) above only pins the calling thread.
        # BLAS / numpy / sklearn pthread_create children inherit the parent's
        # affinity at creation time, but threads created BEFORE this line
        # (during the numpy import inside multiprocessing.spawn unpickling)
        # already exist with cpus_allowed=[0..total-1]. Use `taskset -pca`
        # which iterates /proc/<pid>/task/* and sets affinity for ALL threads
        # (existing + future). Result: bg-worker has zero affinity leak.
        try:
            import subprocess as _subprocess
            _aff_str = f"{_total_cores - _n_bg}-{_total_cores - 1}"
            _r = _subprocess.run(
                ['taskset', '-pac', _aff_str, str(os.getpid())],
                check=False, capture_output=True, text=True,
            )
            if _r.returncode != 0:
                print(f"  [{exp_name}] WARN taskset failed (rc={_r.returncode}): "
                      f"{_r.stderr.strip()[:200]}", flush=True)
            else:
                print(f"  [{exp_name}] CPU affinity hard-pinned to cores {_aff_str} "
                      f"(all threads via taskset -pac)", flush=True)
        except FileNotFoundError:
            print(f"  [{exp_name}] WARN taskset command not found — affinity may leak", flush=True)
        except Exception as _ts_e:  # noqa: BLE001
            print(f"  [{exp_name}] WARN taskset exception: {_ts_e}", flush=True)
    except Exception:
        pass
    warnings.filterwarnings('ignore')

    print(f"  [{exp_name}] Background CPU eval+viz started (PID={os.getpid()})", flush=True)

    # === RECOVERY PAYLOAD (2026-05-28): dump all IPC inputs to disk BEFORE any work. ===
    # If this bg-worker crashes for any reason (code bug, OOM, etc.), the data
    # main process sent via IPC would otherwise be lost forever (main process
    # already discarded its copy). Dumping to disk preserves it so `scripts/
    # recover_bg_worker.py` can re-run the bg-worker logic later (after fixing
    # the underlying bug) WITHOUT losing the training results that took hours
    # to produce. The file is DELETED on full success — zero permanent footprint.
    import pickle as _pickle_recovery
    _recovery_tag = swat_eval_mode if swat_eval_mode else 'full'
    _recovery_path = os.path.join(exp_dir, f'_bg_worker_recovery_{_recovery_tag}.pkl')
    try:
        os.makedirs(exp_dir, exist_ok=True)
        with open(_recovery_path, 'wb') as _rf:
            _pickle_recovery.dump({
                'exp_name': exp_name, 'exp_dir': exp_dir, 'config_dict': config_dict,
                'signals': signals, 'point_labels': point_labels,
                'anomaly_regions_ser': anomaly_regions_ser, 'history': history,
                'train_ratio': train_ratio, 'timing': timing,
                'patch_scores': patch_scores, 'pred_data': pred_data,
                'detailed_data': detailed_data, 'progress_info': progress_info,
                'window_start_indices': window_start_indices,
                'swat_eval_mode': swat_eval_mode,
            }, _rf, protocol=_pickle_recovery.HIGHEST_PROTOCOL)
        print(f"  [{exp_name}] Recovery payload saved: {_recovery_path}", flush=True)
    except Exception as _re:
        print(f"  [{exp_name}] WARN: recovery payload dump failed ({_re}) — proceeding without safety net", flush=True)

    # Wrap entire bg-worker body so we can preserve recovery file on any failure
    try:
        _bg_worker_body(
            exp_name, exp_dir, config_dict, signals, point_labels,
            anomaly_regions_ser, history, train_ratio, timing,
            patch_scores, pred_data, detailed_data,
            progress_info, window_start_indices, swat_eval_mode,
        )
    except Exception as _exc:
        print(f"  [{exp_name}] BG-WORKER FAILED — recovery payload preserved at {_recovery_path}", flush=True)
        print(f"  [{exp_name}] To retry after fixing the bug: `python scripts/recover_bg_worker.py`", flush=True)
        raise
    else:
        # Success — delete recovery payload (zero permanent footprint)
        try:
            os.unlink(_recovery_path)
        except OSError:
            pass


def _bg_worker_body(exp_name, exp_dir, config_dict, signals, point_labels,
                    anomaly_regions_ser, history, train_ratio, timing,
                    patch_scores, pred_data, detailed_data,
                    progress_info, window_start_indices, swat_eval_mode):
    """Original bg-worker work — split out so _cpu_eval_viz_worker can wrap with
    recovery-payload try/except. All imports/setup live in the caller; this
    function only contains the heavy work."""
    # Re-import (cheap; modules already cached in this subprocess)
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
        from mae_anomaly.utils.experiment import resolve_test_stride
        test_length = len(test_point_labels)
        stride = resolve_test_stride(config)
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
    from mae_anomaly.types import PatchScoresBundle
    evaluator.set_precomputed_patch_scores(
        PatchScoresBundle.from_patch_scores_dict(patch_scores)
    )
    print(f"  [{exp_name}] Computing metrics...", flush=True)
    # Pre-warmup gate (2026-06-01): patch_scores were produced by the best-epoch
    # weights (best_checkpoint.pt). If that epoch is inside the teacher-only
    # warmup window, this final eval must use the recon-only score so the saved
    # experiment_metadata metrics match the gated per-epoch metrics that selected
    # the best epoch. timing['best_epoch'] is the 1-indexed best epoch.
    evaluator.set_eval_context(epoch=timing.get('best_epoch'))
    # (2026-06-23) GATE the final-eval path by MAE_SKIP_VUS so BASE / non-TEP datasets are
    # 100% unchanged. Only VUS-off runs (TEP) take the no-recompute read path; every other
    # dataset keeps the ORIGINAL full final eval verbatim (which legitimately computes VUS
    # — not redundant there — plus the npz@best misalignment finalize).
    if os.environ.get('MAE_SKIP_VUS') == '1':
        # --- TEP / VUS-off: NO REDUNDANT RECOMPUTE ---
        # The per-epoch eval already computed + saved the best epoch's full metric set to
        # epoch_metrics.json (it literally SELECTED best_epoch); with VUS off the final
        # eval adds nothing, so READ adaptive + teacher from epoch_metrics[best] (verified:
        # epoch_metrics[best].pak_auc_f1 == old recompute to 5 dp). Only disc/student
        # score-type diagnostics aren't persisted per-epoch → compute just those, parallel.
        metrics, teacher_recon_metrics = _read_best_epoch_metric_set(exp_dir, timing.get('best_epoch'))
        evaluator._get_cached_scores()  # cheap (~100ms): populate cache for per-fault/detailed/viz
        if metrics is None:
            print(f"  [{exp_name}] WARN epoch_metrics@best unavailable → computing primary metrics", flush=True)
            metrics = evaluator.evaluate(lite=False)
            teacher_recon_metrics = evaluator.evaluate_by_score_type('teacher_recon', lite=False)
        disc_metrics, student_recon_metrics = _score_type_metrics_parallel(
            evaluator, ['disc', 'student_recon'], exp_name)
        eval_time = time.time() - eval_start
    else:
        # --- base / non-TEP (VUS-on final report): ORIGINAL full final eval, UNCHANGED ---
        # 2026-05-28: lite=False for final bg-worker eval — populates VUS in primary
        # metrics dict (per-epoch eval still uses lite=True for speed).
        metrics = evaluator.evaluate(lite=False)
        eval_time = time.time() - eval_start

        # Per-score-type metrics (also lite=False for VUS in final)
        disc_metrics = evaluator.evaluate_by_score_type('disc', lite=False)
        teacher_recon_metrics = evaluator.evaluate_by_score_type('teacher_recon', lite=False)
        student_recon_metrics = evaluator.evaluate_by_score_type('student_recon', lite=False)

        # === 2026-06-08 ROOT-CAUSE FIX: finalize metadata computed at the WRONG epoch ===
        # The evaluate() calls above re-forward best_checkpoint.pt, whose weights can be
        # MISALIGNED from the selected best epoch. The SAVED per-epoch npz@best_epoch is
        # the authoritative best-epoch snapshot — it drove the best-epoch SELECTION — so
        # the metadata metrics MUST be recomputed from it. No-op if the npz is missing.
        try:
            from mae_anomaly.evaluator import compute_full_metric_set as _cfms_best
            _best_npz_path = os.path.join(
                exp_dir, 'epoch_scores', f"epoch_{int(timing.get('best_epoch', 0)):03d}_scores.npz")
            if os.path.exists(_best_npz_path):
                _best_nd = np.load(_best_npz_path)
                _best_lbl = _best_nd['point_labels'].astype(np.int8)
                _best_ml = min(len(_best_lbl), len(test_point_labels))
                _primary_key = ('official_score' if getattr(config, 'official', False)
                                and 'official_score' in _best_nd.files else 'adaptive_score')
                for _best_skey, _best_mdict in ((_primary_key, metrics),
                                                ('discrepancy_error', disc_metrics),
                                                ('teacher_recon_error', teacher_recon_metrics)):
                    if _best_skey in _best_nd.files and _best_mdict is not None:
                        _best_rec = _cfms_best(
                            _best_nd[_best_skey][:_best_ml], _best_lbl[:_best_ml],
                            test_anomaly_regions, eval_mask=None,
                            n_thresholds=200, sliding_window=100, lite=False)
                        for _bk, _bv in _best_rec.items():
                            if not _bk.startswith('_') and isinstance(_bv, (int, float)):
                                _best_mdict[_bk] = float(_bv)
        except Exception as _best_e:
            print(f"  [{exp_name}] WARN npz@best metadata recompute skipped: "
                  f"{type(_best_e).__name__}: {_best_e}", flush=True)

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
                find_swat_region_22, compute_metrics_with_exclusion
            )
            region_22 = find_swat_region_22(test_anomaly_regions)
            if region_22 is not None:
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
                    # [post-warmup-forced] excl22 best epoch (overrides timing['best_epoch']
                    # below) may also only move to a POST-warmup epoch. Init already holds the
                    # full-SWaT post-warmup best (from select_best_epoch above), so no post-warmup
                    # excl22 eval ⇒ it stays post-warmup. Consistent with metric/checkpoint/viz.
                    _warm_excl = resolve_warmup_boundary(config)

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

                    # PERF FIX (2026-05-27): use lite=True for per-epoch scan.
                    # PERF FIX (2026-05-28): parallelize via ProcessPoolExecutor.
                    # Before parallelization: 200 calls (100 epochs × 2 score types)
                    # × ~6.5s each = ~22 min sequential. After: 200 / 8 workers ≈
                    # ~3 min wall. 10x speedup, no GPU usage, +~240MB RAM.
                    # Each task = one compute_metrics_with_exclusion call. Results
                    # collected in epoch order so excl22_epoch_metrics_list ordering
                    # is preserved.
                    import multiprocessing as _mp_lib_excl
                    from concurrent.futures import ProcessPoolExecutor as _PPE_excl
                    # 2026-05-29: default 8 → 2 to bound CPU usage when running
                    # 2 bg-workers (SWaT full + excl22) in parallel. With affinity
                    # capped to 4 cores and OMP=2 per process, max_workers=2
                    # uses 2 procs × 2 OMP = 4 threads — matches the affinity
                    # budget exactly. Override via env for fast hosts.
                    _excl22_workers = int(os.environ.get('TSMAE_EXCL22_WORKERS', 2))
                    _excl22_ctx = _mp_lib_excl.get_context('spawn')

                    # Pre-read npz files (small, fast) + collect score arrays
                    _score_data = []
                    for sf in score_files:
                        try:
                            ep_num = int(os.path.basename(sf).split('_')[1])
                            sd = np.load(sf)
                            a_scores = sd['official_score' if getattr(config, 'official', False) and 'official_score' in sd.files else 'adaptive_score']
                            t_scores = sd['teacher_recon_error']
                            ml = min(len(a_scores), len(test_point_labels))
                            _score_data.append((ep_num, a_scores[:ml], t_scores[:ml]))
                        except Exception:
                            continue

                    # Parallel dispatch to Pool
                    _excl22_pool = _PPE_excl(max_workers=_excl22_workers, mp_context=_excl22_ctx)
                    try:
                        _excl22_futures = {}
                        for ep_num, a_scores, t_scores in _score_data:
                            f_em = _excl22_pool.submit(
                                compute_metrics_with_exclusion,
                                a_scores, test_point_labels[:len(a_scores)],
                                test_anomaly_regions, region_22, True  # lite=True (positional)
                            )
                            f_teacher = _excl22_pool.submit(
                                compute_metrics_with_exclusion,
                                t_scores, test_point_labels[:len(t_scores)],
                                test_anomaly_regions, region_22, True
                            )
                            _excl22_futures[ep_num] = (f_em, f_teacher)

                        # Collect in epoch order
                        for ep_num in sorted(_excl22_futures.keys()):
                            f_em, f_teacher = _excl22_futures[ep_num]
                            try:
                                em = f_em.result()
                                teacher_em = f_teacher.result()
                            except Exception:
                                continue
                            pak_f1 = em.get('pak_auc_f1', 0)
                            if ep_num > _warm_excl and pak_f1 > best_excl22_pak_f1:
                                best_excl22_pak_f1 = pak_f1
                                best_excl22_epoch = ep_num
                                # Defer best-epoch full-metric recompute until after scan
                                best_excl22_em = em  # lite, will overwrite below
                                best_excl22_teacher_em = teacher_em  # lite, will overwrite

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
                                # 2026-06-01 FIX: recon_snr (teacher recon separation) and fm_loss
                                # (train feature-matching loss) are dataset-wide diagnostics —
                                # independent of which test region is masked, exactly like disc_snr —
                                # but were omitted from this copy list, leaving excl22 epoch_metrics
                                # recon_snr/fm_loss=None for ALL evals while disc_snr was copied from
                                # full-SWaT. Copy them too. (recon_snr added 2026-05-29, fm_loss
                                # earlier — both never propagated to excl22.)
                                for key in ['disc_snr', 'recon_snr', 'fm_loss', 'd_loss', 'd_real_acc', 'd_fake_acc',
                                            'adv_loss', 'adaptive_lambda',
                                            'grl_cls_loss', 'grl_balanced_acc', 'grl_anomaly_acc',
                                            'grl_normal_acc', 'grl_acc_gap', 'grl_lambda', 'grl_effective_weight']:
                                    if key in full_epoch_data[ep_num]:
                                        entry[key] = full_epoch_data[ep_num][key]
                            excl22_epoch_metrics_list.append(entry)
                    finally:
                        _excl22_pool.shutdown(wait=True)

                    excl22_epoch_metrics_list.sort(key=lambda x: x['epoch'])

                    # Recompute FULL metric set for the best epoch only (VUS+Aff+R-F1+AR)
                    if best_excl22_epoch > 0 and best_excl22_em is not None:
                        best_sf = os.path.join(
                            epoch_scores_dir, f'epoch_{best_excl22_epoch:03d}_scores.npz'
                        )
                        if os.path.exists(best_sf):
                            sd = np.load(best_sf)
                            a_scores = sd['official_score' if getattr(config, 'official', False) and 'official_score' in sd.files else 'adaptive_score']
                            t_scores = sd['teacher_recon_error']
                            ml = min(len(a_scores), len(test_point_labels))
                            best_excl22_em = compute_metrics_with_exclusion(
                                a_scores[:ml], test_point_labels[:ml],
                                test_anomaly_regions, region_22, lite=False
                            )
                            best_excl22_teacher_em = compute_metrics_with_exclusion(
                                t_scores[:ml], test_point_labels[:ml],
                                test_anomaly_regions, region_22, lite=False
                            )

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
                        excl_len = region_22.end - region_22.start
                        excl22_info = {
                            'region_start': region_22.start,
                            'region_end': region_22.end,
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
                        adaptive_scores = scores_data['official_score' if getattr(config, 'official', False) and 'official_score' in scores_data.files else 'adaptive_score']
                        teacher_scores = scores_data['teacher_recon_error']

                        min_len = min(len(adaptive_scores), len(test_point_labels))
                        excl22_metrics = compute_metrics_with_exclusion(
                            adaptive_scores[:min_len], test_point_labels[:min_len],
                            test_anomaly_regions, region_22
                        )
                        excl22_teacher_metrics = compute_metrics_with_exclusion(
                            teacher_scores[:min_len], test_point_labels[:min_len],
                            test_anomaly_regions, region_22
                        )
                        excl_len = region_22.end - region_22.start
                        excl22_info = {
                            'region_start': region_22.start,
                            'region_end': region_22.end,
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

    # SCAD-C repulsion diagnostics — additive, guarded no-op for non-Form-C runs.
    # Only fires when this experiment trained with scad_form == 'C' (e.g. exp321);
    # for every other run the train_scad_c_* series are constant 0.0 and nothing
    # is written, so the existing pipeline is byte-for-byte unchanged.
    if getattr(config, 'use_scad', False) and str(getattr(config, 'scad_form', '')) == 'C' and history:
        try:
            from mae_anomaly.visualization import ScadDiagnosticsVisualizer
            scad_diag_dir = os.path.join(exp_dir, 'visualization', 'scad_diagnostics')
            ScadDiagnosticsVisualizer(
                history=history, output_dir=scad_diag_dir, exp_dir=exp_dir, config=config,
            ).generate_all()
            plt.close('all')
        except Exception as _scad_e:
            print(f"  - [scad_diagnostics] skipped due to error: {_scad_e}")

    viz_time = time.time() - viz_start

    # Update metadata with viz_time and total
    total_time = timing.get('gpu_total', 0) + eval_time + viz_time
    metadata['timing']['cpu_viz_time'] = viz_time
    metadata['timing']['total_time'] = total_time
    with open(os.path.join(exp_dir, 'experiment_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=float)

    # ============================================================
    # Phase 5 (2026-05-29): Post-training VUS / Aff-F1 / R-F1 sweep
    # over all saved epoch_NNN_scores.npz files. Per-epoch training-time
    # evaluation runs lite=True (skip VUS) so this is where the bottom
    # row of the epoch_dashboard finally gets populated. The sweep runs
    # entirely on CPU inside the bg-worker process, so the main process
    # is free to start the next dataset's GPU training in parallel.
    # On failure (any exception) the dashboard simply stays with the
    # zero-filled VUS row written by the main process — no other
    # downstream artefact depends on these values, so we swallow the
    # error and continue rather than killing the bg-worker.
    # ============================================================
    # 2026-05-29: refactored to try-except-finally so the epoch_dashboard
    # is rendered EXACTLY ONCE regardless of VUS sweep outcome:
    #   - VUS succeeds → dashboard with full VUS row (the happy path)
    #   - VUS fails    → dashboard with zero-filled VUS row (graceful degradation)
    # Previously the dashboard was rendered twice (main process at L2702 with
    # zero VUS, then bg-worker after sweep with full VUS). If user looked at
    # the dashboard while bg-worker was still running, they saw the no-VUS
    # version. With L2702 removed, this finally block is the sole render site.
    vus_results = {}
    try:
        vus_t0 = time.time()
        # === excl22 VUS via array slicing (2026-05-29) ===
        # bg-B (excl22 worker) historically called the same VUS sweep as bg-A
        # against symlinked npz files, yielding byte-identical full VUS results
        # because compute_extra_metrics (evaluator.py L669) only consumes
        # `point_scores, point_labels, threshold, sliding_window` — no region
        # argument. So a true excl22 VUS just needs score/label arrays with the
        # region22 timesteps deleted before VUS computation. That's pure numpy
        # slicing inside _compute_vus_for_npz_file (no algorithm change, no new
        # dependency). On the full worker, excl_region stays None → original
        # behavior. On excl22 worker, region_22 is resolved from
        # test_anomaly_regions and passed through; the resulting vus_pr / vus_roc
        # / affiliation_f1 / r_based_f1 reflect the genuine region22-excluded
        # data, not full data.
        _excl_for_vus = None
        if swat_eval_mode == 'excl22' and test_anomaly_regions:
            try:
                from mae_anomaly.evaluator import find_swat_region_22 as _frv
                _excl_for_vus = _frv(test_anomaly_regions)
            except Exception:
                _excl_for_vus = None

        _vus_workers = int(os.environ.get('TSMAE_VUS_WORKERS', 2))
        if os.environ.get('MAE_SKIP_VUS') == '1':
            # (2026-06-23) VUS off (e.g. TEP) → this post-training VUS sweep over ALL
            # saved epoch_NNN_scores.npz has NO useful output (its only product is the
            # epoch_dashboard's VUS row, which is empty when VUS is off). It is NOT gated
            # by MAE_SKIP_VUS internally, so it was running VUS × 30 npz and hanging the
            # bg-worker for ~40 min on TEP-scale tests (and the dashboard render below was
            # blocked behind it). Skip the sweep entirely; the finally block still renders
            # the epoch dashboard directly from epoch_metrics.json (VUS row left empty).
            print(f"  [{exp_name}] VUS sweep SKIPPED (MAE_SKIP_VUS=1) — "
                  f"dashboard renders from epoch_metrics (empty VUS row)", flush=True)
            vus_results = {}
        else:
            vus_results = _run_vus_sweep_on_saved_npz(
                exp_dir,
                test_anomaly_regions,
                max_workers=_vus_workers,
                log_prefix=f"[{exp_name}] ",
                excl_region=_excl_for_vus,
            )
        if vus_results:
            # Load epoch_metrics.json (written by main process pre-spawn) and
            # merge the sweep results in.
            epm_path = os.path.join(exp_dir, 'epoch_metrics.json')
            if os.path.exists(epm_path):
                with open(epm_path) as f:
                    epm_data = json.load(f)
                _apply_vus_sweep_results(epm_data.get('epochs', []), vus_results)
                with open(epm_path, 'w') as f:
                    json.dump(epm_data, f, indent=2)
    except Exception as _vus_e:  # noqa: BLE001
        print(f"  [{exp_name}] WARN VUS sweep failed: "
              f"{type(_vus_e).__name__}: {_vus_e} — rendering dashboard with zero-filled VUS row", flush=True)
    finally:
        # Always render the dashboard at the end — with VUS if sweep
        # succeeded, with zero-filled VUS row if sweep failed.
        try:
            epm_path = os.path.join(exp_dir, 'epoch_metrics.json')
            if os.path.exists(epm_path):
                with open(epm_path) as f:
                    epm_data = json.load(f)
                epoch_viz_dir = os.path.join(exp_dir, 'visualization', 'epoch_metrics')
                plot_epoch_metrics(epm_data.get('epochs', []), epoch_viz_dir)
                _vus_state = "with VUS" if vus_results else "VUS missing"
                print(f"  [{exp_name}] Epoch dashboard rendered ({_vus_state}) "
                      f"({time.time() - vus_t0:.0f}s)", flush=True)
        except Exception as _render_e:  # noqa: BLE001
            print(f"  [{exp_name}] WARN dashboard render failed: "
                  f"{type(_render_e).__name__}: {_render_e}", flush=True)

    # GRL adversarial-game diagnostics — additive, guarded no-op for non-GRL runs.
    # Placed AFTER the epoch dashboard (epoch_grl.png, rendered in the finally above)
    # AND the best_model viz (GRL_contribution_trend.png) so _relocate_existing() can
    # consolidate BOTH legacy GRL pngs into grl_diagnostics/. has_grl() is False for
    # SCAD/plain/WDGRL → nothing written (existing pipeline byte-for-byte unchanged).
    if (getattr(config, 'use_grl', False)
            and str(getattr(config, 'grl_mode', 'classifier')) == 'classifier' and history):
        try:
            from mae_anomaly.visualization import GrlDiagnosticsVisualizer
            grl_diag_dir = os.path.join(exp_dir, 'visualization', 'grl_diagnostics')
            GrlDiagnosticsVisualizer(
                history=history, output_dir=grl_diag_dir, exp_dir=exp_dir, config=config,
            ).generate_all()
            plt.close('all')
        except Exception as _grl_e:
            print(f"  - [grl_diagnostics] skipped due to error: {_grl_e}")

    pt = timing.get('pure_train_time', timing.get('train_time', 0))
    tpe = timing.get('train_per_epoch', 0)
    be = timing.get('best_epoch', '?')
    # Training losses from history (last epoch).
    # Same naming convention as the per-epoch eval log line (see L2275-2288):
    #   t_loss / s_loss  — legacy aliases (joint recon / discrepancy)
    #   recon_t / recon_s / dis — 2026-05-29 explicit fields
    t_loss = history['train_rec_loss'][-1] if history and history.get('train_rec_loss') else 0
    s_loss = history['train_disc_loss'][-1] if history and history.get('train_disc_loss') else 0
    recon_t = (history['train_teacher_recon_normal'][-1]
               if history and history.get('train_teacher_recon_normal') else 0)
    recon_s = (history['train_student_recon_normal'][-1]
               if history and history.get('train_student_recon_normal') else 0)
    dis_loss = s_loss  # alias
    d_loss_str = ""
    if history and history.get('train_d_loss'):
        d_loss_str = f" d_loss={history['train_d_loss'][-1]:.4f}"
    print(f"  [{exp_name}] {progress_info} COMPLETE ({total_time:.0f}s): "
          f"PRC={primary_metrics.get('prc_auc',0):.4f} "
          f"PAK_AUC_F1={primary_metrics.get('pak_auc_f1',0):.4f} PAK_AUC_PRC={primary_metrics.get('pak_auc_prc_auc',0):.4f} "
          f"F1_T={primary_metrics.get('f1_t',0):.4f} | "
          f"d_SNR={loss_stats.get('disc_SNR',0):.4f} recon_SNR={loss_stats.get('recon_SNR',0):.4f} "
          f"t_loss={t_loss:.4f} s_loss={s_loss:.4f} "
          f"recon_t={recon_t:.4f} recon_s={recon_s:.4f} dis={dis_loss:.4f}"
          f"{d_loss_str} | "
          f"best_ep={be} | "
          f"Time: train={pt:.0f}s({tpe:.1f}s/ep) "
          f"infer={timing.get('inference_time',0):.0f}s eval={eval_time:.0f}s viz={viz_time:.0f}s", flush=True)


# =============================================================================
# Main Experiment Runner
# =============================================================================

def _write_tep_experiment_info(exp_dir, key, config, data_info):
    """[TEP] Write a human-readable EXPERIMENT_INFO.md into the run dir documenting the
    condition, dataset composition, and (carefully) the LABELING, so each TEP run is
    self-documenting. Best-effort — never raises into the run."""
    try:
        di = data_info or {}
        gv = lambda k, d='?': getattr(config, k, d)
        blind = bool(getattr(config, 'blind_train_labels', False))
        uf = float(di.get('unlabeled_frac', 0.0) or 0.0)
        proto = di.get('protocol', '')
        fold = di.get('fold', '')
        is_ffonly = 'ffonly' in str(key)
        tl = int(di.get('train_len', 0)); te = int(di.get('test_len', 0))
        tar = float(di.get('train_attack_ratio', 0)); tear = float(di.get('test_attack_ratio', 0))
        tr = float(di.get('train_ratio', 0)); n_runs = tl // 960 if tl else 0
        seen = di.get('seen_fault_set', []); ho_set = di.get('heldout_fault_set')
        nfaulty = di.get('n_faulty_runs'); nunlab = di.get('n_unlabeled_runs')
        if proto == 'lofo':
            ho = di.get('held_out_family'); cont = di.get('contaminate_heldout', False)
            cond = (f"LOFO (3 family seen / held-out={ho}, "
                    f"{'held-out 무라벨 오염' if cont else 'held-out 제외'})"
                    + (" + label-blind(B)" if blind else " — seen 3 family 라벨(A류)"))
        elif is_ffonly:
            cond = "B0 — clean fault-free reference (정상 FF만 학습)"
        elif uf > 0:
            cond = f"noisy-label (부분 라벨; seen 오염 중 {nunlab}/{nfaulty} run 무라벨, unlabeled_frac={uf})"
        elif blind:
            cond = "B — label-blind (오염 train, 라벨 전부 차단)"
        else:
            cond = "A — LASAD/ours (오염 train, seen-family 라벨 사용)"
        if is_ffonly:
            lab = "train에 faulty run 없음 → 전부 **정상(0)**."
        elif blind:
            lab = ("**blind_train_labels=True** → train point label 전부 **0** (faulty도 무라벨 오염으로만 존재). "
                   "라벨 0이라 GRL/force_mask/anomaly_loss 자동 inert.")
        elif uf > 0:
            lab = (f"seen-family faulty run 중 per-fault 뒤쪽 **{nunlab}/{nfaulty} run을 무라벨(0)**, "
                   "나머지는 onset(sample 161)+ 구간을 **anomaly(1)**로 라벨.")
        else:
            lab = ("seen-family faulty run의 **onset(sample 161)+ 구간 = anomaly(1)**, "
                   "앞 160 + FF run = 정상(0).")
        seen_s = (', '.join(map(str, seen)) if seen else '없음(clean)')
        _ei_raw = int(getattr(config, 'eval_interval', -1) or -1); ei = _ei_raw if _ei_raw > 0 else 1
        md = (
            f"# TEP 실험 정보 — {key}\n\n"
            f"> 자동 생성(`run_base_experiments.py`). 데이터셋 구성·**라벨링**·설정 기록.\n\n"
            f"## 조건\n**{cond}**\n\n"
            f"## 데이터셋 구성\n"
            f"- protocol: {proto or ('ffonly(B0)' if is_ffonly else f'1-seen fold ({fold})')}\n"
            f"- **train**: {tl:,} samples ({n_runs} runs × 960)"
            f"{'' if is_ffonly else f' = FF(정상) + faulty {nfaulty} runs'}\n"
            f"  - seen faults: {seen_s}" + (f" / held-out: {ho_set}" if ho_set else "") + "\n"
            f"  - train **labeled-anomaly ratio**: {tar:.4f}\n"
            f"- **test** (공유 frozen test_stream): {te:,} samples = 440 runs (faulty 400 + FF 40), "
            f"anomaly {tear:.4f}, regions {di.get('n_anomaly_regions_total','?')}\n"
            f"- train_ratio: {tr:.4f} · fault onset: sample {di.get('fault_onset_sample',161)} "
            f"(faulty run = 앞 160 정상 / 뒤 800 이상)\n"
            f"- **IDV 3·9·15 = excluded-hard** (headline 집계 제외)\n\n"
            f"## 라벨링 (주의)\n{lab}\n- test는 조건 무관 **항상 실제 라벨**로 평가.\n\n"
            f"## 설정\n"
            f"- official=True (CANON_271), normalize={gv('normalize_mode')}/{gv('minmax_range')}\n"
            f"- num_epochs={gv('num_epochs')}, teacher_only_warmup_epochs={gv('teacher_only_warmup_epochs')}, "
            f"batch_size={gv('batch_size')}, seed={gv('random_seed')}\n"
            f"- use_grl={gv('use_grl')} / force_mask_anomaly={gv('force_mask_anomaly')} "
            f"(라벨 0이면 자동 inert) / use_output_discrepancy={gv('use_output_discrepancy')}\n"
            f"- weight 미저장(official_keep_checkpoints={gv('official_keep_checkpoints')}), eval interval={ei} (epoch)\n"
            f"- **headline metric = pak_auc_f1** (PA%K F1-AUC). D(recon-only) = teacher_pak_auc_f1.\n\n"
            f"## 데이터 출처\n`scripts/TEP/data/*.npz` (frozen; `scripts/TEP/build_tep_data.py`, Rieth 2017 TEP). "
            f"`manifest.json` 참조.\n"
        )
        with open(os.path.join(exp_dir, 'EXPERIMENT_INFO.md'), 'w') as f:
            f.write(md)
        print("  [TEP] EXPERIMENT_INFO.md written")
    except Exception as e:
        print(f"  [TEP] EXPERIMENT_INFO.md skipped: {e}")


def run_base_experiment(dataset_def, config_preset, results_base, progress_info="",
                        save_weights=False):
    """Run a single base experiment with epoch-wise monitoring."""
    key = dataset_def['key']
    train_stride = dataset_def['train_stride']
    is_normal50 = dataset_def['normal50']
    results_dir = os.path.join(results_base, dataset_def['results_subdir'])

    # 2026-05-28: dataset-level skip for resume. If experiment_metadata.json exists
    # at the target dataset dir (or _full for SWaT dual-eval), this dataset is
    # already completed — skip. Manual override: delete experiment_metadata.json
    # to force re-run. Allows queue/manual resume from arbitrary dataset position
    # within an exp_dir (e.g., resume exp285 from WaDi/A1 after deleting WaDi/).
    _is_swat_skip = 'SWaT' in key and 'swap' not in key
    _check_path = results_dir + '_full' if _is_swat_skip else results_dir
    _skip_marker = os.path.join(_check_path, 'experiment_metadata.json')
    if os.path.exists(_skip_marker):
        print(f"\n{'='*80}")
        print(f"⏭️  SKIP {key}: already completed ({_skip_marker} exists)")
        print(f"  To force re-run, delete: {_check_path}")
        print(f"{'='*80}")
        return

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
    # Per-entity normalization segments for concat multi-entity datasets
    # (SMAP/MSL/SMD/Exathlon). When present, SlidingWindowDataset fits a scaler
    # per entity on its own train portion (leakage-free) instead of one
    # whole-array fit that mixes entities of differing scale (2026-06-02 fix).
    entity_segments = data_info.get('entity_norm_segments') if data_info else None

    print(f"  Signals: {signals.shape}")
    print(f"  Labels: normal={np.sum(point_labels==0):,}, anomaly={np.sum(point_labels==1):,}")
    print(f"  Train ratio: {train_ratio:.4f}")
    if run_boundaries:
        print(f"  Run boundaries: {len(run_boundaries)} (windows will not cross block boundaries)")
    if entity_segments:
        print(f"  Per-entity normalization: {len(entity_segments)} entities "
              f"(each fit on its own train portion; NO whole-array fit)")

    # Apply normal50 noise if needed
    noisy_labels = None
    if is_normal50:
        noisy_labels = apply_normal50_noise(point_labels, anomaly_regions, train_ratio)
        noisy_train_end = int(len(point_labels) * train_ratio)
        print(f"  Normal50 noise applied: train anomaly ratio {noisy_labels[:noisy_train_end].mean():.2%}")

    # Create config from preset + dataset-specific overrides
    if config_preset['overrides'].get('official'):
        # [official] 271-base layering: CANON_271 is the default for every config
        # the user did NOT explicitly pass; the explicit --config-override keys win
        # over it; make_config's apply_official_overrides then FORCES the official
        # bundle on top. (Set C preset geometry is intentionally bypassed — 271
        # supplies its own.) official=False ⇒ this branch is skipped entirely.
        _user_keys = config_preset.get('_user_override_keys', [])
        overrides = dict(CANON_271)
        # Official defaults that are USER-OVERRIDABLE: num_epochs=30 and
        # teacher_only_warmup_epochs=num_epochs//2. Set BEFORE the user merge so an
        # explicit num_epochs / teacher_only_warmup_epochs in config_override wins.
        overrides['num_epochs'] = 30
        for _k in _user_keys:
            if _k in config_preset['overrides']:
                overrides[_k] = config_preset['overrides'][_k]
        if 'teacher_only_warmup_epochs' not in _user_keys:
            overrides['teacher_only_warmup_epochs'] = int(overrides['num_epochs']) // 2
        overrides['official'] = True
    else:
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

    # [official] Per-experiment LOCAL eval interval (1 = every epoch). The module
    # global EVAL_INTERVAL stays 5 and is NEVER mutated, so a non-official
    # experiment later in the same queue process is unaffected.
    # config.eval_interval override (>0) wins → eval only every N epochs (+ final epoch),
    # eval-bound 완화용. -1(default) → auto: 1 official / EVAL_INTERVAL else.
    _ei_ovr = int(getattr(config, 'eval_interval', -1) or -1)
    if _ei_ovr > 0:
        eval_interval = _ei_ovr          # explicit user override always wins
    elif str(key).startswith('TEP_typegen'):
        eval_interval = 3                # (2026-06-23) TEP_typegen 경로 전용 default = 3
    else:
        eval_interval = 1 if getattr(config, 'official', False) else EVAL_INTERVAL  # 비-TEP 원래대로 (official=1)

    # [official] Force the LOCAL train stride to 1 (no offset). The TRAIN datasets
    # below (2537/2552) read this local `train_stride`, NOT config.sliding_window_stride,
    # so the config-field override alone would NOT take effect — this is the actual
    # train-stride gate. test_stride (resolve_test_stride) and the train-inference
    # loader stride are untouched (req 1 is train-only).
    if getattr(config, 'official', False):
        train_stride = 1

    # [official] Whether to keep this dataset's checkpoints (per-dataset override
    # with global fallback). False ⇒ skip official_epochs/ writes + delete the
    # best/last checkpoints at the end (eval + viz still run). Only used when official.
    _official_keep_ckpt = _official_keep_ckpt_for(config, key)

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
    if str(key).startswith('TEP_typegen'):
        _write_tep_experiment_info(exp_dir, key, config, data_info)
        # [TEP] VUS off — final/bg eval skips VUS (~40s/call, eval-tail 병목; 분석에 미사용).
        # env는 spawn된 bg eval+viz / pool worker에 상속됨. per-epoch eval은 이미 lite로 skip.
        os.environ['MAE_SKIP_VUS'] = '1'

    # Create datasets
    print("Creating datasets...")
    # [official] Thorough single-knob seeding (adds PYTHONHASHSEED + a seeded
    # generator; same cudnn flags as set_seed → no speed loss). Everything still
    # derives from config.random_seed, so changing that one value changes all RNG.
    if getattr(config, 'official', False):
        set_seed_official(config.random_seed)
    else:
        set_seed(config.random_seed)

    from mae_anomaly.utils.experiment import resolve_test_stride
    test_stride = resolve_test_stride(config)
    test_dataset = SlidingWindowDataset(
        signals=signals, point_labels=point_labels, anomaly_regions=anomaly_regions,
        window_size=config.seq_length, stride=test_stride, mask_last_n=config.patch_size,
        split='test', train_ratio=train_ratio, seed=config.random_seed,
        run_boundaries=run_boundaries,
        normalize_mode=config.normalize_mode,
        minmax_range=getattr(config, 'minmax_range', '0_1'),
        minmax_clamp_min=getattr(config, 'minmax_clamp_min', None),
        minmax_clamp_max=getattr(config, 'minmax_clamp_max', None),
        entity_segments=entity_segments,
    )
    # === Per-dataset DataLoader num_workers (2026-05-29) ===
    # WaDi/A1 and WaDi/A2 have 123 features → batch tensor is ~252 MB at
    # batch_size=1024, bf16. With num_workers>0, PyTorch CUDA caching allocator
    # holds ~5 GB extra residency (CUDA stream concurrency + pin-thread + IPC
    # transitions), pushing total to 24 GB / 24.6 GB capacity → allocator
    # thrashing → batches degrade from 0.5 s/batch to 14 s/batch after ~30
    # batches. Confirmed empirically with prefetch_factor=1, pin_memory=False,
    # and PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6
    # (expandable_segments:True crashed with INTERNAL ASSERT FAILED at
    # CUDACachingAllocator.cpp:417 — driver incompatibility).
    # SWaT_A1A2 (8 features) and PSM (25 features) have ample memory
    # headroom → workers fine (measured 1.8× speedup on SWaT).
    # 2026-05-29 confirmed: nw=2 on WaDi/A1 hits GPU OOM thrashing even from a
    # genuinely fresh GPU state. PyTorch num_workers>0 forces ~5 GB extra CUDA
    # residency on top of WaDi/A1's 18.6 GB working set → exceeds 24.5 GB →
    # allocator thrashes. WaDi → nw=0.
    # 2026-05-29 22:00: 271_lr SWaT crashed at ep 7 batch 31 with IndexError
    # (idx 34236 OOB size 34236) — root cause: epoch_offset=True changes
    # len(window_start_indices) ±1 per epoch but persistent DataLoader workers
    # hold stale dataset state. Fixed in dataset_sliding.py:_extract_windows
    # via length stabilization (lock to first-call N, truncate/pad if drift).
    # → nw=2 on small-feature datasets now safe; WaDi stays nw=0 for GPU mem.
    # 2026-05-29 23:45: even SWaT (45 feat) at POST-warmup + dual-eval hit GPU
    # mem 98.2% (24111/24564 MiB) with nw=2 → allocator thrashing (24-26 s/ep,
    # 6.6 s/it spikes), same signature as WaDi. Post-warmup adds student
    # decoder + GRL + disc forward on top of dual-eval residency. Per user
    # decision, force nw=0 for ALL datasets — the nw=2 speedup is not worth the
    # OOM-thrashing risk at post-warmup. Branch kept for easy re-enable.
    _force_nw0_all = True  # 2026-05-29: global nw=0 (post-warmup OOM thrashing)
    _is_high_feature = _force_nw0_all or (key in ('WaDi_A1', 'WaDi_A2'))
    _dataloader_nw = 0 if _is_high_feature else 2
    _dataloader_persistent = (_dataloader_nw > 0)
    _dataloader_prefetch = 1 if _dataloader_nw > 0 else None
    print(f"  DataLoader: num_workers={_dataloader_nw} ({'WaDi → 0 due to GPU mem' if _is_high_feature else 'default 2 for GPU sat'})")
    # test_loader: same workers config as train_loader.
    # Used at every per-epoch eval (every 5 epochs) → worker reuse via
    # persistent_workers cuts eval cycle by ~5-10s. shuffle=False keeps
    # deterministic eval ordering.
    _test_loader_kwargs = dict(num_workers=_dataloader_nw, pin_memory=False)
    if _dataloader_nw > 0:
        _test_loader_kwargs['persistent_workers'] = _dataloader_persistent
        _test_loader_kwargs['prefetch_factor'] = _dataloader_prefetch
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        **_test_loader_kwargs,
    )
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
            entity_segments=entity_segments,
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
            entity_segments=entity_segments,
            blind_train_labels=getattr(config, 'blind_train_labels', False),
            train_label_mask_frac=getattr(config, 'train_label_mask_frac', 0.0),
            train_label_mask_random=getattr(config, 'train_label_mask_random', False),
            train_label_mask_group_size=getattr(config, 'train_label_mask_group_size', 100),
            train_label_mask_exclude=getattr(config, 'train_label_mask_exclude', False),
            train_exclude_anomaly_segments=getattr(config, 'train_exclude_anomaly_segments', False),
        )
    # drop_last=True (when dataset is large enough): keeps batch shape constant across steps,
    # which stabilizes cuDNN.benchmark heuristic search and removes a small last-batch overhead.
    # Guard: if the dataset is smaller than batch_size, drop_last would yield an empty loader.
    _drop_last = len(train_dataset) >= config.batch_size
    # Resume-safe DataLoader (2026-05-28): explicit Generator + RandomSampler.
    # Sample order per epoch is determined ONLY by g.manual_seed(seed+epoch),
    # called by pre_epoch_hook in trainer.train(). This makes resume reproducible
    # without saving sampler state in checkpoint. Differs from prior shuffle=True
    # which used torch's global RNG (whose history depends on every torch.rand
    # consumer everywhere — fragile for resume).
    from torch.utils.data import RandomSampler as _RandomSampler
    _train_generator = torch.Generator()  # CPU generator (DataLoader's default context)
    _train_generator.manual_seed(config.random_seed)
    _train_sampler = _RandomSampler(train_dataset, generator=_train_generator)
    # === DataLoader workers (2026-05-29) ===
    # num_workers=0 caused dataloader-bound GPU idle (~27% gap, util plateau at 73%
    # peak 96% under bf16 small-model regime). Adding workers fetches/collates
    # next batch while GPU is forwarding current → reduces idle gap, raises
    # sustained util. pin_memory uses page-locked CPU staging buffer for faster
    # async H2D copy (bf16 saturates PCIe quickly). persistent_workers keeps
    # workers alive across epochs → ~0 spawn overhead after first epoch.
    # Reproducibility preserved: RandomSampler runs in main thread with
    # _train_generator (deterministic per-epoch via pre_epoch_hook reseed at
    # L2754), workers only fetch dataset[idx] deterministically. Workers
    # inherit main affinity [0-11] (fork default) and OMP=2 env (launcher
    # export) — no contention with bg-worker [12-15].
    _train_loader_kwargs = dict(num_workers=_dataloader_nw, pin_memory=False)
    if _dataloader_nw > 0:
        _train_loader_kwargs['persistent_workers'] = _dataloader_persistent
        _train_loader_kwargs['prefetch_factor'] = _dataloader_prefetch
    if getattr(config, 'official', False):
        # Defensive per-worker seeding (moot at num_workers=0; reproducible if >0).
        _train_loader_kwargs['worker_init_fn'] = official_worker_init_fn
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        sampler=_train_sampler, drop_last=_drop_last,
        **_train_loader_kwargs,
    )
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
    _async_eval_thread = [None]        # [thread] — for join before checkpoint ops
    _pending_eval = [None]             # [eval_data] staged by eval callback (eval
                                       # epochs only), consumed by post_epoch_save_callback
    _official_train_infer = [None, None]  # [official] [train_infer_loader, dataset] built once, reused per eval epoch
    _best_epoch_metric_key = config.best_epoch_metric  # e.g. 'pak_auc_f1'
    _best_ckpt_score = [0.0]  # best score seen so far (mutable for nonlocal)
    # [post-warmup-forced] best_checkpoint + reported best_epoch are chosen ONLY from
    # post-warmup epochs (epoch > _warm_boundary). UNCONDITIONAL (not gated on official):
    # during warmup the student/discrepancy is untrained so a pre-warmup best is
    # misleading. Single source of truth = select_best_epoch / resolve_warmup_boundary.
    _warm_boundary = resolve_warmup_boundary(config)

    # ---- Option A+C (2026-05-28): ProcessPoolExecutor for per-K eval work ----
    # Per-eval CPU cost was ~33s because per-K loops (21 K × 2 funcs +
    # ts_precision_and_recall × 101 K × 2 modes × 4 calls per eval) ran in a
    # single threading.Thread, sharing GIL with the GPU training loop. By
    # delegating per-K work to a Pool of separate processes (spawn, GIL-free),
    # the bg-thread becomes thin (dispatch + await) and the heavy work runs
    # on idle CPU cores. Expected: 33s → ~7s eval (~80% reduction).
    # The pool is created once per experiment (not per eval) — spawn overhead
    # amortized across all ~100 evals.
    import multiprocessing as _mp_lib
    from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
    _EVAL_POOL_WORKERS = int(os.environ.get('TSMAE_EVAL_POOL_WORKERS', 8))
    _eval_executor = _ProcessPoolExecutor(
        max_workers=_EVAL_POOL_WORKERS,
        mp_context=_mp_lib.get_context('spawn'),  # spawn: CUDA-safe, GIL-free
    )
    # Cleanup hook: shutdown pool on process exit (safety net for crashes).
    import atexit as _atexit
    _atexit.register(lambda: _eval_executor.shutdown(wait=False, cancel_futures=True) if _eval_executor is not None else None)

    def _process_eval_result(cb_metrics):
        """Record a completed eval result: append to epoch_metrics_list, update the
        best score, print the metrics line, and RETURN is_best.

        The latest→best checkpoint COPY is intentionally NOT done here — the caller
        (_run_bg_all) performs it AFTER writing latest_checkpoint.pt for this epoch,
        so best_checkpoint.pt captures exactly this epoch's weights + eval list.
        """
        epoch_metrics_list.append(cb_metrics)
        prc = cb_metrics.get('prc_auc', 0)
        f1t = cb_metrics.get('f1_t', 0)
        d_snr = cb_metrics.get('disc_snr', 0)
        r_snr = cb_metrics.get('recon_snr', 0)  # teacher-only recon separation (2026-05-29)
        pak_f1 = cb_metrics.get('pak_auc_f1', 0)
        pak_prc = cb_metrics.get('pak_auc_prc_auc', 0)
        i_t = cb_metrics.get('_inference_time', 0)
        e_t = cb_metrics.get('_eval_time', 0)
        ep = cb_metrics.get('epoch', 0)
        # Update best checkpoint if best_epoch_metric improved
        # [post-warmup-forced] only post-warmup epochs (ep > _warm_boundary) may become
        # best_checkpoint, so the saved best_model never lands on a pre-warmup epoch.
        # Greedy running-max over post-warmup == full-list max ⇒ stays consistent with the
        # final best_epoch selection (select_best_epoch) below.
        epoch_score = cb_metrics.get(_best_epoch_metric_key, 0)
        is_best = (ep > _warm_boundary) and (epoch_score > _best_ckpt_score[0])
        best_marker = ""
        if is_best:
            _best_ckpt_score[0] = epoch_score
            best_marker = " ★"
            # latest→best copy deferred to _run_bg_all (after it saves THIS epoch's
            # latest_checkpoint.pt) — see this function's docstring.

        # Training losses from history
        h = _trainer_ref[0].history if _trainer_ref[0] is not None else {}
        idx = ep - 1  # epoch 1-indexed, history 0-indexed
        # Legacy fields (kept for parser backward compat):
        #   t_loss = train_rec_loss   (joint reconstruction loss — misleading legacy name)
        #   s_loss = train_disc_loss  (output discrepancy — misleading legacy name)
        # Explicit fields (added 2026-05-29 for monitoring clarity):
        #   recon_t = train_teacher_recon_normal  (teacher-only recon, normal samples)
        #   recon_s = train_student_recon_normal  (student-only recon, normal samples;
        #             non-zero but stagnant pre-warmup — student forward runs every step
        #             but loss.backward() doesn't update student params; see loss.py:179)
        #   dis     = train_disc_loss             (alias for s_loss with semantically correct name;
        #             0 pre-warmup since discrepancy block is gated by `not teacher_only`)
        t_loss = h['train_rec_loss'][idx] if idx < len(h.get('train_rec_loss', [])) else 0
        s_loss = h['train_disc_loss'][idx] if idx < len(h.get('train_disc_loss', [])) else 0
        recon_t = (h['train_teacher_recon_normal'][idx]
                   if idx < len(h.get('train_teacher_recon_normal', [])) else 0)
        recon_s = (h['train_student_recon_normal'][idx]
                   if idx < len(h.get('train_student_recon_normal', [])) else 0)
        dis_loss = s_loss  # alias — same value, clearer name
        d_loss_str = ""
        if idx < len(h.get('train_d_loss', [])):
            d_loss_str = f" d_loss={h['train_d_loss'][idx]:.4f}"
        # Family ordering: point-strict | PA%K | Event/Range | VUS | diagnostic
        # f1 = sklearn point-level F1 at optimal threshold (NO point-adjustment).
        # cb_metrics['f1_score'] is computed in evaluator via sklearn.metrics.f1_score
        # at the F1-optimal threshold over the test set. Do NOT use pa_0_f1 here:
        # pa_0_f1 is the K=0 PA-adjusted F1 (lenient PA, traditional PA literature),
        # which inflates recall to 1.0 and misrepresents point-level performance.
        f1_pt  = cb_metrics.get('f1_score', 0)
        vus_pr = cb_metrics.get('vus_pr', 0)
        vus_roc = cb_metrics.get('vus_roc', 0)
        aff_f1 = cb_metrics.get('affiliation_f1', 0)
        rf1 = cb_metrics.get('r_based_f1', 0)
        # SWaT excl22 dual eval (2026-05-27): if present in cb_metrics, append second line.
        # Only emitted when dataset is SWaT* and evaluator.evaluate(also_excl22=True) ran.
        has_excl22 = 'excl22_pak_auc_f1' in cb_metrics
        excl22_str = ""
        if has_excl22:
            e_prc = cb_metrics.get('excl22_prc_auc', 0)
            e_f1 = cb_metrics.get('excl22_f1_score', 0)
            e_f1t = cb_metrics.get('excl22_f1_t', 0)
            e_pak_f1 = cb_metrics.get('excl22_pak_auc_f1', 0)
            e_pak_prc = cb_metrics.get('excl22_pak_auc_prc_auc', 0)
            e_aff = cb_metrics.get('excl22_affiliation_f1', 0)
            e_rf1 = cb_metrics.get('excl22_r_based_f1', 0)
            excl22_str = (
                f"\n              [excl22] "
                f"PRC={e_prc:.4f} F1={e_f1:.4f} F1_T={e_f1t:.4f} "
                f"PAK_F1={e_pak_f1:.4f} PAK_PRC={e_pak_prc:.4f} "
                f"AFF_F1={e_aff:.4f} RF1={e_rf1:.4f}"
            )
        tqdm.write(
            f"  [Epoch {ep:>2}] "
            # point-strict
            f"PRC={prc:.4f} F1={f1_pt:.4f} F1_T={f1t:.4f} "
            # PA%K
            f"PAK_F1={pak_f1:.4f} PAK_PRC={pak_prc:.4f} "
            # event/range-aware
            f"AFF_F1={aff_f1:.4f} RF1={rf1:.4f} "
            # VUS
            f"VUS_PR={vus_pr:.4f} VUS_ROC={vus_roc:.4f} "
            # diagnostic
            f"d_SNR={d_snr:.4f} recon_SNR={r_snr:.4f} | "
            f"t_loss={t_loss:.4f} s_loss={s_loss:.4f}"
            f" recon_t={recon_t:.4f} recon_s={recon_s:.4f} dis={dis_loss:.4f}"
            f"{d_loss_str} "
            f"(infer={i_t:.0f}s eval={e_t:.0f}s) [async]{best_marker}"
            + excl22_str
        )
        return is_best

    def _collect_async_eval(blocking=False):
        """Wait for the in-flight eval-epoch thread to finish. Each thread records its
        own eval AND saves its checkpoint inside _run_bg_all, so once the final thread
        is joined epoch_metrics_list is already complete — there is nothing to drain."""
        if blocking and _async_eval_thread[0] is not None:
            _async_eval_thread[0].join()
            _async_eval_thread[0] = None

    def _run_bg_all(eval_data, ep, ckpt_data, prev_thread):
        """Background thread for ONE eval epoch: join prev → run eval → record →
        save checkpoint → promote best. GPU is free to train while this runs on CPU.

        Ordering rationale (2026-05-30 — eval-before-checkpoint invariant):
          The checkpoint for epoch N is written ONLY AFTER epoch N's eval has been
          recorded into epoch_metrics_list. Therefore "latest_checkpoint.pt holds
          epoch N" implies "every eval for epochs ≤ N is in its epoch_metrics_list".
          A resume can never drop an eval record: if N's eval had not finished, the
          checkpoint would still read epoch N-EVAL_INTERVAL (a few epochs back, which
          the user explicitly accepts), whose own eval list is itself complete.

        Thread chaining (join prev) keeps checkpoint-file order, epoch_metrics_list
        order, and the latest→best copy strictly serial. `eval_data` is always
        non-None here (post_epoch_save_callback returns early on non-eval epochs).
        """
        try:
            # A. Serialize after the previous eval-epoch thread (its checkpoint write
            #    + best-copy are fully done before we touch latest_checkpoint.pt).
            if prev_thread is not None:
                prev_thread.join()

            # B. Run THIS epoch's CPU eval (the heavy part; uses the process pool).
            cb_metrics = _compute_cpu_eval(eval_data, ep)

            # C. Record it (append to epoch_metrics_list + update best score + print)
            #    BEFORE snapshotting the list into the checkpoint. NO copy yet — the
            #    latest checkpoint for this epoch is written in step E.
            is_best = False
            if cb_metrics is not None:
                is_best = _process_eval_result(cb_metrics)

            # D. Fold the now-current eval list + best score into the checkpoint, so
            #    ckpt['epoch'] == N  ⟺  evals through N are persisted with it.
            ckpt_data['epoch_metrics_list'] = list(epoch_metrics_list)
            ckpt_data['best_ckpt_score'] = _best_ckpt_score[0]

            # E. Save this epoch's checkpoint (CPU-cloned, epoch-consistent state).
            latest_path = os.path.join(checkpoints_dir, 'latest_checkpoint.pt')
            torch.save(ckpt_data, latest_path)

            # F. Promote to best AFTER latest == this epoch's checkpoint, so
            #    best_checkpoint.pt captures exactly this epoch's weights + eval list.
            if is_best:
                shutil.copy2(latest_path, os.path.join(checkpoints_dir, 'best_checkpoint.pt'))

        except Exception as e:
            tqdm.write(f"  [Epoch {ep:>2}] BG ALL ERROR: {e}")

    def _compute_cpu_eval(eval_data, ep):
        """Compute ONE epoch's CPU eval (heavy per-K metric work via the process pool)
        and save the per-epoch point-score npz. RETURNS the metrics dict (or None on
        failure) for the caller to record. Does NOT touch epoch_metrics_list or any
        checkpoint file — that ordering is owned entirely by _run_bg_all."""
        try:
            # Pass dataset_key so SWaT enables excl22 dual eval automatically.
            # `_eval_executor` lets per-K work run in 8 separate Pool processes
            # instead of inside this thread (avoids GIL contention with GPU loop).
            cb_metrics = compute_epoch_test_eval(
                eval_data, config, test_loader, test_dataset=test_dataset,
                dataset_key=key, executor=_eval_executor, epoch=ep,
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
                    # |normal-anomaly| degeneracy gap (balanced_acc=0.5 is deceptive; see trainer.py)
                    cb_metrics['grl_acc_gap'] = abs(cb_metrics['grl_normal_acc'] - cb_metrics['grl_anomaly_acc'])
                    cb_metrics['grl_lambda'] = h['train_grl_lambda'][idx]
                    cb_metrics['grl_effective_weight'] = h['train_grl_effective_weight'][idx] if idx < len(h.get('train_grl_effective_weight', [])) else 0.0

            # NOTE: cb_metrics is RETURNED to _run_bg_all (no queue) — see end of fn.
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

                # Adaptive patch-level score.
                # SINGLE SOURCE: mae_anomaly.scoring.compute_score.
                # Pre-warmup gate: ``ep`` is the 1-indexed eval epoch; during
                # the teacher-only warmup window the saved adaptive_score must
                # be recon-only so it matches the per-epoch metrics (which use
                # the same gate via set_eval_context above). The raw
                # teacher_recon_error/discrepancy_error arrays below are saved
                # UN-gated (always raw) so offline recompute keeps full info.
                from mae_anomaly.scoring import compute_score, is_prewarmup_epoch
                adaptive_patch = compute_score(
                    recon_p, disc_p, fm_p, config,
                    force_recon_only=is_prewarmup_epoch(config, ep),
                )

                adaptive_scores = _aggregate_with_map(
                    adaptive_patch.ravel(), flat_t, flat_wp, coverage, covered, total_len, method='mean'
                )
                teacher_recon_scores = _aggregate_with_map(
                    recon_p.ravel(), flat_t, flat_wp, coverage, covered, total_len, method='mean'
                )
                disc_scores = _aggregate_with_map(
                    disc_p.ravel(), flat_t, flat_wp, coverage, covered, total_len, method='mean'
                )
                # Save FM-aggregated point score only when the bundle actually
                # carries one (Phase 1 2026-05-29 single-source refactor removed
                # the local ``use_fm_eval`` definition; re-derive from the
                # patch-level array directly so the test is local to this block).
                fm_scores = None
                if fm_p is not None and getattr(config, 'use_feature_matching', False):
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
                # [official] Also save the causal/online score (same R_tr/D_tr seed
                # as the best-epoch metric). 'adaptive_score' is preserved untouched.
                if getattr(config, 'official', False) and 'official_R_tr' in eval_data:
                    from mae_anomaly.scoring import compute_official_causal_score
                    save_dict['official_score'] = compute_official_causal_score(
                        teacher_recon_scores, disc_scores,
                        R_tr=eval_data['official_R_tr'], D_tr=eval_data['official_D_tr'],
                        force_recon_only=is_prewarmup_epoch(config, ep))
                np.savez_compressed(
                    os.path.join(epoch_scores_dir, f'epoch_{ep:03d}_scores.npz'),
                    **save_dict,
                )

            except Exception as e_scores:
                tqdm.write(f"  [Epoch {ep:>2}] SCORE SAVE ERROR: {e_scores}")
            return cb_metrics  # hand metrics back to _run_bg_all for recording
        except Exception as e:
            tqdm.write(f"  [Epoch {ep:>2}] ASYNC EVAL ERROR: {e}")
            return None

    def epoch_eval_callback(epoch, model, history):
        nonlocal callback_infer_time
        ep = epoch + 1
        if ep % eval_interval != 0 and ep != config.num_epochs:
            return

        # GPU inference (synchronous — must block while model is available)
        cb_start = time.time()
        try:
            eval_data = compute_epoch_test_inference(
                model, test_loader, config, test_dataset=test_dataset
            )

            # [official] Causal-score seed: run a TRAIN-normal inference at THIS
            # epoch's weights → (R_tr, D_tr), staged on eval_data for the bg eval
            # (drives best-epoch selection + the official_score npz). The
            # train-inference loader (stride=test_stride, no offset) is built once
            # and reused. No-op on the non-official path.
            if getattr(config, 'official', False):
                if _official_train_infer[0] is None:
                    _ti_ds = SlidingWindowDataset(
                        signals=signals, point_labels=point_labels, anomaly_regions=anomaly_regions,
                        window_size=config.seq_length, stride=test_stride, mask_last_n=config.patch_size,
                        split='train', train_ratio=train_ratio, seed=config.random_seed,
                        run_boundaries=run_boundaries, normalize_mode=config.normalize_mode,
                        minmax_range=getattr(config, 'minmax_range', '0_1'),
                        minmax_clamp_min=getattr(config, 'minmax_clamp_min', None),
                        minmax_clamp_max=getattr(config, 'minmax_clamp_max', None),
                        entity_segments=entity_segments,
                    )
                    _ti_kwargs = dict(num_workers=_dataloader_nw, pin_memory=False)
                    if _dataloader_nw > 0:
                        _ti_kwargs['prefetch_factor'] = _dataloader_prefetch
                    _official_train_infer[0] = DataLoader(
                        _ti_ds, batch_size=config.batch_size, shuffle=False, **_ti_kwargs)
                    _official_train_infer[1] = _ti_ds
                _R_tr, _D_tr = _official_train_seed(
                    model, _official_train_infer[0], _official_train_infer[1], config)
                eval_data['official_R_tr'] = _R_tr
                eval_data['official_D_tr'] = _D_tr

            # Compute contribution ratios from eval_data (pure numpy, ~1ms).
            # Pass epoch=ep so pre-warmup contribution drops the frozen-student
            # disc/FM term (matches the gated anomaly score).
            contrib = compute_contrib_from_eval_data(eval_data, config, epoch=ep)
            _trainer_ref[0]._pending_contrib = contrib

            cb_time = time.time() - cb_start
            callback_infer_time += cb_time

            # Stash eval_data for the end-of-epoch save callback (which also runs the
            # CPU eval on this data). The model/optimizer snapshot + checkpoint save
            # happen in post_epoch_save_callback, AFTER all per-epoch history appends,
            # so the persisted history is COMPLETE and resume-consistent (2026-05-30
            # root-cause fix). The GPU returns to training as soon as this returns.
            _pending_eval[0] = eval_data
        except Exception as e:
            cb_time = time.time() - cb_start
            callback_infer_time += cb_time
            print(f"  [Epoch {ep:>2}] EVAL ERROR: {e} ({cb_time:.0f}s)")

    def _clone_state_to_cpu(sd):
        """Detach+clone every tensor in a state_dict to CPU (race-free snapshot).

        Required because model.state_dict() returns LIVE param tensors (shared
        storage); the background torch.save runs concurrently with the next epoch's
        training, which would otherwise serialise post-mutation weights — a torn,
        epoch-inconsistent checkpoint. Cloning here pins this epoch's exact state.
        """
        out = {}
        for k, v in sd.items():
            if torch.is_tensor(v):
                out[k] = v.detach().to('cpu', copy=True)
            else:
                out[k] = v
        return out

    def _clone_optim_state(opt_sd):
        """Deep-clone an optimizer state_dict (Adam moments are CUDA tensors)."""
        import copy as _copy
        new = {'param_groups': _copy.deepcopy(opt_sd.get('param_groups', []))}
        st = {}
        for pid, pstate in opt_sd.get('state', {}).items():
            st[pid] = {kk: (vv.detach().to('cpu', copy=True) if torch.is_tensor(vv) else vv)
                       for kk, vv in pstate.items()}
        new['state'] = st
        return new

    def post_epoch_save_callback(epoch, history):
        """Run by Trainer at the END of EVERY epoch, AFTER all per-epoch history
        appends complete. On EVAL epochs it builds a race-free, epoch-consistent
        checkpoint and saves it (and runs the CPU eval) in the background; on
        non-eval epochs it returns immediately.

        Two root-cause guarantees (2026-05-30):
          1. History is snapshotted HERE (not mid-epoch in epoch_eval_callback), so
             every per-epoch key (contrib ratios, anomaly-type scores, timings, …) has
             length == epoch — a resumed run can never inherit an off-by-one.
          2. Model/optimizer state is CPU-cloned (not a live reference), so the async
             torch.save records exactly THIS epoch's weights even though the next
             epoch trains concurrently. ckpt['epoch'] == len(history) == model epoch.

        Checkpoints are written on eval epochs only (every EVAL_INTERVAL + the final
        epoch). Resume re-trains at most EVAL_INTERVAL epochs — fully acceptable and
        far cheaper than a 312 MB write every epoch. Resume never skips or drops a
        record: strict-normalize on load forces history to len == ckpt epoch and
        training resumes at ckpt_epoch + 1.
        """
        # [official] Save a MODEL-ONLY snapshot EVERY epoch (req 4) to a SEPARATE
        # namespace so it never clobbers best_checkpoint.pt / best_model.pt. Placed
        # BEFORE the eval-gated early-return so it fires on non-eval epochs too.
        # Skipped entirely when this dataset opted out of checkpoint saving
        # (_official_keep_ckpt=False ⇒ '저장 안함'); eval + viz are unaffected.
        if getattr(config, 'official', False) and _official_keep_ckpt:
            _ep_off = epoch + 1
            _off_dir = os.path.join(exp_dir, 'official_epochs')
            os.makedirs(_off_dir, exist_ok=True)
            torch.save(
                {'epoch': _ep_off, 'model_state_dict': _clone_state_to_cpu(model.state_dict())},
                os.path.join(_off_dir, f'epoch_{_ep_off:03d}.pt'),
            )

        # Staged by epoch_eval_callback ONLY on eval epochs. None → non-eval epoch →
        # nothing to persist (the next eval epoch's snapshot captures this epoch too).
        eval_data = _pending_eval[0]
        _pending_eval[0] = None
        if eval_data is None:
            return

        ep = epoch + 1
        _tr = _trainer_ref[0]
        # CPU-cloned, race-free snapshot of this epoch's exact state.
        ckpt_data = {
            'epoch': ep,
            'model_state_dict': _clone_state_to_cpu(model.state_dict()),
            'config': asdict(config),
            'resume_version': 1,
            'optimizer_state_dict': _clone_optim_state(_tr.optimizer.state_dict()),
            'scheduler_state_dict': _tr.scheduler.state_dict(),  # small, plain dict
            'history': {k: list(v) if isinstance(v, list) else v
                        for k, v in history.items()},  # COMPLETE for this epoch
            # epoch_metrics_list + best_ckpt_score are filled by _run_bg_all AFTER it
            # runs+records THIS epoch's eval (eval-before-checkpoint invariant), so the
            # checkpoint's epoch counter and its eval list are always consistent.
            'prev_adv_lambda': _tr._prev_epoch_adv_lambda,
            'prev_fm_lambda': _tr._prev_epoch_fm_lambda,
            'prev_grl_lambda': _tr._prev_epoch_grl_lambda,
            'lbm_state': _tr._lbm_state_dict(),  # loss_balance_mode runtime state (resume)
            '_frozen_eval_modules': getattr(_tr, '_frozen_eval_modules', None),
            # [신규 2026-06-14] freeze_encoder_only 활성 시 frozen 모듈 목록. 미트리거면 None.
            # resume에서 재적용 안 하면 epoch==warmup 트리거를 놓쳐 encoder가 조용히 un-freeze됨.
            '_frozen_encoder_modules': getattr(_tr, '_frozen_encoder_modules', None),
            # [신규 2026-06-01] teacher warmup early-stop으로 동적 단축된 warmup 종료점.
            # 미트리거면 None. 트리거 시 (epoch+1). resume에서 config.teacher_only_warmup_epochs
            # 복원에 사용 → post-trigger resume가 warmup으로 재진입하지 않도록 보장.
            '_early_stopped_warmup_end': getattr(_tr, '_early_stopped_warmup_end', None),
        }
        if _tr.discriminator is not None:
            ckpt_data['discriminator_state_dict'] = _clone_state_to_cpu(_tr.discriminator.state_dict())
            ckpt_data['d_optimizer_state_dict'] = _clone_optim_state(_tr.d_optimizer.state_dict())
            ckpt_data['d_scheduler_state_dict'] = _tr.d_scheduler.state_dict()
        if _tr.wdgrl_critic_optimizer is not None:
            ckpt_data['wdgrl_critic_state_dict'] = _clone_state_to_cpu(_tr.wdgrl_critic.state_dict())
            ckpt_data['wdgrl_optimizer_state_dict'] = _clone_optim_state(_tr.wdgrl_critic_optimizer.state_dict())
        if _tr.scaler is not None:
            ckpt_data['scaler_state_dict'] = _tr.scaler.state_dict()

        prev_thread = _async_eval_thread[0]
        t = threading.Thread(
            target=_run_bg_all,
            args=(eval_data, ep, ckpt_data, prev_thread),
            daemon=True,
        )
        t.start()
        _async_eval_thread[0] = t

    wall_start = time.time()
    _trainer_ref = [None]  # Shared reference for callback to access discriminator
    trainer = Trainer(model, config, train_loader, test_loader, verbose=True)
    _trainer_ref[0] = trainer

    # --- Resume detection (2026-05-28, v1) ---
    # If latest_checkpoint.pt exists with resume_version >= 1, restore optimizer/
    # scheduler/RNG state and start from saved epoch+1. Otherwise fresh start.
    _latest_ckpt_path = os.path.join(checkpoints_dir, 'latest_checkpoint.pt')
    _start_epoch = 0
    if os.path.exists(_latest_ckpt_path):
        try:
            _rsm = torch.load(_latest_ckpt_path, map_location=config.device, weights_only=False)
            if _rsm.get('resume_version', 0) >= 1:
                print(f"  📂 Resuming from epoch {_rsm['epoch']} ({_latest_ckpt_path})", flush=True)
                model.load_state_dict(_rsm['model_state_dict'])
                trainer.optimizer.load_state_dict(_rsm['optimizer_state_dict'])
                trainer.scheduler.load_state_dict(_rsm['scheduler_state_dict'])
                trainer.history = _rsm['history']
                # === Strict resume normalization (2026-05-30) ===
                # The checkpoint epoch is AUTHORITATIVE: model/optimizer state and the
                # `epoch` counter all correspond to exactly `_rsm['epoch']` completed
                # epochs. Force EVERY per-epoch history list to that length so a resumed
                # run can never carry an off-by-one to the end of training (epoch=500
                # but epoch_recon_ratio_*=499 → score-contribution stackplot crash).
                #   • `epoch` itself is rebuilt as 1..N (bulletproof epoch numbering).
                #   • shorter lists (pre-2026-05-30 buggy checkpoints) are back-filled
                #     with their last value (lost epoch's value is unrecoverable);
                #   • longer lists are truncated.
                # No-op for checkpoints written by the fixed save-side code (already
                # consistent at len == epoch).
                # NON_PER_EPOCH: lists NOT indexed by epoch — must be left untouched.
                #   - `epoch` is rebuilt explicitly below.
                #   - `batch_profiling` is a per-BATCH diagnostic assigned once at the
                #     epoch-1 profiling pass (len == n_profiled_batches, e.g. 9); padding
                #     it to the epoch count would fabricate duplicate batch rows.
                _NON_PER_EPOCH = {'epoch', 'batch_profiling'}
                _tgt = int(_rsm['epoch'])
                trainer.history['epoch'] = list(range(1, _tgt + 1))
                _n_norm = 0
                for _hk, _lst in trainer.history.items():
                    if _hk in _NON_PER_EPOCH or not isinstance(_lst, list) or len(_lst) == 0:
                        continue
                    if len(_lst) < _tgt:
                        _lst.extend([_lst[-1]] * (_tgt - len(_lst))); _n_norm += 1
                    elif len(_lst) > _tgt:
                        del _lst[_tgt:]; _n_norm += 1
                if _n_norm:
                    print(f"  ⚠️  resume history strict-normalize: {_n_norm} per-epoch keys forced to len={_tgt} (=ckpt epoch)", flush=True)
                epoch_metrics_list.extend(_rsm['epoch_metrics_list'])  # mutate (closures hold ref)
                _best_ckpt_score[0] = _rsm['best_ckpt_score']
                trainer._prev_epoch_adv_lambda = _rsm['prev_adv_lambda']
                trainer._prev_epoch_fm_lambda = _rsm['prev_fm_lambda']
                trainer._prev_epoch_grl_lambda = _rsm['prev_grl_lambda']
                trainer._lbm_load_state_dict(_rsm.get('lbm_state'))  # back-compat: absent → no-op
                if 'discriminator_state_dict' in _rsm and trainer.discriminator is not None:
                    trainer.discriminator.load_state_dict(_rsm['discriminator_state_dict'])
                    trainer.d_optimizer.load_state_dict(_rsm['d_optimizer_state_dict'])
                    trainer.d_scheduler.load_state_dict(_rsm['d_scheduler_state_dict'])
                if 'wdgrl_optimizer_state_dict' in _rsm and trainer.wdgrl_critic is not None:
                    trainer.wdgrl_critic.load_state_dict(_rsm['wdgrl_critic_state_dict'])
                    trainer.wdgrl_critic_optimizer.load_state_dict(_rsm['wdgrl_optimizer_state_dict'])
                if 'scaler_state_dict' in _rsm and trainer.scaler is not None:
                    trainer.scaler.load_state_dict(_rsm['scaler_state_dict'])
                # Re-apply teacher freeze if it was already activated pre-crash
                if _rsm.get('_frozen_eval_modules'):
                    trainer._frozen_eval_modules = _rsm['_frozen_eval_modules']
                    for _name in trainer._frozen_eval_modules:
                        _m = getattr(trainer.model, _name, None)
                        if _m is not None:
                            _m.eval()
                            for _p in _m.parameters():
                                _p.requires_grad_(False)
                    for _name in ['teacher_mask_token']:
                        _p = getattr(trainer.model, _name, None)
                        if _p is not None:
                            _p.requires_grad_(False)
                # [신규 2026-06-14] Re-apply encoder-only freeze if it activated pre-crash.
                # 트레이너의 freeze 트리거는 epoch==warmup에서만 발화하므로, warmup 이후
                # resume 시 이 재적용이 없으면 encoder가 다시 학습되어 freeze_encoder_only가 무효화됨.
                if _rsm.get('_frozen_encoder_modules'):
                    trainer._frozen_encoder_modules = _rsm['_frozen_encoder_modules']
                    for _name in trainer._frozen_encoder_modules:
                        _m = getattr(trainer.model, _name, None)
                        if _m is not None:
                            _m.eval()
                            for _p in _m.parameters():
                                _p.requires_grad_(False)
                # [신규 2026-06-01] teacher warmup early-stop으로 동적 단축된 warmup 종료점 복원.
                # 트리거 후 저장된 ckpt에서 재개 시, config의 원래 warmup(상한)으로 되돌아가
                # 다시 warmup에 진입하는 것을 방지. 트리거 전(또는 flag OFF)이면 _eswe is None
                # 또는 == 원래값 → 변경 없음(기존 거동 동일).
                if getattr(config, 'use_teacher_warmup_early_stop', False):
                    _eswe = _rsm.get('_early_stopped_warmup_end', None)
                    if _eswe is None:
                        _eswe = (_rsm.get('config') or {}).get('teacher_only_warmup_epochs', None)
                    if _eswe is not None and int(_eswe) < int(config.teacher_only_warmup_epochs):
                        print(f"  [Resume] early-stop warmup end 복원: teacher_only_warmup_epochs "
                              f"{config.teacher_only_warmup_epochs} → {int(_eswe)}", flush=True)
                        config.teacher_only_warmup_epochs = int(_eswe)
                        trainer._early_stopped_warmup_end = int(_eswe)
                _start_epoch = _rsm['epoch']  # ckpt epoch is 1-indexed; resume from this python index
                print(f"  Resumed: optimizer + scheduler + history + best_score={_best_ckpt_score[0]:.4f} "
                      f"+ {len(epoch_metrics_list)} prev evals", flush=True)
            else:
                print(f"  Found ckpt but no resume_version → fresh start", flush=True)
        except Exception as _rsm_e:
            print(f"  Resume load failed ({_rsm_e}) → fresh start", flush=True)

    # Pre-epoch hook: deterministic DataLoader generator reseed per epoch (resume-safe)
    def _reseed_train_loader(epoch_idx):
        _train_generator.manual_seed(config.random_seed + epoch_idx)

    trainer.train(
        epoch_callback=epoch_eval_callback,
        post_epoch_callback=post_epoch_save_callback,
        profile_n_batches=PROFILE_N_BATCHES,
        start_epoch=_start_epoch,
        pre_epoch_hook=_reseed_train_loader,
    )
    # Collect last async eval result (blocking: wait for final thread)
    _collect_async_eval(blocking=True)
    # Shutdown the per-K eval pool — training done, no more callbacks coming.
    try:
        _eval_executor.shutdown(wait=True, cancel_futures=False)
        print(f"  Eval pool shutdown OK ({_EVAL_POOL_WORKERS} workers)")
    except Exception as _e_shutdown:
        print(f"  Eval pool shutdown warning: {_e_shutdown}")
    wall_time = time.time() - wall_start
    train_time = wall_time - callback_infer_time  # Pure training time (no inference callback)
    history = trainer.history
    epochs_done = config.num_epochs
    per_epoch = train_time / max(epochs_done, 1)
    n_evals = len(epoch_metrics_list)
    print(f"Training complete: wall={wall_time:.0f}s, pure_train={train_time:.0f}s "
          f"({per_epoch:.1f}s/ep), gpu_infer_callback={callback_infer_time:.0f}s ({n_evals} evals) | {mem_status()}")

    # 2026-06-11: release training-loop host-RAM bloat BEFORE the memory-heavy
    # finalize (best-epoch inference + VUS sweep + viz). Ultra-fast tiny datasets
    # (e.g. MSL C-2 = 38 windows, ~0.1s/epoch) accumulate cyclic-ref/d_model=768
    # bloat during 500 epochs that, added to the finalize, OOMs (~30GB host-RAM).
    # An explicit collect here lets the finalize start from a low baseline.
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save batch profiling from epoch 1 (first N batches with per-component sync timing)
    batch_profiling = history.get('batch_profiling', [])
    if batch_profiling:
        save_batch_profiling(batch_profiling, config, len(train_loader), exp_dir)

    # Save epoch metrics
    epoch_metrics_path = os.path.join(exp_dir, 'epoch_metrics.json')
    with open(epoch_metrics_path, 'w') as f:
        json.dump({'eval_interval': eval_interval, 'epochs': epoch_metrics_list}, f, indent=2)
    print(f"  Epoch metrics saved: {epoch_metrics_path}")

    # Generate epoch-wise visualizations (feature stats only).
    # 2026-05-29: epoch_dashboard (plot_epoch_metrics) deferred to bg-worker —
    # rendered AFTER the post-train VUS sweep so the VUS / Aff-F1 / R-F1 row
    # is populated from the start (rather than rendered twice, first with
    # zero-filled VUS then re-rendered after sweep). bg-worker handles fallback
    # render if VUS sweep itself fails (see L1982 finally block). Feature
    # stats viz remains here — it is independent of VUS sweep and does not
    # require the bg-worker.
    epoch_viz_dir = os.path.join(exp_dir, 'visualization', 'epoch_metrics')
    plot_epoch_feature_stats(epoch_metrics_list, epoch_viz_dir)
    print(f"  Epoch feature-stats viz saved: {epoch_viz_dir} "
          f"(epoch_dashboard.png will be rendered by bg-worker after VUS sweep)")

    # ========== Find Best Epoch ==========
    # [post-warmup-forced] choose best_epoch ONLY among post-warmup epochs (epoch >
    # _warm_boundary); falls back to all epochs if none. UNCONDITIONAL. Matches the
    # greedy best_checkpoint guard above ⇒ the loaded best_checkpoint.pt and best_epoch
    # reference the SAME post-warmup epoch (consistent weights + label + metric).
    _best_em = select_best_epoch(epoch_metrics_list, _best_epoch_metric_key, _warm_boundary)
    if _best_em is not None:
        best_epoch = _best_em.get('epoch', config.num_epochs)
        best_score = _best_em.get(_best_epoch_metric_key, 0)
        best_prc = _best_em.get('prc_auc', 0)
    else:
        best_epoch = config.num_epochs
        best_score = -1.0
        best_prc = -1.0

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
            entity_segments=entity_segments,
        )
        # train_infer_loader: best-epoch single inference pass. workers reduce
        # one-shot load latency; persistent_workers=False since this loader is
        # used exactly once per dataset (no reuse benefit).
        _train_infer_loader_kwargs = dict(num_workers=_dataloader_nw, pin_memory=False)
        if _dataloader_nw > 0:
            _train_infer_loader_kwargs['prefetch_factor'] = _dataloader_prefetch
        train_infer_loader = DataLoader(
            train_infer_dataset, batch_size=config.batch_size, shuffle=False,
            **_train_infer_loader_kwargs,
        )
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

        # Scoring (same formula as test).
        # SINGLE SOURCE: mae_anomaly.scoring.compute_score.
        # Gate on best_epoch: the loaded weights are best_checkpoint.pt (the
        # best epoch). If that epoch is pre-warmup, the train reference score
        # must be recon-only too, matching the test-side scoring for the same
        # epoch (otherwise train/test score distributions would be inconsistent).
        from mae_anomaly.scoring import compute_score, is_prewarmup_epoch
        t_adaptive_patch = compute_score(
            train_recon_p, train_disc_p, train_fm_p, config,
            force_recon_only=is_prewarmup_epoch(config, best_epoch),
        )

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
        # Phase 1 2026-05-29: local re-derivation of fm-active flag, matching
        # the per-epoch save path (use_fm_eval was removed in the single-source
        # refactor — fm is now a first-class field on the bundle).
        if train_fm_p is not None and getattr(config, 'use_feature_matching', False):
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
        # KEEP_BEST_CKPT=1 환경변수 설정 시 보존 (274 재실험용).
        # Phase 4 (2026-05-29): the four KEEP_CHECKPOINT_DATASETS (SWaT_A1A2,
        # WaDi_A1, WaDi_A2, PSM) also preserve best_checkpoint.pt so the
        # weights can be reloaded for ad-hoc inference. ``key`` is the dataset
        # identifier and is already in scope from earlier in this function.
        _keep_for_dataset = key in KEEP_CHECKPOINT_DATASETS
        if os.environ.get('KEEP_BEST_CKPT') == '1' or _keep_for_dataset:
            _reason = 'KEEP_BEST_CKPT=1' if os.environ.get('KEEP_BEST_CKPT') == '1' \
                else f"dataset '{key}' in KEEP_CHECKPOINT_DATASETS"
            print(f"    [{_reason}] — preserving {best_ckpt_path}")
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
        # FM patches must travel with the dict so PatchScoresBundle.from_patch_scores_dict
        # can route them into the bg-worker evaluator. evaluator.fm_patches is populated
        # by _compute_patch_scores_all_patches when use_feature_matching=True (else None).
        'fm': getattr(evaluator, 'fm_patches', None),
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

    # Route FM through derive_pred_data so the visualization score uses the same
    # FM-inclusive formula as the NPZ adaptive_score (fixing the 2026-05-28
    # viz vs NPZ inconsistency). ``evaluator.fm_patches`` is populated by
    # ``_compute_patch_scores_all_patches`` when use_feature_matching=True.
    # patch_scores were computed with the best-epoch weights (best_checkpoint.pt
    # loaded above), so gate the viz score on best_epoch: a pre-warmup best epoch
    # gets a recon-only viz score consistent with its recon-only metrics.
    from mae_anomaly.scoring import is_prewarmup_epoch as _is_prewarmup_epoch
    pred_data = derive_pred_data(
        recon_p, disc_p, student_p, labels, sample_types,
        config, test_dataset, subset_indices=viz_indices,
        fm_patches=patch_scores.get('fm'),
        force_recon_only=_is_prewarmup_epoch(config, best_epoch),
    )
    # [official] Visualize the CAUSAL score at the best epoch (req 5+6). The
    # best-epoch npz holds the authoritative full-length official_score (the same
    # one that selected the best epoch + drives the final metrics); overriding the
    # full-length point score keeps viz consistent with the reported metrics.
    # Guarded by length-match so a shape mismatch can never corrupt the plot.
    if getattr(config, 'official', False):
        _best_off_npz = os.path.join(exp_dir, 'epoch_scores', f'epoch_{best_epoch:03d}_scores.npz')
        if os.path.exists(_best_off_npz):
            _off_nd = np.load(_best_off_npz)
            if ('official_score' in _off_nd.files
                    and pred_data.get('scores') is not None
                    and len(_off_nd['official_score']) == len(pred_data['scores'])):
                _off_sc = _off_nd['official_score']
                pred_data['scores'] = _off_sc
                pred_data['point_scores'] = _off_sc
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
        from mae_anomaly.evaluator import find_swat_region_22
        from mae_anomaly.dataset_sliding import AnomalyRegion
        train_end = int(len(signals) * train_ratio)
        test_anomaly_regions = [
            AnomalyRegion(start=r['start'] - train_end, end=r['end'] - train_end,
                          anomaly_type=r['anomaly_type'])
            for r in anomaly_regions_ser if r['start'] >= train_end
        ]
        region_22 = find_swat_region_22(test_anomaly_regions)
        if region_22 is not None:
            pred_data_excl22, detailed_data_excl22, excl22_keep_win = _filter_excl22_viz_data(
                pred_data, detailed_data, config, region_22.start, region_22.end)
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
    # KEEP_BEST_CKPT=1 환경변수 설정 시 best_checkpoint.pt 보존 (274 재실험용).
    # Phase 4 (2026-05-29): the four KEEP_CHECKPOINT_DATASETS (SWaT_A1A2,
    # WaDi_A1, WaDi_A2, PSM) preserve best_model.pt + best_checkpoint.pt +
    # latest_checkpoint.pt so a downstream reader can reload either the
    # best-epoch or last-epoch weights. For SWaT the excl22 best_model.pt is
    # also preserved alongside the full one. Other datasets keep the legacy
    # delete-after-inference path to bound disk usage across the 28 SMD
    # machines and 6 Exathlon apps.
    if not save_weights:
        keep_env = os.environ.get('KEEP_BEST_CKPT') == '1'
        keep_for_dataset = key in KEEP_CHECKPOINT_DATASETS
        keep_all_weights = keep_env or keep_for_dataset
        if keep_all_weights:
            _reason = 'KEEP_BEST_CKPT=1' if keep_env else f"dataset '{key}' in KEEP_CHECKPOINT_DATASETS"
            print(f"  [{_reason}] — preserving best_model.pt + best_checkpoint.pt + latest_checkpoint.pt")
            if is_swat_dual and exp_dir_excl22:
                print(f"    + SWaT excl22 best_model.pt + checkpoints symlink (excl22_epoch={timing.get('excl22_best_epoch', 'n/a')})")
        else:
            cleanup_list = [os.path.join(exp_dir, 'best_model.pt'),
                            os.path.join(checkpoints_dir, 'latest_checkpoint.pt'),
                            os.path.join(checkpoints_dir, 'best_checkpoint.pt')]
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

    # [official] '저장 안함' (per-dataset, global fallback): eval + viz have already
    # run (the bg-worker holds its data in memory — proven safe by the existing
    # cleanup above that also deletes best_model.pt mid-viz). Remove ALL checkpoint
    # artifacts for this dataset regardless of save_weights / KEEP_CHECKPOINT_DATASETS.
    # official_epochs/ was skipped during training, so this mostly clears best/last.
    if getattr(config, 'official', False) and not _official_keep_ckpt:
        import shutil as _sh_off
        _off_dir = os.path.join(exp_dir, 'official_epochs')
        if os.path.isdir(_off_dir):
            _sh_off.rmtree(_off_dir, ignore_errors=True)
        for _wf in [os.path.join(exp_dir, 'best_model.pt'),
                    os.path.join(checkpoints_dir, 'latest_checkpoint.pt'),
                    os.path.join(checkpoints_dir, 'best_checkpoint.pt')]:
            try:
                if os.path.exists(_wf):
                    os.remove(_wf)
            except OSError:
                pass
        if os.path.isdir(checkpoints_dir) and not os.listdir(checkpoints_dir):
            try:
                os.rmdir(checkpoints_dir)
            except OSError:
                pass
        print(f"  [official 저장안함] removed checkpoints for '{key}' (eval + viz preserved)")

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
    # === Patch 2 (2026-05-29): Main worker CPU affinity isolation ===
    # Reserve cores [0..total-n_bg-1] for main training process + its thread
    # children (background daemon thread launched per-eval) + _eval_executor
    # spawn processes (inherit parent affinity). Bg-worker (_cpu_eval_viz_worker)
    # extends its own affinity to [total-n_bg..total-1] in its body, see L1538.
    # On a 16-core system: main=[0-11] (12 cores), bg=[12-15] (4 cores).
    # Goal: prevent bg-worker BLAS thread leak from starving GPU dataloader.
    try:
        _total_cpu = os.cpu_count() or 16
        _n_bg = max(2, _total_cpu // 4)  # 16c→4, 24c→6, 32c→8
        _main_cores = set(range(_total_cpu - _n_bg))
        os.sched_setaffinity(0, _main_cores)
        _actual = sorted(os.sched_getaffinity(0))
        print(f"  [main] CPU affinity reserved: cores {_actual[0]}-{_actual[-1]} "
              f"({len(_actual)} cores; bg-worker will use cores {_total_cpu - _n_bg}-{_total_cpu - 1})",
              flush=True)
    except Exception as _aff_e:  # noqa: BLE001
        print(f"  [main] WARN CPU affinity set failed: {_aff_e}", flush=True)

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

    all_datasets = DATASETS + SMD_DATASETS + EXATHLON_DATASETS + SMAP_MSL_SIMPLE_DATASETS + TEP_TYPEGEN_DATASETS

    if args.list:
        print(f"\nSet {args.set}: {config_preset['description']}")
        print(f"{'#':>3} {'Key':<30} {'Loader':<30} {'Stride':>7} {'N50':>5} {'Subdir':<35}")
        print("-" * 115)
        for i, d in enumerate(all_datasets):
            print(f"{i:>3} {d['key']:<30} {d['loader']:<30} {d['train_stride']:>7} "
                  f"{'Yes' if d['normal50'] else 'No':>5} {d['results_subdir']:<35}")
        print(f"\nTotal: {len(DATASETS)} base + {len(SMD_DATASETS)} SMD + {len(EXATHLON_DATASETS)} Exathlon "
              f"+ {len(SMAP_MSL_SIMPLE_DATASETS)} SMAP/MSL-simple = {len(all_datasets)}")
        return

    # Apply config overrides BEFORE creating output dir (so suffix reflects overrides)
    if args.config_override:
        # Flatten: each item may contain multiple space-separated key=value pairs
        flat_kvs = []
        for item in args.config_override:
            flat_kvs.extend(item.split())
        _user_override_keys = []  # [official] keys the user EXPLICITLY passed (271-base layering)
        for kv in flat_kvs:
            if '=' not in kv:
                continue  # defensive: skip malformed bare token (neutral for every k=v)
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
            _user_override_keys.append(key)
        # [official] Stash the explicit-key list so run_base_experiment can layer
        # CANON_271 (base) < user-explicit < forced official bundle.
        config_preset['_user_override_keys'] = _user_override_keys
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
            if result is None:
                # Dataset was skipped via dataset-skip marker (experiment_metadata.json exists).
                # Do NOT append None to results — it pollutes downstream iteration over results.
                print(f"  Skipped: {dataset_def.get('key', '?')} (already completed)")
                continue
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
        # (2026-06-23) 600s join+terminate 제거. 그 timeout이 느린 viz(예: win100 detailed
        # 84만 샘플)를 렌더 도중 강제 종료해 viz 파일이 누락됐다. 기본은 **무제한 대기**로
        # bg eval+viz worker가 항상 끝까지 완료되게 한다. 진짜 hang 방지용 backstop이 필요하면
        # MAE_BG_JOIN_TIMEOUT(초)를 명시할 때만 그 시점에 terminate한다(미설정 시 None=무제한).
        _bg_to = os.environ.get('MAE_BG_JOIN_TIMEOUT')
        _bg_to = float(_bg_to) if _bg_to else None
        for name, p in _background_processes:
            p.join(timeout=_bg_to)
            if p.is_alive():
                print(f"  {name}: still running after MAE_BG_JOIN_TIMEOUT={_bg_to}s — terminating", flush=True)
                p.terminate()
            else:
                print(f"  {name}: completed", flush=True)

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
