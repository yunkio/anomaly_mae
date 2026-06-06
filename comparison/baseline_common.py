#!/usr/bin/env python
"""
Shared utilities for baseline comparison experiments.

All metric computation uses mae_anomaly/evaluator.py directly — no duplicate code.
Result format matches MAE output: epoch_metrics.json, scores.npz, epoch_scores/.

Contains:
- Metrics computation (MAE evaluator wrappers)
- Baseline model execution (train, predict, evaluate, save)
- Results file management (epoch_metrics.json, scores.npz, metadata.json)
- Model factory (create_model with per-experiment hyperparameters)
"""

import os
import sys
import json
import time
import inspect
import importlib
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# MAE Evaluator Imports (single source of truth)
# ============================================================
from mae_anomaly.evaluator import (
    compute_full_metric_set,   # UNIFIED SINGLE SOURCE OF TRUTH (2026-06-06)
    _zero_metric_set,          # MAE degenerate zero template (unified)
    find_f1_optimal_idx,
    compute_f1_t_at_threshold,
    compute_pa_k_metrics_from_mean_scores,
    compute_pa_k_roc_prc_from_mean_scores,
    compute_pa_k_auc,
    find_swat_largest_region,
    compute_metrics_with_exclusion,
    compute_extra_metrics,
    compute_ar_threshold_metric_set,
    EXTRA_METRIC_KEYS,
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_score, recall_score, f1_score as sklearn_f1_score,
)

# ============================================================
# Baseline Model Imports
# ============================================================
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
)

# anomaly_transformer: 2026-05-22 module dir 손실. import 격리 (legacy 결과는 보존).
try:
    from comparison.baselines import AnomalyTransformerBaseline
    HAS_ANOMALY_TRANSFORMER = AnomalyTransformerBaseline is not None
except ImportError:
    AnomalyTransformerBaseline = None
    HAS_ANOMALY_TRANSFORMER = False

# Optional SOTA imports
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

# --- 2026-05-19 batch (10 SOTA additions) ---
try:
    from comparison.baselines import TFMAEBaseline
    HAS_TFMAE = True
except ImportError:
    HAS_TFMAE = False

try:
    from comparison.baselines import TimesNetBaseline
    HAS_TIMESNET = True
except ImportError:
    HAS_TIMESNET = False

try:
    from comparison.baselines import DCdetectorBaseline
    HAS_DCDETECTOR = True
except ImportError:
    HAS_DCDETECTOR = False

try:
    from comparison.baselines import MEMTOBaseline
    HAS_MEMTO = True
except ImportError:
    HAS_MEMTO = False

try:
    from comparison.baselines import ModernTCNBaseline
    HAS_MODERNTCN = True
except ImportError:
    HAS_MODERNTCN = False

try:
    from comparison.baselines import CATCHBaseline
    HAS_CATCH = True
except ImportError:
    HAS_CATCH = False

try:
    from comparison.baselines import NPSRBaseline
    HAS_NPSR = True
except ImportError:
    HAS_NPSR = False

# --- 2026-05-29/30 weakly-supervised batch (4 models, Q1-only, additive) ---
# Import-isolated: a failure here leaves the 22 unsupervised baselines unaffected.
try:
    from comparison.baselines import WETASBaseline
    HAS_WETAS = WETASBaseline is not None
except ImportError:
    WETASBaseline = None
    HAS_WETAS = False

try:
    from comparison.baselines import TreeMILBaseline
    HAS_TREEMIL = TreeMILBaseline is not None
except ImportError:
    TreeMILBaseline = None
    HAS_TREEMIL = False

try:
    from comparison.baselines import NRdetectorBaseline
    HAS_NRDETECTOR = NRdetectorBaseline is not None
except ImportError:
    NRdetectorBaseline = None
    HAS_NRDETECTOR = False

try:
    from comparison.baselines import DeepMILBaseline
    HAS_DEEPMIL = DeepMILBaseline is not None
except ImportError:
    DeepMILBaseline = None
    HAS_DEEPMIL = False


# ============================================================
# Constants
# ============================================================

PA_K_VALUES = list(range(0, 101, 5))  # 0, 5, 10, ..., 100 (MAE와 동일)

BASELINE_MODELS = [
    'random', 'sensor_range', 'pca_error', 'l2_norm', 'nn_distance',
    'mlp', 'mlpmixer', 'transformer', 'gcn_lstm',
    'anomaly_transformer', 'tranad', 'usad', 'dagmm', 'gdn', 'omnianomaly',
    # === 2026-05-19 batch (7 active SOTA, post 3-model removal 2026-05-26) ===
    'tfmae', 'timesnet', 'dcdetector', 'memto', 'moderntcn',
    'catch', 'npsr',
]

SIMPLE_MODELS = ['random', 'sensor_range', 'pca_error', 'l2_norm', 'nn_distance']
NEURAL_MODELS = ['mlp', 'mlpmixer', 'transformer']
SOTA_MODELS = ['gcn_lstm', 'anomaly_transformer', 'tranad', 'usad', 'dagmm', 'gdn', 'omnianomaly',
               # 2026-05-19 batch (7 SOTA, epoch_callback path)
               'tfmae', 'timesnet', 'dcdetector', 'memto', 'moderntcn',
               'catch', 'npsr']

SOTA_AVAILABILITY = {
    'gcn_lstm': True,
    'anomaly_transformer': HAS_ANOMALY_TRANSFORMER,
    'tranad': HAS_TRANAD,
    'usad': HAS_USAD,
    'dagmm': HAS_DAGMM,
    'gdn': HAS_GDN,
    'omnianomaly': HAS_OMNIANOMALY,
    # 2026-05-19 batch (7 models)
    'tfmae': HAS_TFMAE,
    'timesnet': HAS_TIMESNET,
    'dcdetector': HAS_DCDETECTOR,
    'memto': HAS_MEMTO,
    'moderntcn': HAS_MODERNTCN,
    'catch': HAS_CATCH,
    'npsr': HAS_NPSR,
}

# ============================================================
# Weakly-supervised models (2026-05-29/30 SSL porting) — ADDITIVE / ISOLATED.
# These consume train_y (weak window/bag labels) and run via the dedicated
# `run_weak_sota_baseline_with_epoch_eval` path. They are NOT in BASELINE_MODELS /
# SOTA_MODELS, so the 22 existing baselines and their execution paths are untouched.
# Q1-only: each wrapper raises RuntimeError under normal-only (Q3) — see model wrappers.
# ============================================================
WEAK_SUPERVISED_MODELS = ['wetas', 'treemil', 'nrdetector', 'nrdetector_full', 'deepmil']

WEAK_SUPERVISED_AVAILABILITY = {
    'wetas': HAS_WETAS,
    'treemil': HAS_TREEMIL,
    'nrdetector': HAS_NRDETECTOR,
    'nrdetector_full': HAS_NRDETECTOR,  # noisy_rate=1.0 ablation variant (same wrapper)
    'deepmil': HAS_DEEPMIL,
}
# Expose weak-model availability through the same dict the runner already consults.
SOTA_AVAILABILITY.update(WEAK_SUPERVISED_AVAILABILITY)

# MAE-only metric keys that baselines set to null
_MAE_ONLY_KEYS = [
    'teacher_prc_auc', 'teacher_f1_t', 'teacher_pa_20_f1',
    'teacher_pak_auc_prc_auc', 'teacher_pak_auc_roc_auc',
    'teacher_pak_auc_f1', 'teacher_pak_auc_f1_t',
    'teacher_pak_auc_precision', 'teacher_pak_auc_recall',
    'teacher_pak_auc_f1_raw', 'teacher_pak_auc_f1_t_raw',
    'teacher_pak_auc_precision_raw', 'teacher_pak_auc_recall_raw',
    'disc_snr',
    'disturbing_roc_auc', 'disturbing_precision', 'disturbing_recall', 'disturbing_f1',
    'n_pure_normal', 'n_disturbing_normal', 'n_anomaly',
]


# ============================================================
# Model Hyperparameter Presets
# ============================================================

def _get_default_model_params():
    """QuoVadis-paper-faithful hyperparameters (epochs=50 user override, otherwise paper yaml)."""
    return {
        # ====== QuoVadis paper-faithful 9 baselines ======
        # Simple baselines — match quovadis_tad/baselines/simple_baselines.py defaults
        # + eval_simple_baselines.py:methods_dict actual call signatures.
        'random': {'seed': None},                                          # np.random.randint(0,2)
        'sensor_range': {'sensor_range': (0, 1), 'count_sensors': False},  # eval_simple_baselines.py:36
        'pca_error': {'pca_dim': 'auto', 'svd_solver': 'full'},          # eval_simple_baselines.py:39
        'l2_norm': {'ord': 2},                                            # eval_simple_baselines.py:37 (LNorm(ord=2))
        'nn_distance': {'distance': 'euclidean'},                         # eval_simple_baselines.py:38
        # Neural baselines — paper yaml (model_configs/*.yaml) with epochs=50 user override
        # mlp_small_embedd_32_seq_5.yaml: seq_len=5, embed=32, epochs=200, lr=0.001,
        #   batch=512, dropout=0.0, weight_decay=0.0001, optimizer=adam
        'mlp': {
            'seq_len': 5, 'embedding_dim': 32, 'dropout': 0.0,
            'lr': 0.001, 'weight_decay': 0.0,  # QuoVadis legacy-Keras Adam never applies wd (effective 0.0)
            'train_stride': 1, 'epochs': 10, 'batch_size': 512,  # 2026-06-06: unsupervised unified to 10
        },
        # MLPmixer_blocks_1_embedd_128_seq_5.yaml: seq_len=5, embed=128, epochs=100,
        #   lr=0.0002, batch=512, dropout=0.1, num_blocks=1
        'mlpmixer': {
            'seq_len': 5, 'embedding_dim': 128, 'num_blocks': 1, 'dropout': 0.1,
            'lr': 0.0002, 'weight_decay': 0.0,  # QuoVadis legacy-Keras Adam never applies wd (effective 0.0)
            'train_stride': 1, 'epochs': 10, 'batch_size': 512,  # 2026-06-06: unsupervised unified to 10
        },
        # Transformer_blocks_1_1_embedd_128_seq_5.yaml: seq_len=5, embed=128, MHA=1,
        #   num_blocks=1, epochs=100, lr=0.001, batch=512, dropout=0.1
        'transformer': {
            'seq_len': 5, 'embedding_dim': 128, 'num_heads': 1, 'num_blocks': 1, 'dropout': 0.1,
            'lr': 0.001, 'weight_decay': 0.0,  # QuoVadis legacy-Keras Adam never applies wd (effective 0.0)
            'train_stride': 1, 'epochs': 10, 'batch_size': 512,  # 2026-06-06: unsupervised unified to 10
        },
        # gcn_lstm_model_seq_5.yaml: seq_len=5, embed=10, lstm_units=64, output_dim=1,
        #   epochs=100, lr=0.001, batch=100, dropout=0.1
        'gcn_lstm': {
            'seq_len': 5, 'gcn_out_dim': 10, 'lstm_units': 64, 'output_seq_len': 1,
            'aggregation_type': 'max', 'combination_type': 'concat',
            'graph_conv_activation': 'relu',
            'adj_symmetric': True,
            'score_norm': 'median-iqr', 'score_smooth_window': 5, 'score_iqr_epsilon': 1e-2,
            'lr': 0.001, 'weight_decay': 0.0,  # faithful: upstream legacy-Adam ignores wd (gcn_lstm)
            'train_stride': 1, 'epochs': 10, 'batch_size': 100,  # 2026-06-06: unsupervised unified to 10
        },
        # SOTA baselines
        'anomaly_transformer': {'win_size': 100, 'd_model': 512, 'n_heads': 8, 'e_layers': 3, 'train_stride': 1, 'epochs': 10, 'batch_size': 128},
        'tranad': {'seq_len': 10, 'd_ff': 16, 'train_stride': 1, 'epochs': 10, 'batch_size': 128},
        'usad': {'seq_len': 5, 'latent_dim': 40, 'train_stride': 1, 'epochs': 10, 'batch_size': 256},  # latent: USAD Table-7 SWaT=40 (~median; no uniform anomaly default; seq_len 5=modal; epochs frozen)
        # DAGMM (TranAD-author impl, imperial-qore/TranAD): n_window=5 (flatten),
        # n_hidden=16, n_latent=8, AdamW(lr=1e-4, wd=1e-5), StepLR(5, 0.9), epochs=5.
        # n_gmm is derived (= n_feats × n_window) and not user-configurable.
        'dagmm': {'n_window': 5, 'n_hidden': 16, 'n_latent': 8,
                  'lr': 1e-4, 'weight_decay': 1e-5, 'lr_step_size': 5, 'lr_gamma': 0.9,
                  'train_stride': 1, 'epochs': 10, 'batch_size': 256},
        # GDN (Deng & Hooi, AAAI 2021): seq_len/embed_dim/top_k/out_layer_* match upstream run.sh
        # (slide_win=5, dim=64, topk=5, out_layer_num=1, out_layer_inter_dim=128).
        # batch_size=256 kept for uniformity with other SOTA presets (vs upstream 32 in run.sh).
        'gdn': {'seq_len': 5, 'embed_dim': 64, 'n_heads': 1,
                'out_layer_num': 1, 'out_layer_inter_dim': 128, 'output_dropout': 0.2,
                'lr': 0.001, 'train_stride': 1, 'epochs': 10, 'batch_size': 32},  # batch: GDN run.sh repro value 32 (argparse default 128 = wrong layer)
        # OmniAnomaly — paper-faithful preset (CLEAN_REIMPL 2026-05-24)
        # Upstream defaults (main.py ExpConfig): rnn_num_hidden=500, dense_dim=500, z_dim=3,
        #   window_length=100, nf_layers=20, std_epsilon=1e-4, batch_size=50, max_epoch=10,
        #   initial_lr=0.001, l2_reg=1e-4, test_n_z=1, beta=1.0 (full SGVB ELBO, no β-VAE).
        'omnianomaly': {
            'seq_len': 100, 'hidden_dim': 500, 'dense_dim': 500, 'z_dim': 3,
            'nf_layers': 20, 'std_epsilon': 1e-4,
            'beta': 1.0, 'n_mc_samples': 1, 'l2_reg': 1e-4,
            'train_stride': 1, 'epochs': 10, 'batch_size': 50,
        },
        # === Weakly-supervised batch (2026-05-29/30 SSL porting) — Q1-only ===
        # Values are official-config-sourced (see phase2_implementation/models/<M>/07_*).
        # WETAS — donalee/WETAS@cb149dc train_classifier.py:218-242 (epochs upstream=200; pipeline may
        #   override via --sota-epochs). split_size=window; output_size MUST equal hidden_size (fc quirk).
        'wetas': {
            'hidden_size': 128, 'output_size': 128, 'kernel_size': 2, 'n_layers': 7,
            'pooling_type': 'avg', 'local_threshold': 0.3, 'granularity': 4,
            'beta': 0.1, 'gamma': 0.1, 'split_size': 500,
            'batch_size': 32, 'lr': 1e-4, 'epochs': 50, 'train_stride': 500,  # 2026-06-06: weak unified to 50
        },
        # TreeMIL — fly-orange/TreeMIL@16f166c train.py argparse (BCE-only active loss; DTW dropped as dead code).
        'treemil': {
            'split_size': 500, 'epochs': 50, 'batch_size': 32,  # 2026-06-06: weak unified to 50 'lr': 1e-4,
            'ary_size': 2, 'inner_size': 3, 'd_model': 128, 'd_k': 128, 'd_v': 128,
            'd_inner_hid': 32, 'n_head': 5, 'n_layer': 2, 'dropout': 0.5,
            'agg_type': 'max', 'pooling_type': 'max', 'train_stride': 1,
        },
        # NRdetector — UCSC-REAL/NRdetector@bd5592b main.py/solver.py (PU; HOC+softDTW dropped).
        #   win_size=100 kept (NOT pipeline 500).
        #   prior=None → estimated dynamically from train wlabel rate (PU class prior = intrinsic
        #     anomaly ratio; dataset-dependent → must be estimated, NOT a fixed constant).
        #   noisy_rate=0.4 = experimenter-imposed REVEAL fraction (keep first 40% of positive
        #     segments as labeled-P, demote rest to unlabeled; selector.py:31-39). NOT a dataset
        #     property → kept as a fixed parameter (estimation would be a category error).
        #   normalization: original = per-split z-score StandardScaler (paper §5.2 "following Xu 2021";
        #     data_loader.py:50-53). Applied inside the wrapper on raw data (run_baseline SELF_NORMALIZING_WEAK).
        #   encoder_epochs/encoder_lr = IMPL-INVENTED (official loads pretrained .pth, no training recipe).
        'nrdetector': {
            'win_size': 100, 'noisy_rate': 0.4, 'prior': 0.25,  # upstream main.py:98 default 0.25
            'hidden_size': 64, 'output_size': 64, 'kernel_size': 2, 'n_layers': 7, 'd_model': 64,
            'classifier_hidden': 128, 'batch_size': 32, 'epochs': 50, 'encoder_epochs': 50,  # 2026-06-06: weak unified to 50
            'encoder_lr': 1e-3,  # IMPL-INVENTED (parameterized 2026-05-30; no official recipe)
            'lr': 1e-5, 'knn_k': 5, 'seed': 0, 'train_stride': 1,
        },
        # NRdetector full-reveal ablation variant (2026-05-31). Same wrapper, same preset
        # EXCEPT noisy_rate=1.0 — all positive train segments kept as labeled-P (no demotion to
        # unlabeled). Negative samples remain unlabeled (PU learning invariant). Produces a
        # separate output directory (model_name='nrdetector_full') so results don't collide
        # with the paper-default noisy_rate=0.4 run.
        'nrdetector_full': {
            'win_size': 100, 'noisy_rate': 1.0, 'prior': 0.25,  # upstream main.py:98 default 0.25
            'hidden_size': 64, 'output_size': 64, 'kernel_size': 2, 'n_layers': 7, 'd_model': 64,
            'classifier_hidden': 128, 'batch_size': 32, 'epochs': 50, 'encoder_epochs': 50,  # 2026-06-06: weak unified to 50
            'encoder_lr': 1e-3,
            'lr': 1e-5, 'knn_k': 5, 'seed': 0, 'train_stride': 1,
        },
        # DeepMIL — Sultani CVPR'18 MIL ranking head+loss (code-sourced, FAITHFUL) on the
        #   WETAS DiCNN encoder (DERIVATIVE_CITED, WETAS ICCV'21 p.7360 "DeepMIL ... = DiCNN").
        #   optimizer = Adam lr=1e-4 (WETAS train_classifier.py:234, the encoder's OWN optimizer):
        #   Sultani's Adagrad lr=0.01 (frozen-C3D shallow head) DIVERGES with the deep trainable
        #   DiCNN encoder (logits→-200/-440, score collapse). head/loss stay Sultani; only the
        #   optimizer follows the encoder source. (60 bags/batch = 30 pos + 30 neg.)
        #   normalization = per-recording StandardScaler (WETAS-lineage; wrapper applies on raw).
        #   n_segments vestigial (config back-compat; dense per-timestep MIL, not 32-seg).
        'deepmil': {
            'n_segments': 32, 'dropout': 0.6, 'ranking_margin': 1.0,
            'lambda_smooth': 8e-5, 'lambda_sparse': 8e-5, 'l2_reg': 0.001,
            'optimizer': 'adam', 'lr': 1e-4,  # WETAS-sourced (Adagrad 0.01 diverges w/ deep DiCNN)
            'bags_per_batch': 60, 'seq_len': 128, 'encoder_dim': 128,
            'test_stride': 1, 'aggregation': 'mean', 'epochs': 50,  # 2026-06-06: weak unified to 50
            'iters_per_epoch': 50, 'train_stride': 1,
        },
        # === 2026-05-19 batch (10 SOTA additions) ===
        # Phase 1: TFMAE — MAE-direct competitor (ICDE 2024)
        # Paper SWaT.conf (LMissher/TFMAE/config/SWaT.conf 2026-05-22 검증):
        #   ws=100, d=128, l=3, fr=0.4, tr=0.25, ss=10, bs=64, lr=1e-4
        # tr 0.5→0.25, seq_size 5→10 (paper-faithful).
        'tfmae': {
            'win_size': 100, 'seq_size': 10,
            'd_model': 128, 'e_layers': 3, 'n_heads': 8,
            'temporal_mask_ratio': 0.25, 'freq_mask_ratio': 0.4,
            'dropout': 0.05, 'lr': 1e-4,
            'train_stride': 1, 'epochs': 10, 'batch_size': 64,
            'k_temperature': 50.0,
        },
        # Phase 2: TimesNet — General SOTA (ICLR 2023)
        'timesnet': {
            'win_size': 100, 'd_model': 128, 'd_ff': 128,  # SMAP anomaly script (scripts/anomaly_detection/SMAP/TimesNet.sh): d_model=128, d_ff=128
            'e_layers': 3, 'top_k': 3, 'num_kernels': 6,  # SMAP anomaly script: e_layers=3, top_k=3
            'dropout': 0.1, 'lr': 1e-4,
            'train_stride': 1, 'epochs': 10, 'batch_size': 128,
        },
        # Phase 2: DCdetector — Dual-attention contrastive (KDD 2023)
        'dcdetector': {
            'win_size': 105, 'patch_size': [3, 5, 7],
            'd_model': 256, 'n_heads': 1, 'e_layers': 3,
            'dropout': 0.0, 'lr': 1e-4,
            'train_stride': 1, 'epochs': 10, 'batch_size': 128,
            'k_temperature': 50.0,
        },
        # Phase 3: MEMTO — Memory-guided Transformer (NeurIPS 2023) — paper-faithful
        # Upstream (gunny97/MEMTO):
        #   solver.py:195-198  loss = rec_loss + lambd * entropy_loss     (lambd=0.01, NO gather term)
        #   solver.py:258-261  score = softmax(gather/τ, dim=-1) * recon  (τ=0.1 over seq dim)
        #   test.sh            Phase 1 lr=1e-4, Phase 2 (memory_initial) lr=5e-5
        #   data_loader.py     StandardScaler train-only fit (Path B; memto ∈ SELF_NORMALIZING_SOTA)
        'memto': {
            'win_size': 100, 'n_memory': 10,
            'd_model': 512, 'n_heads': 8, 'e_layers': 3, 'd_ff': 512,
            'dropout': 0.0, 'shrink_thres': 0.0,
            'lambda_entropy': 0.01,   # was 0.1; upstream lambd=0.01 (test.sh, main.py)
            'temperature': 0.1,       # softmax τ in inference score formula (main.py default)
            'lr': 1e-4,               # Phase 1 lr (test.sh mode=train)
            'phase2_lr': 5e-5,        # Phase 2 lr (test.sh mode=memory_initial)
            'train_stride': 100, 'epochs': 10, 'phase1_epochs': 3,  # stride: upstream step=100 (non-overlap)
            'batch_size': 128,
        },
        # Phase 3: ModernTCN — Modern convolutional TSA (ICLR 2024 Spotlight)
        # Paper SWaT.sh (luodhhh/ModernTCN/ModernTCN-detection/scripts/SWaT.sh 2026-05-22 검증):
        #   seq_len 100, ffn_ratio 1, patch 8/4, num_blocks 3, large_size 51, small_size 5,
        #   dims 128, dropout 0.1, head_dropout 0.0, lr 0.0003, bs 128, affine 0, subtract_last 0,
        #   use_multi_scale False, small_kernel_merged False
        # LR schedule (upstream `exp_anomaly_detection.py` + `utils/tools.py:adjust_learning_rate`):
        #   argparse default `--lradj=type1` → lr_new = base_lr * 0.5^((epoch_1based-1)//1)
        #   i.e. per-epoch halving starting after epoch 1. We model this as
        #   `lr_decay_per_epoch=0.5`. The OneCycleLR object in upstream is constructed but
        #   `.step()` is never called (dead code for SWaT).
        'moderntcn': {
            'win_size': 100, 'patch_size': 8, 'patch_stride': 4,
            'dims': [128], 'num_blocks': [3],
            'large_size': [51], 'small_size': [5],
            'ffn_ratio': 1, 'dropout': 0.1, 'head_dropout': 0.0,
            'use_revin': True, 'affine': False, 'subtract_last': False,
            'use_multi_scale': False, 'small_kernel_merged': False,
            'stem_ratio': 6, 'downsample_ratio': 2,
            'lr': 3e-4,                  # paper SWaT 0.0003
            'lr_decay_per_epoch': 0.5,   # upstream lradj='type1' = per-epoch halving
            'train_stride': 1, 'epochs': 10, 'batch_size': 128,
        },
        # Phase 5: CATCH — channel-aware freq reconstruction (ICLR 2025)
        # Paper: upstream DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS + SWAT_script/CATCH.sh override.
        #
        # DEFAULT (decisionintelligence/CATCH/ts_benchmark/baselines/catch/CATCH.py):
        #   lr=1e-4, Mlr=1e-5, e_layers=3, n_heads=2, cf_dim=64, d_ff=256, d_model=128,
        #   head_dim=64, individual=0, dropout=0.2, head_dropout=0.1,
        #   auxi_loss="MAE", auxi_type="complex", auxi_mode="fft",
        #   auxi_lambda=0.005, score_lambda=0.05, dc_lambda=0.005,
        #   regular_lambda=0.5, temperature=0.07,
        #   patch_stride=8, patch_size=16, inference_patch_stride=1, inference_patch_size=32,
        #   num_epochs=3, batch_size=128, seq_len=192, pct_start=0.3,
        #   revin=1, affine=0, subtract_last=0
        #
        # SWAT_script/CATCH.sh override:
        #   auxi_lambda=0, batch_size=32, inference_patch_size=256, inference_patch_stride=32,
        #   patch_size=256, patch_stride=64, score_lambda=0, seq_len=2048
        #
        # 우리는 SWaT 위주이므로 SWaT override를 default로 사용. 다른 데이터셋도 동일 preset
        # (single-preset 정책). 단 seq_len=2048은 작은 데이터셋(sim 등)에서 windowing 불가하므로
        # paper default seq_len=192 사용 + 다른 SWaT override는 모두 반영. patch_size=16 (default)
        # 도 짧은 시퀀스에 적합. 즉 paper default fully (single preset).
        'catch': {
            'win_size': 192,                      # paper default seq_len
            'patch_size': 16, 'patch_stride': 8,  # paper default
            'd_model': 128, 'n_heads': 2,         # paper default (was 8 → 2)
            'e_layers': 3,                        # paper default (was 2 → 3)
            'cf_dim': 64, 'd_ff': 256, 'head_dim': 64,  # paper defaults
            'individual': False,
            'dropout': 0.2, 'head_dropout': 0.1,  # paper default (was 0.1/0.0)
            'auxi_loss': "MAE", 'auxi_type': "complex", 'auxi_mode': "fft",
            'lambda_ch_discover': 0.005,          # paper dc_lambda (was 0.5)
            'lambda_freq': 0.005,                 # paper auxi_lambda (was 0.5)
            'score_lambda': 0.05,                 # paper default
            'regular_lambda': 0.5, 'temperature': 0.07,
            'inference_patch_size': 32, 'inference_patch_stride': 1,
            'module_first': True, 'mask': False,
            'use_revin': True, 'affine': False, 'subtract_last': False,
            'mlr_ratio': 0.1,                     # Mlr = lr × 0.1 (paper Mlr=1e-5 vs lr=1e-4)
            'pct_start': 0.3,                     # OneCycleLR pct_start (paper default)
            'lr': 1e-4,
            'train_stride': 1, 'epochs': 10, 'batch_size': 128,  # 2026-06-06: unsupervised unified to 10
        },
        # NPSR — Nominality Score Conditioned Anomaly Detection (NeurIPS 2023)
        # Paper-faithful preset (clean reimpl, 2026-05-24):
        #   - architecture: Performer-based M_pt (with z_dim latent bottleneck) +
        #                   M_seq sequence reducer (PerfPredSqz).
        #   - normalization: Path B (wrapper-internal MinMax(-1,1) + test clamp ±4).
        #   - scoring: nominality + induced anomaly score, default d=16 gate='soft'.
        'npsr': {
            # Windowing
            'win_size':       100,    # M_pt window length (reference `dl`, SWaT/WADI/PSM=100)
            'pred_dl':        100,    # M_seq input length (left+right halves concat)
            'delta':          20,     # M_seq target length / induction span
            'train_stride':   10,     # reference default
            # Architecture
            'z_dim':          10,    # paper Table 4 (real datasets); config.txt default 4 = trimSyn-synthetic only      # M_pt latent bottleneck (reference `lat`)
            'ff_mult':        4,
            'enc_depth':      4,      # M_pt Performer depth (reference `enc_depth`)
            'pred_depth':     8,     # paper Table 4 N_enc=8 (real datasets); config.txt default 4 = trimSyn-only      # M_seq Performer depth (reference `pred_depth`)
            'n_heads':        4,      # auto-adjusted to divide d_model=enc_in at runtime
            'dropout':        0.0,    # reference uses Performer defaults (no extra dropout)
            # Normalization (Path B)
            'clamp_max':      4.0,
            'clamp_min':     -4.0,
            # Scoring (induced anomaly score; paper Sec 3.4 default)
            'theta_N_ratio':  0.9985,
            'induction_d':    16,
            'gate_func':      'soft',
            # Training
            'lr':             1e-4,
            'batch_size':     64,
            'epochs':         10,     # KEPT (user: do not touch epochs)
        },
    }


MODEL_PRESETS = {
    'default': _get_default_model_params,
    # Legacy aliases (point to same config)
    'wadi_14days': _get_default_model_params,
    'general': _get_default_model_params,
}


# ============================================================
# Metrics Computation (MAE evaluator 직접 사용)
# ============================================================

def compute_all_metrics(
    point_scores: np.ndarray,
    point_labels: np.ndarray,
    anomaly_regions: list,
    eval_mask: np.ndarray = None,
    *,
    lite: bool = False,
) -> dict:
    """Compute all metrics — THIN WRAPPER over the UNIFIED single source of truth.

    UNIFIED CODE PATH (2026-06-06): every metric value is produced by
    ``mae_anomaly.evaluator.compute_full_metric_set`` — the EXACT function the MAE
    pipeline uses (``Evaluator.evaluate``). Baseline and MAE therefore compute
    metrics through ONE code path (identical threshold convention, PA%K, VUS, AR,
    etc.); no duplicate orchestration can drift apart again. This wrapper adds ONLY
    the baseline presentation layer (NOT metric logic):
      - convenience aliases ``aff_f1`` / ``r_f1`` (exp-monitor SKILL.md columns),
      - MAE-only keys (``teacher_*``, ``disc_snr``, ``disturbing_*``) set to None so
        the baseline epoch_metrics.json shares MAE's column schema,
      - drops ``_``-prefixed diagnostic arrays (matches MAE's own save filter).

    ``lite=False`` (default) computes the full set incl. VUS (prior baseline
    behaviour, byte-identical output except the threshold-convention fix below);
    ``lite=True`` skips ONLY VUS for per-epoch speed (same semantics as MAE).

    NOTE (2026-06-06): point ``precision``/``recall``/``f1_score`` now use the strict
    ``>`` decision convention (Kim et al. AAAI 2022) inherited from
    ``compute_full_metric_set`` — previously this function used ``>=``. All other
    keys are unchanged (they were already computed by the shared evaluator primitives).

    Args:
        point_scores: 1D float array, anomaly score per test timestep.
        point_labels: 1D int array, ground-truth label per test timestep.
        anomaly_regions: list of region objects with ``.start`` / ``.end`` (test-local).
        eval_mask: optional boolean mask (True = evaluate). Default: all True.
        lite: if True skip VUS (per-epoch speed); default False = full set.

    Returns:
        Flat dict matching MAE epoch_metrics.json schema (MAE-only keys = None).
    """
    results = compute_full_metric_set(
        point_scores, point_labels, anomaly_regions, eval_mask, lite=lite,
    )
    # ---- baseline presentation layer ONLY (NOT metric logic) ----
    # Drop `_`-prefixed diagnostic arrays (matches MAE's own save filter, evaluator.py:2213).
    results = {k: v for k, v in results.items() if not k.startswith('_')}
    # Convenience aliases matching exp-monitor SKILL.md column names.
    results['aff_f1'] = float(results.get('affiliation_f1', 0.0))
    results['r_f1']   = float(results.get('r_based_f1', 0.0))
    # MAE-only keys (computed only by the MAE model): null in baseline output.
    for key in _MAE_ONLY_KEYS:
        results[key] = None
    return results


def compute_all_metrics_with_excl(
    point_scores: np.ndarray,
    point_labels: np.ndarray,
    anomaly_regions: list,
    excl_region,
) -> dict:
    """Compute metrics excluding largest test anomaly region (excl22).

    Uses MAE's compute_metrics_with_exclusion — returns ALL metrics with excl22_ prefix.
    This matches MAE's run_base_experiments.py behavior (full metric set for excl22).
    """
    excl_metrics = compute_metrics_with_exclusion(
        point_scores, point_labels, anomaly_regions, excl_region
    )
    return {f'excl22_{k}': v for k, v in excl_metrics.items()}


def _zero_metrics() -> dict:
    """Zero-initialized metrics — delegates to MAE ``_zero_metric_set`` (UNIFIED 2026-06-06).

    Same schema/keys as ``compute_all_metrics`` on a degenerate input, so
    epoch_metrics.json keys stay consistent across normal and degenerate epochs.
    Adds only the baseline presentation keys (aliases + MAE-only None).
    """
    results = dict(_zero_metric_set())          # MAE single-source zero template
    results['aff_f1'] = 0.0
    results['r_f1']   = 0.0
    for key in _MAE_ONLY_KEYS:
        results[key] = None
    return results


# ============================================================
# Score & Metric Saving (MAE 형식)
# ============================================================

def save_scores_npz(output_dir: Path, scores: np.ndarray, filename: str = 'scores.npz'):
    """Save point-level anomaly scores as .npz (MAE format)."""
    np.savez_compressed(
        output_dir / filename,
        anomaly_score=np.nan_to_num(scores, nan=0.0).astype(np.float32),
    )


def save_epoch_scores_npz(epoch_scores_dir: Path, epoch: int, scores: np.ndarray):
    """Save epoch-level scores to epoch_scores/ directory."""
    epoch_scores_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        epoch_scores_dir / f'epoch_{epoch:03d}_scores.npz',
        anomaly_score=np.nan_to_num(scores, nan=0.0).astype(np.float32),
    )


def save_epoch_metrics(output_dir: Path, epoch_metrics_list: list, eval_interval: int = 1):
    """Save epoch_metrics.json in MAE format (atomic write).

    Atomicity matters because the resume path (2026-05-31) now calls this
    function after EVERY epoch (was: once at end-of-fit). A `kill -9` mid-write
    on the old `open('w')` truncated path would leave a zero-byte or partial
    json that breaks the resume reconciliation. Mirror the same write-tmp +
    os.replace pattern as ``_checkpoint.atomic_save`` to make the swap
    POSIX-atomic.
    """
    target = output_dir / 'epoch_metrics.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump({'eval_interval': eval_interval, 'epochs': epoch_metrics_list}, f, indent=2)
    os.replace(tmp, target)


def save_metadata(output_dir: Path, model_name: str, experiment_name: str,
                  train_time: float, inference_time: float, extra: dict = None):
    """Save metadata.json."""
    metadata = {
        "model_name": model_name,
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "timing": {
            "train_time": train_time,
            "inference_time": inference_time,
        },
    }
    if extra:
        metadata.update(extra)
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


# ============================================================
# Simple Baseline Execution (no epochs)
# ============================================================

def _aggregate_run_metrics(per_run_metrics: list) -> dict:
    """Aggregate metrics across N independent runs (paper §4.2 random baseline).

    Scalar metric values become the MEAN across runs (the reported value), with an
    added ``<key>_std`` sample-std for every aggregated numeric key. Non-numeric/None
    values and bookkeeping keys are taken from the first run unchanged. For a single
    run the output equals that run's dict (no _std keys added).
    """
    if not per_run_metrics:
        return {}
    base = dict(per_run_metrics[0])
    if len(per_run_metrics) == 1:
        return base
    _SKIP = {'epoch', '_inference_time', '_eval_time'}
    for key, v0 in list(base.items()):
        if key in _SKIP:
            continue
        vals = [r.get(key) for r in per_run_metrics]
        if any(v is None or isinstance(v, bool) or not isinstance(v, (int, float))
               for v in vals):
            continue  # leave first-run value (e.g. None MAE-only keys)
        arr = np.asarray(vals, dtype=np.float64)
        base[key] = float(arr.mean())
        base[f'{key}_std'] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    base['epoch'] = 1
    return base


def run_simple_baseline(
    model,
    train_X: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    anomaly_regions: list,
    model_name: str,
    output_dir: Path,
    experiment_name: str,
    excl_region=None,
) -> dict:
    """Run a simple (non-DL) baseline: fit, predict, evaluate, save.

    Result format:
    - scores.npz (key: anomaly_score)
    - epoch_metrics.json (1 epoch)
    - metadata.json

    Returns: metrics dict (flat, MAE format)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running {model_name}")
    print(f"{'='*60}")
    print(f"  Train shape: {train_X.shape}")
    print(f"  Test shape: {test_X.shape}")

    # Fit (stateless / negligible for simple baselines; done once and reused
    # across runs — only predict() is re-randomized for the random baseline).
    start_time = time.time()
    model.fit(train_X)
    train_time = time.time() - start_time
    print(f"  Train time: {train_time:.2f}s")

    # ---- Number of independent runs ----
    # Paper §4.2 (QuoVadis): the random baseline reports "the score achieved over
    # five independent runs". With seed=None, each model.predict() re-seeds NumPy
    # from OS entropy → independent draw. Deterministic baselines run once.
    n_runs = 5 if model_name == 'random' else 1

    per_run_metrics = []
    inference_time = 0.0
    scores = None  # keep last run's scores for scores.npz / viz
    for run_idx in range(n_runs):
        # Predict
        start_time = time.time()
        scores = model.predict(test_X)
        run_infer = time.time() - start_time
        inference_time += run_infer

        # Evaluate
        eval_start = time.time()
        m = compute_all_metrics(scores, test_y, anomaly_regions)
        m['_inference_time'] = run_infer
        m['_eval_time'] = time.time() - eval_start
        m['epoch'] = 1

        # SWaT excl22
        if excl_region is not None:
            excl = compute_all_metrics_with_excl(scores, test_y, anomaly_regions, excl_region)
            m.update(excl)

        per_run_metrics.append(m)
        if n_runs > 1:
            print(f"  [run {run_idx + 1}/{n_runs}] "
                  f"PAK_F1={m.get('pak_auc_f1', 0):.4f} PRC={m.get('prc_auc', 0):.4f}")

    # ---- Aggregate: mean over runs (reported value) + std (uncertainty) ----
    # For n_runs == 1 this is identical to the old single-run dict.
    metrics = _aggregate_run_metrics(per_run_metrics)

    # Save (scores.npz = last run; representative single-run artifact for viz)
    save_scores_npz(output_dir, scores)
    save_epoch_metrics(output_dir, [metrics])
    save_metadata(
        output_dir, model_name, experiment_name, train_time, inference_time,
        extra=({'n_runs': n_runs,
                'per_run_metrics': per_run_metrics} if n_runs > 1 else None),
    )

    # Save model if supported
    if hasattr(model, 'save'):
        try:
            model.save(output_dir / "model")
        except Exception as e:
            print(f"  Warning: Could not save model: {e}")

    _print_metrics(metrics, excl_region is not None)
    if n_runs > 1:
        print(f"  [aggregate over {n_runs} runs] "
              f"PAK_F1={metrics.get('pak_auc_f1', 0):.4f} "
              f"± {metrics.get('pak_auc_f1_std', 0):.4f}")
    return metrics


# ============================================================
# DL Baseline Execution (with epochs)
# ============================================================

def run_dl_baseline(
    model,
    train_X: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    anomaly_regions: list,
    model_name: str,
    output_dir: Path,
    experiment_name: str,
    excl_region=None,
) -> dict:
    """Run a DL baseline with epoch-level scoring and async CPU eval.

    For each epoch:
    - GPU: training step
    - GPU: test inference (synchronous)
    - CPU: metric computation (async background thread)
    - Save epoch_scores/epoch_{NNN}_scores.npz

    Result format:
    - scores.npz (final epoch)
    - epoch_scores/epoch_{NNN}_scores.npz (each epoch)
    - epoch_metrics.json (all epochs)
    - metadata.json
    - model/ (saved weights)

    Returns: final epoch metrics dict
    """
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_scores_dir = output_dir / 'epoch_scores'
    epoch_scores_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running {model_name} (DL baseline)")
    print(f"{'='*60}")
    print(f"  Device: {getattr(model, 'device', 'cpu')}")
    print(f"  Train shape: {train_X.shape}")
    print(f"  Test shape: {test_X.shape}")
    print(f"  Epochs: {getattr(model, 'epochs', '?')}")

    # Fit model (handles its own training loop)
    start_time = time.time()
    model.fit(train_X)
    train_time = time.time() - start_time
    print(f"\n  Train time: {train_time:.2f}s")

    # After training complete, do per-epoch eval with scores
    # DL baselines train internally — we get final predictions only
    # For epoch-level scoring, we need to modify the training loop
    # Since model.fit() handles training, we do a single eval here
    infer_start = time.time()
    scores = model.predict(test_X)
    inference_time = time.time() - infer_start
    print(f"  Inference time: {inference_time:.2f}s")

    # Evaluate (async eval for consistency, but single eval here)
    eval_start = time.time()
    metrics = compute_all_metrics(scores, test_y, anomaly_regions)
    metrics['_inference_time'] = inference_time
    metrics['_eval_time'] = time.time() - eval_start
    metrics['epoch'] = getattr(model, 'epochs', 1)

    # SWaT excl22
    if excl_region is not None:
        excl = compute_all_metrics_with_excl(scores, test_y, anomaly_regions, excl_region)
        metrics.update(excl)

    # Save final scores
    save_scores_npz(output_dir, scores)
    # Also save as epoch score
    final_epoch = getattr(model, 'epochs', 1)
    save_epoch_scores_npz(epoch_scores_dir, final_epoch, scores)

    # If model has train_loss_history, create epoch entries for each
    epoch_metrics_list = []
    if hasattr(model, 'train_loss_history') and model.train_loss_history:
        # We only have final scores, so all intermediate epochs get null metrics
        # Only the final epoch gets real metrics
        for ep_idx, loss in enumerate(model.train_loss_history):
            if ep_idx == len(model.train_loss_history) - 1:
                # Final epoch — real metrics
                ep_metrics = dict(metrics)
                ep_metrics['epoch'] = ep_idx + 1
                ep_metrics['train_loss'] = float(loss)
            else:
                # Intermediate epochs — just loss
                ep_metrics = _zero_metrics()
                ep_metrics['epoch'] = ep_idx + 1
                ep_metrics['train_loss'] = float(loss)
                ep_metrics['_inference_time'] = None
                ep_metrics['_eval_time'] = None
            epoch_metrics_list.append(ep_metrics)
    else:
        epoch_metrics_list = [metrics]

    save_epoch_metrics(output_dir, epoch_metrics_list)
    save_metadata(output_dir, model_name, experiment_name, train_time, inference_time)

    # Save model
    if hasattr(model, 'save'):
        try:
            model.save(output_dir / "model")
        except Exception as e:
            print(f"  Warning: Could not save model: {e}")

    _print_metrics(metrics, excl_region is not None)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def run_dl_baseline_with_epoch_eval(
    model,
    train_X: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    anomaly_regions: list,
    model_name: str,
    output_dir: Path,
    experiment_name: str,
    excl_region=None,
    eval_interval: int = 1,
    max_eval_workers: int = 10,
    train_windows: np.ndarray = None,
    train_targets: np.ndarray = None,
    test_segments=None,
) -> dict:
    """Run a DL baseline with per-epoch inference + synchronous CPU eval.

    Unlike run_dl_baseline(), this function takes control of the training loop
    to insert per-epoch inference and evaluation.

    This works for NeuralBaselineBase subclasses (MLP, MLPMixer, Transformer, GCN-LSTM)
    which expose the training loop internals.

    Args:
        eval_interval: Evaluate every N epochs (default: every epoch)
        max_eval_workers: unused (kept for API compatibility)
        train_windows: Pre-built training windows (e.g. segment-aware). Skips internal windowing.
        train_targets: Pre-built training targets. Required if train_windows is provided.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_scores_dir = output_dir / 'epoch_scores'
    epoch_scores_dir.mkdir(parents=True, exist_ok=True)

    segment_aware = train_windows is not None
    label = "segment-aware DL" if segment_aware else "DL baseline"

    print(f"\n{'='*60}")
    print(f"Running {model_name} ({label} with epoch eval)")
    print(f"{'='*60}")
    print(f"  Device: {getattr(model, 'device', 'cpu')}")
    print(f"  Train shape: {train_X.shape}")
    print(f"  Test shape: {test_X.shape}")
    print(f"  Epochs: {model.epochs}")
    print(f"  Eval interval: {eval_interval}")

    # ---- Setup model ----
    model.n_features = train_X.shape[1]
    model.model = model._build_model().to(model.device)

    if model.verbose:
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"  Parameters: {total_params:,}")

    # Create windows (or use pre-built segment-aware windows)
    if segment_aware:
        windows, targets = train_windows, train_targets
        print(f"  Training windows (segment-aware): {len(windows):,}")
    else:
        stride = getattr(model, 'train_stride', 1)
        windows, targets = model._create_windows(train_X, stride=stride)
        print(f"  Training windows: {len(windows):,} (stride={stride})")

    dataset = TensorDataset(torch.FloatTensor(windows), torch.FloatTensor(targets))
    dataloader = DataLoader(dataset, batch_size=model.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=model.lr)
    criterion = nn.MSELoss()

    # ---- Epoch eval state ----
    epoch_metrics_list = []
    model.train_loss_history = []
    best_pak_f1 = 0.0
    train_start = time.time()

    # ---- Training loop ----
    for epoch in range(model.epochs):
        ep = epoch + 1
        epoch_start = time.time()
        epoch_losses = []

        # GPU training
        model.model.train()
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

        # Should we evaluate this epoch?
        should_eval = (ep % eval_interval == 0) or (ep == model.epochs)

        if should_eval:
            # GPU inference (synchronous) — boundary-safe per-entity windowing
            # via test_segments iff the wrapper accepts it (single-arg wrappers
            # fall through to the legacy whole-array predict).
            scores = _predict_gated(model, test_X, test_segments)

            # CPU eval (synchronous — avoids ThreadPool deadlock with CUDA)
            try:
                eval_start = time.time()
                ep_metrics = compute_all_metrics(scores, test_y, anomaly_regions)
                ep_metrics['_eval_time'] = time.time() - eval_start
                ep_metrics['epoch'] = ep

                if excl_region is not None:
                    excl = compute_all_metrics_with_excl(scores, test_y, anomaly_regions, excl_region)
                    ep_metrics.update(excl)

                save_epoch_scores_npz(epoch_scores_dir, ep, scores)
                epoch_metrics_list.append(ep_metrics)

                pak_f1 = ep_metrics.get('pak_auc_f1', 0)
                prc = ep_metrics.get('prc_auc', 0)
                best_marker = ""
                if pak_f1 > best_pak_f1:
                    best_pak_f1 = pak_f1
                    best_marker = " ★"
                print(f"  [Epoch {ep:>2}] PRC={prc:.4f} PAK_F1={pak_f1:.4f} "
                      f"(eval={ep_metrics.get('_eval_time', 0):.1f}s){best_marker}")
            except Exception as e:
                print(f"\n  [Epoch {ep}] EVAL ERROR: {e}")

        elapsed = time.time() - train_start
        remaining = elapsed / ep * (model.epochs - ep)
        print(f"\r  [{ep}/{model.epochs}] Loss: {avg_loss:.6f} | "
              f"{epoch_time:.1f}s/ep | ETA: {remaining:.0f}s", end="")
        sys.stdout.flush()

    train_time = time.time() - train_start
    print(f"\n  Training complete: {train_time:.1f}s")

    # Final scores and metrics
    # scores.npz must correspond to the BEST epoch (by pak_auc_f1) so that any
    # downstream visualization or recomputation matches the reported best metric.
    if epoch_metrics_list:
        best_idx = max(
            range(len(epoch_metrics_list)),
            key=lambda i: epoch_metrics_list[i].get('pak_auc_f1', 0) or 0,
        )
        final_metrics = epoch_metrics_list[best_idx]
        best_ep_num = final_metrics.get('epoch', best_idx + 1)
        best_scores_file = output_dir / 'epoch_scores' / f'epoch_{best_ep_num:03d}_scores.npz'
        if best_scores_file.exists():
            final_scores = np.load(best_scores_file)['anomaly_score']
        else:
            # eval_interval>1 or save failed — fall back to last-epoch predict
            final_scores = _predict_gated(model, test_X, test_segments)
    else:
        # No per-epoch eval done — compute metrics on last-epoch predict
        final_scores = _predict_gated(model, test_X, test_segments)
        final_metrics = compute_all_metrics(final_scores, test_y, anomaly_regions)
        final_metrics['epoch'] = model.epochs
        epoch_metrics_list = [final_metrics]

    final_metrics['_inference_time'] = 0  # already timed above

    # Save
    save_scores_npz(output_dir, final_scores)
    save_epoch_metrics(output_dir, epoch_metrics_list, eval_interval=eval_interval)
    save_metadata(output_dir, model_name, experiment_name, train_time,
                  final_metrics.get('_inference_time', 0))

    if hasattr(model, 'save'):
        try:
            model.save(output_dir / "model")
        except Exception as e:
            print(f"  Warning: Could not save model: {e}")

    _print_metrics(final_metrics, excl_region is not None)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_metrics


# ============================================================
# Per-source-FILE NORM plumbing helpers (leak-free per-file norm)
# ============================================================

def _predict_gated(model_instance, test_X, test_segments):
    """Call ``predict`` forwarding per-file ``test_segments`` ONLY if the wrapper
    accepts it. Uses ``inspect.signature`` (NOT a bare try/except) so a real
    TypeError raised inside ``predict`` is never masked.

    All 21 windowing wrappers (incl. the 6 RevIN SOTA timesnet/tfmae/memto/
    moderntcn/dcdetector/catch) now accept ``test_segments`` and use it for BOTH
    leak-free per-entity normalization AND boundary-safe per-entity TEST windowing.
    The stateless/simple wrappers (random/sensor_range/l2_norm/nn_distance/
    pca_error) keep a single-arg ``predict(test_X)`` -> the gate falls through to
    the legacy call, leaving them bit-for-bit unchanged.
    """
    try:
        params = inspect.signature(model_instance.predict).parameters
    except (TypeError, ValueError):
        params = {}
    if 'test_segments' in params:
        return model_instance.predict(test_X, test_segments=test_segments)
    return model_instance.predict(test_X)


def _fit_kwargs_with_norm(model, base_kwargs, norm_train_segments):
    """Augment ``model.fit`` kwargs with ``norm_train_segs=`` ONLY if the
    wrapper's ``fit`` declares that parameter (sibling-gated via inspect).
    Returns a NEW dict; ``base_kwargs`` is not mutated.
    """
    kwargs = dict(base_kwargs)
    try:
        fit_params = inspect.signature(model.fit).parameters
    except (TypeError, ValueError):
        fit_params = {}
    if 'norm_train_segs' in fit_params:
        kwargs['norm_train_segs'] = norm_train_segments
    return kwargs


def _fit_kwargs_with_test(model, base_kwargs, test_X, test_segments):
    """Augment ``model.fit`` kwargs with ``test_X=`` / ``test_segments=`` ONLY if
    the wrapper's ``fit`` declares those params (sibling-gated via inspect).

    Supports a TRANSDUCTIVE graph build (nrdetector PU-LP #1=B: upstream builds the
    cosine-kNN A + Katz W over train+valid+TEST embeddings — data_loader.py:106,135;
    solver.py:43,49 — while RP/RN SELECTION stays train-only, enforced in the wrapper).
    Label-free: test LABELS are never passed. Returns a NEW dict; wrappers without
    these params (all other baselines) are untouched.
    """
    kwargs = dict(base_kwargs)
    try:
        fit_params = inspect.signature(model.fit).parameters
    except (TypeError, ValueError):
        fit_params = {}
    if 'test_X' in fit_params:
        kwargs['test_X'] = test_X
    if 'test_segments' in fit_params:
        kwargs['test_segments'] = test_segments
    return kwargs


# ============================================================
# SOTA Baseline Execution (with per-epoch eval via callback)
# ============================================================

def run_sota_baseline_with_epoch_eval(
    model,
    train_X: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    anomaly_regions: list,
    model_name: str,
    output_dir: Path,
    experiment_name: str,
    excl_region=None,
    eval_interval: int = 1,
    max_eval_workers: int = 10,
    train_segments=None,
    norm_train_segments=None,
    test_segments=None,
) -> dict:
    """Run a SOTA baseline with per-epoch inference + synchronous CPU eval.

    Uses the model's epoch_callback mechanism: model.fit() calls
    the callback after each epoch, which does predict + eval synchronously.

    Args:
        eval_interval: Evaluate every N epochs (default: every epoch)
        max_eval_workers: unused (kept for API compatibility)
        train_segments: Optional list of ``(start, end)`` tuples (end exclusive)
            in ``train_X`` coordinate. When provided (normalonly variant), the
            wrapper drops every sliding window that would span a segment
            boundary. ``None`` keeps the legacy concat-array behaviour.
    """
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_scores_dir = output_dir / 'epoch_scores'
    epoch_scores_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running {model_name} (SOTA with epoch eval)")
    print(f"{'='*60}")
    print(f"  Device: {getattr(model, 'device', 'cpu')}")
    print(f"  Train shape: {train_X.shape}")
    print(f"  Test shape: {test_X.shape}")
    print(f"  Epochs: {getattr(model, 'epochs', '?')}")
    print(f"  Eval interval: {eval_interval}")

    # ---- Epoch eval state ----
    epoch_metrics_list = []
    best_pak_f1 = 0.0

    def epoch_callback(model_instance, ep):
        """Called by model.fit() after each epoch. Synchronous eval."""
        nonlocal best_pak_f1

        # Should we evaluate this epoch?
        should_eval = (ep % eval_interval == 0) or (ep == model_instance.epochs)
        if not should_eval:
            return

        # Inference (forward per-file test_segments iff the wrapper accepts it)
        scores = _predict_gated(model_instance, test_X, test_segments)

        # CPU eval (synchronous — avoids ThreadPool deadlock with CUDA)
        try:
            eval_start = time.time()
            ep_metrics = compute_all_metrics(scores, test_y, anomaly_regions)
            ep_metrics['_eval_time'] = time.time() - eval_start
            ep_metrics['epoch'] = ep

            if excl_region is not None:
                excl = compute_all_metrics_with_excl(scores, test_y, anomaly_regions, excl_region)
                ep_metrics.update(excl)

            save_epoch_scores_npz(epoch_scores_dir, ep, scores)
            epoch_metrics_list.append(ep_metrics)

            pak_f1 = ep_metrics.get('pak_auc_f1', 0)
            prc = ep_metrics.get('prc_auc', 0)
            best_marker = ""
            if pak_f1 > best_pak_f1:
                best_pak_f1 = pak_f1
                best_marker = " ★"
            print(f"\n  [Epoch {ep:>2}] PRC={prc:.4f} PAK_F1={pak_f1:.4f} "
                  f"(eval={ep_metrics.get('_eval_time', 0):.1f}s){best_marker}")
        except Exception as e:
            print(f"\n  [Epoch {ep}] EVAL ERROR: {e}")

    # ---- Train with callback ----
    start_time = time.time()
    if train_segments is not None:
        # normalonly: wrapper opts into segment-safe windowing.
        # All 14 SOTA wrappers accept `train_segments=...` and filter windows that
        # cross a boundary. (DAGMM was previously row-by-row and ignored this; the
        # TranAD-author DAGMM impl now uses 5-step windows and respects segments.)
        _fit_kw = {'epoch_callback': epoch_callback, 'train_segments': train_segments}
    else:
        _fit_kw = {'epoch_callback': epoch_callback}
    # SEPARATE per-file NORM segments (window-safety train_segments is unchanged):
    # forwarded as `norm_train_segs=` only to wrappers whose fit declares it.
    _fit_kw = _fit_kwargs_with_norm(model, _fit_kw, norm_train_segments)
    model.fit(train_X, **_fit_kw)
    train_time = time.time() - start_time

    print(f"\n  Training complete: {train_time:.1f}s")

    # Final scores and metrics
    # scores.npz must correspond to the BEST epoch (by pak_auc_f1) so that any
    # downstream visualization or recomputation matches the reported best metric.
    if epoch_metrics_list:
        best_idx = max(
            range(len(epoch_metrics_list)),
            key=lambda i: epoch_metrics_list[i].get('pak_auc_f1', 0) or 0,
        )
        final_metrics = epoch_metrics_list[best_idx]
        best_ep_num = final_metrics.get('epoch', best_idx + 1)
        best_scores_file = output_dir / 'epoch_scores' / f'epoch_{best_ep_num:03d}_scores.npz'
        if best_scores_file.exists():
            final_scores = np.load(best_scores_file)['anomaly_score']
        else:
            # eval_interval>1 or save failed — fall back to last-epoch predict
            final_scores = _predict_gated(model, test_X, test_segments)
    else:
        # No per-epoch eval done — compute metrics on last-epoch predict
        final_scores = _predict_gated(model, test_X, test_segments)
        final_metrics = compute_all_metrics(final_scores, test_y, anomaly_regions)
        final_metrics['epoch'] = getattr(model, 'epochs', 1)
        if excl_region is not None:
            excl = compute_all_metrics_with_excl(final_scores, test_y, anomaly_regions, excl_region)
            final_metrics.update(excl)
        epoch_metrics_list = [final_metrics]

    final_metrics['_inference_time'] = 0

    # Save
    save_scores_npz(output_dir, final_scores)
    save_epoch_metrics(output_dir, epoch_metrics_list, eval_interval=eval_interval)
    save_metadata(output_dir, model_name, experiment_name, train_time,
                  final_metrics.get('_inference_time', 0))

    if hasattr(model, 'save'):
        try:
            model.save(output_dir / "model")
        except Exception as e:
            print(f"  Warning: Could not save model: {e}")

    _print_metrics(final_metrics, excl_region is not None)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_metrics


def run_weak_sota_baseline_with_epoch_eval(
    model,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    anomaly_regions: list,
    model_name: str,
    output_dir: Path,
    experiment_name: str,
    excl_region=None,
    eval_interval: int = 1,
    max_eval_workers: int = 10,
    train_segments=None,
    norm_train_segments=None,
    test_segments=None,
    resume: bool = False,
) -> dict:
    """Weakly-supervised SOTA execution path (2026-05-29/30 SSL porting).

    Byte-for-byte identical to ``run_sota_baseline_with_epoch_eval`` EXCEPT it
    forwards ``train_y`` (point labels for the train split, from
    ``loader.get_train_data()[1]``) into ``model.fit(..., train_y=train_y)``.
    The wrapper derives its own weak window/bag labels (``max(train_y over
    window)``) internally — leak-free, train-split-only.

    Kept as a SEPARATE function so the 22 unsupervised baselines' SOTA path
    (``run_sota_baseline_with_epoch_eval``) is NOT modified. Per-epoch eval,
    best-epoch-by-``pak_auc_f1`` selection, ``scores.npz`` (= best epoch),
    ``epoch_metrics.json``, ``metadata.json`` and viz all reuse the SAME logic
    and the SAME (frozen) metric layer as every other model — so results are
    directly comparable. Q1-only: the wrapper raises RuntimeError under Q3.
    """
    import json
    import shutil
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_scores_dir = output_dir / 'epoch_scores'
    epoch_scores_dir.mkdir(parents=True, exist_ok=True)
    # Checkpoint directory (resume infrastructure, 2026-05-31). Wrappers write
    # ``last.pt`` here at the end of every visible epoch; we mirror it to
    # ``best.pt`` whenever pak_auc_f1 improves. Pre-create unconditionally so
    # the directory exists for the first save without an extra mkdir round-trip.
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running {model_name} (WEAK-SUPERVISED with epoch eval)")
    print(f"{'='*60}")
    print(f"  Device: {getattr(model, 'device', 'cpu')}")
    print(f"  Train shape: {train_X.shape}  (train_y anomaly ratio: {float(np.mean(train_y)) if train_y is not None and len(train_y) else 0.0:.4%})")
    print(f"  Test shape: {test_X.shape}")
    print(f"  Epochs: {getattr(model, 'epochs', '?')}")
    print(f"  Eval interval: {eval_interval}")
    print(f"  Resume: {resume}")

    # --- Resume bookkeeping ---------------------------------------------------
    # We need to (a) seed `epoch_metrics_list` with all previously-saved epochs
    # so the appended json stays a continuous 1..target range, and (b) re-derive
    # `best_pak_f1` from those entries so the in-loop "is this a new best?"
    # check is correct on resume. The wrapper handles model/optimizer/RNG
    # restore via set_resume()/_maybe_resume(); this block only handles the
    # eval-side state that lives outside the wrapper.
    epoch_metrics_list: list = []
    best_pak_f1 = 0.0
    resume_start_epoch = 0
    metrics_file = output_dir / 'epoch_metrics.json'
    last_ckpt = ckpt_dir / 'last.pt'
    if resume and metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                existing = json.load(f)
            epoch_metrics_list = list(existing.get('epochs', []))
            # Reconcile json ↔ last.pt: if json is AHEAD of last.pt (json had
            # ep K logged but the wrapper crashed before saving last.pt for
            # ep K), drop the ghost entries so the appended training starts
            # from the SAME epoch the wrapper resumes from — otherwise the
            # final json would have a "double ep K" or skipped epoch. If json
            # is BEHIND last.pt (json save was killed mid-write before this
            # commit added atomic write; or older runs), warn explicitly so
            # the user knows ep N→last.pt's epoch are forever missing.
            ck_epoch = 0
            if last_ckpt.exists():
                try:
                    import torch as _torch
                    ck_epoch = int(_torch.load(last_ckpt, map_location='cpu').get('epoch', 0))
                except Exception as _ck_e:
                    print(f"  [RESUME] could not read last.pt for reconciliation: {_ck_e}")
            json_max_epoch = max(((e.get('epoch') or 0) for e in epoch_metrics_list), default=0)
            if ck_epoch and json_max_epoch > ck_epoch:
                # JSON has phantom entries beyond ck_epoch → drop them.
                dropped = [e for e in epoch_metrics_list if (e.get('epoch') or 0) > ck_epoch]
                epoch_metrics_list = [e for e in epoch_metrics_list if (e.get('epoch') or 0) <= ck_epoch]
                print(f"  [RESUME] dropped {len(dropped)} ghost epoch entries beyond last.pt epoch={ck_epoch}")
            elif ck_epoch and json_max_epoch < ck_epoch:
                # last.pt has trained past json's last entry → ep (json_max+1)..ck_epoch
                # have NO metrics saved. They cannot be recovered (eval would
                # need predictions from those specific snapshots) — warn loudly.
                missing = list(range(json_max_epoch + 1, ck_epoch + 1))
                print(f"  [RESUME] WARNING: epochs {missing[0]}..{missing[-1]} have no eval metrics "
                      f"in json (likely a pre-atomic-write crash). These epochs will be missing "
                      f"from the final epoch_metrics.json. Training continues from ep {ck_epoch + 1}.")
            if epoch_metrics_list:
                best_pak_f1 = max(
                    (e.get('pak_auc_f1', 0) or 0) for e in epoch_metrics_list
                )
                resume_start_epoch = max(
                    (e.get('epoch', 0) or 0) for e in epoch_metrics_list
                )
                print(f"  [RESUME] loaded {len(epoch_metrics_list)} prior epochs from {metrics_file.name}; "
                      f"best_pak_f1={best_pak_f1:.4f} at epoch≤{resume_start_epoch}")
        except Exception as e:
            print(f"  [RESUME] could not parse {metrics_file}: {e} — treating as fresh run")
            epoch_metrics_list = []
            best_pak_f1 = 0.0
            resume_start_epoch = 0
    # Hand the wrapper the resume + per-epoch save hooks. ``set_resume`` is a
    # no-op when ``resume=False`` (we still need to call ``set_checkpoint_dir``
    # so the wrapper writes last.pt going forward).
    if resume:
        if last_ckpt.exists() and hasattr(model, 'set_resume'):
            model.set_resume(last_ckpt)
    if hasattr(model, 'set_checkpoint_dir'):
        model.set_checkpoint_dir(ckpt_dir)

    def epoch_callback(model_instance, ep):
        """Called by model.fit() after each epoch. Synchronous eval (same as SOTA path)."""
        nonlocal best_pak_f1
        should_eval = (ep % eval_interval == 0) or (ep == model_instance.epochs)
        if not should_eval:
            return
        scores = _predict_gated(model_instance, test_X, test_segments)
        try:
            eval_start = time.time()
            ep_metrics = compute_all_metrics(scores, test_y, anomaly_regions)
            ep_metrics['_eval_time'] = time.time() - eval_start
            ep_metrics['epoch'] = ep
            if excl_region is not None:
                excl = compute_all_metrics_with_excl(scores, test_y, anomaly_regions, excl_region)
                ep_metrics.update(excl)
            save_epoch_scores_npz(epoch_scores_dir, ep, scores)
            epoch_metrics_list.append(ep_metrics)
            # Persist epoch_metrics.json incrementally so an interrupted run
            # is resumable from THIS epoch (otherwise the last 50 ep of data
            # would be lost on crash because of the end-of-fit() one-shot save).
            save_epoch_metrics(output_dir, epoch_metrics_list, eval_interval=eval_interval)
            pak_f1 = ep_metrics.get('pak_auc_f1', 0)
            prc = ep_metrics.get('prc_auc', 0)
            best_marker = ""
            if pak_f1 > best_pak_f1:
                best_pak_f1 = pak_f1
                best_marker = " ★"
                # Mirror this epoch's wrapper-saved last.pt → best.pt. The
                # wrapper writes last.pt BEFORE epoch_callback (see each
                # wrapper's fit() loop) so the file is guaranteed to be on
                # disk with the just-finished epoch's state.
                last_ckpt = ckpt_dir / 'last.pt'
                if last_ckpt.exists():
                    try:
                        shutil.copy2(last_ckpt, ckpt_dir / 'best.pt')
                    except Exception as cp_e:
                        print(f"  [WARN] best.pt copy failed: {cp_e}")
            print(f"\n  [Epoch {ep:>2}] PRC={prc:.4f} PAK_F1={pak_f1:.4f} "
                  f"(eval={ep_metrics.get('_eval_time', 0):.1f}s){best_marker}")
        except Exception as e:
            print(f"\n  [Epoch {ep}] EVAL ERROR: {e}")

    # ---- Train with callback — ONLY behavioral delta vs SOTA path: forward train_y ----
    start_time = time.time()
    if train_segments is not None:
        _fit_kw = {'train_y': train_y, 'epoch_callback': epoch_callback,
                   'train_segments': train_segments}
    else:
        _fit_kw = {'train_y': train_y, 'epoch_callback': epoch_callback}
    # SEPARATE per-file NORM segments (window-safety train_segments is unchanged):
    # forwarded as `norm_train_segs=` only to wrappers whose fit declares it.
    _fit_kw = _fit_kwargs_with_norm(model, _fit_kw, norm_train_segments)
    # TRANSDUCTIVE graph support (nrdetector #1=B): forward TEST features (label-free)
    # + per-entity test_segments ONLY to wrappers whose fit declares them (inspect-gated).
    _fit_kw = _fit_kwargs_with_test(model, _fit_kw, test_X, test_segments)
    model.fit(train_X, **_fit_kw)
    train_time = time.time() - start_time

    print(f"\n  Training complete: {train_time:.1f}s")

    # Final scores = BEST epoch (by pak_auc_f1) — identical to SOTA path.
    if epoch_metrics_list:
        best_idx = max(
            range(len(epoch_metrics_list)),
            key=lambda i: epoch_metrics_list[i].get('pak_auc_f1', 0) or 0,
        )
        final_metrics = epoch_metrics_list[best_idx]
        best_ep_num = final_metrics.get('epoch', best_idx + 1)
        best_scores_file = output_dir / 'epoch_scores' / f'epoch_{best_ep_num:03d}_scores.npz'
        if best_scores_file.exists():
            final_scores = np.load(best_scores_file)['anomaly_score']
        else:
            final_scores = _predict_gated(model, test_X, test_segments)
    else:
        final_scores = _predict_gated(model, test_X, test_segments)
        final_metrics = compute_all_metrics(final_scores, test_y, anomaly_regions)
        final_metrics['epoch'] = getattr(model, 'epochs', 1)
        if excl_region is not None:
            excl = compute_all_metrics_with_excl(final_scores, test_y, anomaly_regions, excl_region)
            final_metrics.update(excl)
        epoch_metrics_list = [final_metrics]

    final_metrics['_inference_time'] = 0

    save_scores_npz(output_dir, final_scores)
    save_epoch_metrics(output_dir, epoch_metrics_list, eval_interval=eval_interval)
    save_metadata(output_dir, model_name, experiment_name, train_time,
                  final_metrics.get('_inference_time', 0))

    if hasattr(model, 'save'):
        try:
            model.save(output_dir / "model")
        except Exception as e:
            print(f"  Warning: Could not save model: {e}")

    _print_metrics(final_metrics, excl_region is not None)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_metrics


# ============================================================
# Segment-Aware Training (normalonly variant)
# ============================================================

def run_segment_aware_dl_baseline(
    model,
    loader,
    test_X: np.ndarray,
    test_y: np.ndarray,
    anomaly_regions: list,
    model_name: str,
    output_dir: Path,
    experiment_name: str,
    seq_len: int = 5,
    excl_region=None,
) -> dict:
    """Run DL baseline with segment-aware training (normalonly variant).

    Creates windows from normal segments only, ensuring no window spans
    deletion boundaries.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running {model_name} (segment-aware)")
    print(f"{'='*60}")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    windows, targets = loader.create_windows_from_segments(seq_len=seq_len)
    print(f"  Training windows (boundary-safe): {len(windows):,}")

    model.n_features = loader.n_features
    model.model = model._build_model().to(model.device)

    dataset = TensorDataset(torch.FloatTensor(windows), torch.FloatTensor(targets))
    dataloader = DataLoader(dataset, batch_size=model.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=model.lr)
    criterion = nn.MSELoss()
    model.train_loss_history = []
    model.model.train()

    start_time = time.time()
    for epoch in range(model.epochs):
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
        if model.verbose:
            elapsed = time.time() - start_time
            remaining = elapsed / (epoch + 1) * (model.epochs - epoch - 1)
            print(f"\r  [{epoch+1}/{model.epochs}] Loss: {avg_loss:.6f} | ETA: {remaining:.0f}s", end="")
            sys.stdout.flush()

    train_time = time.time() - start_time
    if model.verbose:
        print(f"\n  Training complete: {train_time:.1f}s")

    # Inference & eval
    infer_start = time.time()
    scores = model.predict(test_X)
    inference_time = time.time() - infer_start

    eval_start = time.time()
    metrics = compute_all_metrics(scores, test_y, anomaly_regions)
    metrics['_inference_time'] = inference_time
    metrics['_eval_time'] = time.time() - eval_start
    metrics['epoch'] = model.epochs

    if excl_region is not None:
        excl = compute_all_metrics_with_excl(scores, test_y, anomaly_regions, excl_region)
        metrics.update(excl)

    save_scores_npz(output_dir, scores)
    save_epoch_metrics(output_dir, [metrics])
    save_metadata(output_dir, model_name, experiment_name, train_time, inference_time,
                  extra={"training_note": "segment-aware windowing (no boundary crossing)"})

    if hasattr(model, 'save'):
        try:
            model.save(output_dir / "model")
        except Exception as e:
            print(f"  Warning: Could not save model: {e}")

    _print_metrics(metrics, excl_region is not None)
    return metrics


# ============================================================
# Helper: Print Metrics
# ============================================================

def _print_metrics(metrics: dict, has_excl: bool = False):
    """Print key metrics."""
    prefix = "[Full] " if has_excl else ""
    print(f"\n  {prefix}ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
    print(f"  {prefix}PRC-AUC: {metrics.get('prc_auc', 0):.4f}")
    print(f"  {prefix}F1: {metrics.get('f1_score', 0):.4f}")
    print(f"  {prefix}F1_T: {metrics.get('f1_t', 0):.4f}")
    print(f"  {prefix}PAK_F1: {metrics.get('pak_auc_f1', 0):.4f}")
    print(f"  {prefix}PAK_PRC: {metrics.get('pak_auc_prc_auc', 0):.4f}")

    if has_excl and 'excl22_pak_auc_f1' in metrics:
        print(f"\n  [Excl22] PRC-AUC: {metrics.get('excl22_prc_auc', 0):.4f}")
        print(f"  [Excl22] F1_T: {metrics.get('excl22_f1_t', 0):.4f}")
        print(f"  [Excl22] PAK_F1: {metrics.get('excl22_pak_auc_f1', 0):.4f}")
        print(f"  [Excl22] PAK_PRC: {metrics.get('excl22_pak_auc_prc_auc', 0):.4f}")


# ============================================================
# Status Display
# ============================================================

def print_status(results_dir: Path, all_models: list, experiment_name: str = ""):
    """Print status table showing completed/pending models."""
    print(f"\n{'='*120}")
    print(f"STATUS — {experiment_name}")
    print(f"{'='*120}")

    header = (f"{'#':<3} {'Model':<22} {'Status':<8} {'PRC':>7} {'F1_T':>7} "
              f"{'PAK_PRC':>8} {'PAK_F1':>7} {'PA20_F1':>8} {'PA80_F1':>8}")
    print(header)
    print("-" * 120)

    for i, model_name in enumerate(all_models, 1):
        model_dir = results_dir / model_name
        metrics_file = model_dir / 'epoch_metrics.json'

        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            epochs = data.get('epochs', [])
            if epochs:
                m = epochs[-1]  # Last epoch metrics
                prc = m.get('prc_auc', 0)
                f1t = m.get('f1_t', 0)
                pak_prc = m.get('pak_auc_prc_auc', 0)
                pak_f1 = m.get('pak_auc_f1', 0)
                pa20 = m.get('pa_20_f1', 0)
                pa80 = m.get('pa_80_f1', 0)
                print(f"{i:<3} {model_name:<22} {'OK':<8} {prc:>7.3f} {f1t:>7.3f} "
                      f"{pak_prc:>8.3f} {pak_f1:>7.3f} {pa20:>8.3f} {pa80:>8.3f}")
            else:
                print(f"{i:<3} {model_name:<22} {'EMPTY':<8}")
        else:
            print(f"{i:<3} {model_name:<22} {'-':<8}")

    print("-" * 120)

    completed = sum(1 for m in all_models if (results_dir / m / 'epoch_metrics.json').exists())
    print(f"  {completed}/{len(all_models)} completed")


def print_summary_sorted(results_dir: Path, all_models: list, excl22: bool = False):
    """Print final summary sorted by PAK_PRC descending."""
    model_metrics = {}
    for model_name in all_models:
        metrics_file = results_dir / model_name / 'epoch_metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            epochs = data.get('epochs', [])
            if epochs:
                model_metrics[model_name] = epochs[-1]

    if not model_metrics:
        return

    print(f"\n{'Model':<25} {'PRC':>8} {'F1_T':>8} {'PAK_PRC':>9} {'PAK_F1':>8}")
    print("-" * 65)

    sorted_models = sorted(model_metrics.items(),
                          key=lambda x: x[1].get('pak_auc_prc_auc', 0),
                          reverse=True)

    for model_name, m in sorted_models:
        print(f"{model_name:<25} {m.get('prc_auc', 0):>8.4f} {m.get('f1_t', 0):>8.4f} "
              f"{m.get('pak_auc_prc_auc', 0):>9.4f} {m.get('pak_auc_f1', 0):>8.4f}")

    print("-" * 65)

    if excl22:
        has_excl = any('excl22_pak_auc_f1' in m for m in model_metrics.values())
        if has_excl:
            print(f"\n  Excl22:")
            for model_name, m in sorted_models:
                if 'excl22_pak_auc_f1' in m:
                    print(f"  {model_name:<25} PRC={m.get('excl22_prc_auc', 0):.4f} "
                          f"PAK_PRC={m.get('excl22_pak_auc_prc_auc', 0):.4f} "
                          f"PAK_F1={m.get('excl22_pak_auc_f1', 0):.4f}")


# ============================================================
# Model Factory
# ============================================================

def create_model(
    model_name: str,
    n_features: int,
    model_preset: str = 'general',
    epoch_overrides: dict = None,
    train_stride: int = None,
    nn_subsample: int = None,
):
    """Create a baseline model instance with the correct hyperparameters."""
    params_fn = MODEL_PRESETS.get(model_preset, _get_default_model_params)
    all_params = params_fn()
    params = dict(all_params.get(model_name, {}))

    epoch_overrides = epoch_overrides or {}
    neural_epochs = epoch_overrides.get('neural_epochs')
    sota_epochs = epoch_overrides.get('sota_epochs')
    at_epochs = epoch_overrides.get('at_epochs')

    if train_stride is not None and model_name not in SIMPLE_MODELS:
        params['train_stride'] = train_stride

    if model_name == 'nn_distance' and nn_subsample is not None:
        params['subsample'] = nn_subsample

    # Simple baselines
    if model_name == 'random':
        return RandomBaseline(**params)
    elif model_name == 'sensor_range':
        return SensorRangeDeviation(**params)
    elif model_name == 'pca_error':
        return PCAError(**params)
    elif model_name == 'l2_norm':
        return L2Norm(**params)
    elif model_name == 'nn_distance':
        return NNDistance(**params)

    # Neural baselines
    elif model_name == 'mlp':
        if neural_epochs is not None:
            params['epochs'] = neural_epochs
        params['verbose'] = True
        return MLPBaseline(**params)
    elif model_name == 'mlpmixer':
        if neural_epochs is not None:
            params['epochs'] = neural_epochs
        params['verbose'] = True
        return MLPMixerBaseline(**params)
    elif model_name == 'transformer':
        if neural_epochs is not None:
            params['epochs'] = neural_epochs
        params['verbose'] = True
        return TransformerBaseline(**params)
    elif model_name == 'gcn_lstm':
        if neural_epochs is not None:
            params['epochs'] = neural_epochs
        params['verbose'] = True
        return GCNLSTMBaseline(**params)

    # SOTA models
    elif model_name == 'anomaly_transformer':
        if not SOTA_AVAILABILITY.get('anomaly_transformer', False):
            raise ValueError("AnomalyTransformerBaseline not available")
        if at_epochs is not None:
            params['epochs'] = at_epochs
        params['verbose'] = True
        return AnomalyTransformerBaseline(**params)
    elif model_name == 'tranad':
        if not HAS_TRANAD:
            raise ValueError("TranADBaseline not available")
        if sota_epochs is not None:
            params['epochs'] = sota_epochs
        params['verbose'] = True
        return TranADBaseline(**params)
    elif model_name == 'usad':
        if not HAS_USAD:
            raise ValueError("USADBaseline not available")
        if sota_epochs is not None:
            params['epochs'] = sota_epochs
        params['verbose'] = True
        return USADBaseline(**params)
    elif model_name == 'dagmm':
        if not HAS_DAGMM:
            raise ValueError("DAGMMBaseline not available")
        if sota_epochs is not None:
            params['epochs'] = sota_epochs
        params['verbose'] = True
        return DAGMMBaseline(**params)
    elif model_name == 'gdn':
        if not HAS_GDN:
            raise ValueError("GDNBaseline not available")
        if sota_epochs is not None:
            params['epochs'] = sota_epochs
        if 'top_k' not in params:
            params['top_k'] = min(5, n_features - 1)
        params['verbose'] = True
        return GDNBaseline(**params)
    elif model_name == 'omnianomaly':
        if not HAS_OMNIANOMALY:
            raise ValueError("OmniAnomalyBaseline not available")
        if sota_epochs is not None:
            params['epochs'] = sota_epochs
        params['verbose'] = True
        return OmniAnomalyBaseline(**params)

    # === 2026-05-19 batch (10 SOTA additions) ===
    elif model_name == 'tfmae':
        if not HAS_TFMAE:
            raise ValueError("TFMAEBaseline not available")
        if sota_epochs is not None:
            params['epochs'] = sota_epochs
        params['verbose'] = True
        return TFMAEBaseline(**params)
    elif model_name == 'timesnet':
        if not HAS_TIMESNET:
            raise ValueError("TimesNetBaseline not available")
        if sota_epochs is not None:
            params['epochs'] = sota_epochs
        params['verbose'] = True
        return TimesNetBaseline(**params)
    elif model_name == 'dcdetector':
        if not HAS_DCDETECTOR:
            raise ValueError("DCdetectorBaseline not available")
        if sota_epochs is not None:
            params['epochs'] = sota_epochs
        params['verbose'] = True
        return DCdetectorBaseline(**params)
    elif model_name == 'memto':
        if not HAS_MEMTO:
            raise ValueError("MEMTOBaseline not available")
        if sota_epochs is not None:
            params['epochs'] = sota_epochs
        params['verbose'] = True
        return MEMTOBaseline(**params)
    elif model_name == 'moderntcn':
        if not HAS_MODERNTCN:
            raise ValueError("ModernTCNBaseline not available")
        if sota_epochs is not None:
            params['epochs'] = sota_epochs
        params['verbose'] = True
        return ModernTCNBaseline(**params)
    elif model_name == 'catch':
        if not HAS_CATCH:
            raise ValueError("CATCHBaseline not available")
        if sota_epochs is not None:
            params['epochs'] = sota_epochs
        params['verbose'] = True
        return CATCHBaseline(**params)
    elif model_name == 'npsr':
        if not HAS_NPSR:
            raise ValueError("NPSRBaseline not available")
        if sota_epochs is not None:
            params['epochs'] = sota_epochs
        params['verbose'] = True
        return NPSRBaseline(**params)

    # === Weakly-supervised models (Q1-only) — instantiated like SOTA; train_y is
    # supplied later via run_weak_sota_baseline_with_epoch_eval. n_features is
    # re-derived from train_X inside each wrapper's fit(), so it is not passed here. ===
    elif model_name == 'wetas':
        if not HAS_WETAS:
            raise ValueError("WETASBaseline not available")
        if sota_epochs is not None:
            params['epochs'] = sota_epochs
        params['verbose'] = True
        return WETASBaseline(**params)
    elif model_name == 'treemil':
        if not HAS_TREEMIL:
            raise ValueError("TreeMILBaseline not available")
        if sota_epochs is not None:
            params['epochs'] = sota_epochs
        params['verbose'] = True
        return TreeMILBaseline(**params)
    elif model_name in ('nrdetector', 'nrdetector_full'):
        if not HAS_NRDETECTOR:
            raise ValueError("NRdetectorBaseline not available")
        if sota_epochs is not None:
            params['epochs'] = sota_epochs
        params['verbose'] = True
        return NRdetectorBaseline(**params)
    elif model_name == 'deepmil':
        if not HAS_DEEPMIL:
            raise ValueError("DeepMILBaseline not available")
        if sota_epochs is not None:
            params['epochs'] = sota_epochs
        params['verbose'] = True
        return DeepMILBaseline(**params)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def is_model_available(model_name: str) -> bool:
    """Check if a model is available."""
    if model_name in SIMPLE_MODELS or model_name in NEURAL_MODELS:
        return True
    return SOTA_AVAILABILITY.get(model_name, False)


def filter_models(
    models: list,
    only_simple: bool = False,
    only_neural: bool = False,
    skip_at: bool = False,
    skip_sota: bool = False,
) -> list:
    """Filter model list based on CLI flags."""
    if only_simple:
        return [m for m in models if m in SIMPLE_MODELS]
    if only_neural:
        return [m for m in models if m not in SIMPLE_MODELS]

    result = list(models)
    if skip_sota:
        result = [m for m in result if m not in SOTA_MODELS]
    elif skip_at:
        result = [m for m in result if m != 'anomaly_transformer']
    return result


# ============================================================
# Data Loader Helper
# ============================================================

def load_data_from_config(config: dict, normalize_mode: str = None):
    """Instantiate UnifiedLoader from experiment config.

    Args:
        config: Experiment config dict from experiment_configs.py
        normalize_mode: Override normalization mode ('zscore' or 'minmax').
                        If None, uses 'zscore' (UnifiedLoader default).
    """
    from comparison.data.unified_loader import UnifiedLoader
    kwargs = dict(config.get('loader_kwargs', {}))
    if normalize_mode:
        kwargs['normalize_mode'] = normalize_mode
    loader = UnifiedLoader(**kwargs)
    loader.load()
    return loader


# ============================================================
# Auto-Numbered Experiment Directory
# ============================================================

BASELINE_EXPERIMENTS_DIR = PROJECT_ROOT / "comparison" / "results" / "experiments"


def make_numbered_baseline_dir(description: str = "") -> Path:
    """Create a numbered experiment directory under comparison/results/experiments/.

    Mirrors MAE's make_numbered_experiment_dir() pattern:
    {N}_{YYYYMMDD_HHMMSS}_{description}/

    Returns the created directory path.
    """
    BASELINE_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Find max existing number
    max_num = 0
    for d in BASELINE_EXPERIMENTS_DIR.iterdir():
        if d.is_dir():
            try:
                num = int(d.name.split('_')[0])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                pass

    next_num = max_num + 1
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    desc_part = f"_{description}" if description else ""
    dir_name = f"{next_num}_{timestamp}{desc_part}"

    exp_dir = BASELINE_EXPERIMENTS_DIR / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


# ============================================================
# SWaT excl22 Directory Split Post-processing
# ============================================================

def generate_excl22_directory(full_dir: Path):
    """Generate SWaT/A1A2_excl22/ from SWaT/A1A2_full/ results.

    For each model in full_dir, extracts excl22_* metrics from epoch_metrics.json,
    removes the excl22_ prefix, and saves to the parallel _excl22 directory.
    This matches MAE's dual-directory structure (A1A2_full + A1A2_excl22).

    Args:
        full_dir: Path to SWaT/A1A2_full/ directory containing model subdirs.
    """
    excl22_dir = full_dir.parent / full_dir.name.replace('_full', '_excl22')

    model_count = 0
    for model_dir in sorted(full_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        metrics_file = model_dir / 'epoch_metrics.json'
        if not metrics_file.exists():
            continue

        with open(metrics_file, 'r') as f:
            data = json.load(f)

        epochs = data.get('epochs', [])
        if not epochs or 'excl22_prc_auc' not in epochs[0]:
            continue

        # Extract excl22 metrics: remove excl22_ prefix
        excl22_epochs = []
        best_pak_f1 = -1.0
        best_epoch_num = 0
        for ep in epochs:
            excl_ep = {'epoch': ep['epoch']}
            for k, v in ep.items():
                if k.startswith('excl22_'):
                    excl_ep[k[7:]] = v  # Remove 'excl22_' prefix
            excl22_epochs.append(excl_ep)

            pak_f1 = excl_ep.get('pak_auc_f1', 0)
            if pak_f1 > best_pak_f1:
                best_pak_f1 = pak_f1
                best_epoch_num = ep['epoch']

        # Save to excl22 directory
        out_dir = excl22_dir / model_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / 'epoch_metrics.json', 'w') as f:
            json.dump({
                'eval_interval': data.get('eval_interval', 1),
                'best_epoch': best_epoch_num,
                'best_epoch_metric': 'pak_auc_f1',
                'best_epoch_score': best_pak_f1,
                'epochs': excl22_epochs,
            }, f, indent=2)

        # Copy metadata.json with excl22 note
        meta_file = model_dir / 'metadata.json'
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            meta['excl22'] = True
            meta['source_dir'] = str(full_dir.name)
            with open(out_dir / 'metadata.json', 'w') as f:
                json.dump(meta, f, indent=2)

        model_count += 1

    if model_count > 0:
        print(f"\n  [excl22] Generated {excl22_dir.name}/ with {model_count} models")
