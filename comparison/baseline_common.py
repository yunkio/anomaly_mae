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

import sys
import json
import time
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
    find_f1_optimal_idx,
    compute_f1_t_at_threshold,
    compute_pa_k_metrics_from_mean_scores,
    compute_pa_k_roc_prc_from_mean_scores,
    compute_pa_k_auc,
    find_swat_largest_region,
    compute_metrics_with_exclusion,
    # Single-source metric pipeline (2026-05-27): any metric added to MAE
    # auto-flows to comparison baselines through these imports — no duplication.
    compute_full_metric_set,
    _zero_metric_set,
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
        'random': {'seed': 42},                                          # np.random.randint(0,2)
        'sensor_range': {'sensor_range': (0, 1), 'count_sensors': False},  # eval_simple_baselines.py:36
        'pca_error': {'pca_dim': 'auto', 'svd_solver': 'full'},          # eval_simple_baselines.py:39
        'l2_norm': {'ord': 2},                                            # eval_simple_baselines.py:37 (LNorm(ord=2))
        'nn_distance': {'distance': 'euclidean'},                         # eval_simple_baselines.py:38
        # Neural baselines — paper yaml (model_configs/*.yaml) with epochs=50 user override
        # mlp_small_embedd_32_seq_5.yaml: seq_len=5, embed=32, epochs=200, lr=0.001,
        #   batch=512, dropout=0.0, weight_decay=0.0001, optimizer=adam
        'mlp': {
            'seq_len': 5, 'embedding_dim': 32, 'dropout': 0.0,
            'lr': 0.001, 'weight_decay': 0.0001,
            'train_stride': 1, 'epochs': 50, 'batch_size': 512,
        },
        # MLPmixer_blocks_1_embedd_128_seq_5.yaml: seq_len=5, embed=128, epochs=100,
        #   lr=0.0002, batch=512, dropout=0.1, num_blocks=1
        'mlpmixer': {
            'seq_len': 5, 'embedding_dim': 128, 'num_blocks': 1, 'dropout': 0.1,
            'lr': 0.0002, 'weight_decay': 0.0001,
            'train_stride': 1, 'epochs': 50, 'batch_size': 512,
        },
        # Transformer_blocks_1_1_embedd_128_seq_5.yaml: seq_len=5, embed=128, MHA=1,
        #   num_blocks=1, epochs=100, lr=0.001, batch=512, dropout=0.1
        'transformer': {
            'seq_len': 5, 'embedding_dim': 128, 'num_heads': 1, 'num_blocks': 1, 'dropout': 0.1,
            'lr': 0.001, 'weight_decay': 0.0001,
            'train_stride': 1, 'epochs': 50, 'batch_size': 512,
        },
        # gcn_lstm_model_seq_5.yaml: seq_len=5, embed=10, lstm_units=64, output_dim=1,
        #   epochs=100, lr=0.001, batch=100, dropout=0.1
        'gcn_lstm': {
            'seq_len': 5, 'gcn_out_dim': 10, 'lstm_units': 64, 'output_seq_len': 1,
            'aggregation_type': 'max', 'combination_type': 'concat',
            'graph_conv_activation': 'relu',
            'adj_symmetric': True,
            'score_norm': 'median-iqr', 'score_smooth_window': 5, 'score_iqr_epsilon': 1e-2,
            'lr': 0.001, 'weight_decay': 0.0001,
            'train_stride': 1, 'epochs': 50, 'batch_size': 100,
        },
        # SOTA baselines
        'anomaly_transformer': {'win_size': 100, 'd_model': 512, 'n_heads': 8, 'e_layers': 3, 'train_stride': 1, 'epochs': 10, 'batch_size': 128},
        'tranad': {'seq_len': 10, 'd_ff': 16, 'train_stride': 1, 'epochs': 10, 'batch_size': 128},
        'usad': {'seq_len': 5, 'latent_dim': 32, 'train_stride': 1, 'epochs': 10, 'batch_size': 256},
        # DAGMM (TranAD-author impl, imperial-qore/TranAD): n_window=5 (flatten),
        # n_hidden=16, n_latent=8, AdamW(lr=1e-4, wd=1e-5), StepLR(5, 0.9), epochs=5.
        # n_gmm is derived (= n_feats × n_window) and not user-configurable.
        'dagmm': {'n_window': 5, 'n_hidden': 16, 'n_latent': 8,
                  'lr': 1e-4, 'weight_decay': 1e-5, 'lr_step_size': 5, 'lr_gamma': 0.9,
                  'train_stride': 1, 'epochs': 5, 'batch_size': 256},
        # GDN (Deng & Hooi, AAAI 2021): seq_len/embed_dim/top_k/out_layer_* match upstream run.sh
        # (slide_win=5, dim=64, topk=5, out_layer_num=1, out_layer_inter_dim=128).
        # batch_size=256 kept for uniformity with other SOTA presets (vs upstream 32 in run.sh).
        'gdn': {'seq_len': 5, 'embed_dim': 64, 'n_heads': 1,
                'out_layer_num': 1, 'out_layer_inter_dim': 128, 'output_dropout': 0.2,
                'lr': 0.001, 'train_stride': 1, 'epochs': 10, 'batch_size': 256},
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
            'win_size': 100, 'd_model': 64, 'd_ff': 64,
            'e_layers': 3, 'top_k': 3, 'num_kernels': 6,
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
            'train_stride': 1, 'epochs': 10, 'phase1_epochs': 3,
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
            'train_stride': 1, 'epochs': 10, 'batch_size': 128,
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
            'z_dim':          4,      # M_pt latent bottleneck (reference `lat`)
            'ff_mult':        4,
            'enc_depth':      4,      # M_pt Performer depth (reference `enc_depth`)
            'pred_depth':     4,      # M_seq Performer depth (reference `pred_depth`)
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
            'epochs':         10,     # Domain B policy (reference uses 25)
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
) -> dict:
    """Compute all metrics using MAE evaluator — returns flat dict matching epoch_metrics.json format.

    This is the SINGLE metric computation function for all baselines.
    Output format is identical to MAE's Evaluator.evaluate() output.

    Args:
        point_scores: 1D float array, anomaly score per test timestep
        point_labels: 1D int array, ground truth labels per test timestep
        anomaly_regions: List of AnomalyRegion objects (test-local)
        eval_mask: Optional boolean mask (True = evaluate). Default: all True.

    Returns:
        Flat dict with all metric keys matching MAE epoch_metrics.json format.
        MAE-only keys (teacher, disc_snr, disturbing) are set to null.
    """
    if eval_mask is None:
        eval_mask = np.ones(len(point_labels), dtype=bool)

    # Single source of truth: MAE evaluator's compute_full_metric_set.
    # Any metric added to MAE auto-flows here (no duplication needed).
    # Returns ~146 keys: base + per-K PA%K + PA%K AUC + VUS/Aff/RF1 + AR variants.
    raw = compute_full_metric_set(point_scores, point_labels, anomaly_regions, eval_mask)

    # Filter out _per_k_* diagnostic arrays (not used by baseline plots).
    results = {k: v for k, v in raw.items() if not k.startswith('_')}

    # MAE-only keys (teacher_*, disc_snr, disturbing_*, n_*) — baselines have no
    # source for these — explicit null.
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
    """Return zero-initialized metrics dict matching MAE format.

    Delegates to MAE evaluator's _zero_metric_set() — any new metric added to MAE
    auto-appears here (e.g., VUS-PR/ROC, Affiliation-F1, R-based F1, AR variants
    added 2026-05-27).
    """
    results = dict(_zero_metric_set())
    # MAE-only keys (teacher_*, disc_snr, disturbing_*, n_*) — baselines have no source.
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
    """Save epoch_metrics.json in MAE format."""
    with open(output_dir / 'epoch_metrics.json', 'w') as f:
        json.dump({'eval_interval': eval_interval, 'epochs': epoch_metrics_list}, f, indent=2)


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

    # Fit
    start_time = time.time()
    model.fit(train_X)
    train_time = time.time() - start_time
    print(f"  Train time: {train_time:.2f}s")

    # Predict
    start_time = time.time()
    scores = model.predict(test_X)
    inference_time = time.time() - start_time
    print(f"  Inference time: {inference_time:.2f}s")

    # Evaluate
    eval_start = time.time()
    metrics = compute_all_metrics(scores, test_y, anomaly_regions)
    metrics['_inference_time'] = inference_time
    metrics['_eval_time'] = time.time() - eval_start
    metrics['epoch'] = 1

    # SWaT excl22
    if excl_region is not None:
        excl = compute_all_metrics_with_excl(scores, test_y, anomaly_regions, excl_region)
        metrics.update(excl)

    # Save
    save_scores_npz(output_dir, scores)
    save_epoch_metrics(output_dir, [metrics])
    save_metadata(output_dir, model_name, experiment_name, train_time, inference_time)

    # Save model if supported
    if hasattr(model, 'save'):
        try:
            model.save(output_dir / "model")
        except Exception as e:
            print(f"  Warning: Could not save model: {e}")

    _print_metrics(metrics, excl_region is not None)
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
            # GPU inference (synchronous)
            scores = model.predict(test_X)

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
            final_scores = model.predict(test_X)
    else:
        # No per-epoch eval done — compute metrics on last-epoch predict
        final_scores = model.predict(test_X)
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

        # Inference
        scores = model_instance.predict(test_X)

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
        model.fit(train_X, epoch_callback=epoch_callback, train_segments=train_segments)
    else:
        model.fit(train_X, epoch_callback=epoch_callback)
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
            final_scores = model.predict(test_X)
    else:
        # No per-epoch eval done — compute metrics on last-epoch predict
        final_scores = model.predict(test_X)
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
