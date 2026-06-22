"""TEP type-generalization experiment #12 — shared constants.

Pre-registered design: temp/tep_design/80_experiment_design_final.md (frozen 2026-06-10).
This temp pipeline implements the SIMPLE-BASELINE reference runs (pca_error,
sensor_range) of that design, as comparison anchors for the upcoming MAE runs.

Standalone scripts — existing comparison/ and mae_anomaly/ code is imported,
never modified.
"""
import os
import sys

# ---- paths -----------------------------------------------------------------
# Derive repo root from this file's location (scripts/TEP/tep_common.py -> repo root)
# so the build works on any machine, not a single hardcoded checkout path.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKSPACE = os.path.join(REPO_ROOT, 'scripts', 'TEP')
DATA_DIR = os.path.join(WORKSPACE, 'data')
RESULTS_BASE = os.path.join(WORKSPACE, 'results')
TEP_RAW_DIR = os.path.join(REPO_ROOT, 'dataset', 'TEP')

EXPERIMENT_NUMBER = 12  # user-assigned experiment numbering

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---- fault taxonomy (Downs & Vogel; design §2.2) ---------------------------
FAMILY = {
    'step':   [1, 2, 4, 5, 6, 7],
    'random': [8, 10, 11, 12],
    'drift_sticking': [13, 14],
    'unknown': [16, 17, 18, 19, 20],
}
EXCLUDED_HARD = [3, 9, 15]          # closed-loop fully compensated (design §2.2 rule)
USABLE_FAULTS = sorted(f for fam in FAMILY.values() for f in fam)   # 17 faults
ALL_FAULTS = sorted(USABLE_FAULTS + EXCLUDED_HARD)                  # 20 faults

# ---- fold definitions: seen family + train-run allocation (design §2.2-2.3) -
# Every fold injects exactly 60 faulty runs into train (label-budget equalized).
FOLDS = {
    'f_step': {'seen_family': 'step',           'train_runs_per_fault': 10},  # 6 x 10
    'f_rand': {'seen_family': 'random',         'train_runs_per_fault': 15},  # 4 x 15
    'f_ds':   {'seen_family': 'drift_sticking', 'train_runs_per_fault': 30},  # 2 x 30
    'f_unk':  {'seen_family': 'unknown',        'train_runs_per_fault': 12},  # 5 x 12
}

def seen_faults(fold: str):
    return list(FAMILY[FOLDS[fold]['seen_family']])

def unseen_faults(fold: str):
    s = set(seen_faults(fold))
    return [f for f in USABLE_FAULTS if f not in s]

# ---- run allocation (design §2.3; deterministic simulationRun IDs) ----------
FF_TRAIN_RUNS = list(range(1, 241))      # 240 runs  (FaultFree_Testing)
FF_VAL_RUNS   = list(range(241, 281))    # 40 runs   (reserved; unused for simple models)
FF_TEST_RUNS  = list(range(461, 501))    # 40 runs
FAULTY_VAL_RUNS  = list(range(301, 321)) # reserved; unused for simple models
FAULTY_TEST_RUNS = list(range(441, 461)) # 20 runs per fault, ALL 20 faults

RUN_LEN = 960            # samples per Testing run
FAULT_ONSET_IDX = 160    # 0-indexed: samples 1..160 (1-indexed) normal, 161.. faulty
                         # (design §2.4: onset corrected to sample 161, i.e. 160 normal pts)

# near/far pairs (same physical variable, step<->random; design §2.2)
NEAR_PAIRS = [(4, 11), (5, 12)]

# ---- model settings ---------------------------------------------------------
# Identical to comparison/baseline_common.py:_get_default_model_params
MODEL_PARAMS = {
    'sensor_range': {'sensor_range': (0, 1), 'count_sensors': False},
    'pca_error':    {'pca_dim': 'auto', 'svd_solver': 'full'},
    'l2_norm':      {'ord': 2},                       # eval_simple_baselines.py:37
    'nn_distance':  {'distance': 'euclidean'},        # full train, no subsample (paper-faithful)
    'random':       {'seed': None},                   # 5 independent runs, mean±std (paper §4.2)
}
ALL_SIMPLE_MODELS = ['random', 'sensor_range', 'pca_error', 'l2_norm', 'nn_distance']
# Normalization: comparison-pipeline Q1 standard = minmax, train-only fit,
# NO clip (sklearn MinMaxScaler semantics; required for sensor_range to work).
# NOTE deviation from design §2.6 (zscore): the design's zscore rule targets the
# MAE-pipeline minmax_clamp saturation; the comparison pipeline's minmax has no
# clip, and sensor_range's (0,1) range is defined against minmax. We therefore
# keep the comparison-standard minmax for these two simple models ("기존 baseline
# 실험과 그대로" per user instruction) and document this in the analysis.
NORMALIZE_MODE = 'minmax_noclip'
