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

# Both Rieth et al. Testing RData sets carry simulationRun 1..500 (per fault for
# the faulty set) — the sampling pool for the data-seed axis.
RUN_POOL_SIZE = 500


def train_runs_needed():
    """fault -> #train runs required (union over folds).

    Each fault family is 'seen' in exactly one fold, so the union per fault is
    simply that fold's train_runs_per_fault; EXCLUDED_HARD faults (3/9/15) are
    never trained on -> absent (0)."""
    need = {}
    for fold, cfg in FOLDS.items():
        for f in seen_faults(fold):
            need[f] = max(need.get(f, 0), cfg['train_runs_per_fault'])
    return need


def allocate_runs(data_seed=None):
    """Run-ID allocation for the typegen streams (data-seed axis, 2026-07-24).

    data_seed None  -> the frozen canonical allocation above (FF train 1-240 /
      FF test 461-500 / faulty test 441-460 / faulty train 1..N), byte-identical
      to the original #12 build.
    data_seed N     -> sampling WITHOUT replacement via np.random.default_rng(N):
      FF pool 1..500        : train 240 drawn first, then test 40   (disjoint);
      per fault (1..20 asc) : test 20 drawn first, then train N_f   (disjoint),
      N_f = train_runs_needed()[f] (0 for EXCLUDED_HARD 3/9/15 — test-only).
    Set sizes, fold composition, onset and stream-ordering rules are unchanged
    from canonical; only the run IDs move.

    Returns {'data_seed', 'ff_train', 'ff_test',
             'faulty_train': {fault: [...]}, 'faulty_test': {fault: [...]}}
    with every run-ID list ascending (stream layout sorts runs ascending).
    """
    need = train_runs_needed()
    if data_seed is None:
        return {
            'data_seed': None,
            'ff_train': list(FF_TRAIN_RUNS),
            'ff_test': list(FF_TEST_RUNS),
            'faulty_train': {f: list(range(1, need.get(f, 0) + 1)) for f in ALL_FAULTS},
            'faulty_test': {f: list(FAULTY_TEST_RUNS) for f in ALL_FAULTS},
        }
    import numpy as np
    rng = np.random.default_rng(int(data_seed))
    pool = np.arange(1, RUN_POOL_SIZE + 1)
    n_tr, n_te = len(FF_TRAIN_RUNS), len(FF_TEST_RUNS)
    ff_pick = rng.choice(pool, size=n_tr + n_te, replace=False)
    alloc = {
        'data_seed': int(data_seed),
        'ff_train': sorted(int(x) for x in ff_pick[:n_tr]),
        'ff_test': sorted(int(x) for x in ff_pick[n_tr:]),
        'faulty_train': {},
        'faulty_test': {},
    }
    n_fte = len(FAULTY_TEST_RUNS)
    for f in ALL_FAULTS:              # ascending fault order -> deterministic rng stream
        k = need.get(f, 0)
        pick = rng.choice(pool, size=n_fte + k, replace=False)
        alloc['faulty_test'][f] = sorted(int(x) for x in pick[:n_fte])
        alloc['faulty_train'][f] = sorted(int(x) for x in pick[n_fte:])
    # invariants (canonical passes trivially; seeded must hold by construction)
    assert len(alloc['ff_train']) == n_tr and len(alloc['ff_test']) == n_te
    assert not set(alloc['ff_train']) & set(alloc['ff_test'])
    for f in ALL_FAULTS:
        assert len(alloc['faulty_test'][f]) == n_fte
        assert len(alloc['faulty_train'][f]) == need.get(f, 0)
        assert not set(alloc['faulty_train'][f]) & set(alloc['faulty_test'][f])
    return alloc


def data_dir_for(data_seed=None):
    """Default stream dir: canonical -> data/, data-seed N -> data_dataseed{N}/."""
    if data_seed is None:
        return DATA_DIR
    return os.path.join(WORKSPACE, f'data_dataseed{int(data_seed)}')

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
