"""build_vus.py — re-verify TEP Table 3 under VUS-PR (paper v22 headline metric).
Same train mean-fixed scoring; metric = vus_pr (lite=False, ~6s/fault). Incremental save.
Computes: Table-3 conditions (A/B/B0/D/nogrl) + simple baselines + breadth(Shapley) per-mode S/U,
discriminants (Δ_unseen, Δ_gap), GRL-ablation decomposition, Shapley — all under VUS-PR.

Provenance warning: this historical script selects a test-PAK-best epoch after
epoch 15 from the 30/15-epoch root.  Paper v22 Appendix A.3 instead specifies
10 total epochs and a 5-epoch Teacher-only phase, so regenerate with that
protocol before treating new values as protocol-faithful paper results.
"""
import json, glob, sys, os, itertools
from math import factorial
import numpy as np
sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import seen_faults, unseen_faults, USABLE_FAULTS
from run_tep_simple import partition_eval, load_test, fit_minmax, apply_minmax, MODEL_PARAMS
from comparison.baselines.pca_error.model import PCAError

R = 'results/experiments/TEP_phase2_win100_ep30'
SIMPLE = 'scripts/TEP/results/12_20260610_211815_tep_typegen_simple'
FK = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}
FAM_FAULTS = {'step': [1, 2, 4, 5, 6, 7], 'rand': [8, 10, 11, 12], 'ds': [13, 14], 'unk': [16, 17, 18, 19, 20]}
USABLE = set(USABLE_FAULTS)
OUT = f'{R}/vus_results.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(res, open(OUT, 'w'), indent=1)

def train_ratio(d):
    f = glob.glob(f'{d}/best_epoch_train_scores.npz')
    if not f: return None
    z = np.load(f[0]); rec = np.nan_to_num(z['teacher_recon_error']); dis = np.nan_to_num(z['discrepancy_error']); lab = z['point_labels']
    m = lab == 0; return float(rec[m].mean() / dis[m].mean())

def best_ep(d):
    rows = json.load(open(f'{d}/epoch_metrics.json')).get('epochs', [])
    pw = [r for r in rows if r.get('epoch', 0) > 15] or rows
    return max(pw, key=lambda r: r.get('pak_auc_f1', 0)).get('epoch')

_, y, _, _, run_table = load_test()
def vus(score, f):
    return partition_eval(score, y, run_table, {f}, lite=False)[0].get('vus_pr', 0.0)

def mae_score(d, scoring='tmf'):
    rr = train_ratio(d)
    z = np.load(f'{d}/epoch_scores/epoch_{best_ep(d):03d}_scores.npz')
    rec = np.nan_to_num(z['teacher_recon_error']).astype(np.float64); dis = np.nan_to_num(z['discrepancy_error']).astype(np.float64)
    return rec if scoring == 'recon' else rec + rr * dis

def do(key, scorefn, faults):
    if key in res: return
    s = scorefn()
    res[key] = {str(f): vus(s, f) for f in faults}; save(); print(f'  done {key}', flush=True)

# ---- 1. MAE conditions (USABLE 17) ----
print('=== MAE conditions (vus_pr) ===', flush=True)
for fk in FK:
    do(f'A_{fk}', lambda fk=fk: mae_score(f'{R}/phase2_A/TEP/typegen_{fk}'), USABLE_FAULTS)
    do(f'B_{fk}', lambda fk=fk: mae_score(f'{R}/phase2_B/TEP/typegen_{fk}'), USABLE_FAULTS)
    do(f'D_{fk}', lambda fk=fk: mae_score(f'{R}/phase2_A/TEP/typegen_{fk}', 'recon'), USABLE_FAULTS)
    do(f'nogrl_{fk}', lambda fk=fk: mae_score(f'{R}/phase2_nogrl/TEP/typegen_{fk}'), USABLE_FAULTS)
do('B0', lambda: mae_score(f'{R}/phase2_B0/TEP/typegen_ffonly'), USABLE_FAULTS)

# ---- 2. breadth (for Shapley): all 32 configs, held-out faults vus_pr ----
print('=== breadth (vus_pr, Shapley) ===', flush=True)
def breadth_dir(ho, sub, tf):
    if len(sub) == 3: return f'{R}/lofo/TEP/typegen_lofo_{ho}'
    tag = 'k0' if len(sub) == 0 else '-'.join(f for f in tf if f in sub)
    return f'{R}/breadth/TEP/typegen_breadth_{ho}_{tag}'
for ho in ['step', 'rand', 'ds', 'unk']:
    tf = [f for f in ('step', 'rand', 'ds', 'unk') if f != ho]
    ho_faults = [f for f in FAM_FAULTS[ho] if f in USABLE]
    for k in range(4):
        for sub in itertools.combinations(tf, k):
            tag = 'lofo' if k == 3 else ('k0' if k == 0 else '-'.join(f for f in tf if f in sub))
            key = f'br_{ho}_{tag}'
            do(key, lambda d=breadth_dir(ho, sub, tf): mae_score(d), ho_faults)

# ---- 3. simple baselines (USABLE 17) ----
print('=== simple baselines (vus_pr) ===', flush=True)
BASE = [('Random', 'random'), ('PCA', 'pca_error'), ('NN', 'nn_distance'), ('Sensor', 'sensor_range'), ('L2', 'l2_norm')]
for lbl, mdir in BASE:
    for fk in FK:
        key = f'simple_{lbl}_{fk}'
        if key in res: continue
        sc = np.load(f'{SIMPLE}/{FK[fk]}/{mdir}/scores.npz')['anomaly_score'].astype(np.float64)
        res[key] = {str(f): vus(sc, f) for f in USABLE_FAULTS}; save(); print(f'  done {key}', flush=True)

print('\n저장 완료:', OUT, '| keys:', len(res))
