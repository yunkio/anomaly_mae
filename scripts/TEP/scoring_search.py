"""scoring_search.py — search alternative score-fusion rules so MAE-A > MAE-B per fault.

Re-combines the SAVED per-point components (teacher_recon_error R, discrepancy_error D)
of each Phase2 A/B best-epoch npz into candidate fusion rules, then re-evaluates
per-fault separation (roc) for A vs B. No retraining (weights not needed).

Rule families: constant ratio (R+w·D), per-head z-norm (stream & causal per-run-prefix),
rank fusion, and adaptive (weight D by the run's training disc_snr).
Goal metric: # seen faults where A>B, and seen/unseen macro A-B.
"""
import json, os, sys
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import seen_faults, unseen_faults, USABLE_FAULTS, FAMILY

ROOT = 'results/experiments/TEP_phase2_win100_ep30'
FOLD_KEYS = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}
ONSET = 160
rt = json.load(open('scripts/TEP/data/test_run_table.json'))
PIDX = {f: np.concatenate([np.arange(r['start'], r['end']) for r in rt if r['fault'] == f or r['fault'] == 0])
        for f in USABLE_FAULTS}

def best_ep(d):
    rows = json.load(open(f'{d}/epoch_metrics.json')).get('epochs', [])
    pw = [r for r in rows if r.get('epoch', 0) > 15] or rows
    b = max(pw, key=lambda r: r.get('pak_auc_f1', 0))
    return b.get('epoch'), b.get('disc_snr', 1.0), b.get('recon_snr', 1.0)

def load(cond, fs):
    d = f'{ROOT}/{cond}/TEP/typegen_{fs}'
    be, dsnr, rsnr = best_ep(d)
    z = np.load(f'{d}/epoch_scores/epoch_{be:03d}_scores.npz')
    return (np.nan_to_num(z['teacher_recon_error']), np.nan_to_num(z['discrepancy_error']),
            np.nan_to_num(z['official_score']), z['point_labels'].astype(int), dsnr)

def zsc(x):
    return (x - x.mean()) / (x.std() + 1e-12)

def rnk(x):
    return rankdata(x) / len(x)

def z_prefix(X):
    """causal per-run z-norm using each run's first-ONSET normal prefix."""
    out = np.zeros_like(X)
    for r in rt:
        s, e = r['start'], r['end']
        p = X[s:s + ONSET]
        out[s:e] = (X[s:e] - p.mean()) / (p.std() + 1e-12)
    return out

# rule(R, D, O, dsnr) -> fused point score (higher = more anomalous)
RULES = {
    'official(base)':  lambda R, D, O, s: O,
    'recon_only':      lambda R, D, O, s: R,
    'disc_only':       lambda R, D, O, s: D,
    'R+1·D':           lambda R, D, O, s: R + D,
    'R+(σR/σD)·D':     lambda R, D, O, s: R + (R.std() / (D.std() + 1e-12)) * D,
    'z_sum(stream)':   lambda R, D, O, s: zsc(R) + zsc(D),
    'z_max(stream)':   lambda R, D, O, s: np.maximum(zsc(R), zsc(D)),
    'z_R+2D(stream)':  lambda R, D, O, s: zsc(R) + 2 * zsc(D),
    'rank_sum':        lambda R, D, O, s: rnk(R) + rnk(D),
    'rank_max':        lambda R, D, O, s: np.maximum(rnk(R), rnk(D)),
    'zpfx_sum(causal)': lambda R, D, O, s: z_prefix(R) + z_prefix(D),
    'zpfx_max(causal)': lambda R, D, O, s: np.maximum(z_prefix(R), z_prefix(D)),
    'adapt_dsnr·zD':   lambda R, D, O, s: zsc(R) + max(s, 0.05) / 0.2 * zsc(D),
}

# precompute fused scores + per-fault roc for A and B in each fold
roc = {rn: {'A': {}, 'B': {}} for rn in RULES}
for fs, fk in FOLD_KEYS.items():
    RA, DA, OA, yA, sA = load('phase2_A', fs)
    RB, DB, OB, yB, sB = load('phase2_B', fs)
    for rn, fn in RULES.items():
        scA = fn(RA, DA, OA, sA); scB = fn(RB, DB, OB, sB)
        for f in USABLE_FAULTS:
            ix = PIDX[f]
            roc[rn]['A'][(fs, f)] = roc_auc_score(yA[ix], scA[ix])
            roc[rn]['B'][(fs, f)] = roc_auc_score(yB[ix], scB[ix])

seen_pairs = [(fs, f) for fs, fk in FOLD_KEYS.items() for f in seen_faults(fk)]      # 17
unseen_pairs = [(fs, f) for fs, fk in FOLD_KEYS.items() for f in unseen_faults(fk)]  # 17*? per fold

print("=== SEEN faults (labeled): rule이 A>B를 만드는가 (roc) ===")
print(f"{'rule':18} {'A>B':>7} {'A_S':>7} {'B_S':>7} {'A-B':>8} {'worst(A-B)':>11} {'A_U':>7} {'B_U':>7}")
rows = []
for rn in RULES:
    A = [roc[rn]['A'][p] for p in seen_pairs]; B = [roc[rn]['B'][p] for p in seen_pairs]
    d = [a - b for a, b in zip(A, B)]
    AU = [roc[rn]['A'][p] for p in unseen_pairs]; BU = [roc[rn]['B'][p] for p in unseen_pairs]
    cnt = sum(1 for x in d if x > 0)
    rows.append((rn, cnt, np.mean(A), np.mean(B), np.mean(d), min(d), np.mean(AU), np.mean(BU)))
    print(f"{rn:18} {cnt:>2}/{len(seen_pairs):<4} {np.mean(A):>7.4f} {np.mean(B):>7.4f} {np.mean(d):>+8.4f} {min(d):>+11.4f} {np.mean(AU):>7.4f} {np.mean(BU):>7.4f}")

# detail for the best rule (max A>B count, tiebreak mean diff)
best = max(rows, key=lambda r: (r[1], r[4]))
rn = best[0]
print(f"\n=== best rule = '{rn}' — per seen-fault A vs B (roc) ===")
print(f"{'fold':6} {'IDV':>4} {'A':>7} {'B':>7} {'A-B':>8}  vs official: {'Aoff':>7} {'Boff':>7}")
for fs, f in seen_pairs:
    a, b = roc[rn]['A'][(fs, f)], roc[rn]['B'][(fs, f)]
    ao, bo = roc['official(base)']['A'][(fs, f)], roc['official(base)']['B'][(fs, f)]
    flag = '' if a > b else '  <-- A<=B'
    print(f"{fs:6} {f:>4} {a:>7.4f} {b:>7.4f} {a-b:>+8.4f}      {ao:>7.4f} {bo:>7.4f}{flag}")

json.dump({rn: {'seen': {f'{fs}_{f}': {'A': roc[rn]['A'][(fs, f)], 'B': roc[rn]['B'][(fs, f)]} for fs, f in seen_pairs}}
           for rn in RULES}, open(f'{ROOT}/scoring_search_roc.json', 'w'), indent=1)
print(f"\n저장: {ROOT}/scoring_search_roc.json")
