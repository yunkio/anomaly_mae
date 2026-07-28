"""std_balance_test.py — does matching STD (not just mean) fix the official?

Compares, per-mode seen/unseen (prc), 4 ways to combine recon(R) and disc(D):
  official    : R + 0.25·D·s_t                  (mean-balanced via s_t, PER-POINT varying)
  mean-fixed  : R + (μR/μD)·D                    (FIXED mean-ratio, no s_t)
  std-fixed(z): (R−μR)/σR + (D−μD)/σD            (FIXED mean+STD, no s_t)
  raw_w25     : R + 25·D                          (constant ref)
μ,σ from each run's NORMAL prefix (causal proxy for train-normal stats; deploy = save train μ,σ).
"""
import json, sys
import numpy as np
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import seen_faults, unseen_faults, USABLE_FAULTS

ROOT = 'results/experiments/TEP_phase2_win100_ep30'
FOLD_KEYS = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}
ONSET = 160
rt = json.load(open('scripts/TEP/data/test_run_table.json'))
PIDX = {f: np.concatenate([np.arange(r['start'], r['end']) for r in rt if r['fault'] == f or r['fault'] == 0])
        for f in USABLE_FAULTS}

def best_ep(d):
    rows = json.load(open(f'{d}/epoch_metrics.json')).get('epochs', [])
    pw = [r for r in rows if r.get('epoch', 0) > 15] or rows
    return max(pw, key=lambda r: r.get('pak_auc_f1', 0)).get('epoch')

def load(sub, fs):
    d = f'{ROOT}/{sub}/TEP/typegen_{fs}'
    z = np.load(f'{d}/epoch_scores/epoch_{best_ep(d):03d}_scores.npz')
    return (np.nan_to_num(z['teacher_recon_error']), np.nan_to_num(z['discrepancy_error']),
            np.nan_to_num(z['official_score']), z['point_labels'].astype(int))

def normal_stats(X):
    """global mean/std over all run NORMAL prefixes (proxy for train-normal)."""
    pref = np.concatenate([X[r['start']:r['start'] + ONSET] for r in rt])
    return float(pref.mean()), float(pref.std() + 1e-12)

def permode(scoreA, scoreB, yA, yB):
    pA = {f: ap(yA[PIDX[f]], scoreA[PIDX[f]]) for f in USABLE_FAULTS}
    pB = {f: ap(yB[PIDX[f]], scoreB[PIDX[f]]) for f in USABLE_FAULTS}
    return pA, pB

METHODS = ['official', 'mean-fixed', 'std-fixed(z)', 'raw_w25']
acc = {m: {'A': {}, 'B': {}} for m in METHODS}
ratios = []
for fs, fk in FOLD_KEYS.items():
    RA, DA, OA, yA = load('phase2_A', fs); RB, DB, OB, yB = load('phase2_B', fs)
    for tag, (R, D, O, y) in [('A', (RA, DA, OA, yA)), ('B', (RB, DB, OB, yB))]:
        muR, sgR = normal_stats(R); muD, sgD = normal_stats(D)
        if tag == 'A':
            ratios.append((fs, muR / muD, sgR / sgD))
        scores = {
            'official': O,
            'mean-fixed': R + (muR / muD) * D,
            'std-fixed(z)': (R - muR) / sgR + (D - muD) / sgD,
            'raw_w25': R + 25 * D,
        }
        for m in METHODS:
            for f in USABLE_FAULTS:
                acc[m][tag][(fs, f)] = ap(y[PIDX[f]], scores[m][PIDX[f]])

print("=== normal-prefix scale ratios (A) : mean-ratio μR/μD vs std-ratio σR/σD ===")
for fs, mr, sr in ratios:
    print(f"  {fs:6}: μR/μD={mr:7.1f}   σR/σD={sr:7.1f}")
print(f"  (official은 0.25·μR/μD ≈ {0.25*np.mean([r[1] for r in ratios]):.1f}배로 disc를 down-weight)")

seen = [(fs, f) for fs, fk in FOLD_KEYS.items() for f in seen_faults(fk)]
uns = [(fs, f) for fs, fk in FOLD_KEYS.items() for f in unseen_faults(fk)]
print(f"\n{'method':14} {'A_S':>6} {'B_S':>6} {'ΔS':>8} | {'A_U':>6} {'B_U':>6} {'Δ_unseen':>9} {'H1?':>4}")
for m in METHODS:
    AS = np.mean([acc[m]['A'][p] for p in seen]); BS = np.mean([acc[m]['B'][p] for p in seen])
    AU = np.mean([acc[m]['A'][p] for p in uns]); BU = np.mean([acc[m]['B'][p] for p in uns])
    print(f"{m:14} {AS:>6.3f} {BS:>6.3f} {AS-BS:>+8.4f} | {AU:>6.3f} {BU:>6.3f} {AU-BU:>+9.4f} {'YES' if AU>BU else 'no':>4}")
