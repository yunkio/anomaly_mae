"""method_summary.py — every scoring rule tried, with per-mode seen/unseen (prc) + causality tag."""
import json, sys
import numpy as np
from sklearn.metrics import average_precision_score as ap
from scipy.stats import rankdata
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

def zpfx(X):
    out = np.zeros_like(X)
    for r in rt:
        s, e = r['start'], r['end']; p = X[s:s + ONSET]
        out[s:e] = (X[s:e] - p.mean()) / (p.std() + 1e-9)
    return out

def zstream(X):
    return (X - X.mean()) / (X.std() + 1e-9)

# (name, causality, fn)  — higher score = more anomalous
RULES = [
    ('official  recon+0.25·disc·sₜ', 'causal(현행)', lambda R, D, O: O),
    ('recon-only (=D 조건)', 'causal', lambda R, D, O: R),
    ('disc-only', 'causal', lambda R, D, O: D),
    ('raw  recon+1·disc', 'causal', lambda R, D, O: R + D),
    ('raw  recon+10·disc', 'causal', lambda R, D, O: R + 10 * D),
    ('raw  recon+25·disc', 'causal', lambda R, D, O: R + 25 * D),
    ('raw  recon+50·disc', 'causal', lambda R, D, O: R + 50 * D),
    ('raw  recon+100·disc', 'causal', lambda R, D, O: R + 100 * D),
    ('amp_k3  recon+0.75·disc·sₜ', 'causal', lambda R, D, O: R + 3 * (O - R)),
    ('amp_k5  recon+1.25·disc·sₜ', 'causal', lambda R, D, O: R + 5 * (O - R)),
    ('amp_k10 recon+2.5·disc·sₜ', 'causal', lambda R, D, O: R + 10 * (O - R)),
    ('zpfx  zPREFIX(R)+zPREFIX(D)', 'prefix가정(약leak)', lambda R, D, O: zpfx(R) + zpfx(D)),
    ('rank  rank(R)+rank(D)', 'LEAKAGE', lambda R, D, O: rankdata(R) / len(R) + rankdata(D) / len(D)),
    ('zstream  zSTREAM(R)+zSTREAM(D)', 'LEAKAGE', lambda R, D, O: zstream(R) + zstream(D)),
]

prc = {nm: {'A': {}, 'B': {}} for nm, _, _ in RULES}
for fs, fk in FOLD_KEYS.items():
    RA, DA, OA, yA = load('phase2_A', fs); RB, DB, OB, yB = load('phase2_B', fs)
    for nm, tag, fn in RULES:
        sA, sB = fn(RA, DA, OA), fn(RB, DB, OB)
        for f in USABLE_FAULTS:
            ix = PIDX[f]
            prc[nm]['A'][(fs, f)] = ap(yA[ix], sA[ix]); prc[nm]['B'][(fs, f)] = ap(yB[ix], sB[ix])

seen = [(fs, f) for fs, fk in FOLD_KEYS.items() for f in seen_faults(fk)]
uns = [(fs, f) for fs, fk in FOLD_KEYS.items() for f in unseen_faults(fk)]

print(f"{'method':32} {'causality':16} {'A_S':>6} {'B_S':>6} {'ΔS':>7} | {'A_U':>6} {'B_U':>6} {'Δ_unseen':>9} {'H1?':>4}")
print('-' * 110)
for nm, tag, _ in RULES:
    AS = np.mean([prc[nm]['A'][p] for p in seen]); BS = np.mean([prc[nm]['B'][p] for p in seen])
    AU = np.mean([prc[nm]['A'][p] for p in uns]); BU = np.mean([prc[nm]['B'][p] for p in uns])
    h1 = 'YES' if AU - BU > 0 else 'no'
    print(f"{nm:32} {tag:16} {AS:>6.3f} {BS:>6.3f} {AS-BS:>+7.4f} | {AU:>6.3f} {BU:>6.3f} {AU-BU:>+9.4f} {h1:>4}")
