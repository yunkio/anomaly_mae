"""train_meanfix_test.py — recover the TRAIN mean-ratio R_tr/D_tr from the official score
and test a TRULY train-only mean-fixed:  recon + (R_tr/D_tr)·disc  (no test stats at all).

official = recon + 0.25·disc·s_t,  s_t = (R_tr + cumsum_recon)/(D_tr + cumsum_disc).
→ s_t = (official − recon)/(0.25·disc).  Then R_tr,D_tr solve the linear relation
   R_tr − s_t·D_tr = s_t·cumsum_disc − cumsum_recon   (least squares).
R_tr/D_tr is the train-normal recon/disc MEAN ratio (N cancels) = fully causal, train-only.
"""
import json, sys
import numpy as np
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import seen_faults, unseen_faults, USABLE_FAULTS

ROOT = 'results/experiments/TEP_phase2_win100_ep30'
FOLD_KEYS = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}
W_OFF = 0.25
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
    return (np.nan_to_num(z['teacher_recon_error']).astype(np.float64),
            np.nan_to_num(z['discrepancy_error']).astype(np.float64),
            np.nan_to_num(z['official_score']).astype(np.float64), z['point_labels'].astype(int))

def recover_RtrDtr(R, D, O):
    """least-squares recover (R_tr, D_tr) from official score."""
    cR = np.cumsum(R); cD = np.cumsum(D)
    mask = D > (np.median(D) * 0.5)           # avoid disc≈0 (s_t ill-defined)
    s = (O[mask] - R[mask]) / (W_OFF * D[mask])
    A_ = np.stack([np.ones(mask.sum()), -s], axis=1)   # [1, -s_t]
    b_ = s * cD[mask] - cR[mask]
    sol, *_ = np.linalg.lstsq(A_, b_, rcond=None)
    return sol[0], sol[1]                       # R_tr, D_tr

METHODS = ['official', 'meanfix_TRAIN (R_tr/D_tr)', 'raw_w25']
acc = {m: {'A': {}, 'B': {}} for m in METHODS}
print("=== 복원된 train ratio R_tr/D_tr (= train-normal recon/disc 평균비) ===")
for fs, fk in FOLD_KEYS.items():
    for tag, sub in [('A', 'phase2_A'), ('B', 'phase2_B')]:
        R, D, O, y = load(sub, fs)
        Rtr, Dtr = recover_RtrDtr(R, D, O)
        ratio = Rtr / Dtr if Dtr != 0 else float('nan')
        if tag == 'A':
            print(f"  {fs:6} A: R_tr={Rtr:.1f}  D_tr={Dtr:.3f}  R_tr/D_tr={ratio:.1f}")
        scores = {'official': O, 'meanfix_TRAIN (R_tr/D_tr)': R + ratio * D, 'raw_w25': R + 25 * D}
        for m in METHODS:
            for f in USABLE_FAULTS:
                acc[m][tag][(fs, f)] = ap(y[PIDX[f]], scores[m][PIDX[f]])

seen = [(fs, f) for fs, fk in FOLD_KEYS.items() for f in seen_faults(fk)]
uns = [(fs, f) for fs, fk in FOLD_KEYS.items() for f in unseen_faults(fk)]
print(f"\n{'method':26} {'A_S':>6} {'B_S':>6} {'ΔS':>8} | {'A_U':>6} {'B_U':>6} {'Δ_unseen':>9} {'H1?':>4} {'leakage':>10}")
leak = {'official': 'none', 'meanfix_TRAIN (R_tr/D_tr)': 'NONE(train)', 'raw_w25': 'const(tuned)'}
for m in METHODS:
    AS = np.mean([acc[m]['A'][p] for p in seen]); BS = np.mean([acc[m]['B'][p] for p in seen])
    AU = np.mean([acc[m]['A'][p] for p in uns]); BU = np.mean([acc[m]['B'][p] for p in uns])
    print(f"{m:26} {AS:>6.3f} {BS:>6.3f} {AS-BS:>+8.4f} | {AU:>6.3f} {BU:>6.3f} {AU-BU:>+9.4f} {'YES' if AU>BU else 'no':>4} {leak[m]:>10}")
