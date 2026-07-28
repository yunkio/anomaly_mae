"""pak_v2.py — headline pak_auc_f1 under official vs train mean-fixed (tmf), A/B/B0.
Confirms the prc Δ_unseen finding in the project's headline metric. Saves per-fault pak."""
import json, sys, os
import numpy as np
sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import seen_faults, unseen_faults, USABLE_FAULTS
from run_tep_simple import partition_eval, load_test

ROOT = 'results/experiments/TEP_phase2_win100_ep30'
FOLD_KEYS = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}
W_OFF = 0.25
rt = json.load(open('scripts/TEP/data/test_run_table.json'))

def best_ep(d):
    rows = json.load(open(f'{d}/epoch_metrics.json')).get('epochs', [])
    pw = [r for r in rows if r.get('epoch', 0) > 15] or rows
    return max(pw, key=lambda r: r.get('pak_auc_f1', 0)).get('epoch')

def load(path):
    z = np.load(f'{ROOT}/{path}/epoch_scores/epoch_{best_ep(f"{ROOT}/{path}"):03d}_scores.npz')
    return (np.nan_to_num(z['teacher_recon_error']).astype(np.float64),
            np.nan_to_num(z['discrepancy_error']).astype(np.float64),
            np.nan_to_num(z['official_score']).astype(np.float64))

def recover_ratio(R, D, O):
    cR, cD = np.cumsum(R), np.cumsum(D); m = D > (np.median(D) * 0.5)
    s = (O[m] - R[m]) / (W_OFF * D[m]); A_ = np.stack([np.ones(m.sum()), -s], 1); b_ = s * cD[m] - cR[m]
    sol, *_ = np.linalg.lstsq(A_, b_, rcond=None); return sol[0] / sol[1]

_, y, _, _, run_table = load_test()
def fpak(score, f):
    m, _, _ = partition_eval(score, y, run_table, {f}, lite=True); return m.get('pak_auc_f1', 0.0)

CONDP = {'A': 'phase2_A/TEP/typegen_{}', 'B': 'phase2_B/TEP/typegen_{}'}
pak = {f'{c}_{sc}': {} for c in ['A', 'B', 'B0'] for sc in ['official', 'tmf']}
for c, tmpl in CONDP.items():
    for fs in FOLD_KEYS:
        R, D, O = load(tmpl.format(fs)); rr = recover_ratio(R, D, O)
        for sc, s in [('official', O), ('tmf', R + rr * D)]:
            for f in USABLE_FAULTS:
                pak[f'{c}_{sc}'][(fs, f)] = fpak(s, f)
# B0
R, D, O = load('phase2_B0/TEP/typegen_ffonly'); rr0 = recover_ratio(R, D, O)
for sc, s in [('official', O), ('tmf', R + rr0 * D)]:
    for fs in FOLD_KEYS:
        for f in USABLE_FAULTS:
            pak[f'B0_{sc}'][(fs, f)] = fpak(s, f)

seen = [(fs, f) for fs, fk in FOLD_KEYS.items() for f in seen_faults(fk)]
uns = [(fs, f) for fs, fk in FOLD_KEYS.items() for f in unseen_faults(fk)]
print("=== pak_auc_f1 (headline) per-mode ===")
print(f"{'key':14} {'S':>7} {'U':>7}")
for k in pak:
    S = np.mean([pak[k][p] for p in seen]); U = np.mean([pak[k][p] for p in uns])
    print(f"{k:14} {S:>7.4f} {U:>7.4f}")
print("\n=== Δ (A−B) ===")
for sc in ['official', 'tmf']:
    AS = np.mean([pak[f'A_{sc}'][p] for p in seen]); BS = np.mean([pak[f'B_{sc}'][p] for p in seen])
    AU = np.mean([pak[f'A_{sc}'][p] for p in uns]); BU = np.mean([pak[f'B_{sc}'][p] for p in uns])
    print(f"  {sc}: ΔS={AS-BS:+.4f} Δ_unseen={AU-BU:+.4f}")
json.dump({k: {f'{fs}_{f}': v for (fs, f), v in d.items()} for k, d in pak.items()},
          open(f'{ROOT}/pak_v2.json', 'w'), indent=1)
print("저장: pak_v2.json")
