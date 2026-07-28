"""validate_scoring.py — confirm the winning fusion rule(s) with pak_auc_f1 (headline metric).

Recomputes per-fault pak_auc_f1 (partition_eval, same metric stack as everything else)
for A and B under candidate fusion rules, then reports per-mode S/U macro + #faults A>B.
"""
import json, os, sys
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import seen_faults, unseen_faults, USABLE_FAULTS
from run_tep_simple import partition_eval, load_test

ROOT = 'results/experiments/TEP_phase2_win100_ep30'
FOLD_KEYS = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}

def best_ep(d):
    rows = json.load(open(f'{d}/epoch_metrics.json')).get('epochs', [])
    pw = [r for r in rows if r.get('epoch', 0) > 15] or rows
    return max(pw, key=lambda r: r.get('pak_auc_f1', 0)).get('epoch')

def load(cond, fs):
    d = f'{ROOT}/{cond}/TEP/typegen_{fs}'
    z = np.load(f'{d}/epoch_scores/epoch_{best_ep(d):03d}_scores.npz')
    return np.nan_to_num(z['teacher_recon_error']), np.nan_to_num(z['discrepancy_error']), np.nan_to_num(z['official_score'])

def rnk(x):
    return rankdata(x) / len(x)

RULES = {
    'official':  lambda R, D, O: O,
    'rank_sum':  lambda R, D, O: rnk(R) + rnk(D),
    'rank_max':  lambda R, D, O: np.maximum(rnk(R), rnk(D)),
    'R+1D':      lambda R, D, O: R + D,
}

_, y, _, _, run_table = load_test()

def fault_pak(score, fault):
    m, _, _ = partition_eval(score, y, run_table, {fault}, lite=True)
    return m.get('pak_auc_f1', 0.0)

# per-fault pak for A and B under each rule
res = {rn: {'A': {}, 'B': {}} for rn in RULES}
for fs, fk in FOLD_KEYS.items():
    RA, DA, OA = load('phase2_A', fs); RB, DB, OB = load('phase2_B', fs)
    for rn, fn in RULES.items():
        sA, sB = fn(RA, DA, OA), fn(RB, DB, OB)
        for f in USABLE_FAULTS:
            res[rn]['A'][(fs, f)] = fault_pak(sA, f)
            res[rn]['B'][(fs, f)] = fault_pak(sB, f)

seen = [(fs, f) for fs, fk in FOLD_KEYS.items() for f in seen_faults(fk)]
unseen = [(fs, f) for fs, fk in FOLD_KEYS.items() for f in unseen_faults(fk)]

print(f"\n=== pak_auc_f1 validation (per-mode macro + A>B count) ===")
print(f"{'rule':10} {'seenA>B':>8} {'A_S':>7} {'B_S':>7} {'ΔS':>+8} {'unsA>B':>7} {'A_U':>7} {'B_U':>7} {'ΔU':>+8}")
for rn in RULES:
    As = [res[rn]['A'][p] for p in seen]; Bs = [res[rn]['B'][p] for p in seen]
    Au = [res[rn]['A'][p] for p in unseen]; Bu = [res[rn]['B'][p] for p in unseen]
    cs = sum(1 for a, b in zip(As, Bs) if a > b); cu = sum(1 for a, b in zip(Au, Bu) if a > b)
    print(f"{rn:10} {cs:>3}/{len(seen):<4} {np.mean(As):>7.4f} {np.mean(Bs):>7.4f} {np.mean(As)-np.mean(Bs):>+8.4f} "
          f"{cu:>3}/{len(unseen):<3} {np.mean(Au):>7.4f} {np.mean(Bu):>7.4f} {np.mean(Au)-np.mean(Bu):>+8.4f}")

# best rule per-fault detail
rn = 'rank_sum'
print(f"\n=== '{rn}' per seen-fault pak (vs official) ===")
print(f"{'fold':6} {'IDV':>4} {'A_new':>7} {'B_new':>7} {'Δnew':>+7} | {'A_off':>7} {'B_off':>7} {'Δoff':>+7}")
for fs, f in seen:
    a, b = res[rn]['A'][(fs, f)], res[rn]['B'][(fs, f)]
    ao, bo = res['official']['A'][(fs, f)], res['official']['B'][(fs, f)]
    fl = '' if a > b else '  A<=B'
    print(f"{fs:6} {f:>4} {a:>7.4f} {b:>7.4f} {a-b:>+7.4f} | {ao:>7.4f} {bo:>7.4f} {ao-bo:>+7.4f}{fl}")

json.dump({rn: {f'{fs}_{f}': {'A': res[rn]['A'][(fs, f)], 'B': res[rn]['B'][(fs, f)]}
                for fs, f in seen + unseen} for rn in RULES},
          open(f'{ROOT}/scoring_validate_pak.json', 'w'), indent=1)
print(f"\n저장: {ROOT}/scoring_validate_pak.json")
