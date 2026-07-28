"""validate_redo.py — confirm the causal weighted-sum winner(s) with pak_auc_f1 (headline)."""
import json, sys
import numpy as np
sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import seen_faults, unseen_faults, USABLE_FAULTS
from run_tep_simple import partition_eval, load_test

ROOT = 'results/experiments/TEP_phase2_win100_ep30'
FOLD_KEYS = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}
ONSET = 160
rt = json.load(open('scripts/TEP/data/test_run_table.json'))

def best_ep(d):
    rows = json.load(open(f'{d}/epoch_metrics.json')).get('epochs', [])
    pw = [r for r in rows if r.get('epoch', 0) > 15] or rows
    return max(pw, key=lambda r: r.get('pak_auc_f1', 0)).get('epoch')

def load(cond, fs):
    d = f'{ROOT}/{cond}/TEP/typegen_{fs}'
    z = np.load(f'{d}/epoch_scores/epoch_{best_ep(d):03d}_scores.npz')
    return (np.nan_to_num(z['teacher_recon_error']), np.nan_to_num(z['discrepancy_error']),
            np.nan_to_num(z['official_score']))

def zpfx(X):
    out = np.zeros_like(X)
    for r in rt:
        s, e = r['start'], r['end']; p = X[s:s + ONSET]
        out[s:e] = (X[s:e] - p.mean()) / (p.std() + 1e-9)
    return out

RULES = {
    'official': lambda R, D, O: O,
    'raw_w25':  lambda R, D, O: R + 25 * D,
    'zpfx_w1':  lambda R, D, O: zpfx(R) + zpfx(D),
}
_, y, _, _, run_table = load_test()

def fpak(score, f):
    m, _, _ = partition_eval(score, y, run_table, {f}, lite=True)
    return m.get('pak_auc_f1', 0.0)

pak = {rn: {'A': {}, 'B': {}} for rn in RULES}
for fs in FOLD_KEYS:
    RA, DA, OA = load('phase2_A', fs); RB, DB, OB = load('phase2_B', fs)
    for rn, fn in RULES.items():
        sA, sB = fn(RA, DA, OA), fn(RB, DB, OB)
        for f in USABLE_FAULTS:
            pak[rn]['A'][(fs, f)] = fpak(sA, f)
            pak[rn]['B'][(fs, f)] = fpak(sB, f)

seen = [(fs, f) for fs, fk in FOLD_KEYS.items() for f in seen_faults(fk)]
unseen = [(fs, f) for fs, fk in FOLD_KEYS.items() for f in unseen_faults(fk)]
print(f"\n=== pak_auc_f1 검증 (per-mode macro + A>B) ===")
print(f"{'rule':10} {'seenA>B':>9} {'A_S':>7} {'B_S':>7} {'dS':>8} {'unsA>B':>8} {'A_U':>7} {'B_U':>7} {'dU':>8}")
for rn in RULES:
    As = [pak[rn]['A'][p] for p in seen]; Bs = [pak[rn]['B'][p] for p in seen]
    Au = [pak[rn]['A'][p] for p in unseen]; Bu = [pak[rn]['B'][p] for p in unseen]
    cs = sum(1 for a, b in zip(As, Bs) if a > b); cu = sum(1 for a, b in zip(Au, Bu) if a > b)
    print(f"{rn:10} {cs:>3}/{len(seen):<5} {np.mean(As):>7.4f} {np.mean(Bs):>7.4f} {np.mean(As)-np.mean(Bs):>+8.4f} "
          f"{cu:>3}/{len(unseen):<4} {np.mean(Au):>7.4f} {np.mean(Bu):>7.4f} {np.mean(Au)-np.mean(Bu):>+8.4f}")

rn = 'raw_w25'
print(f"\n=== '{rn}' per seen-fault pak (vs official) ===")
print(f"{'fold':6} {'IDV':>4} {'A_new':>7} {'B_new':>7} {'d':>8} | {'A_off':>7} {'B_off':>7}")
for fs, f in seen:
    a, b = pak[rn]['A'][(fs, f)], pak[rn]['B'][(fs, f)]
    ao, bo = pak['official']['A'][(fs, f)], pak['official']['B'][(fs, f)]
    print(f"{fs:6} {f:>4} {a:>7.4f} {b:>7.4f} {a-b:>+8.4f} | {ao:>7.4f} {bo:>7.4f}{'' if a>b else '  A<=B'}")

json.dump({rn: {f'{fs}_{f}': {'A': pak[rn]['A'][(fs, f)], 'B': pak[rn]['B'][(fs, f)]} for fs, f in seen + unseen}
           for rn in RULES}, open(f'{ROOT}/validate_redo_pak.json', 'w'), indent=1)
print(f"\n저장: {ROOT}/validate_redo_pak.json")
