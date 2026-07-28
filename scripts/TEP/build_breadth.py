"""build_breadth.py — analyze the Label-Breadth sweep (handles partial/full data).
For each held-out (unseen) family, compute its detection (macro per-mode pak, train mean-fixed) as a
function of k = #labeled families among the 3 fixed train families (k=0,1,2 from breadth dirs;
k=3 = LOFO-A). Subset-averaged at k=1,2. Tests the TRUE hypothesis: more labeled TYPES -> better unseen.
"""
import json, glob, sys, os, itertools
import numpy as np
sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import USABLE_FAULTS
from run_tep_simple import partition_eval, load_test

R = 'results/experiments/TEP_phase2_win100_ep30'
FAM_FAULTS = {'step': [1, 2, 4, 5, 6, 7], 'rand': [8, 10, 11, 12], 'ds': [13, 14], 'unk': [16, 17, 18, 19, 20]}
USABLE = set(USABLE_FAULTS)

def train_ratio(d):
    f = glob.glob(f'{d}/best_epoch_train_scores.npz')
    if not f: return None
    z = np.load(f[0]); rec = np.nan_to_num(z['teacher_recon_error']); dis = np.nan_to_num(z['discrepancy_error']); lab = z['point_labels']
    m = lab == 0
    return float(rec[m].mean() / dis[m].mean())

def best_ep(d):
    rows = json.load(open(f'{d}/epoch_metrics.json')).get('epochs', [])
    pw = [r for r in rows if r.get('epoch', 0) > 15] or rows
    return max(pw, key=lambda r: r.get('pak_auc_f1', 0)).get('epoch')

_, y, _, _, run_table = load_test()
def fold_SU(d, ho):
    """held-out family macro pak (U, unseen) + seen-3 macro pak (S), train mean-fixed."""
    rr = train_ratio(d)
    if rr is None or not os.path.exists(f'{d}/epoch_scores'): return None
    try:
        z = np.load(f'{d}/epoch_scores/epoch_{best_ep(d):03d}_scores.npz')
    except Exception:
        return None
    score = np.nan_to_num(z['teacher_recon_error']).astype(np.float64) + rr * np.nan_to_num(z['discrepancy_error']).astype(np.float64)
    hf = [f for f in FAM_FAULTS[ho] if f in USABLE]
    sf = [f for fam in ('step', 'rand', 'ds', 'unk') if fam != ho for f in FAM_FAULTS[fam] if f in USABLE]
    pak = lambda f: partition_eval(score, y, run_table, {f}, lite=True)[0].get('pak_auc_f1', 0.0)
    return {'U': float(np.mean([pak(f) for f in hf])), 'S': float(np.mean([pak(f) for f in sf])), 'ratio': rr}

results = {}
for ho in ['unk', 'ds', 'rand', 'step']:
    tf = [f for f in ('step', 'rand', 'ds', 'unk') if f != ho]
    perk = {0: [], 1: [], 2: [], 3: []}
    detail = {}
    for k in (0, 1, 2):
        for sub in itertools.combinations(tf, k):
            tag = 'k0' if k == 0 else '-'.join(sub)
            d = f'{R}/breadth/TEP/typegen_breadth_{ho}_{tag}'
            su = fold_SU(d, ho)
            if su: perk[k].append(su['U']); detail[tag] = su
    su = fold_SU(f'{R}/lofo/TEP/typegen_lofo_{ho}', ho)   # k=3 = LOFO-A
    if su: perk[3].append(su['U']); detail['k3_lofo'] = su
    results[ho] = {'U_by_k': {k: (float(np.mean(v)) if v else None) for k, v in perk.items()},
                   'n_by_k': {k: len(v) for k, v in perk.items()}, 'detail': detail}

print('=== held-out (UNSEEN) detection vs k = #labeled families (train data FIXED) ===')
print(f"{'held-out':>8} {'k=0':>9} {'k=1':>9} {'k=2':>9} {'k=3':>9}   slope(k3-k0)")
for ho in ['step', 'rand', 'ds', 'unk']:
    v = [results[ho]['U_by_k'][k] for k in (0, 1, 2, 3)]
    s = '  '.join(f'{x:.4f}' if x is not None else '  ---  ' for x in v)
    sl = f'{v[3]-v[0]:+.4f}' if (v[0] is not None and v[3] is not None) else '?'
    print(f'{ho:>8} {s}   {sl}')
print('\n=== pooled over held-outs (mean) — THE hypothesis test ===')
pooled = {}
for k in (0, 1, 2, 3):
    vs = [results[ho]['U_by_k'][k] for ho in ['step', 'rand', 'ds', 'unk'] if results[ho]['U_by_k'][k] is not None]
    pooled[k] = float(np.mean(vs)) if vs else None
    print(f'  k={k}: ' + (f'{pooled[k]:.4f}  (n={len(vs)} held-outs)' if vs else '---'))
if all(pooled[k] is not None for k in (0, 1, 2, 3)):
    mono = all(pooled[k + 1] >= pooled[k] - 0.003 for k in range(3))
    print(f"\n  monotone↑ (more labeled types -> better unseen)? {'YES' if mono else 'NO'}  | k3-k0 = {pooled[3]-pooled[0]:+.4f}")
json.dump({'results': results, 'pooled': pooled}, open(f'{R}/breadth_results.json', 'w'), indent=1)
print('\n저장: breadth_results.json')
