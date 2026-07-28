"""build_shapley.py — fair attribution controlling for WHICH family is labeled.
Shapley value of each train family = its average marginal contribution to held-out (unseen) detection
over ALL coalitions (subsets) of the other families. Uses the full 2^3 breadth subset grid per held-out.
Sum of Shapley = U(all labeled) - U(none labeled). All families' Shapley>0 ⇔ "labeling any type helps
unseen on avg (controlling for which)" = hypothesis supported.
"""
import json, glob, sys, os, itertools
from math import factorial
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
def U_of(d, ho):
    if not os.path.exists(f'{d}/epoch_metrics.json') or not os.path.exists(f'{d}/epoch_scores'): return None
    rr = train_ratio(d)
    if rr is None: return None
    try:
        z = np.load(f'{d}/epoch_scores/epoch_{best_ep(d):03d}_scores.npz')
    except Exception:
        return None
    score = np.nan_to_num(z['teacher_recon_error']).astype(np.float64) + rr * np.nan_to_num(z['discrepancy_error']).astype(np.float64)
    hf = [f for f in FAM_FAULTS[ho] if f in USABLE]
    return float(np.mean([partition_eval(score, y, run_table, {f}, lite=True)[0].get('pak_auc_f1', 0.0) for f in hf]))

def dir_for(ho, sub, tf):
    if len(sub) == 3: return f'{R}/lofo/TEP/typegen_lofo_{ho}'
    tag = 'k0' if len(sub) == 0 else '-'.join(f for f in tf if f in sub)   # tf-ordered (matches registration)
    return f'{R}/breadth/TEP/typegen_breadth_{ho}_{tag}'

def shapley(v, players):
    n = len(players); sh = {}
    for X in players:
        others = [p for p in players if p != X]; s = 0.0
        for k in range(len(others) + 1):
            for S in itertools.combinations(others, k):
                Sf, SXf = frozenset(S), frozenset(S + (X,))
                if Sf not in v or SXf not in v: return None
                s += factorial(k) * factorial(n - k - 1) / factorial(n) * (v[SXf] - v[Sf])
        sh[X] = s
    return sh

allsh = {}
for ho in ['unk', 'ds', 'rand', 'step']:
    tf = [f for f in ('step', 'rand', 'ds', 'unk') if f != ho]
    v = {}
    for k in range(4):
        for sub in itertools.combinations(tf, k):
            u = U_of(dir_for(ho, sub, tf), ho)
            if u is not None: v[frozenset(sub)] = u
    have = len(v); print(f'\n=== held-out={ho}: {have}/8 subsets available ===')
    # marginal table (available)
    print('  marginal ΔU of adding each family (per coalition):')
    for X in tf:
        others = [p for p in tf if p != X]
        for k in range(len(others) + 1):
            for S in itertools.combinations(others, k):
                Sf, SXf = frozenset(S), frozenset(S + (X,))
                if Sf in v and SXf in v:
                    print(f'    +{X:<5} | given {{{",".join(S) or "—"}}}: {v[SXf]-v[Sf]:+.4f}  (U {v[Sf]:.4f}→{v[SXf]:.4f})')
    sh = shapley(v, tf)
    if sh:
        allsh[ho] = sh
        tot = v[frozenset(tf)] - v[frozenset()]
        print(f'  >>> Shapley (fair contribution to unseen): ' + '  '.join(f'{k}={val:+.4f}' for k, val in sh.items()))
        print(f'      sum={sum(sh.values()):+.4f} = U(all){v[frozenset(tf)]:.4f} − U(none){v[frozenset()]:.4f} = {tot:+.4f}')
        print(f'      all positive? {"YES (labeling any type helps unseen, controlling which)" if all(x > 0 for x in sh.values()) else "NO — manifold-dependent"}')
    else:
        print('  (full Shapley pending — need all 8 subsets)')

if allsh:
    print('\n=== POOLED Shapley over completed held-outs ===')
    fams = ['step', 'rand', 'ds', 'unk']
    for f in fams:
        vals = [allsh[ho][f] for ho in allsh if f in allsh[ho]]
        if vals: print(f'  {f}: mean Shapley {np.mean(vals):+.4f} (n={len(vals)})')
    json.dump({ho: allsh[ho] for ho in allsh}, open(f'{R}/shapley.json', 'w'), indent=1)
    print('\n저장: shapley.json')
