"""deep_prep.py — per-fault score decomposition (recon/disc/official) for A/B/B0.

For each run × fault: roc_auc + anomaly/normal mean + normalized separation of
each score component, so the mechanism (collapse = anomaly reconstructed well →
recon_sep≈0; vs A-recovery = purging restores recon_sep) can be read off directly.
Fast (sklearn roc, no metric pool). Output: per_fault_deep.json.
"""
import os, sys, json
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import ALL_FAULTS, FAMILY

ROOT = 'results/experiments/TEP_phase2_win100_ep30'
FOLD_KEYS = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}
FAM_OF = {f: fam for fam, fs in FAMILY.items() for f in fs}

rt = json.load(open('scripts/TEP/data/test_run_table.json'))
def part_idx(f):
    return np.concatenate([np.arange(r['start'], r['end'])
                           for r in rt if r['fault'] == f or r['fault'] == 0])
PIDX = {f: part_idx(f) for f in ALL_FAULTS}

def best_ep(d):
    rows = json.load(open(f'{d}/epoch_metrics.json')).get('epochs', [])
    pw = [r for r in rows if r.get('epoch', 0) > 15] or rows
    return max(pw, key=lambda r: r.get('pak_auc_f1', 0)).get('epoch')

def run_stats(d):
    be = best_ep(d)
    z = np.load(f'{d}/epoch_scores/epoch_{be:03d}_scores.npz')
    comps = {'recon': np.nan_to_num(z['teacher_recon_error']),
             'disc':  np.nan_to_num(z['discrepancy_error']),
             'off':   np.nan_to_num(z['official_score'])}
    y = z['point_labels'].astype(int)
    out = {'best_ep': int(be), 'faults': {}}
    for f in ALL_FAULTS:
        ix = PIDX[f]; yy = y[ix]
        rec = {'pos_rate': float(yy.mean())}
        for nm, sc in comps.items():
            s = sc[ix]
            rec[f'{nm}_roc'] = float(roc_auc_score(yy, s)) if len(set(yy.tolist())) > 1 else float('nan')
            a, n = s[yy == 1], s[yy == 0]
            rec[f'{nm}_anom'] = float(a.mean()); rec[f'{nm}_norm'] = float(n.mean())
            rec[f'{nm}_sep'] = float((a.mean() - n.mean()) / (s.std() + 1e-9))
        out['faults'][str(f)] = rec
    return out

res = {}
for cond, sub in [('A', 'phase2_A'), ('B', 'phase2_B')]:
    for fs in FOLD_KEYS:
        d = f'{ROOT}/{sub}/TEP/typegen_{fs}'
        if os.path.exists(f'{d}/epoch_metrics.json'):
            res[f'{cond}_{fs}'] = run_stats(d)
res['B0'] = run_stats(f'{ROOT}/phase2_B0/TEP/typegen_ffonly')

json.dump(res, open(f'{ROOT}/per_fault_deep.json', 'w'), indent=1)
print(f'saved {len(res)} runs → {ROOT}/per_fault_deep.json')
# quick sanity: recon_sep for a known collapse (IDV4 fstep) vs recovery (IDV14 fds)
for k, f in [('A_fstep', '4'), ('B_fstep', '4'), ('A_fds', '14'), ('B_fds', '14')]:
    r = res[k]['faults'][f]
    print(f"{k} IDV{f}: recon_roc={r['recon_roc']:.3f} recon_sep={r['recon_sep']:.3f} "
          f"disc_roc={r['disc_roc']:.3f} off_roc={r['off_roc']:.3f}")
