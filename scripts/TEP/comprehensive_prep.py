"""comprehensive_prep.py — full per-fault × per-fold × per-condition × per-scoring prc table.

Enables seen/unseen/per-mode/Δ_unseen/spillover analysis under official AND improved
(causal) scoring. Core for validating the MAE claim: A improves UNSEEN (H1) vs H2.
prc only (fast, no pool). Output: comprehensive_prc.json.
"""
import json, sys
import numpy as np
from sklearn.metrics import average_precision_score as ap

sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import seen_faults, unseen_faults, USABLE_FAULTS, FAMILY

ROOT = 'results/experiments/TEP_phase2_win100_ep30'
FOLD_KEYS = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}
FAM = {f: fam for fam, fs in FAMILY.items() for f in fs}
NEAR = {4: 11, 11: 4, 5: 12, 12: 5}
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
    import os
    if not os.path.exists(f'{d}/epoch_metrics.json'):
        return None
    z = np.load(f'{d}/epoch_scores/epoch_{best_ep(d):03d}_scores.npz')
    return (np.nan_to_num(z['teacher_recon_error']), np.nan_to_num(z['discrepancy_error']),
            np.nan_to_num(z['official_score']), z['point_labels'].astype(int))

def zpfx(X):
    out = np.zeros_like(X)
    for r in rt:
        s, e = r['start'], r['end']; p = X[s:s + ONSET]
        out[s:e] = (X[s:e] - p.mean()) / (p.std() + 1e-9)
    return out

SCOR = {
    'official': lambda R, D, O: O,
    'raw_w25':  lambda R, D, O: R + 25 * D,
    'zpfx':     lambda R, D, O: zpfx(R) + zpfx(D),
    'recon':    lambda R, D, O: R,
    'disc':     lambda R, D, O: D,
}

# conditions: (label, output-subdir-template per fold). B0 fold-independent (ffonly).
COND = {
    'A':  lambda fs: f'phase2_A/TEP/typegen_{fs}',
    'B':  lambda fs: f'phase2_B/TEP/typegen_{fs}',
    'lab80': lambda fs: f'noisy/TEP/typegen_{fs}_lab80',
    'lab50': lambda fs: f'noisy/TEP/typegen_{fs}_lab50',
    'lab25': lambda fs: f'noisy/TEP/typegen_{fs}_lab25',
    'lab10': lambda fs: f'noisy/TEP/typegen_{fs}_lab10',
}

# prc[cond][scor][fold][fault]
data = {}
for cond, tmpl in COND.items():
    data[cond] = {sc: {} for sc in SCOR}
    for fs in FOLD_KEYS:
        sub = tmpl(fs).rsplit('/', 2)[0] + '/' + '/'.join(tmpl(fs).rsplit('/', 2)[1:])
        loaded = load('/'.join(tmpl(fs).split('/')[:-2]) if False else tmpl(fs).rsplit('/TEP/', 1)[0], None) if False else None
        # simpler: load by full path
        import os
        d = f'{ROOT}/{tmpl(fs)}'
        if not os.path.exists(f'{d}/epoch_metrics.json'):
            continue
        z = np.load(f'{d}/epoch_scores/epoch_{best_ep(d):03d}_scores.npz')
        R, Dd, O, y = (np.nan_to_num(z['teacher_recon_error']), np.nan_to_num(z['discrepancy_error']),
                       np.nan_to_num(z['official_score']), z['point_labels'].astype(int))
        for sc, fn in SCOR.items():
            s = fn(R, Dd, O)
            data[cond][sc][fs] = {str(f): float(ap(y[PIDX[f]], s[PIDX[f]])) for f in USABLE_FAULTS}

# B0 (ffonly, fold-independent scores) + D (= A teacher_recon = recon scoring of A)
import os
d0 = f'{ROOT}/phase2_B0/TEP/typegen_ffonly'
if os.path.exists(f'{d0}/epoch_metrics.json'):
    z = np.load(f'{d0}/epoch_scores/epoch_{best_ep(d0):03d}_scores.npz')
    R, Dd, O, y = (np.nan_to_num(z['teacher_recon_error']), np.nan_to_num(z['discrepancy_error']),
                   np.nan_to_num(z['official_score']), z['point_labels'].astype(int))
    data['B0'] = {sc: {fs: {str(f): float(ap(y[PIDX[f]], fn(R, Dd, O)[PIDX[f]])) for f in USABLE_FAULTS}
                       for fs in FOLD_KEYS} for sc, fn in SCOR.items()}
# D = A's recon-only (already captured as data['A']['recon'])
data['D'] = {'recon-only': data['A']['recon']}

# ---- derive per-mode S/U for each cond×scor ----
def macro(cond, sc, which):
    out = {}
    if cond not in data or sc not in data[cond]:
        return out
    for fs, fk in FOLD_KEYS.items():
        if fs not in data[cond][sc]:
            continue
        faults = seen_faults(fk) if which == 'S' else unseen_faults(fk)
        out[fs] = float(np.mean([data[cond][sc][fs][str(f)] for f in faults]))
    return out

summary = {}
for cond in data:
    summary[cond] = {}
    for sc in data[cond]:
        S = macro(cond, sc, 'S'); U = macro(cond, sc, 'U')
        summary[cond][sc] = {
            'S_perfold': S, 'U_perfold': U,
            'S_macro': float(np.mean(list(S.values()))) if S else None,
            'U_macro': float(np.mean(list(U.values()))) if U else None}

# ---- spillover: each fault's prc across folds (seen vs unseen), official + raw_w25 ----
spill = {}
for f in USABLE_FAULTS:
    spill[str(f)] = {'family': FAM.get(f), 'near_pair': NEAR.get(f)}
    for cond in ['A', 'B', 'B0']:
        for sc in ['official', 'raw_w25']:
            if cond in data and sc in data[cond]:
                spill[str(f)][f'{cond}_{sc}'] = {fs: data[cond][sc].get(fs, {}).get(str(f)) for fs in FOLD_KEYS}

out = {'prc': data, 'summary': summary, 'spillover': spill}
json.dump(out, open(f'{ROOT}/comprehensive_prc.json', 'w'), indent=1)

# ---- print headline: Δ_unseen A vs B under official vs raw_w25 ----
print("=== Δ_unseen = A_U − B_U  (per-mode macro) — H1 검증의 핵심 ===")
for sc in ['official', 'raw_w25', 'zpfx', 'recon', 'disc']:
    au = summary['A'][sc]['U_macro']; bu = summary['B'][sc]['U_macro']
    as_ = summary['A'][sc]['S_macro']; bs = summary['B'][sc]['S_macro']
    print(f"  scoring={sc:9}: A_U={au:.4f} B_U={bu:.4f} Δ_unseen={au-bu:+.4f}  |  A_S={as_:.4f} B_S={bs:.4f} ΔS={as_-bs:+.4f}")
print(f"\n저장: {ROOT}/comprehensive_prc.json (conditions={list(data.keys())})")
