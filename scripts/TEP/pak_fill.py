"""pak_fill.py — ALL pak_auc_f1 under train mean-fixed (tmf) for the Table D.1 page.
tmf = recon + (R_tr/D_tr)·disc.  D = recon-only.  Computes A/B/B0/D + noisy per-fault per-fold.
Saves incrementally (core A/B/B0 first) so partial tables are usable. partition_eval = same
metric stack as the page's existing values.

Multi-seed (2026-07-22): --root selects the phase2 results root (default = seed-42 root).
pak_fill.json is written INTO that root. Sections whose inputs are absent under the root
(phase2_B0, noisy/ — not run for the extra-seed roots) are skipped automatically.
  ~/anaconda3/envs/dc_vis/bin/python scripts/TEP/pak_fill.py                       # seed-42 (unchanged)
  ~/anaconda3/envs/dc_vis/bin/python scripts/TEP/pak_fill.py --root results/experiments/TEP_phase2_win100_ep30_s40
"""
import argparse, json, sys, os
import numpy as np
sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import seen_faults, unseen_faults, USABLE_FAULTS, ALL_FAULTS, FAMILY
from run_tep_simple import partition_eval, load_test

DEFAULT_ROOT = 'results/experiments/TEP_phase2_win100_ep30'
ap = argparse.ArgumentParser()
ap.add_argument('--root', default=DEFAULT_ROOT,
                help='phase2 results root (pak_fill.json is created inside it)')
args = ap.parse_args()
ROOT = args.root
FOLD_KEYS = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}
W_OFF = 0.25
rt = json.load(open('scripts/TEP/data/test_run_table.json'))
HAVE_B0 = os.path.isdir(f'{ROOT}/phase2_B0')
HAVE_NOISY = os.path.isdir(f'{ROOT}/noisy')

def best_ep(d):
    rows = json.load(open(f'{d}/epoch_metrics.json')).get('epochs', [])
    pw = [r for r in rows if r.get('epoch', 0) > 15] or rows
    return max(pw, key=lambda r: r.get('pak_auc_f1', 0)).get('epoch')

def load(path):
    z = np.load(f'{ROOT}/{path}/epoch_scores/epoch_{best_ep(f"{ROOT}/{path}"):03d}_scores.npz')
    return (np.nan_to_num(z['teacher_recon_error']).astype(np.float64),
            np.nan_to_num(z['discrepancy_error']).astype(np.float64),
            np.nan_to_num(z['official_score']).astype(np.float64))

def ratio(R, D, O):
    cR, cD = np.cumsum(R), np.cumsum(D); m = D > (np.median(D) * 0.5)
    s = (O[m] - R[m]) / (W_OFF * D[m]); A_ = np.stack([np.ones(m.sum()), -s], 1); b_ = s * cD[m] - cR[m]
    sol, *_ = np.linalg.lstsq(A_, b_, rcond=None); return sol[0] / sol[1]

_, y, _, _, run_table = load_test()
def fpak(score, f):
    return partition_eval(score, y, run_table, {f}, lite=True)[0].get('pak_auc_f1', 0.0)

OUT = f'{ROOT}/pak_fill.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(res, open(OUT, 'w'), indent=1)

def do(key, path, scoring, faults):
    if key in res: return
    R, D, O = load(path); rr = ratio(R, D, O)
    s = {'tmf': R + rr * D, 'official': O, 'recon': R}[scoring]
    res[key] = {str(f): fpak(s, f) for f in faults}
    save(); print(f'  done {key} (ratio {rr:.1f})', flush=True)

# --- core: A/B per-fault ALL 20, per fold (§5,§6,§10) + B0 + D ---
print('=== core A/B/B0/D (tmf) ===', flush=True)
for fs in FOLD_KEYS:
    do(f'A_tmf_{fs}', f'phase2_A/TEP/typegen_{fs}', 'tmf', ALL_FAULTS)
    do(f'B_tmf_{fs}', f'phase2_B/TEP/typegen_{fs}', 'tmf', ALL_FAULTS)
    do(f'D_recon_{fs}', f'phase2_A/TEP/typegen_{fs}', 'recon', USABLE_FAULTS)
if HAVE_B0:
    do('B0_tmf', 'phase2_B0/TEP/typegen_ffonly', 'tmf', ALL_FAULTS)
    do('B0_official', 'phase2_B0/TEP/typegen_ffonly', 'official', ALL_FAULTS)
else:
    print('  (no phase2_B0 under root — B0 skipped)', flush=True)
# --- noisy (§A): USABLE per-fault per fold ---
if HAVE_NOISY:
    print('=== noisy (tmf) ===', flush=True)
    for lab in ['lab80', 'lab50', 'lab25', 'lab10']:
        for fs in FOLD_KEYS:
            do(f'{lab}_tmf_{fs}', f'noisy/TEP/typegen_{fs}_{lab}', 'tmf', USABLE_FAULTS)
else:
    print('=== noisy — skipped (no noisy/ under root) ===', flush=True)

# ============ derive + print fill-ready tables ============
def smac(key, fs): return float(np.mean([res[key][str(f)] for f in seen_faults(FOLD_KEYS[fs])]))
def umac(key, fs): return float(np.mean([res[key][str(f)] for f in unseen_faults(FOLD_KEYS[fs])]))

print('\n## §5 per-mode S/U (tmf pak)')
for cond, kt in [('B0', 'B0_tmf'), ('B', 'B_tmf'), ('A', 'A_tmf'), ('D', 'D_recon')]:
    if cond == 'B0' and 'B0_tmf' not in res:
        continue
    vals = []
    for fs in FOLD_KEYS:
        k = kt if cond in ('B0',) else (kt if cond != 'D' else f'D_recon_{fs}')
        k = 'B0_tmf' if cond == 'B0' else f'{cond}_tmf_{fs}' if cond in ('A', 'B') else f'D_recon_{fs}'
        vals += [f'{smac(k, fs):.3f}', f'{umac(k, fs):.3f}']
    print(f'  {cond}: ' + ' '.join(vals))

print('\n## §7 Δ_unseen (A_U−B_U) per fold')
print('  ' + ' '.join(f'{umac(f"A_tmf_{fs}", fs) - umac(f"B_tmf_{fs}", fs):+.3f}' for fs in FOLD_KEYS))
if 'B0_tmf' in res:
    print('## §6 C_dmg  MAE-B(B0−B) / MAE-A(B0−A) seen, per fold')
    for tag, c in [('MAE-B', 'B'), ('MAE-A', 'A')]:
        print(f'  {tag}: ' + ' '.join(f'{smac("B0_tmf", fs) - smac(f"{c}_tmf_{fs}", fs):+.3f}' for fs in FOLD_KEYS))

if any(f'{lab}_tmf_fstep' in res for lab in ['lab80', 'lab50', 'lab25', 'lab10']):
    print('\n## §A noisy unseen (tmf pak)')
    for lab in ['lab80', 'lab50', 'lab25', 'lab10']:
        if f'{lab}_tmf_fstep' not in res:
            continue
        print(f'  {lab}: ' + ' '.join(f'{umac(f"{lab}_tmf_{fs}", fs):.3f}' for fs in FOLD_KEYS))

print('\n## §10 per-fault (tmf pak): IDV | B0 | A@fstep/frand/fds/funk | B@...')
for f in ALL_FAULTS:
    fam = next((k for k, v in FAMILY.items() if f in v), 'EXCL')
    b0s = f"B0={res['B0_tmf'][str(f)]:.3f}" if 'B0_tmf' in res else 'B0=  -  '
    a = [res[f'A_tmf_{fs}'][str(f)] for fs in FOLD_KEYS]
    b = [res[f'B_tmf_{fs}'][str(f)] for fs in FOLD_KEYS]
    print(f'  IDV{f:>2}({fam[:4]}) {b0s} | A ' + '/'.join(f'{x:.3f}' for x in a) + ' | B ' + '/'.join(f'{x:.3f}' for x in b))
print('\n저장:', OUT)
