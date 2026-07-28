"""build_extras.py — noisy label sweep (Figure 5 format) + LOFO tables, train mean-fixed.
Noisy: unseen-mode pak per fold per labeled-fraction p (from pak_fill.json, consistent with Table 4).
LOFO: condition-A only (3 seen labeled, 1 held-out novel). tmf via train-based R_tr/D_tr.
  S = 3 seen families' macro pak, U = held-out family's macro pak. non-cont vs cont.
  + main-fold(1-seen) unseen detection of each family for the 3-seen vs 1-seen comparison.
Output: extras_data.json + extras_values.txt.
"""
import json, glob, sys
import numpy as np
sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import seen_faults, unseen_faults, USABLE_FAULTS
from run_tep_simple import partition_eval, load_test

R = 'results/experiments/TEP_phase2_win100_ep30'
FOLDS = ['fstep', 'frand', 'fds', 'funk']
FAM = {'step': [1, 2, 4, 5, 6, 7], 'rand': [8, 10, 11, 12], 'ds': [13, 14], 'unk': [16, 17, 18, 19, 20]}

# ===== NOISY sweep (Figure 5 format) from pak_fill.json =====
pf = json.load(open(f'{R}/pak_fill.json'))
def umac(key, fk):
    fkk = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}[fk]
    return float(np.mean([pf[key][str(f)] for f in unseen_faults(fkk)]))
# train ratio per noisy run (instability mechanism)
def train_ratio(d):
    f = glob.glob(f'{d}/best_epoch_train_scores.npz')
    if not f: return None
    z = np.load(f[0]); rec = np.nan_to_num(z['teacher_recon_error']); dis = np.nan_to_num(z['discrepancy_error']); lab = z['point_labels']
    m = lab == 0
    return float(rec[m].mean() / dis[m].mean())

SWEEP = [('1.00', 'A'), ('0.80', 'lab80'), ('0.50', 'lab50'), ('0.25', 'lab25'), ('0.10', 'lab10'), ('0.00', 'B')]
noisy = {'p_rows': [], 'ratio_rows': []}
for p, tag in SWEEP:
    row = {'p': p, 'cond': tag}
    for fk in FOLDS:
        key = ('A_tmf_' + fk if tag == 'A' else 'B_tmf_' + fk if tag == 'B' else f'{tag}_tmf_{fk}')
        row[fk] = umac(key, fk)
    row['mean'] = float(np.mean([row[fk] for fk in FOLDS]))
    noisy['p_rows'].append(row)
    # ratio per fold
    rr = {'p': p}
    for fk in FOLDS:
        if tag == 'A': rr[fk] = train_ratio(f'{R}/phase2_A/TEP/typegen_{fk}')
        elif tag == 'B': rr[fk] = train_ratio(f'{R}/phase2_B/TEP/typegen_{fk}')
        else: rr[fk] = train_ratio(f'{R}/noisy/TEP/typegen_{fk}_{tag}')
    noisy['ratio_rows'].append(rr)

# ===== LOFO (condition A: 3 seen labeled, 1 held-out novel) =====
_, y, _, _, run_table = load_test()
def best_ep(d):
    rows = json.load(open(f'{d}/epoch_metrics.json')).get('epochs', [])
    pw = [r for r in rows if r.get('epoch', 0) > 15] or rows
    return max(pw, key=lambda r: r.get('pak_auc_f1', 0)).get('epoch')
def lofo_paks(d):
    rr = train_ratio(d)
    z = np.load(f'{d}/epoch_scores/epoch_{best_ep(d):03d}_scores.npz')
    rec = np.nan_to_num(z['teacher_recon_error']).astype(np.float64)
    dis = np.nan_to_num(z['discrepancy_error']).astype(np.float64)
    score = rec + rr * dis
    return {f: partition_eval(score, y, run_table, {f}, lite=True)[0].get('pak_auc_f1', 0.0) for f in USABLE_FAULTS}, rr

# main-fold unseen detection of family X = avg of X's faults' pak over the 3 main folds where X is unseen
fkk = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}
SELF_FOLD = {'step': 'fstep', 'rand': 'frand', 'ds': 'fds', 'unk': 'funk'}
def main_unseen_U(fam):
    fs = [f for f in FAM[fam] if f in USABLE_FAULTS]
    vals = []
    for fk in FOLDS:
        if fk == SELF_FOLD[fam]: continue  # skip the fold where fam is seen
        vals += [pf[f'A_tmf_{fk}'][str(f)] for f in fs]
    return float(np.mean(vals))

lofo = {}
for ho in ['step', 'rand', 'ds', 'unk']:
    seen_fams = [f for f in ('step', 'rand', 'ds', 'unk') if f != ho]
    seen_fl = [f for fam in seen_fams for f in FAM[fam] if f in USABLE_FAULTS]
    ho_fl = [f for f in FAM[ho] if f in USABLE_FAULTS]
    entry = {}
    for variant, suff in [('A', ''), ('A_cont', '_cont')]:
        paks, rr = lofo_paks(f'{R}/lofo/TEP/typegen_lofo_{ho}{suff}')
        S = float(np.mean([paks[f] for f in seen_fl]))
        U = float(np.mean([paks[f] for f in ho_fl]))
        entry[variant] = {'S': S, 'U': U, 'ratio': rr, 'paks': {str(f): paks[f] for f in USABLE_FAULTS}}
    entry['main_unseen_U'] = main_unseen_U(ho)  # 1-seen baseline for held-out family
    lofo[ho] = entry

json.dump({'noisy': noisy, 'lofo': lofo}, open(f'{R}/extras_data.json', 'w'), indent=1)

# ===== fill-ready text =====
L = ['### NOISY label sweep — unseen-mode pak (Figure 5 format)',
     'p(labeled)  F-STEP   F-RAND   F-DS     F-UNK    Mean   (cond)']
for row in noisy['p_rows']:
    L.append(f"  {row['p']}     " + '  '.join(f'{row[fk]:.4f}' for fk in FOLDS) + f"  {row['mean']:.4f}  ({row['cond']})")
L.append('\n  train R_tr/D_tr per p (instability mechanism):')
for rr in noisy['ratio_rows']:
    L.append(f"  {rr['p']}     " + '  '.join(f'{rr[fk]:.1f}' for fk in FOLDS))
L.append('\n### LOFO (3 seen labeled / 1 held-out novel) — condition A')
L.append('held-out | Seen-3 S | LOFO-A U(held-out) | LOFO-A-cont U | main 1-seen U | Δ(3seen−1seen)')
for ho in ['step', 'rand', 'ds', 'unk']:
    e = lofo[ho]
    d3 = e['A']['U'] - e['main_unseen_U']
    L.append(f"  {ho:<6} | {e['A']['S']:.4f}  | {e['A']['U']:.4f}          | {e['A_cont']['U']:.4f}      | {e['main_unseen_U']:.4f}     | {d3:+.4f}")
L.append('\n  LOFO train ratios: ' + '  '.join(f"{ho}={lofo[ho]['A']['ratio']:.1f}/cont {lofo[ho]['A_cont']['ratio']:.1f}" for ho in ['step', 'rand', 'ds', 'unk']))
txt = '\n'.join(L)
open(f'{R}/extras_values.txt', 'w').write(txt)
print(txt)
print('\n저장: extras_data.json, extras_values.txt')
