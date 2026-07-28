"""build_noisy_vus.py — noisy label sweep (lab80/50/25/10) unseen VUS-PR (train mean-fixed, lite=False).
To check whether substituting a partial-label result for A keeps the paper logic under VUS-PR too."""
import json, glob, sys
import numpy as np
sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import unseen_faults, USABLE_FAULTS
from run_tep_simple import partition_eval, load_test

R = 'results/experiments/TEP_phase2_win100_ep30'
FK = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}
USABLE = set(USABLE_FAULTS)
def train_ratio(d):
    f = glob.glob(f'{d}/best_epoch_train_scores.npz')
    z = np.load(f[0]); rec = np.nan_to_num(z['teacher_recon_error']); dis = np.nan_to_num(z['discrepancy_error']); lab = z['point_labels']
    m = lab == 0; return float(rec[m].mean() / dis[m].mean())
def best_ep(d):
    rows = json.load(open(f'{d}/epoch_metrics.json')).get('epochs', [])
    pw = [r for r in rows if r.get('epoch', 0) > 15] or rows
    return max(pw, key=lambda r: r.get('pak_auc_f1', 0)).get('epoch')
_, y, _, _, run_table = load_test()
def uvus(d, fk):
    rr = train_ratio(d)
    z = np.load(f'{d}/epoch_scores/epoch_{best_ep(d):03d}_scores.npz')
    s = np.nan_to_num(z['teacher_recon_error']).astype(np.float64) + rr * np.nan_to_num(z['discrepancy_error']).astype(np.float64)
    uf = [f for f in unseen_faults(FK[fk]) if f in USABLE]
    return float(np.mean([partition_eval(s, y, run_table, {f}, lite=False)[0].get('vus_pr', 0.0) for f in uf]))

res = {}
for lab in ['lab80', 'lab50', 'lab25', 'lab10']:
    res[lab] = {fk: uvus(f'{R}/noisy/TEP/typegen_{fk}_{lab}', fk) for fk in FK}
    print(f'  {lab}: ' + ' '.join(f'{res[lab][fk]:.4f}' for fk in FK) + f'  mean {np.mean(list(res[lab].values())):.4f}', flush=True)
# A, B unseen VUS-PR from vus_results
vus = json.load(open(f'{R}/vus_results.json'))
def uvus_cond(pref, fk):
    key = f'{pref}_{fk}'
    return float(np.mean([vus[key][str(f)] for f in unseen_faults(FK[fk]) if f in USABLE]))
resA = {fk: uvus_cond('A', fk) for fk in FK}
resB = {fk: uvus_cond('B', fk) for fk in FK}
json.dump({'noisy_vus': res, 'A': resA, 'B': resB}, open(f'{R}/noisy_vus.json', 'w'), indent=1)
print(f'\n  A(100%): ' + ' '.join(f'{resA[fk]:.4f}' for fk in FK) + f'  mean {np.mean(list(resA.values())):.4f}')
print(f'  B(0%):   ' + ' '.join(f'{resB[fk]:.4f}' for fk in FK) + f'  mean {np.mean(list(resB.values())):.4f}')
print('\n저장: noisy_vus.json')
