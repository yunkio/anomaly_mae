"""build_nogrl.py — legacy PAK per-mode S/U for the w/o-GRL diagnostic.

Paper v22 TEP Table 3 uses VUS-PR; this script computes pak_auc_f1 only.
nogrl = condition A with use_grl=False + anomaly_loss_weight=0.0 (student loss=0 on labeled anomalies).
Computes per-fault pak under tmf (train-based R_tr/D_tr), then per-mode S/U per fold. Prints alongside
A / B / B0 / D for the GRL-contribution analysis.
"""
import json, glob, sys
import numpy as np
sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import seen_faults, unseen_faults, USABLE_FAULTS
from run_tep_simple import partition_eval, load_test

R = 'results/experiments/TEP_phase2_win100_ep30'
FK = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}

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
def per_fault(d, scoring='tmf'):
    rr = train_ratio(d)
    z = np.load(f'{d}/epoch_scores/epoch_{best_ep(d):03d}_scores.npz')
    rec = np.nan_to_num(z['teacher_recon_error']).astype(np.float64)
    dis = np.nan_to_num(z['discrepancy_error']).astype(np.float64)
    score = rec if scoring == 'recon' else rec + rr * dis
    return {f: partition_eval(score, y, run_table, {f}, lite=True)[0].get('pak_auc_f1', 0.0) for f in USABLE_FAULTS}, rr

res = {}
for fk in FK:
    paks, rr = per_fault(f'{R}/phase2_nogrl/TEP/typegen_{fk}')
    S = float(np.mean([paks[f] for f in seen_faults(FK[fk])]))
    U = float(np.mean([paks[f] for f in unseen_faults(FK[fk])]))
    res[fk] = {'S': S, 'U': U, 'ratio': rr, 'paks': {str(f): paks[f] for f in USABLE_FAULTS}}
json.dump(res, open(f'{R}/nogrl_results.json', 'w'), indent=1)

# compare to A/B/B0/D from table4_data.json (its keys are 'f_step'... = FK[fk])
t4 = json.load(open(f'{R}/table4_data.json'))['mae']
def t4v(cond, fk, m): return t4[cond][FK[fk]][m]
print('=== Legacy TEP PAK diagnostic: MAE conditions S/U per fold — incl. w/o-GRL ===')
print(f"{'condition':<16} " + '  '.join(f'{fk[1:].upper():>13}' for fk in FK) + '   Mean S/U')
for cond, lab in [('B0', 'Clean(upper)'), ('B', 'Unlabeled'), ('A', 'LASAD(ours)'), ('nogrl', 'w/o-GRL'), ('D', 'Recon-only')]:
    cells = []
    Ss, Us = [], []
    for fk in FK:
        if cond == 'nogrl':
            S, U = res[fk]['S'], res[fk]['U']
        else:
            S, U = t4v(cond, fk, 'S'), t4v(cond, fk, 'U')
        Ss.append(S); Us.append(U); cells.append(f'{S:.3f}/{U:.3f}')
    print(f'{lab:<16} ' + '  '.join(cells) + f'   {np.mean(Ss):.3f}/{np.mean(Us):.3f}')

print('\n=== GRL ablation: A − (w/o-GRL) = GRL push의 기여 (per fold, S / U) ===')
for fk in FK:
    dS = t4v('A', fk, 'S') - res[fk]['S']; dU = t4v('A', fk, 'U') - res[fk]['U']
    print(f'  {fk[1:].upper():>6}: ΔS={dS:+.4f}  ΔU={dU:+.4f}  (nogrl ratio={res[fk]["ratio"]:.1f})')
mS = np.mean([t4v('A', fk, 'S') - res[fk]['S'] for fk in FK])
mU = np.mean([t4v('A', fk, 'U') - res[fk]['U'] for fk in FK])
print(f'  MEAN  : ΔS={mS:+.4f}  ΔU={mU:+.4f}')
print('\n=== w/o-GRL vs Unlabeled(B) = "무시" vs "흡수" (per fold U) ===')
for fk in FK:
    print(f'  {fk[1:].upper():>6}: w/o-GRL U={res[fk]["U"]:.4f}  B U={t4v("B",fk,"U"):.4f}  Δ(nogrl−B)={res[fk]["U"]-t4v("B",fk,"U"):+.4f}')
print('\n저장: nogrl_results.json')
