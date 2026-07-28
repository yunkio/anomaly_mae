"""build_vus_verify.py — verify the 3 LASAD claims hold under VUS-PR (from vus_results.json).
Computes per-mode S/U (A/B/B0/D/nogrl), Δ_unseen, Δ_gap, GRL-ablation, Shapley — all under VUS-PR —
and compares the SIGN/conclusion to the pak results."""
import json, itertools
from math import factorial
import numpy as np
import sys
sys.path.insert(0, 'scripts/TEP')
from tep_common import seen_faults, unseen_faults, USABLE_FAULTS
R = 'results/experiments/TEP_phase2_win100_ep30'
v = json.load(open(f'{R}/vus_results.json'))
FK = ['fstep', 'frand', 'fds', 'funk']; FKR = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}
FAM_FAULTS = {'step': [1, 2, 4, 5, 6, 7], 'rand': [8, 10, 11, 12], 'ds': [13, 14], 'unk': [16, 17, 18, 19, 20]}
USABLE = set(USABLE_FAULTS)

def SU(key, fk):
    S = np.mean([v[key][str(f)] for f in seen_faults(FKR[fk])])
    U = np.mean([v[key][str(f)] for f in unseen_faults(FKR[fk])])
    return float(S), float(U)
def Bkey_SU(fk):  # B0 fold-indep
    S = np.mean([v['B0'][str(f)] for f in seen_faults(FKR[fk])]); U = np.mean([v['B0'][str(f)] for f in unseen_faults(FKR[fk])])
    return float(S), float(U)

print('=== VUS-PR: Table-4 conditions per-mode S/U ===')
print(f"{'cond':<10}" + '  '.join(f'{fk[1:].upper():>13}' for fk in FK) + '   Mean S/U')
rows = {}
for cond in ['B0', 'B', 'A', 'nogrl', 'D']:
    cells, Ss, Us = [], [], []
    for fk in FK:
        S, U = Bkey_SU(fk) if cond == 'B0' else SU(f'{cond}_{fk}', fk)
        Ss.append(S); Us.append(U); cells.append(f'{S:.3f}/{U:.3f}')
    rows[cond] = (Ss, Us)
    print(f'{cond:<10}' + '  '.join(cells) + f'   {np.mean(Ss):.3f}/{np.mean(Us):.3f}')

print('\n=== claim (a)/(c): Δ_unseen = A_U − B_U (VUS-PR) ===')
dU = [rows['A'][1][i] - rows['B'][1][i] for i in range(4)]
dgap = [(rows['A'][0][i] - rows['A'][1][i]) - (rows['B'][0][i] - rows['B'][1][i]) for i in range(4)]
print('  Δ_unseen: ' + '  '.join(f'{x:+.4f}' for x in dU) + f'  mean {np.mean(dU):+.4f}  | pak mean +0.0188')
print('  Δ_gap   : ' + '  '.join(f'{x:+.4f}' for x in dgap))
print(f'  → 4 fold 전부 양수? {all(x > 0 for x in dU)}  (claim a/c {"유지" if all(x>0 for x in dU) else "불일치"})')

print('\n=== claim (b): A vs D (discrepancy) + w/o-GRL 분해 (VUS-PR) ===')
adU = [rows['A'][1][i] - rows['D'][1][i] for i in range(4)]; adS = [rows['A'][0][i] - rows['D'][0][i] for i in range(4)]
print(f'  A−D (discrepancy): seen mean {np.mean(adS):+.4f}  unseen mean {np.mean(adU):+.4f}  | pak seen+0.151/unseen+0.031')
push_U = [rows['A'][1][i] - rows['nogrl'][1][i] for i in range(4)]; push_S = [rows['A'][0][i] - rows['nogrl'][0][i] for i in range(4)]
absb_U = [rows['nogrl'][1][i] - rows['B'][1][i] for i in range(4)]
print(f'  GRL-push (A−w/oGRL): seen mean {np.mean(push_S):+.4f}  unseen mean {np.mean(push_U):+.4f}  | pak S+0.025/U+0.012')
print(f'  흡수방지 (w/oGRL−B): unseen mean {np.mean(absb_U):+.4f}  | pak +0.007')

print('\n=== claim (c)+bound: Shapley per family (VUS-PR) ===')
def shapley(vf, players):
    n = len(players); sh = {}
    for X in players:
        others = [p for p in players if p != X]; s = 0.0
        for k in range(len(others) + 1):
            for S in itertools.combinations(others, k):
                Sf, SXf = frozenset(S), frozenset(S + (X,))
                if Sf not in vf or SXf not in vf: return None
                s += factorial(k) * factorial(n - k - 1) / factorial(n) * (vf[SXf] - vf[Sf])
        sh[X] = s
    return sh
allsh = {}
for ho in ['unk', 'ds', 'rand', 'step']:
    tf = [f for f in ('step', 'rand', 'ds', 'unk') if f != ho]
    ho_f = [f for f in FAM_FAULTS[ho] if f in USABLE]
    vf = {}
    for k in range(4):
        for sub in itertools.combinations(tf, k):
            tag = 'lofo' if k == 3 else ('k0' if k == 0 else '-'.join(f for f in tf if f in sub))
            key = f'br_{ho}_{tag}'
            if key in v: vf[frozenset(sub)] = np.mean([v[key][str(f)] for f in ho_f])
    sh = shapley(vf, tf)
    if sh: allsh[ho] = sh; print(f'  held-out={ho}: ' + '  '.join(f'{k}={x:+.4f}' for k, x in sh.items()))
print('  pooled mean Shapley:')
for f in ['step', 'rand', 'ds', 'unk']:
    vals = [allsh[ho][f] for ho in allsh if f in allsh[ho]]
    if vals: print(f'    {f}: {np.mean(vals):+.4f}  | pak {{step:+0.0146,rand:-0.0057,ds:-0.0010,unk:+0.0045}}')
json.dump({'permode': {c: rows[c] for c in rows}, 'dU': dU, 'shapley': allsh}, open(f'{R}/vus_verify.json', 'w'), indent=1)
print('\n저장: vus_verify.json')
