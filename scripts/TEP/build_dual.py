"""build_dual.py — legacy side-by-side PAK/VUS audit for historical TEP artifacts.

Paper v22 Table 3 uses VUS-PR only; the PAK block is diagnostic.
"""
import json, numpy as np, sys, itertools
from math import factorial
sys.path.insert(0, 'scripts/TEP')
from tep_common import seen_faults, unseen_faults, USABLE_FAULTS
R = 'results/experiments/TEP_phase2_win100_ep30'
t4 = json.load(open(f'{R}/table4_data.json')); nogrl = json.load(open(f'{R}/nogrl_results.json'))
vv = json.load(open(f'{R}/vus_verify.json')); vus = json.load(open(f'{R}/vus_results.json'))
hom = json.load(open(f'{R}/homogeneity.json'))
FK = ['fstep', 'frand', 'fds', 'funk']; FKR = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}
FAM_FAULTS = {'step': [1, 2, 4, 5, 6, 7], 'rand': [8, 10, 11, 12], 'ds': [13, 14], 'unk': [16, 17, 18, 19, 20]}
USABLE = set(USABLE_FAULTS)
out = []
def P(*a): out.append(' '.join(str(x) for x in a))

def vSU(key_prefix, fk):
    if key_prefix == 'B0':
        k = 'B0'
    else:
        k = f'{key_prefix}_{fk}'
    S = np.mean([vus[k][str(f)] for f in seen_faults(FKR[fk])]); U = np.mean([vus[k][str(f)] for f in unseen_faults(FKR[fk])])
    return float(S), float(U)

P('# TEP legacy dual-metric audit (paper Table 3 uses VUS-PR only)\n')
P('## (A) pak_auc_f1 — diagnostic, not a paper Table 3 metric')
P(f"{'Condition':<14}" + '  '.join(f'{fk[1:].upper():>11}' for fk in FK) + '   Mean S/U')
def t4row_pak(name, fn):
    Ss, Us, c = [], [], []
    for fk in FK:
        S, U = fn(fk); Ss.append(S); Us.append(U); c.append(f'{S:.3f}/{U:.3f}')
    P(f'{name:<14}' + '  '.join(c) + f'   {np.mean(Ss):.3f}/{np.mean(Us):.3f}')
for lbl, key in [('Random', 'Random'), ('PCA recon.', 'PCA recon.'), ('NN-distance', 'NN-distance'), ('Sensor range', 'Sensor range'), ('L2-norm', 'L2-norm')]:
    t4row_pak(lbl, lambda fk, k=key: (t4['simple'][k][FKR[fk]]['S'], t4['simple'][k][FKR[fk]]['U']))
for lbl, c in [('B0 Clean', 'B0'), ('B Unlabeled', 'B'), ('A LASAD', 'A'), ('w/o-GRL', None), ('D Recon-only', 'D')]:
    if lbl == 'w/o-GRL':
        t4row_pak(lbl, lambda fk: (nogrl[fk]['S'], nogrl[fk]['U']))
    else:
        t4row_pak(lbl, lambda fk, c=c: (t4['mae'][c][FKR[fk]]['S'], t4['mae'][c][FKR[fk]]['U']))

P('\n## (B) VUS-PR — paper Table 3 metric')
P(f"{'Condition':<14}" + '  '.join(f'{fk[1:].upper():>11}' for fk in FK) + '   Mean S/U')
SIMPLE_V = {'Random': 'Random', 'PCA recon.': 'PCA', 'NN-distance': 'NN', 'Sensor range': 'Sensor', 'L2-norm': 'L2'}
def t4row_vus(name, fn):
    Ss, Us, c = [], [], []
    for fk in FK:
        S, U = fn(fk); Ss.append(S); Us.append(U); c.append(f'{S:.3f}/{U:.3f}')
    P(f'{name:<14}' + '  '.join(c) + f'   {np.mean(Ss):.3f}/{np.mean(Us):.3f}')
for lbl, vk in SIMPLE_V.items():
    def f(fk, vk=vk):
        k = f'simple_{vk}_{fk}'; return (np.mean([vus[k][str(x)] for x in seen_faults(FKR[fk])]), np.mean([vus[k][str(x)] for x in unseen_faults(FKR[fk])]))
    t4row_vus(lbl, f)
for lbl, pref in [('B0 Clean', 'B0'), ('B Unlabeled', 'B'), ('A LASAD', 'A'), ('w/o-GRL', 'nogrl'), ('D Recon-only', 'D')]:
    t4row_vus(lbl, lambda fk, pref=pref: vSU(pref, fk))

P('\n## Discriminants & decompositions (pak | VUS-PR)')
dU_pak = [t4['discriminant'][FKR[fk]]['dU_AB'] for fk in FK]
P('Δ_unseen(A−B) per fold:')
P('  pak   : ' + '  '.join(f'{x:+.4f}' for x in dU_pak) + f'  mean {np.mean(dU_pak):+.4f}')
P('  VUS-PR: ' + '  '.join(f'{x:+.4f}' for x in vv['dU']) + f'  mean {np.mean(vv["dU"]):+.4f}')
# decompositions from vus_verify rows + pak
vr = {c: vv['permode'][c] for c in vv['permode']}  # [Ss,Us]
def mean(lst): return float(np.mean(lst))
P('A−D (discrepancy) unseen mean:  pak +0.0305   VUS-PR ' + f'{mean([vr["A"][1][i]-vr["D"][1][i] for i in range(4)]):+.4f}')
P('A−D (discrepancy) seen   mean:  pak +0.1507   VUS-PR ' + f'{mean([vr["A"][0][i]-vr["D"][0][i] for i in range(4)]):+.4f}')
P('GRL-push (A−w/oGRL) unseen:     pak +0.0120   VUS-PR ' + f'{mean([vr["A"][1][i]-vr["nogrl"][1][i] for i in range(4)]):+.4f}')
P('흡수방지 (w/oGRL−B) unseen:      pak +0.0070   VUS-PR ' + f'{mean([vr["nogrl"][1][i]-vr["B"][1][i] for i in range(4)]):+.4f}')

P('\n## Shapley pooled (pak | VUS-PR) + homogeneity')
shp_pak = {'step': 0.0146, 'rand': -0.0057, 'ds': -0.0010, 'unk': 0.0045}
for fam in ['step', 'unk', 'rand', 'ds']:
    vals = [vv['shapley'][ho][fam] for ho in vv['shapley'] if fam in vv['shapley'][ho]]
    P(f"  {fam:<5} pak {shp_pak[fam]:+.4f} | VUS-PR {np.mean(vals):+.4f}  (homogeneity {hom['per_family_hom'][fam]:.3f})")

open(f'{R}/all_results_dual.md', 'w').write('\n'.join(out))
print('\n'.join(out))
print('\n저장: all_results_dual.md')
