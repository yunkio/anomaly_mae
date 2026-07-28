"""derive_fill.py — print fill-ready Table D.1 values from pak_fill.json (any state)."""
import json, numpy as np, sys
sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import seen_faults, unseen_faults, ALL_FAULTS, FAMILY
R = 'results/experiments/TEP_phase2_win100_ep30'
res = json.load(open(f'{R}/pak_fill.json'))
FK = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}
FAM = {f: fam for fam, fs in FAMILY.items() for f in fs}
def sm(k, fs): return np.mean([res[k][str(f)] for f in seen_faults(FK[fs])])
def um(k, fs): return np.mean([res[k][str(f)] for f in unseen_faults(FK[fs])])
def has(k): return k in res

print('### §5 per-mode S/U (tmf pak) — cols: fstepS fstepU randS randU dsS dsU unkS unkU')
for c in ['B0', 'B', 'A', 'D']:
    v = []
    for fs in FK:
        k = 'B0_tmf' if c == 'B0' else (f'{c}_tmf_{fs}' if c in ('A', 'B') else f'D_recon_{fs}')
        v += [f'{sm(k, fs):.3f}', f'{um(k, fs):.3f}']
    print(f'{c}: ' + '  '.join(v))

print('\n### §6 clean-train ref: B0 (clean MAE) S/U per fold')
print('B0: ' + '  '.join(f'{sm("B0_tmf", fs):.3f} / {um("B0_tmf", fs):.3f}' for fs in FK))
print('### §6 C_dmg (seen): MAE-B=B0_S−B_S, MAE-A=B0_S−A_S  cols: fstep frand fds funk')
for t, c in [('MAE-B', 'B'), ('MAE-A', 'A')]:
    print(f'{t}: ' + '  '.join(f'{sm("B0_tmf", fs) - sm(f"{c}_tmf_{fs}", fs):+.3f}' for fs in FK))

print('\n### §7  cols: fstep frand fds funk')
print('Δ_unseen(A−B): ' + '  '.join(f'{um(f"A_tmf_{fs}", fs) - um(f"B_tmf_{fs}", fs):+.3f}' for fs in FK))
print('Ĝ_A(GRL bal_acc−0.5): -0.000  +0.000  -0.186  -0.001')

print('\n### §A noisy unseen (tmf pak)  cols: fstepU randU dsU unkU')
print('100% (A): ' + '  '.join(f'{um(f"A_tmf_{fs}", fs):.3f}' for fs in FK))
for lab in ['lab80', 'lab50', 'lab25', 'lab10']:
    if all(has(f'{lab}_tmf_{fs}') for fs in FK):
        print(f'{lab}: ' + '  '.join(f'{um(f"{lab}_tmf_{fs}", fs):.3f}' for fs in FK))
    else:
        print(f'{lab}: (계산중)')
print('0% (B): ' + '  '.join(f'{um(f"B_tmf_{fs}", fs):.3f}' for fs in FK))

print('\n### §10 per-fault (tmf pak)  cols: B0 | A@fstep frand fds funk | B@fstep frand fds funk  (★=seen fold)')
for f in ALL_FAULTS:
    fam = FAM.get(f, 'EXCL')
    star = {fs: ('*' if f in seen_faults(FK[fs]) else '') for fs in FK}
    a = '  '.join(f'{res[f"A_tmf_{fs}"][str(f)]:.3f}{star[fs]}' for fs in FK)
    b = '  '.join(f'{res[f"B_tmf_{fs}"][str(f)]:.3f}{star[fs]}' for fs in FK)
    print(f'IDV{f:>2}({fam[:4]}): B0={res["B0_tmf"][str(f)]:.3f} | A {a} | B {b}')
