"""comprehensive_v2.py — FULL recompute under the canonical scoring 'train mean-fixed'
   tmf = recon + (R_tr/D_tr)·disc   (R_tr/D_tr recovered from each run's official score).

Emits every table the Notion study page needs: R_tr/D_tr, per-mode S/U (per-fold+macro),
Δ_unseen/ΔS/Ĝ, per-fault (all 17 × 4 folds), spillover (near-pair), damage(B0−B)/recovery(A−B),
noisy sweep, method comparison, component (recon/disc) prc.  prc (fast).  Saves JSON + prints MD.
"""
import json, sys, os
import numpy as np
from sklearn.metrics import average_precision_score as ap

sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import seen_faults, unseen_faults, USABLE_FAULTS, FAMILY

ROOT = 'results/experiments/TEP_phase2_win100_ep30'
FOLD_KEYS = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}
FAM = {f: fam for fam, fs in FAMILY.items() for f in fs}
NEAR = {4: 11, 11: 4, 5: 12, 12: 5}
W_OFF = 0.25
rt = json.load(open('scripts/TEP/data/test_run_table.json'))
PIDX = {f: np.concatenate([np.arange(r['start'], r['end']) for r in rt if r['fault'] == f or r['fault'] == 0])
        for f in USABLE_FAULTS}

def best_ep(d):
    rows = json.load(open(f'{d}/epoch_metrics.json')).get('epochs', [])
    pw = [r for r in rows if r.get('epoch', 0) > 15] or rows
    return max(pw, key=lambda r: r.get('pak_auc_f1', 0)).get('epoch')

def load(path):
    z = np.load(f'{ROOT}/{path}/epoch_scores/epoch_{best_ep(f"{ROOT}/{path}"):03d}_scores.npz')
    return (np.nan_to_num(z['teacher_recon_error']).astype(np.float64),
            np.nan_to_num(z['discrepancy_error']).astype(np.float64),
            np.nan_to_num(z['official_score']).astype(np.float64), z['point_labels'].astype(int))

def recover_ratio(R, D, O):
    cR, cD = np.cumsum(R), np.cumsum(D)
    m = D > (np.median(D) * 0.5)
    s = (O[m] - R[m]) / (W_OFF * D[m])
    A_ = np.stack([np.ones(m.sum()), -s], 1); b_ = s * cD[m] - cR[m]
    sol, *_ = np.linalg.lstsq(A_, b_, rcond=None)
    return sol[0] / sol[1] if sol[1] != 0 else float('nan')

COND = {'A': 'phase2_A/TEP/typegen_{}', 'B': 'phase2_B/TEP/typegen_{}',
        'lab80': 'noisy/TEP/typegen_{}_lab80', 'lab50': 'noisy/TEP/typegen_{}_lab50',
        'lab25': 'noisy/TEP/typegen_{}_lab25', 'lab10': 'noisy/TEP/typegen_{}_lab10'}

# prc[cond][scoring][fold][fault]; ratio[cond][fold]
prc = {}; ratio = {}
for cond, tmpl in COND.items():
    prc[cond] = {sc: {} for sc in ['official', 'tmf', 'recon', 'disc']}; ratio[cond] = {}
    for fs in FOLD_KEYS:
        path = tmpl.format(fs)
        if not os.path.exists(f'{ROOT}/{path}/epoch_metrics.json'):
            continue
        R, D, O, y = load(path); rr = recover_ratio(R, D, O); ratio[cond][fs] = float(rr)
        scs = {'official': O, 'tmf': R + rr * D, 'recon': R, 'disc': D}
        for sc, s in scs.items():
            prc[cond][sc][fs] = {str(f): float(ap(y[PIDX[f]], s[PIDX[f]])) for f in USABLE_FAULTS}
# B0 fold-independent
p0 = 'phase2_B0/TEP/typegen_ffonly'
R, D, O, y = load(p0); rr0 = recover_ratio(R, D, O)
scs0 = {'official': O, 'tmf': R + rr0 * D, 'recon': R, 'disc': D}
prc['B0'] = {sc: {fs: {str(f): float(ap(y[PIDX[f]], scs0[sc][PIDX[f]])) for f in USABLE_FAULTS} for fs in FOLD_KEYS}
             for sc in scs0}
ratio['B0'] = {'(fold-indep)': float(rr0)}
prc['D'] = {'recon-only': prc['A']['recon']}

def smacro(c, sc, fs): return float(np.mean([prc[c][sc][fs][str(f)] for f in seen_faults(FOLD_KEYS[fs])]))
def umacro(c, sc, fs): return float(np.mean([prc[c][sc][fs][str(f)] for f in unseen_faults(FOLD_KEYS[fs])]))
def SU(c, sc):
    S = {fs: smacro(c, sc, fs) for fs in FOLD_KEYS if fs in prc[c][sc]}
    U = {fs: umacro(c, sc, fs) for fs in FOLD_KEYS if fs in prc[c][sc]}
    return S, U, (np.mean(list(S.values())) if S else None), (np.mean(list(U.values())) if U else None)

out = {'ratio': ratio, 'prc': prc,
       'summary': {c: {sc: dict(zip(['S_perfold', 'U_perfold', 'S_macro', 'U_macro'], SU(c, sc)))
                       for sc in prc[c]} for c in prc}}
json.dump(out, open(f'{ROOT}/comprehensive_v2.json', 'w'), indent=1)

# ============ PRINT MARKDOWN TABLES ============
def row(*xs): print('| ' + ' | '.join(str(x) for x in xs) + ' |')
print("\n## R_tr/D_tr (복원)")
row('cond', *FOLD_KEYS); row(*['---'] * (1 + len(FOLD_KEYS)))
for c in ['A', 'B', 'lab80', 'lab50', 'lab25', 'lab10']:
    row(c, *[f"{ratio[c].get(fs, float('nan')):.1f}" for fs in FOLD_KEYS])
row('B0', f"{rr0:.1f} (fold-indep)", '', '', '')

print("\n## per-mode (prc) — official vs tmf, 전 조건")
row('cond', 'sc', 'A/—_S', '_U', 'fstep_U', 'frand_U', 'fds_U', 'funk_U'); row(*['---'] * 8)
for c in ['A', 'B', 'B0', 'lab80', 'lab50', 'lab25', 'lab10']:
    for sc in ['official', 'tmf']:
        S, U, Sm, Um = SU(c, sc)
        row(c, sc, f"{Sm:.3f}", f"{Um:.3f}", *[f"{U.get(fs, float('nan')):.3f}" for fs in FOLD_KEYS])

print("\n## Δ_unseen·ΔS·Ĝ — 전 scoring (A vs B)")
row('scoring', 'A_S', 'B_S', 'ΔS', 'A_U', 'B_U', 'Δ_unseen', 'Ĝ=ΔS−ΔU'); row(*['---'] * 8)
for sc in ['official', 'tmf', 'recon', 'disc']:
    _, _, As, Au = SU('A', sc); _, _, Bs, Bu = SU('B', sc)
    row(sc, f"{As:.3f}", f"{Bs:.3f}", f"{As-Bs:+.4f}", f"{Au:.3f}", f"{Bu:.3f}", f"{Au-Bu:+.4f}", f"{(As-Bs)-(Au-Bu):+.4f}")

print("\n## per-fault (tmf prc) — A/B/B0, seen-fold + unseen-avg")
row('IDV', 'fam', 'B0', 'A_seen', 'B_seen', 'A−B_seen', 'A_uns', 'B_uns', 'A−B_uns'); row(*['---'] * 9)
for f in USABLE_FAULTS:
    sf = next(fs for fs in FOLD_KEYS if f in seen_faults(FOLD_KEYS[fs]))
    uf = [fs for fs in FOLD_KEYS if f not in seen_faults(FOLD_KEYS[fs])]
    As, Bs = prc['A']['tmf'][sf][str(f)], prc['B']['tmf'][sf][str(f)]
    Au = np.mean([prc['A']['tmf'][fs][str(f)] for fs in uf]); Bu = np.mean([prc['B']['tmf'][fs][str(f)] for fs in uf])
    b0 = prc['B0']['tmf'][sf][str(f)]
    row(f, FAM.get(f, ''), f"{b0:.3f}", f"{As:.3f}", f"{Bs:.3f}", f"{As-Bs:+.3f}", f"{Au:.3f}", f"{Bu:.3f}", f"{Au-Bu:+.3f}")

print("\n## damage(B0−B)/recovery(A−B) on UNSEEN, per-fold (tmf)")
row('fold', 'B0_U', 'B_U', 'A_U', 'damage', 'recovery', 'recov%'); row(*['---'] * 7)
for fs in FOLD_KEYS:
    b0u = umacro('B0', 'tmf', fs); bu = umacro('B', 'tmf', fs); au = umacro('A', 'tmf', fs)
    dmg = b0u - bu; rec = au - bu
    row(fs, f"{b0u:.3f}", f"{bu:.3f}", f"{au:.3f}", f"{dmg:+.3f}", f"{rec:+.3f}", f"{100*rec/dmg:.1f}%" if dmg else '—')

print("\n## noisy sweep (tmf) — labeled% → S/U")
row('labeled%', 'cond', 'S_macro', 'U_macro', 'ΔU vs B'); row(*['---'] * 5)
_, _, _, BU0 = SU('B', 'tmf')
for lab, c in [('100% (A)', 'A'), ('80%', 'lab80'), ('50%', 'lab50'), ('25%', 'lab25'), ('10%', 'lab10'), ('0% (B)', 'B')]:
    _, _, Sm, Um = SU(c, 'tmf'); row(lab, c, f"{Sm:.3f}", f"{Um:.3f}", f"{Um-BU0:+.3f}")

print("\n저장: comprehensive_v2.json")
