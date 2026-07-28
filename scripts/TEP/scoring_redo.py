"""scoring_redo.py — CAUSAL weighted-sum score fusion, prc-based insight.

Constraints (per user):
 (1) prc_auc for insight (roc dropped); pak validated separately.
 (2) LIVE-STREAM / causal only — no whole-stream rank/z (= leakage). Allowed:
     - amp_k:  recon + k·(official−recon)  = recon + 0.25k·disc·s_t   (retune the causal constant)
     - zpfx_w: zpfx(R) + w·zpfx(D)          (per-run normal-PREFIX z-norm: online cold-start calib)
     - raw_w:  recon + w·disc               (fixed constant weighted sum)
 (3) NOISE (deprioritized) = low in BOTH A,B AND low in BOTH recon,disc
     → max(A_recon, A_disc, B_recon, B_disc) prc < thr  (no channel×condition has signal).
 (4) method = weighted SUM of recon and disc.
 (+) report disc-ONLY A vs B per fault (does labeling sharpen the disc channel itself?).
Goal: A > B on (almost) all SIGNAL faults.
"""
import json, sys
import numpy as np
from sklearn.metrics import average_precision_score as ap

sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import seen_faults, unseen_faults, USABLE_FAULTS, FAMILY

ROOT = 'results/experiments/TEP_phase2_win100_ep30'
FOLD_KEYS = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}
FAM = {f: fam for fam, fs in FAMILY.items() for f in fs}
ONSET = 160
NOISE_THR = 0.40           # max(A_recon,A_disc,B_recon,B_disc) prc < thr → noise (floor≈0.278)
rt = json.load(open('scripts/TEP/data/test_run_table.json'))
PIDX = {f: np.concatenate([np.arange(r['start'], r['end']) for r in rt if r['fault'] == f or r['fault'] == 0])
        for f in USABLE_FAULTS}

def best_ep(d):
    rows = json.load(open(f'{d}/epoch_metrics.json')).get('epochs', [])
    pw = [r for r in rows if r.get('epoch', 0) > 15] or rows
    return max(pw, key=lambda r: r.get('pak_auc_f1', 0)).get('epoch')

def load(cond, fs):
    d = f'{ROOT}/{cond}/TEP/typegen_{fs}'
    z = np.load(f'{d}/epoch_scores/epoch_{best_ep(d):03d}_scores.npz')
    return (np.nan_to_num(z['teacher_recon_error']), np.nan_to_num(z['discrepancy_error']),
            np.nan_to_num(z['official_score']), z['point_labels'].astype(int))

def zpfx(X):
    out = np.zeros_like(X)
    for r in rt:
        s, e = r['start'], r['end']; p = X[s:s + ONSET]
        out[s:e] = (X[s:e] - p.mean()) / (p.std() + 1e-9)
    return out

RULES = {'official': lambda R, D, O: O, 'recon': lambda R, D, O: R, 'disc': lambda R, D, O: D}
for k in [2, 3, 5, 8, 12, 20]:
    RULES[f'amp_k{k}'] = (lambda R, D, O, k=k: R + k * (O - R))
for w in [1, 2, 3, 5]:
    RULES[f'zpfx_w{w}'] = (lambda R, D, O, w=w: zpfx(R) + w * zpfx(D))
for w in [25, 50, 100, 200]:
    RULES[f'raw_w{w}'] = (lambda R, D, O, w=w: R + w * D)

prc = {rn: {'A': {}, 'B': {}} for rn in RULES}
for fs, fk in FOLD_KEYS.items():
    RA, DA, OA, yA = load('phase2_A', fs); RB, DB, OB, yB = load('phase2_B', fs)
    for rn, fn in RULES.items():
        sA, sB = fn(RA, DA, OA), fn(RB, DB, OB)
        for f in USABLE_FAULTS:
            ix = PIDX[f]
            prc[rn]['A'][(fs, f)] = ap(yA[ix], sA[ix])
            prc[rn]['B'][(fs, f)] = ap(yB[ix], sB[ix])

seen = [(fs, f) for fs, fk in FOLD_KEYS.items() for f in seen_faults(fk)]
def chan_max(fs, f):
    return max(prc['recon']['A'][(fs, f)], prc['disc']['A'][(fs, f)],
              prc['recon']['B'][(fs, f)], prc['disc']['B'][(fs, f)])
noise = [(fs, f) for fs, f in seen if chan_max(fs, f) < NOISE_THR]
signal = [(fs, f) for fs, f in seen if (fs, f) not in noise]

print(f"=== per-component prc (seen fold) + NOISE 분류 (floor≈0.278, thr={NOISE_THR}) ===")
print(f"{'fold':6} {'IDV':>4} {'A_rec':>6} {'A_dis':>6} {'B_rec':>6} {'B_dis':>6} {'A_off':>6} {'B_off':>6} {'max4':>6}")
for fs, f in seen:
    pr = prc
    vals = (pr['recon']['A'][(fs, f)], pr['disc']['A'][(fs, f)], pr['recon']['B'][(fs, f)],
            pr['disc']['B'][(fs, f)], pr['official']['A'][(fs, f)], pr['official']['B'][(fs, f)])
    tag = ' <NOISE>' if (fs, f) in noise else ''
    print(f"{fs:6} {f:>4} " + " ".join(f"{v:>6.3f}" for v in vals) + f" {chan_max(fs,f):>6.3f}{tag}")
print(f"\nSIGNAL {len(signal)}개 / NOISE {len(noise)}개 → {[f'{fs}-IDV{f}' for fs,f in noise]}\n")

# (2) disc-ONLY A vs B
print(f"=== (2) disc-ONLY A vs B (signal faults) — labeling이 disc 채널 자체를 sharpen하나? ===")
dA = [prc['disc']['A'][p] for p in signal]; dB = [prc['disc']['B'][p] for p in signal]
dd = [a - b for a, b in zip(dA, dB)]
print(f"  disc-only: A>B {sum(1 for x in dd if x>0)}/{len(signal)}, A_macro={np.mean(dA):.4f} B_macro={np.mean(dB):.4f} Δ={np.mean(dd):+.4f} worst={min(dd):+.4f}")
print(f"  {'fold':6} {'IDV':>4} {'A_disc':>7} {'B_disc':>7} {'Δ':>7}")
for fs, f in signal:
    a, b = prc['disc']['A'][(fs, f)], prc['disc']['B'][(fs, f)]
    print(f"  {fs:6} {f:>4} {a:>7.4f} {b:>7.4f} {a-b:>+7.4f}{'' if a>b else '  A<=B'}")

print(f"\n=== causal 가중합 rule별 prc — A>B on SIGNAL seen faults ===")
print(f"{'rule':10} {'A>B':>8} {'A_S':>7} {'B_S':>7} {'ΔS':>8} {'worst':>8}")
scored = []
for rn in RULES:
    A = [prc[rn]['A'][p] for p in signal]; B = [prc[rn]['B'][p] for p in signal]
    d = [a - b for a, b in zip(A, B)]; cnt = sum(1 for x in d if x > 0)
    scored.append((rn, cnt, np.mean(A), np.mean(B), np.mean(d), min(d)))
    print(f"{rn:10} {cnt:>3}/{len(signal):<4} {np.mean(A):>7.4f} {np.mean(B):>7.4f} {np.mean(d):>+8.4f} {min(d):>+8.4f}")

best = max([s for s in scored if s[0] not in ('recon', 'disc', 'official')],
           key=lambda r: (r[1], r[4]))[0]
print(f"\n=== best causal weighted-sum = '{best}' — SIGNAL seen-fault detail (prc) ===")
print(f"{'fold':6} {'IDV':>4} {'A_new':>7} {'B_new':>7} {'Δ':>7} | {'A_off':>7} {'B_off':>7}")
for fs, f in signal:
    a, b = prc[best]['A'][(fs, f)], prc[best]['B'][(fs, f)]
    ao, bo = prc['official']['A'][(fs, f)], prc['official']['B'][(fs, f)]
    print(f"{fs:6} {f:>4} {a:>7.4f} {b:>7.4f} {a-b:>+7.4f} | {ao:>7.4f} {bo:>7.4f}{'' if a>b else '  A<=B'}")

json.dump({'noise': [f'{fs}_{f}' for fs, f in noise], 'signal': [f'{fs}_{f}' for fs, f in signal], 'best_rule': best,
           'prc': {rn: {f'{fs}_{f}': {'A': prc[rn]['A'][(fs, f)], 'B': prc[rn]['B'][(fs, f)]} for fs, f in seen} for rn in RULES}},
          open(f'{ROOT}/scoring_redo_prc.json', 'w'), indent=1)
print(f"\n저장: {ROOT}/scoring_redo_prc.json  (best={best})")
