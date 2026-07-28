"""build_brief_all.py — consolidate ALL experiment results (pak_auc_f1) into one comprehensive
tables document for the subpage. Reads every results JSON. vus_pr appended later from vus_results.json.
"""
import json, numpy as np, sys
sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import seen_faults, unseen_faults, USABLE_FAULTS
R = 'results/experiments/TEP_phase2_win100_ep30'
def L(p):
    try: return json.load(open(f'{R}/{p}'))
    except Exception: return None
t4 = L('table4_data.json'); nogrl = L('nogrl_results.json'); breadth = L('breadth_results.json')
shap = L('shapley.json'); hom = L('homogeneity.json'); extras = L('extras_data.json'); vus = L('vus_results.json')
FK = ['fstep', 'frand', 'fds', 'funk']; FKR = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}
FAML = {'fstep': 'F-STEP', 'frand': 'F-RAND', 'fds': 'F-DS', 'funk': 'F-UNK'}
out = []
def P(*a): out.append(' '.join(str(x) for x in a))

P('# 전체 실험 결과 데이터 (pak_auc_f1, train mean-fixed scoring)\n')

# ===== Table 4 =====
P('## 1. Table 4 — Type-Disjoint Generalization (per-fold Seen/Unseen)')
P('cols: F-STEP S/U  F-RAND S/U  F-DS S/U  F-UNK S/U  Mean S/U')
def row(name, getter):
    cells = []; Ss = []; Us = []
    for fk in FK:
        S, U = getter(fk); Ss.append(S); Us.append(U); cells.append(f'{S:.4f}/{U:.4f}')
    P(f'{name:<16} ' + '  '.join(cells) + f'   {np.mean(Ss):.3f}/{np.mean(Us):.3f}')
for lbl, key in [('Random', 'Random'), ('PCA recon.', 'PCA recon.'), ('NN-distance', 'NN-distance'), ('Sensor range', 'Sensor range'), ('L2-norm', 'L2-norm')]:
    row(lbl, lambda fk, k=key: (t4['simple'][k][FKR[fk]]['S'], t4['simple'][k][FKR[fk]]['U']))
for lbl, c in [('B0 Clean', 'B0'), ('B Unlabeled', 'B'), ('A LASAD', 'A'), ('D Recon-only', 'D')]:
    row(lbl, lambda fk, c=c: (t4['mae'][c][FKR[fk]]['S'], t4['mae'][c][FKR[fk]]['U']))
if nogrl:
    row('w/o-GRL', lambda fk: (nogrl[fk]['S'], nogrl[fk]['U']))
P('\nDiscriminants (unseen-mode):')
P('Δ_unseen(A−B): ' + '  '.join(f'{t4["discriminant"][FKR[fk]]["dU_AB"]:+.4f}' for fk in FK) + f"  mean {np.mean([t4['discriminant'][FKR[fk]]['dU_AB'] for fk in FK]):+.4f}")
P('Δ_gap(Γ̂):     ' + '  '.join(f'{t4["discriminant"][FKR[fk]]["ghat_A"]:+.4f}' for fk in FK))

# ===== per-fault Table 4 (A/B/B0/D) =====
pf = L('pak_fill.json')
if pf:
    P('\n## 2. Per-fault pak (USABLE 17) — A/B/B0/D per fold')
    P('IDV  fam | B0 | A@step/rand/ds/unk | B@... | D@...')
    FAM = {f: fam for fam, fs in {'step': [1, 2, 4, 5, 6, 7], 'rand': [8, 10, 11, 12], 'ds': [13, 14], 'unk': [16, 17, 18, 19, 20]}.items() for f in fs}
    for f in USABLE_FAULTS:
        a = '/'.join(f'{pf[f"A_tmf_{fk}"][str(f)]:.3f}' for fk in FK)
        b = '/'.join(f'{pf[f"B_tmf_{fk}"][str(f)]:.3f}' for fk in FK)
        d = '/'.join(f'{pf[f"D_recon_{fk}"][str(f)]:.3f}' for fk in FK)
        P(f'IDV{f:>2} {FAM[f]:<4} | B0={pf["B0_tmf"][str(f)]:.3f} | A {a} | B {b} | D {d}')

# ===== w/o-GRL decomposition =====
if nogrl:
    P('\n## 3. GRL Ablation (w/o-GRL) — A−w/o-GRL = GRL push; w/o-GRL−B = ignore-vs-absorb')
    for fk in FK:
        dSg = t4['mae']['A'][FKR[fk]]['S'] - nogrl[fk]['S']; dUg = t4['mae']['A'][FKR[fk]]['U'] - nogrl[fk]['U']
        dUb = nogrl[fk]['U'] - t4['mae']['B'][FKR[fk]]['U']
        P(f'{FAML[fk]}: GRL-push ΔS={dSg:+.4f} ΔU={dUg:+.4f} | (w/oGRL−B) U={dUb:+.4f}')
    mSg = np.mean([t4['mae']['A'][FKR[fk]]['S'] - nogrl[fk]['S'] for fk in FK]); mUg = np.mean([t4['mae']['A'][FKR[fk]]['U'] - nogrl[fk]['U'] for fk in FK])
    P(f'MEAN GRL-push: ΔS={mSg:+.4f} ΔU={mUg:+.4f}')

# ===== Breadth + Shapley =====
if breadth:
    P('\n## 4. Label-Breadth pooled k-sweep (held-out unseen detection vs #labeled families)')
    pooled = breadth.get('pooled', {})
    P('  k=0/1/2/3: ' + ' / '.join(f'{pooled.get(str(k), 0):.4f}' for k in range(4)) if pooled else '  (k-sweep pending)')
    P('  per held-out U_by_k:')
    for ho in ['step', 'rand', 'ds', 'unk']:
        u = breadth['results'][ho]['U_by_k']
        P(f'    {ho}: ' + ' / '.join(f'{u.get(str(k)) if u.get(str(k)) is not None else "-":.4f}' if u.get(str(k)) is not None else '  -  ' for k in range(4)))
if shap:
    P('\n## 5. Shapley (fair per-family contribution to unseen, controlling which family)')
    fams = ['step', 'rand', 'ds', 'unk']
    P('  per held-out:')
    for ho in shap:
        P(f'    held-out={ho}: ' + '  '.join(f'{k}={v:+.4f}' for k, v in shap[ho].items()))
    P('  pooled mean Shapley:')
    for f in fams:
        vals = [shap[ho][f] for ho in shap if f in shap[ho]]
        if vals: P(f'    {f}: {np.mean(vals):+.4f} (n={len(vals)})  homogeneity={hom["per_family_hom"][f]:.3f}' if hom else f'    {f}: {np.mean(vals):+.4f}')

# ===== Homogeneity =====
if hom:
    P('\n## 6. Within-fault homogeneity vs noisy-valley (mechanism)')
    P(f'  corr(homogeneity, valley_depth) = {hom["corr"]:+.3f}')
    P('  per-family: homogeneity(within_cos) | noisy valley_depth(min−B) | A−B')
    for fam in ['step', 'rand', 'ds', 'unk']:
        v = hom['valley'][fam]
        P(f'    {fam}: hom={hom["per_family_hom"][fam]:.3f} | valley={v["depth_vs_B"]:+.4f} | A−B={v["A_minus_B"]:+.4f}')

# ===== Absorption (manual) =====
P('\n## 7. Absorption (train recon anomaly/normal ratio, F-STEP)')
P('  MAE Teacher (A·B): normal=0.0029 anomaly=0.0028 ratio=0.97 (흡수)')
P('  PCA (rigid):       normal=0.0001 anomaly=0.0003 ratio=2.62 (flag)')

# ===== Noisy sweep =====
if extras:
    P('\n## 8. Noisy label sweep (per-mode-mode F1, unseen, per fold) — within-type stratified')
    for r in extras['noisy']['p_rows']:
        P(f"  p={r['p']} ({r['cond']}): " + ' / '.join(f'{r[fk]:.4f}' for fk in FK) + f"  mean {r['mean']:.4f}")

# ===== LOFO =====
if extras and 'lofo' in extras:
    P('\n## 9. LOFO (3-seen/1-held-out): Seen-3 S / held-out U / cont U / main-1seen U')
    for ho in ['step', 'rand', 'ds', 'unk']:
        e = extras['lofo'][ho]
        P(f"  {ho}: S={e['A']['S']:.4f} U={e['A']['U']:.4f} contU={e['A_cont']['U']:.4f} main1seen={e['main_unseen_U']:.4f}")

# ===== vus_pr (if available) =====
P('\n## 10. VUS-PR 검증 상태:', f'{len(vus)} keys' if vus else '계산 중 (vus_results.json)')

open(f'{R}/all_results_tables.md', 'w').write('\n'.join(out))
print('\n'.join(out))
print('\n저장: all_results_tables.md')
