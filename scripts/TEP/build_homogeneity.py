"""build_homogeneity.py — test the hypothesis: U-valley is driven by within-fault-type run homogeneity.
TEP is a deterministic simulator → runs of the same fault share an (almost) identical signature +
measurement noise. Partial labeling drops near-duplicate faulty runs into the normal set as y=0,
creating a maximal label contradiction → unstable discrepancy normalization → U-valley.

Discriminating prediction: faults whose runs are MORE homogeneous → deeper valley when their family
is the partially-labeled seen contamination. step (deterministic) should be most homogeneous & deepest
valley; random-variation (8-12) least homogeneous & shallow valley.
"""
import json, sys
import numpy as np
sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import USABLE_FAULTS

DATA = 'scripts/TEP/data'
R = 'results/experiments/TEP_phase2_win100_ep30'
FAM_FILE = {'step': 'train_f_step.npz', 'rand': 'train_f_rand.npz', 'ds': 'train_f_ds.npz', 'unk': 'train_f_unk.npz'}
FAM_FAULTS = {'step': [1, 2, 4, 5, 6, 7], 'rand': [8, 10, 11, 12], 'ds': [13, 14], 'unk': [16, 17, 18, 19, 20]}
RL = 960

def run_signatures(fam):
    """Per fault: list of run-level z-scored mean-feature signatures over the faulty region."""
    z = np.load(f'{DATA}/{FAM_FILE[fam]}')
    X, y, fid = z['X'].astype(np.float64), z['y'], z['fault_id'].astype(int)
    n = len(y) // RL
    # FF baseline stats (fault_id==0 runs)
    ff_mask = fid == 0
    mu, sd = X[ff_mask].mean(0), X[ff_mask].std(0) + 1e-9
    sigs = {}  # fault -> list of (52,) signatures
    for i in range(n):
        sl = slice(i * RL, (i + 1) * RL)
        rf = fid[sl]; ry = y[sl]
        f = int(np.bincount(rf[rf > 0]).argmax()) if (rf > 0).any() else 0
        if f == 0:
            continue
        anom = ry == 1
        if anom.sum() < 5:
            continue
        sig = ((X[sl][anom] - mu) / sd).mean(0)   # z-scored mean deviation over faulty region
        sigs.setdefault(f, []).append(sig)
    return sigs, mu, sd

def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# ---- per-fault homogeneity ----
hom = {}   # fault -> within-fault mean pairwise cosine sim + L2 stats
fam_of = {f: fam for fam, fs in FAM_FAULTS.items() for f in fs}
for fam in FAM_FILE:
    sigs, mu, sd = run_signatures(fam)
    for f, S in sigs.items():
        S = np.array(S)
        # within-fault pairwise cosine sim
        pcs = [cos(S[i], S[j]) for i in range(len(S)) for j in range(i + 1, len(S))]
        cen = S.mean(0)
        within_d = np.mean([np.linalg.norm(s - cen) for s in S])     # run scatter around fault centroid
        mag = np.linalg.norm(cen)                                     # fault signal magnitude (vs FF=0)
        hom[f] = {'within_cos': float(np.mean(pcs)), 'within_scatter': float(within_d),
                  'signal_mag': float(mag), 'cv': float(within_d / (mag + 1e-9)), 'n_runs': len(S)}

# ---- valley depth per fold (from extras_data) ----
ex = json.load(open(f'{R}/extras_data.json'))
FOLD_FAM = {'fstep': 'step', 'frand': 'rand', 'fds': 'ds', 'funk': 'unk'}
prow = {r['cond']: r for r in ex['noisy']['p_rows']}
valley = {}   # fold -> (min partial U) - (B U); negative = valley below floor
for fk, fam in FOLD_FAM.items():
    partials = [prow[c][fk] for c in ['lab80', 'lab50', 'lab25', 'lab10']]
    Bu = prow['B'][fk]; Au = prow['A'][fk]
    valley[fam] = {'min_partial': float(min(partials)), 'B': float(Bu), 'A': float(Au),
                   'depth_vs_B': float(min(partials) - Bu), 'A_minus_B': float(Au - Bu)}

# ---- per-family homogeneity (mean over faults) ----
print('=== per-fault within-fault run homogeneity (raw features) ===')
print(f"{'fault':>6} {'fam':>5} {'within_cos':>11} {'CV(scatter/mag)':>16} {'signal_mag':>11} runs")
for f in sorted(hom):
    h = hom[f]
    print(f"  IDV{f:>2} {fam_of[f]:>5} {h['within_cos']:>10.4f} {h['cv']:>15.4f} {h['signal_mag']:>11.3f} {h['n_runs']}")

print('\n=== per-family: homogeneity vs valley depth (THE discriminating test) ===')
print(f"{'family':>6} {'mean_within_cos':>16} {'mean_CV':>9} | {'valley_depth(min−B)':>20} {'A−B':>8}")
fam_hom = {}
for fam, fs in FAM_FAULTS.items():
    fs_u = [f for f in fs if f in USABLE_FAULTS]
    mc = np.mean([hom[f]['within_cos'] for f in fs_u])
    mcv = np.mean([hom[f]['cv'] for f in fs_u])
    fam_hom[fam] = mc
    v = valley[fam]
    print(f"  {fam:>6} {mc:>15.4f} {mcv:>9.4f} | {v['depth_vs_B']:>+19.4f} {v['A_minus_B']:>+8.4f}")

# ---- correlation: homogeneity vs valley depth ----
fams = list(FAM_FAULTS)
x = np.array([fam_hom[f] for f in fams])           # homogeneity (within_cos)
yv = np.array([valley[f]['depth_vs_B'] for f in fams])  # valley depth (more neg = deeper)
r = np.corrcoef(x, yv)[0, 1]
print(f'\ncorr(homogeneity, valley_depth) = {r:+.3f}  (가설: 더 균일 → 더 깊은 valley = 음의 상관, depth가 음수일수록 깊음 → r 음수 기대)')
print('  homogeneity rank:', [f for f in sorted(fams, key=lambda k: -fam_hom[k])])
print('  valley depth  rank (deepest first):', [f for f in sorted(fams, key=lambda k: valley[k]['depth_vs_B'])])
json.dump({'per_fault': hom, 'per_family_hom': fam_hom, 'valley': valley, 'corr': float(r)},
          open(f'{R}/homogeneity.json', 'w'), indent=1)
print('\n저장: homogeneity.json')
