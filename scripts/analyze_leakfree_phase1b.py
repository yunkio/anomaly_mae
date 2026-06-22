"""Phase 1b: add two insight-driven LEAK-FREE entity signals to temp/leakfree_signals.json.
(1) disc/recon EPOCH-STABILITY: mean consecutive-epoch Spearman of point rankings over the last
    K eval epochs <= best_epoch (no labels). High = stabilized = trustworthy.
(2) disc MAGNITUDE / over-mimicry: raw disc.mean, disc CV, disc/recon scale ratio (no labels).
Merges into each entity as r['esig'].  Also prints the (2) verification correlation vs disc-PAK.
"""
import numpy as np, json, os, glob

RUN = "/home/ykio/notebooks/TSMAE/results/experiments/271_20260602_020545_271canon_baseline"
R = json.load(open('/home/ykio/notebooks/TSMAE/temp/leakfree_signals.json'))
rng = np.random.default_rng(0)
KWIN = 10  # last K eval epochs (<= best) for stability

def srank(a):
    return a.argsort().argsort().astype(np.float64)

def stability(sub, be, key, idx):
    d = f"{RUN}/{sub}/epoch_scores"
    eps = sorted(int(os.path.basename(f).split('_')[1]) for f in glob.glob(f"{d}/epoch_*_scores.npz"))
    eps = [e for e in eps if e <= be][-KWIN:]
    if len(eps) < 2:
        eps = sorted(int(os.path.basename(f).split('_')[1]) for f in glob.glob(f"{d}/epoch_*_scores.npz"))[:KWIN]
    arrs = []
    for e in eps:
        x = np.load(f"{d}/epoch_{e:03d}_scores.npz")[key].astype(np.float64)[idx]
        arrs.append(srank(x))
    cors = []
    for i in range(1, len(arrs)):
        a, b = arrs[i - 1], arrs[i]
        if a.std() and b.std():
            cors.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(cors)) if cors else 0.0

for r in R:
    sub = r['sub']
    d = f"{RUN}/{sub}"
    meta = json.load(open(f"{d}/experiment_metadata.json")); be = meta['timing']['best_epoch']
    z = np.load(f"{d}/epoch_scores/epoch_{be:03d}_scores.npz")
    disc = z['discrepancy_error'].astype(np.float64); recon = z['teacher_recon_error'].astype(np.float64)
    n = len(disc); idx = rng.choice(n, size=min(20000, n), replace=False)
    r['esig'] = {
        'disc_stab': stability(sub, be, 'discrepancy_error', idx),
        'recon_stab': stability(sub, be, 'teacher_recon_error', idx),
        'disc_mag': float(disc.mean()),
        'disc_cv': float(disc.std() / (disc.mean() + 1e-12)),
        'disc_recon_ratio': float(disc.mean() / (recon.mean() + 1e-12)),
        'best_ep': be,
    }
    print(f"{sub:20s} be={be:3d} disc_stab={r['esig']['disc_stab']:+.3f} recon_stab={r['esig']['recon_stab']:+.3f} "
          f"disc_mag={r['esig']['disc_mag']:.2e} cv={r['esig']['disc_cv']:.2f} d/r={r['esig']['disc_recon_ratio']:.2f} "
          f"| disc_PAK={r['paks']['disc']:.3f}", flush=True)

json.dump(R, open('/home/ykio/notebooks/TSMAE/temp/leakfree_signals.json', 'w'), indent=1)

# (2) verification: does low disc magnitude / instability predict low disc-PAK?
def sp(a, b):
    a = srank(np.asarray(a, float)); b = srank(np.asarray(b, float))
    return float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else float('nan')
pdisc = [r['paks']['disc'] for r in R]
print("\n=== (2) VERIFY: correlation of each disc-reliability signal with disc-channel PAK ===")
for k in ['disc_stab', 'disc_mag', 'disc_cv', 'disc_recon_ratio', 'best_ep']:
    print(f"  spearman({k:16s}, disc_PAK) = {sp([r['esig'][k] for r in R], pdisc):+.3f}")
print("  (disc_stab>0, disc_mag>0, disc_recon_ratio>0 ⇒ insight holds: low/unstable disc → low disc-PAK)")
