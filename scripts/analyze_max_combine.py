"""(2) Combine score components via MAX instead of SUM. Saved exp271 data only (no inference).
baseline (sum)   : recon + scaled_disc/4          (scaled_disc = disc*(recon_mean/disc_mean))
A meanmatch max  : max(recon, w*scaled_disc)        over w grid
B zscore max     : max(z_recon, z_disc), z_x=(x-mu_trainNormal)/sigma_trainNormal  (leak-free)
Report per dataset whether ANY max variant beats the sum baseline.
"""
import numpy as np, json, os, sys
from types import SimpleNamespace
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
from mae_anomaly.evaluator import compute_full_metric_set

RUN = "/home/ykio/notebooks/TSMAE/results/experiments/271_20260602_020545_271canon_baseline"
EPS = 1e-4
MAXW = [0.25, 0.5, 1, 2, 4]

def regions_from(lb):
    d = np.diff(np.concatenate([[0], (lb > 0).astype(int), [0]]))
    s = np.where(d == 1)[0]; e = np.where(d == -1)[0] - 1
    return [SimpleNamespace(start=int(a), end=int(b)) for a, b in zip(s, e)]

def pak(sc, lb, regs):
    return compute_full_metric_set(sc.astype(np.float64), lb, regs, eval_mask=None, lite=True)['pak_auc_f1']

def analyze(sub):
    d = f"{RUN}/{sub}"
    meta = json.load(open(f"{d}/experiment_metadata.json"))
    be = meta['timing']['best_epoch']
    tz = np.load(f"{d}/epoch_scores/epoch_{be:03d}_scores.npz")
    recon = tz['teacher_recon_error'].astype(np.float64); disc = tz['discrepancy_error'].astype(np.float64)
    lb = tz['point_labels'].astype(int); regs = regions_from(lb)
    rm = recon.mean() + EPS; dm = disc.mean() + EPS
    sd = disc * (rm / dm)
    base = pak(recon + sd / 4.0, lb, regs)
    # A: mean-matched max
    maxA = {w: pak(np.maximum(recon, w * sd), lb, regs) for w in MAXW}
    bwA = max(maxA, key=maxA.get)
    # B: train-normal z-score max (leak-free)
    tr_path = f"{d}/best_epoch_train_scores.npz"
    if not os.path.exists(tr_path) and 'excl22' in sub:
        tr_path = f"{RUN}/{sub.replace('A1A2_excl22','A1A2_full')}/best_epoch_train_scores.npz"
    tr = np.load(tr_path); trl = tr['point_labels'].astype(int)
    rn = tr['teacher_recon_error'].astype(np.float64)[trl == 0]
    dn = tr['discrepancy_error'].astype(np.float64)[trl == 0]
    zr = (recon - rn.mean()) / (rn.std() + 1e-12)
    zd = (disc - dn.mean()) / (dn.std() + 1e-12)
    zmax = pak(np.maximum(zr, zd), lb, regs)
    # zsum for reference (z_recon + z_disc)
    zsum = pak(zr + zd, lb, regs)
    return dict(sub=sub, base=base, maxA_best=maxA[bwA], maxA_w=bwA, maxA1=maxA[1],
                zmax=zmax, zsum=zsum)

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'subset'
    SINGLE = ['SWaT/A1A2_full', 'SWaT/A1A2_excl22', 'WaDi/A1', 'WaDi/A2', 'PSM']
    SMD = [f'SMD/machine-{m}' for m in ['1-2','1-3','1-4','1-5','1-6','1-7','1-8','2-1','2-3','2-4','2-5','2-6','2-7','3-1','3-2','3-3','3-4','3-5','3-6','3-7','3-9','3-10']]
    SMAP = [f'SMAP/{e}' for e in ['G-7','P-1','P-4','T-1','T-3']]
    MSL = [f'MSL/{e}' for e in ['C-1','C-2','F-7','P-11','T-13']]
    targets = (SINGLE + SMD[:3] + SMAP[:2] + MSL[:2]) if which == 'subset' else (SINGLE + SMD + SMAP + MSL)
    print(f"{'dataset':20s} sum_base  maxA(w)   Δ        zmax     Δ       zsum     Δ")
    rows = []
    for t in targets:
        try: r = analyze(t); rows.append(r)
        except Exception as e:
            print(f"{t:20s} ERR {e}"); continue
        print(f"{t:20s} {r['base']:.4f}  {r['maxA_best']:.4f}(w{r['maxA_w']:<4}) {r['maxA_best']-r['base']:+.4f}  "
              f"{r['zmax']:.4f} {r['zmax']-r['base']:+.4f}  {r['zsum']:.4f} {r['zsum']-r['base']:+.4f}")
    if rows:
        import numpy as _np
        def mΔ(k): return _np.mean([r[k]-r['base'] for r in rows])
        def wins(k): return sum(1 for r in rows if r[k] > r['base']+1e-4)
        n = len(rows)
        print(f"\nMEAN over {n}: base={_np.mean([r['base'] for r in rows]):.4f}")
        print(f"  maxA(mean-match, best-w): Δ{mΔ('maxA_best'):+.4f}  win {wins('maxA_best')}/{n}")
        print(f"  zmax (train-normal z)   : Δ{mΔ('zmax'):+.4f}  win {wins('zmax')}/{n}")
        print(f"  zsum (z_recon+z_disc)   : Δ{mΔ('zsum'):+.4f}  win {wins('zsum')}/{n}")
        json.dump(rows, open('/home/ykio/notebooks/TSMAE/temp/max_combine_results.json','w'), indent=1)
