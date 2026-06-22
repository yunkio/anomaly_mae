"""(1) Does a per-dataset dynamic score_recon_disc_ratio (from train-side disc_SNR/recon_SNR)
beat the fixed 4.0? Uses ONLY saved exp271 data (no inference).

score = recon + scaled_disc/ratio, scaled_disc = disc*(recon_mean/disc_mean)  [scoring.py faithful]
SNR_x = (mu_x[anom]-mu_x[norm])/(sigma_x[anom]+sigma_x[norm]+eps)  on TRAIN (leak-free)
dynamic ratio = clamp(recon_SNR/disc_SNR)  [weight each component by its discriminability]
oracle = best PAK_F1 over a ratio grid (upper bound / "is there room").
"""
import numpy as np, json, os, glob, sys
from types import SimpleNamespace
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
from mae_anomaly.evaluator import compute_full_metric_set

RUN = "/home/ykio/notebooks/TSMAE/results/experiments/271_20260602_020545_271canon_baseline"
EPS = 1e-4
GRID = [0.1, 0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 40]

def regions_from(lb):
    d = np.diff(np.concatenate([[0], (lb > 0).astype(int), [0]]))
    s = np.where(d == 1)[0]; e = np.where(d == -1)[0] - 1
    return [SimpleNamespace(start=int(a), end=int(b)) for a, b in zip(s, e)]

def snr(x, lb):
    a = x[lb > 0]; n = x[lb == 0]
    if len(a) < 5 or len(n) < 5: return None
    return float((a.mean() - n.mean()) / (a.std() + n.std() + 1e-9))

def score(recon, disc, ratio):
    rm = recon.mean() + EPS; dm = disc.mean() + EPS
    return recon + (disc * (rm / dm)) / ratio

def pak(sc, lb, regs):
    return compute_full_metric_set(sc.astype(np.float64), lb, regs, eval_mask=None, lite=True)['pak_auc_f1']

def analyze(sub):
    d = f"{RUN}/{sub}"
    meta = json.load(open(f"{d}/experiment_metadata.json"))
    be = meta['timing']['best_epoch']; rep = meta['metrics']['pak_auc_f1']
    tz = np.load(f"{d}/epoch_scores/epoch_{be:03d}_scores.npz")
    recon = tz['teacher_recon_error'].astype(np.float64); disc = tz['discrepancy_error'].astype(np.float64)
    lb = tz['point_labels'].astype(int); regs = regions_from(lb)
    # train SNRs (excl22 shares training with _full -> use sibling train scores)
    tr_path = f"{d}/best_epoch_train_scores.npz"
    if not os.path.exists(tr_path) and 'excl22' in sub:
        tr_path = f"{RUN}/{sub.replace('A1A2_excl22','A1A2_full')}/best_epoch_train_scores.npz"
    tr = np.load(tr_path); trl = tr['point_labels'].astype(int)
    rsnr = snr(tr['teacher_recon_error'].astype(np.float64), trl)
    dsnr = snr(tr['discrepancy_error'].astype(np.float64), trl)
    # TEST SNRs (diagnostic / leaky)
    rsnr_te = snr(recon, lb); dsnr_te = snr(disc, lb)
    def dratio(rs, ds):
        return float(np.clip(rs / ds, 0.1, 40)) if (rs and ds and ds > 0) else 4.0
    base = pak(score(recon, disc, 4.0), lb, regs)
    dyn_ratio = dratio(rsnr, dsnr); dyn = pak(score(recon, disc, dyn_ratio), lb, regs)
    dyn_te_ratio = dratio(rsnr_te, dsnr_te); dyn_te = pak(score(recon, disc, dyn_te_ratio), lb, regs)
    grid = {r: pak(score(recon, disc, r), lb, regs) for r in GRID}
    orc_r = max(grid, key=grid.get); orc = grid[orc_r]
    return dict(sub=sub, best_ep=be, rep=rep, base=base, rsnr=rsnr, dsnr=dsnr,
               dyn_ratio=dyn_ratio, dyn=dyn, dyn_te_ratio=dyn_te_ratio, dyn_te=dyn_te,
               rsnr_te=rsnr_te, dsnr_te=dsnr_te, orc_ratio=orc_r, orc=orc, grid=grid)

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'subset'
    SINGLE = ['SWaT/A1A2_full', 'SWaT/A1A2_excl22', 'WaDi/A1', 'WaDi/A2', 'PSM']
    SMD = [f'SMD/machine-{m}' for m in ['1-2','1-3','1-4','1-5','1-6','1-7','1-8','2-1','2-3','2-4','2-5','2-6','2-7','3-1','3-2','3-3','3-4','3-5','3-6','3-7','3-9','3-10']]
    SMAP = [f'SMAP/{e}' for e in ['G-7','P-1','P-4','T-1','T-3']]
    MSL = [f'MSL/{e}' for e in ['C-1','C-2','F-7','P-11','T-13']]
    if which == 'subset':
        targets = SINGLE + SMD[:3] + SMAP[:2] + MSL[:2]
    else:
        targets = SINGLE + SMD + SMAP + MSL
    print(f"{'dataset':20s} ep  base   trSNR(r/d) dynTr Δ      teSNR(r/d) dynTe Δ      orcR orc    Δorc")
    rows = []
    for t in targets:
        try:
            r = analyze(t); rows.append(r)
        except Exception as e:
            print(f"{t:20s} ERR {e}"); continue
        print(f"{t:20s} {r['best_ep']:3d} {r['base']:.4f} "
              f"{(r['rsnr'] or 0):.2f}/{(r['dsnr'] or 0):.2f} {r['dyn_ratio']:5.1f} {r['dyn']-r['base']:+.4f}  "
              f"{(r['rsnr_te'] or 0):.2f}/{(r['dsnr_te'] or 0):.2f} {r['dyn_te_ratio']:5.1f} {r['dyn_te']-r['base']:+.4f}  "
              f"{r['orc_ratio']:4.1f} {r['orc']:.4f} {r['orc']-r['base']:+.4f}")
    if rows:
        import numpy as _np
        def mΔ(k): return _np.mean([r[k]-r['base'] for r in rows])
        def wins(k): return sum(1 for r in rows if r[k] > r['base']+1e-4)
        print(f"\nMEAN over {len(rows)}: base={_np.mean([r['base'] for r in rows]):.4f}")
        print(f"  dyn_train: Δ{mΔ('dyn'):+.4f}  win {wins('dyn')}/{len(rows)}")
        print(f"  dyn_test : Δ{mΔ('dyn_te'):+.4f}  win {wins('dyn_te')}/{len(rows)}  (leaky diagnostic)")
        print(f"  oracle   : Δ{mΔ('orc'):+.4f}  win {wins('orc')}/{len(rows)}  (upper bound)")
        import json as _json
        _json.dump([{k:v for k,v in r.items() if k!='grid'} for r in rows],
                   open('/home/ykio/notebooks/TSMAE/temp/dynamic_ratio_results.json','w'), indent=1)
