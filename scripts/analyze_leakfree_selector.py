"""Solve the per-entity channel mis-scoring (MSL/C-1: disc=0.95 but sum=0.45) using ONLY
train-collected values + test UNLABELED values (no labels, no future where marked).

Phase 1 (this script): per entity compute
  - candidate scores' PAK: recon, disc, sum4(current), sum1, max   (oracle = best)
  - leak-free signals per channel (recon / disc):
      STRICT (train only, fully causal/deployable):
        trAUC   = roc_auc on TRAIN sparse labels (rank separability, robust vs SNR)
        trNcv   = train-normal coeff-of-variation (sigma/mu) -- channel stability on normals
      TRANSDUCTIVE (test unlabeled, uses whole test batch -> not strictly causal; marked):
        kurt, skew                 = test distribution shape (heavy/right tail = detecting)
        p99z, tailmass             = test tail vs TRAIN-NORMAL baseline (excess above normal)
        coh                        = temporal coherence of top-10% test points (anomalies are runs)
Saves temp/leakfree_signals.json for Phase 2 (selector design, cheap).
"""
import numpy as np, json, os, sys
from types import SimpleNamespace
from scipy import stats as ss
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
from mae_anomaly.evaluator import compute_full_metric_set

RUN = "/home/ykio/notebooks/TSMAE/results/experiments/271_20260602_020545_271canon_baseline"
EPS = 1e-4

def regions_from(lb):
    d = np.diff(np.concatenate([[0], (lb > 0).astype(int), [0]]))
    s = np.where(d == 1)[0]; e = np.where(d == -1)[0] - 1
    return [SimpleNamespace(start=int(a), end=int(b)) for a, b in zip(s, e)]

def pak(sc, lb, regs):
    return compute_full_metric_set(sc.astype(np.float64), lb, regs, eval_mask=None, lite=True)['pak_auc_f1']

def chan_signals(test_x, tr_x, trl):
    """leak-free signals for one channel."""
    tn = tr_x[trl == 0]
    mu, sd = float(tn.mean()), float(tn.std() + 1e-12)
    out = {}
    # STRICT train-only
    out['trAUC'] = float(roc_auc_score(trl, tr_x)) if (trl.sum() > 0 and trl.sum() < len(trl)) else 0.5
    out['trNcv'] = sd / (abs(mu) + 1e-12)
    # TRANSDUCTIVE test-unlabeled
    out['kurt'] = float(ss.kurtosis(test_x))
    out['skew'] = float(ss.skew(test_x))
    out['p99z'] = (float(np.percentile(test_x, 99)) - mu) / sd
    out['tailmass'] = float((test_x > mu + 4 * sd).mean())
    k = max(1, int(0.10 * len(test_x)))
    thr = np.partition(test_x, -k)[-k]
    b = (test_x >= thr).astype(float)
    out['coh'] = float(np.corrcoef(b[:-1], b[1:])[0, 1]) if b.std() > 0 else 0.0
    return out

def analyze(sub):
    d = f"{RUN}/{sub}"
    meta = json.load(open(f"{d}/experiment_metadata.json")); be = meta['timing']['best_epoch']
    z = np.load(f"{d}/epoch_scores/epoch_{be:03d}_scores.npz")
    recon = z['teacher_recon_error'].astype(np.float64); disc = z['discrepancy_error'].astype(np.float64)
    lb = z['point_labels'].astype(int); regs = regions_from(lb)
    sd = disc * ((recon.mean() + EPS) / (disc.mean() + EPS))
    cand = {'recon': recon, 'disc': sd, 'sum4': recon + sd / 4, 'sum1': recon + sd, 'max': np.maximum(recon, sd)}
    paks = {k: pak(v, lb, regs) for k, v in cand.items()}
    # train signals
    tr_path = f"{d}/best_epoch_train_scores.npz"
    if not os.path.exists(tr_path) and 'excl22' in sub:
        tr_path = f"{RUN}/{sub.replace('A1A2_excl22','A1A2_full')}/best_epoch_train_scores.npz"
    tr = np.load(tr_path); trl = tr['point_labels'].astype(int)
    sig = {'recon': chan_signals(recon, tr['teacher_recon_error'].astype(np.float64), trl),
           'disc':  chan_signals(disc,  tr['discrepancy_error'].astype(np.float64), trl)}
    return dict(sub=sub, paks=paks, sig=sig)

if __name__ == '__main__':
    SINGLE = ['SWaT/A1A2_full', 'SWaT/A1A2_excl22', 'WaDi/A1', 'WaDi/A2', 'PSM']
    SMD = [f'SMD/machine-{m}' for m in ['1-2','1-3','1-4','1-5','1-6','1-7','1-8','2-1','2-3','2-4','2-5','2-6','2-7','3-1','3-2','3-3','3-4','3-5','3-6','3-7','3-9','3-10']]
    SMAP = [f'SMAP/{e}' for e in ['G-7','P-1','P-4','T-1','T-3']]
    MSL = [f'MSL/{e}' for e in ['C-1','C-2','F-7','P-11','T-13']]
    targets = SINGLE + SMD + SMAP + MSL
    rows = []
    for t in targets:
        try:
            r = analyze(t); rows.append(r)
            p = r['paks']
            print(f"{t:20s} recon={p['recon']:.3f} disc={p['disc']:.3f} sum4={p['sum4']:.3f} "
                  f"sum1={p['sum1']:.3f} max={p['max']:.3f} | best={max(p,key=p.get)}", flush=True)
        except Exception as e:
            print(f"{t:20s} ERR {e}", flush=True)
    json.dump(rows, open('/home/ykio/notebooks/TSMAE/temp/leakfree_signals.json', 'w'), indent=1)
    print(f"\nsaved {len(rows)} entities -> temp/leakfree_signals.json")
