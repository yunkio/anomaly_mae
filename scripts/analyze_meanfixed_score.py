#!/usr/bin/env python
"""mean-fixed vs official-causal score comparison (no inference, saved npz only).

Concern: official s_t = (R_tr + cumsum(recon)) / (D_tr + cumsum(disc) + eps) DECREASES
within a long fault as disc accumulates in the denominator, so disc_t * s_t distorts disc's
clean point ranking. mean-fixed freezes the multiplier to the train-normal mean ratio:
    score_t = recon_t + (mu_R / mu_D) * disc_t          (mu_R/mu_D == R_tr/D_tr)
We also report the 0.25-matched variant (recon + 0.25*(R_tr/D_tr)*disc) which isolates the
time-varying-vs-frozen effect at the official's own disc weight.

R_tr/D_tr are NOT stored; recovered EXACTLY from official_score via linear least squares:
    y_t = official - recon = 0.25*disc*(R_tr+Cr)/(D_tr+Cd)
    => y_t*D_tr - 0.25*disc*R_tr = 0.25*disc*Cr - y_t*Cd   (linear in D_tr,R_tr; no division)
"""
import numpy as np, json, os, glob, sys
from types import SimpleNamespace
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
from mae_anomaly.evaluator import compute_full_metric_set

W = 0.25; EPS = 1e-8; WARM = 15
ROOT = 'results/experiments/official/271_20260622_164359_30ep_42'
CELLS = [('PSM', 'PSM'), ('SWaT_full', 'SWaT/A1A2_full'),
         ('WaDi_A1', 'WaDi/A1'), ('WaDi_A2', 'WaDi/A2')]


def regions(lb):
    lb = (np.asarray(lb) == 1).astype(int)
    d = np.diff(np.concatenate([[0], lb, [0]]))
    s = np.where(d == 1)[0]; e = np.where(d == -1)[0] - 1
    return [SimpleNamespace(start=int(a), end=int(b)) for a, b in zip(s, e)]


def pak(score, lb, regs):
    return compute_full_metric_set(score.astype(np.float64), lb, regs, eval_mask=None,
                                   lite=True)['pak_auc_f1']


def recover_seed(recon, disc, offi):
    Cr = np.cumsum(recon); Cd = np.cumsum(disc)
    y = offi - recon
    A = np.column_stack([y, -W * disc]); b = W * disc * Cr - y * Cd
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(sol[1]), float(sol[0])  # R_tr, D_tr


def scores_at(npz):
    z = np.load(npz)
    recon = z['teacher_recon_error'].astype(np.float64)
    disc = z['discrepancy_error'].astype(np.float64)
    offi = z['official_score'].astype(np.float64)
    lb = z['point_labels'].astype(int)
    R_tr, D_tr = recover_seed(recon, disc, offi)
    ratio = R_tr / D_tr
    return {
        'lb': lb, 'ratio': ratio,
        'official': offi,                                  # current (time-varying s_t)
        'meanfixed': recon + ratio * disc,                 # user formula (R_tr/D_tr coeff)
        'meanfixed_025': recon + W * ratio * disc,         # 0.25-matched (isolate s_t freeze)
        'recon': recon, 'disc': disc,
    }


def best_post_epoch(cell_dir):
    em = json.load(open(os.path.join(cell_dir, 'epoch_metrics.json')))['epochs']
    post = [e for e in em if int(e['epoch']) > WARM] or em
    return max(post, key=lambda e: e.get('pak_auc_f1', 0))['epoch']


def main():
    print(f"{'dataset':10s} {'ep':>3} {'ratio(R/D)':>11} | "
          f"{'official':>9} {'mean-fix':>9} {'mf·0.25':>9} {'recon':>7} {'disc':>7}")
    print('-' * 84)
    rows = []
    for name, sub in CELLS:
        cd = os.path.join(ROOT, sub)
        if not os.path.exists(os.path.join(cd, 'epoch_metrics.json')):
            print(f"{name:10s}  (no epoch_metrics)"); continue
        be = best_post_epoch(cd)
        npz = os.path.join(cd, 'epoch_scores', f'epoch_{be:03d}_scores.npz')
        if not os.path.exists(npz):
            print(f"{name:10s}  (no npz ep{be})"); continue
        S = scores_at(npz); regs = regions(S['lb'])
        m = {k: pak(S[k], S['lb'], regs) for k in
             ['official', 'meanfixed', 'meanfixed_025', 'recon', 'disc']}
        print(f"{name:10s} {be:>3} {S['ratio']:>11.5f} | "
              f"{m['official']:>9.4f} {m['meanfixed']:>9.4f} {m['meanfixed_025']:>9.4f} "
              f"{m['recon']:>7.4f} {m['disc']:>7.4f}")
        rows.append((name, m))
    print('-' * 84)
    print("Δ(mean-fixed − official):  " +
          "  ".join(f"{n}={m['meanfixed']-m['official']:+.4f}" for n, m in rows))
    print("Δ(mf·0.25  − official):    " +
          "  ".join(f"{n}={m['meanfixed_025']-m['official']:+.4f}" for n, m in rows))

    # per-method best over ALL post-warmup epochs (each method at its own best epoch)
    print("\n=== 각 방법의 post-warmup 최적 epoch에서 best pak_auc_f1 ===")
    print(f"{'dataset':10s} | {'official':>16} {'mean-fixed':>16} {'mf·0.25':>16}")
    for name, sub in CELLS:
        cd = os.path.join(ROOT, sub)
        npzs = sorted(glob.glob(os.path.join(cd, 'epoch_scores', 'epoch_*_scores.npz')))
        npzs = [p for p in npzs if int(os.path.basename(p)[6:9]) > WARM]
        if not npzs:
            continue
        bestm = {k: (-1, 0) for k in ['official', 'meanfixed', 'meanfixed_025']}
        for p in npzs:
            ep = int(os.path.basename(p)[6:9])
            S = scores_at(p); regs = regions(S['lb'])
            for k in bestm:
                v = pak(S[k], S['lb'], regs)
                if v > bestm[k][0]:
                    bestm[k] = (v, ep)
        print(f"{name:10s} | " + " ".join(
            f"{bestm[k][0]:.4f}@ep{bestm[k][1]:<2}" .rjust(16) for k in
            ['official', 'meanfixed', 'meanfixed_025']))


if __name__ == '__main__':
    main()
