#!/usr/bin/env python
"""DRY-RUN (read-only): determine best-epoch FLIPS under the new 4:1 / no-FM score.

New adaptive score:  score = recon + 0.25 * scaled_disc
  scaled_disc = disc * (recon.mean()+eps)/(disc.mean()+eps)   (scale corr kept)
  FM removed from the score.  Pre-warmup (ep<=warmup) stays recon-only (== stored).

For each of the 9 completed runs x 5 cells:
  - post-warmup epochs: recompute pak_auc_f1 on NEW score (lite=True, fast).
  - pre-warmup epochs:  pak_auc_f1 unchanged (recon-only) -> use stored.
  - new_best = argmax;  flip = (new_best != stored timing.best_epoch).
Faithfulness gate: reconstruct OLD score from components -> recompute OLD
  pak_auc_f1 -> compare to stored row (must be ~0).  No writes.
"""
import os, sys, json, glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, '/home/ykio/notebooks/TSMAE')

RUNS = ['271_20260529_100418_271canon_baseline', '271_lr2_20260529_225351_baseline',
        '271_lr_20260529_225351_baseline', '274_20260529_100418_274canon_balsamp',
        '274_lr2_20260529_225351_balsamp', '274_lr_20260529_225351_balsamp',
        '285_20260529_225351_no_fm', '286_20260529_225351_clamp_pm4',
        '287_20260529_225351_unmask']
DATASETS = ['PSM', 'SWaT/A1A2_full', 'SWaT/A1A2_excl22', 'WaDi/A1', 'WaDi/A2']
EPS = 1e-4
N_THRESHOLDS = 200     # AUTHORITATIVE — matches production per-epoch eval (run_base L588)
SLIDING_WINDOW = 100
DISC_FACTOR = 0.25  # recon:disc = 4:1


def regions_from_labels(lbl):
    lbl = np.asarray(lbl).astype(int); regs = []; i, n = 0, len(lbl)
    while i < n:
        if lbl[i] == 1:
            j = i
            while j < n and lbl[j] == 1: j += 1
            regs.append((i, j)); i = j
        else: i += 1
    return regs


def _task(t):
    """t=(cell_idx, ep, npz_path, is_excl22, region22, want_old, use_fm).
    Returns (cell_idx, ep, new_pak, old_pak_or_None)."""
    import numpy as _np
    from mae_anomaly.evaluator import compute_full_metric_set, compute_metrics_with_exclusion
    class _R:
        __slots__ = ('start', 'end')
        def __init__(s, a, b): s.start = a; s.end = b
    ci, ep, p, is_excl, r22, want_old, use_fm = t
    d = _np.load(p)
    recon = d['teacher_recon_error'].astype(_np.float64)
    disc = d['discrepancy_error'].astype(_np.float64)
    fm = d['fm_error'].astype(_np.float64) if 'fm_error' in d.files else None
    lbl = d['point_labels'].astype(int)
    ml = min(len(recon), len(disc), len(lbl))
    recon, disc, lbl = recon[:ml], disc[:ml], lbl[:ml]
    if fm is not None: fm = fm[:ml]
    rmean = recon.mean() + EPS
    scaled_disc = disc * (rmean / (disc.mean() + EPS))
    new_score = recon + DISC_FACTOR * scaled_disc
    regs = [_R(a, b) for a, b in regions_from_labels(lbl)]

    def _pak(score):
        if is_excl and r22 is not None:
            m = compute_metrics_with_exclusion(score, lbl, regs, _R(r22[0], r22[1]), lite=True)
        else:
            m = compute_full_metric_set(score, lbl, regs, n_thresholds=N_THRESHOLDS,
                                        sliding_window=SLIDING_WINDOW, lite=True)
        return float(m['pak_auc_f1'])
    new_pak = _pak(new_score)
    old_pak = None
    if want_old:
        if use_fm and fm is not None:
            scaled_fm = fm * (rmean / (fm.mean() + EPS))
            old_score = recon + 0.5 * scaled_disc + 0.5 * scaled_fm
        else:
            old_score = recon + scaled_disc
        old_pak = _pak(old_score)
    return ci, ep, new_pak, old_pak


def main():
    cells = []
    for r in RUNS:
        for ds in DATASETS:
            cd = f'results/experiments/{r}/{ds}'
            em = f'{cd}/epoch_metrics.json'
            mp = f'{cd}/experiment_metadata.json'
            if not os.path.exists(em) or not os.path.exists(mp):
                cells.append({'run': r, 'ds': ds, 'missing': True}); continue
            meta = json.load(open(mp))
            cfg = meta.get('config', {})
            warmup = int(cfg.get('teacher_only_warmup_epochs', 250))
            use_fm = bool(cfg.get('use_feature_matching', True))
            old_best = meta.get('timing', {}).get('best_epoch')
            is_excl = ds.endswith('excl22')
            r22 = None
            info = meta.get('excl_region22_info')
            if is_excl and isinstance(info, dict):
                r22 = (int(info['region_start']), int(info['region_end']))
            # full SWaT excl region also needed? no — full cell uses full metric.
            rows = {int(x['epoch']): x for x in json.load(open(em))['epochs']}
            npz = {int(os.path.basename(p).split('_')[1]): p
                   for p in glob.glob(f'{cd}/epoch_scores/epoch_*_scores.npz')}
            cells.append({'run': r, 'ds': ds, 'cell_dir': cd, 'warmup': warmup,
                          'use_fm': use_fm, 'old_best': old_best, 'is_excl': is_excl,
                          'r22': r22, 'rows': rows, 'npz': npz, 'missing': False})

    nlive = sum(1 for c in cells if not c.get('missing'))
    print(f"cells={nlive} (AUTHORITATIVE n_thresholds={N_THRESHOLDS}, pak_auc_f1 only, new 4:1/no-FM)", flush=True)
    print(f"{'CELL':<46}{'old_best':>9}{'new_best':>9}{'flip':>6}{'newpre?':>8}{'oldfaith':>11}", flush=True)
    print("-" * 92, flush=True)

    out = []
    with ProcessPoolExecutor(max_workers=14) as ex:
        for ci, c in enumerate(cells):
            label = (c['run'].split('_2026')[0] + '/' + c['ds'])
            if c.get('missing'):
                print(f"{label:<46}{'MISSING':>9}", flush=True); continue
            warmup = c['warmup']; rows = c['rows']
            post = sorted(e for e in c['npz'] if e > warmup)
            tasks = [(ci, ep, c['npz'][ep], c['is_excl'], c['r22'], k == 0, c['use_fm'])
                     for k, ep in enumerate(post)]
            new_pak = {}; faith = None
            for _ci, ep, npak, opak in ex.map(_task, tasks):
                new_pak[ep] = npak
                if opak is not None: faith = (ep, opak)
            cand = []
            for ep, row in rows.items():
                if ep <= warmup:
                    v = row.get('pak_auc_f1')
                    cand.append((ep, float(v) if v is not None else -1.0))
                elif ep in new_pak:
                    cand.append((ep, new_pak[ep]))
            if not cand:
                print(f"{label:<46}{'NOCAND':>9}", flush=True); continue
            nb_ep, nb_val = max(cand, key=lambda x: x[1])
            flip = (c['old_best'] is not None and nb_ep != c['old_best'])
            chk = 'n/a'
            if faith is not None:
                stored = rows.get(faith[0], {}).get('pak_auc_f1')
                if stored is not None: chk = f"{abs(faith[1]-float(stored)):.1e}"
            print(f"{label:<46}{str(c['old_best']):>9}{str(nb_ep):>9}"
                  f"{('FLIP' if flip else '-'):>6}{('YES' if nb_ep<=warmup else 'no'):>8}{chk:>11}",
                  flush=True)
            out.append({'run': c['run'], 'ds': c['ds'], 'old_best': c['old_best'],
                        'new_best': nb_ep, 'new_best_val': round(nb_val, 6), 'flip': flip,
                        'new_best_prewarmup': nb_ep <= warmup, 'use_fm': c['use_fm'],
                        'oldpak_faith_delta': chk})
            json.dump(out, open('/tmp/lambda_bestepoch_classification.json', 'w'), indent=2)

    flips = [o for o in out if o['flip']]
    noflip = [o for o in out if not o['flip']]
    print("-" * 92, flush=True)
    print(f"TOTAL cells={len(out)}  RETRAIN(flip)={len(flips)}  RECOMPUTE-ONLY(no-flip)={len(noflip)}", flush=True)
    fa = [float(o['oldpak_faith_delta']) for o in out if o['oldpak_faith_delta'] not in ('n/a',)]
    print(f"worst OLD-recompute faithfulness delta @nt200 (must be ~0): {max(fa) if fa else 0:.1e}", flush=True)
    print("\n### RETRAIN NEEDED (best epoch FLIPPED) ###", flush=True)
    for o in flips:
        print(f"  {o['run'].split('_2026')[0]}/{o['ds']:<16} best {o['old_best']} -> {o['new_best']}"
              f"  (newpak={o['new_best_val']})", flush=True)
    print("\n### RECOMPUTE-ONLY (best epoch unchanged) ###", flush=True)
    for o in noflip:
        print(f"  {o['run'].split('_2026')[0]}/{o['ds']:<16} best={o['new_best']} (stable)", flush=True)
    print("\nsaved /tmp/lambda_bestepoch_classification.json", flush=True)


if __name__ == '__main__':
    main()
