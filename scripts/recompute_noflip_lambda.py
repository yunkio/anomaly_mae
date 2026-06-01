#!/usr/bin/env python
"""Recompute score + ALL metrics (VUS included) for NO-FLIP canonical cells
under the 2026-06-01 4:1 / no-FM lambda change.

Scope: canonical experiments {271,274,285,286,287}, cells whose best epoch did
NOT flip (from /tmp/lambda_bestepoch_classification.json). lr/lr2 + 288 + every
flipped cell are EXCLUDED (flipped -> retrain queue; lr/lr2 -> untouched).

Only POST-warmup rows change (pre-warmup score is recon-only in both old and
new formula -> identical, already carry VUS -> left untouched). For each
post-warmup epoch:
  - new score = mae_anomaly.scoring.compute_adaptive_components(... )['score']
                (single source: recon + scaled_disc/ratio, ratio=4, no FM)
  - metrics  = compute_full_metric_set(lite=False)  (VUS-PR/ROC included)
               / compute_metrics_with_exclusion(lite=False) for excl22
  - npz['adaptive_score'] := new score   (skip symlinked excl22 dir)
  - epoch_metrics post-warmup row := new metrics
Best epoch is unchanged (no-flip) -> experiment_metadata['metrics'] recomputed
at the SAME best epoch with the new score (lite=False). best_checkpoint.pt stays
valid -> model-forward viz remain correct.

FAITHFULNESS GATE (per cell, before any write): reconstruct the OLD score from
the npz components and recompute its pak_auc_f1 (nt=200) -> must match the stored
row (d < 2e-3). Proves the offline pipeline reproduces production before we
overwrite anything.

Usage:
  python scripts/recompute_noflip_lambda.py            # dry-run (plan only)
  python scripts/recompute_noflip_lambda.py --apply
  python scripts/recompute_noflip_lambda.py --apply --workers 12
"""
import os, sys, json, glob, shutil, argparse, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, '/home/ykio/notebooks/TSMAE')

CLASS = '/tmp/lambda_bestepoch_classification.json'
BACKUP_ROOT = '/home/ykio/notebooks/TSMAE/.trash/0601/lambda'
CANON = {'271', '274', '285', '286', '287'}
N_THRESHOLDS = 200
SLIDING_WINDOW = 100
REPORT = '/tmp/lambda_recompute_report.json'


def base_of(run):
    p = run.split('_')
    return p[0] if p[1].startswith('2026') else f"{p[0]}_{p[1]}"


def regions_from_labels(lbl):
    lbl = np.asarray(lbl).astype(int); regs = []; i, n = 0, len(lbl)
    while i < n:
        if lbl[i] == 1:
            j = i
            while j < n and lbl[j] == 1: j += 1
            regs.append((i, j)); i = j
        else: i += 1
    return regs


def load_json(p):
    with open(p) as f: return json.load(f)


def backup_file(src, cell_rel):
    dst = os.path.join(BACKUP_ROOT, cell_rel, os.path.relpath(src, _cell_dirs[cell_rel]))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


_cell_dirs = {}


# ---- worker (module-level, picklable) ----
def _recompute_one(task):
    import numpy as _np
    from mae_anomaly.evaluator import compute_full_metric_set, compute_metrics_with_exclusion
    from mae_anomaly.scoring import compute_adaptive_components
    class _R:
        __slots__ = ('start', 'end')
        def __init__(s, a, b): s.start = a; s.end = b
    npz_path, cfg, region22, is_excl, want_oldcheck = task
    ep = int(os.path.basename(npz_path).split('_')[1])
    d = _np.load(npz_path)
    recon = d['teacher_recon_error'].astype(_np.float64)
    disc = d['discrepancy_error'].astype(_np.float64)
    fm = d['fm_error'].astype(_np.float64) if 'fm_error' in d.files else None
    lbl = d['point_labels'].astype(int)
    ml = min(len(recon), len(disc), len(lbl))
    recon, disc, lbl = recon[:ml], disc[:ml], lbl[:ml]
    if fm is not None: fm = fm[:ml]
    # NEW score via single source (post-warmup -> force_recon_only=False)
    new_score = compute_adaptive_components(recon, disc, fm, cfg, force_recon_only=False)['score']
    regs = [_R(a, b) for a, b in regions_from_labels(lbl)]
    if is_excl and region22 is not None:
        m = compute_metrics_with_exclusion(new_score.astype(_np.float64), lbl, regs,
                                            _R(region22[0], region22[1]), lite=False)
    else:
        m = compute_full_metric_set(new_score.astype(_np.float64), lbl, regs,
                                    n_thresholds=N_THRESHOLDS, sliding_window=SLIDING_WINDOW, lite=False)
    out = {k: (float(v) if isinstance(v, (int, float, _np.floating)) else v) for k, v in m.items()}
    old_pak = None
    if want_oldcheck:
        eps = 1e-4; rm = recon.mean() + eps
        sd = disc * (rm / (disc.mean() + eps))
        use_fm = bool(cfg.get('use_feature_matching', True)) and fm is not None
        old_score = recon + (0.5 * sd + 0.5 * fm * (rm / (fm.mean() + eps))) if use_fm else recon + sd
        if is_excl and region22 is not None:
            om = compute_metrics_with_exclusion(old_score.astype(_np.float64), lbl, regs,
                                                _R(region22[0], region22[1]), lite=True)
        else:
            om = compute_full_metric_set(old_score.astype(_np.float64), lbl, regs,
                                         n_thresholds=N_THRESHOLDS, sliding_window=SLIDING_WINDOW, lite=True)
        old_pak = float(om['pak_auc_f1'])
    return ep, out, new_score.astype(np.float32), old_pak


def region22_of(cell_dir, ds):
    if 'SWaT' not in ds: return None
    mp = os.path.join(cell_dir, 'experiment_metadata.json')
    if not os.path.exists(mp): return None
    info = load_json(mp).get('excl_region22_info')
    if isinstance(info, dict):
        return (int(info['region_start']), int(info['region_end']))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--only', default=None)
    args = ap.parse_args()

    cls = load_json(CLASS)
    # SWaT full + excl22 SHARE one training (symlinked npz). If EITHER SWaT cell
    # of an experiment flipped, BOTH must go to retrain (the shared model is
    # re-trained), so neither is recompute-eligible. Build the set of runs whose
    # SWaT flipped.
    swat_flip_runs = {c['run'] for c in cls if c['flip'] and 'SWaT' in c['ds']}
    # no-flip canonical cells, with SWaT coupling applied
    targets = []
    for c in cls:
        if c['flip'] or base_of(c['run']) not in CANON:
            continue
        if 'SWaT' in c['ds'] and c['run'] in swat_flip_runs:
            continue  # sibling SWaT cell flipped -> whole SWaT retrains
        targets.append(c)
    if args.only:
        targets = [c for c in targets if f"{c['run']}/{c['ds']}" == args.only]

    # resume-skip: cells already recomputed (idempotent restart)
    def _done(c):
        mp = f"results/experiments/{c['run']}/{c['ds']}/experiment_metadata.json"
        return os.path.exists(mp) and load_json(mp).get('_lambda_recompute', {}).get('applied', False)
    skipped_done = [c for c in targets if _done(c)]
    targets = [c for c in targets if not _done(c)]
    if skipped_done:
        print(f"(resume) {len(skipped_done)} already-recomputed cells skipped: "
              + ", ".join(f"{c['run'].split('_2026')[0]}/{c['ds']}" for c in skipped_done))
    print(f"{'APPLY' if args.apply else 'DRY-RUN'} — no-flip canonical cells: {len(targets)}")
    for c in targets:
        print(f"  {c['run'].split('_2026')[0]}/{c['ds']:<16} best={c['new_best']} (stable)")
    if not args.apply:
        print("\n(dry-run; use --apply)"); return

    os.makedirs(BACKUP_ROOT, exist_ok=True)
    reports = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in targets:
            run, ds = c['run'], c['ds']
            cell_dir = f'results/experiments/{run}/{ds}'
            cell_rel = f'{run}/{ds}'
            _cell_dirs[cell_rel] = cell_dir
            warmup = 250
            is_excl = ds.endswith('excl22')
            is_swat_full = ('SWaT' in ds) and ds.endswith('full')
            r22 = region22_of(cell_dir, ds)
            cfg = load_json(os.path.join(cell_dir, 'experiment_metadata.json')).get('config', {})
            npz_dir = os.path.join(cell_dir, 'epoch_scores')
            npz_is_link = os.path.islink(npz_dir)
            ep2npz = {int(os.path.basename(p).split('_')[1]): p
                      for p in glob.glob(os.path.join(npz_dir, 'epoch_*_scores.npz'))}
            post = sorted(e for e in ep2npz if e > warmup)
            rep = {'cell': cell_rel, 'best': c['new_best'], 'n_post': len(post),
                   'rows_updated': 0, 'npz_updated': 0, 'errors': [], 'faith_delta': None}

            # backup epoch_metrics + metadata + (post-warmup npz)
            em_path = os.path.join(cell_dir, 'epoch_metrics.json')
            backup_file(em_path, cell_rel)
            mp = os.path.join(cell_dir, 'experiment_metadata.json')
            if os.path.exists(mp): backup_file(mp, cell_rel)

            em = load_json(em_path)
            rows = {int(r['epoch']): r for r in em['epochs']}
            tasks = [(ep2npz[ep], cfg, r22, is_excl, (k == 0)) for k, ep in enumerate(post)]
            results = list(ex.map(_recompute_one, tasks))

            # faithfulness gate
            for ep, newm, newscore, old_pak in results:
                if old_pak is not None:
                    stored = rows.get(ep, {}).get('pak_auc_f1')
                    if stored is not None:
                        rep['faith_delta'] = abs(old_pak - float(stored))
                    break
            if rep['faith_delta'] is not None and rep['faith_delta'] > 2e-3:
                rep['errors'].append(f"FAITHFULNESS FAIL d={rep['faith_delta']:.2e} — SKIP cell (no write)")
                reports.append(rep)
                print(f"  !! {cell_rel} faith d={rep['faith_delta']:.2e} > 2e-3 — SKIPPED")
                continue

            # apply: update rows + npz
            for ep, newm, newscore, _ in results:
                row = rows.get(ep)
                if row is None:
                    rep['errors'].append(f"no row ep{ep}"); continue
                if is_excl:
                    row.update(newm)  # excl22 rows store masked metrics under unprefixed keys
                else:
                    upd = dict(newm)
                    # swat_full also carries excl22_* keys — recompute via exclusion
                    row.update(upd)
                rep['rows_updated'] += 1
                # overwrite npz adaptive_score (skip symlinked excl22 dir to avoid double-write)
                if not npz_is_link:
                    p = ep2npz[ep]
                    backup_file(p, cell_rel)
                    dd = np.load(p)
                    save = {k: dd[k] for k in dd.files}
                    save['adaptive_score'] = newscore
                    tmp = p + '.rebuild.npz'
                    np.savez_compressed(tmp, **save)
                    os.replace(tmp, p)
                    rep['npz_updated'] += 1

            # swat_full: recompute excl22_* keys on post-warmup rows
            if is_swat_full and r22 is not None:
                etasks = [(ep2npz[ep], cfg, r22, True, False) for ep in post]
                eres = list(ex.map(_recompute_one, etasks))
                for ep, em_excl, _, _ in eres:
                    row = rows.get(ep)
                    if row is None: continue
                    for k, v in em_excl.items():
                        ek = 'excl22_' + k
                        if ek in row:
                            row[ek] = v

            # write epoch_metrics
            tmp = em_path + '.tmp'
            with open(tmp, 'w') as f: json.dump(em, f, indent=2)
            os.replace(tmp, em_path)

            # recompute experiment_metadata best metrics (best epoch unchanged)
            if os.path.exists(mp):
                meta = load_json(mp)
                nb = c['new_best']
                if nb in ep2npz:
                    _, bm, _, _ = _recompute_one((ep2npz[nb], cfg, r22, is_excl, False))
                    blk = meta.get('metrics') or {}
                    for k, v in bm.items(): blk[k] = v
                    meta['metrics'] = blk
                    if is_swat_full and r22 is not None:
                        _, bex, _, _ = _recompute_one((ep2npz[nb], cfg, r22, True, False))
                        b2 = meta.get('metrics_excl_region22') or {}
                        for k, v in bex.items(): b2[k] = v
                        meta['metrics_excl_region22'] = b2
                    meta['_lambda_recompute'] = {'applied': True, 'date': '2026-06-01',
                                                 'ratio': '4:1', 'fm_in_score': False,
                                                 'best_epoch': nb, 'flipped': False}
                    tmp = mp + '.tmp'
                    with open(tmp, 'w') as f: json.dump(meta, f, indent=2)
                    os.replace(tmp, mp)

            reports.append(rep)
            fe = (' ERR:' + '|'.join(rep['errors'])) if rep['errors'] else ''
            print(f"  OK {cell_rel:<46} rows={rep['rows_updated']} npz={rep['npz_updated']} "
                  f"faith={rep['faith_delta']:.1e}{fe}" if rep['faith_delta'] is not None
                  else f"  OK {cell_rel} rows={rep['rows_updated']}{fe}")

    json.dump({'n': len(reports), 'reports': reports}, open(REPORT, 'w'), indent=2)
    nerr = sum(1 for r in reports if r['errors'])
    print(f"\nDONE {len(reports)} cells in {time.time()-t0:.0f}s, errors={nerr}")
    print(f"report {REPORT}  backups {BACKUP_ROOT}")
    if nerr: sys.exit(2)


if __name__ == '__main__':
    main()
