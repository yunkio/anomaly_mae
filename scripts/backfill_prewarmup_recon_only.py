#!/usr/bin/env python
"""Backfill: retro-correct completed experiments for the 2026-06-01 pre-warmup
recon-only anomaly-score fix.

For every IN-SCOPE cell (adaptive + teacher_only_warmup_epochs>0, from the
survey manifest), the saved anomaly score for pre-warmup epochs (ep <= warmup)
wrongly included the frozen/random-init student's disc/FM terms. This rewrites
those pre-warmup artifacts to the recon-only score and re-records derived
metrics + best-epoch selection.

GROUND TRUTH that makes this exact + retrain-free:
  recon-only score == teacher reconstruction error; the npz already stores the
  per-epoch teacher_recon_error array un-gated. Validated identities:
    - recompute on the *buggy* adaptive_score reproduces stored rows exactly (d=0)
      => offline pipeline (compute_full_metric_set + labels-derived regions) is faithful.
    - recon-only pak_auc_f1 == stored teacher_pak_auc_f1 (FULL and EXCL22), d=0
      => best-epoch projection needs NO recompute (read teacher_pak_auc_f1).

Operations (pre-warmup epochs only; post-warmup rows/npz NEVER touched):
  R1  npz['adaptive_score'] := npz['teacher_recon_error']  (skip symlinked excl22 dir)
  R2/R3  epoch_metrics pre-warmup rows := full recon-only metric set (parallel)
  R4  re-select best_epoch over ALL rows; recompute experiment_metadata metrics
      only when corrected best is pre-warmup or flips.

Not offline-recomputable (documented, untouched): disturbing_* (npz lacks
sample_types); best-model per-sample PNGs + best_epoch_train_scores.npz for
cells whose corrected best_epoch != saved best_checkpoint.pt epoch (no per-epoch
weights). Their metrics SCALARS are still recomputed.

Recon-only source is ALWAYS the invariant teacher_recon_error array → order-
independent + idempotent.

Usage:
  python scripts/backfill_prewarmup_recon_only.py            # dry-run (default)
  python scripts/backfill_prewarmup_recon_only.py --apply    # writes + backups
  python scripts/backfill_prewarmup_recon_only.py --apply --only GROUP/DATASET
  python scripts/backfill_prewarmup_recon_only.py --apply --workers 8
"""
import os, sys, json, glob, shutil, argparse, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
from mae_anomaly.scoring import is_prewarmup_epoch

MANIFEST = '/tmp/prewarmup_backfill/manifest.json'
BACKUP_ROOT = '/home/ykio/notebooks/TSMAE/.trash/0531/backfill_backups'
REPORT = '/tmp/prewarmup_backfill/backfill_report.json'
N_THRESHOLDS = 200
SLIDING_WINDOW = 100
BEST_KEY = 'pak_auc_f1'
TEACHER_BEST_KEY = 'teacher_pak_auc_f1'  # == recon-only pak_auc_f1 (validated identity)


def _prewarmup(warmup, ep):
    return is_prewarmup_epoch({'teacher_only_warmup_epochs': warmup}, ep)


def regions_from_labels(lbl):
    """Maximal contiguous runs of label==1 → events. Validated to reproduce
    stored affiliation/r_based/vus exactly."""
    lbl = np.asarray(lbl).astype(int)
    regs = []
    i, n = 0, len(lbl)
    while i < n:
        if lbl[i] == 1:
            j = i
            while j < n and lbl[j] == 1:
                j += 1
            regs.append((i, j)); i = j
        else:
            i += 1
    return regs


# ---- worker (module-level, picklable) ------------------------------------
def _recompute_one(task):
    """task = (npz_path, mode, region22 or None). Returns (ep, metric_dict)
    computed on the recon-only score (teacher_recon_error)."""
    import numpy as _np
    from mae_anomaly.evaluator import compute_full_metric_set, compute_metrics_with_exclusion

    class _R:
        __slots__ = ('start', 'end')
        def __init__(s, a, b): s.start = a; s.end = b

    npz_path, mode, region22 = task
    ep = int(os.path.basename(npz_path).split('_')[1])
    d = _np.load(npz_path)
    score = d['teacher_recon_error'].astype(_np.float64)
    lbl = d['point_labels'].astype(int)
    ml = min(len(score), len(lbl))
    score = score[:ml]; lbl = lbl[:ml]
    # regions from labels
    regs = []
    i, n = 0, len(lbl)
    while i < n:
        if lbl[i] == 1:
            j = i
            while j < n and lbl[j] == 1:
                j += 1
            regs.append(_R(i, j)); i = j
        else:
            i += 1
    out = {}
    full = compute_full_metric_set(score, lbl, regs, n_thresholds=N_THRESHOLDS,
                                   sliding_window=SLIDING_WINDOW, lite=False)
    out['full'] = {k: (float(v) if isinstance(v, (int, float, _np.floating)) else v)
                   for k, v in full.items()}
    if region22 is not None:
        r22 = _R(region22[0], region22[1])
        ex = compute_metrics_with_exclusion(score, lbl, regs, r22, lite=False)
        out['excl'] = {k: (float(v) if isinstance(v, (int, float, _np.floating)) else v)
                       for k, v in ex.items()}
    return ep, mode, out


def load_json(p):
    with open(p) as f:
        return json.load(f)


def backup_file(src, cell_dir, cell_rel):
    rel = os.path.relpath(src, start=cell_dir)
    dst = os.path.join(BACKUP_ROOT, cell_rel, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
    return dst


def project_best(cell, warmup):
    """Fast: project new best_epoch using teacher_pak_auc_f1 for pre-warmup rows
    (== recon-only pak_auc_f1) and pak_auc_f1 for post-warmup rows. No compute."""
    em = load_json(os.path.join(cell['cell_dir'], 'epoch_metrics.json'))
    proj = []
    for r in em['epochs']:
        ep = int(r['epoch'])
        if _prewarmup(warmup, ep):
            v = r.get(TEACHER_BEST_KEY)
        else:
            v = r.get(BEST_KEY)
        proj.append((ep, float(v) if v is not None else -1.0))
    if not proj:
        return None
    new_ep, new_val = max(proj, key=lambda x: x[1])
    return new_ep, new_val


def region22_of(cell):
    meta_p = os.path.join(cell['cell_dir'], 'experiment_metadata.json')
    if 'SWaT' not in cell['dataset'] or not os.path.exists(meta_p):
        return None
    meta = load_json(meta_p)
    info = meta.get('excl_region22_info')
    if isinstance(info, dict):
        return (int(info['region_start']), int(info['region_end']))
    return None


def plan_cell(cell, warmup):
    cell_dir = cell['cell_dir']
    dataset = cell['dataset']
    is_excl22 = dataset.endswith('excl22')
    is_swat_full = ('SWaT' in dataset) and dataset.endswith('full')
    npz_dir = os.path.join(cell_dir, 'epoch_scores')
    npz_is_symlink = os.path.islink(npz_dir)
    npz_paths = sorted(glob.glob(os.path.join(npz_dir, 'epoch_*_scores.npz')))
    pre_eps = [int(os.path.basename(p).split('_')[1]) for p in npz_paths]
    pre_eps = [e for e in pre_eps if _prewarmup(warmup, e)]
    old_best = cell['stored_best_epoch']
    new_best, new_val = project_best(cell, warmup)
    rep = {
        'cell': f"{cell['group']}/{cell['dataset']}", 'cell_dir': cell_dir,
        'is_excl22': is_excl22, 'is_swat_full': is_swat_full,
        'npz_is_symlink': npz_is_symlink,
        'n_pre_warmup_npz': len(pre_eps),
        'n_pre_rows_to_recompute': sum(1 for r in load_json(os.path.join(cell_dir, 'epoch_metrics.json'))['epochs']
                                       if _prewarmup(warmup, int(r['epoch']))),
        'old_best_epoch': old_best, 'new_best_epoch': new_best,
        'new_best_val': round(new_val, 8),
        'best_flipped': (old_best is not None and new_best != old_best),
        'old_best_was_prewarmup': cell['best_epoch_is_prewarmup'],
        'new_best_is_prewarmup': bool(_prewarmup(warmup, new_best)),
        'stale_best_model_viz': (old_best is not None and new_best != old_best),
        'region22': region22_of(cell),
    }
    return rep


def apply_cell(cell, warmup, plan, executor):
    cell_dir = cell['cell_dir']
    cell_rel = f"{cell['group']}/{cell['dataset']}"
    dataset = cell['dataset']
    is_excl22 = plan['is_excl22']; is_swat_full = plan['is_swat_full']
    npz_is_symlink = plan['npz_is_symlink']
    region22 = plan['region22']
    npz_dir = os.path.join(cell_dir, 'epoch_scores')
    npz_paths = sorted(glob.glob(os.path.join(npz_dir, 'epoch_*_scores.npz')))
    ep2npz = {int(os.path.basename(p).split('_')[1]): p for p in npz_paths}
    pre_eps = sorted(e for e in ep2npz if _prewarmup(warmup, e))
    rep = dict(plan)
    rep.update({'r1_overwritten': 0, 'rows_updated': 0, 'metadata_recomputed': False,
                'asserts': [], 'errors': []})

    # ---- R1: overwrite pre-warmup npz adaptive_score (skip symlink dir) ----
    if not npz_is_symlink:
        for ep in pre_eps:
            p = ep2npz[ep]
            d = np.load(p)
            if np.array_equal(d['adaptive_score'], d['teacher_recon_error']):
                continue
            backup_file(p, cell_dir, cell_rel)
            save = {k: d[k] for k in d.files}
            save['adaptive_score'] = d['teacher_recon_error'].astype(np.float32)
            # NB: np.savez_compressed appends '.npz' unless the path already ends
            # in '.npz' — so the temp path MUST end in '.npz' (else os.replace
            # below would target a non-existent file).
            tmp = p + '.rebuild.npz'
            np.savez_compressed(tmp, **save)
            os.replace(tmp, p)
            d2 = np.load(p)
            assert np.array_equal(d2['adaptive_score'], d2['teacher_recon_error']), f"R1 verify {p}"
            assert np.array_equal(d2['teacher_recon_error'], d['teacher_recon_error']), f"R1 raw mutate {p}"
            rep['r1_overwritten'] += 1

    # ---- R2/R3: parallel recompute pre-warmup rows ----
    em_path = os.path.join(cell_dir, 'epoch_metrics.json')
    backup_file(em_path, cell_dir, cell_rel)
    em = load_json(em_path)
    rows = {int(r['epoch']): r for r in em['epochs']}
    tasks = [(ep2npz[ep], 'cell', region22 if (is_excl22 or is_swat_full) else None)
             for ep in pre_eps]
    results = list(executor.map(_recompute_one, tasks))

    # one-time identity validation on the first pre-warmup epoch
    val_done = False
    for ep, _mode, out in results:
        row = rows.get(ep)
        if row is None:
            rep['errors'].append(f"no row for ep{ep}"); continue
        if is_excl22:
            new_metrics = out['excl']  # rows store excl22-masked under unprefixed keys
            if not val_done and row.get(TEACHER_BEST_KEY) is not None:
                ok = abs(new_metrics['pak_auc_f1'] - float(row[TEACHER_BEST_KEY])) < 1e-6
                rep['asserts'].append(f"ep{ep} excl recon pak_auc_f1 vs teacher {'OK' if ok else 'MISMATCH'}")
                if not ok: rep['errors'].append("excl identity broke")
                val_done = True
            row.update(new_metrics)
        else:
            new_metrics = out['full']
            if not val_done and row.get(TEACHER_BEST_KEY) is not None:
                ok = abs(new_metrics['pak_auc_f1'] - float(row[TEACHER_BEST_KEY])) < 1e-6
                rep['asserts'].append(f"ep{ep} full recon pak_auc_f1 vs teacher {'OK' if ok else 'MISMATCH'}")
                if not ok: rep['errors'].append("full identity broke")
                val_done = True
            upd = dict(new_metrics)
            if is_swat_full and 'excl' in out:
                for k, v in out['excl'].items():
                    ek = 'excl22_' + k
                    if ek in row:
                        upd[ek] = v
            row.update(upd)
        rep['rows_updated'] += 1

    # ---- post-warmup invariance check vs backup ----
    bem = load_json(os.path.join(BACKUP_ROOT, cell_rel, 'epoch_metrics.json'))
    bmap = {int(r['epoch']): r for r in bem['epochs']}
    for ep, row in rows.items():
        if not _prewarmup(warmup, ep):
            if json.dumps(bmap.get(ep), sort_keys=True) != json.dumps(row, sort_keys=True):
                rep['errors'].append(f"POST-WARMUP ROW ep{ep} CHANGED")

    # ---- write epoch_metrics ----
    tmp = em_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(em, f, indent=2)
    os.replace(tmp, em_path)

    # ---- R4: metadata metrics if best pre-warmup or flipped ----
    need_meta = rep['best_flipped'] or rep['new_best_is_prewarmup']
    meta_p = os.path.join(cell_dir, 'experiment_metadata.json')
    if need_meta and os.path.exists(meta_p):
        backup_file(meta_p, cell_dir, cell_rel)
        meta = load_json(meta_p)
        nb = rep['new_best_epoch']
        p = ep2npz.get(nb)
        if p is not None:
            d = np.load(p); ml = min(len(d['teacher_recon_error']), len(d['point_labels']))
            lbl = d['point_labels'][:ml]
            score = (d['teacher_recon_error'] if rep['new_best_is_prewarmup'] else d['adaptive_score'])[:ml]
            regs = [type('R', (), {'start': a, 'end': b})() for a, b in regions_from_labels(lbl)]
            from mae_anomaly.evaluator import compute_full_metric_set, compute_metrics_with_exclusion
            if is_excl22:
                nm = compute_metrics_with_exclusion(score.astype(np.float64), lbl, regs,
                                                    type('R', (), {'start': region22[0], 'end': region22[1]})(),
                                                    lite=False)
            else:
                nm = compute_full_metric_set(score.astype(np.float64), lbl, regs,
                                             n_thresholds=N_THRESHOLDS, sliding_window=SLIDING_WINDOW, lite=False)
            mblock = meta.get('metrics') or {}
            for k, v in nm.items():
                mblock[k] = float(v) if isinstance(v, (int, float, np.floating)) else v
            meta['metrics'] = mblock
            if is_swat_full and region22 is not None:
                ex = compute_metrics_with_exclusion(score.astype(np.float64), lbl, regs,
                                                    type('R', (), {'start': region22[0], 'end': region22[1]})(),
                                                    lite=False)
                mb2 = meta.get('metrics_excl_region22') or {}
                for k, v in ex.items():
                    mb2[k] = float(v) if isinstance(v, (int, float, np.floating)) else v
                meta['metrics_excl_region22'] = mb2
            if isinstance(meta.get('timing'), dict):
                meta['timing']['best_epoch'] = int(nb)
                meta['timing']['best_epoch_score'] = float(rep['new_best_val'])
            meta['_prewarmup_backfill'] = {
                'applied': True, 'date': '2026-06-01',
                'old_best_epoch': rep['old_best_epoch'], 'new_best_epoch': int(nb),
                'new_best_is_prewarmup': rep['new_best_is_prewarmup'],
                'stale_best_model_viz': rep['stale_best_model_viz'],
                'note': ('best-model per-sample PNGs + best_epoch_train_scores.npz reflect the OLD '
                         'best_checkpoint.pt epoch and are NOT regenerable offline (no per-epoch '
                         'weights); metrics SCALARS recomputed. disturbing_* not recomputed '
                         '(npz lacks sample_types).')
                        if rep['stale_best_model_viz'] else
                        'metrics recomputed (best epoch pre-warmup); best_checkpoint epoch unchanged.'
            }
            tmp = meta_p + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(meta, f, indent=2)
            os.replace(tmp, meta_p)
            rep['metadata_recomputed'] = True
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--only', default=None)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    apply = args.apply

    man = load_json(MANIFEST)
    cells = [c for c in man['cells'] if c['in_scope']]
    if args.only:
        cells = [c for c in cells if f"{c['group']}/{c['dataset']}" == args.only]
    print(f"{'APPLY' if apply else 'DRY-RUN'} — {len(cells)} in-scope cells\n")

    # PLAN (fast, both modes)
    plans = []
    for c in cells:
        warmup = int(c['teacher_only_warmup_epochs'])
        plans.append((c, warmup, plan_cell(c, warmup)))

    print(f"{'CELL':<54} {'pre':>4} {'best':>16} flip stale")
    for c, warmup, pl in plans:
        flip = 'YES' if pl['best_flipped'] else ' - '
        stale = 'STALE' if pl['stale_best_model_viz'] else '  -  '
        print(f"  {pl['cell']:<52} {pl['n_pre_warmup_npz']:>4} "
              f"{str(pl['old_best_epoch'])+'->'+str(pl['new_best_epoch']):>16} {flip:>4} {stale}")
    n_flip = sum(1 for _, _, p in plans if p['best_flipped'])
    n_stale = sum(1 for _, _, p in plans if p['stale_best_model_viz'])
    print(f"\nPLAN: cells={len(plans)} flips={n_flip} stale-viz={n_stale}")

    if not apply:
        with open(REPORT.replace('.json', '_plan.json'), 'w') as f:
            json.dump({'apply': False, 'plans': [p for _, _, p in plans]}, f, indent=2)
        print(f"Dry-run plan: {REPORT.replace('.json', '_plan.json')}\n(use --apply to execute)")
        return

    os.makedirs(BACKUP_ROOT, exist_ok=True)
    reports = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c, warmup, pl in plans:
            try:
                rep = apply_cell(c, warmup, pl, ex)
            except Exception as e:
                import traceback
                rep = dict(pl); rep['errors'] = [f"EXC:{e}"]; rep['tb'] = traceback.format_exc()
            reports.append(rep)
            errs = (' ERR:' + '|'.join(rep['errors'])) if rep.get('errors') else ''
            print(f"  {rep['cell']:<52} npz={rep.get('r1_overwritten',0):>2} "
                  f"rows={rep.get('rows_updated',0):>2} meta={int(rep.get('metadata_recomputed',False))} "
                  f"best {rep['old_best_epoch']}->{rep['new_best_epoch']}"
                  f"{' FLIP' if rep['best_flipped'] else ''}"
                  f"{' STALE' if rep['stale_best_model_viz'] else ''}{errs}")
            for a in rep.get('asserts', []):
                print(f"        {a}")

    with open(REPORT, 'w') as f:
        json.dump({'apply': True, 'n_cells': len(reports), 'reports': reports}, f, indent=2)
    n_err = sum(1 for r in reports if r.get('errors'))
    print(f"\nAPPLIED {len(reports)} cells in {time.time()-t0:.0f}s | "
          f"flips={sum(1 for r in reports if r['best_flipped'])} "
          f"stale-viz={sum(1 for r in reports if r['stale_best_model_viz'])} errors={n_err}")
    print(f"Report: {REPORT}  Backups: {BACKUP_ROOT}")
    if n_err:
        print("!! ERRORS — inspect report")
        sys.exit(2)


if __name__ == '__main__':
    main()
