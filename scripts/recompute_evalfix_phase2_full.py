#!/usr/bin/env python
"""Phase 2 — recompute ALL non-VUS metrics for ALL 100 epochs of every no-flip
cell of {271,274,285,286} with the FIXED evaluator (2026-06-03). Every cell is
no-flip (Phase 1: 0 flips), so this covers every non-concat cell.

- Score UNCHANGED (read stored npz['adaptive_score']); only metric formula changed.
- compute_full_metric_set(lite=True)  -> NO VUS recompute (vus_* keys preserved as
  stored; full VUS is fix-invariant per user instruction).
- excl22: compute_metrics_with_exclusion(lite=True) on sibling A1A2_full npz +
  region22.  swat_full also carries excl22_* keys (recomputed, non-VUS).
- EXCLUDED: concat cells; 286/WaDi/A2 (absent anyway).
- BACKUP every epoch_metrics.json + experiment_metadata.json before any write.
- FAITHFULNESS GATE per cell: recomputed roc_auc must match stored (d<2e-3, the
  rank metric is fix-invariant). LOUD + skip-write on failure (no silent corruption).
- experiment_metadata['metrics'] (+ ['metrics_excl_region22'] for swat_full)
  recomputed at the UNCHANGED best epoch.

Usage: python scripts/recompute_evalfix_phase2_full.py [--apply] [--workers 14]
"""
import os, sys, json, glob, shutil, argparse, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
EXP_ROOT = '/home/ykio/notebooks/TSMAE/results/experiments'
EXPS = ['271', '274', '285', '286']
EXCLUDE = {('286', 'WaDi/A2')}
N_THRESHOLDS = 200
SLIDING_WINDOW = 100
FAITH_TOL = 2e-3
BACKUP_ROOT = '/home/ykio/notebooks/TSMAE/.trash/0603/eval_fix_pre'
REPORT = '/tmp/evalfix_phase2_report.json'
CLASS = '/tmp/evalfix_phase1_classification.json'  # for the known best epoch


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


def find_run_dir(num):
    ds = sorted(glob.glob(os.path.join(EXP_ROOT, f'{num}_*')))
    return ds[-1] if ds else None


def list_cells(run_dir):
    out = []
    for emp in glob.glob(os.path.join(run_dir, '**', 'epoch_metrics.json'), recursive=True):
        out.append(os.path.relpath(os.path.dirname(emp), run_dir))
    return sorted(out)


def region22_of(cell_dir):
    mp = os.path.join(cell_dir, 'experiment_metadata.json')
    if not os.path.exists(mp): return None
    info = load_json(mp).get('excl_region22_info')
    return (int(info['region_start']), int(info['region_end'])) if isinstance(info, dict) else None


def backup(src, run, ds):
    if not os.path.exists(src): return
    dst = os.path.join(BACKUP_ROOT, run, ds, os.path.basename(src))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


# ---- worker: returns full (and optionally excl) metric dicts for one epoch ----
def _metrics_one(task):
    import numpy as _np
    from mae_anomaly.evaluator import compute_full_metric_set, compute_metrics_with_exclusion
    class _R:
        __slots__ = ('start', 'end')
        def __init__(s, a, b): s.start = a; s.end = b
    npz_path, mode, r22 = task  # mode: 'full' | 'excl'
    ep = int(os.path.basename(npz_path).split('_')[1])
    d = _np.load(npz_path)
    score = d['adaptive_score'].astype(_np.float64)
    lbl = d['point_labels'].astype(int)
    ml = min(len(score), len(lbl)); score, lbl = score[:ml], lbl[:ml]
    regs = [_R(a, b) for a, b in regions_from_labels(lbl)]
    if mode == 'excl':
        m = compute_metrics_with_exclusion(score, lbl, regs, _R(r22[0], r22[1]), lite=True)
    else:
        m = compute_full_metric_set(score, lbl, regs, n_thresholds=N_THRESHOLDS,
                                    sliding_window=SLIDING_WINDOW, lite=True)
    return ep, {k: (float(v) if isinstance(v, (int, float, _np.floating)) else v)
                for k, v in m.items() if not k.startswith('_')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--workers', type=int, default=14)
    args = ap.parse_args()

    cells = []
    for num in EXPS:
        rd = find_run_dir(num)
        if rd is None:
            print(f"  !! exp{num} 부재"); continue
        run = os.path.basename(rd)
        for ds in list_cells(rd):
            if 'concat' in ds or (num, ds) in EXCLUDE:
                continue
            cells.append({'run': run, 'num': num, 'ds': ds, 'run_dir': rd,
                          'cell_dir': os.path.join(rd, ds), 'is_excl': ds.endswith('excl22'),
                          'is_swat_full': ('SWaT' in ds and ds.endswith('full'))})
    print(f"{'APPLY' if args.apply else 'DRY-RUN'} — 대상 셀(concat/286-WaDiA2 제외): {len(cells)}")
    if not args.apply:
        for c in cells: print(f"  {c['num']}/{c['ds']}")
        print("\n(dry-run; --apply 로 실행)"); return

    os.makedirs(BACKUP_ROOT, exist_ok=True)
    reports = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in cells:
            run, ds, rd, cell_dir = c['run'], c['ds'], c['run_dir'], c['cell_dir']
            emp = os.path.join(cell_dir, 'epoch_metrics.json')
            mp = os.path.join(cell_dir, 'experiment_metadata.json')
            # npz source (excl22 -> sibling full)
            npz_dir = os.path.join(cell_dir, 'epoch_scores')
            if c['is_excl']:
                sib = os.path.join(rd, ds.replace('excl22', 'full'), 'epoch_scores')
                if glob.glob(os.path.join(sib, 'epoch_*_scores.npz')): npz_dir = sib
            r22 = region22_of(cell_dir) or (region22_of(os.path.join(rd, ds.replace('excl22', 'full'))) if c['is_excl'] else None)
            ep2npz = {int(os.path.basename(p).split('_')[1]): p
                      for p in glob.glob(os.path.join(npz_dir, 'epoch_*_scores.npz'))}
            rep = {'cell': f'{c["num"]}/{ds}', 'n_ep': len(ep2npz), 'rows': 0,
                   'faith_max': None, 'errors': []}
            if not ep2npz:
                rep['errors'].append('NO_NPZ'); reports.append(rep)
                print(f"  !! {rep['cell']} NO_NPZ"); continue
            em = load_json(emp)
            rows = {int(r['epoch']): r for r in em['epochs']}

            mode = 'excl' if c['is_excl'] else 'full'
            tasks = [(ep2npz[ep], mode, r22) for ep in sorted(ep2npz)]
            res = {ep: m for ep, m in ex.map(_metrics_one, tasks)}
            # swat_full also needs excl22_* (separate exclusion pass)
            eres = {}
            if c['is_swat_full'] and r22 is not None:
                etasks = [(ep2npz[ep], 'excl', r22) for ep in sorted(ep2npz)]
                eres = {ep: m for ep, m in ex.map(_metrics_one, etasks)}

            # faith gate (roc_auc, fix-invariant)
            fds = []
            for ep, m in res.items():
                sr = rows.get(ep, {}).get('roc_auc')
                if sr is not None and 'roc_auc' in m and not np.isnan(m['roc_auc']):
                    fds.append(abs(m['roc_auc'] - float(sr)))
            rep['faith_max'] = max(fds) if fds else None
            if rep['faith_max'] is not None and rep['faith_max'] > FAITH_TOL:
                rep['errors'].append(f'FAITH_FAIL roc d={rep["faith_max"]:.2e} — SKIP write')
                reports.append(rep); print(f"  !! {rep['cell']} {rep['errors'][-1]}"); continue

            # backup then apply
            backup(emp, run, ds); backup(mp, run, ds)
            for ep, m in res.items():
                row = rows.get(ep)
                if row is None: continue
                row.update(m)  # non-VUS keys; vus_* not in m -> preserved
                if ep in eres:  # swat_full excl22_* keys
                    for k, v in eres[ep].items():
                        ek = 'excl22_' + k
                        if ek in row and not k.startswith('vus'):
                            row[ek] = v
                rep['rows'] += 1
            tmp = emp + '.tmp'
            with open(tmp, 'w') as f: json.dump(em, f, indent=2)
            os.replace(tmp, emp)

            # experiment_metadata best block (best epoch unchanged)
            if os.path.exists(mp):
                meta = load_json(mp)
                # best epoch = argmax stored pak (== new best, no-flip); read from rows
                sp = {ep: rows[ep].get('pak_auc_f1') for ep in rows if rows[ep].get('pak_auc_f1') is not None}
                nb = max(sp, key=lambda e: sp[e]) if sp else None
                if nb in res:
                    blk = meta.get('metrics') or {}
                    blk.update({k: v for k, v in res[nb].items()})
                    meta['metrics'] = blk
                    if nb in eres:
                        b2 = meta.get('metrics_excl_region22') or {}
                        b2.update({k: v for k, v in eres[nb].items() if not k.startswith('vus')})
                        meta['metrics_excl_region22'] = b2
                    meta['_evalfix_recompute'] = {'applied': True, 'date': '2026-06-03',
                                                  'vus_recomputed': False, 'best_epoch': nb, 'flipped': False}
                    tmp = mp + '.tmp'
                    with open(tmp, 'w') as f: json.dump(meta, f, indent=2)
                    os.replace(tmp, mp)
            reports.append(rep)
            print(f"  OK {rep['cell']:<24} rows={rep['rows']} faith={rep['faith_max'] if rep['faith_max'] is not None else -1:.1e}")

    json.dump({'n': len(reports), 'reports': reports}, open(REPORT, 'w'), indent=2)
    nerr = sum(1 for r in reports if r['errors'])
    print(f"\nDONE {len(reports)} cells in {time.time()-t0:.0f}s — errors={nerr}")
    print(f"report {REPORT}  backups {BACKUP_ROOT}")
    if nerr:
        for r in reports:
            if r['errors']: print(f"  ERR {r['cell']}: {r['errors']}")
        sys.exit(2)


if __name__ == '__main__':
    main()
