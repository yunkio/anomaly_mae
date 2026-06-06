#!/usr/bin/env python
"""Phase 1 — recompute pak_auc_f1 for ALL epochs of ALL non-concat cells of
{271,274,285,286} with the FIXED evaluator (2026-06-03: >= convention, excl22
mask, K=0 PA guard), then census best-epoch FLIPS (old stored best vs new best).

Fast path: pak_auc_f1 only via compute_pa_k_auc (9x faster than full lite, gives
BYTE-IDENTICAL pak_auc_f1, verified d=0.0). Score is UNCHANGED (read stored
npz['adaptive_score']); only the metric formula changed.

FAITHFULNESS GATE: roc_auc (auc of the same roc_curve used to pick the threshold)
is rank-based and fix-invariant -> must match stored roc_auc (d<2e-3). Proves the
offline pipeline reproduces production. LOUD on any failure (no silent skip).

excl22 cell: own epoch_metrics (selection metric = its unprefixed pak_auc_f1 =
region-22-excluded) but NO own epoch_scores -> read sibling A1A2_full npz, apply
eval_mask over region22 (verified == compute_metrics_with_exclusion, d=0.0).

concat cells (MSL/concat, SMAP/concat, SMD/concat): EXCLUDED (deleted / invalid
whole-array normalization, separate rerun).

Outputs:
  /tmp/evalfix_phase1_classification.json   (machine-readable per-cell)
  ./temp/evalfix_bestepoch_FLIPS_<stamp>.md  (FLIP cases only, human doc)
Usage: python scripts/recompute_evalfix_phase1_pak.py [--workers 14]
"""
import os, sys, json, glob, argparse, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
EXP_ROOT = '/home/ykio/notebooks/TSMAE/results/experiments'
EXPS = ['271', '274', '285', '286']
N_THRESHOLDS = 200
FAITH_TOL = 2e-3
OUT_JSON = '/tmp/evalfix_phase1_classification.json'
TEMP_DIR = '/home/ykio/notebooks/TSMAE/temp'


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
    cells = []
    for emp in glob.glob(os.path.join(run_dir, '**', 'epoch_metrics.json'), recursive=True):
        ds = os.path.relpath(os.path.dirname(emp), run_dir)
        cells.append(ds)
    return sorted(cells)


def region22_of(cell_dir):
    mp = os.path.join(cell_dir, 'experiment_metadata.json')
    if not os.path.exists(mp): return None
    info = load_json(mp).get('excl_region22_info')
    if isinstance(info, dict):
        return (int(info['region_start']), int(info['region_end']))
    return None


# ---- worker ----
def _compute_one(task):
    import numpy as _np
    from mae_anomaly import evaluator as E
    from mae_anomaly.evaluator import compute_pa_k_auc
    from sklearn.metrics import roc_curve, auc
    class _R:
        __slots__ = ('start', 'end')
        def __init__(s, a, b): s.start = a; s.end = b
    npz_path, is_excl, region22 = task
    ep = int(os.path.basename(npz_path).split('_')[1])
    d = _np.load(npz_path)
    score = d['adaptive_score'].astype(_np.float64)
    lbl = d['point_labels'].astype(int)
    ml = min(len(score), len(lbl)); score, lbl = score[:ml], lbl[:ml]
    regs = [_R(a, b) for a, b in regions_from_labels(lbl)]
    if is_excl and region22 is not None:
        mask = _np.ones(len(score), bool); mask[region22[0]:region22[1]] = False
        sl, ss = lbl[mask], score[mask]
    else:
        mask, sl, ss = None, lbl, score
    if len(_np.unique(sl)) < 2:
        return ep, float('nan'), float('nan')
    fpr, tpr, thr = roc_curve(sl, ss)
    idx = E.find_f1_optimal_idx(fpr, tpr, sl)
    th = float(thr[idx]); roc = float(auc(fpr, tpr))
    pak = compute_pa_k_auc(score, lbl, regs, th, eval_mask=mask, n_thresholds=N_THRESHOLDS)['pak_auc_f1']
    return ep, float(pak), roc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=14)
    args = ap.parse_args()

    real, missing = [], []
    for num in EXPS:
        rd = find_run_dir(num)
        if rd is None:
            missing.append(num); print(f"  !! exp{num}: run dir 없음"); continue
        run = os.path.basename(rd)
        for ds in list_cells(rd):
            if 'concat' in ds:
                continue  # excluded (deleted / invalid whole-array norm)
            real.append({'run': run, 'num': num, 'ds': ds, 'run_dir': rd})
    print(f"대상 셀(concat 제외): {len(real)}  | run-missing: {missing or '없음'}")

    # flatten ALL (cell, epoch) tasks for max parallel utilization
    cell_tasks = []
    for c in real:
        rd, ds = c['run_dir'], c['ds']
        cell_dir = os.path.join(rd, ds)
        is_excl = ds.endswith('excl22')
        npz_dir = os.path.join(cell_dir, 'epoch_scores')
        if is_excl:
            sib = os.path.join(rd, ds.replace('excl22', 'full'), 'epoch_scores')
            if glob.glob(os.path.join(sib, 'epoch_*_scores.npz')):
                npz_dir = sib
        r22 = region22_of(cell_dir) or (region22_of(os.path.join(rd, ds.replace('excl22', 'full'))) if is_excl else None)
        npzs = sorted(glob.glob(os.path.join(npz_dir, 'epoch_*_scores.npz')),
                      key=lambda p: int(os.path.basename(p).split('_')[1]))
        c['is_excl'] = is_excl; c['npzs'] = npzs; c['r22'] = r22
        for p in npzs:
            cell_tasks.append((len(cell_tasks), c, p, is_excl, r22))
    print(f"총 (셀,epoch) task: {len(cell_tasks)}")

    t0 = time.time()
    by_cell = {}  # id(c) -> {ep: (pak, roc)}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = ex.map(_compute_one, [(p, ie, r) for (_, c, p, ie, r) in cell_tasks], chunksize=4)
        for (idx, c, p, ie, r), (ep, pak, roc) in zip(cell_tasks, futs):
            by_cell.setdefault(id(c), {})[ep] = (pak, roc)
    print(f"재계산 {len(cell_tasks)} tasks in {time.time()-t0:.0f}s")

    results = []
    for c in real:
        rd, ds, run, num = c['run_dir'], c['ds'], c['run'], c['num']
        emp = os.path.join(rd, ds, 'epoch_metrics.json')
        em = load_json(emp)
        stored = {int(r['epoch']): r for r in em['epochs']}
        newv = by_cell.get(id(c), {})
        rec = {'run': run, 'num': num, 'ds': ds, 'is_excl': c['is_excl'],
               'n_npz': len(c['npzs']), 'errors': []}
        # faith
        fd = [abs(newv[ep][1] - float(stored[ep]['roc_auc']))
              for ep in newv if ep in stored and stored[ep].get('roc_auc') is not None
              and not np.isnan(newv[ep][1])]
        rec['faith_max_roc_delta'] = max(fd) if fd else None
        if rec['faith_max_roc_delta'] is not None and rec['faith_max_roc_delta'] > FAITH_TOL:
            rec['errors'].append(f"FAITH_FAIL roc d={rec['faith_max_roc_delta']:.2e}")
        # flip
        sp = {ep: stored[ep].get('pak_auc_f1') for ep in stored if stored[ep].get('pak_auc_f1') is not None}
        npak = {ep: newv[ep][0] for ep in newv if not np.isnan(newv[ep][0])}
        common = sorted(set(sp) & set(npak))
        if not common:
            rec['errors'].append('NO_COMMON_EPOCHS'); results.append(rec); continue
        old_best = max(common, key=lambda e: sp[e])
        new_best = max(common, key=lambda e: npak[e])
        rec.update({
            'old_best': old_best, 'new_best': new_best, 'flip': old_best != new_best,
            'old_best_pak_stored': round(sp[old_best], 6),
            'new_best_pak_new': round(npak[new_best], 6),
            'new_pak_at_old_best': round(npak[old_best], 6),
            'pak_gain_from_flip': round(npak[new_best] - npak[old_best], 6),
            'max_abs_pak_delta': round(max(abs(npak[e] - sp[e]) for e in common), 6),
            'n_common': len(common),
        })
        results.append(rec)

    flips = [r for r in results if r.get('flip')]
    stables = [r for r in results if 'flip' in r and not r['flip']]
    errs = [r for r in results if r['errors']]
    for r in sorted(results, key=lambda x: (x['num'], x['ds'])):
        if 'flip' not in r:
            print(f"  {r['num']}/{r['ds']:<20} ERR {r['errors']}"); continue
        tag = '🔴FLIP' if r['flip'] else 'stable'
        print(f"  {r['num']}/{r['ds']:<20} old={r['old_best']:>3} new={r['new_best']:>3} "
              f"[{tag}] dpak_max={r['max_abs_pak_delta']:+.4f} faith={r['faith_max_roc_delta'] or -1:.1e}"
              + (' '+'|'.join(r['errors']) if r['errors'] else ''))

    json.dump({'exps': EXPS, 'n_cells': len(results), 'n_flip': len(flips),
               'n_stable': len(stables), 'n_err': len(errs), 'results': results},
              open(OUT_JSON, 'w'), indent=2)

    # FLIP doc -> ./temp/
    os.makedirs(TEMP_DIR, exist_ok=True)
    stamp = em.get('_stamp', '20260603')
    doc = os.path.join(TEMP_DIR, 'evalfix_bestepoch_FLIPS_20260603.md')
    with open(doc, 'w') as f:
        f.write("# eval-fix (2026-06-03) best-epoch FLIP census — pak_auc_f1\n\n")
        f.write("Evaluator fix: `>=` convention + excl22 mask + **K=0 PA guard**. "
                "Score unchanged; only `pak_auc_f1` formula changed. Best epoch = argmax pak_auc_f1 "
                "(excl22 cell uses its region-22-excluded pak_auc_f1).\n\n")
        f.write(f"- Scope: exp {', '.join(EXPS)}, non-concat cells = **{len(results)}**\n")
        f.write(f"- **FLIP (best epoch changed): {len(flips)}**  |  stable: {len(stables)}  |  errors: {len(errs)}\n")
        f.write(f"- Faithfulness gate (roc_auc invariant): max delta over all cells = "
                f"{max((r.get('faith_max_roc_delta') or 0) for r in results):.2e} (tol {FAITH_TOL})\n\n")
        if flips:
            f.write("## 🔴 FLIP cells (best epoch CHANGED)\n\n")
            f.write("| exp | dataset/cell | old_best | new_best | old_best pak(stored) | "
                    "new_best pak(new) | new pak@old_best | gain | max|Δpak| |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for r in sorted(flips, key=lambda x: (x['num'], x['ds'])):
                f.write(f"| {r['num']} | {r['ds']} | {r['old_best']} | {r['new_best']} | "
                        f"{r['old_best_pak_stored']:.6f} | {r['new_best_pak_new']:.6f} | "
                        f"{r['new_pak_at_old_best']:.6f} | {r['pak_gain_from_flip']:+.6f} | "
                        f"{r['max_abs_pak_delta']:.6f} |\n")
        else:
            f.write("## ✅ No best-epoch flips — every cell's best epoch is unchanged.\n")
        if errs:
            f.write("\n## ⚠️ errors / faith-fail\n\n")
            for r in errs: f.write(f"- {r['num']}/{r['ds']}: {r['errors']}\n")

    print(f"\nDONE {len(results)} cells in {time.time()-t0:.0f}s — "
          f"🔴FLIP={len(flips)} stable={len(stables)} ERR={len(errs)}")
    print(f"json -> {OUT_JSON}\ndoc  -> {doc}")
    if errs:
        print("⚠️ 오류 셀:")
        for r in errs: print(f"  {r['num']}/{r['ds']}: {r['errors']}")
        sys.exit(2)


if __name__ == '__main__':
    main()
