#!/usr/bin/env python
"""2026-06-06 — recompute ALL results/experiments cells under the reverted evaluator
(79f8f1b: PA%K strict `>` + K=0 guard removed, paper-faithful Kim et al. AAAI 2022).

Affected metrics (verified): pa_K_{f1,precision,recall,roc_auc,prc_auc} for K>=10,
pak_auc_{f1,prc_auc,roc_auc,precision,recall,f1_t,*_raw}, r_based_f1, and tie-level
noise on f1/f1_t/recall/affiliation. UNAFFECTED: roc_auc, prc_auc, vus_*, *_ar,
pa_0_f1, pa_5_f1.

Two modes:
  --mode census : pak_auc_f1 only (fast, compute_pa_k_auc) -> best-epoch FLIP census.
  --mode full   : compute_full_metric_set(lite) -> overwrite ALL non-VUS metrics
                  (VUS preserved: lite returns vus_*=0.0 -> we DROP vus_* from update),
                  backup, faith gate (roc_auc invariant), best epoch, flips, metadata.

Score UNCHANGED (stored npz adaptive_score). concat cells EXCLUDED.
excl22 cell: sibling A1A2_full npz + region22 mask (compute_metrics_with_exclusion).

Outputs:
  census: /tmp/evalrevert_census.json + ./temp/evalrevert_FLIPS_20260606.md
  full  : /tmp/evalrevert_full_report.json , backups ./.trash/0606/eval_revert_pre/
Usage: python scripts/recompute_evalrevert_20260606.py --mode census|full [--workers 14]
"""
import os, sys, json, glob, shutil, argparse, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
EXP_ROOT = '/home/ykio/notebooks/TSMAE/results/experiments'
N_THRESHOLDS = 200
SLIDING_WINDOW = 100
FAITH_TOL = 2e-3
BACKUP_ROOT = '/home/ykio/notebooks/TSMAE/.trash/0606/eval_revert_pre'
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


def lj(p):
    with open(p) as f: return json.load(f)


def wj(p, o):
    t = p + '.tmp'
    with open(t, 'w') as f: json.dump(o, f, indent=2)
    os.replace(t, p)


def lj_safe(p):
    try:
        return lj(p)
    except Exception:
        return None


def region22_of(cell_dir):
    mp = os.path.join(cell_dir, 'experiment_metadata.json')
    if not os.path.exists(mp): return None
    meta = lj_safe(mp)
    if meta is None: return None
    info = meta.get('excl_region22_info')
    return (int(info['region_start']), int(info['region_end'])) if isinstance(info, dict) else None


def enumerate_cells():
    cells = []
    for rd in sorted(glob.glob(os.path.join(EXP_ROOT, '*'))):
        if not os.path.isdir(rd): continue
        run = os.path.basename(rd)
        if run.startswith('legacy_'):   # legacy npz lack 'point_labels' (old format) -> un-recomputable
            continue
        for emp in glob.glob(os.path.join(rd, '**', 'epoch_metrics.json'), recursive=True):
            ds = os.path.relpath(os.path.dirname(emp), rd)
            if 'concat' in ds: continue
            cells.append({'run': run, 'ds': ds, 'rd': rd, 'cell_dir': os.path.join(rd, ds),
                          'is_excl': ds.endswith('excl22'),
                          'is_swat_full': ('SWaT' in ds and ds.endswith('full'))})
    return cells


def _npz_dir(c):
    nd = os.path.join(c['cell_dir'], 'epoch_scores')
    if c['is_excl']:
        sib = os.path.join(c['rd'], c['ds'].replace('excl22', 'full'), 'epoch_scores')
        if glob.glob(os.path.join(sib, 'epoch_*_scores.npz')): return sib
    return nd


def _r22(c):
    r = region22_of(c['cell_dir'])
    if r is None and c['is_excl']:
        r = region22_of(os.path.join(c['rd'], c['ds'].replace('excl22', 'full')))
    return r


# ---- census worker: pak_auc_f1 + roc_auc (faith) ----
def _census_one(task):
    import numpy as _np
    from mae_anomaly import evaluator as E
    from mae_anomaly.evaluator import compute_pa_k_auc
    from sklearn.metrics import roc_curve, auc
    class _R:
        __slots__ = ('start', 'end')
        def __init__(s, a, b): s.start = a; s.end = b
    npz_path, is_excl, r22 = task
    ep = int(os.path.basename(npz_path).split('_')[1])
    d = _np.load(npz_path)
    score = d['adaptive_score'].astype(_np.float64); lbl = d['point_labels'].astype(int)
    ml = min(len(score), len(lbl)); score, lbl = score[:ml], lbl[:ml]
    regs = [_R(a, b) for a, b in regions_from_labels(lbl)]
    if is_excl and r22 is not None:
        mask = _np.ones(len(score), bool); mask[r22[0]:r22[1]] = False; sl, ss = lbl[mask], score[mask]
    else:
        mask, sl, ss = None, lbl, score
    if len(_np.unique(sl)) < 2: return ep, float('nan'), float('nan')
    fpr, tpr, thr = roc_curve(sl, ss); idx = E.find_f1_optimal_idx(fpr, tpr, sl)
    th = float(thr[idx]); roc = float(auc(fpr, tpr))
    pak = compute_pa_k_auc(score, lbl, regs, th, eval_mask=mask, n_thresholds=N_THRESHOLDS)['pak_auc_f1']
    return ep, float(pak), roc


# ---- full worker: compute_full_metric_set(lite) / exclusion ----
def _full_one(task):
    import numpy as _np
    from mae_anomaly.evaluator import compute_full_metric_set, compute_metrics_with_exclusion
    class _R:
        __slots__ = ('start', 'end')
        def __init__(s, a, b): s.start = a; s.end = b
    idx, npz_path, mode, r22, compute_vus = task
    try:
        ep = int(os.path.basename(npz_path).split('_')[1])
    except Exception:
        return idx, mode, -1, {'__err__': 'bad_npz_name:' + os.path.basename(npz_path)}
    try:
        d = _np.load(npz_path)
        if 'adaptive_score' not in d.files or 'point_labels' not in d.files:
            return idx, mode, ep, {'__err__': 'missing_keys:' + str(d.files)}
        score = d['adaptive_score'].astype(_np.float64); lbl = d['point_labels'].astype(int)
        ml = min(len(score), len(lbl)); score, lbl = score[:ml], lbl[:ml]
        regs = [_R(a, b) for a, b in regions_from_labels(lbl)]
        lite = not compute_vus   # compute_vus -> lite=False so VUS is produced
        if mode == 'excl':
            m = compute_metrics_with_exclusion(score, lbl, regs, _R(r22[0], r22[1]), lite=lite)
        else:
            m = compute_full_metric_set(score, lbl, regs, n_thresholds=N_THRESHOLDS,
                                        sliding_window=SLIDING_WINDOW, lite=lite)
    except Exception as e:
        return idx, mode, ep, {'__err__': type(e).__name__ + ':' + str(e)[:60]}
    # If NOT computing VUS, DROP vus_* (lite returns 0.0 -> would zero stored VUS).
    # If computing VUS, KEEP vus_* (cell had empty/null VUS -> fill it).
    return idx, mode, ep, {k: (float(v) if isinstance(v, (int, float, _np.floating)) else v)
                           for k, v in m.items()
                           if not k.startswith('_') and (compute_vus or not k.startswith('vus'))}


def backup(src, run, ds):
    if not os.path.exists(src): return
    dst = os.path.join(BACKUP_ROOT, run, ds, os.path.basename(src))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst): shutil.copy2(src, dst)


def run_census(cells, workers):
    flat = []
    for c in cells:
        c['npzs'] = sorted(glob.glob(os.path.join(_npz_dir(c), 'epoch_*_scores.npz')),
                           key=lambda p: int(os.path.basename(p).split('_')[1]))
        c['r22v'] = _r22(c)
        for p in c['npzs']: flat.append((c, p))
    print(f"census: {len(cells)} cells, {len(flat)} tasks", flush=True)
    by = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        outs = ex.map(_census_one, [(p, c['is_excl'], c['r22v']) for (c, p) in flat], chunksize=8)
        for (c, p), (ep, pak, roc) in zip(flat, outs):
            by.setdefault(id(c), {})[ep] = (pak, roc)
    print(f"census compute {time.time()-t0:.0f}s", flush=True)
    results = []
    for c in cells:
        em = lj(os.path.join(c['cell_dir'], 'epoch_metrics.json'))
        stored = {int(r['epoch']): r for r in em['epochs']}
        nv = by.get(id(c), {})
        sp = {ep: stored[ep].get('pak_auc_f1') for ep in stored if stored[ep].get('pak_auc_f1') is not None}
        npak = {ep: nv[ep][0] for ep in nv if not np.isnan(nv[ep][0])}
        common = sorted(set(sp) & set(npak))
        fd = [abs(nv[ep][1] - float(stored[ep]['roc_auc'])) for ep in nv
              if ep in stored and stored[ep].get('roc_auc') is not None and not np.isnan(nv[ep][1])]
        rec = {'run': c['run'], 'ds': c['ds'], 'faith': (max(fd) if fd else None)}
        if not common:
            rec['err'] = 'NO_COMMON'; results.append(rec); continue
        ob = max(common, key=lambda e: sp[e]); nb = max(common, key=lambda e: npak[e])
        rec.update({'old_best': ob, 'new_best': nb, 'flip': ob != nb,
                    'old_best_pak': round(sp[ob], 6), 'new_best_pak': round(npak[nb], 6),
                    'new_pak_at_old': round(npak[ob], 6),
                    'max_dpak': round(max(abs(npak[e] - sp[e]) for e in common), 6)})
        results.append(rec)
    return results


def write_flip_doc(results):
    os.makedirs(TEMP_DIR, exist_ok=True)
    flips = [r for r in results if r.get('flip')]
    doc = os.path.join(TEMP_DIR, 'evalrevert_FLIPS_20260606.md')
    by_exp = {}
    for r in results:
        by_exp.setdefault(r['run'].split('_2026')[0].split('_2025')[0], {'n': 0, 'flip': 0})
        e = by_exp[r['run'].split('_2026')[0].split('_2025')[0]]; e['n'] += 1
        if r.get('flip'): e['flip'] += 1
    with open(doc, 'w') as f:
        f.write("# 2026-06-06 eval-revert (`>=`->`>`, K=0 guard 제거) best-epoch FLIP 전수조사\n\n")
        f.write("커밋 `79f8f1b` (PA%K strict `>`, paper-faithful Kim et al. AAAI 2022). "
                "선정키 `pak_auc_f1` (excl22 셀은 region-22 제외).\n\n")
        f.write(f"- 전체 셀(concat 제외): **{len(results)}** · **FLIP: {len(flips)}** · stable: {len(results)-len(flips)}\n\n")
        f.write("## 실험별 flip 집계\n\n| 실험 | 셀 | FLIP |\n|---|---|---|\n")
        for k in sorted(by_exp):
            f.write(f"| {k} | {by_exp[k]['n']} | {'**'+str(by_exp[k]['flip'])+'**' if by_exp[k]['flip'] else '0'} |\n")
        if flips:
            f.write("\n## 🔴 best epoch 바뀐 셀 전수\n\n")
            f.write("| 실험 | 셀 | old_best | new_best | old pak | new pak | new@old | max|Δpak| |\n|---|---|---|---|---|---|---|---|\n")
            for r in sorted(flips, key=lambda x: (x['run'], x['ds'])):
                f.write(f"| {r['run'].split('_2026')[0].split('_2025')[0]} | {r['ds']} | {r['old_best']} | "
                        f"{r['new_best']} | {r['old_best_pak']:.4f} | {r['new_best_pak']:.4f} | "
                        f"{r['new_pak_at_old']:.4f} | {r['max_dpak']:.4f} |\n")
    return doc


def run_full(cells, workers):
    os.makedirs(BACKUP_ROOT, exist_ok=True)
    # build flat task list across ALL cells x epochs (max parallel utilization)
    info = []   # idx -> {c, ep2, r22, need_vus}
    flat = []
    n_vus_cells = 0
    for idx, c in enumerate(cells):
        ep2 = {int(os.path.basename(p).split('_')[1]): p
               for p in glob.glob(os.path.join(_npz_dir(c), 'epoch_*_scores.npz'))}
        r22 = _r22(c)
        # VUS empty? (stored vus_pr null/missing for ALL epochs -> recompute VUS too)
        emj = lj_safe(os.path.join(c['cell_dir'], 'epoch_metrics.json'))
        need_vus = False
        if emj and emj.get('epochs'):
            need_vus = all(r.get('vus_pr') is None for r in emj['epochs'])
        if need_vus: n_vus_cells += 1
        info.append({'c': c, 'ep2': ep2, 'r22': r22, 'need_vus': need_vus})
        mode = 'excl' if c['is_excl'] else 'full'
        for ep, p in ep2.items():
            flat.append((idx, p, mode, r22, need_vus))
            if c['is_swat_full'] and r22 is not None:
                flat.append((idx, p, 'excl', r22, need_vus))   # excl22_* keys for swat_full
    print(f"full: {len(cells)} cells ({n_vus_cells} need VUS recompute), {len(flat)} tasks, {workers} workers", flush=True)
    collected = {}  # idx -> {'full': {ep: m}, 'excl': {ep: m}}
    t0 = time.time()
    done = 0; err_tasks = 0; err_samples = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for idx, mode, ep, m in ex.map(_full_one, flat, chunksize=4):
            done += 1
            if '__err__' in m:
                err_tasks += 1
                if len(err_samples) < 15:
                    err_samples.append(f"{info[idx]['c']['run']}/{info[idx]['c']['ds']} ep{ep}: {m['__err__']}")
            else:
                collected.setdefault(idx, {}).setdefault(mode, {})[ep] = m
            if done % 5000 == 0:
                print(f"  ... {done}/{len(flat)} tasks ({time.time()-t0:.0f}s, task-err={err_tasks})", flush=True)
    print(f"compute {time.time()-t0:.0f}s — task errors={err_tasks} — writing per-cell...", flush=True)
    if err_samples:
        print("  task-err 샘플:", flush=True)
        for s in err_samples: print("    " + s, flush=True)

    reports = []
    for idx, ci in enumerate(info):
        c = ci['c']; ep2 = ci['ep2']
        emp = os.path.join(c['cell_dir'], 'epoch_metrics.json')
        mp = os.path.join(c['cell_dir'], 'experiment_metadata.json')
        rep = {'cell': f"{c['run']}/{c['ds']}", 'n_ep': len(ep2), 'rows': 0, 'faith': None, 'flip': None, 'errors': []}
        res = collected.get(idx, {}).get('full', {}) or collected.get(idx, {}).get('excl', {})
        eres = collected.get(idx, {}).get('excl', {}) if c['is_swat_full'] else {}
        if not res:
            rep['errors'].append('NO_RESULT'); reports.append(rep); print(f"  !! {rep['cell']} NO_RESULT", flush=True); continue
        em = lj(emp); rows = {int(r['epoch']): r for r in em['epochs']}
        old_sp = {ep: rows[ep].get('pak_auc_f1') for ep in rows if rows[ep].get('pak_auc_f1') is not None}
        fds = [abs(res[ep]['roc_auc'] - float(rows[ep]['roc_auc'])) for ep in res
               if ep in rows and 'roc_auc' in res[ep] and rows[ep].get('roc_auc') is not None]
        rep['faith'] = max(fds) if fds else None
        if rep['faith'] is not None and rep['faith'] > FAITH_TOL:
            rep['errors'].append(f"FAITH_FAIL roc d={rep['faith']:.2e}")
            reports.append(rep); print(f"  !! {rep['cell']} {rep['errors'][-1]} — SKIP", flush=True); continue
        backup(emp, c['run'], c['ds']); backup(mp, c['run'], c['ds'])
        for ep, m in res.items():
            r = rows.get(ep)
            if r is None: continue
            r.update(m)
            if ep in eres:
                for k, v in eres[ep].items():
                    ek = 'excl22_' + k
                    if ek in r: r[ek] = v
            rep['rows'] += 1
        wj(emp, em)
        new_sp = {ep: rows[ep].get('pak_auc_f1') for ep in rows if rows[ep].get('pak_auc_f1') is not None}
        if old_sp and new_sp:
            ob = max(old_sp, key=lambda e: old_sp[e]); nb = max(new_sp, key=lambda e: new_sp[e])
            rep['flip'] = (ob != nb); rep['old_best'] = ob; rep['new_best'] = nb
        meta = lj_safe(mp) if os.path.exists(mp) else None
        if meta is not None and rep.get('new_best') in res:
            nb = rep['new_best']
            blk = meta.get('metrics') or {}; blk.update(res[nb]); meta['metrics'] = blk
            if nb in eres:
                b2 = meta.get('metrics_excl_region22') or {}; b2.update(eres[nb]); meta['metrics_excl_region22'] = b2
            meta['_evalrevert_recompute'] = {'applied': True, 'date': '2026-06-06',
                                             'convention': 'strict > , no K=0 guard (79f8f1b)',
                                             'vus_recomputed': False, 'best_epoch': nb, 'flipped': rep['flip']}
            wj(mp, meta)
        reports.append(rep)

    json.dump({'n': len(reports), 'reports': reports}, open('/tmp/evalrevert_full_report.json', 'w'), indent=2)
    nerr = sum(1 for r in reports if r['errors']); nflip = sum(1 for r in reports if r.get('flip'))
    flipped = [r for r in reports if r.get('flip')]
    # FLIP doc
    os.makedirs(TEMP_DIR, exist_ok=True)
    doc = os.path.join(TEMP_DIR, 'evalrevert_FLIPS_20260606.md')
    by_exp = {}
    for r in reports:
        k = r['cell'].split('/')[0].split('_2026')[0].split('_2025')[0]
        by_exp.setdefault(k, [0, 0]); by_exp[k][0] += 1
        if r.get('flip'): by_exp[k][1] += 1
    with open(doc, 'w') as f:
        f.write("# 2026-06-06 eval-revert 전체지표 재계산 + best-epoch FLIP (전체 실험)\n\n")
        f.write("`79f8f1b`: PA%K strict `>` + K=0 guard 제거 (paper-faithful Kim et al. AAAI 2022). "
                "전 지표(VUS 제외) 새 evaluator로 재계산. 선정키 `pak_auc_f1`.\n\n")
        f.write(f"- 셀(concat 제외): **{len(reports)}** · **FLIP {len(flipped)}** · errors {nerr}\n\n")
        f.write("## 실험별 flip\n\n| 실험 | 셀 | FLIP |\n|---|---|---|\n")
        for k in sorted(by_exp):
            f.write(f"| {k} | {by_exp[k][0]} | {'**'+str(by_exp[k][1])+'**' if by_exp[k][1] else '0'} |\n")
        if flipped:
            f.write("\n## 🔴 best epoch 바뀐 셀 전수\n\n| 실험 | 셀 | old_best | new_best |\n|---|---|---|---|\n")
            for r in sorted(flipped, key=lambda x: x['cell']):
                ex_, ds_ = r['cell'].split('/', 1)
                f.write(f"| {ex_.split('_2026')[0].split('_2025')[0]} | {ds_} | {r['old_best']} | {r['new_best']} |\n")
    print(f"\nFULL DONE {len(reports)} cells in {time.time()-t0:.0f}s — errors={nerr} flips={nflip}", flush=True)
    print(f"doc -> {doc}", flush=True)
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['census', 'full'], required=True)
    ap.add_argument('--workers', type=int, default=14)
    args = ap.parse_args()
    cells = enumerate_cells()
    print(f"전체 셀(concat 제외): {len(cells)}  실험: {len(set(c['run'] for c in cells))}", flush=True)
    if args.mode == 'census':
        results = run_census(cells, args.workers)
        json.dump({'n': len(results), 'results': results}, open('/tmp/evalrevert_census.json', 'w'), indent=2)
        doc = write_flip_doc(results)
        flips = [r for r in results if r.get('flip')]
        errs = [r for r in results if r.get('err')]
        print(f"\nCENSUS DONE — cells={len(results)} FLIP={len(flips)} err={len(errs)}", flush=True)
        print(f"doc -> {doc}", flush=True)
        for r in sorted(flips, key=lambda x: (x['run'], x['ds'])):
            print(f"  FLIP {r['run'].split('_2026')[0].split('_2025')[0]}/{r['ds']:<18} {r['old_best']}->{r['new_best']} dpak_max={r['max_dpak']:.4f}", flush=True)
    else:
        run_full(cells, args.workers)


if __name__ == '__main__':
    main()
