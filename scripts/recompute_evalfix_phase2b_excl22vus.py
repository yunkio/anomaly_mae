#!/usr/bin/env python
"""Phase 2b — recompute SWaT excl22 VUS (the ONE VUS family the 2026-06-03 fix
actually changed: the excl22 mask-leak made VUS/Aff/AR byte-identical to full).
ALL epochs. Run AFTER Phase 2 (non-VUS) to avoid write races on the same files.

Per SWaT training (271/274/285/286), the A1A2_full and A1A2_excl22 cells SHARE
one npz (A1A2_full/epoch_scores). For each epoch:
  compute_metrics_with_exclusion(adaptive_score, labels, regions, region22,
                                 lite=False)  -> vus_pr, vus_roc on MASKED data
Apply ONLY the VUS keys (leave Phase-2's non-VUS metrics untouched):
  - A1A2_excl22 cell row:  vus_pr, vus_roc            (unprefixed = excl22 metrics)
  - A1A2_full   cell row:  excl22_vus_pr, excl22_vus_roc
  - both experiment_metadata best blocks updated at the (unchanged) best epoch.

Original (pre-Phase-2) values already backed up under .trash/0603/eval_fix_pre/.
Usage: python scripts/recompute_evalfix_phase2b_excl22vus.py [--apply] [--workers 14]
"""
import os, sys, json, glob, argparse, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
EXP_ROOT = '/home/ykio/notebooks/TSMAE/results/experiments'
EXPS = ['271', '274', '285', '286']
REPORT = '/tmp/evalfix_phase2b_report.json'


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


def _vus_one(task):
    import numpy as _np
    from mae_anomaly.evaluator import compute_metrics_with_exclusion
    class _R:
        __slots__ = ('start', 'end')
        def __init__(s, a, b): s.start = a; s.end = b
    npz_path, r22 = task
    ep = int(os.path.basename(npz_path).split('_')[1])
    d = _np.load(npz_path)
    score = d['adaptive_score'].astype(_np.float64)
    lbl = d['point_labels'].astype(int)
    ml = min(len(score), len(lbl)); score, lbl = score[:ml], lbl[:ml]
    regs = [_R(a, b) for a, b in regions_from_labels(lbl)]
    m = compute_metrics_with_exclusion(score, lbl, regs, _R(r22[0], r22[1]), lite=False)
    return ep, float(m.get('vus_pr', float('nan'))), float(m.get('vus_roc', float('nan')))


def write_json(p, obj):
    tmp = p + '.tmp'
    with open(tmp, 'w') as f: json.dump(obj, f, indent=2)
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--workers', type=int, default=14)
    args = ap.parse_args()

    jobs = []
    for num in EXPS:
        rd = find_run_dir(num)
        if rd is None:
            print(f"  !! exp{num} 부재", flush=True); continue
        full = os.path.join(rd, 'SWaT', 'A1A2_full')
        excl = os.path.join(rd, 'SWaT', 'A1A2_excl22')
        if not os.path.isdir(full):
            print(f"  !! {num} SWaT/A1A2_full 없음", flush=True); continue
        info = load_json(os.path.join(full, 'experiment_metadata.json')).get('excl_region22_info')
        r22 = (int(info['region_start']), int(info['region_end']))
        npzs = sorted(glob.glob(os.path.join(full, 'epoch_scores', 'epoch_*_scores.npz')),
                      key=lambda p: int(os.path.basename(p).split('_')[1]))
        jobs.append({'num': num, 'full': full, 'excl': excl, 'r22': r22, 'npzs': npzs})
        print(f"  {num}: SWaT {len(npzs)} epochs, region22={r22}", flush=True)

    print(f"{'APPLY' if args.apply else 'DRY-RUN'} — SWaT excl22 VUS 재계산: {len(jobs)} 실험", flush=True)
    if not args.apply:
        print("(dry-run; --apply)"); return

    reports = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for j in jobs:
            tasks = [(p, j['r22']) for p in j['npzs']]
            vus = {ep: (vp, vr) for ep, vp, vr in ex.map(_vus_one, tasks)}
            rep = {'num': j['num'], 'n_ep': len(vus), 'full_rows': 0, 'excl_rows': 0}

            # A1A2_full: excl22_vus_*
            emp = os.path.join(j['full'], 'epoch_metrics.json')
            em = load_json(emp); rows = {int(r['epoch']): r for r in em['epochs']}
            for ep, (vp, vr) in vus.items():
                r = rows.get(ep)
                if r is None: continue
                r['excl22_vus_pr'] = vp; r['excl22_vus_roc'] = vr; rep['full_rows'] += 1
            write_json(emp, em)

            # A1A2_excl22: vus_* (unprefixed)
            eemp = os.path.join(j['excl'], 'epoch_metrics.json')
            if os.path.exists(eemp):
                eem = load_json(eemp); erows = {int(r['epoch']): r for r in eem['epochs']}
                for ep, (vp, vr) in vus.items():
                    r = erows.get(ep)
                    if r is None: continue
                    r['vus_pr'] = vp; r['vus_roc'] = vr; rep['excl_rows'] += 1
                write_json(eemp, eem)

            # metadata best blocks (best epoch = argmax pak_auc_f1 in each cell)
            for cell, prefix in [(j['full'], 'excl22_'), (j['excl'], '')]:
                mp = os.path.join(cell, 'experiment_metadata.json')
                em2 = load_json(os.path.join(cell, 'epoch_metrics.json'))
                sp = {int(r['epoch']): (r.get('excl22_pak_auc_f1') if prefix else r.get('pak_auc_f1'))
                      for r in em2['epochs']}
                sp = {k: v for k, v in sp.items() if v is not None}
                nb = max(sp, key=lambda e: sp[e]) if sp else None
                if os.path.exists(mp) and nb in vus:
                    meta = load_json(mp)
                    blk_key = 'metrics_excl_region22' if prefix else 'metrics'
                    blk = meta.get(blk_key) or {}
                    blk[f'{prefix}vus_pr' if prefix else 'vus_pr'] = vus[nb][0]
                    blk[f'{prefix}vus_roc' if prefix else 'vus_roc'] = vus[nb][1]
                    meta[blk_key] = blk
                    write_json(mp, meta)

            reports.append(rep)
            ex_show = {ep: round(vp, 4) for ep, (vp, vr) in list(vus.items())[:1]}
            print(f"  OK {j['num']} SWaT  full_excl22_vus={rep['full_rows']} excl_vus={rep['excl_rows']}  "
                  f"sample vus_pr={ex_show}", flush=True)

    json.dump({'n': len(reports), 'reports': reports}, open(REPORT, 'w'), indent=2)
    print(f"\nDONE {len(reports)} 실험 in {time.time()-t0:.0f}s — report {REPORT}", flush=True)


if __name__ == '__main__':
    main()
