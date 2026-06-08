#!/usr/bin/env python
"""Phase 3 (L2) — recompute metadata vus_pr/vus_roc (float32) for non-flip cells.

WHY: non-flip cells were recomputed offline on 2026-06-06 with lite=True
(VUS skipped → metadata kept the ORIGINAL float64 evaluate() VUS). Every other
metric + epoch_metrics is already float32-recompute. This restores VUS to the
SAME float32 value a re-run (flip cell) produces, by calling run_base's own
``_compute_vus_for_npz_file`` on the best-epoch npz (verified bit-identical to a
flip cell's metadata VUS).

Non-flip set has NO SWaT (all SWaT cells flipped) → no excl22 dual handling.
Usage:
  python scripts/reexp_phase3_vus.py --spot-check    # 1 cell, no write
  python scripts/reexp_phase3_vus.py --apply         # all 160, write
"""
import json, os, sys, argparse
import numpy as np
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
sys.path.insert(0, '/home/ykio/notebooks/TSMAE/scripts')
from run_base_experiments import _compute_vus_for_npz_file


def regions_from_npz(npz):
    d = np.load(npz)
    labels = d['point_labels'].astype(np.int8)
    starts, ends, i, n = [], [], 0, len(labels)
    while i < n:
        if labels[i] == 1:
            j = i
            while j < n and labels[j] == 1:
                j += 1
            starts.append(i); ends.append(j); i = j
        else:
            i += 1
    return starts, ends


def recompute_cell(cdir, apply=False):
    mdp = os.path.join(cdir, 'experiment_metadata.json')
    md = json.load(open(mdp))
    be = md['timing']['best_epoch']
    npz = os.path.join(cdir, 'epoch_scores', f'epoch_{be:03d}_scores.npz')
    if not os.path.exists(npz):
        return ('NO_NPZ', be, None, None, None, None)
    rs, re_ = regions_from_npz(npz)
    res = _compute_vus_for_npz_file(npz, rs, re_)
    old_pr = md['metrics'].get('vus_pr')
    old_roc = md['metrics'].get('vus_roc')
    new_pr, new_roc = float(res['vus_pr']), float(res['vus_roc'])
    if apply:
        md['metrics']['vus_pr'] = new_pr
        md['metrics']['vus_roc'] = new_roc
        p3 = md.setdefault('_phase3_vus_recompute', {})
        p3['applied'] = True
        p3['date'] = '2026-06-08'
        p3['old_vus_pr'] = old_pr
        p3['old_vus_roc'] = old_roc
        if '_evalrevert_recompute' in md:
            md['_evalrevert_recompute']['vus_recomputed'] = True
        json.dump(md, open(mdp, 'w'), indent=2)
    return ('OK', be, old_pr, new_pr, old_roc, new_roc)


def iter_cells():
    m = json.load(open('temp/reexp_manifest.json'))
    for e in m['exps']:
        for c in e['reviz_cells']:
            yield e['exp'], os.path.join(e['dir'], c), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spot-check', action='store_true')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    cells = list(iter_cells())
    if args.spot_check:
        # one simple + one base
        sel = [c for c in cells if 'MSL/F-7' in c[1]][:1] + \
              [c for c in cells if c[1].endswith('/PSM')][:1]
        for exp, cdir, cname in sel:
            st, be, opr, npr, oroc, nroc = recompute_cell(cdir, apply=False)
            print(f'  exp{exp} {cname} best={be} [{st}]')
            print(f'    vus_pr : {opr} -> {npr}  (Δ={abs((opr or 0)-(npr or 0)):.2e})')
            print(f'    vus_roc: {oroc} -> {nroc}  (Δ={abs((oroc or 0)-(nroc or 0)):.2e})')
        return

    if args.apply:
        ok = nochg = nonpz = 0
        for exp, cdir, cname in cells:
            st, be, opr, npr, oroc, nroc = recompute_cell(cdir, apply=True)
            if st == 'NO_NPZ':
                nonpz += 1; print(f'  !! NO_NPZ exp{exp} {cname} (best={be})'); continue
            changed = (opr != npr) or (oroc != nroc)
            ok += 1; nochg += (0 if changed else 1)
        print(f'\n=== Phase3 VUS recompute: {ok} updated, {nochg} already-equal, {nonpz} no-npz / {len(cells)} cells ===')
        return

    ap.print_help()


if __name__ == '__main__':
    main()
