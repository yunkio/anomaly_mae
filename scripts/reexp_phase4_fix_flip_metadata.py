#!/usr/bin/env python
"""Phase-4 fix — recompute FLIP cells' metadata.metrics from npz@best_epoch.

ROOT CAUSE: run_base finalize computes metadata via a re-forward of best_checkpoint.pt,
which is saved by ONLINE best-metric tracking. The (III) float32 fix corrected the
post-hoc selection (epoch_metrics + timing.best_epoch) but NOT the online checkpoint
tracking, so for FLIP cells (float32-best != float64-best) the finalize evaluates at
the WRONG (old float64-best) epoch. e.g. SMAP/P-4: timing.best_epoch=255 (npz@255 pak
0.4858 == epoch_metrics@255) but metadata pak=0.4337 == epoch 470.

FIX (same path as recompute_evalrevert used for non-flip): recompute metadata.metrics
from npz@best_epoch via compute_full_metric_set / compute_metrics_with_exclusion
(lite=False so VUS is included). No retrain (npz@best already saved).

Usage: --spot-check | --apply
"""
import json, os, sys, argparse
import numpy as np
_argv_backup = list(sys.argv)
sys.argv = ['fix']  # neutralize argparse in imported modules
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
from mae_anomaly.evaluator import compute_full_metric_set, compute_metrics_with_exclusion
sys.argv = _argv_backup

N_THRESHOLDS = 200
SLIDING_WINDOW = 100


class _R:
    def __init__(s, a, b):
        s.start = a; s.end = b


def regions_from_labels(lbl):
    lbl = np.asarray(lbl).astype(int)
    regs, i, n = [], 0, len(lbl)
    while i < n:
        if lbl[i] == 1:
            j = i
            while j < n and lbl[j] == 1:
                j += 1
            regs.append(_R(i, j)); i = j
        else:
            i += 1
    return regs


def recompute_metrics(cell_dir, best_epoch, is_excl):
    if is_excl:
        full_dir = cell_dir[:-len('A1A2_excl22')] + 'A1A2_full'
        npz = os.path.join(full_dir, 'epoch_scores', f'epoch_{best_epoch:03d}_scores.npz')
        meta = json.load(open(os.path.join(cell_dir, 'experiment_metadata.json')))
        info = meta.get('excl_region22_info')
        r22 = (int(info['region_start']), int(info['region_end'])) if isinstance(info, dict) else None
        if r22 is None:
            return 'NO_R22'
    else:
        npz = os.path.join(cell_dir, 'epoch_scores', f'epoch_{best_epoch:03d}_scores.npz')
        r22 = None
    if not os.path.exists(npz):
        return 'NO_NPZ'
    d = np.load(npz)
    score = d['adaptive_score'].astype(np.float64)
    lbl = d['point_labels'].astype(int)
    regs = regions_from_labels(lbl)
    if r22:
        m = compute_metrics_with_exclusion(score, lbl, regs, _R(r22[0], r22[1]), lite=False)
    else:
        m = compute_full_metric_set(score, lbl, regs, n_thresholds=N_THRESHOLDS,
                                    sliding_window=SLIDING_WINDOW, lite=False)
    return {k: float(v) for k, v in m.items()
            if not k.startswith('_') and isinstance(v, (int, float, np.floating, np.integer))}


def fix_cell(cell_dir, apply=False):
    mp = os.path.join(cell_dir, 'experiment_metadata.json')
    md = json.load(open(mp))
    be = md['timing']['best_epoch']
    is_excl = cell_dir.endswith('A1A2_excl22')
    new = recompute_metrics(cell_dir, be, is_excl)
    if isinstance(new, str):
        return (new, be, None, None)
    old_pak = md['metrics'].get('pak_auc_f1')
    new_pak = new.get('pak_auc_f1')
    if apply:
        bk = md.get('_phase4_flip_metadata_fix', {})
        bk.update({'applied': True, 'date': '2026-06-08', 'reason': 'finalize wrong-epoch (best_checkpoint online-tracking gap); recomputed from npz@best',
                   'old_pak_auc_f1': old_pak, 'old_vus_pr': md['metrics'].get('vus_pr')})
        md['_phase4_flip_metadata_fix'] = bk
        md['metrics'].update(new)
        json.dump(md, open(mp, 'w'), indent=2)
    return ('OK', be, old_pak, new_pak)


def iter_flip():
    m = json.load(open('temp/reexp_manifest.json'))
    for e in m['exps']:
        for c in e['reexp_cells']:
            yield e['exp'], os.path.join(e['dir'], c), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spot-check', action='store_true')
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    cells = list(iter_flip())
    if a.spot_check:
        sel = [c for c in cells if c[1].endswith('SMAP/P-4')][:1] + \
              [c for c in cells if c[1].endswith('A1A2_excl22')][:1] + \
              [c for c in cells if c[1].endswith('SMD/machine-3-1')][:1]
        for exp, cd, cn in sel:
            st, be, op, npk = fix_cell(cd, apply=False)
            dp = abs((op or 0) - (npk or 0)) if npk is not None else 0
            print(f'  exp{exp} {cn} best={be} [{st}]: pak {op} -> {npk} (Δ={dp:.4f})')
        return
    if a.apply:
        ok = bad = 0
        for exp, cd, cn in cells:
            st, be, op, npk = fix_cell(cd, apply=True)
            if st == 'OK':
                ok += 1
            else:
                bad += 1; print(f'  {st} exp{exp} {cn} (best={be})')
        print(f'\n=== flip metadata fix: {ok} ok, {bad} fail / {len(cells)} ===')
        return
    ap.print_help()


if __name__ == '__main__':
    main()
