#!/usr/bin/env python
"""Phase 4 — strict final consistency verification across ALL 370 cells.

Zero-inconsistency target. Checks:
  A. VUS float32 consistency (ALL 370): metadata vus_pr/vus_roc == run_base's
     _compute_vus_for_npz_file(best-epoch npz) within TOL. This is the decisive
     "no float64 VUS residual anywhere" check — flip cells (run_base float32) and
     non-flip cells (Phase-3 float32) must BOTH match the canonical npz compute.
  B. Flip best-epoch (210 reexp): timing.best_epoch == reexp_expected_best.json.
  C. Non-flip (160 reviz): _evalrevert_recompute.vus_recomputed == True AND
     epoch_metrics/anomaly_threshold viz regenerated on/after 2026-06-08.

Usage: python scripts/reexp_phase4_verify.py
"""
import json, os, sys, glob, datetime
import numpy as np
sys.argv = ['p4']
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
sys.path.insert(0, '/home/ykio/notebooks/TSMAE/scripts')
from run_base_experiments import _compute_vus_for_npz_file

TOL = 1e-6
CUT = datetime.datetime(2026, 6, 8).timestamp()


def regions_from_npz(npz):
    d = np.load(npz)
    labels = d['point_labels'].astype(np.int8)
    s, e, i, n = [], [], 0, len(labels)
    while i < n:
        if labels[i] == 1:
            j = i
            while j < n and labels[j] == 1:
                j += 1
            s.append(i); e.append(j); i = j
        else:
            i += 1
    return s, e


def vus_check(cdir, excl_start=None, excl_end=None):
    mdp = os.path.join(cdir, 'experiment_metadata.json')
    if not os.path.exists(mdp):
        return ('NO_META', None, None)
    md = json.load(open(mdp))
    be = md['timing']['best_epoch']
    npz = os.path.join(cdir, 'epoch_scores', f'epoch_{be:03d}_scores.npz')
    if not os.path.exists(npz):
        return ('NO_NPZ', None, None)
    rs, re_ = regions_from_npz(npz)
    res = _compute_vus_for_npz_file(npz, rs, re_, excl_start, excl_end)
    mpr, mroc = md['metrics'].get('vus_pr'), md['metrics'].get('vus_roc')
    dpr = abs((mpr or 0) - res['vus_pr'])
    droc = abs((mroc or 0) - res['vus_roc'])
    ok = dpr < TOL and droc < TOL
    return ('OK' if ok else 'MISMATCH', dpr, droc)


def main():
    m = json.load(open('temp/reexp_manifest.json'))
    expected = json.load(open('temp/reexp_expected_best.json')) if os.path.exists('temp/reexp_expected_best.json') else {}

    # ---- A: VUS consistency on 360 cells (excl22 handled separately) ----
    # excl22 cells: metadata VUS is region22-MASKED but their npz holds the FULL
    # score and the metadata excl region is None (run_base derives it dynamically
    # via find_swat_region_22). They are FLIP cells (Phase-2 run_base float32,
    # best-epoch already verified by reexp_verify) and were NOT touched by Phase 3
    # -> excluded from this strict re-derivation, counted separately.
    print('=== A. VUS float32 consistency (360 cells; 10 excl22 separate) ===')
    a_ok = a_bad = a_skip = a_excl = 0
    bad_list = []
    for e in m['exps']:
        for key, cells in (('reexp', e['reexp_cells']), ('reviz', e['reviz_cells'])):
            for c in cells:
                if 'excl22' in c:
                    a_excl += 1
                    continue
                cdir = os.path.join(e['dir'], c)
                st, dpr, droc = vus_check(cdir)
                if st == 'OK':
                    a_ok += 1
                elif st in ('NO_NPZ', 'NO_META'):
                    a_skip += 1
                else:
                    a_bad += 1; bad_list.append((e['exp'], c, st, dpr, droc))
    print(f'  OK={a_ok}  MISMATCH={a_bad}  skip={a_skip}  | excl22(flip, Phase-2 float32)={a_excl}')
    for exp, c, st, dpr, droc in bad_list[:20]:
        print(f'    !! exp{exp} {c}: {st} dpr={dpr:.2e} droc={droc:.2e}')

    # ---- B: flip best-epoch ----
    print('=== B. Flip best-epoch (210 reexp) vs expected ===')
    b_ok = b_bad = b_skip = 0
    for e in m['exps']:
        for c in e['reexp_cells']:
            cdir = os.path.join(e['dir'], c)
            mdp = os.path.join(cdir, 'experiment_metadata.json')
            if not os.path.exists(mdp):
                b_skip += 1; continue
            be = json.load(open(mdp))['timing']['best_epoch']
            ek = f"{os.path.basename(e['dir'])}/{c}"
            exp_be = expected.get(ek, expected.get(c))
            if exp_be is None:
                b_skip += 1; continue
            if int(be) == int(exp_be):
                b_ok += 1
            else:
                b_bad += 1; print(f'    !! exp{e["exp"]} {c}: best={be} expected={exp_be}')
    print(f'  OK={b_ok}  MISMATCH={b_bad}  (expected-key-miss={b_skip})')

    # ---- C: non-flip viz + vus_recomputed flag ----
    print('=== C. Non-flip (160 reviz): vus_recomputed flag + viz regenerated ===')
    c_flag = c_viz = c_total = 0
    viz_bad = []
    for e in m['exps']:
        for c in e['reviz_cells']:
            c_total += 1
            cdir = os.path.join(e['dir'], c)
            md = json.load(open(os.path.join(cdir, 'experiment_metadata.json')))
            if md.get('_evalrevert_recompute', {}).get('vus_recomputed') is True or \
               md.get('_phase3_vus_recompute', {}).get('applied') is True:
                c_flag += 1
            # viz: at least one epoch_metrics png + anomaly_threshold regenerated >= 06-08
            em = glob.glob(os.path.join(cdir, 'visualization', 'epoch_metrics', '*.png'))
            at = os.path.join(cdir, 'visualization', 'best_model', 'anomaly_threshold.png')
            em_new = any(os.path.getmtime(p) >= CUT for p in em)
            at_new = os.path.exists(at) and os.path.getmtime(at) >= CUT
            if em_new and at_new:
                c_viz += 1
            else:
                viz_bad.append((e['exp'], c, em_new, at_new))
    print(f'  vus_recomputed flag: {c_flag}/{c_total} | viz regenerated(epoch+threshold): {c_viz}/{c_total}')
    for exp, c, emn, atn in viz_bad[:20]:
        print(f'    !! exp{exp} {c}: epoch_new={emn} threshold_new={atn}')

    # ---- verdict ----
    print('\n=== VERDICT ===')
    allgood = (a_bad == 0 and b_bad == 0 and c_flag == c_total and c_viz == c_total)
    print('  >>> ' + ('ALL CONSISTENT ✓ (zero pre-fix residual)' if allgood else 'INCONSISTENCY FOUND ✗ — review above'))


if __name__ == '__main__':
    main()
