#!/usr/bin/env python
"""Comprehensive consistency audit — ALL 370 cells, NO re-run.

For every cell (flip + non-flip, simple + base + excl22), verify the recorded
metadata is internally consistent with the saved best-epoch artifact:

  C1. npz@best_epoch exists.
  C2. metadata.timing.best_epoch == argmax(epoch_metrics[best_epoch_metric])
      (float32 selection). The stored best epoch IS the selected one.
  C3. metadata.metrics[m] == recompute_from_npz@best_epoch[m] for every key
      metric m (pak_auc_f1, pak_auc_prc_auc, prc_auc, f1_score, vus_pr,
      vus_roc, affiliation_f1, r_based_f1). i.e. the "computed values" the
      paper/Notion would cite are exactly what the saved best-epoch score
      produces. excl22 cells use compute_metrics_with_exclusion on the FULL
      sibling npz + region22 mask.

Anything outside TOL is printed per-cell. Verdict = zero mismatch.

Usage: python scripts/reexp_comprehensive_audit.py
"""
import json, os, sys
import numpy as np
_argv = list(sys.argv); sys.argv = ['audit']
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
from mae_anomaly.evaluator import compute_full_metric_set, compute_metrics_with_exclusion
sys.argv = _argv

N_THR, SW, TOL = 200, 100, 1e-6
KEYS = ['pak_auc_f1', 'pak_auc_prc_auc', 'prc_auc', 'f1_score',
        'vus_pr', 'vus_roc', 'affiliation_f1', 'r_based_f1']


class _R:
    def __init__(s, a, b): s.start = a; s.end = b


def regions(lbl):
    lbl = np.asarray(lbl).astype(int); regs = []; i = 0; n = len(lbl)
    while i < n:
        if lbl[i] == 1:
            j = i
            while j < n and lbl[j] == 1: j += 1
            regs.append(_R(i, j)); i = j
        else: i += 1
    return regs


def recompute(cell_dir, best_epoch, is_excl):
    if is_excl:
        full = cell_dir[:-len('A1A2_excl22')] + 'A1A2_full'
        npz = os.path.join(full, 'epoch_scores', f'epoch_{best_epoch:03d}_scores.npz')
        md = json.load(open(os.path.join(cell_dir, 'experiment_metadata.json')))
        info = md.get('excl_region22_info')
        if not isinstance(info, dict): return None, 'NO_R22'
        r22 = _R(int(info['region_start']), int(info['region_end']))
    else:
        npz = os.path.join(cell_dir, 'epoch_scores', f'epoch_{best_epoch:03d}_scores.npz')
        r22 = None
    if not os.path.exists(npz): return None, 'NO_NPZ'
    d = np.load(npz)
    score = d['adaptive_score'].astype(np.float64)
    lbl = d['point_labels'].astype(int)
    regs = regions(lbl)
    if r22 is not None:
        m = compute_metrics_with_exclusion(score, lbl, regs, r22, lite=False)
    else:
        m = compute_full_metric_set(score, lbl, regs, n_thresholds=N_THR,
                                    sliding_window=SW, lite=False)
    return m, 'OK'


def audit_cell(cell_dir):
    """Return (status, list-of-issue-strings)."""
    mdp = os.path.join(cell_dir, 'experiment_metadata.json')
    if not os.path.exists(mdp): return 'NO_META', ['no experiment_metadata.json']
    md = json.load(open(mdp))
    issues = []
    be = md.get('timing', {}).get('best_epoch')
    metric_key = md.get('timing', {}).get('best_epoch_metric',
                                          md.get('metrics', {}).get('_best_epoch_metric', 'pak_auc_f1'))
    is_excl = cell_dir.endswith('A1A2_excl22')

    # C2: best_epoch == argmax(epoch_metrics[metric_key])  (skip excl22: its metric is excl22_*)
    emp = os.path.join(cell_dir, 'epoch_metrics.json')
    if os.path.exists(emp) and not is_excl:
        rows = json.load(open(emp))['epochs']
        mk = metric_key if any(metric_key in r for r in rows) else 'pak_auc_f1'
        best_row = max(rows, key=lambda r: r.get(mk, -1))
        if int(best_row['epoch']) != int(be):
            issues.append(f'C2 best_epoch={be} != argmax({mk})@{best_row["epoch"]}')

    # C1+C3: recompute from npz@best and compare every metric
    rec, st = recompute(cell_dir, be, is_excl)
    if st != 'OK':
        issues.append(f'C1 {st}')
        return ('ISSUE' if issues else 'OK'), issues
    for k in KEYS:
        mk = ('excl22_' + k) if is_excl else k
        mv = md.get('metrics', {}).get(mk)
        rv = rec.get(k)
        if mv is None or rv is None:
            continue
        if abs(float(mv) - float(rv)) > TOL:
            issues.append(f'C3 {mk}: meta={float(mv):.6f} recompute={float(rv):.6f} Δ={abs(float(mv)-float(rv)):.2e}')
    return ('ISSUE' if issues else 'OK'), issues


def main():
    m = json.load(open('temp/reexp_manifest.json'))
    n_ok = n_issue = n_meta = 0
    by_kind = {'flip': [0, 0], 'nonflip': [0, 0]}
    issue_cells = []
    for e in m['exps']:
        for kind, cells in (('flip', e['reexp_cells']), ('nonflip', e['reviz_cells'])):
            for c in cells:
                cd = os.path.join(e['dir'], c)
                st, iss = audit_cell(cd)
                if st == 'NO_META':
                    n_meta += 1; issue_cells.append((e['exp'], c, iss)); continue
                if st == 'OK':
                    n_ok += 1; by_kind[kind][0] += 1
                else:
                    n_issue += 1; by_kind[kind][1] += 1
                    issue_cells.append((e['exp'], c, iss))
    total = n_ok + n_issue + n_meta
    print(f'=== Comprehensive audit: {total} cells ===')
    print(f'  OK={n_ok}  ISSUE={n_issue}  NO_META={n_meta}')
    print(f'  flip: OK={by_kind["flip"][0]} ISSUE={by_kind["flip"][1]} | '
          f'nonflip: OK={by_kind["nonflip"][0]} ISSUE={by_kind["nonflip"][1]}')
    if issue_cells:
        print('\n--- inconsistencies ---')
        for exp, c, iss in issue_cells[:80]:
            print(f'  exp{exp} {c}:')
            for x in iss: print(f'      {x}')
    print('\n=== VERDICT ===')
    print('  >>> ' + ('ALL CONSISTENT ✓ (zero inconsistency across all cells)'
                      if n_issue == 0 and n_meta == 0 else
                      f'{n_issue + n_meta} cell(s) INCONSISTENT ✗'))


if __name__ == '__main__':
    main()
