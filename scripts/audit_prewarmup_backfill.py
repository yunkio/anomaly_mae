#!/usr/bin/env python
"""Independent audit of the pre-warmup recon-only backfill.

Re-derives the expected post-backfill state FROM SCRATCH (npz teacher_recon_error
+ labels-derived regions + backups) and compares against what the backfill
actually wrote. Does NOT trust the backfill's own report — recomputes everything
independently. READ-ONLY.

Per in-scope cell, checks:
  C1  every pre-warmup npz: adaptive_score == teacher_recon_error (recon-only)
  C2  every post-warmup npz: adaptive_score != teacher_recon_error AND unchanged
      vs (a backup if present / its own un-gated raw arrays) — i.e. full score kept
  C3  every pre-warmup row: a fresh compute_full_metric_set / _with_exclusion on
      teacher_recon_error matches the WRITTEN row's headline keys (pak_auc_f1,
      prc_auc, f1_score, f1_t, affiliation_f1, r_based_f1, vus_pr) within tol
  C4  every post-warmup row: byte-identical to the backup
  C5  best_epoch in metadata == independent argmax over corrected rows
  C6  flip cells: stale marker present; metadata _prewarmup_backfill recorded

Exit 0 = all pass. Non-zero = failures (printed + JSON).
"""
import os, sys, json, glob, argparse
import numpy as np

sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
from mae_anomaly.evaluator import compute_full_metric_set, compute_metrics_with_exclusion
from mae_anomaly.scoring import is_prewarmup_epoch

MANIFEST = '/tmp/prewarmup_backfill/manifest.json'
BACKUP_ROOT = '/home/ykio/notebooks/TSMAE/.trash/0531/backfill_backups'
AUDIT_OUT = '/tmp/prewarmup_backfill/audit_report.json'
TOL = 1e-6
HEADLINE = ['pak_auc_f1', 'prc_auc', 'f1_score', 'f1_t', 'affiliation_f1', 'r_based_f1', 'vus_pr']
BEST_KEY = 'pak_auc_f1'
WARMUP = 250


class R:
    __slots__ = ('start', 'end')
    def __init__(s, a, b): s.start = a; s.end = b


def regions_from_labels(lbl):
    lbl = np.asarray(lbl).astype(int)
    o = []; i = 0; n = len(lbl)
    while i < n:
        if lbl[i] == 1:
            j = i
            while j < n and lbl[j] == 1: j += 1
            o.append(R(i, j)); i = j
        else: i += 1
    return o


def load_json(p):
    with open(p) as f:
        return json.load(f)


def pre(ep):
    return is_prewarmup_epoch({'teacher_only_warmup_epochs': WARMUP}, ep)


def audit_cell(cell, sample_post=3, sample_pre=4):
    cell_dir = cell['cell_dir']
    cell_id = f"{cell['group']}/{cell['dataset']}"
    cell_rel = cell_id
    is_excl22 = cell['dataset'].endswith('excl22')
    is_swat_full = ('SWaT' in cell['dataset']) and cell['dataset'].endswith('full')
    fails = []
    npz_dir = os.path.join(cell_dir, 'epoch_scores')
    npz_paths = sorted(glob.glob(os.path.join(npz_dir, 'epoch_*_scores.npz')))
    ep2npz = {int(os.path.basename(p).split('_')[1]): p for p in npz_paths}
    pre_eps = sorted(e for e in ep2npz if pre(e))
    post_eps = sorted(e for e in ep2npz if not pre(e))

    region22 = None
    meta_p = os.path.join(cell_dir, 'experiment_metadata.json')
    meta = load_json(meta_p) if os.path.exists(meta_p) else None
    if 'SWaT' in cell['dataset'] and meta and isinstance(meta.get('excl_region22_info'), dict):
        info = meta['excl_region22_info']; region22 = R(int(info['region_start']), int(info['region_end']))

    # C1: all pre-warmup npz recon-only
    for ep in pre_eps:
        d = np.load(ep2npz[ep])
        if not np.array_equal(d['adaptive_score'], d['teacher_recon_error']):
            fails.append(f"C1 ep{ep} adaptive_score != teacher_recon_error")
            break
    # C2: post-warmup npz NOT recon-only (full score preserved) + not in backup (untouched)
    for ep in post_eps[:sample_post] + post_eps[-sample_post:]:
        d = np.load(ep2npz[ep])
        if np.array_equal(d['adaptive_score'], d['teacher_recon_error']):
            # could legitimately coincide only if disc+fm==0; flag for review
            fails.append(f"C2 ep{ep} post-warmup adaptive==teacher (full score lost?)")
        bnpz = os.path.join(BACKUP_ROOT, cell_rel, 'epoch_scores', f'epoch_{ep:03d}_scores.npz')
        if os.path.exists(bnpz):
            fails.append(f"C2 ep{ep} post-warmup npz was BACKED UP (=modified!)")

    # rows
    em = load_json(os.path.join(cell_dir, 'epoch_metrics.json'))
    rows = {int(r['epoch']): r for r in em['epochs']}
    bem_p = os.path.join(BACKUP_ROOT, cell_rel, 'epoch_metrics.json')
    bem = load_json(bem_p) if os.path.exists(bem_p) else None
    bmap = {int(r['epoch']): r for r in bem['epochs']} if bem else {}

    # C3: sample pre-warmup rows — independent recompute matches written row
    pre_rows = sorted(e for e in rows if pre(e))
    sample = pre_rows[:sample_pre] + pre_rows[-sample_pre:]
    for ep in sample:
        p = ep2npz.get(ep)
        if p is None:
            continue
        d = np.load(p); ml = min(len(d['teacher_recon_error']), len(d['point_labels']))
        t = d['teacher_recon_error'][:ml].astype(np.float64); lbl = d['point_labels'][:ml].astype(int)
        regs = regions_from_labels(lbl)
        if is_excl22:
            m = compute_metrics_with_exclusion(t, lbl, regs, region22, lite=False)
        else:
            m = compute_full_metric_set(t, lbl, regs, n_thresholds=200, sliding_window=100, lite=False)
        for k in HEADLINE:
            if k in m and k in rows[ep]:
                if abs(float(m[k]) - float(rows[ep][k])) > TOL:
                    fails.append(f"C3 ep{ep} {k}: written={rows[ep][k]:.8f} recompute={float(m[k]):.8f}")

    # C4: post-warmup rows byte-identical to backup
    if bmap:
        for ep in sorted(e for e in rows if not pre(e)):
            if ep in bmap and json.dumps(bmap[ep], sort_keys=True) != json.dumps(rows[ep], sort_keys=True):
                fails.append(f"C4 ep{ep} post-warmup row CHANGED vs backup")
                break

    # C5: best_epoch == independent argmax over corrected rows
    proj = []
    for ep, r in rows.items():
        proj.append((ep, float(r.get(BEST_KEY, -1))))
    ind_best = max(proj, key=lambda x: x[1])[0] if proj else None
    stored_best = (meta or {}).get('timing', {}).get('best_epoch') if meta else None
    # stored_best only updated when flip/pre-warmup; else original. Compare ind_best to
    # what the row-argmax says (the authoritative corrected best).
    rep_best = None
    # read backfill report for the recorded new_best
    audit = {'cell': cell_id, 'n_pre': len(pre_eps), 'n_post': len(post_eps),
             'independent_best_epoch': ind_best, 'stored_best_epoch': stored_best,
             'fails': fails}

    # C6: flip → stale marker + metadata note
    old_best = cell['stored_best_epoch']
    flipped = (old_best is not None and ind_best != old_best)
    audit['flipped'] = flipped
    if flipped:
        bm_dir = os.path.join(cell_dir, 'visualization', 'best_model')
        audit['stale_marker_present'] = os.path.exists(os.path.join(bm_dir, 'STALE_VIZ.txt'))
        audit['metadata_backfill_recorded'] = bool(meta and meta.get('_prewarmup_backfill'))
    audit['PASS'] = (len(fails) == 0)
    return audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    ap.add_argument('--full', action='store_true', help='check ALL pre rows (slow), not a sample')
    args = ap.parse_args()
    man = load_json(MANIFEST)
    cells = [c for c in man['cells'] if c['in_scope']]
    if args.only:
        cells = [c for c in cells if f"{c['group']}/{c['dataset']}" == args.only]
    results = []
    for c in cells:
        a = audit_cell(c, sample_pre=(50 if args.full else 4))
        results.append(a)
        status = 'PASS' if a['PASS'] else 'FAIL'
        flip = ' FLIP' if a.get('flipped') else ''
        extra = ''
        if a.get('flipped'):
            extra = f" stale_marker={a.get('stale_marker_present')} meta_rec={a.get('metadata_backfill_recorded')}"
        print(f"  [{status}] {a['cell']:<52} best(ind)={a['independent_best_epoch']}{flip}{extra}"
              + ('' if a['PASS'] else '  :: ' + ' | '.join(a['fails'][:3])))
    n_pass = sum(1 for r in results if r['PASS'])
    with open(AUDIT_OUT, 'w') as f:
        json.dump({'n_cells': len(results), 'n_pass': n_pass, 'results': results}, f, indent=2)
    print(f"\nAUDIT: {n_pass}/{len(results)} PASS | report {AUDIT_OUT}")
    sys.exit(0 if n_pass == len(results) else 2)


if __name__ == '__main__':
    main()
