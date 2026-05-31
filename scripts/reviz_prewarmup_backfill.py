#!/usr/bin/env python
"""Re-visualize after the pre-warmup recon-only backfill.

Two things:
  (A) Regenerate epoch-metric trend curves (visualization/epoch_metrics/*.png)
      for every in-scope cell from the CORRECTED epoch_metrics.json. These are
      pure data->PNG (no model), and the pre-warmup adaptive-metric trend
      changed (now == teacher), so they MUST be refreshed. Backs up the old
      PNGs first.
  (B) Classify best_model/*.png staleness (model-forward viz, needs weights):
        - post-warmup best, not flipped  -> CORRECT (full score unaffected): keep
        - pre-warmup best, not flipped    -> STALE-REGENERABLE (best_checkpoint.pt
          weights exist; best-model score should be recon-only). Needs GPU viz
          pipeline; flagged for optional regen, NOT done here.
        - flipped best                    -> STALE-NOT-REGENERABLE (no per-epoch
          weights for the new best epoch). Documented caveat.
      Writes a JSON + Markdown stale report; drops a STALE_VIZ.txt marker into
      each affected cell's best_model dir.

Usage:
  python scripts/reviz_prewarmup_backfill.py            # do (A) + (B)
  python scripts/reviz_prewarmup_backfill.py --report-only   # only (B)
"""
import os, sys, json, glob, shutil, argparse, time
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
sys.argv = ['reviz']  # neutralize argparse in imported module
from scripts.run_base_experiments import plot_epoch_metrics

MANIFEST = '/tmp/prewarmup_backfill/manifest.json'
APPLY_REPORT = '/tmp/prewarmup_backfill/backfill_report.json'
PNG_BACKUP_ROOT = '/home/ykio/notebooks/TSMAE/.trash/0531/backfill_viz_backups'
STALE_REPORT_JSON = '/tmp/prewarmup_backfill/stale_viz_report.json'
STALE_REPORT_MD = '/home/ykio/notebooks/TSMAE/docs/POST_MORTEMS/2026-06-01_prewarmup_backfill_stale_viz.md'
WARMUP = 250


def load_json(p):
    with open(p) as f:
        return json.load(f)


def regen_epoch_curves(cell_dir, cell_rel):
    em_path = os.path.join(cell_dir, 'epoch_metrics.json')
    if not os.path.exists(em_path):
        return {'regenerated': False, 'reason': 'no epoch_metrics.json'}
    rows = load_json(em_path)['epochs']
    out_dir = os.path.join(cell_dir, 'visualization', 'epoch_metrics')
    # back up existing PNGs first
    if os.path.isdir(out_dir):
        for png in glob.glob(os.path.join(out_dir, '*.png')):
            dst = os.path.join(PNG_BACKUP_ROOT, cell_rel, 'epoch_metrics', os.path.basename(png))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copy2(png, dst)
    try:
        plot_epoch_metrics(rows, out_dir)
        return {'regenerated': True, 'n_rows': len(rows), 'out_dir': out_dir}
    except Exception as e:
        return {'regenerated': False, 'reason': f'{e}'}


def classify_best_model(cell, rep):
    """rep = backfill apply report entry for this cell (or None)."""
    old_best = cell.get('stored_best_epoch')
    new_best = (rep or {}).get('new_best_epoch', old_best)
    flipped = (rep or {}).get('best_flipped', False)
    new_pre = (new_best is not None and new_best <= WARMUP)
    if flipped:
        status = 'STALE_NOT_REGENERABLE'
        reason = (f'best_epoch flipped {old_best}->{new_best}; no per-epoch weights for '
                  f'the new best epoch (only best_checkpoint.pt={old_best} on disk). '
                  f'metrics SCALARS recomputed; best-model per-sample PNGs reflect the OLD epoch.')
    elif new_pre:
        status = 'STALE_REGENERABLE'
        reason = (f'best_epoch={new_best} is pre-warmup; best-model PNGs were rendered with the '
                  f'buggy full score. best_checkpoint.pt weights exist -> regenerable with '
                  f'force_recon_only=True via the GPU viz pipeline (deferred).')
    else:
        status = 'CORRECT'
        reason = f'best_epoch={new_best} is post-warmup; full score unaffected by the fix.'
    return {'status': status, 'old_best': old_best, 'new_best': new_best,
            'flipped': flipped, 'reason': reason}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report-only', action='store_true')
    args, _ = ap.parse_known_args()

    man = load_json(MANIFEST)
    cells = [c for c in man['cells'] if c['in_scope']]
    apply_rep = {}
    if os.path.exists(APPLY_REPORT):
        for r in load_json(APPLY_REPORT)['reports']:
            apply_rep[r['cell']] = r

    results = []
    t0 = time.time()
    for c in cells:
        cell_rel = f"{c['group']}/{c['dataset']}"
        rep = apply_rep.get(cell_rel)
        cls = classify_best_model(c, rep)
        curve = {'skipped': True} if args.report_only else regen_epoch_curves(c['cell_dir'], cell_rel)
        # drop a marker into stale best_model dirs
        if cls['status'].startswith('STALE'):
            bm = os.path.join(c['cell_dir'], 'visualization', 'best_model')
            if os.path.isdir(bm):
                with open(os.path.join(bm, 'STALE_VIZ.txt'), 'w') as f:
                    f.write(f"[2026-06-01 pre-warmup recon-only backfill]\n{cls['status']}\n{cls['reason']}\n")
        results.append({'cell': cell_rel, 'epoch_curves': curve, 'best_model': cls})
        tag = cls['status']
        print(f"  {cell_rel:<52} curves={'OK' if curve.get('regenerated') else curve.get('skipped','FAIL')}"
              f"  best_model={tag}")

    n_regen = sum(1 for r in results if r['epoch_curves'].get('regenerated'))
    by = {}
    for r in results:
        by[r['best_model']['status']] = by.get(r['best_model']['status'], 0) + 1
    os.makedirs(os.path.dirname(STALE_REPORT_JSON), exist_ok=True)
    with open(STALE_REPORT_JSON, 'w') as f:
        json.dump({'results': results, 'summary': by, 'n_curves_regen': n_regen}, f, indent=2)

    # Markdown stale report
    lines = ['# Pre-warmup backfill — visualization status (2026-06-01)', '',
             f'Epoch-metric curves regenerated: **{n_regen}/{len(results)}** cells '
             f'(from corrected `epoch_metrics.json`; old PNGs backed up under '
             f'`.trash/0531/backfill_viz_backups/`).', '',
             '## best_model/*.png status (model-forward viz)', '',
             '| status | count | meaning |', '|---|---|---|',
             f"| CORRECT | {by.get('CORRECT',0)} | post-warmup best; full score unaffected → kept as-is |",
             f"| STALE_REGENERABLE | {by.get('STALE_REGENERABLE',0)} | pre-warmup best, not flipped; "
             f"weights exist → regenerable with `force_recon_only=True` (deferred) |",
             f"| STALE_NOT_REGENERABLE | {by.get('STALE_NOT_REGENERABLE',0)} | best_epoch flipped; "
             f"no per-epoch weights → metrics scalars recomputed, per-sample PNGs reflect OLD epoch |",
             '', '## Per-cell detail', '',
             '| cell | best (old→new) | best_model viz |', '|---|---|---|']
    for r in results:
        bm = r['best_model']
        lines.append(f"| {r['cell']} | {bm['old_best']}→{bm['new_best']}"
                     f"{' (FLIP)' if bm['flipped'] else ''} | {bm['status']} |")
    lines += ['', '### STALE_NOT_REGENERABLE cells (flipped — require retrain to regenerate best-model PNGs)']
    for r in results:
        if r['best_model']['status'] == 'STALE_NOT_REGENERABLE':
            lines.append(f"- **{r['cell']}**: {r['best_model']['reason']}")
    lines += ['', '### STALE_REGENERABLE cells (pre-warmup best; regenerable offline w/ weights)']
    for r in results:
        if r['best_model']['status'] == 'STALE_REGENERABLE':
            lines.append(f"- {r['cell']}: best_epoch={r['best_model']['new_best']} (pre-warmup)")
    with open(STALE_REPORT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"\nepoch curves regen: {n_regen}/{len(results)} | best_model: {by} | {time.time()-t0:.0f}s")
    print(f"Report: {STALE_REPORT_JSON}\nMarkdown: {STALE_REPORT_MD}")


if __name__ == '__main__':
    main()
