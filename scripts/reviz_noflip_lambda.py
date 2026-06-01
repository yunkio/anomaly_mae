#!/usr/bin/env python
"""Re-visualize the NO-FLIP canonical cells after the 4:1 / no-FM recompute.

For each no-flip canonical cell (same selection as recompute_noflip_lambda.py):
  1. back up existing visualization/{epoch_metrics,best_model}/*.png -> ./.trash/0601/lambda
  2. regenerate epoch-metric trend PNGs from the RECOMPUTED epoch_metrics.json
     (pure data->PNG, CPU): visualization/epoch_metrics/*.png
  3. regenerate best_model/*.png via the production GPU pipeline
     (scripts/visualize_all.py, best_checkpoint is valid since best epoch
     did NOT flip) -> reflects the new score via single-source scoring.py.

Run AFTER recompute_noflip_lambda.py --apply has finished.

Usage:
  python scripts/reviz_noflip_lambda.py            # all no-flip canonical cells
  python scripts/reviz_noflip_lambda.py --only 286_..._clamp_pm4/WaDi/A1
"""
import os, sys, json, glob, shutil, argparse, subprocess
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
sys.argv_backup = list(sys.argv)
sys.argv = ['reviz']  # neutralize argparse in imported module
from scripts.run_base_experiments import plot_epoch_metrics
sys.argv = sys.argv_backup

CLASS = '/tmp/lambda_bestepoch_classification.json'
BACKUP_ROOT = '/home/ykio/notebooks/TSMAE/.trash/0601/lambda'
CANON = {'271', '274', '285', '286', '287'}


def base_of(run):
    p = run.split('_')
    return p[0] if p[1].startswith('2026') else f"{p[0]}_{p[1]}"


def backup_png(src, cell_rel, sub):
    dst = os.path.join(BACKUP_ROOT, cell_rel, 'visualization', sub, os.path.basename(src))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    args = ap.parse_args()
    cls = json.load(open(CLASS))
    swat_flip_runs = {c['run'] for c in cls if c['flip'] and 'SWaT' in c['ds']}
    targets = []
    for c in cls:
        if c['flip'] or base_of(c['run']) not in CANON:
            continue
        if 'SWaT' in c['ds'] and c['run'] in swat_flip_runs:
            continue
        targets.append(c)
    if args.only:
        targets = [c for c in targets if f"{c['run']}/{c['ds']}" == args.only]
    print(f"reviz no-flip canonical cells: {len(targets)}")

    for c in targets:
        run, ds = c['run'], c['ds']
        cell_dir = f'results/experiments/{run}/{ds}'
        cell_rel = f'{run}/{ds}'
        epoch_viz = os.path.join(cell_dir, 'visualization', 'epoch_metrics')
        best_viz = os.path.join(cell_dir, 'visualization', 'best_model')
        print(f"\n=== {base_of(run)}/{ds} (best={c['new_best']}) ===")
        # 1. backup
        for png in glob.glob(os.path.join(epoch_viz, '*.png')):
            backup_png(png, cell_rel, 'epoch_metrics')
        for png in glob.glob(os.path.join(best_viz, '*.png')):
            backup_png(png, cell_rel, 'best_model')
        # 2. epoch trends (CPU)
        em = os.path.join(cell_dir, 'epoch_metrics.json')
        if os.path.exists(em):
            rows = json.load(open(em))['epochs']
            os.makedirs(epoch_viz, exist_ok=True)
            try:
                plot_epoch_metrics(rows, epoch_viz)
                print(f"  [epoch_metrics] {len(rows)} rows -> trends regenerated")
            except Exception as e:
                print(f"  [epoch_metrics] FAIL: {e}")
        # 3. anomaly_threshold.png — npz-based (recomputed best-epoch npz), no GPU.
        #    Other best_model PNGs are model-forward (reconstruction/feature) and
        #    UNCHANGED for no-flip cells (same model + same best epoch); the
        #    score-overlay curves need the production GPU pipeline (offline path
        #    is stale) — reported, not regenerated here.
        try:
            from mae_anomaly.visualization import BestModelVisualizer
            from mae_anomaly import Config
            bc = {}
            bcp = os.path.join(cell_dir, 'best_config.json')
            if os.path.exists(bcp):
                bc = json.load(open(bcp))
            cfg = Config()
            for k, v in bc.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            viz = BestModelVisualizer.__new__(BestModelVisualizer)
            viz.output_dir = best_viz
            viz.config = cfg
            viz.test_loader = None
            os.makedirs(best_viz, exist_ok=True)
            viz.plot_anomaly_threshold(experiment_dir=cell_dir)
            print(f"  [anomaly_threshold] regenerated (npz-based)")
        except Exception as e:
            print(f"  [anomaly_threshold] FAIL: {e}")

    print("\nreviz done.")


if __name__ == '__main__':
    main()
