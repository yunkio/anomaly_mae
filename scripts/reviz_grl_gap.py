#!/usr/bin/env python
"""Re-visualize GRL panels after adding the |normal-anomaly| degeneracy gap.

2026-06-01. balanced_acc=0.5 is deceptive (coexists with full degeneracy
normal=0/anomaly=1). We added `grl_acc_gap = |normal_acc - anomaly_acc|` to the
two GRL plots:
  (1) visualization/epoch_metrics/epoch_grl.png   (plot_epoch_metrics, accuracy panel)
  (2) visualization/best_model/GRL_contribution_trend.png  ((C) panel)

Both are pure data->PNG (no model forward / no GPU). This driver, for every
GRL-active cell of the 9 affected runs:
  - backs up the existing epoch_metrics/*.png + GRL_contribution_trend.png
  - regenerates (1) from epoch_metrics.json (now carrying grl_acc_gap)
  - regenerates (2) from training_histories.json (gap derived in-plot)

Backups -> ./.trash/0601/grl_gap_viz_backup/<run>/<cell>/...
"""
import os, sys, json, glob, shutil
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
sys.argv = ['reviz']  # neutralize argparse in imported module
from scripts.run_base_experiments import plot_epoch_metrics
from mae_anomaly.visualization.best_model_visualizer import BestModelVisualizer

RUNS = ['271_20260529_100418_271canon_baseline', '271_lr2_20260529_225351_baseline',
        '271_lr_20260529_225351_baseline', '274_20260529_100418_274canon_balsamp',
        '274_lr2_20260529_225351_balsamp', '274_lr_20260529_225351_balsamp',
        '285_20260529_225351_no_fm', '286_20260529_225351_clamp_pm4',
        '287_20260529_225351_unmask']
BK = '/home/ykio/notebooks/TSMAE/.trash/0601/grl_gap_viz_backup'


def backup(src, rel):
    dst = os.path.join(BK, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)


def main():
    cells = []
    for r in RUNS:
        cells += sorted(glob.glob(f'results/experiments/{r}/**/epoch_metrics.json', recursive=True))
    print(f"GRL-active cells: {len(cells)}")
    stats = {'epoch_grl': 0, 'grl_contrib': 0, 'skip_contrib': 0, 'err': 0}
    for em in cells:
        cell_dir = os.path.dirname(em)
        rel = cell_dir.split('experiments/')[1]
        epoch_viz = os.path.join(cell_dir, 'visualization', 'epoch_metrics')
        best_viz = os.path.join(cell_dir, 'visualization', 'best_model')
        # --- backup existing PNGs ---
        for png in glob.glob(os.path.join(epoch_viz, '*.png')):
            backup(png, os.path.join(rel, 'epoch_metrics', os.path.basename(png)))
        backup(os.path.join(best_viz, 'GRL_contribution_trend.png'),
               os.path.join(rel, 'best_model', 'GRL_contribution_trend.png'))
        # --- (1) epoch_grl.png (regenerates the whole epoch_metrics PNG set) ---
        try:
            rows = json.load(open(em))['epochs']
            os.makedirs(epoch_viz, exist_ok=True)
            plot_epoch_metrics(rows, epoch_viz)
            if os.path.exists(os.path.join(epoch_viz, 'epoch_grl.png')):
                stats['epoch_grl'] += 1
        except Exception as e:
            print(f"  ! epoch_grl FAIL {rel}: {e}"); stats['err'] += 1
        # --- (2) GRL_contribution_trend.png ---
        th = os.path.join(cell_dir, 'training_histories.json')
        if os.path.exists(th):
            try:
                os.makedirs(best_viz, exist_ok=True)
                # bypass __init__ (it forces GPU collect_predictions); this plot
                # only needs self.output_dir + on-disk training_histories.json
                viz = BestModelVisualizer.__new__(BestModelVisualizer)
                viz.output_dir = best_viz
                viz.plot_grl_contribution_trend(experiment_dir=cell_dir, history=None)
                if os.path.exists(os.path.join(best_viz, 'GRL_contribution_trend.png')):
                    stats['grl_contrib'] += 1
            except Exception as e:
                print(f"  ! grl_contrib FAIL {rel}: {e}"); stats['err'] += 1
        else:
            stats['skip_contrib'] += 1
    print(f"\nepoch_grl.png regenerated : {stats['epoch_grl']}/{len(cells)}")
    print(f"GRL_contribution_trend.png: {stats['grl_contrib']} (skip no-history {stats['skip_contrib']})")
    print(f"errors: {stats['err']}")
    print(f"backups -> {BK}")


if __name__ == '__main__':
    main()
