#!/usr/bin/env python
"""Run extra official masking-ratio ablations requested after the first queue.

The first maskratio queue was already running when these were requested, so this
append queue runs only the newly requested non-default ratios:
  maskr090 -> masking_ratio=0.90
  maskr060 -> masking_ratio=0.60

Same official 271 setup otherwise: seed=42, 30 epochs, keep=False, 4 datasets.

Usage: PYTHONHASHSEED=42 python scripts/run_official_maskratio_extra_after.py
"""
import datetime
import json
import os
import subprocess
import sys

sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
SEED = 42
ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {
    'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
    'WaDi_A1': ['WaDi/A1'],
    'WaDi_A2': ['WaDi/A2'],
    'PSM': ['PSM'],
}
BASE = 'official=True num_epochs=30 random_seed=42 official_keep_checkpoints=False'
RATIOS = [0.90, 0.60]


def ratio_tag(ratio):
    return f"maskr{int(round(ratio * 100)):03d}"


def run(outdir, ratio):
    override = f'{BASE} masking_ratio={ratio:g}'
    cmd = [
        sys.executable,
        'scripts/run_base_experiments.py',
        '--set',
        'C',
        '--no-wait',
        '--output-base',
        outdir,
        '--dataset',
        *ALL4,
        '--config-override',
        override,
    ]
    env = dict(os.environ, PYTHONHASHSEED=str(SEED))
    print(
        f"[maskratio-extra] START {ratio_tag(ratio)} -> {os.path.relpath(outdir, PROJECT)}\n"
        f"  {override}",
        flush=True,
    )
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[maskratio-extra] DONE {ratio_tag(ratio)} rc={rc}", flush=True)
    if rc != 0:
        raise SystemExit(rc)


def reviz(outdir, ratio):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[maskratio-extra] reviz import FAIL: {e}", flush=True)
        return
    for ds in ALL4:
        for sub in SUBDIRS[ds]:
            cell = os.path.join(outdir, sub)
            if not os.path.exists(os.path.join(cell, 'epoch_metrics.json')):
                continue
            try:
                bc = json.load(open(os.path.join(cell, 'best_config.json')))
                bc = bc.get('config', bc)
                cfg = Config()
                for k, v in bc.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
                viz = BestModelVisualizer.__new__(BestModelVisualizer)
                viz.output_dir = os.path.join(cell, 'visualization', 'best_model')
                viz.config = cfg
                viz.test_loader = None
                os.makedirs(viz.output_dir, exist_ok=True)
                viz.plot_anomaly_threshold(experiment_dir=cell)
                viz.plot_anomaly_threshold_test_event(experiment_dir=cell)
                print(
                    f"[maskratio-extra] reviz {ratio_tag(ratio)}/{sub} OK (post-warmup)",
                    flush=True,
                )
            except Exception as e:
                print(f"[maskratio-extra] reviz {ratio_tag(ratio)}/{sub} FAIL: {e}", flush=True)


def main():
    try:
        open('/tmp/official_maskratio_extra_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(
        f"[maskratio-extra] START {datetime.datetime.now():%Y%m%d_%H%M%S} "
        f"ratios={RATIOS}",
        flush=True,
    )
    for ratio in RATIOS:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_42_{ratio_tag(ratio)}"
        run(outdir, ratio)
        reviz(outdir, ratio)
    print(f"[maskratio-extra] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
