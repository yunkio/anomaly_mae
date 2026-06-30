#!/usr/bin/env python
"""Resume the interrupted unlab50 run, then launch masking-ratio official runs.

This is a recovery wrapper for:
  results/experiments/official/271_20260625_160747_30ep_42_unlab50

It keeps the original unlab50 config exactly as scheduled, resumes from
latest_checkpoint.pt inside that output directory, refreshes post-warmup viz, and
then runs scripts/run_official_maskratio_after.py.

Usage:
  PYTHONHASHSEED=42 python scripts/run_official_unlab50_resume_then_maskratio.py
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
UNLAB50_OUTDIR = (
    f'{PROJECT}/results/experiments/official/271_20260625_160747_30ep_42_unlab50'
)
UNLAB50_OVERRIDE = (
    'official=True num_epochs=30 random_seed=42 official_keep_checkpoints=False '
    'train_label_mask_frac=0.5'
)


def run_unlab50_resume():
    cmd = [
        sys.executable,
        'scripts/run_base_experiments.py',
        '--set',
        'C',
        '--no-wait',
        '--output-base',
        UNLAB50_OUTDIR,
        '--dataset',
        *ALL4,
        '--config-override',
        UNLAB50_OVERRIDE,
    ]
    env = dict(os.environ, PYTHONHASHSEED=str(SEED))
    print(
        f"[resume+maskratio] RESUME unlab50 -> {os.path.relpath(UNLAB50_OUTDIR, PROJECT)}\n"
        f"  {UNLAB50_OVERRIDE}",
        flush=True,
    )
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[resume+maskratio] DONE unlab50 resume rc={rc}", flush=True)
    if rc != 0:
        raise SystemExit(rc)


def reviz_unlab50():
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[resume+maskratio] unlab50 reviz import FAIL: {e}", flush=True)
        return
    for ds in ALL4:
        for sub in SUBDIRS[ds]:
            cell = os.path.join(UNLAB50_OUTDIR, sub)
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
                print(f"[resume+maskratio] reviz unlab50/{sub} OK (post-warmup)", flush=True)
            except Exception as e:
                print(f"[resume+maskratio] reviz unlab50/{sub} FAIL: {e}", flush=True)


def run_maskratio_queue():
    cmd = [sys.executable, 'scripts/run_official_maskratio_after.py']
    env = dict(os.environ, PYTHONHASHSEED=str(SEED))
    print("[resume+maskratio] START maskratio queue", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[resume+maskratio] DONE maskratio queue rc={rc}", flush=True)
    if rc != 0:
        raise SystemExit(rc)


def main():
    try:
        open('/tmp/official_unlab50_resume_then_maskratio_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[resume+maskratio] START {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)
    run_unlab50_resume()
    reviz_unlab50()
    run_maskratio_queue()
    print(f"[resume+maskratio] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
