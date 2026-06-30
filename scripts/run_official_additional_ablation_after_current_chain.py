#!/usr/bin/env python
"""Append additional official ablations after the currently active official chain.

This watcher waits for scripts/run_official_wait_maskratio_extra_then_unlab.py
to finish. It then runs only the requested official-271 ablations, keeping every
other setting identical to the official baseline:

  exclanom : train_exclude_anomaly_segments=True
  noforce  : force_mask_anomaly=False
  nostudent: use_student=False
  td3sd3   : num_teacher_decoder_layers=3, num_student_decoder_layers=3
  td2sd2   : num_teacher_decoder_layers=2, num_student_decoder_layers=2
  warmup0  : teacher_only_warmup_epochs=0

Usage:
  PYTHONHASHSEED=42 python scripts/run_official_additional_ablation_after_current_chain.py
"""
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/ykio/notebooks/TSMAE')

PROJECT = '/home/ykio/notebooks/TSMAE'
OFFICIAL_DIR = Path(PROJECT) / 'results' / 'experiments' / 'official'
SEED = 42

WAIT_TOKEN = 'scripts/run_official_wait_maskratio_extra_then_unlab.py'

ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {
    'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
    'WaDi_A1': ['WaDi/A1'],
    'WaDi_A2': ['WaDi/A2'],
    'PSM': ['PSM'],
}
EXPECTED_CELLS = [cell for cells in SUBDIRS.values() for cell in cells]

BASE = 'official=True num_epochs=30 random_seed=42 official_keep_checkpoints=False'
EXPERIMENTS = [
    ('exclanom', 'train_exclude_anomaly_segments=True'),
    ('noforce', 'force_mask_anomaly=False'),
    ('nostudent', 'use_student=False'),
    ('td3sd3', 'num_teacher_decoder_layers=3 num_student_decoder_layers=3'),
    ('td2sd2', 'num_teacher_decoder_layers=2 num_student_decoder_layers=2'),
    ('warmup0', 'teacher_only_warmup_epochs=0'),
]


def active_wait_pids():
    """Return live PIDs whose command line is the current official append chain."""
    result = subprocess.run(
        ['ps', '-eo', 'pid=,cmd='],
        capture_output=True,
        text=True,
        check=False,
    )
    pids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, cmd = line.partition(' ')
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        if WAIT_TOKEN in cmd:
            pids.append(pid)
    return pids


def completed_tag(tag):
    """Skip a tag only if a prior official directory has all expected cells."""
    for run_dir in sorted(OFFICIAL_DIR.glob(f'*_{tag}')):
        if all((run_dir / cell / 'epoch_metrics.json').exists() for cell in EXPECTED_CELLS):
            return run_dir
    return None


def run_experiment(outdir, override, label):
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
        f"[official-extra-ablation] START {label} -> {os.path.relpath(outdir, PROJECT)}\n"
        f"  {override}",
        flush=True,
    )
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[official-extra-ablation] DONE {label} rc={rc}", flush=True)
    if rc != 0:
        raise SystemExit(rc)


def reviz(outdir, label):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as exc:
        print(f"[official-extra-ablation] reviz import FAIL: {exc}", flush=True)
        return

    for dataset in ALL4:
        for subdir in SUBDIRS[dataset]:
            cell = os.path.join(outdir, subdir)
            if not os.path.exists(os.path.join(cell, 'epoch_metrics.json')):
                continue
            try:
                with open(os.path.join(cell, 'best_config.json')) as handle:
                    best_config = json.load(handle)
                best_config = best_config.get('config', best_config)
                config = Config()
                for key, value in best_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                viz = BestModelVisualizer.__new__(BestModelVisualizer)
                viz.output_dir = os.path.join(cell, 'visualization', 'best_model')
                viz.config = config
                viz.test_loader = None
                os.makedirs(viz.output_dir, exist_ok=True)
                viz.plot_anomaly_threshold(experiment_dir=cell)
                viz.plot_anomaly_threshold_test_event(experiment_dir=cell)
                print(f"[official-extra-ablation] reviz {label}/{subdir} OK", flush=True)
            except Exception as exc:
                print(f"[official-extra-ablation] reviz {label}/{subdir} FAIL: {exc}", flush=True)


def main():
    try:
        with open('/tmp/official_additional_ablation_pid.txt', 'w') as handle:
            handle.write(str(os.getpid()))
    except Exception:
        pass

    print(
        f"[official-extra-ablation] START {datetime.datetime.now():%Y%m%d_%H%M%S} "
        f"waiting for {WAIT_TOKEN}",
        flush=True,
    )
    waited = 0
    while True:
        pids = active_wait_pids()
        if not pids:
            break
        time.sleep(60)
        waited += 1
        if waited % 30 == 0:
            print(f"[official-extra-ablation] still waiting ({waited} min), pids={pids}", flush=True)
    print(f"[official-extra-ablation] prior chain done after {waited} min - settling 20s.", flush=True)
    time.sleep(20)

    for tag, extra in EXPERIMENTS:
        prior = completed_tag(tag)
        if prior is not None:
            print(f"[official-extra-ablation] SKIP {tag}: complete prior run {prior}", flush=True)
            continue
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_42_{tag}"
        run_experiment(outdir, f"{BASE} {extra}", tag)
        reviz(outdir, tag)

    print(f"[official-extra-ablation] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
