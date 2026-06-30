#!/usr/bin/env python
"""Run OD-off and feature-wise official ablations after the current full queue.

This script is intentionally separate from the already-running master queue:
that process has its QUEUE list loaded in memory, so editing the source file
cannot append work to it. This watcher waits for the full queue process to end,
verifies that every predecessor tag is complete, then runs only the two new
experiments with state-aware skip/resume behavior.

Run with:
  PYTHONHASHSEED=42 conda run --no-capture-output -n dc_vis python \
    scripts/run_official_odoff_featurewise_after_full_queue.py
"""
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/ykio/notebooks/TSMAE')

PROJECT = Path('/home/ykio/notebooks/TSMAE')
OFFICIAL_DIR = PROJECT / 'results' / 'experiments' / 'official'
SEED = 42
WAIT_PID_FILE = Path('/tmp/official_resume_full_queue_after_pause_pid.txt')

ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {
    'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
    'WaDi_A1': ['WaDi/A1'],
    'WaDi_A2': ['WaDi/A2'],
    'PSM': ['PSM'],
}
EXPECTED_CELLS = [cell for cells in SUBDIRS.values() for cell in cells]

BASE = 'official=True num_epochs=30 random_seed=42 official_keep_checkpoints=False'

PREDECESSOR_TAGS = [
    'maskr005',
    'maskr010',
    'maskr030',
    'maskr050',
    'maskr075',
    'maskr090',
    'maskr060',
    'unlab10',
    'unlab75',
    'unlab25',
    'exclanom',
    'noforce',
    'nostudent',
    'td3sd3',
    'td2sd2',
    'warmup0',
]

QUEUE = [
    ('odoff', 'use_output_discrepancy=False'),
    ('featwise', 'masking_strategy=feature_wise force_mask_anomaly=False'),
    ('featwise_maskr050', 'masking_strategy=feature_wise force_mask_anomaly=False masking_ratio=0.50'),
]


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def wait_for_full_queue():
    if WAIT_PID_FILE.exists():
        try:
            pid = int(WAIT_PID_FILE.read_text().strip())
        except ValueError:
            pid = None
        if pid:
            while pid_alive(pid):
                print(
                    f"[official-extra] waiting for full queue pid={pid} "
                    f"at {datetime.datetime.now():%Y%m%d_%H%M%S}",
                    flush=True,
                )
                time.sleep(300)

    missing = [tag for tag in PREDECESSOR_TAGS if completed_run(tag) is None]
    if missing:
        print(
            "[official-extra] predecessor queue is not complete; "
            f"not starting extras. missing={missing}",
            flush=True,
        )
        raise SystemExit(2)


def is_complete(run_dir):
    return all((run_dir / cell / 'epoch_metrics.json').exists() for cell in EXPECTED_CELLS)


def completed_run(tag):
    for run_dir in sorted(OFFICIAL_DIR.glob(f'*_{tag}')):
        if is_complete(run_dir):
            return run_dir
    return None


def incomplete_runs(tag):
    runs = [p for p in OFFICIAL_DIR.glob(f'*_{tag}') if p.is_dir() and not is_complete(p)]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def has_resume_checkpoint(run_dir):
    return any((run_dir / cell / 'checkpoints' / 'latest_checkpoint.pt').exists()
               for cell in EXPECTED_CELLS)


def select_outdir(tag):
    done = completed_run(tag)
    if done is not None:
        return None, done
    for run_dir in incomplete_runs(tag):
        if has_resume_checkpoint(run_dir):
            return run_dir, None
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return OFFICIAL_DIR / f'271_{ts}_30ep_42_{tag}', None


def run_experiment(outdir, override, tag):
    cmd = [
        sys.executable,
        'scripts/run_base_experiments.py',
        '--set',
        'C',
        '--no-wait',
        '--output-base',
        str(outdir),
        '--dataset',
        *ALL4,
        '--config-override',
        override,
    ]
    env = dict(os.environ, PYTHONHASHSEED=str(SEED))
    print(f"[official-extra] START {tag} -> {outdir.relative_to(PROJECT)}", flush=True)
    print(f"[official-extra] override: {override}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[official-extra] DONE {tag} rc={rc}", flush=True)
    if rc != 0:
        raise SystemExit(rc)


def reviz(outdir, tag):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as exc:
        print(f"[official-extra] reviz import FAIL: {exc}", flush=True)
        return

    for dataset in ALL4:
        for subdir in SUBDIRS[dataset]:
            cell = outdir / subdir
            if not (cell / 'epoch_metrics.json').exists():
                continue
            try:
                with open(cell / 'best_config.json') as handle:
                    best_config = json.load(handle)
                best_config = best_config.get('config', best_config)
                config = Config()
                for key, value in best_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                viz = BestModelVisualizer.__new__(BestModelVisualizer)
                viz.output_dir = str(cell / 'visualization' / 'best_model')
                viz.config = config
                viz.test_loader = None
                os.makedirs(viz.output_dir, exist_ok=True)
                viz.plot_anomaly_threshold(experiment_dir=str(cell))
                viz.plot_anomaly_threshold_test_event(experiment_dir=str(cell))
                print(f"[official-extra] reviz {tag}/{subdir} OK", flush=True)
            except Exception as exc:
                print(f"[official-extra] reviz {tag}/{subdir} FAIL: {exc}", flush=True)


def main():
    try:
        with open('/tmp/official_odoff_featurewise_after_full_queue_pid.txt', 'w') as handle:
            handle.write(str(os.getpid()))
    except Exception:
        pass

    print(f"[official-extra] START {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)
    wait_for_full_queue()
    for tag, extra in QUEUE:
        outdir, done = select_outdir(tag)
        if done is not None:
            print(f"[official-extra] SKIP {tag}: complete prior run {done}", flush=True)
            continue
        run_experiment(outdir, f"{BASE} {extra}", tag)
        reviz(outdir, tag)
    print(f"[official-extra] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
