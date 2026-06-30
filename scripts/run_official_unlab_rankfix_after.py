#!/usr/bin/env python
"""Re-run the unlab10~100 official ablations AFTER the current campaign finishes.

Why this exists
---------------
The `train_label_mask_frac` masking was fixed on 2026-06-30 from position-based
(zero the back frac of the TRAIN timeline) to rank-based (unlabel the
chronologically-last frac of the TRAIN ANOMALY timepoints) — see
mae_anomaly/dataset_sliding.py + config.py. The old unlab10~100 results were
produced with the buggy position-based masking (which wiped ~all train anomalies
at the smallest frac because every train anomaly sits in the back few % of the
timeline → unlab10≈unlab100) and have been deleted. This watcher waits for the
in-flight official launchers to finish, then re-runs the five unlab experiments
with FRESH timestamped dirs so they pick up the corrected code.

The currently-running master queue has its QUEUE list loaded in memory, so the
source edit cannot append to it — hence this separate watcher (same chaining
pattern as run_official_odoff_featurewise_after_full_queue.py).

Run with:
  PYTHONHASHSEED=42 conda run --no-capture-output -n dc_vis python \
    scripts/run_official_unlab_rankfix_after.py
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

# Wait until BOTH in-flight launchers are dead AND no run_base is active, so we
# never contend for the GPU with the current campaign.
WAIT_PID_FILES = [
    Path('/tmp/official_resume_full_queue_after_pause_pid.txt'),
    Path('/tmp/official_odoff_featurewise_after_full_queue_pid.txt'),
]

ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {
    'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
    'WaDi_A1': ['WaDi/A1'],
    'WaDi_A2': ['WaDi/A2'],
    'PSM': ['PSM'],
}
EXPECTED_CELLS = [cell for cells in SUBDIRS.values() for cell in cells]

BASE = 'official=True num_epochs=30 random_seed=42 official_keep_checkpoints=False'

# Same override pattern as the original unlab runs (only train_label_mask_frac);
# all other settings inherit the current campaign defaults → internally consistent
# with the rest of the recent (June-25+) ablations.
QUEUE = [
    ('unlab10', 'train_label_mask_frac=0.10'),
    ('unlab25', 'train_label_mask_frac=0.25'),
    ('unlab50', 'train_label_mask_frac=0.50'),
    ('unlab75', 'train_label_mask_frac=0.75'),
    ('unlab100', 'train_label_mask_frac=1.00'),
]


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def any_launcher_alive():
    for f in WAIT_PID_FILES:
        if f.exists():
            try:
                pid = int(f.read_text().strip())
            except ValueError:
                continue
            if pid_alive(pid):
                return pid
    return None


def run_base_alive():
    out = subprocess.run(['ps', '-eo', 'cmd', '-ww'], capture_output=True, text=True).stdout
    return any('run_base_experiments.py' in ln and 'grep' not in ln
               for ln in out.splitlines())


def wait_for_campaign():
    while True:
        pid = any_launcher_alive()
        if pid:
            print(f"[unlab-rankfix] waiting: launcher pid={pid} alive "
                  f"at {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)
            time.sleep(300)
            continue
        if run_base_alive():
            print(f"[unlab-rankfix] waiting: run_base still active "
                  f"at {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)
            time.sleep(120)
            continue
        break
    print(f"[unlab-rankfix] campaign idle — starting reruns "
          f"at {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


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
        '--set', 'C',
        '--no-wait',
        '--output-base', str(outdir),
        '--dataset', *ALL4,
        '--config-override', override,
    ]
    env = dict(os.environ, PYTHONHASHSEED=str(SEED))
    print(f"[unlab-rankfix] START {tag} -> {outdir.relative_to(PROJECT)}", flush=True)
    print(f"[unlab-rankfix] override: {override}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[unlab-rankfix] DONE {tag} rc={rc}", flush=True)
    if rc != 0:
        raise SystemExit(rc)


def reviz(outdir, tag):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as exc:
        print(f"[unlab-rankfix] reviz import FAIL: {exc}", flush=True)
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
                print(f"[unlab-rankfix] reviz {tag}/{subdir} OK", flush=True)
            except Exception as exc:
                print(f"[unlab-rankfix] reviz {tag}/{subdir} FAIL: {exc}", flush=True)


def main():
    try:
        with open('/tmp/official_unlab_rankfix_after_pid.txt', 'w') as handle:
            handle.write(str(os.getpid()))
    except Exception:
        pass

    print(f"[unlab-rankfix] START {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)
    wait_for_campaign()
    for tag, extra in QUEUE:
        outdir, done = select_outdir(tag)
        if done is not None:
            print(f"[unlab-rankfix] SKIP {tag}: complete prior run {done}", flush=True)
            continue
        run_experiment(outdir, f"{BASE} {extra}", tag)
        reviz(outdir, tag)
    print(f"[unlab-rankfix] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
