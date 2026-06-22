#!/usr/bin/env python
"""Run official Exp-271 runs after the current resume_dedup_v2f queue (336->337) finishes.

- Waits until no `resume_dedup_v2f.py` driver process remains (avoids GPU double-booking).
- Then runs, SEQUENTIALLY, each entry in OFFICIAL_RUNS via run_base_experiments with
  official=True (CANON_271 base = exact 271), random_seed=42, over 4 datasets ONLY
  (SWaT_A1A2, WaDi_A1, WaDi_A2, PSM).
- Each run: output base results/experiments/official/271_<TS>_<EP>ep_42 (TS = that run's
  start time), its own disk guard (skip if free < MIN_FREE_GB).
- KEEP_CKPT controls official_epochs/ per-epoch weight dumps. False (disk-safe) keeps
  every-epoch eval + per-epoch score npz + causal score + viz + best result, but NOT the
  112MB/epoch model weights (~13G for 4 DS x 30 ep). Set True ONLY if enough free disk.

OFFICIAL_RUNS (queued in order):
  1. 30 epochs, warmup auto (= num_epochs//2 = 15)
  2. 10 epochs, warmup 5 (explicit)

Usage:
  PYTHONHASHSEED=42 python scripts/run_official_271_after_queue.py
"""
import json, os, sys, subprocess, datetime, time, shutil

PROJECT = '/home/ykio/notebooks/TSMAE'
SEED = 42
KEEP_CKPT = False   # disk-safe default; see module docstring
MIN_FREE_GB = 8     # abort a run (don't blow disk) if free space below this at its start
BASE4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']

# Each run: epochs + warmup (None = auto num_epochs//2). Run sequentially, in order.
OFFICIAL_RUNS = [
    {'epochs': 30, 'warmup': None},   # original official 271
    {'epochs': 10, 'warmup': 5},      # user (2026-06-22): epoch 10, teacher warm-up 5
]


def queue_running():
    r = subprocess.run(['pgrep', '-f', 'resume_dedup_v2f.py'], capture_output=True, text=True)
    return bool(r.stdout.strip())


def run_one(epochs, warmup):
    free_gb = shutil.disk_usage(PROJECT).free / 1e9
    if free_gb < MIN_FREE_GB:
        print(f"[official-271] ABORT {epochs}ep — only {free_gb:.1f}G free (< {MIN_FREE_GB}G guard). "
              f"Free disk then re-run: python scripts/run_official_271_after_queue.py", flush=True)
        return 2
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(PROJECT, f'results/experiments/official/271_{ts}_{epochs}ep_{SEED}')
    os.makedirs(os.path.dirname(outdir), exist_ok=True)
    datasets = list(BASE4)   # SWaT_A1A2, WaDi_A1, WaDi_A2, PSM only
    override = (f'official=True num_epochs={epochs} random_seed={SEED} '
               f'official_keep_checkpoints={KEEP_CKPT}')
    if warmup is not None:
        override += f' teacher_only_warmup_epochs={warmup}'
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
           '--output-base', outdir, '--dataset'] + datasets + ['--config-override', override]
    env = dict(os.environ, PYTHONHASHSEED=str(SEED))
    print(f"[official-271] disk OK: {free_gb:.1f}G free. START {ts} {epochs}ep "
          f"-> {os.path.relpath(outdir, PROJECT)}", flush=True)
    print(f"[official-271] {len(datasets)} datasets | override: {override}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[official-271] DONE {epochs}ep rc={rc}", flush=True)
    return rc


def main():
    try:
        open('/tmp/official_271_waiter_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[official-waiter] START {datetime.datetime.now():%Y%m%d_%H%M%S} — waiting for "
          f"resume_dedup_v2f queue (336->337) to finish...", flush=True)
    waited = 0
    while queue_running():
        time.sleep(60); waited += 1
        if waited % 30 == 0:
            print(f"[official-waiter] still waiting ({waited} min)...", flush=True)
    print(f"[official-waiter] queue gone after {waited} min — settling 15s.", flush=True)
    time.sleep(15)

    for i, cfg in enumerate(OFFICIAL_RUNS, 1):
        print(f"[official-waiter] === run {i}/{len(OFFICIAL_RUNS)}: "
              f"{cfg['epochs']}ep warmup={cfg['warmup']} ===", flush=True)
        run_one(cfg['epochs'], cfg['warmup'])
    print(f"[official-waiter] ALL OFFICIAL RUNS DONE", flush=True)


if __name__ == '__main__':
    main()
