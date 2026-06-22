#!/usr/bin/env python
"""Run ONE official Exp-271 with freeze_encoder_only=True, AFTER the current official waiter
(run_official_271_after_queue.py = the 30ep + 10ep sequence) finishes. Does not disturb it.

Same as the current official 271 (4 datasets SWaT_A1A2/WaDi_A1/WaDi_A2/PSM, seed 42,
num_epochs=30 -> warmup 15 auto, official_keep_checkpoints=False) PLUS freeze_encoder_only=True
(encoder frozen at the warmup boundary; teacher/student decoders keep training).

Usage: PYTHONHASHSEED=42 python scripts/run_official_freeze_after.py
"""
import os, sys, subprocess, datetime, time, shutil

PROJECT = '/home/ykio/notebooks/TSMAE'
SEED = 42
EPOCHS = 30
KEEP_CKPT = False
MIN_FREE_GB = 8
BASE4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
WAIT_TOKEN = 'run_official_271_after_queue'  # waiter1 (30ep + 10ep)


def waiter1_alive():
    r = subprocess.run(['pgrep', '-f', WAIT_TOKEN], capture_output=True, text=True)
    pids = [p for p in r.stdout.split() if p and int(p) != os.getpid()]
    return bool(pids)


def main():
    try:
        open('/tmp/official_freeze_waiter_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[freeze-waiter] START {datetime.datetime.now():%Y%m%d_%H%M%S} — waiting for "
          f"the 30ep+10ep official waiter to finish...", flush=True)
    waited = 0
    while waiter1_alive():
        time.sleep(60); waited += 1
        if waited % 30 == 0:
            print(f"[freeze-waiter] still waiting ({waited} min)...", flush=True)
    print(f"[freeze-waiter] waiter1 gone after {waited} min — settling 20s.", flush=True)
    time.sleep(20)

    free = shutil.disk_usage(PROJECT).free / 1e9
    if free < MIN_FREE_GB:
        print(f"[freeze-271] ABORT — only {free:.1f}G free (< {MIN_FREE_GB}G guard).", flush=True)
        sys.exit(2)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(PROJECT, f'results/experiments/official/271_{ts}_{EPOCHS}ep_{SEED}_freezeenc')
    os.makedirs(os.path.dirname(outdir), exist_ok=True)
    override = (f'official=True num_epochs={EPOCHS} random_seed={SEED} '
               f'official_keep_checkpoints={KEEP_CKPT} freeze_encoder_only=True')
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
           '--output-base', outdir, '--dataset'] + list(BASE4) + ['--config-override', override]
    env = dict(os.environ, PYTHONHASHSEED=str(SEED))
    print(f"[freeze-271] disk OK {free:.1f}G. START {ts} -> {os.path.relpath(outdir, PROJECT)}", flush=True)
    print(f"[freeze-271] override: {override}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[freeze-271] DONE rc={rc}", flush=True)


if __name__ == '__main__':
    main()
