#!/usr/bin/env python
"""Wait for the current maskratio queue, then run extra maskratio and unlab queues."""
import datetime
import os
import subprocess
import sys
import time

sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
WAIT_TOKEN = 'scripts/run_official_maskratio_after.py'
MASKRATIO_EXTRA_SCRIPT = 'scripts/run_official_maskratio_extra_after.py'
UNLAB_EXTRA_SCRIPT = 'scripts/run_official_unlabeled_extra_after_maskratio.py'


def queue_alive():
    r = subprocess.run(['pgrep', '-f', WAIT_TOKEN], capture_output=True, text=True)
    pids = []
    for p in r.stdout.split():
        try:
            pid = int(p)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)
    return pids


def run_script(path, label):
    env = dict(os.environ, PYTHONHASHSEED='42')
    print(f"[official-chain] START {label}: {path}", flush=True)
    rc = subprocess.run([sys.executable, path], cwd=PROJECT, env=env).returncode
    print(f"[official-chain] DONE {label}: rc={rc}", flush=True)
    if rc != 0:
        raise SystemExit(rc)


def main():
    try:
        open('/tmp/official_wait_maskratio_extra_then_unlab_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(
        f"[official-chain] START {datetime.datetime.now():%Y%m%d_%H%M%S} "
        f"waiting for {WAIT_TOKEN}",
        flush=True,
    )
    waited = 0
    while True:
        pids = queue_alive()
        if not pids:
            break
        time.sleep(60)
        waited += 1
        if waited % 30 == 0:
            print(f"[official-chain] still waiting ({waited} min), pids={pids}", flush=True)
    print(f"[official-chain] current maskratio done after {waited} min - settling 20s.", flush=True)
    time.sleep(20)
    run_script(MASKRATIO_EXTRA_SCRIPT, 'maskratio-extra')
    run_script(UNLAB_EXTRA_SCRIPT, 'unlab-extra')
    print(f"[official-chain] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
