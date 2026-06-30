#!/usr/bin/env python
"""Wait for the current maskratio queue, then run extra unlabeled official runs."""
import datetime
import os
import subprocess
import sys
import time

sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
WAIT_TOKEN = 'scripts/run_official_maskratio_after.py'
NEXT_SCRIPT = 'scripts/run_official_unlabeled_extra_after_maskratio.py'


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


def main():
    try:
        open('/tmp/official_unlab_extra_wait_maskratio_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(
        f"[unlab-extra-wait] START {datetime.datetime.now():%Y%m%d_%H%M%S} "
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
            print(
                f"[unlab-extra-wait] still waiting ({waited} min), pids={pids}",
                flush=True,
            )
    print(f"[unlab-extra-wait] maskratio done after {waited} min - settling 20s.", flush=True)
    time.sleep(20)
    env = dict(os.environ, PYTHONHASHSEED='42')
    rc = subprocess.run([sys.executable, NEXT_SCRIPT], cwd=PROJECT, env=env).returncode
    print(f"[unlab-extra-wait] DONE {NEXT_SCRIPT} rc={rc}", flush=True)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == '__main__':
    main()
