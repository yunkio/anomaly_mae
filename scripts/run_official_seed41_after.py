#!/usr/bin/env python
"""After the resume chain (run_official_resume_chain.py: 30ep-resume -> 10ep -> freeze)
finishes, run ONE more official 30ep with seed=41 (keep=False, same config as the current
30ep) + post-warmup npz-viz. Queued per user request (2026-06-23).

Usage: PYTHONHASHSEED=41 python scripts/run_official_seed41_after.py
"""
import json, os, sys, subprocess, datetime, time
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
SEED = 41
ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
           'WaDi_A1': ['WaDi/A1'], 'WaDi_A2': ['WaDi/A2'], 'PSM': ['PSM']}
WAIT_TOKEN = 'run_official_resume_chain'


def chain_alive():
    r = subprocess.run(['pgrep', '-f', WAIT_TOKEN], capture_output=True, text=True)
    pids = [p for p in r.stdout.split() if p and int(p) != os.getpid()]
    return bool(pids)


def run(outdir, datasets, override, label):
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
           '--output-base', outdir, '--dataset'] + datasets + ['--config-override', override]
    env = dict(os.environ, PYTHONHASHSEED=str(SEED))
    print(f"[seed41] START {label} -> {os.path.relpath(outdir, PROJECT)} | {override}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[seed41] DONE {label} rc={rc}", flush=True)


def reviz(outdir, datasets, label):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[seed41] reviz import FAIL: {e}", flush=True); return
    for ds in datasets:
        for sub in SUBDIRS[ds]:
            cell = os.path.join(outdir, sub)
            if not os.path.exists(os.path.join(cell, 'epoch_metrics.json')):
                continue
            try:
                bc = json.load(open(os.path.join(cell, 'best_config.json'))); bc = bc.get('config', bc)
                cfg = Config()
                for k, v in bc.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
                viz = BestModelVisualizer.__new__(BestModelVisualizer)
                viz.output_dir = os.path.join(cell, 'visualization', 'best_model')
                viz.config = cfg; viz.test_loader = None
                os.makedirs(viz.output_dir, exist_ok=True)
                viz.plot_anomaly_threshold(experiment_dir=cell)
                viz.plot_anomaly_threshold_test_event(experiment_dir=cell)
                print(f"[seed41] reviz {sub} OK (post-warmup)", flush=True)
            except Exception as e:
                print(f"[seed41] reviz {sub} FAIL: {e}", flush=True)


def main():
    try:
        open('/tmp/official_seed41_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[seed41] START {datetime.datetime.now():%Y%m%d_%H%M%S} — waiting for the resume "
          f"chain (30ep->10ep->freeze) to finish...", flush=True)
    waited = 0
    while chain_alive():
        time.sleep(60); waited += 1
        if waited % 60 == 0:
            print(f"[seed41] still waiting ({waited} min)...", flush=True)
    print(f"[seed41] chain gone after {waited} min — settling 20s.", flush=True)
    time.sleep(20)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_41"
    run(outdir, ALL4, 'official=True num_epochs=30 random_seed=41 official_keep_checkpoints=False', '30ep-seed41')
    reviz(outdir, ALL4, 'seed41')
    print(f"[seed41] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
