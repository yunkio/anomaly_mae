#!/usr/bin/env python
"""After the unlabeled queue (run_official_unlabeled_after.py: unlab100/unlab50) finishes,
run the official 271 BASELINE (CANON_271, NO geometry/loss override) for seeds 40, 41, 43, 44
— seed-variance companions to the existing seed-42 baseline. keep=False, 30ep, 4 datasets.
Each run -> post-warmup npz-viz. (2026-06-24)

Config per run = pure official baseline: `official=True num_epochs=30 random_seed=<S>
official_keep_checkpoints=False` (nothing else; CANON_271 supplies the geometry/loss).

Usage: python scripts/run_official_seeds_after.py
"""
import json, os, sys, subprocess, datetime, time
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
           'WaDi_A1': ['WaDi/A1'], 'WaDi_A2': ['WaDi/A2'], 'PSM': ['PSM']}
# Wait for the ENTIRE current campaign: master queue + both follow-up waiters. Seeds run
# only once all three are gone (robust to whatever inter-waiter chain order they use).
WAIT_TOKENS = ['run_official_resume_full_queue_after_pause',
               'run_official_odoff_featurewise_after_full_queue',
               'run_official_unlab_rankfix_after']
SEEDS = [40, 41, 43, 44]
BASE = 'official=True num_epochs=30 official_keep_checkpoints=False'


def queue_alive():
    for tok in WAIT_TOKENS:
        r = subprocess.run(['pgrep', '-f', tok], capture_output=True, text=True)
        pids = [p for p in r.stdout.split() if p and int(p) != os.getpid()]
        if pids:
            return True
    return False


def run(outdir, seed):
    override = f'{BASE} random_seed={seed}'
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
           '--output-base', outdir, '--dataset'] + ALL4 + ['--config-override', override]
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    print(f"[seeds] START seed{seed} -> {os.path.relpath(outdir, PROJECT)}\n  {override}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[seeds] DONE seed{seed} rc={rc}", flush=True)


def reviz(outdir, seed):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[seeds] reviz import FAIL: {e}", flush=True); return
    for ds in ALL4:
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
                print(f"[seeds] reviz seed{seed}/{sub} OK (post-warmup)", flush=True)
            except Exception as e:
                print(f"[seeds] reviz seed{seed}/{sub} FAIL: {e}", flush=True)


def main():
    try:
        open('/tmp/official_seeds_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[seeds] START {datetime.datetime.now():%Y%m%d_%H%M%S} — waiting for the unlabeled "
          f"queue (unlab100/unlab50) to finish...", flush=True)
    waited = 0
    while queue_alive():
        time.sleep(60); waited += 1
        if waited % 60 == 0:
            print(f"[seeds] still waiting ({waited} min)...", flush=True)
    print(f"[seeds] unlabeled queue gone after {waited} min — settling 20s.", flush=True)
    time.sleep(20)
    for seed in SEEDS:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_{seed}"
        run(outdir, seed)
        reviz(outdir, seed)
    print(f"[seeds] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
