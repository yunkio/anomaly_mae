#!/usr/bin/env python
"""Extra seed-variance runs 45..49 — appended AFTER the first seeds launcher (seed43/44).

[2026-07-08 — user] Seeds 43/44 are already running via run_official_seeds_after.py. This launcher
adds seeds 45,46,47,48,49 right after them (before the dense/exclude sweeps). It waits on the first
seeds launcher's token so 43/44 finish first; dense/discsnr were rewired to also wait on THIS token.

Final order: seed43,44 (seeds_after) -> seed45..49 (this) -> dense[excl30/70/90r] -> exclR[excl25/50/75r]
-> discsnr_refill -> odofffeat_redo.

Config per run = pure official baseline: `official=True num_epochs=30 random_seed=<S>
official_keep_checkpoints=False`. keep=False, 30ep, 4 datasets. Each run -> post-warmup npz-viz.

Usage: python scripts/run_official_seeds_extra_after.py
"""
import json, os, sys, subprocess, datetime, time
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
           'WaDi_A1': ['WaDi/A1'], 'WaDi_A2': ['WaDi/A2'], 'PSM': ['PSM']}
WAIT_TOKENS = ['run_official_seeds_after']  # wait for seed43/44 to finish first
SEEDS = [49]  # [2026-07-10 pause/resume #2] seed45..48 COMPLETED (4/4 each) — dropped; resume from 49 only (partial 49 removed → fresh). Last seed.
BASE = 'official=True num_epochs=30 official_keep_checkpoints=False'


def queue_alive():
    for tok in WAIT_TOKENS:
        r = subprocess.run(['pgrep', '-f', tok], capture_output=True, text=True)
        pids = [p for p in r.stdout.split() if p and int(p) != os.getpid()]
        if pids:
            return True
    # extra safety: never start while any training run_base is active
    r = subprocess.run(['pgrep', '-f', 'run_base_experiments.py --set'], capture_output=True, text=True)
    if [p for p in r.stdout.split() if p and int(p) != os.getpid()]:
        return True
    return False


def run(outdir, seed):
    override = f'{BASE} random_seed={seed}'
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
           '--output-base', outdir, '--dataset'] + ALL4 + ['--config-override', override]
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    print(f"[seedsX] START seed{seed} -> {os.path.relpath(outdir, PROJECT)}\n  {override}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[seedsX] DONE seed{seed} rc={rc}", flush=True)


def reviz(outdir, seed):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[seedsX] reviz import FAIL: {e}", flush=True); return
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
                print(f"[seedsX] reviz seed{seed}/{sub} OK (post-warmup)", flush=True)
            except Exception as e:
                print(f"[seedsX] reviz seed{seed}/{sub} FAIL: {e}", flush=True)


def main():
    try:
        open('/tmp/official_seeds_extra_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[seedsX] START {datetime.datetime.now():%Y%m%d_%H%M%S} — waiting for seeds_after "
          f"(seed43/44) + any active run_base to clear...", flush=True)
    waited = 0
    while queue_alive():
        time.sleep(60); waited += 1
        if waited % 60 == 0:
            print(f"[seedsX] still waiting ({waited} min)...", flush=True)
    print(f"[seedsX] clear after {waited} min — settling 20s.", flush=True)
    time.sleep(20)
    for seed in SEEDS:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_{seed}"
        run(outdir, seed)
        reviz(outdir, seed)
    print(f"[seedsX] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
