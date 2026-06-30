#!/usr/bin/env python
"""After the resume queue (run_official_resume_remaining.py: sd1/enc1sd1/w100p5/nogrl) finishes,
run ONE more official ablation as the 5th: no-fm = 271 config with feature-matching OFF only.
use_feature_matching=False -> loss.py sets fm_loss=0, discrepancy_loss = normal+anomaly (no FM);
GRL/SCAD/discrepancy unchanged (= "271과 같은데 fm만 뺀" 구성). keep=False, seed=42, 4 datasets.
Logs train_disc_snr (new code). Each run -> post-warmup npz-viz.

Usage: PYTHONHASHSEED=42 python scripts/run_official_nofm_after.py
"""
import json, os, sys, subprocess, datetime, time
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
SEED = 42
ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
           'WaDi_A1': ['WaDi/A1'], 'WaDi_A2': ['WaDi/A2'], 'PSM': ['PSM']}
WAIT_TOKEN = 'run_official_resume_remaining'
OVERRIDE = ('official=True num_epochs=30 random_seed=42 official_keep_checkpoints=False '
            'use_feature_matching=False')
TAG = 'nofm'


def queue_alive():
    r = subprocess.run(['pgrep', '-f', WAIT_TOKEN], capture_output=True, text=True)
    return bool([p for p in r.stdout.split() if p and int(p) != os.getpid()])


def run(outdir):
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
           '--output-base', outdir, '--dataset'] + ALL4 + ['--config-override', OVERRIDE]
    env = dict(os.environ, PYTHONHASHSEED=str(SEED))
    print(f"[nofm] START {TAG} -> {os.path.relpath(outdir, PROJECT)}\n  {OVERRIDE}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[nofm] DONE {TAG} rc={rc}", flush=True)


def reviz(outdir):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[nofm] reviz import FAIL: {e}", flush=True); return
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
                print(f"[nofm] reviz {sub} OK (post-warmup)", flush=True)
            except Exception as e:
                print(f"[nofm] reviz {sub} FAIL: {e}", flush=True)


def main():
    try:
        open('/tmp/official_nofm_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[nofm] START {datetime.datetime.now():%Y%m%d_%H%M%S} — waiting for resume queue "
          f"(sd1/enc1sd1/w100p5/nogrl) to finish...", flush=True)
    waited = 0
    while queue_alive():
        time.sleep(60); waited += 1
        if waited % 60 == 0:
            print(f"[nofm] still waiting ({waited} min)...", flush=True)
    print(f"[nofm] resume queue gone after {waited} min — settling 20s.", flush=True)
    time.sleep(20)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_42_{TAG}"
    run(outdir)
    reviz(outdir)
    print(f"[nofm] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
