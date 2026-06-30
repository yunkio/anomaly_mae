#!/usr/bin/env python
"""After the no-fm run (run_official_nofm_after.py) finishes, run TWO more official ablations:
  6th: unlab100 = 271 with ALL train labels masked (train_label_mask_frac=1.0 → fully unlabeled;
       GRL/anomaly supervision inactive — no anomaly labels).
  7th: unlab50  = back 50% of train data unlabeled (train_label_mask_frac=0.5).
Same as 271 otherwise. keep=False, seed=42, 4 datasets. Test labels + best_epoch_train_scores
keep TRUE labels (masking is .copy()-local to the training train_dataset). Each run -> post-warmup viz.
Logs train_disc_snr (new code); for frac=1.0 it is None (no anomaly labels → SNR undefined).

Usage: PYTHONHASHSEED=42 python scripts/run_official_unlabeled_after.py
"""
import json, os, sys, subprocess, datetime, time
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
SEED = 42
ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
           'WaDi_A1': ['WaDi/A1'], 'WaDi_A2': ['WaDi/A2'], 'PSM': ['PSM']}
WAIT_TOKEN = 'run_official_nofm_after'
BASE = 'official=True num_epochs=30 random_seed=42 official_keep_checkpoints=False'
EXPERIMENTS = [
    ('unlab100', 'train_label_mask_frac=1.0'),
    ('unlab50',  'train_label_mask_frac=0.5'),
]


def queue_alive():
    r = subprocess.run(['pgrep', '-f', WAIT_TOKEN], capture_output=True, text=True)
    return bool([p for p in r.stdout.split() if p and int(p) != os.getpid()])


def run(outdir, override, label):
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
           '--output-base', outdir, '--dataset'] + ALL4 + ['--config-override', override]
    env = dict(os.environ, PYTHONHASHSEED=str(SEED))
    print(f"[unlab] START {label} -> {os.path.relpath(outdir, PROJECT)}\n  {override}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[unlab] DONE {label} rc={rc}", flush=True)


def reviz(outdir, label):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[unlab] reviz import FAIL: {e}", flush=True); return
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
                print(f"[unlab] reviz {label}/{sub} OK (post-warmup)", flush=True)
            except Exception as e:
                print(f"[unlab] reviz {label}/{sub} FAIL: {e}", flush=True)


def main():
    try:
        open('/tmp/official_unlabeled_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[unlab] START {datetime.datetime.now():%Y%m%d_%H%M%S} — waiting for no-fm run to finish...", flush=True)
    waited = 0
    while queue_alive():
        time.sleep(60); waited += 1
        if waited % 60 == 0:
            print(f"[unlab] still waiting ({waited} min)...", flush=True)
    print(f"[unlab] no-fm gone after {waited} min — settling 20s.", flush=True)
    time.sleep(20)
    for tag, extra in EXPERIMENTS:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_42_{tag}"
        run(outdir, f"{BASE} {extra}", tag)
        reviz(outdir, tag)
    print(f"[unlab] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
