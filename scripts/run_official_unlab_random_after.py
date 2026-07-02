#!/usr/bin/env python
"""Queue-JUMP re-run of the unlab sweep with RANDOM anomaly masking (train_label_mask_random=True)
instead of the chronologically-last (rank) masking. 5 experiments: frac 0.10/0.25/0.50/0.75/1.00.
official=True, 30ep, seed 42, keep=False, 4 datasets. Cuts in line BEFORE seeds/discsnr/odofffeat.

Runs as soon as no run_base is active (the interrupted seed40 was stopped for this jump). Dir tag
`unlab<X>r`. NOTE: unlab100r ≡ unlab100 (frac=1.0 ⇒ all anomalies masked regardless of random).

Usage: PYTHONHASHSEED=42 python scripts/run_official_unlab_random_after.py
"""
import json, os, sys, subprocess, datetime, time
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
SEED = 42
ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
           'WaDi_A1': ['WaDi/A1'], 'WaDi_A2': ['WaDi/A2'], 'PSM': ['PSM']}
BASE = 'official=True num_epochs=30 random_seed=42 official_keep_checkpoints=False'
EXPERIMENTS = [
    ('unlab10r',  'train_label_mask_frac=0.10 train_label_mask_random=True'),
    ('unlab25r',  'train_label_mask_frac=0.25 train_label_mask_random=True'),
    ('unlab50r',  'train_label_mask_frac=0.50 train_label_mask_random=True'),
    ('unlab75r',  'train_label_mask_frac=0.75 train_label_mask_random=True'),
    # unlab100r dropped 2026-07-02 (user): frac=1.0 masks ALL anomalies ⇒ identical to unlab100.
]


def run_base_active():
    r = subprocess.run(['pgrep', '-f', 'run_base_experiments.py --set'], capture_output=True, text=True)
    return bool([p for p in r.stdout.split() if p and int(p) != os.getpid()])


def run(outdir, override, tag):
    ov = (BASE + ' ' + override).strip()
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
           '--output-base', outdir, '--dataset'] + ALL4 + ['--config-override', ov]
    env = dict(os.environ, PYTHONHASHSEED=str(SEED))
    print(f"[unlabR] START {tag} -> {os.path.relpath(outdir, PROJECT)}\n  {ov}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[unlabR] DONE {tag} rc={rc}", flush=True)


def reviz(outdir, tag):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[unlabR] reviz import FAIL: {e}", flush=True); return
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
                print(f"[unlabR] reviz {tag}/{sub} OK (post-warmup)", flush=True)
            except Exception as e:
                print(f"[unlabR] reviz {tag}/{sub} FAIL: {e}", flush=True)


def main():
    try:
        open('/tmp/official_unlabR_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[unlabR] START {datetime.datetime.now():%Y%m%d_%H%M%S} — queue-jump; waiting for any "
          f"active run_base to clear...", flush=True)
    waited = 0
    while run_base_active():
        time.sleep(15); waited += 1
    print(f"[unlabR] no run_base after {waited*15}s — settling 10s then running RANDOM unlab sweep.", flush=True)
    time.sleep(10)
    for tag, extra in EXPERIMENTS:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_42_{tag}"
        run(outdir, extra, tag)
        reviz(outdir, tag)
    print(f"[unlabR] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
