#!/usr/bin/env python
"""RESUME the official ablation queue after the 2026-06-23 pause for disc_snr re-inference.
Already done: enc1, dmodel256 (+ baseline30/10ep/freezeenc earlier). sd1 was killed ~9min in
(minimal) -> re-run from scratch. Remaining, in order: sd1, enc1sd1, w100p5, nogrl.
All official=True, 30ep, seed=42, keep=False, 4 datasets. Each run -> post-warmup npz-viz.

Waits for any lingering disc_snr re-inference process to exit (GPU free) before starting.
Usage: PYTHONHASHSEED=42 python scripts/run_official_resume_remaining.py
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
    ('sd1',      'num_student_decoder_layers=1'),
    ('enc1sd1',  'num_encoder_layers=1 num_student_decoder_layers=1'),
    ('w100p5',   'seq_length=100 patch_size=5 num_patches=20'),
    ('nogrl',    'use_grl=False anomaly_loss_weight=0.0'),
]


def gpu_busy():
    r = subprocess.run(['pgrep', '-f', 'disc_snr_reinfer'], capture_output=True, text=True)
    return bool([p for p in r.stdout.split() if p and int(p) != os.getpid()])


def run(outdir, override, label):
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
           '--output-base', outdir, '--dataset'] + ALL4 + ['--config-override', override]
    env = dict(os.environ, PYTHONHASHSEED=str(SEED))
    print(f"[resume] START {label} -> {os.path.relpath(outdir, PROJECT)}\n  {override}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[resume] DONE {label} rc={rc}", flush=True)


def reviz(outdir, label):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[resume] reviz import FAIL: {e}", flush=True); return
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
                print(f"[resume] reviz {label}/{sub} OK (post-warmup)", flush=True)
            except Exception as e:
                print(f"[resume] reviz {label}/{sub} FAIL: {e}", flush=True)


def main():
    try:
        open('/tmp/official_resume_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[resume] START {datetime.datetime.now():%Y%m%d_%H%M%S} — waiting for disc_snr re-inference to exit (GPU free)...", flush=True)
    waited = 0
    while gpu_busy():
        time.sleep(15); waited += 1
        if waited % 8 == 0:
            print(f"[resume] still waiting for GPU ({waited*15}s)...", flush=True)
    print(f"[resume] GPU free — starting queue. settle 10s.", flush=True)
    time.sleep(10)
    for tag, extra in EXPERIMENTS:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_42_{tag}"
        run(outdir, f"{BASE} {extra}", tag)
        reviz(outdir, tag)
    print(f"[resume] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
