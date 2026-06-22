#!/usr/bin/env python
"""Resume the official chain after the manual stop (2026-06-23):
  1. 30ep — EXISTING dir, ONLY incomplete datasets: WaDi_A2 (resume from ep24 ckpt) + PSM. keep=False.
  2. 10ep — new dir, all 4 datasets. keep=False, warmup 5.
  3. freeze_encoder_only — new dir, all 4. keep=True.
After EACH run completes, re-generate the post-warmup npz-viz (anomaly_threshold +
anomaly_threshold_test_event) for that run's datasets. (run_base already emits post-warmup
viz via the new code; this is an explicit, idempotent guarantee per the user's request.)

Usage: PYTHONHASHSEED=42 python scripts/run_official_resume_chain.py
"""
import json, os, sys, subprocess, datetime
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
SEED = 42
ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
EXISTING = f'{PROJECT}/results/experiments/official/271_20260622_164359_30ep_42'
SUBDIRS = {'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
           'WaDi_A1': ['WaDi/A1'], 'WaDi_A2': ['WaDi/A2'], 'PSM': ['PSM']}


def run(outdir, datasets, override, label):
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
           '--output-base', outdir, '--dataset'] + datasets + ['--config-override', override]
    env = dict(os.environ, PYTHONHASHSEED=str(SEED))
    print(f"[chain] START {label} -> {os.path.relpath(outdir, PROJECT)} | datasets={datasets}", flush=True)
    print(f"[chain]   override: {override}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[chain] DONE {label} rc={rc}", flush=True)


def reviz(outdir, datasets, label):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[chain] reviz import FAIL: {e}", flush=True); return
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
                print(f"[chain] reviz {label}/{sub} OK (post-warmup)", flush=True)
            except Exception as e:
                print(f"[chain] reviz {label}/{sub} FAIL: {e}", flush=True)


def ts():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def main():
    try:
        open('/tmp/official_chain_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[chain] resume chain START {ts()}", flush=True)

    # 1. 30ep — resume EXISTING dir, only incomplete datasets (WaDi_A2 resume + PSM fresh). keep=False.
    run(EXISTING, ['WaDi_A2', 'PSM'],
        'official=True num_epochs=30 random_seed=42 official_keep_checkpoints=False', '30ep-resume')
    reviz(EXISTING, ['WaDi_A2', 'PSM'], '30ep')

    # 2. 10ep — new dir, all 4 datasets. keep=False, warmup 5.
    d10 = f"{PROJECT}/results/experiments/official/271_{ts()}_10ep_42"
    run(d10, ALL4,
        'official=True num_epochs=10 random_seed=42 official_keep_checkpoints=False teacher_only_warmup_epochs=5', '10ep')
    reviz(d10, ALL4, '10ep')

    # 3. freeze_encoder_only — new dir, all 4. keep=True.
    df = f"{PROJECT}/results/experiments/official/271_{ts()}_30ep_42_freezeenc"
    run(df, ALL4,
        'official=True num_epochs=30 random_seed=42 official_keep_checkpoints=True freeze_encoder_only=True', 'freeze')
    reviz(df, ALL4, 'freeze')

    print(f"[chain] ALL DONE {ts()}", flush=True)


if __name__ == '__main__':
    main()
