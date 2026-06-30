#!/usr/bin/env python
"""After the architecture-ablation queue (run_official_ablation_after.py:
enc1/dmodel256/sd1/enc1sd1/w100p5) finishes, run ONE more official 30ep ablation with the
GRL path OFF — and the OD anomaly_loss (margin push) that would otherwise re-activate also
OFF, so GRL is purely IGNORED (no compensating mechanism). keep=False, seed=42, 4 datasets.

GRL-off config (verified against codebase 2026-06-23):
  use_grl=False           — GRL adversarial classifier path off (trainer gates every GRL
                            branch on use_grl; noadv = use_grl & use_scad both False is valid,
                            trainer.py only forbids BOTH True).
  anomaly_loss_weight=0.0 — when use_grl=False, loss.py's disable_anomaly_loss
                            =((use_grl & grl_disable_anomaly_loss) or use_scad) becomes False,
                            so the margin-based OD anomaly_loss (maximize direction) revives.
                            loss.py's own guidance: "to ablate it there, set anomaly_loss_weight
                            =0.0". This zeroes the margin/anomaly push (margin_type irrelevant
                            at weight 0).
Everything else = CANON_271: normal_loss (normal distillation) + FM stay, use_scad=False,
use_discriminator=False, loss_balance_mode=adaptive_lambda_legacy (does not require GRL).

Usage: PYTHONHASHSEED=42 python scripts/run_official_nogrl_after.py
"""
import json, os, sys, subprocess, datetime, time
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
SEED = 42
ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
           'WaDi_A1': ['WaDi/A1'], 'WaDi_A2': ['WaDi/A2'], 'PSM': ['PSM']}
WAIT_TOKEN = 'run_official_ablation_after'
OVERRIDE = ('official=True num_epochs=30 random_seed=42 official_keep_checkpoints=False '
            'use_grl=False anomaly_loss_weight=0.0')
TAG = 'nogrl'


def queue_alive():
    r = subprocess.run(['pgrep', '-f', WAIT_TOKEN], capture_output=True, text=True)
    pids = [p for p in r.stdout.split() if p and int(p) != os.getpid()]
    return bool(pids)


def run(outdir):
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
           '--output-base', outdir, '--dataset'] + ALL4 + ['--config-override', OVERRIDE]
    env = dict(os.environ, PYTHONHASHSEED=str(SEED))
    print(f"[nogrl] START {TAG} -> {os.path.relpath(outdir, PROJECT)}\n  {OVERRIDE}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[nogrl] DONE {TAG} rc={rc}", flush=True)


def reviz(outdir):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[nogrl] reviz import FAIL: {e}", flush=True); return
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
                print(f"[nogrl] reviz {sub} OK (post-warmup)", flush=True)
            except Exception as e:
                print(f"[nogrl] reviz {sub} FAIL: {e}", flush=True)


def main():
    try:
        open('/tmp/official_nogrl_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[nogrl] START {datetime.datetime.now():%Y%m%d_%H%M%S} — waiting for the ablation "
          f"queue (enc1..w100p5) to finish...", flush=True)
    waited = 0
    while queue_alive():
        time.sleep(60); waited += 1
        if waited % 60 == 0:
            print(f"[nogrl] still waiting ({waited} min)...", flush=True)
    print(f"[nogrl] ablation queue gone after {waited} min — settling 20s.", flush=True)
    time.sleep(20)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_42_{TAG}"
    run(outdir)
    reviz(outdir)
    print(f"[nogrl] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
