#!/usr/bin/env python
"""After the resume chain (run_official_resume_chain.py: 30ep-resume -> 10ep -> freeze)
finishes, run 4 architecture-ablation official runs (all official=True, 30ep, seed=42,
keep=False, 4 datasets), each overriding ONE/TWO geometry params on top of CANON_271:

  (3) enc1       : num_encoder_layers       4 -> 1
  (4) dmodel256  : d_model 512->256, dim_feedforward 2048->1024 (4x kept; NOT auto in
                   official path because CANON_271 pins dim_feedforward, so set explicitly)
  (5) sd1        : num_student_decoder_layers 2 -> 1
  (6) enc1sd1    : num_encoder_layers 4->1 AND num_student_decoder_layers 2->1

Replaces the cancelled seed-41 run. Each run -> post-warmup npz-viz.
Verified code path: run_base parses --config-override (int auto-cast) -> official branch
overrides=dict(CANON_271), user-explicit keys WIN -> make_config -> apply_official_overrides
(no geometry) -> model.py builds layers from num_encoder_layers / num_student_decoder_layers
/ d_model. (2026-06-23)

Usage: PYTHONHASHSEED=42 python scripts/run_official_ablation_after.py
"""
import json, os, sys, subprocess, datetime, time
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
SEED = 42
ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
           'WaDi_A1': ['WaDi/A1'], 'WaDi_A2': ['WaDi/A2'], 'PSM': ['PSM']}
WAIT_TOKEN = 'run_official_resume_chain'
BASE = 'official=True num_epochs=30 random_seed=42 official_keep_checkpoints=False'
# (tag, extra geometry override appended to BASE)
EXPERIMENTS = [
    ('enc1',      'num_encoder_layers=1'),
    ('dmodel256', 'd_model=256 dim_feedforward=1024'),
    ('sd1',       'num_student_decoder_layers=1'),
    ('enc1sd1',   'num_encoder_layers=1 num_student_decoder_layers=1'),
    # (7) window 100 + patch 5: seq_length 500->100, patch_size 10->5. num_patches MUST be
    #     set to seq_length//patch_size = 20 (CANON_271 pins it at 50; make_config's
    #     consistency check raises ValueError unless seq_length == patch_size*num_patches).
    ('w100p5',    'seq_length=100 patch_size=5 num_patches=20'),
]


def chain_alive():
    r = subprocess.run(['pgrep', '-f', WAIT_TOKEN], capture_output=True, text=True)
    pids = [p for p in r.stdout.split() if p and int(p) != os.getpid()]
    return bool(pids)


def run(outdir, override, label):
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
           '--output-base', outdir, '--dataset'] + ALL4 + ['--config-override', override]
    env = dict(os.environ, PYTHONHASHSEED=str(SEED))
    print(f"[ablat] START {label} -> {os.path.relpath(outdir, PROJECT)}\n  {override}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[ablat] DONE {label} rc={rc}", flush=True)


def reviz(outdir, label):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[ablat] reviz import FAIL: {e}", flush=True); return
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
                print(f"[ablat] reviz {label}/{sub} OK (post-warmup)", flush=True)
            except Exception as e:
                print(f"[ablat] reviz {label}/{sub} FAIL: {e}", flush=True)


def main():
    try:
        open('/tmp/official_ablation_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[ablat] START {datetime.datetime.now():%Y%m%d_%H%M%S} — waiting for resume chain "
          f"(10ep->freeze) to finish...", flush=True)
    waited = 0
    while chain_alive():
        time.sleep(60); waited += 1
        if waited % 60 == 0:
            print(f"[ablat] still waiting ({waited} min)...", flush=True)
    print(f"[ablat] chain gone after {waited} min — settling 20s.", flush=True)
    time.sleep(20)
    for tag, extra in EXPERIMENTS:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_42_{tag}"
        run(outdir, f"{BASE} {extra}", tag)
        reviz(outdir, tag)
    print(f"[ablat] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
