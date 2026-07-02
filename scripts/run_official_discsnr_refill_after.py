#!/usr/bin/env python
"""Re-run the 5 official experiments that PRE-date the train_disc_snr feature (added
2026-06-24) so their training_histories.json gets the train-side train_disc_snr filled.
These were completed before the feature, and train_disc_snr CANNOT be post-computed from
saved data (per-epoch train discrepancy variance not stored; checkpoints deleted; the
best_epoch_train_scores.npz uses a different forward/masking than training-time, so its
recomputed value disagrees with the stored values — verified 2026-06-30).

Faithful re-creation of each original config (verified from best_config.json), seed 42,
30ep, keep=False, 4 datasets. NEW dir suffix `_v2` to avoid colliding with the originals.
(exclanom is NOT here: train_exclude_anomaly_segments=True ⇒ no train anomalies ⇒
train_disc_snr is structurally None even on re-run.)

Runs AFTER the entire current campaign + seed runs finish.

Usage: PYTHONHASHSEED=42 python scripts/run_official_discsnr_refill_after.py
"""
import json, os, sys, subprocess, datetime, time
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
SEED = 42
ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
           'WaDi_A1': ['WaDi/A1'], 'WaDi_A2': ['WaDi/A2'], 'PSM': ['PSM']}
# wait for the WHOLE remaining queue: campaign (3 orchestrators) + seeds
WAIT_TOKENS = ['run_official_resume_full_queue_after_pause',
               'run_official_odoff_featurewise_after_full_queue',
               'run_official_unlab_rankfix_after',
               'run_official_seeds_after']
BASE = 'official=True num_epochs=30 random_seed=42 official_keep_checkpoints=False'
# (dir-tag, extra override) — faithful to each original; `_v2` keeps dirs distinct.
EXPERIMENTS = [
    ('v2',            ''),                                   # baseline (pure CANON_271)
    ('freezeenc_v2',  'freeze_encoder_only=True'),
    ('enc1_v2',       'num_encoder_layers=1'),
    ('dmodel256_v2',  'd_model=256 dim_feedforward=1024'),
    ('sd1_v2',        'num_student_decoder_layers=1'),
]


def queue_alive():
    for tok in WAIT_TOKENS:
        r = subprocess.run(['pgrep', '-f', tok], capture_output=True, text=True)
        if [p for p in r.stdout.split() if p and int(p) != os.getpid()]:
            return True
    return False


def run(outdir, override, tag):
    ov = (BASE + ' ' + override).strip()
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
           '--output-base', outdir, '--dataset'] + ALL4 + ['--config-override', ov]
    env = dict(os.environ, PYTHONHASHSEED=str(SEED))
    print(f"[discsnr] START {tag} -> {os.path.relpath(outdir, PROJECT)}\n  {ov}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[discsnr] DONE {tag} rc={rc}", flush=True)


def reviz(outdir, tag):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[discsnr] reviz import FAIL: {e}", flush=True); return
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
                print(f"[discsnr] reviz {tag}/{sub} OK (post-warmup)", flush=True)
            except Exception as e:
                print(f"[discsnr] reviz {tag}/{sub} FAIL: {e}", flush=True)


def main():
    try:
        open('/tmp/official_discsnr_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[discsnr] START {datetime.datetime.now():%Y%m%d_%H%M%S} — waiting for the whole "
          f"campaign + seeds to finish...", flush=True)
    waited = 0
    while queue_alive():
        time.sleep(60); waited += 1
        if waited % 60 == 0:
            print(f"[discsnr] still waiting ({waited} min)...", flush=True)
    print(f"[discsnr] queue gone after {waited} min — settling 20s.", flush=True)
    time.sleep(20)
    for tag, extra in EXPERIMENTS:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_42_{tag}"
        run(outdir, extra, tag)
        reviz(outdir, tag)
    print(f"[discsnr] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
