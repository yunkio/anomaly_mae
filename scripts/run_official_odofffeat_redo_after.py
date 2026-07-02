#!/usr/bin/env python
"""Re-queue odoff / featwise / featwise_maskr050 — they were DROPPED on 2026-07-01 when
run_official_odoff_featurewise_after_full_queue.py aborted: its completeness check flagged
the master queue as incomplete (missing unlab10/75/25, which were intentionally deferred to
the unlab_rankfix campaign), so it refused to launch the extras and exited.

This is a clean re-queue WITHOUT that completeness gate. It waits for every other live
orchestrator (unlab_rankfix + seeds + discsnr) AND for no run_base to be active, then runs
the 3 experiments last. official=True, 30ep, seed 42, keep=False, 4 datasets. (2026-07-01)
"""
import json, os, sys, subprocess, datetime, time
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
SEED = 42
ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
           'WaDi_A1': ['WaDi/A1'], 'WaDi_A2': ['WaDi/A2'], 'PSM': ['PSM']}
WAIT_TOKENS = ['run_official_unlab_rankfix_after',
               'run_official_seeds_after',
               'run_official_discsnr_refill_after']
BASE = 'official=True num_epochs=30 random_seed=42 official_keep_checkpoints=False'
EXPERIMENTS = [
    ('odoff',             'use_output_discrepancy=False'),
    ('featwise',          'masking_strategy=feature_wise force_mask_anomaly=False'),
    ('featwise_maskr050', 'masking_strategy=feature_wise force_mask_anomaly=False masking_ratio=0.50'),
]


def _pgrep(tok):
    r = subprocess.run(['pgrep', '-f', tok], capture_output=True, text=True)
    return [p for p in r.stdout.split() if p and int(p) != os.getpid()]


def queue_alive():
    for tok in WAIT_TOKENS:
        if _pgrep(tok):
            return True
    # extra safety: never start while any training run_base is active
    if _pgrep('run_base_experiments.py --set'):
        return True
    return False


def run(outdir, override, tag):
    ov = (BASE + ' ' + override).strip()
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
           '--output-base', outdir, '--dataset'] + ALL4 + ['--config-override', ov]
    env = dict(os.environ, PYTHONHASHSEED=str(SEED))
    print(f"[odofffeat] START {tag} -> {os.path.relpath(outdir, PROJECT)}\n  {ov}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[odofffeat] DONE {tag} rc={rc}", flush=True)


def reviz(outdir, tag):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[odofffeat] reviz import FAIL: {e}", flush=True); return
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
                print(f"[odofffeat] reviz {tag}/{sub} OK (post-warmup)", flush=True)
            except Exception as e:
                print(f"[odofffeat] reviz {tag}/{sub} FAIL: {e}", flush=True)


def main():
    try:
        open('/tmp/official_odofffeat_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[odofffeat] START {datetime.datetime.now():%Y%m%d_%H%M%S} — waiting for "
          f"unlab_rankfix + seeds + discsnr (+ no active run_base) to finish...", flush=True)
    waited = 0
    while queue_alive():
        time.sleep(60); waited += 1
        if waited % 60 == 0:
            print(f"[odofffeat] still waiting ({waited} min)...", flush=True)
    print(f"[odofffeat] queue idle after {waited} min — settling 20s.", flush=True)
    time.sleep(20)
    for tag, extra in EXPERIMENTS:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_42_{tag}"
        run(outdir, extra, tag)
        reviz(outdir, tag)
    print(f"[odofffeat] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
