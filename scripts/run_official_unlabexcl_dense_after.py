#!/usr/bin/env python
"""Densify the group-random sweeps at frac 0.30 / 0.70 / 0.90 — for BOTH the MASK variant
(unlab*r: train_label_mask_random=True) and the EXCLUDE variant (excl*r: + train_label_mask_exclude=True).

User order (2026-07-05): run unlab30r/70r/90r FIRST, then excl30r/70r/90r. These run AHEAD of the
resumed exclR remainder (excl25r/50r/75r) and the rest of the chain.

official=True, 30ep, seed 42, keep=False, 4 datasets. Dir tags unlab<X>r / excl<X>r.

Placement: runs FIRST (WAIT_TOKENS empty; only the run_base-active guard gates it). run_official_
exclude_grouprandom_after (exclR) now waits on THIS launcher's token, so the resumed order is:
  [unlab30r,70r,90r, excl30r,70r,90r]  ->  exclR[excl25r,50r,75r]  ->  seed43,44  ->  discsnr  ->  odofffeat

Usage: PYTHONHASHSEED=42 python scripts/run_official_unlabexcl_dense_after.py
"""
import json, os, sys, subprocess, datetime, time
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
SEED = 42
ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
           'WaDi_A1': ['WaDi/A1'], 'WaDi_A2': ['WaDi/A2'], 'PSM': ['PSM']}
BASE = 'official=True num_epochs=30 random_seed=42 official_keep_checkpoints=False'
WAIT_TOKENS = ['run_official_paper5seed_after']  # [2026-07-11] paper 5-seed fill preempts; dense resumes after.
EXPERIMENTS = [
    # [2026-07-11 preempt/resume] unlab30/70/90r + excl30r + excl70r ALL COMPLETED. excl90r was
    # interrupted (partial removed) when the paper 5-seed fill preempted → resume redoes excl90r fresh.
    # ('unlab30r', ...), ('unlab70r', ...), ('unlab90r', ...) — done.
    # ('excl30r',  '... exclude=True'),  # done
    # ('excl70r',  '... exclude=True'),  # done
    ('excl90r',  'train_label_mask_frac=0.90 train_label_mask_random=True train_label_mask_exclude=True'),
]


def _pgrep(tok):
    r = subprocess.run(['pgrep', '-f', tok], capture_output=True, text=True)
    return [p for p in r.stdout.split() if p and int(p) != os.getpid()]


def queue_alive():
    for tok in WAIT_TOKENS:
        if _pgrep(tok):
            return True
    if _pgrep('run_base_experiments.py --set'):  # never start while a training run_base is active
        return True
    return False


def run(outdir, override, tag):
    ov = (BASE + ' ' + override).strip()
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
           '--output-base', outdir, '--dataset'] + ALL4 + ['--config-override', ov]
    env = dict(os.environ, PYTHONHASHSEED=str(SEED))
    print(f"[dense] START {tag} -> {os.path.relpath(outdir, PROJECT)}\n  {ov}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[dense] DONE {tag} rc={rc}", flush=True)


def reviz(outdir, tag):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[dense] reviz import FAIL: {e}", flush=True); return
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
                print(f"[dense] reviz {tag}/{sub} OK (post-warmup)", flush=True)
            except Exception as e:
                print(f"[dense] reviz {tag}/{sub} FAIL: {e}", flush=True)


def main():
    try:
        open('/tmp/official_dense_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[dense] START {datetime.datetime.now():%Y%m%d_%H%M%S} — waiting for any active run_base "
          f"to clear, then running dense group-random sweep (30/70/90% mask + exclude).", flush=True)
    waited = 0
    while queue_alive():
        time.sleep(15); waited += 1
    print(f"[dense] no run_base after {waited*15}s — settling 10s then running.", flush=True)
    time.sleep(10)
    for tag, extra in EXPERIMENTS:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_42_{tag}"
        run(outdir, extra, tag)
        reviz(outdir, tag)
    print(f"[dense] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
