#!/usr/bin/env python
"""Group-random EXCLUDE sweep — companion to the group-random MASK sweep (unlab10r/25r/50r/75r).

For each frac in {0.10, 0.25, 0.50, 0.75}, take the SAME seeded 100-ts anomaly-group selection
used by the mask sweep, but instead of unlabeling the selected anomaly timepoints, SPLICE THEM OUT
of the training signal (train_label_mask_exclude=True). Non-selected anomaly groups keep TRUE
labels. This isolates "keep the hidden anomalies as unlabeled data" (mask) vs "drop them entirely"
(exclude) on the exact same timesteps.

frac=1.00 (all groups) ≡ removing ALL anomalies ≡ the existing `exclanom`
(train_exclude_anomaly_segments=True, already completed 271_20260629_071207_30ep_42_exclanom),
so it is intentionally NOT re-run here.

official=True, 30ep, seed 42, keep=False, 4 datasets. Dir tag `excl<X>r`.

Placement [2026-07-05 reorder — user]: runs FIRST. The seed campaign was stopped mid-seed43 and
re-ordered to exclude-first; this launcher now waits only for any active run_base to clear (no token
wait), then runs. run_official_seeds_after (SEEDS=[43,44]) waits on THIS launcher's token, so the
resumed order is: exclR → seed43 → seed44 → discsnr_refill → odofffeat_redo.

Usage: PYTHONHASHSEED=42 python scripts/run_official_exclude_grouprandom_after.py
"""
import json, os, sys, subprocess, datetime, time
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
SEED = 42
ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
           'WaDi_A1': ['WaDi/A1'], 'WaDi_A2': ['WaDi/A2'], 'PSM': ['PSM']}
BASE = 'official=True num_epochs=30 random_seed=42 official_keep_checkpoints=False'
# [2026-07-05 reorder — user] Originally ran first; now the dense sweep (unlab/excl 30/70/90%)
# runs ahead of it, so wait on that launcher's token. Resumed order:
#   dense[unlab30/70/90r, excl30/70/90r] -> exclR[excl25r,50r,75r] -> seeds -> discsnr -> odofffeat.
WAIT_TOKENS = ['run_official_unlabexcl_dense_after']
EXPERIMENTS = [
    # [2026-07-05 pause/resume] excl10r COMPLETED (271_20260705_024006_..._excl10r) — dropped so a
    # resume does not redo it. excl25r was interrupted mid-run (partial dir removed) → resume redoes
    # it fresh from here. Restore the excl10r line only if a full re-run of all four is ever wanted.
    # ('excl10r', 'train_label_mask_frac=0.10 train_label_mask_random=True train_label_mask_exclude=True'),
    ('excl25r', 'train_label_mask_frac=0.25 train_label_mask_random=True train_label_mask_exclude=True'),
    ('excl50r', 'train_label_mask_frac=0.50 train_label_mask_random=True train_label_mask_exclude=True'),
    ('excl75r', 'train_label_mask_frac=0.75 train_label_mask_random=True train_label_mask_exclude=True'),
    # frac=1.00 dropped: removes ALL anomalies ≡ exclanom (already completed).
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
    print(f"[exclR] START {tag} -> {os.path.relpath(outdir, PROJECT)}\n  {ov}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[exclR] DONE {tag} rc={rc}", flush=True)


def reviz(outdir, tag):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[exclR] reviz import FAIL: {e}", flush=True); return
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
                print(f"[exclR] reviz {tag}/{sub} OK (post-warmup)", flush=True)
            except Exception as e:
                print(f"[exclR] reviz {tag}/{sub} FAIL: {e}", flush=True)


def main():
    try:
        open('/tmp/official_exclR_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[exclR] START {datetime.datetime.now():%Y%m%d_%H%M%S} — waiting for the running chain "
          f"(seeds -> discsnr -> odofffeat) + any active run_base to clear...", flush=True)
    waited = 0
    while queue_alive():
        time.sleep(60); waited += 1
        if waited % 60 == 0:
            print(f"[exclR] still waiting ({waited} min)...", flush=True)
    print(f"[exclR] chain clear after {waited} min — settling 20s then running EXCLUDE sweep.", flush=True)
    time.sleep(20)
    for tag, extra in EXPERIMENTS:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_42_{tag}"
        run(outdir, extra, tag)
        reviz(outdir, tag)
    print(f"[exclR] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
