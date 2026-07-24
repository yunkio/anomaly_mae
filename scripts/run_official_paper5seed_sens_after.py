#!/usr/bin/env python
"""5-seed fill for the paper's remaining LASAD analyses (option C, appended LAST):
  (1) Label-sparsity sweep (fig:labelsparsity / tab:labelsparsity_grid) — group-random (100-ts group)
      anomaly-label masking at fractions {10,25,30,50,70,75,90}% (0%=full baseline, 100%=label-blind
      are covered elsewhere).
  (2) Parameter sensitivity — masking ratio rho (fig:param_sensitivity b) — rho in
      {0.05,0.10,0.30,0.50,0.60,0.75,0.90} (default 0.15 = baseline).
The score-weight w sensitivity (fig:param_sensitivity a) needs NO training — it is recomputed post-hoc
from each seed's saved epoch_scores NPZ (recon/disc/official_score) — so it is not in this launcher.

[2026-07-11 — user: option C, run LAST] Seeds {40,41,43,44,45,46} (seed42 already exists). Two blocks:
label-sparsity first (all seeds, seed-major), then rho (all seeds, seed-major). All runs use the REAL
recon_snr ES halt (labels present in every condition → recon_snr defined → halt fires). official=True,
30ep, keep=False, 4 datasets. Dir tag matches the existing seed42 tags (unlab<X>r / maskr0<XX>).

Placement: waits for the chain sens3seed -> paper5seed to finish (dense/exclR/discsnr/odofffeat were
REMOVED — non-paper). Usage: python scripts/run_official_paper5seed_sens_after.py
"""
import json, os, sys, subprocess, datetime, time, glob
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
           'WaDi_A1': ['WaDi/A1'], 'WaDi_A2': ['WaDi/A2'], 'PSM': ['PSM']}
BASE = 'official=True num_epochs=30 official_keep_checkpoints=False use_reconsnr_es_halt=True'
# [2026-07-19 reorder — user] New chain: sens3seed{40,41} -> paper5seed{40,41,43,44} -> THIS launcher
# (sparsity/rho seeds {43,44}; 40,41 done by sens3seed and skipped via already_done). The dense/exclR/
# discsnr/odofffeat launchers were REMOVED from the queue (non-paper experiments — user directive).
WAIT_TOKENS = ['run_official_sens3seed_after',
               'run_official_paper5seed_after']
# [2026-07-19 — user] Paper 5-seed set = consecutive {40,41,42,43,44}; 45/46 removed.
SEEDS = [40, 41, 43, 44]
LABELSPARSITY = [  # paper levels ONLY (Table B.2 / Fig 6) — 30r/70r/90r removed (non-paper, user directive)
    ('unlab10r', 'train_label_mask_frac=0.10 train_label_mask_random=True'),
    ('unlab25r', 'train_label_mask_frac=0.25 train_label_mask_random=True'),
    ('unlab50r', 'train_label_mask_frac=0.50 train_label_mask_random=True'),
    ('unlab75r', 'train_label_mask_frac=0.75 train_label_mask_random=True'),
]
RHO = [  # masking-ratio sensitivity (fig:param_sensitivity b); default 0.15 = baseline
    ('maskr005', 'masking_ratio=0.05'),
    ('maskr010', 'masking_ratio=0.10'),
    ('maskr030', 'masking_ratio=0.30'),
    ('maskr050', 'masking_ratio=0.50'),
    ('maskr060', 'masking_ratio=0.60'),
    ('maskr075', 'masking_ratio=0.75'),
    ('maskr090', 'masking_ratio=0.90'),
]
BLOCKS = [('labelsparsity', LABELSPARSITY), ('rho', RHO)]


def already_done(seed, tag):
    """Resume-safe skip: True if a completed run dir for (seed, tag) exists (all 4 datasets finalized)."""
    cells = ['PSM', 'SWaT/A1A2_full', 'WaDi/A1', 'WaDi/A2']
    for d in glob.glob(f"{PROJECT}/results/experiments/official/271_*_30ep_{seed}_{tag}"):
        if all(os.path.exists(os.path.join(d, c, _f)) for c in cells
               for _f in ('epoch_metrics.json', 'best_config.json', 'training_histories.json')):
            return True
    return False


def _pgrep(tok):
    r = subprocess.run(['pgrep', '-f', tok], capture_output=True, text=True)
    return [p for p in r.stdout.split() if p and int(p) != os.getpid()]


def queue_alive():
    for tok in WAIT_TOKENS:
        if _pgrep(tok):
            return True
    if _pgrep('run_base_experiments.py --set'):
        return True
    return False


def run(outdir, override, tag, seed):
    ov = (BASE + ' ' + override + f' random_seed={seed}').strip()
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
           '--output-base', outdir, '--dataset'] + ALL4 + ['--config-override', ov]
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    print(f"[p5sens] START {tag} seed{seed} -> {os.path.relpath(outdir, PROJECT)}\n  {ov}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[p5sens] DONE {tag} seed{seed} rc={rc}", flush=True)


def reviz(outdir, tag, seed):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[p5sens] reviz import FAIL: {e}", flush=True); return
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
                print(f"[p5sens] reviz {tag}/seed{seed}/{sub} OK", flush=True)
            except Exception as e:
                print(f"[p5sens] reviz {tag}/seed{seed}/{sub} FAIL: {e}", flush=True)


def main():
    try:
        open('/tmp/official_paper5seed_sens_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[p5sens] START {datetime.datetime.now():%Y%m%d_%H%M%S} — waiting for the ENTIRE current "
          f"chain to finish (sens3seed -> paper5seed), then fills sparsity/rho seeds 43,44...", flush=True)
    waited = 0
    while queue_alive():
        time.sleep(60); waited += 1
        if waited % 60 == 0:
            print(f"[p5sens] still waiting ({waited} min)...", flush=True)
    print(f"[p5sens] campaign gone after {waited} min — settling 20s.", flush=True)
    time.sleep(20)
    for block_name, conds in BLOCKS:
        print(f"[p5sens] === block: {block_name} ({len(conds)} conditions x {len(SEEDS)} seeds) ===", flush=True)
        for seed in SEEDS:                       # seed-major within each block
            for tag, cond_ov in conds:
                if already_done(seed, tag):      # resume-safe: skip completed (seed, condition)
                    print(f"[p5sens] SKIP {tag} seed{seed} (already complete)", flush=True)
                    continue
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_{seed}_{tag}"
                run(outdir, cond_ov, tag, seed)
                reviz(outdir, tag, seed)
                if not already_done(seed, tag):  # [2026-07-24] rc!=0 / cut-off before final files -> retry ONCE
                    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_{seed}_{tag}"
                    print(f"[p5sens] RETRY {tag} seed{seed} (incomplete after 1st attempt)", flush=True)
                    run(outdir, cond_ov, tag, seed)
                    reviz(outdir, tag, seed)
                    if not already_done(seed, tag):
                        print(f"[p5sens] WARN {tag} seed{seed} STILL INCOMPLETE after retry -- manual attention", flush=True)
    print(f"[p5sens] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
