#!/usr/bin/env python
"""3-SEED-FIRST fill for the paper's label-sparsity + rho-sensitivity analyses (seeds {40,41} only).

[2026-07-19 — user] Queue reorder: complete a 3-seed basis {42,40,41} for EVERY paper table BEFORE any
5-seed extension. The paper-table variants (exclanom/blind/nogrl/nofm/noforce/nostudent/td3sd3) already
have {42,40,41}; the ONLY missing 3-seed pieces are:
  (1) label-sparsity (Table B.2 / Fig 6): unlab10r/25r/50r/75r — paper levels ONLY (user: non-paper
      levels 30r/70r/90r are NOT to be run);
  (2) masking-ratio rho (Fig B.1(b)): maskr005/010/030/050/060/075/090.
Each currently has seed42 only -> run seeds {40,41} here (11 conditions x 2 seeds = 22 runs).
After this launcher finishes, run_official_paper5seed_after (rewired to WAIT on this token, SEEDS
trimmed to [40,41,43,44]) extends variants to 5-seed {40-44}, then run_official_paper5seed_sens_after
(SEEDS [40,41,43,44], paper-scope) fills sparsity/rho seeds {43,44}. blind seed42 stand-in = unlab100@42
(user-approved functional equivalence, applies to 3-seed AND 5-seed). TEP stays single-seed (user).

All runs use the REAL recon_snr ES halt (labels present in every condition here). official=True, 30ep,
keep=False, 4 datasets. Dir tags match existing seed42 tags. Placement: FIRST (run_base guard only —
waits for the in-flight blind@43 run_base to finish, which was intentionally left running).

Usage: python scripts/run_official_sens3seed_after.py
"""
import json, os, sys, subprocess, datetime, time, glob
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
           'WaDi_A1': ['WaDi/A1'], 'WaDi_A2': ['WaDi/A2'], 'PSM': ['PSM']}
BASE = 'official=True num_epochs=30 official_keep_checkpoints=False use_reconsnr_es_halt=True'
WAIT_TOKENS = []  # runs first; run_base-active guard below is the only gate.
SEEDS = [40, 41]  # 3-seed-first: {42(exists), 40, 41}
LABELSPARSITY = [  # paper levels ONLY (Table B.2 / Fig 6); 30r/70r/90r excluded by user directive
    ('unlab10r', 'train_label_mask_frac=0.10 train_label_mask_random=True'),
    ('unlab25r', 'train_label_mask_frac=0.25 train_label_mask_random=True'),
    ('unlab50r', 'train_label_mask_frac=0.50 train_label_mask_random=True'),
    ('unlab75r', 'train_label_mask_frac=0.75 train_label_mask_random=True'),
]
RHO = [  # masking-ratio sensitivity (Fig B.1(b)); default 0.15 = baseline
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
    if _pgrep('run_base_experiments.py --set'):  # never start while a training run_base is active
        return True
    return False


def run(outdir, override, tag, seed):
    ov = (BASE + ' ' + override + f' random_seed={seed}').strip()
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
           '--output-base', outdir, '--dataset'] + ALL4 + ['--config-override', ov]
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    print(f"[s3s] START {tag} seed{seed} -> {os.path.relpath(outdir, PROJECT)}\n  {ov}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[s3s] DONE {tag} seed{seed} rc={rc}", flush=True)


def reviz(outdir, tag, seed):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[s3s] reviz import FAIL: {e}", flush=True); return
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
                print(f"[s3s] reviz {tag}/seed{seed}/{sub} OK", flush=True)
            except Exception as e:
                print(f"[s3s] reviz {tag}/seed{seed}/{sub} FAIL: {e}", flush=True)


def main():
    try:
        open('/tmp/official_sens3seed_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[s3s] START {datetime.datetime.now():%Y%m%d_%H%M%S} — waiting for any active run_base "
          f"(blind@43 in flight) to clear, then 3-seed-first sparsity+rho fill (seeds 40,41).", flush=True)
    waited = 0
    while queue_alive():
        time.sleep(15); waited += 1
    print(f"[s3s] no run_base after {waited*15}s — settling 10s then running.", flush=True)
    time.sleep(10)
    for block_name, conds in BLOCKS:
        print(f"[s3s] === block: {block_name} ({len(conds)} conds x {len(SEEDS)} seeds) ===", flush=True)
        for seed in SEEDS:                       # seed-major within each block
            for tag, cond_ov in conds:
                if already_done(seed, tag):
                    print(f"[s3s] SKIP {tag} seed{seed} (already complete)", flush=True)
                    continue
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_{seed}_{tag}"
                run(outdir, cond_ov, tag, seed)
                reviz(outdir, tag, seed)
                if not already_done(seed, tag):  # [2026-07-24] rc!=0 / cut-off before final files -> retry ONCE
                    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_{seed}_{tag}"
                    print(f"[s3s] RETRY {tag} seed{seed} (incomplete after 1st attempt)", flush=True)
                    run(outdir, cond_ov, tag, seed)
                    reviz(outdir, tag, seed)
                    if not already_done(seed, tag):
                        print(f"[s3s] WARN {tag} seed{seed} STILL INCOMPLETE after retry -- manual attention", flush=True)
    print(f"[s3s] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
