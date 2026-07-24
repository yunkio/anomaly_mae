#!/usr/bin/env python
"""5-seed fill for the paper's LASAD-variant rows: excised, label-blind, and the ablation table
(w/o GRL / w/o FM / w/o anomaly-priority masking / w/o Student / symmetric decoders).

[2026-07-11 — user] The main table + ablation currently report seed42 only. We add the SAME top-5
seeds' remaining seeds {40,41,43,44} for each of these 7 conditions (seed42 already exists), so the
paper can report the 5-seed mean for LASAD(ours) AND its architecture-matched variants consistently.

Order (user): excised -> blind -> ablations, seeds within each. official=True, 30ep, keep=False, 4
datasets. Dir tag matches the existing seed42 tags so downstream extraction globs `_30ep_<seed>_<tag>`.

Early stopping (user D1=B): the ablations use the REAL recon_snr ES HALT (`use_reconsnr_es_halt=True`)
— training stops at the post-warmup recon_snr ES epoch (verified bit-identical to the post-hoc read).
excised/blind have NO train anomaly labels → recon_snr undefined → they run full 30ep and are read at
the per-seed baseline ES epoch (proxy), matching the paper's convention; so the halt is OFF for them.

Placement: runs FIRST (WAIT_TOKENS empty; run_base guard only). The caller preempts the current queue
(D4=B) and rewires the remaining launchers to wait on this token.

Usage: PYTHONHASHSEED=<seed> is set per run. python scripts/run_official_paper5seed_after.py
"""
import json, os, sys, subprocess, datetime, time, glob
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
PROJECT = '/home/ykio/notebooks/TSMAE'
ALL4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SUBDIRS = {'SWaT_A1A2': ['SWaT/A1A2_full', 'SWaT/A1A2_excl22'],
           'WaDi_A1': ['WaDi/A1'], 'WaDi_A2': ['WaDi/A2'], 'PSM': ['PSM']}
BASE = 'official=True num_epochs=30 official_keep_checkpoints=False'
# [2026-07-19 reorder — user] 3-SEED-FIRST: this launcher now WAITS for run_official_sens3seed_after
# (sparsity+rho seeds {40,41}) so every paper table reaches 3-seed {42,40,41} before ANY 5-seed work.
WAIT_TOKENS = ['run_official_sens3seed_after']
# [2026-07-19 — user] Paper 5-seed set fixed to CONSECUTIVE {40,41,42,43,44}; seeds 45/46 REMOVED
# (non-paper runs are not to be executed). seed42 already exists for every condition (blind@42
# stand-in = unlab100@42, user-approved equivalence for 3-seed AND 5-seed).
SEEDS = [40, 41, 43, 44]
# (tag, condition override, use_reconsnr_es_halt) — order: excised -> blind -> ablations
CONDITIONS = [
    ('exclanom',  'train_exclude_anomaly_segments=True',                       False),  # excised (no labels → no halt)
    ('blind',     'blind_train_labels=True',                                   False),  # label-blind (no labels → no halt)
    ('nogrl',     'use_grl=False anomaly_loss_weight=0',                       True),   # w/o GRL
    ('nofm',      'use_feature_matching=False',                                True),   # w/o FM
    ('noforce',   'force_mask_anomaly=False',                                  True),   # w/o anomaly-priority masking
    ('nostudent', 'use_student=False',                                         True),   # w/o Student (Teacher-only)
    ('td3sd3',    'num_teacher_decoder_layers=3 num_student_decoder_layers=3', True),   # symmetric decoders
]


def already_done(seed, tag):
    """Resume-safe skip: True if a completed run dir for (seed, tag) exists (all 4 datasets finalized).
    keep=False makes each condition a self-contained run, so a completed dir is safe to reuse."""
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
    if _pgrep('run_base_experiments.py --set'):   # never start while a training run_base is active
        return True
    return False


def run(outdir, override, tag, seed):
    ov = (BASE + ' ' + override + f' random_seed={seed}').strip()
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
           '--output-base', outdir, '--dataset'] + ALL4 + ['--config-override', ov]
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    print(f"[p5s] START {tag} seed{seed} -> {os.path.relpath(outdir, PROJECT)}\n  {ov}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT, env=env).returncode
    print(f"[p5s] DONE {tag} seed{seed} rc={rc}", flush=True)


def reviz(outdir, tag, seed):
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
    except Exception as e:
        print(f"[p5s] reviz import FAIL: {e}", flush=True); return
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
                print(f"[p5s] reviz {tag}/seed{seed}/{sub} OK", flush=True)
            except Exception as e:
                print(f"[p5s] reviz {tag}/seed{seed}/{sub} FAIL: {e}", flush=True)


def main():
    try:
        open('/tmp/official_paper5seed_pid.txt', 'w').write(str(os.getpid()))
    except Exception:
        pass
    print(f"[p5s] START {datetime.datetime.now():%Y%m%d_%H%M%S} — waiting for any active run_base to "
          f"clear, then running 5-seed variant fill (excised->blind->ablations); SEEDS=[40,41,43,44] (40/41 already done, skipped).",
          flush=True)
    waited = 0
    while queue_alive():
        time.sleep(15); waited += 1
    print(f"[p5s] no run_base after {waited*15}s — settling 10s then running.", flush=True)
    time.sleep(10)
    for seed in SEEDS:                                   # SEED-MAJOR: all conditions per seed
        for tag, cond_ov, halt in CONDITIONS:            # excised -> blind -> ablations
            if already_done(seed, tag):                  # resume-safe: skip completed (seed, condition)
                print(f"[p5s] SKIP {tag} seed{seed} (already complete)", flush=True)
                continue
            ov = cond_ov + (' use_reconsnr_es_halt=True' if halt else '')
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_{seed}_{tag}"
            run(outdir, ov, tag, seed)
            reviz(outdir, tag, seed)
            if not already_done(seed, tag):  # [2026-07-24] rc!=0 / cut-off before final files -> retry ONCE
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                outdir = f"{PROJECT}/results/experiments/official/271_{ts}_30ep_{seed}_{tag}"
                print(f"[p5s] RETRY {tag} seed{seed} (incomplete after 1st attempt)", flush=True)
                run(outdir, ov, tag, seed)
                reviz(outdir, tag, seed)
                if not already_done(seed, tag):
                    print(f"[p5s] WARN {tag} seed{seed} STILL INCOMPLETE after retry -- manual attention", flush=True)
    print(f"[p5s] ALL DONE {datetime.datetime.now():%Y%m%d_%H%M%S}", flush=True)


if __name__ == '__main__':
    main()
