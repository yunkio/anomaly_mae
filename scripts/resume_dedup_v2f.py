#!/usr/bin/env python
"""Resume v2f (2026-06-14, ext 2026-06-15) — run NEW experiments 316-323 (after v2e finishes 311-315).

Context: new ablations appended to the queue (configs/queue_dedup_renumbered_v6.json):
  - 316 = freeze_enc_after_warmup  (271 + freeze_encoder_only=True; encoder frozen post-warmup)
  - 317 = lbm_mse_norm_dann        (271 + loss_balance_mode=mse_norm_dann)
  - 318 = lbm_relobralo            (271 + loss_balance_mode=relobralo)
  - 319 = lbm_famo                 (271 + loss_balance_mode=famo)
  - 320 = lbm_uwso                 (271 + loss_balance_mode=uwso)
  - 321 = scadC_w10_linear         (271 + SCAD Form C: one-sided thresholded repulsion, gamma=0)
  - 322 = scadA_w10_linear         (271 + SCAD Form A, linear head, w=1.0 — C 비교군)
  - 323 = scadB_w10_linear         (271 + SCAD Form B, linear head, w=1.0 — C 비교군)
  - 324 = grl_first_layer          (271 + grl_attach_layer=first: GRL on student decoder layer-1)
  - 325 = dec_dmodel_half          (271 + decoder_half_dim=True: MAE-style decoder width = d_model//2)
All 316-325 have NO existing dir -> fresh runs. They re-import the current
mae_anomaly code (subprocess per experiment), so loss_balance_mode, the
freeze_encoder_only resume fix, SCAD Form C, grl_attach_layer, and
decoder_half_dim are all active.

⚠️ DO NOT launch this while the v2e queue (311-315) is still running — it would
double-book the GPU. Launch only AFTER v2e prints "=== RESUME DEDUP-v2e DONE ===".

FIRST_TORUN = 316. Reuse-existing-dir logic kept (defensive); all 316-323 are fresh.

Usage:
  python scripts/resume_dedup_v2f.py configs/queue_dedup_renumbered_v6.json
"""
import json, sys, os, subprocess, datetime, glob

PROJECT = '/home/ykio/notebooks/TSMAE'
EXP_ROOT = os.path.join(PROJECT, 'results/experiments')
BASE4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SIMPLE = json.load(open(os.path.join(PROJECT, 'temp/simple_keep_keys.json')))  # 32 kept
FIRST_TORUN = 316   # run 316 (freeze enc) + 317-320 (loss_balance_mode ablation)


def run_base(outdir, datasets, override, set_='C'):
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', set_, '--no-wait',
           '--output-base', outdir, '--dataset'] + datasets + ['--config-override', override]
    print(f"  CMD: run_base --output-base {os.path.basename(outdir)} --dataset ({len(datasets)} ds) [batch=1024]", flush=True)
    return subprocess.run(cmd, cwd=PROJECT).returncode


def main():
    q = json.load(open(sys.argv[1]))
    exps = [e for e in q['experiments'] if e['exp_num'] >= FIRST_TORUN]
    exps.sort(key=lambda e: e['exp_num'])   # 316, 317, 318, 319, 320
    print(f"RESUME DEDUP-v2f START {datetime.datetime.now():%Y%m%d_%H%M%S} — {len(exps)} to-run "
          f"({', '.join(str(e['exp_num']) for e in exps)}) ; {len(SIMPLE)} simple", flush=True)

    for i, e in enumerate(exps):
        num, suffix = e['exp_num'], e['dir_suffix']
        existing = sorted(glob.glob(os.path.join(EXP_ROOT, f"{num}_*_{suffix}")))
        if existing:
            outdir = existing[-1]; tag = 'RESUME (existing dir)'
        else:
            TS = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            outdir = os.path.join(EXP_ROOT, f"{num}_{TS}_{suffix}"); tag = 'fresh'
        print(f"\n##### [{i + 1}/{len(exps)}] exp{num} ({e['name']}) {tag} -> {os.path.basename(outdir)} #####", flush=True)
        t0 = datetime.datetime.now()
        rc = run_base(outdir, list(BASE4) + list(SIMPLE), e['config_override'], e['set'])
        dt = (datetime.datetime.now() - t0).total_seconds() / 60
        print(f"##### exp{num} done rc={rc} in {dt:.1f}min #####", flush=True)

    print("\n=== RESUME DEDUP-v2f DONE ===", flush=True)


if __name__ == '__main__':
    main()
