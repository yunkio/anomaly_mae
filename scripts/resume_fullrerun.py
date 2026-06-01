#!/usr/bin/env python
"""Resume the PAUSED fullrerun (paused 2026-06-02 ~01:28).

State at pause:
  exp271 (1/31) — 6/7 datasets COMPLETE
    DONE : SWaT_A1A2 (full+excl22, weights kept), WaDi_A1, WaDi_A2, PSM (weights kept),
           SMAP_concat, MSL_concat (metrics/scores/viz complete; weights deleted BY DESIGN —
           only KEEP_CHECKPOINT_DATASETS={SWaT_A1A2,WaDi_A1,WaDi_A2,PSM} keep best_model.pt).
    TODO : SMD_concat (was at ~ep7-15; no epoch_metrics/best_config → incomplete, restart fresh).
  exp274..313 (30 experiments) — NOT STARTED.

Resume strategy (run_base has NO mid-training checkpoint resume; --start-from is a dataset
INDEX skip in run_base's own order, which is confusing → we resume by explicit --dataset):
  Phase 1: finish exp271 by running ONLY SMD_concat into the EXISTING exp271 dir.
  Phase 2: run exp274..313 (all 7 datasets each) into fresh numbered dirs (new TS).

Usage:
  python scripts/resume_fullrerun.py configs/queue_fullrerun_20260601_190603.json
"""
import json, sys, os, subprocess, datetime, shutil

PROJECT = '/home/ykio/notebooks/TSMAE'
EXP_ROOT = os.path.join(PROJECT, 'results/experiments')
EXP271_DIR = os.path.join(EXP_ROOT, '271_20260601_190639_271canon_baseline')


def run_base(outdir, datasets, override, set_='C'):
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', set_, '--no-wait',
           '--output-base', outdir, '--dataset'] + datasets + ['--config-override', override]
    print('  CMD:', ' '.join(cmd[:9]), '...', flush=True)
    return subprocess.run(cmd, cwd=PROJECT).returncode


def main():
    queue_path = sys.argv[1]
    q = json.load(open(queue_path))
    exps = q['experiments']
    e271 = exps[0]
    assert e271['exp_num'] == 271, f"expected exp271 first, got {e271['exp_num']}"
    TS = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"RESUME START {TS}  (exp271 SMD_concat + exp274..313)", flush=True)

    # ---- Phase 1: finish exp271 — SMD_concat only ----
    smd_dir = os.path.join(EXP271_DIR, 'SMD', 'concat')
    if os.path.isdir(smd_dir):
        print(f"[resume P1] removing partial SMD dir: {smd_dir}", flush=True)
        shutil.rmtree(smd_dir)
    print("##### [resume P1] exp271 -> SMD_concat (finish 7/7) #####", flush=True)
    rc = run_base(EXP271_DIR, ['SMD_concat'], e271['config_override'], e271['set'])
    print(f"##### exp271 SMD_concat done rc={rc} #####", flush=True)

    # ---- Phase 2: exp274..313 (all 7 datasets each) ----
    rest = exps[1:]
    print(f"##### [resume P2] {len(rest)} experiments (exp{rest[0]['exp_num']}..{rest[-1]['exp_num']}) #####", flush=True)
    for i, e in enumerate(rest):
        num, suffix = e['exp_num'], e['dir_suffix']
        outdir = os.path.join(EXP_ROOT, f"{num}_{TS}_{suffix}")
        print(f"\n##### [{i + 1}/{len(rest)}] exp{num} ({e['name']}) -> {os.path.basename(outdir)} #####", flush=True)
        t0 = datetime.datetime.now()
        rc = run_base(outdir, e['dataset'], e['config_override'], e['set'])
        dt = (datetime.datetime.now() - t0).total_seconds() / 60
        print(f"##### exp{num} done rc={rc} in {dt:.1f}min #####", flush=True)

    print("\n=== RESUME FULLRERUN DONE ===", flush=True)


if __name__ == '__main__':
    main()
