#!/usr/bin/env python
"""Resume after w500_p5 RESTORE + renumber (2026-06-09).

Dedup correction: w500_p5 was wrongly removed (Set C preset patch_size=10 ≠
override patch_size=5 → distinct from 271). Restored as new exp295; old 295-310
shifted to 296-311. Queue = configs/queue_dedup_renumbered_v2.json (29 exps).

Done (NOT re-run): 271,274,285-294 + 296 (w300_p10, finished under old numbering,
dir renamed 295->296).
To-run this script: new exp **295 (w500_p5) then 297..311** (296 excluded = done).
Order: 295 first, then 297,298,...,311 (sorted ascending, 296 skipped).

Each = [PSM, SWaT_A1A2, WaDi_A1, WaDi_A2] + 32 simple, fresh dir, per-experiment TS.

Usage:
  python scripts/resume_dedup_v2.py configs/queue_dedup_renumbered_v2.json
"""
import json, sys, os, subprocess, datetime

PROJECT = '/home/ykio/notebooks/TSMAE'
EXP_ROOT = os.path.join(PROJECT, 'results/experiments')
BASE4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SIMPLE = json.load(open(os.path.join(PROJECT, 'temp/simple_keep_keys.json')))  # 32 kept
DONE_SKIP = {296}            # w300_p10 finished under old numbering (dir renamed 295->296)
FIRST_TORUN = 295            # run 295 (w500_p5) onward


def run_base(outdir, datasets, override, set_='C'):
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', set_, '--no-wait',
           '--output-base', outdir, '--dataset'] + datasets + ['--config-override', override]
    print(f"  CMD: run_base --output-base {os.path.basename(outdir)} --dataset ({len(datasets)} ds)", flush=True)
    return subprocess.run(cmd, cwd=PROJECT).returncode


def main():
    q = json.load(open(sys.argv[1]))
    exps = [e for e in q['experiments'] if e['exp_num'] >= FIRST_TORUN and e['exp_num'] not in DONE_SKIP]
    exps.sort(key=lambda e: e['exp_num'])   # 295, 297, 298, ..., 311
    print(f"RESUME DEDUP-v2 START {datetime.datetime.now():%Y%m%d_%H%M%S} — {len(exps)} to-run "
          f"({', '.join(str(e['exp_num']) for e in exps)}) ; {len(SIMPLE)} simple", flush=True)

    for i, e in enumerate(exps):
        num, suffix = e['exp_num'], e['dir_suffix']
        TS = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')   # per-experiment start time
        outdir = os.path.join(EXP_ROOT, f"{num}_{TS}_{suffix}")
        print(f"\n##### [{i + 1}/{len(exps)}] exp{num} ({e['name']}) -> {os.path.basename(outdir)} #####", flush=True)
        t0 = datetime.datetime.now()
        rc = run_base(outdir, list(BASE4) + list(SIMPLE), e['config_override'], e['set'])
        dt = (datetime.datetime.now() - t0).total_seconds() / 60
        print(f"##### exp{num} done rc={rc} in {dt:.1f}min #####", flush=True)

    print("\n=== RESUME DEDUP-v2 DONE ===", flush=True)


if __name__ == '__main__':
    main()
