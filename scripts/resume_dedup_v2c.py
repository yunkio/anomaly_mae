#!/usr/bin/env python
"""Resume v2c (2026-06-11) — resume exp298 (interrupted) + run 299-314.

Context: exp297(dyn_dmodel) OOM-killed during MSL → recovered separately.
exp298(ep300_warm150) was manually stopped (SWaT done, WaDi_A1 just started) to
free GPU for the MSL recovery. Now resume exp298 into its EXISTING dir, then
299-314 fresh.

Key change vs v2b: **reuse an existing experiment dir if one is present** (glob
{num}_*_{suffix}) instead of always creating a fresh-TS dir — so exp298 resumes
(run_base skips finalized datasets via metadata marker, resume-finalizes SWaT
from latest_checkpoint if its finalize was interrupted, resumes the partial
WaDi_A1, runs the rest). 299-314 have no existing dir → fresh.

All remaining exps (298-314) are fixed-d_model / window variants @ batch_size=1024
(NOT dyn_dmodel) → normal memory profile, like 271/295/296 which completed. No
WaDi@512 split needed (295/297 are done).

Usage:
  python scripts/resume_dedup_v2c.py configs/queue_dedup_renumbered_v5.json
"""
import json, sys, os, subprocess, datetime, glob

PROJECT = '/home/ykio/notebooks/TSMAE'
EXP_ROOT = os.path.join(PROJECT, 'results/experiments')
BASE4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SIMPLE = json.load(open(os.path.join(PROJECT, 'temp/simple_keep_keys.json')))  # 32 kept
DONE_SKIP = {295, 296, 297}   # 295 done, 296 done, 297 done (incl MSL recovery)
FIRST_TORUN = 298


def run_base(outdir, datasets, override, set_='C'):
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', set_, '--no-wait',
           '--output-base', outdir, '--dataset'] + datasets + ['--config-override', override]
    print(f"  CMD: run_base --output-base {os.path.basename(outdir)} --dataset ({len(datasets)} ds) [batch=1024]", flush=True)
    return subprocess.run(cmd, cwd=PROJECT).returncode


def main():
    q = json.load(open(sys.argv[1]))
    exps = [e for e in q['experiments'] if e['exp_num'] >= FIRST_TORUN and e['exp_num'] not in DONE_SKIP]
    exps.sort(key=lambda e: e['exp_num'])   # 298, 299, ..., 314
    print(f"RESUME DEDUP-v2c START {datetime.datetime.now():%Y%m%d_%H%M%S} — {len(exps)} to-run "
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

    print("\n=== RESUME DEDUP-v2c DONE ===", flush=True)


if __name__ == '__main__':
    main()
