#!/usr/bin/env python
"""Resume v2b (2026-06-10) — WaDi batch_size=512 for memory-heavy exps + new exp297.

Queue = configs/queue_dedup_renumbered_v3.json (30 exps: 271,274,285-312).
New exp297 (dyn_dmodel) inserted; old 297-311 renumbered to 298-312.

WaDi OOM avoidance (user 2026-06-10): exp **295 (w500_p5)** and **297 (dyn_dmodel)**
run WaDi_A1/A2 at **batch_size=512** (rest of their datasets stay 1024).
- 295: GPU mem ~98% on SWaT @1024 (num_patches=100). Resumes into EXISTING dir
  (SWaT already done -> skipped via metadata marker): WaDi@512 + PSM+simple@1024.
- 297: fresh dir, SWaT+PSM+simple @1024 + WaDi@512.
All other exps (298-312) run normally @1024 in fresh dirs. 296 (w300_p10) done.

Usage:
  python scripts/resume_dedup_v2b.py configs/queue_dedup_renumbered_v3.json
"""
import json, sys, os, subprocess, datetime, glob

PROJECT = '/home/ykio/notebooks/TSMAE'
EXP_ROOT = os.path.join(PROJECT, 'results/experiments')
BASE4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SIMPLE = json.load(open(os.path.join(PROJECT, 'temp/simple_keep_keys.json')))  # 32 kept
DONE_SKIP = {296}            # w300_p10 finished (dir renamed 295->296)
FIRST_TORUN = 295
WADI_HALF = {295, 297}       # WaDi @ batch 512 (OOM avoidance)
RESUME_EXISTING = {295}      # resume into existing dir (SWaT already done)


def run_base(outdir, datasets, override, set_='C'):
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', set_, '--no-wait',
           '--output-base', outdir, '--dataset'] + datasets + ['--config-override', override]
    print(f"  CMD: run_base --output-base {os.path.basename(outdir)} --dataset ({len(datasets)} ds) "
          f"[batch={'512' if 'batch_size=512' in override else '1024'}]", flush=True)
    return subprocess.run(cmd, cwd=PROJECT).returncode


def main():
    q = json.load(open(sys.argv[1]))
    exps = [e for e in q['experiments'] if e['exp_num'] >= FIRST_TORUN and e['exp_num'] not in DONE_SKIP]
    exps.sort(key=lambda e: e['exp_num'])   # 295, 297, 298, ..., 312
    print(f"RESUME DEDUP-v2b START {datetime.datetime.now():%Y%m%d_%H%M%S} — {len(exps)} to-run "
          f"({', '.join(str(e['exp_num']) for e in exps)}) ; {len(SIMPLE)} simple "
          f"; WaDi@batch512 for {sorted(WADI_HALF)}", flush=True)

    for i, e in enumerate(exps):
        num, suffix = e['exp_num'], e['dir_suffix']
        ov = e['config_override']
        t0 = datetime.datetime.now()

        if num in WADI_HALF:
            ov512 = ov.replace('batch_size=1024', 'batch_size=512')
            assert 'batch_size=512' in ov512, f"exp{num}: batch_size=1024 not in override"
            if num in RESUME_EXISTING:
                cand = sorted(glob.glob(os.path.join(EXP_ROOT, f"{num}_*_{suffix}")))
                assert cand, f"existing dir for exp{num} not found"
                outdir = cand[-1]
                rest = ['PSM'] + list(SIMPLE)            # SWaT done -> skipped via marker
                tag = 'RESUME'
            else:
                TS = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                outdir = os.path.join(EXP_ROOT, f"{num}_{TS}_{suffix}")
                rest = ['SWaT_A1A2', 'PSM'] + list(SIMPLE)
                tag = 'fresh'
            print(f"\n##### [{i + 1}/{len(exps)}] exp{num} ({e['name']}) {tag} -> {os.path.basename(outdir)} #####", flush=True)
            print(f"  -- WaDi_A1/A2 @ batch_size=512 (OOM avoidance) --", flush=True)
            rc1 = run_base(outdir, ['WaDi_A1', 'WaDi_A2'], ov512, e['set'])
            print(f"  -- {len(rest)} ds (SWaT/PSM/simple) @ batch_size=1024 --", flush=True)
            rc2 = run_base(outdir, rest, ov, e['set'])
            rc = rc1 or rc2
        else:
            TS = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')   # per-experiment start time
            outdir = os.path.join(EXP_ROOT, f"{num}_{TS}_{suffix}")
            print(f"\n##### [{i + 1}/{len(exps)}] exp{num} ({e['name']}) -> {os.path.basename(outdir)} #####", flush=True)
            rc = run_base(outdir, list(BASE4) + list(SIMPLE), ov, e['set'])

        dt = (datetime.datetime.now() - t0).total_seconds() / 60
        print(f"##### exp{num} done rc={rc} in {dt:.1f}min #####", flush=True)

    print("\n=== RESUME DEDUP-v2b DONE ===", flush=True)


if __name__ == '__main__':
    main()
