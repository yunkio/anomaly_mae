#!/usr/bin/env python
"""Resume the PAUSED fullrerun — v4 (2026-06-03 ~08:54, during exp286/WaDi_A2).

SMAP/MSL/SMD run as SIMPLE ONLY (32 kept entities; see temp/simple_keep_keys.json).

State at THIS pause (run log = latest temp/phase1_logs/resume3_fullrerun_*.log):
  exp271, exp274, exp285 : FULLY DONE (base 4 + 32 simple; 271 also has the old concat).
  exp286 (clamp_pm4, Phase C 2/29): 3/36 done
    DONE : SWaT_A1A2 (full+excl22), WaDi_A1.
    TODO : WaDi_A2 (~ep140 — restart ep1, no mid-resume), PSM, + 32 simple.
  exp287..313 (27 experiments): NOT STARTED.

Resume:
  Phase 1: finish exp286 -> WaDi_A2 (rmtree partial) + PSM + 32 simple into EXISTING exp286 dir.
  Phase 2: exp287..313 (queue experiments[4:]) each = [PSM,SWaT_A1A2,WaDi_A1,WaDi_A2] + 32 simple, fresh dirs.

Usage:
  python scripts/resume_fullrerun.py configs/queue_fullrerun_20260601_190603.json
"""
import json, sys, os, subprocess, datetime, shutil

PROJECT = '/home/ykio/notebooks/TSMAE'
EXP_ROOT = os.path.join(PROJECT, 'results/experiments')
EXP286_DIR = os.path.join(EXP_ROOT, '286_20260602_164648_clamp_pm4')
BASE4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SIMPLE = json.load(open(os.path.join(PROJECT, 'temp/simple_keep_keys.json')))  # 32 kept


def run_base(outdir, datasets, override, set_='C'):
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', set_, '--no-wait',
           '--output-base', outdir, '--dataset'] + datasets + ['--config-override', override]
    print(f"  CMD: run_base --output-base {os.path.basename(outdir)} --dataset ({len(datasets)} ds)", flush=True)
    return subprocess.run(cmd, cwd=PROJECT).returncode


def main():
    q = json.load(open(sys.argv[1]))
    exps = q['experiments']
    e286 = exps[3]
    assert e286['exp_num'] == 286, f"expected exp286 at index 3, got {e286['exp_num']}"
    TS = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"RESUME v4 START {TS} — exp286 remaining + exp287..313 ; {len(SIMPLE)} simple", flush=True)

    # ---- Phase 1: finish exp286 (WaDi_A2 + PSM + 32 simple) ----
    wadi_a2 = os.path.join(EXP286_DIR, 'WaDi', 'A2')
    if os.path.isdir(wadi_a2):
        print(f"[Phase 1] removing partial WaDi/A2: {wadi_a2}", flush=True)
        shutil.rmtree(wadi_a2)
    print("\n##### [Phase 1] exp286 remaining = WaDi_A2 + PSM + simple (34) #####", flush=True)
    rc = run_base(EXP286_DIR, ['WaDi_A2', 'PSM'] + list(SIMPLE), e286['config_override'], e286['set'])
    print(f"##### exp286 remaining done rc={rc} #####", flush=True)

    # ---- Phase 2: exp287..313 (queue experiments[4:]) ----
    rest = exps[4:]
    print(f"\n##### [Phase 2] {len(rest)} experiments exp{rest[0]['exp_num']}..{rest[-1]['exp_num']} (36 ds each) #####", flush=True)
    for i, e in enumerate(rest):
        num, suffix = e['exp_num'], e['dir_suffix']
        outdir = os.path.join(EXP_ROOT, f"{num}_{TS}_{suffix}")
        print(f"\n##### [{i + 1}/{len(rest)}] exp{num} ({e['name']}) -> {os.path.basename(outdir)} #####", flush=True)
        t0 = datetime.datetime.now()
        rc = run_base(outdir, list(BASE4) + list(SIMPLE), e['config_override'], e['set'])
        dt = (datetime.datetime.now() - t0).total_seconds() / 60
        print(f"##### exp{num} done rc={rc} in {dt:.1f}min #####", flush=True)

    print("\n=== RESUME v4 FULLRERUN DONE ===", flush=True)


if __name__ == '__main__':
    main()
