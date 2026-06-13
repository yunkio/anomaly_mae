#!/usr/bin/env python
"""Resume v2d (2026-06-13) — resume after a SYSTEM REBOOT killed the v2c queue.

Context: the host/WSL rebooted ~20:43 on 2026-06-13, killing the v2c launcher
(447887) mid-exp312 (PSM ep251) and wiping /tmp. Completed/decided so far:
  - 298-306, 310, 311 : DONE (37/37; 298 missing SWaT excl22, deferred)
  - 307, 308, 309     : RevIN variants, intentionally SKIPPED (SWaT collapsed
                        ~0.35-0.38). MUST NOT be resumed.
  - 312 (scadA_w10)   : SWaT(full+excl22)/WaDi_A1/WaDi_A2 done, PSM partial @ep250
                        ckpt → resume; then 32 simple.
  - 313, 314          : not started (fresh).

=> FIRST_TORUN = 312 so we ONLY resume 312 and run 313/314. Setting it to 298
would re-enter the SKIPPED RevIN dirs (307-309 exist, partial) and wrongly
continue them, so 312 is required. Reuse-existing-dir logic identical to v2c
(run_base skips finalized datasets, resumes the partial PSM from latest_checkpoint).

Usage:
  python scripts/resume_dedup_v2d.py configs/queue_dedup_renumbered_v5.json
"""
import json, sys, os, subprocess, datetime, glob

PROJECT = '/home/ykio/notebooks/TSMAE'
EXP_ROOT = os.path.join(PROJECT, 'results/experiments')
BASE4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SIMPLE = json.load(open(os.path.join(PROJECT, 'temp/simple_keep_keys.json')))  # 32 kept
FIRST_TORUN = 312   # resume 312 (PSM partial) + 313, 314; skip done 298-311 & RevIN 307-309


def run_base(outdir, datasets, override, set_='C'):
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', set_, '--no-wait',
           '--output-base', outdir, '--dataset'] + datasets + ['--config-override', override]
    print(f"  CMD: run_base --output-base {os.path.basename(outdir)} --dataset ({len(datasets)} ds) [batch=1024]", flush=True)
    return subprocess.run(cmd, cwd=PROJECT).returncode


def main():
    q = json.load(open(sys.argv[1]))
    exps = [e for e in q['experiments'] if e['exp_num'] >= FIRST_TORUN]
    exps.sort(key=lambda e: e['exp_num'])   # 312, 313, 314
    print(f"RESUME DEDUP-v2d START {datetime.datetime.now():%Y%m%d_%H%M%S} — {len(exps)} to-run "
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

    print("\n=== RESUME DEDUP-v2d DONE ===", flush=True)


if __name__ == '__main__':
    main()
