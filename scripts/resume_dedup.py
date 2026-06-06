#!/usr/bin/env python
"""Resume after DEDUP + RENUMBER (2026-06-05).

Queue deduplicated: removed exp291(freezeoff≡271), exp294(freezeoff_nofm≡285),
exp300(w500_p5≡271, since Config default patch_size=5). Remaining 28 experiments
renumbered contiguously (anchors 271/274 fixed; ablation block 285..310).

Completed (dirs renamed, NOT re-run): 271,274,285,286,287,288,289,290,
  291(freezeoff_ema, was 292), 292(freezeoff_warmstop, was 293).
To-run (this script): new exp 293..310 (18 experiments) = entries[10:].

New queue = configs/queue_dedup_renumbered.json (28 experiments, sorted by new num).
Each to-run experiment = [PSM, SWaT_A1A2, WaDi_A1, WaDi_A2] + 32 simple, fresh dir.

Usage:
  python scripts/resume_dedup.py configs/queue_dedup_renumbered.json
"""
import json, sys, os, subprocess, datetime

PROJECT = '/home/ykio/notebooks/TSMAE'
EXP_ROOT = os.path.join(PROJECT, 'results/experiments')
BASE4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SIMPLE = json.load(open(os.path.join(PROJECT, 'temp/simple_keep_keys.json')))  # 32 kept
FIRST_TORUN = 293  # new numbering: 271,274,285..292 done; run 293..310


def run_base(outdir, datasets, override, set_='C'):
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', set_, '--no-wait',
           '--output-base', outdir, '--dataset'] + datasets + ['--config-override', override]
    print(f"  CMD: run_base --output-base {os.path.basename(outdir)} --dataset ({len(datasets)} ds)", flush=True)
    return subprocess.run(cmd, cwd=PROJECT).returncode


def main():
    q = json.load(open(sys.argv[1]))
    exps = [e for e in q['experiments'] if e['exp_num'] >= FIRST_TORUN]
    exps.sort(key=lambda e: e['exp_num'])
    print(f"RESUME DEDUP START {datetime.datetime.now():%Y%m%d_%H%M%S} — {len(exps)} to-run "
          f"exp{exps[0]['exp_num']}..{exps[-1]['exp_num']} ; {len(SIMPLE)} simple", flush=True)

    for i, e in enumerate(exps):
        num, suffix = e['exp_num'], e['dir_suffix']
        # Per-experiment timestamp = moment this experiment's dir is created (NOT queue start).
        TS = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = os.path.join(EXP_ROOT, f"{num}_{TS}_{suffix}")
        print(f"\n##### [{i + 1}/{len(exps)}] exp{num} ({e['name']}) -> {os.path.basename(outdir)} #####", flush=True)
        t0 = datetime.datetime.now()
        rc = run_base(outdir, list(BASE4) + list(SIMPLE), e['config_override'], e['set'])
        dt = (datetime.datetime.now() - t0).total_seconds() / 60
        print(f"##### exp{num} done rc={rc} in {dt:.1f}min #####", flush=True)

    print("\n=== RESUME DEDUP DONE ===", flush=True)


if __name__ == '__main__':
    main()
