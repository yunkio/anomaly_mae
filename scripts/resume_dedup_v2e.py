#!/usr/bin/env python
"""Resume v2e (2026-06-13) — restart 311-315 with the unified disable_anomaly_loss fix.

Context: a code fix unified the GRL/SCAD anomaly-loss disable (loss.py
self.disable_anomaly_loss = (use_grl and grl_disable_anomaly_loss) or use_scad).
The old SCAD runs (311 scadA_w05, 312 scadA_w10) were contaminated (anomaly_loss was
left active) and have been STOPPED + DELETED. The queue was renumbered (v6):
  - 311 = NEW noanom_nogrl (271 base, use_grl=False, anomaly_loss_weight=0.0)
  - 312 = scadA_w05, 313 = scadA_w10, 314 = scadB_w05, 315 = scadB_w10  (renumbered +1)
All 311-315 have NO existing dir -> fresh runs under the fixed code.

Done/skipped earlier (NOT re-run): 298-306, 310 done; 307-309 RevIN skipped.

FIRST_TORUN = 311. Reuse-existing-dir logic kept (defensive); all 311-315 are fresh.

Usage:
  python scripts/resume_dedup_v2e.py configs/queue_dedup_renumbered_v6.json
"""
import json, sys, os, subprocess, datetime, glob

PROJECT = '/home/ykio/notebooks/TSMAE'
EXP_ROOT = os.path.join(PROJECT, 'results/experiments')
BASE4 = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2']
SIMPLE = json.load(open(os.path.join(PROJECT, 'temp/simple_keep_keys.json')))  # 32 kept
FIRST_TORUN = 311   # run 311 (new noanom_nogrl) + 312-315 (renumbered scad), fixed code


def run_base(outdir, datasets, override, set_='C'):
    cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', set_, '--no-wait',
           '--output-base', outdir, '--dataset'] + datasets + ['--config-override', override]
    print(f"  CMD: run_base --output-base {os.path.basename(outdir)} --dataset ({len(datasets)} ds) [batch=1024]", flush=True)
    return subprocess.run(cmd, cwd=PROJECT).returncode


def main():
    q = json.load(open(sys.argv[1]))
    exps = [e for e in q['experiments'] if e['exp_num'] >= FIRST_TORUN]
    exps.sort(key=lambda e: e['exp_num'])   # 311, 312, 313, 314, 315
    print(f"RESUME DEDUP-v2e START {datetime.datetime.now():%Y%m%d_%H%M%S} — {len(exps)} to-run "
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

    print("\n=== RESUME DEDUP-v2e DONE ===", flush=True)


if __name__ == '__main__':
    main()
