#!/usr/bin/env python
"""Run the full re-run queue sequentially, pinning each experiment's output dir
number (results/experiments/<exp_num>_<TS>_<suffix>) so numbering never tangles.

Usage: python scripts/run_fullrerun.py configs/queue_fullrerun_<stamp>.json
"""
import json, sys, os, subprocess, datetime

PROJECT = '/home/ykio/notebooks/TSMAE'
EXP_ROOT = os.path.join(PROJECT, 'results/experiments')


def main():
    queue_path = sys.argv[1]
    q = json.load(open(queue_path))
    exps = q['experiments']
    TS = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"FULLRERUN START {TS} — {len(exps)} experiments × {len(q['datasets'])} datasets", flush=True)
    print(f"datasets: {q['datasets']}", flush=True)
    print(f"nums: {[e['exp_num'] for e in exps]}", flush=True)

    for i, e in enumerate(exps):
        num, suffix = e['exp_num'], e['dir_suffix']
        outdir = os.path.join(EXP_ROOT, f"{num}_{TS}_{suffix}")
        cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', e['set'], '--no-wait',
               '--output-base', outdir, '--dataset'] + e['dataset'] + \
              ['--config-override', e['config_override']]
        print(f"\n##### [{i+1}/{len(exps)}] exp{num} ({e['name']}) -> {os.path.basename(outdir)} #####", flush=True)
        t0 = datetime.datetime.now()
        rc = subprocess.run(cmd, cwd=PROJECT).returncode
        dt = (datetime.datetime.now() - t0).total_seconds() / 60
        print(f"##### exp{num} done rc={rc} in {dt:.1f}min #####", flush=True)

    print("\n=== ALL FULLRERUN DONE ===", flush=True)


if __name__ == '__main__':
    main()
