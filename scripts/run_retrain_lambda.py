#!/usr/bin/env python
"""Retrain the FLIPPED canonical cells under the 4:1/no-FM lambda change (방법 B).

For each experiment in the retrain queue:
  1. MOVE its flipped dataset dirs (SWaT / WaDi / PSM as applicable) to a backup
     under .trash/0601/lambda_retrain_pre/<run>/ so run_base does NOT skip them
     (skip = experiment_metadata.json present). No-flip cells stay -> auto-skipped.
  2. run_base_experiments.py --set C --no-wait --output-base <orig_run_dir>
     --dataset <flipped datasets> --config-override <orig override>
     -> retrains INTO the original dir; scoring.py (already 4:1/no-FM) makes the
        per-epoch eval pick + checkpoint the NEW best epoch automatically.

All moves happen FIRST (per user instruction "모두 백업으로 옮기고"), then the
experiments run sequentially (one GPU experiment at a time).

Usage:
  python scripts/run_retrain_lambda.py --queue <queue.json> --dry-run
  python scripts/run_retrain_lambda.py --queue <queue.json> --apply
"""
import os, sys, json, glob, shutil, subprocess, argparse, time

PROJECT = '/home/ykio/notebooks/TSMAE'
BACKUP_ROOT = os.path.join(PROJECT, '.trash/0601/lambda_retrain_pre')
EXP_ROOT = os.path.join(PROJECT, 'results/experiments')

# flipped cell ds (e.g. 'SWaT/A1A2_full', 'WaDi/A1', 'PSM') -> top-level dataset dir
def top_dir(ds):
    return ds.split('/')[0]   # 'SWaT', 'WaDi', 'PSM'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', required=True)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    apply = args.apply and not args.dry_run

    q = json.load(open(args.queue))
    exps = q['experiments']

    # ---- plan: per-exp orig dir, top dirs to move, dataset list, override ----
    plans = []
    for e in exps:
        run = e['orig_run_dir']
        orig = os.path.join(EXP_ROOT, run)
        tops = sorted({top_dir(c['ds']) for c in e['flipped_cells']})
        plans.append({'name': e['name'], 'run': run, 'orig': orig, 'tops': tops,
                      'datasets': e['dataset'], 'override': e['config_override']})

    print(f"{'APPLY' if apply else 'DRY-RUN'} — retrain {len(plans)} experiments\n")
    for p in plans:
        exist = [t for t in p['tops'] if os.path.isdir(os.path.join(p['orig'], t))]
        print(f"  {p['name']:<26} dir={p['run']}")
        print(f"      move->backup: {exist}   retrain datasets: {p['datasets']}")

    if not apply:
        print("\n(dry-run; use --apply to execute. WARNING: --apply launches hours of GPU training.)")
        return

    # ---- STEP 1: move ALL flipped cells to backup ----
    print("\n=== STEP 1: backup-move flipped cells ===", flush=True)
    for p in plans:
        for t in p['tops']:
            src = os.path.join(p['orig'], t)
            if not os.path.isdir(src):
                print(f"  (skip, absent) {p['run']}/{t}"); continue
            dst = os.path.join(BACKUP_ROOT, p['run'], t)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                print(f"  (backup exists, leaving source!) {dst}"); continue
            shutil.move(src, dst)
            print(f"  moved {p['run']}/{t} -> backup", flush=True)

    # ---- STEP 2: sequential retrain into orig dirs ----
    print("\n=== STEP 2: retrain (sequential) ===", flush=True)
    for i, p in enumerate(plans):
        print(f"\n##### [{i+1}/{len(plans)}] {p['name']} -> {p['run']} #####", flush=True)
        cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
               '--output-base', p['orig'], '--dataset'] + p['datasets'] + \
              ['--config-override', p['override']]
        print('  CMD:', ' '.join(cmd[:6]), '...', flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=PROJECT).returncode
        print(f"##### {p['name']} done rc={rc} in {(time.time()-t0)/60:.1f}min #####", flush=True)

    print("\n=== ALL RETRAIN DONE ===", flush=True)


if __name__ == '__main__':
    main()
