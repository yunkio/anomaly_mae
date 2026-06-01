"""Recover failed bg-worker(s) by re-running from saved recovery payloads.

When a bg-worker (`_cpu_eval_viz_worker` in run_base_experiments.py) crashes,
the IPC payload it received from the main process is dumped to
`<exp_dir>/_bg_worker_recovery_{full|excl22}.pkl` BEFORE any work begins,
so all data needed for retry is preserved on disk. On full success, the
bg-worker deletes its own recovery payload (zero permanent footprint).

This script scans `results/experiments/**/_bg_worker_recovery_*.pkl` and
re-runs the bg-worker for each. Useful when:
  - bg-worker crashed due to a code bug, code is now fixed
  - bg-worker died from transient resource issue (OOM, disk full)
  - manual recovery after `Ctrl-C` during eval+viz

Usage:
    python scripts/recover_bg_worker.py                  # scan all exp dirs
    python scripts/recover_bg_worker.py <exp_dir>        # one exp dir
    python scripts/recover_bg_worker.py --dry-run        # list only

After each successful re-run, the recovery file is auto-deleted.
"""
import argparse
import glob
import os
import pickle
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def find_recovery_files(root):
    if os.path.isfile(root):
        return [root]
    if os.path.isdir(root):
        files = glob.glob(os.path.join(root, '**', '_bg_worker_recovery_*.pkl'),
                          recursive=True)
        return sorted(files)
    return []


def recover_one(path, dry_run=False):
    print(f"\n{'='*80}")
    print(f"Recovery file: {path}")
    sz_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  size: {sz_mb:.1f} MB")
    if dry_run:
        # Load + report only
        try:
            with open(path, 'rb') as f:
                payload = pickle.load(f)
            print(f"  exp_name: {payload.get('exp_name')}")
            print(f"  exp_dir: {payload.get('exp_dir')}")
            print(f"  swat_eval_mode: {payload.get('swat_eval_mode')}")
            print(f"  signals shape: {payload['signals'].shape}")
            print(f"  best_epoch (timing): {payload['timing'].get('best_epoch')}")
        except Exception as e:
            print(f"  ❌ load failed: {e}")
        return None

    try:
        with open(path, 'rb') as f:
            payload = pickle.load(f)
    except Exception as e:
        print(f"  ❌ load failed: {e}")
        return False

    print(f"  Re-running bg-worker for: {payload['exp_name']} (mode={payload.get('swat_eval_mode') or 'full'})")
    try:
        # Import bg-worker entrypoint. This re-imports the fixed module.
        from scripts.run_base_experiments import _cpu_eval_viz_worker
        _cpu_eval_viz_worker(**payload)
        # _cpu_eval_viz_worker deletes its own recovery file on success
        if not os.path.exists(path):
            print(f"  ✓ SUCCESS — recovery file deleted")
            return True
        else:
            print(f"  ⚠️ bg-worker returned but recovery file still present (unexpected)")
            return False
    except Exception as e:
        print(f"  ❌ recovery re-run FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"  Recovery file still preserved at: {path}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('target', nargs='?',
                   default=os.path.join(PROJECT_ROOT, 'results', 'experiments'),
                   help='exp dir, results root, or a single .pkl path')
    p.add_argument('--dry-run', action='store_true', help='list payloads only')
    args = p.parse_args()

    files = find_recovery_files(args.target)
    if not files:
        print(f"No recovery files found under: {args.target}")
        return 0

    print(f"Found {len(files)} recovery payload(s)")
    results = []
    for f in files:
        ok = recover_one(f, dry_run=args.dry_run)
        results.append((f, ok))

    if not args.dry_run:
        n_ok = sum(1 for _, ok in results if ok)
        n_fail = sum(1 for _, ok in results if ok is False)
        print(f"\n{'='*80}")
        print(f"Recovery summary: {n_ok} ✓ / {n_fail} ❌ / {len(results)} total")
    return 0 if all(ok for _, ok in results if ok is not None) else 1


if __name__ == '__main__':
    sys.exit(main())
