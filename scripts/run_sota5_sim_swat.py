#!/usr/bin/env python
"""
5 SOTA × {simulation, swat_a1a2_normalonly} = 10 tasks sequential driver.

Created 2026-05-23 for paper-faithful re-implementation verification.
Order: tfmae → timesnet → moderntcn → dcdetector → catch (light → heavy).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = "/home/ykio/anaconda3/envs/dc_vis/bin/python"

MODELS = ["tfmae", "timesnet", "moderntcn", "dcdetector", "catch"]
DATASETS = ["simulation", "swat_a1a2_normalonly"]

OUTPUT_BASE = PROJECT_ROOT / "comparison/results/experiments/3_20260312_203923_baseline_minmax_normalonly"


def run_one(experiment_key, model, log_path):
    cmd = [
        PY, "comparison/run_baseline.py",
        "--experiment", experiment_key,
        "--model", model,
        "--output-base", str(OUTPUT_BASE),
        "--normalize-mode", "minmax",
        "--eval-interval", "1",
        "--sota-epochs", "10",
    ]
    print(f"  >> {experiment_key} × {model}", flush=True)
    start = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as f:
        f.write(f"\n=========== {datetime.now().isoformat()}  {experiment_key} × {model} ===========\n".encode())
        f.flush()
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode, time.time() - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="/tmp/0523_sota5_sim_swat")
    p.add_argument("--start-from-model", default=None, help="resume: skip until this model")
    p.add_argument("--start-from-dataset", default=None, help="resume: skip until this dataset")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    skip_m = args.start_from_model
    skip_d = args.start_from_dataset

    n_ok, n_fail = 0, 0
    fails = []
    overall = time.time()

    for model in MODELS:
        if skip_m is not None:
            if model != skip_m:
                print(f"  -- skip {model}")
                continue
            skip_m = None

        for ds in DATASETS:
            if skip_d is not None:
                if ds != skip_d:
                    print(f"  -- skip {ds}")
                    continue
                skip_d = None

            log_path = log_dir / f"{model}_{ds}.log"
            rc, elapsed = run_one(ds, model, log_path)
            if rc == 0:
                n_ok += 1
                print(f"  OK  {model} × {ds}  ({elapsed:.1f}s)")
            else:
                n_fail += 1
                fails.append((model, ds, rc, elapsed))
                print(f"  FAIL {model} × {ds}  rc={rc}  log={log_path}")

    print("\n" + "=" * 60)
    print(f"DONE  ok={n_ok}  fail={n_fail}  total_time={time.time() - overall:.0f}s")
    for m, d, rc, t in fails:
        print(f"  FAIL: {m} × {d}  rc={rc}  ({t:.1f}s)")

    summary = log_dir / "summary.json"
    json.dump({
        "n_ok": n_ok, "n_fail": n_fail,
        "fails": [{"model": m, "dataset": d, "rc": rc, "elapsed": t} for m, d, rc, t in fails],
        "completed_at": datetime.now().isoformat(),
        "total_time": time.time() - overall,
    }, open(summary, "w"), indent=2)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
