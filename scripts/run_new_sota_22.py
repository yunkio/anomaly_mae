#!/usr/bin/env python
"""
Driver: run the 7 newly-added SOTA baselines (TFMAE/NPSR/TimesNet/DCdetector/MEMTO/
ModernTCN/CATCH) on every dataset of a baseline queue.

Usage:
    conda activate dc_vis
    # Q3 (minmax + normalonly)
    python scripts/run_new_sota_22.py --condition q3
    # Q1 (minmax full)
    python scripts/run_new_sota_22.py --condition q1
    # Resume from a specific (dataset, model) pair
    python scripts/run_new_sota_22.py --condition q3 --start-from-dataset simulation_sim_normalonly --start-from-model memto
    # Dry run
    python scripts/run_new_sota_22.py --condition q3 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = "/home/ykio/anaconda3/envs/dc_vis/bin/python"

# 7 newly-added SOTA models for active experimentation
NEW_SOTA_MODELS = [
    "tfmae",
    "npsr",
    "timesnet",
    "dcdetector",
    "memto",
    "moderntcn",
    "catch",
]

CONDITION_QUEUES = {
    "q1": {
        "queue_file": PROJECT_ROOT / "comparison" / "configs" / "baseline_queue_q1_minmax.json",
        "output_base": PROJECT_ROOT / "comparison" / "results" / "experiments" / "1_20260312_041500_baseline_minmax",
        "normalize_mode": "minmax",
    },
    "q3": {
        "queue_file": PROJECT_ROOT / "comparison" / "configs" / "baseline_queue_q3_minmax_normalonly.json",
        "output_base": PROJECT_ROOT / "comparison" / "results" / "experiments" / "3_20260312_203923_baseline_minmax_normalonly",
        "normalize_mode": "minmax",
    },
}


def load_queue(queue_file: Path) -> list[dict]:
    return json.load(open(queue_file))["experiments"]


def run_one(experiment_key: str, model: str, output_base: Path, normalize_mode: str,
            sota_epochs: int = 10, eval_interval: int = 1, log_path: Path | None = None) -> tuple[int, float]:
    cmd = [
        PY, "comparison/run_baseline.py",
        "--experiment", experiment_key,
        "--model", model,
        "--output-base", str(output_base),
        "--normalize-mode", normalize_mode,
        "--eval-interval", str(eval_interval),
        "--sota-epochs", str(sota_epochs),
    ]
    print(f"  >> {' '.join(cmd[2:])}", flush=True)
    start = time.time()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as f:
            f.write(f"\n=========== {datetime.now().isoformat()}  {experiment_key} {model} ===========\n".encode())
            f.flush()
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=f, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start
    return proc.returncode, elapsed


def main() -> int:
    p = argparse.ArgumentParser(description="Run the 7 new SOTA baselines across a queue")
    p.add_argument("--condition", required=True, choices=["q1", "q3"],
                   help="Which baseline queue/output to use")
    p.add_argument("--models", nargs="+", default=NEW_SOTA_MODELS,
                   help=f"Override model list (default: {NEW_SOTA_MODELS})")
    p.add_argument("--start-from-dataset", default=None,
                   help="Resume: skip experiments until this dataset key")
    p.add_argument("--start-from-model", default=None,
                   help="Resume within --start-from-dataset: skip models until this one")
    p.add_argument("--sota-epochs", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-dir", default="/tmp/0522_baseline_logs")
    args = p.parse_args()

    cfg = CONDITION_QUEUES[args.condition]
    queue = load_queue(cfg["queue_file"])
    output_base = cfg["output_base"]
    normalize_mode = cfg["normalize_mode"]
    log_dir = Path(args.log_dir) / args.condition

    print(f"[plan] condition={args.condition}  queue={cfg['queue_file'].name}  "
          f"datasets={len(queue)}  models={len(args.models)}  total_tasks={len(queue) * len(args.models)}")
    print(f"[plan] output_base={output_base}")
    print(f"[plan] models={args.models}")
    print(f"[plan] log_dir={log_dir}")

    skip_until_ds = args.start_from_dataset
    skip_until_model_in_first_ds = args.start_from_model

    n_ok, n_fail = 0, 0
    fails: list[tuple[str, str, int, float]] = []
    overall_start = time.time()

    for ds_idx, exp in enumerate(queue):
        ds_key = exp["experiment"]
        if skip_until_ds is not None:
            if ds_key != skip_until_ds:
                continue
            skip_until_ds = None
        ds_start = time.time()
        print(f"\n[{ds_idx+1}/{len(queue)}] dataset={ds_key}")

        for m_idx, model in enumerate(args.models):
            if skip_until_model_in_first_ds is not None:
                if model != skip_until_model_in_first_ds:
                    print(f"  -- skipping {model} (resume)")
                    continue
                skip_until_model_in_first_ds = None

            if args.dry_run:
                print(f"  [DRY] {ds_key} × {model}")
                continue

            log_path = log_dir / f"{ds_key}.log"
            rc, elapsed = run_one(
                experiment_key=ds_key,
                model=model,
                output_base=output_base,
                normalize_mode=normalize_mode,
                sota_epochs=args.sota_epochs,
                eval_interval=args.eval_interval,
                log_path=log_path,
            )
            if rc == 0:
                n_ok += 1
                print(f"  OK  {ds_key} × {model}  ({elapsed:.1f}s)")
            else:
                n_fail += 1
                fails.append((ds_key, model, rc, elapsed))
                print(f"  FAIL {ds_key} × {model}  rc={rc}  ({elapsed:.1f}s)  log={log_path}")

        print(f"  dataset {ds_key} took {time.time() - ds_start:.1f}s   "
              f"cumulative ok={n_ok} fail={n_fail}  total elapsed={time.time() - overall_start:.0f}s")

    print("\n" + "=" * 60)
    print(f"DONE  ok={n_ok}  fail={n_fail}  total_time={time.time() - overall_start:.0f}s")
    if fails:
        print("Failures:")
        for ds, m, rc, t in fails:
            print(f"  {ds} × {m}  rc={rc}  ({t:.1f}s)")
    summary_path = log_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "condition": args.condition,
            "n_ok": n_ok,
            "n_fail": n_fail,
            "fails": [
                {"dataset": ds, "model": m, "rc": rc, "elapsed": t}
                for ds, m, rc, t in fails
            ],
            "total_time": time.time() - overall_start,
            "models": args.models,
            "completed_at": datetime.now().isoformat(),
        },
        open(summary_path, "w"),
        indent=2,
    )
    print(f"Summary written to {summary_path}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
