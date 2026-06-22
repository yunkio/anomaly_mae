"""Run warm-up teacher-only sweeps by scope with bounded parallelism.

Use from the repository root:

  conda run --no-capture-output -n dc_vis python -u \
    scripts/run_warmup_teacher_only_dynamic_sweep_parallel.py --workers 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


OUT_DIR = Path("temp/warmup_teacher_only_dynamic_sweep")
PROGRESS_DIR = OUT_DIR / "progress"
SCOPES = ("main4", "SMD", "MSL", "SMAP")


def read_last_jsonl(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None
    return None


def status_text(scope: str, progress_file: Path) -> str:
    item = read_last_jsonl(progress_file)
    if item is None:
        return f"{scope}: waiting"
    event = item.get("event")
    if event == "cell":
        return (
            f"{scope}: {item.get('current')}/{item.get('total_cells')} "
            f"{item.get('group')} criteria={item.get('criteria_seen')} "
            f"rss={item.get('rss_mb')}MB elapsed={item.get('elapsed_sec')}s"
        )
    if event == "done":
        return (
            f"{scope}: done cells={item.get('total_cells')} "
            f"criteria={item.get('criteria_seen')} elapsed={item.get('elapsed_sec')}s"
        )
    return f"{scope}: {event}"


def output_name(base_name: str, scope: str) -> str:
    stem, suffix = base_name.rsplit(".", 1)
    return f"{stem}_{scope}.{suffix}"


def combine_outputs(scopes: list[str]) -> None:
    summaries = {}
    top_tables = {}
    pf_tables = {}
    cells = []
    for scope in scopes:
        summary_path = OUT_DIR / output_name("summary.json", scope)
        top_path = OUT_DIR / output_name("top_tables.json", scope)
        pf_path = OUT_DIR / output_name("paper_friendly_tables.json", scope)
        cells_path = OUT_DIR / output_name("cells.json", scope)
        if not summary_path.exists():
            raise RuntimeError(f"Missing {summary_path}")
        summary = json.loads(summary_path.read_text())
        summaries[scope] = summary
        if top_path.exists():
            top_tables.update(json.loads(top_path.read_text()))
        if pf_path.exists():
            pf_tables.update(json.loads(pf_path.read_text()))
        if cells_path.exists():
            cells.extend(json.loads(cells_path.read_text()))

    first = summaries[scopes[0]]
    combined = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_cells_total": sum(item.get("n_cells_total", 0) for item in summaries.values()),
        "n_cells_by_scope": {},
        "n_by_group": {},
        "source_audit_by_scope": {},
        "oracle_by_scope": {},
        "fixed_by_scope": {},
        "top_by_scope": {},
        "paper_friendly_spec": first.get("paper_friendly_spec"),
        "paper_friendly_rows": {},
    }
    for scope, summary in summaries.items():
        combined["n_cells_by_scope"].update(summary.get("n_cells_by_scope", {}))
        combined["n_by_group"].update(summary.get("n_by_group", {}))
        combined["source_audit_by_scope"][scope] = summary.get("source_audit", {})
        combined["oracle_by_scope"].update(summary.get("oracle_by_scope", {}))
        combined["fixed_by_scope"].update(summary.get("fixed_by_scope", {}))
        combined["top_by_scope"].update(summary.get("top_by_scope", {}))
        combined["paper_friendly_rows"].update(summary.get("paper_friendly_rows", {}))

    (OUT_DIR / "summary.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False, allow_nan=True))
    (OUT_DIR / "top_tables.json").write_text(json.dumps(top_tables, indent=2, ensure_ascii=False, allow_nan=True))
    (OUT_DIR / "paper_friendly_tables.json").write_text(
        json.dumps(pf_tables, indent=2, ensure_ascii=False, allow_nan=True)
    )
    (OUT_DIR / "cells.json").write_text(json.dumps(cells, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--scopes", nargs="+", choices=SCOPES, default=list(SCOPES))
    parser.add_argument("--poll-sec", type=float, default=15.0)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    for old in PROGRESS_DIR.glob("progress_*.jsonl"):
        old.unlink()

    pending = list(args.scopes)
    running: dict[str, subprocess.Popen] = {}
    completed: list[str] = []
    failed: dict[str, int] = {}
    t0 = time.time()
    script = Path("scripts/warmup_teacher_only_dynamic_sweep.py")

    while pending or running:
        while pending and len(running) < max(1, args.workers):
            scope = pending.pop(0)
            progress_file = PROGRESS_DIR / f"progress_{scope}.jsonl"
            cmd = [
                sys.executable,
                "-u",
                str(script),
                "--scope",
                scope,
                "--progress-file",
                str(progress_file),
                "--progress-every",
                str(args.progress_every),
            ]
            proc = subprocess.Popen(cmd)
            running[scope] = proc
            print(f"[driver] started {scope} pid={proc.pid}", flush=True)

        time.sleep(args.poll_sec)
        for scope, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            running.pop(scope)
            if rc == 0:
                completed.append(scope)
                print(f"[driver] completed {scope}", flush=True)
            else:
                failed[scope] = rc
                print(f"[driver] FAILED {scope} rc={rc}", flush=True)

        statuses = [status_text(scope, PROGRESS_DIR / f"progress_{scope}.jsonl") for scope in args.scopes]
        print(f"[driver] elapsed={time.time() - t0:.1f}s | " + " | ".join(statuses), flush=True)

        if failed:
            for proc in running.values():
                proc.terminate()
            raise SystemExit(f"Scope failures: {failed}")

    combine_outputs(args.scopes)
    print(f"[driver] combined outputs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
