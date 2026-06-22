"""Summarize outputs from early_stopping_train_metric_sweep_4ds.py."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean


OUT = Path("temp/early_stopping_train_metrics_4ds")
DATASETS = ["SWaT_excl22", "PSM", "WaDi_A1", "WaDi_A2"]


def coerce(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def brief(row: dict) -> str:
    return (
        f"avg_rank={row['avg_rank_4ds']:.2f} mean={row['mean_score_4ds']:.4f} "
        f"drop={row['mean_drop_4ds']:.4f} rel={row['mean_rel_drop_pct_4ds']:.2f}% "
        f"metric={row['metric']} op={row['transform']} dir={row['direction_mode']} "
        f"rule={row['rule']} start={row['start_policy']} "
        f"ranks=({row['SWaT_excl22_rank']:.0f},{row['PSM_rank']:.0f},"
        f"{row['WaDi_A1_rank']:.0f},{row['WaDi_A2_rank']:.0f})"
    )


def fixed_epoch_baselines(summary: dict) -> list[dict]:
    # Re-read cells via the main sweep script helpers to avoid duplicating path logic.
    sys.path.insert(0, str(Path.cwd()))
    from scripts.early_stopping_train_metric_sweep_4ds import load_cells

    cells = load_cells(Path("results/experiments"))
    policies = [
        "epoch50",
        "epoch100",
        "epoch150",
        "epoch200",
        "epoch250",
        "epoch300",
        "epoch350",
        "epoch400",
        "epoch450",
        "epoch500",
        "warmup",
    ]
    rows = []
    for policy in policies:
        ds_stats = {ds: {"n": 0, "score": 0.0, "drop": 0.0, "stop": 0.0} for ds in DATASETS}
        for c in cells:
            if policy == "warmup":
                target = max(1, int(c["warmup"]))
            else:
                target = int(policy.replace("epoch", ""))
            available = [e for e in c["score_epochs"] if e >= target]
            if not available:
                continue
            stop_e = min(available)
            score = c["score_by_epoch"][stop_e]
            st = ds_stats[c["dataset"]]
            st["n"] += 1
            st["score"] += score
            st["drop"] += c["oracle_score"] - score
            st["stop"] += stop_e
        row = {"policy": policy}
        for ds, st in ds_stats.items():
            n = st["n"]
            row[f"{ds}_n"] = n
            row[f"{ds}_mean_score"] = st["score"] / n if n else None
            row[f"{ds}_mean_drop"] = st["drop"] / n if n else None
            row[f"{ds}_mean_stop_epoch"] = st["stop"] / n if n else None
        row["mean_score_4ds"] = mean(row[f"{ds}_mean_score"] for ds in DATASETS)
        row["mean_drop_4ds"] = mean(row[f"{ds}_mean_drop"] for ds in DATASETS)
        rows.append(row)
    return rows


def main() -> None:
    summary = json.loads((OUT / "summary.json").read_text())
    rows = [coerce(r) for r in csv.DictReader((OUT / "leaderboard_full_coverage.csv").open())]
    print(f"criteria total={summary['n_criteria_total']} full={summary['n_criteria_full_coverage']}")
    print(f"cells={summary['n_cells_by_dataset']}")
    print(f"mean_oracle={summary['mean_oracle_by_dataset']}")

    print("\nTOP avg-rank")
    for r in rows[:12]:
        print("  " + brief(r))

    print("\nTOP mean-score")
    for r in sorted(rows, key=lambda x: -x["mean_score_4ds"])[:12]:
        print("  " + brief(r))

    print("\nPer-dataset top")
    for ds in DATASETS:
        print(f"  {ds}")
        for r in sorted(rows, key=lambda x: x[f"{ds}_rank"])[:5]:
            print(
                f"    rank={r[f'{ds}_rank']:.0f} score={r[f'{ds}_mean_score']:.4f} "
                f"drop={r[f'{ds}_mean_drop']:.4f} stop={r[f'{ds}_mean_stop_epoch']:.1f} "
                f"avg_rank={r['avg_rank_4ds']:.2f} metric={r['metric']} "
                f"op={r['transform']} dir={r['direction_mode']} rule={r['rule']} "
                f"start={r['start_policy']}"
            )

    fixed = fixed_epoch_baselines(summary)
    print("\nFixed-epoch baselines")
    for r in fixed:
        print(f"  {r['policy']}: mean={r['mean_score_4ds']:.4f} drop={r['mean_drop_4ds']:.4f}")
        for ds in DATASETS:
            print(
                f"    {ds}: score={r[f'{ds}_mean_score']:.4f} "
                f"drop={r[f'{ds}_mean_drop']:.4f} stop={r[f'{ds}_mean_stop_epoch']:.1f}"
            )

    report = {
        "top_avg_rank": rows[:25],
        "top_mean_score": sorted(rows, key=lambda x: -x["mean_score_4ds"])[:25],
        "per_dataset_top": {
            ds: sorted(rows, key=lambda x: x[f"{ds}_rank"])[:25] for ds in DATASETS
        },
        "fixed_epoch_baselines": fixed,
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nWrote {OUT / 'analysis_summary.json'}")


if __name__ == "__main__":
    main()
