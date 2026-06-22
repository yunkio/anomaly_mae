"""Summarize strict train-only scalar early-stopping sweep outputs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean


OUT = Path("temp/early_stopping_strict_train_scalar_4ds")
DATASETS = ["SWaT_excl22", "PSM", "WaDi_A1", "WaDi_A2"]


def coerce(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = value
    return out


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return [coerce(row) for row in csv.DictReader(f)]


def brief(row: dict) -> str:
    avg_rank = row.get("avg_rank_4ds")
    avg_rank_text = "-" if avg_rank is None else f"{float(avg_rank):.2f}"
    return (
        f"avg_rank={avg_rank_text} mean={row['mean_score_4ds']:.6f} "
        f"drop={row['mean_drop_4ds']:.6f} stop={row['mean_stop_epoch_4ds']:.1f} "
        f"metric={row['metric']} op={row['transform']} dir={row['direction_mode']} "
        f"rule={row['rule']} P={int(row['patience'])} "
        f"T={row['threshold_type']}:{row['threshold_value']:g}"
    )


def find_spec(rows: list[dict], spec: dict) -> dict:
    for row in rows:
        ok = True
        for key, value in spec.items():
            if row.get(key) != value:
                ok = False
                break
        if ok:
            return row
    raise KeyError(spec)


def criterion_from_rank_table(payload: dict, label: str) -> dict:
    tables = [payload["paper_friendly_table"]]
    tables.extend(payload.get("paper_friendly_ema_tables", []))
    tables.extend(payload.get("paper_friendly_ablation_tables", []))
    for table in tables:
        if table.get("index") == label:
            return table["criterion"]
    raise KeyError(label)


def fixed_epoch_baselines() -> list[dict]:
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
        for cell in cells:
            if policy == "warmup":
                target = max(1, int(cell["warmup"]))
            else:
                target = int(policy.replace("epoch", ""))
            available = [epoch for epoch in cell["score_epochs"] if epoch >= target]
            if not available:
                continue
            stop_epoch = min(available)
            score = cell["score_by_epoch"][stop_epoch]
            st = ds_stats[cell["dataset"]]
            st["n"] += 1
            st["score"] += score
            st["drop"] += cell["oracle_score"] - score
            st["stop"] += stop_epoch
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
    history_source_audit = json.loads((OUT / "history_source_audit.json").read_text())
    rows = load_rows(OUT / "leaderboard_post_warmup_full_coverage.csv")
    rank_payload = json.loads((OUT / "top5_model_rank_tables.json").read_text())
    top_mean = sorted(rows, key=lambda row: -row["mean_score_4ds"])[:25]
    top_rank = rows[:25]

    print(f"criteria full={summary['n_criteria_full_coverage']} post_warmup={summary['n_criteria_post_warmup_full_coverage']}")
    print(f"cells={summary['n_cells_by_dataset']}")
    included_keys = summary.get("included_train_scalar_candidate_keys", [])
    print(f"included={included_keys}")
    print(
        "train_anomaly_loss="
        f"{summary['source_audit']['counts'].get('train_anomaly_loss', 0)} cells, "
        f"{summary['source_audit']['nonconstant_counts'].get('train_anomaly_loss', 0)} nonconstant"
    )

    print("\nTOP mean-score")
    for idx, row in enumerate(top_mean[:12], start=1):
        print(f"  {idx:2d}. {brief(row)}")

    print("\nTOP avg-rank")
    for idx, row in enumerate(top_rank[:12], start=1):
        print(f"  {idx:2d}. {brief(row)}")

    pf_specs = [
        (
            "PF-current",
            {
                "metric": "pair_ratio__train_loss__train_normal_loss",
                "transform": "ema03",
                "direction_mode": "force_min",
                "rule": "standard",
                "patience": 3.0,
                "threshold_type": "rel",
                "threshold_value": 0.005,
                "rollback": "best_seen_before_stop",
                "start_policy": "post_warmup",
            },
        ),
        (
            "PF-raw",
            {
                "metric": "pair_ratio__train_loss__train_normal_loss",
                "transform": "raw",
                "direction_mode": "force_min",
                "rule": "standard",
                "patience": 3.0,
                "threshold_type": "rel",
                "threshold_value": 0.005,
                "rollback": "best_seen_before_stop",
                "start_policy": "post_warmup",
            },
        ),
        (
            "PF-no-patience",
            {
                "metric": "pair_ratio__train_loss__train_normal_loss",
                "transform": "ema03",
                "direction_mode": "force_min",
                "rule": "standard",
                "patience": 1.0,
                "threshold_type": "rel",
                "threshold_value": 0.005,
                "rollback": "best_seen_before_stop",
                "start_policy": "post_warmup",
            },
        ),
        (
            "PF-no-EMA-no-patience",
            {
                "metric": "pair_ratio__train_loss__train_normal_loss",
                "transform": "raw",
                "direction_mode": "force_min",
                "rule": "standard",
                "patience": 1.0,
                "threshold_type": "rel",
                "threshold_value": 0.005,
                "rollback": "best_seen_before_stop",
                "start_policy": "post_warmup",
            },
        ),
    ]
    pf_rows = []
    print("\nPF variants")
    for label, spec in pf_specs:
        try:
            row = find_spec(rows, spec)
        except KeyError:
            # P=1 no-patience variants are applied directly in the rank-table
            # builder but are outside the leaderboard sweep grid.
            row = criterion_from_rank_table(rank_payload, label)
        pf_rows.append({"label": label, **row})
        print(f"  {label}: {brief(row)}")

    fixed = fixed_epoch_baselines()
    print("\nFixed baselines")
    for row in fixed:
        print(f"  {row['policy']}: mean={row['mean_score_4ds']:.6f} drop={row['mean_drop_4ds']:.6f}")

    report = {
        "summary": summary,
        "history_source_audit": history_source_audit,
        "top_mean_score": top_mean,
        "top_avg_rank": top_rank,
        "paper_friendly_variants": pf_rows,
        "fixed_epoch_baselines": fixed,
        "recommendation": {
            "paper_friendly_keep_or_replace": "replace",
            "recommended_pf_metric": "pair_ratio__train_normal_loss__train_student_recon_normal",
            "recommended_pf_reason": (
                "It remains train-only, non-feature, non-GRL/SCAD, avoids anomaly-only instability, "
                "uses a simple ratio between normal output discrepancy and normal student reconstruction, "
                "keeps a short P=3 patience, and outperforms the previous "
                "ratio(train_loss, train_normal_loss) PF baseline."
            ),
        },
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {OUT / 'analysis_summary.json'}")


if __name__ == "__main__":
    main()
