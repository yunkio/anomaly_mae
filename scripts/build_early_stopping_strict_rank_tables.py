"""Build model rank tables for strict train-only scalar ES criteria."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np


STRICT_SCRIPT = Path(__file__).with_name("early_stopping_strict_train_scalar_sweep_4ds.py")
spec = importlib.util.spec_from_file_location("strict_es", STRICT_SCRIPT)
strict = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = strict
assert spec.loader is not None
spec.loader.exec_module(strict)
base = strict.base


OUT_DIR = Path("temp/early_stopping_strict_train_scalar_4ds")
LEADERBOARD = OUT_DIR / "leaderboard_post_warmup_full_coverage.csv"
OUT_JSON = OUT_DIR / "top5_model_rank_tables.json"
OUT_MD = OUT_DIR / "top5_model_rank_tables.md"

PAPER_FRIENDLY_SPEC = {
    "metric": "pair_ratio__train_loss__train_normal_loss",
    "transform": "ema03",
    "direction_mode": "force_min",
    "rule": "standard",
    "patience": 3.0,
    "threshold_type": "rel",
    "threshold_value": 0.005,
    "rollback": "best_seen_before_stop",
    "start_policy": "post_warmup",
}
PAPER_FRIENDLY_LASAD_SPEC = {
    "metric": "pair_relgap__train_student_recon_anomaly__train_normal_loss",
    "transform": "ema01",
    "direction_mode": "auto",
    "rule": "standard",
    "patience": 2.0,
    "threshold_type": "rel",
    "threshold_value": 0.01,
    "rollback": "best_seen_before_stop",
    "start_policy": "post_warmup",
}
PAPER_FRIENDLY_EMA_TRANSFORMS = ("raw", "ema01", "ema02", "ema03", "ema05", "ema07")
PAPER_FRIENDLY_ABLATIONS = [
    {
        "label": "PF-no-EMA",
        "description": "PF without EMA smoothing: raw ratio with the same patience rule.",
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
    {
        "label": "PF-no-patience",
        "description": "PF without patience: trigger at the first non-improving evaluation point, then roll back to the best checkpoint.",
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
    {
        "label": "PF-no-EMA-no-patience",
        "description": "PF without both EMA smoothing and patience.",
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
]

TEXT_COLS = {
    "key",
    "metric",
    "transform",
    "direction_mode",
    "rule",
    "threshold_type",
    "rollback",
    "start_policy",
    "full_coverage",
}


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            for key, value in list(row.items()):
                if key not in TEXT_COLS:
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        pass
            rows.append(row)
    return rows


def canonical_signature(metric: str) -> str:
    parts = metric.split("__")
    if len(parts) == 3 and parts[0] == "pair_relgap":
        return "pair_relgap__" + "__".join(sorted(parts[1:]))
    return metric


def select_top_criteria(rows: list[dict], n: int = 5) -> list[dict]:
    best_by_family = {}
    for row in rows:
        family = canonical_signature(row["metric"])
        if family not in best_by_family or row["mean_score_4ds"] > best_by_family[family]["mean_score_4ds"]:
            best_by_family[family] = row
    return sorted(best_by_family.values(), key=lambda r: r["mean_score_4ds"], reverse=True)[:n]


def find_spec(rows: list[dict], spec: dict) -> dict:
    for row in rows:
        if all(row.get(key) == value for key, value in spec.items()):
            return row
    raise KeyError(spec)


def criterion_from_spec(spec: dict) -> dict:
    return {k: v for k, v in spec.items() if k not in {"label", "description"}}


def criterion_key(row: dict) -> str:
    return "|".join(
        [
            row["metric"],
            row["transform"],
            row["direction_mode"],
            row["rule"],
            str(int(row["patience"])),
            row["threshold_type"],
            f"{float(row['threshold_value']):g}",
            row["rollback"],
            row["start_policy"],
        ]
    )


def model_label(exp_name: str) -> str:
    parts = exp_name.split("_", 3)
    if len(parts) == 4:
        return f"{parts[0]} · {parts[3]}"
    return exp_name


def apply_criterion(cell: dict, criterion: dict, metrics: dict[str, np.ndarray]) -> dict | None:
    metric = criterion["metric"]
    if metric not in metrics:
        return None
    values = base.transform_series(metrics[metric], criterion["transform"])
    if base.finite_ratio(values) < 0.2 or np.nanstd(values) <= 1e-12:
        return None
    direction = base.resolve_direction(metric, criterion["direction_mode"])
    start = base.start_epoch(criterion["start_policy"], cell["warmup"])
    points = base.eval_points(values, cell["score_epochs"], start, direction)
    if len(points) < 3:
        return None
    result = base.RULE_FN[criterion["rule"]](
        points,
        int(criterion["patience"]),
        criterion["threshold_type"],
        float(criterion["threshold_value"]),
        criterion["rollback"],
    )
    if result is None:
        return None
    stop_epoch, peak_epoch = result
    score = cell["score_by_epoch"].get(stop_epoch)
    if score is None:
        return None
    return {
        "score": float(score),
        "epoch": int(stop_epoch),
        "peak_epoch": int(peak_epoch),
        "oracle_score": float(cell["oracle_score"]),
        "oracle_epoch": int(cell["oracle_epoch"]),
        "drop": float(cell["oracle_score"] - score),
        "after_warmup": bool(stop_epoch >= max(1, cell["warmup"] + 1)),
        "warmup": int(cell["warmup"]),
    }


def rank_dataset(entries: list[dict]) -> None:
    ranked = sorted(entries, key=lambda x: (-x["score"], x["epoch"], x["exp_num"]))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank


def build_tables(criteria: list[dict]) -> dict:
    cells = base.load_cells(Path("results/experiments"))
    numeric_exps = {
        cell["exp_num"]: {
            "exp": cell["exp"],
            "exp_num": cell["exp_num"],
            "model": model_label(cell["exp"]),
        }
        for cell in cells
    }
    metric_cache = {
        (cell["exp"], cell["dataset"]): strict.build_strict_metric_bank(cell["history"])
        for cell in cells
    }

    tables = []
    for idx, criterion in enumerate(criteria, start=1):
        by_dataset = {ds: [] for ds in base.DATASETS}
        for cell in cells:
            result = apply_criterion(cell, criterion, metric_cache[(cell["exp"], cell["dataset"])])
            if result is None:
                continue
            result.update(
                {
                    "exp": cell["exp"],
                    "exp_num": cell["exp_num"],
                    "model": model_label(cell["exp"]),
                    "dataset": cell["dataset"],
                }
            )
            by_dataset[cell["dataset"]].append(result)
        for entries in by_dataset.values():
            rank_dataset(entries)

        ds_summary = {}
        for ds, entries in by_dataset.items():
            if entries:
                ds_summary[ds] = {
                    "mean_score": mean(item["score"] for item in entries),
                    "mean_drop": mean(item["drop"] for item in entries),
                    "mean_stop_epoch": mean(item["epoch"] for item in entries),
                    "after_warmup_pct": 100.0
                    * mean(1.0 if item["after_warmup"] else 0.0 for item in entries),
                }

        exp_rows = {exp_num: dict(info, datasets={}, ranks=[]) for exp_num, info in numeric_exps.items()}
        for ds, entries in by_dataset.items():
            for item in entries:
                exp_rows[item["exp_num"]]["datasets"][ds] = {
                    "score": item["score"],
                    "rank": item["rank"],
                    "epoch": item["epoch"],
                    "drop": item["drop"],
                    "oracle_score": item["oracle_score"],
                    "oracle_epoch": item["oracle_epoch"],
                    "after_warmup": item["after_warmup"],
                    "warmup": item["warmup"],
                }
                exp_rows[item["exp_num"]]["ranks"].append(item["rank"])

        rows = []
        for row in exp_rows.values():
            ranks = row.pop("ranks")
            row["coverage"] = len(ranks)
            row["avg_rank"] = mean(ranks) if ranks else None
            rows.append(row)
        rows.sort(
            key=lambda r: (
                r["avg_rank"] is None,
                r["avg_rank"] if r["avg_rank"] is not None else 10**9,
                r["exp_num"],
            )
        )
        tables.append(
            {
                "index": idx,
                "criterion": {
                    "key": criterion_key(criterion),
                    "metric": criterion["metric"],
                    "transform": criterion["transform"],
                    "direction_mode": criterion["direction_mode"],
                    "rule": criterion["rule"],
                    "patience": int(criterion["patience"]),
                    "threshold_type": criterion["threshold_type"],
                    "threshold_value": float(criterion["threshold_value"]),
                    "rollback": criterion["rollback"],
                    "start_policy": criterion["start_policy"],
                    "mean_score_4ds": criterion.get(
                        "mean_score_4ds",
                        mean(ds_summary[ds]["mean_score"] for ds in base.DATASETS if ds in ds_summary),
                    ),
                    "mean_drop_4ds": criterion.get(
                        "mean_drop_4ds",
                        mean(ds_summary[ds]["mean_drop"] for ds in base.DATASETS if ds in ds_summary),
                    ),
                    "mean_stop_epoch_4ds": criterion.get(
                        "mean_stop_epoch_4ds",
                        mean(ds_summary[ds]["mean_stop_epoch"] for ds in base.DATASETS if ds in ds_summary),
                    ),
                    "avg_rank_4ds": criterion.get("avg_rank_4ds"),
                },
                "rows": rows,
            }
        )
    return {
        "datasets": list(base.DATASETS),
        "selection_rule": (
            "Top 5 strict post-warmup-start, full-coverage, train-only scalar criteria by "
            "4-dataset mean score after collapsing equivalent relgap(A,B)/relgap(B,A) families. "
            "Inputs exclude eval/test callback metrics, feature/FM, GRL, SCAD, and performance metrics."
        ),
        "criteria": [table["criterion"] for table in tables],
        "tables": tables,
    }


def fmt_cell(ds_value: dict | None) -> str:
    if not ds_value:
        return "-"
    return f"{ds_value['score']:.4f} (#{ds_value['rank']}, e{ds_value['epoch']})"


def write_markdown(payload: dict) -> None:
    lines = [
        "# Strict Train-Only Early-Stopping Criterion Model Rank Tables",
        "",
        payload["selection_rule"],
        "",
    ]
    extra_tables = (
        [payload["paper_friendly_lasad_table"]]
        if "paper_friendly_lasad_table" in payload
        else []
    ) + (
        [payload["paper_friendly_table"]]
        + payload.get("paper_friendly_ema_tables", [])
        + payload.get("paper_friendly_ablation_tables", [])
    )
    for table in payload["tables"] + extra_tables:
        c = table["criterion"]
        label = f"C{table['index']}" if isinstance(table["index"], int) else str(table["index"])
        lines.extend(
            [
                f"## {label}: {c['metric']}",
                "",
                (
                    f"`{c['transform']}`, `{c['direction_mode']}`, `{c['rule']}`, "
                    f"P={c['patience']}, `{c['threshold_type']}={c['threshold_value']:g}`, "
                    f"`{c['start_policy']}`"
                ),
                "",
                "| Avg rank | Model | Coverage | SWaT(excl22) | PSM | WaDi A1 | WaDi A2 |",
                "|---:|---|---:|---|---|---|---|",
            ]
        )
        for row in table["rows"]:
            avg_rank = "-" if row["avg_rank"] is None else f"{row['avg_rank']:.2f}"
            ds = row["datasets"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        avg_rank,
                        row["model"],
                        str(row["coverage"]),
                        fmt_cell(ds.get("SWaT_excl22")),
                        fmt_cell(ds.get("PSM")),
                        fmt_cell(ds.get("WaDi_A1")),
                        fmt_cell(ds.get("WaDi_A2")),
                    ]
                )
                + " |"
            )
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = load_rows(LEADERBOARD)
    criteria = select_top_criteria(rows, n=5)
    payload = build_tables(criteria)

    paper_friendly = build_tables([find_spec(rows, PAPER_FRIENDLY_SPEC)])["tables"][0]
    paper_friendly["index"] = "PF-current"
    payload["paper_friendly_criterion"] = paper_friendly["criterion"]
    payload["paper_friendly_table"] = paper_friendly

    paper_friendly_lasad = build_tables([find_spec(rows, PAPER_FRIENDLY_LASAD_SPEC)])["tables"][0]
    paper_friendly_lasad["index"] = "PF-LASAD"
    paper_friendly_lasad["description"] = (
        "Label-conditioned suppression gap aligned with the LASAD manuscript: "
        "relgap(train_student_recon_anomaly, train_normal_loss), EMA 0.1, P=2."
    )
    payload["paper_friendly_lasad_criterion"] = paper_friendly_lasad["criterion"]
    payload["paper_friendly_lasad_table"] = paper_friendly_lasad

    ema_tables = []
    for transform in PAPER_FRIENDLY_EMA_TRANSFORMS:
        spec = dict(PAPER_FRIENDLY_SPEC, transform=transform)
        table = build_tables([find_spec(rows, spec)])["tables"][0]
        table["index"] = f"PF-{transform}"
        table["description"] = f"PF with transform={transform}, same P=3 rule."
        ema_tables.append(table)
    payload["paper_friendly_ema_criteria"] = [table["criterion"] for table in ema_tables]
    payload["paper_friendly_ema_tables"] = ema_tables

    ablation_tables = []
    for ablation in PAPER_FRIENDLY_ABLATIONS:
        table = build_tables([criterion_from_spec(ablation)])["tables"][0]
        table["index"] = ablation["label"]
        table["description"] = ablation["description"]
        ablation_tables.append(table)
    payload["paper_friendly_ablation_criteria"] = [table["criterion"] for table in ablation_tables]
    payload["paper_friendly_ablation_tables"] = ablation_tables

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload)
    print(json.dumps({"criteria": payload["criteria"], "json": str(OUT_JSON), "markdown": str(OUT_MD)}, indent=2))


if __name__ == "__main__":
    main()
