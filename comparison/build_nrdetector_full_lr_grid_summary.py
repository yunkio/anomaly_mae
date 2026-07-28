"""Summarize the queued five-seed NRDetector-full encoder-LR grid."""

from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    REPO_ROOT
    / "comparison"
    / "results"
    / "experiments"
    / "nrdetector_full_lr_grid_5seed"
)
SEEDS = [42, 43, 40, 41, 44]
CLASSIFIER_LR = 1e-5
VARIANTS = {
    "encoder_1e-4_classifier_1e-5": {
        "encoder_lr": 1e-4,
        "directory": "encoder_lr_1e-4__classifier_lr_1e-5",
    },
    "encoder_1e-5_classifier_1e-5": {
        "encoder_lr": 1e-5,
        "directory": "encoder_lr_1e-5__classifier_lr_1e-5",
    },
}
DATASETS = {
    "PSM": Path("PSM/nrdetector_full"),
    "SWaT": Path("SWaT/A1A2_full/nrdetector_full"),
    "WaDi A1": Path("WaDi/A1/nrdetector_full"),
    "WaDi A2": Path("WaDi/A2/nrdetector_full"),
}
METRICS = {
    "PAK": "pak_auc_f1",
    "VUS-PR": "vus_pr",
    "Affiliation F1": "affiliation_f1",
    "PRC": "prc_auc",
    "VUS-ROC": "vus_roc",
    "F1-T": "f1_t",
}


def close(actual: object, expected: float) -> bool:
    return math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12)


def load_run(path: Path, expected_encoder_lr: float, expected_seed: int) -> dict:
    payload = json.loads((path / "epoch_metrics.json").read_text(encoding="utf-8"))
    rows = payload.get("epochs", payload) if isinstance(payload, dict) else payload
    if [int(row["epoch"]) for row in rows] != list(range(1, 51)):
        raise RuntimeError(f"incomplete epoch sequence: {path}")
    if not all(math.isfinite(float(row[key])) for row in rows for key in METRICS.values()):
        raise RuntimeError(f"non-finite metric: {path}")

    config = json.loads((path / "model" / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    attrs = metadata["parameters"]["all_model_attributes"]
    if metadata["model_name"] != "nrdetector_full":
        raise RuntimeError(f"model mismatch: {path}")
    if int(metadata["parameters"]["seed"]) != expected_seed:
        raise RuntimeError(f"seed mismatch: {path}")
    if not close(config["encoder_lr"], expected_encoder_lr):
        raise RuntimeError(f"encoder LR mismatch: {path}")
    if not close(config["noisy_rate"], 1.0):
        raise RuntimeError(f"noisy_rate mismatch: {path}")
    if not close(attrs["lr"], CLASSIFIER_LR):
        raise RuntimeError(f"classifier LR mismatch: {path}")

    final = rows[-1]
    best = max(rows, key=lambda row: float(row["pak_auc_f1"]))
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "final": {"epoch": 50, **{key: float(final[key]) for key in METRICS.values()}},
        "best_pak_diagnostic": {
            "epoch": int(best["epoch"]),
            **{key: float(best[key]) for key in METRICS.values()},
        },
    }


def describe(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def build_stats(runs: dict) -> dict:
    output = {}
    for selection in ("final", "best_pak_diagnostic"):
        output[selection] = {}
        for label, key in METRICS.items():
            metric_stats = {
                dataset: describe(
                    [runs[str(seed)][dataset][selection][key] for seed in SEEDS]
                )
                for dataset in DATASETS
            }
            per_seed_mean = [
                statistics.mean(
                    [runs[str(seed)][dataset][selection][key] for dataset in DATASETS]
                )
                for seed in SEEDS
            ]
            metric_stats["Four-entity mean"] = describe(per_seed_mean)
            output[selection][label] = metric_stats
    return output


def format_stat(stat: dict) -> str:
    return (
        f"{stat['mean']:.4f} / {stat['std']:.4f} / "
        f"{stat['min']:.4f} / {stat['max']:.4f}"
    )


def append_stats_table(lines: list[str], title: str, stats: dict) -> None:
    lines.extend(
        [
            f"### {title}",
            "",
            "Values are mean / sample std / min / max across five seeds.",
            "",
            "| Metric | PSM | SWaT | WaDi A1 | WaDi A2 | Four-entity mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in METRICS:
        row = stats[metric]
        lines.append(
            f"| {metric} | {format_stat(row['PSM'])} | {format_stat(row['SWaT'])} | "
            f"{format_stat(row['WaDi A1'])} | {format_stat(row['WaDi A2'])} | "
            f"{format_stat(row['Four-entity mean'])} |"
        )
    lines.append("")


def main() -> None:
    all_runs = {}
    all_stats = {}
    for variant, spec in VARIANTS.items():
        runs = {}
        for seed in SEEDS:
            runs[str(seed)] = {}
            for dataset, relative_path in DATASETS.items():
                path = RESULT_ROOT / spec["directory"] / f"seed{seed}" / relative_path
                runs[str(seed)][dataset] = load_run(path, spec["encoder_lr"], seed)
        all_runs[variant] = runs
        all_stats[variant] = build_stats(runs)

    low = "encoder_1e-5_classifier_1e-5"
    default = "encoder_1e-4_classifier_1e-5"
    delta = {
        metric: {
            dataset: (
                all_stats[low]["final"][metric][dataset]["mean"]
                - all_stats[default]["final"][metric][dataset]["mean"]
            )
            for dataset in (*DATASETS.keys(), "Four-entity mean")
        }
        for metric in METRICS
    }

    summary = {
        "model": "nrdetector_full",
        "noisy_rate": 1.0,
        "classifier_lr": CLASSIFIER_LR,
        "seeds": SEEDS,
        "datasets": list(DATASETS),
        "primary_selection": "NO_ES fixed final epoch 50",
        "diagnostic_selection": "test PAK argmax; not paper-primary",
        "variants": VARIANTS,
        "runs": all_runs,
        "stats": all_stats,
        "encoder_1e-5_minus_1e-4_final_mean": delta,
    }
    json_path = RESULT_ROOT / "summary.json"
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(json_tmp, json_path)

    lines = [
        "# NRDetector-full encoder LR grid (five seeds)",
        "",
        "> This is a non-paper full-positive-reveal PU ablation (`noisy_rate=1.0`).",
        "> Both arms use classifier LR 1e-5 and 50 epochs; only encoder LR differs.",
        "> Primary results use fixed final epoch 50. Best-PAK results are diagnostic only.",
        "",
        "## Result directories",
        "",
        "| Encoder LR | Classifier LR | Directory |",
        "|---:|---:|---|",
    ]
    for spec in VARIANTS.values():
        lines.append(
            f"| {spec['encoder_lr']:.0e} | {CLASSIFIER_LR:.0e} | "
            f"`comparison/results/experiments/nrdetector_full_lr_grid_5seed/{spec['directory']}` |"
        )
    lines.append("")

    for variant, spec in VARIANTS.items():
        label = f"encoder={spec['encoder_lr']:.0e}, classifier={CLASSIFIER_LR:.0e}"
        append_stats_table(lines, f"Primary: final epoch 50 ({label})", all_stats[variant]["final"])
        append_stats_table(
            lines,
            f"Diagnostic: best PAK epoch ({label})",
            all_stats[variant]["best_pak_diagnostic"],
        )

    lines.extend(
        [
            "## Final-epoch mean delta: encoder 1e-5 minus encoder 1e-4",
            "",
            "| Metric | PSM | SWaT | WaDi A1 | WaDi A2 | Four-entity mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric, row in delta.items():
        lines.append(
            f"| {metric} | {row['PSM']:+.4f} | {row['SWaT']:+.4f} | "
            f"{row['WaDi A1']:+.4f} | {row['WaDi A2']:+.4f} | "
            f"{row['Four-entity mean']:+.4f} |"
        )
    lines.append("")
    markdown_path = RESULT_ROOT / "results_nrdetector_full_lr_grid.md"
    markdown_tmp = markdown_path.with_suffix(".md.tmp")
    markdown_tmp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(markdown_tmp, markdown_path)
    print(f"WROTE {json_path}")
    print(f"WROTE {markdown_path}")


if __name__ == "__main__":
    main()
