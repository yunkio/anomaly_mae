"""Summarize the queued NRDetector encoder-LR=1e-5 five-seed experiment.

The paper-ranked NRDetector is a NO_ES model, so the primary comparison uses
the fixed final classifier epoch 50.  A test-PAK-best diagnostic is retained
separately and is never mixed into the primary table.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROOT = (
    ROOT / "comparison" / "results" / "experiments"
    / "nrdetector_encoder_lr_1e-5_5seed"
)
SEEDS = [42, 43, 40, 41, 44]
SEED_TO_DEFAULT_RUN = {42: 1, 43: 2, 40: 3, 41: 4, 44: 5}
DATASETS = {
    "PSM": Path("PSM/nrdetector"),
    "SWaT": Path("SWaT/A1A2_full/nrdetector"),
    "WaDi A1": Path("WaDi/A1/nrdetector"),
    "WaDi A2": Path("WaDi/A2/nrdetector"),
}
METRICS = {
    "PAK": "pak_auc_f1",
    "VUS-PR": "vus_pr",
    "Affiliation F1": "affiliation_f1",
    "PRC": "prc_auc",
    "VUS-ROC": "vus_roc",
    "F1-T": "f1_t",
}


def load_run(path: Path, expected_lr: float, expected_seed: int) -> dict:
    payload = json.loads((path / "epoch_metrics.json").read_text(encoding="utf-8"))
    rows = payload.get("epochs", payload) if isinstance(payload, dict) else payload
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, 51)):
        raise RuntimeError(f"incomplete epoch sequence: {path}: {epochs}")
    if not all(math.isfinite(float(row[key])) for row in rows for key in METRICS.values()):
        raise RuntimeError(f"non-finite metric: {path}")
    config = json.loads((path / "model" / "config.json").read_text(encoding="utf-8"))
    if not math.isclose(float(config["encoder_lr"]), expected_lr, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError(f"encoder_lr mismatch: {path}: {config.get('encoder_lr')}")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if int(metadata["parameters"]["seed"]) != expected_seed:
        raise RuntimeError(f"seed mismatch: {path}")
    final = rows[-1]
    best = max(rows, key=lambda row: float(row["pak_auc_f1"]))
    return {
        "path": str(path.relative_to(ROOT)),
        "final": {"epoch": 50, **{key: float(final[key]) for key in METRICS.values()}},
        "best_pak_diagnostic": {
            "epoch": int(best["epoch"]),
            **{key: float(best[key]) for key in METRICS.values()},
        },
    }


def default_root(seed: int) -> Path:
    index = SEED_TO_DEFAULT_RUN[seed]
    matches = sorted((ROOT / "comparison" / "results" / "experiments").glob(
        f"9-{index}_*weak_ssl"
    ))
    if len(matches) != 1:
        raise RuntimeError(f"expected one default root for seed {seed}, found {matches}")
    return matches[0]


def describe(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def build_stats(runs: dict) -> dict:
    out = {}
    for selection in ("final", "best_pak_diagnostic"):
        out[selection] = {}
        for label, key in METRICS.items():
            metric_stats = {}
            for dataset in DATASETS:
                metric_stats[dataset] = describe([
                    runs[str(seed)][dataset][selection][key] for seed in SEEDS
                ])
            per_seed_avg = [
                statistics.mean([
                    runs[str(seed)][dataset][selection][key] for dataset in DATASETS
                ])
                for seed in SEEDS
            ]
            metric_stats["Four-entity mean"] = describe(per_seed_avg)
            out[selection][label] = metric_stats
    return out


def fmt(stat: dict) -> str:
    return (
        f"{stat['mean']:.4f} / {stat['std']:.4f} / "
        f"{stat['min']:.4f} / {stat['max']:.4f}"
    )


def render_table(lines: list[str], title: str, stats: dict) -> None:
    lines.extend([
        f"### {title}",
        "",
        "각 셀: mean / sample std / min / max (n=5).",
        "",
        "| Metric | PSM | SWaT | WaDi A1 | WaDi A2 | Four-entity mean |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for metric in METRICS:
        row = stats[metric]
        lines.append(
            f"| {metric} | {fmt(row['PSM'])} | {fmt(row['SWaT'])} | "
            f"{fmt(row['WaDi A1'])} | {fmt(row['WaDi A2'])} | "
            f"{fmt(row['Four-entity mean'])} |"
        )
    lines.append("")


def main() -> None:
    candidate = {}
    default = {}
    for seed in SEEDS:
        candidate[str(seed)] = {}
        default[str(seed)] = {}
        base = default_root(seed)
        for dataset, rel in DATASETS.items():
            candidate[str(seed)][dataset] = load_run(
                CANDIDATE_ROOT / f"seed{seed}" / rel, 1e-5, seed
            )
            default[str(seed)][dataset] = load_run(base / rel, 1e-4, seed)

    candidate_stats = build_stats(candidate)
    default_stats = build_stats(default)
    delta = {}
    for metric in METRICS:
        delta[metric] = {}
        for dataset in (*DATASETS.keys(), "Four-entity mean"):
            delta[metric][dataset] = (
                candidate_stats["final"][metric][dataset]["mean"]
                - default_stats["final"][metric][dataset]["mean"]
            )

    payload = {
        "model": "nrdetector",
        "candidate_encoder_lr": 1e-5,
        "default_encoder_lr": 1e-4,
        "classifier_lr": 1e-5,
        "seeds": SEEDS,
        "datasets": list(DATASETS),
        "primary_selection": "NO_ES fixed final epoch 50",
        "diagnostic_selection": "test PAK argmax; not paper-primary",
        "candidate_runs": candidate,
        "default_runs": default,
        "candidate_stats": candidate_stats,
        "default_stats": default_stats,
        "candidate_minus_default_final_mean": delta,
    }
    out_json = CANDIDATE_ROOT / "summary.json"
    tmp_json = out_json.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_json, out_json)

    lines = [
        "# NRDetector encoder LR 1e-5: 5-seed sensitivity",
        "",
        "> 별도 진단 실험이며 `results_baseline.md`의 논문 기본 NRDetector(encoder LR 1e-4)를 자동 교체하지 않는다.",
        "> 모델은 공식 행 `nrdetector`(noisy_rate=0.4)만 사용하며 `nrdetector_full`은 제외한다.",
        "> encoder LR만 1e-5로 변경했고 classifier LR=1e-5, 50 epochs, 데이터·seed 규칙은 기본 실험과 동일하다.",
        "",
        "## 구성과 원천",
        "",
        "| Seed | Candidate root | Default comparison root |",
        "|---:|---|---|",
    ]
    for seed in SEEDS:
        lines.append(
            f"| {seed} | `comparison/results/experiments/nrdetector_encoder_lr_1e-5_5seed/seed{seed}` "
            f"| `{default_root(seed).relative_to(ROOT)}` |"
        )
    lines.append("")
    render_table(lines, "Primary: final epoch 50", candidate_stats["final"])
    render_table(lines, "Diagnostic: best PAK epoch", candidate_stats["best_pak_diagnostic"])
    lines.extend([
        "## Candidate - default LR final-epoch mean",
        "",
        "| Metric | PSM | SWaT | WaDi A1 | WaDi A2 | Four-entity mean |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for metric in METRICS:
        row = delta[metric]
        lines.append(
            f"| {metric} | {row['PSM']:+.4f} | {row['SWaT']:+.4f} | "
            f"{row['WaDi A1']:+.4f} | {row['WaDi A2']:+.4f} | "
            f"{row['Four-entity mean']:+.4f} |"
        )
    lines.append("")
    out_md = CANDIDATE_ROOT / "results_nrdetector_encoder_lr_1e-5.md"
    tmp_md = out_md.with_suffix(".md.tmp")
    tmp_md.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp_md, out_md)
    print(f"WROTE {out_json}")
    print(f"WROTE {out_md}")


if __name__ == "__main__":
    main()
