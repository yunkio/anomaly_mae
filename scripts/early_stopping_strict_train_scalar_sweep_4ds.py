"""Strict train-only scalar early-stopping sweep for the 4 main datasets.

This is the source-disciplined sweep requested after the initial exploratory
passes. Criterion inputs are restricted to values recorded from the training
loop and available without eval/test loader output.

Included:
  - non-excluded scalar train-loop histories from training_histories.json
  - pairwise diff/ratio/relgap/absdiff interactions among those scalars

Excluded:
  - eval/test callback metrics: epoch_*, epoch_metrics disc_snr/recon_snr,
    PRC/AUC/F1/PAK/VUS/Affiliation and score contribution histories
  - feature-related metrics: train_feature_*, train_fm_*
  - auxiliary mechanism diagnostics explicitly excluded by the user: train_grl_*,
    train_scad_*
  - timing, profiling, schedules, gradients
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations, permutations
from pathlib import Path
from statistics import mean

import numpy as np


BASE_SCRIPT = Path(__file__).with_name("early_stopping_train_metric_sweep_4ds.py")
spec = importlib.util.spec_from_file_location("early_stopping_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
assert spec.loader is not None
spec.loader.exec_module(base)


OUT_DIR = Path("temp/early_stopping_strict_train_scalar_4ds")

EXCLUDED_PREFIXES = (
    "epoch_",
    "train_feature_",
    "train_fm_",
    "train_grl_",
    "train_scad_",
)
EXCLUDED_EXACT = {
    "epoch",
    "batch_profiling",
    "epoch_timings",
    "train_fm_loss",
}

TRANSFORMS = ("raw", "ema01", "ema02", "ema03", "ema05", "ema07")
DIRECTION_MODES = ("auto", "force_max", "force_min")
RULES = ("standard", "peak_reversal")
PATIENCES = (2, 3, 5, 8, 10)
THRESHOLDS = (("abs", 0.0), ("rel", 0.005), ("rel", 0.01))
START_POLICIES = ("warmup", "post_warmup", "epoch100")
ROLLBACK = "best_seen_before_stop"


def is_excluded_key(key: str) -> bool:
    if key in EXCLUDED_EXACT:
        return True
    return any(key.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def is_candidate_train_scalar_key(key: str) -> bool:
    return key.startswith("train_") and not is_excluded_key(key)


def audit_history_keys(cells: list[dict]) -> dict:
    counts = Counter()
    nonconst = Counter()
    categories = defaultdict(list)

    for cell in cells:
        history = cell["history"]
        n_epochs = len(history.get("epoch", []))
        for key, values in history.items():
            if key in counts:
                continue
            # counted below per-cell; this branch only keeps lints quiet
            _ = values
        for key, values in history.items():
            if not isinstance(values, list) or len(values) != n_epochs:
                continue
            series = base.to_scalar_series(values, n_epochs)
            if series is None:
                continue
            counts[key] += 1
            if np.nanstd(series) > 1e-12:
                nonconst[key] += 1

    for key in sorted(counts):
        if is_candidate_train_scalar_key(key):
            categories["included_train_scalar_candidate"].append(key)
        elif key.startswith("epoch_"):
            categories["excluded_eval_test_callback"].append(key)
        elif key.startswith("train_feature_") or key.startswith("train_fm_"):
            categories["excluded_feature_related"].append(key)
        elif key.startswith("train_grl_") or key.startswith("train_scad_"):
            categories["excluded_grl_scad"].append(key)
        elif key in EXCLUDED_EXACT:
            categories["excluded_bookkeeping"].append(key)
        else:
            categories["excluded_other_train_scalar"].append(key)

    return {
        "counts": dict(counts),
        "nonconstant_counts": dict(nonconst),
        "categories": dict(categories),
    }


def build_strict_metric_bank(history: dict) -> dict[str, np.ndarray]:
    n_epochs = len(history.get("epoch", []))
    scalar: dict[str, np.ndarray] = {}
    for key in sorted(history):
        if not is_candidate_train_scalar_key(key):
            continue
        series = base.to_scalar_series(history.get(key), n_epochs)
        if series is not None and np.nanstd(series) > 1e-12:
            scalar[key] = series

    metrics = dict(scalar)
    keys = sorted(scalar)

    for a_name, b_name in permutations(keys, 2):
        a = scalar[a_name]
        b = scalar[b_name]
        if base.finite_ratio(a) < 0.5 or base.finite_ratio(b) < 0.5:
            continue
        for op in ("ratio", "relgap"):
            name = f"pair_{op}__{a_name}__{b_name}"
            series = base.pair_series(op, a, b)
            if base.finite_ratio(series) >= 0.5 and np.nanstd(series) > 1e-12:
                metrics[name] = series

    for a_name, b_name in combinations(keys, 2):
        a = scalar[a_name]
        b = scalar[b_name]
        for op in ("diff", "absdiff"):
            name = f"pair_{op}__{a_name}__{b_name}"
            series = base.pair_series(op, a, b)
            if base.finite_ratio(series) >= 0.5 and np.nanstd(series) > 1e-12:
                metrics[name] = series

    return metrics


def init_stats() -> dict:
    return {
        "n": 0,
        "score_sum": 0.0,
        "drop_sum": 0.0,
        "rel_drop_sum": 0.0,
        "stop_epoch_sum": 0.0,
        "after_warmup": 0,
    }


def init_accum_entry() -> dict:
    return {ds: init_stats() for ds in base.DATASETS}


def update_stats(stats: dict, score: float, oracle: float, stop_epoch: int, warmup: int) -> None:
    stats["n"] += 1
    stats["score_sum"] += score
    drop = oracle - score
    stats["drop_sum"] += drop
    stats["rel_drop_sum"] += 100.0 * drop / max(abs(oracle), 1e-8)
    stats["stop_epoch_sum"] += stop_epoch
    if stop_epoch >= max(1, warmup + 1):
        stats["after_warmup"] += 1


def key_tuple(
    metric: str,
    transform: str,
    direction_mode: str,
    rule: str,
    patience: int,
    threshold_type: str,
    threshold_value: float,
    start_policy: str,
) -> tuple:
    return (
        metric,
        transform,
        direction_mode,
        rule,
        patience,
        threshold_type,
        threshold_value,
        ROLLBACK,
        start_policy,
    )


def key_to_meta(key: tuple) -> dict:
    return {
        "metric": key[0],
        "transform": key[1],
        "direction_mode": key[2],
        "rule": key[3],
        "patience": key[4],
        "threshold_type": key[5],
        "threshold_value": key[6],
        "rollback": key[7],
        "start_policy": key[8],
    }


def merge_accum(dst: dict, src: dict) -> None:
    for key, ds_stats in src.items():
        if key not in dst:
            dst[key] = init_accum_entry()
        for ds in base.DATASETS:
            for field, value in ds_stats[ds].items():
                dst[key][ds][field] += value


def sweep_cells(cells: list[dict], worker_id: int = 0, progress_every: int = 20) -> dict:
    accum = {}
    for i, cell in enumerate(cells, start=1):
        metrics = build_strict_metric_bank(cell["history"])
        if progress_every and (i == 1 or i % progress_every == 0):
            print(
                f"worker {worker_id}: cell {i:3d}/{len(cells)} "
                f"{cell['dataset']:11s} metrics={len(metrics)}",
                flush=True,
            )
        for metric_name, raw_values in metrics.items():
            for transform in TRANSFORMS:
                values = base.transform_series(raw_values, transform)
                if base.finite_ratio(values) < 0.2 or np.nanstd(values) <= 1e-12:
                    continue
                for direction_mode in DIRECTION_MODES:
                    direction = base.resolve_direction(metric_name, direction_mode)
                    for start_policy in START_POLICIES:
                        start = base.start_epoch(start_policy, cell["warmup"])
                        pts = base.eval_points(values, cell["score_epochs"], start, direction)
                        if len(pts) < 3:
                            continue
                        for rule in RULES:
                            fn = base.RULE_FN[rule]
                            for patience in PATIENCES:
                                for threshold_type, threshold_value in THRESHOLDS:
                                    res = fn(pts, patience, threshold_type, threshold_value, ROLLBACK)
                                    if res is None:
                                        continue
                                    stop_epoch, _ = res
                                    score = cell["score_by_epoch"].get(stop_epoch)
                                    if score is None:
                                        continue
                                    key = key_tuple(
                                        metric_name,
                                        transform,
                                        direction_mode,
                                        rule,
                                        patience,
                                        threshold_type,
                                        threshold_value,
                                        start_policy,
                                    )
                                    if key not in accum:
                                        accum[key] = init_accum_entry()
                                    update_stats(
                                        accum[key][cell["dataset"]],
                                        score,
                                        cell["oracle_score"],
                                        stop_epoch,
                                        cell["warmup"],
                                    )
    print(f"worker {worker_id}: done criteria={len(accum)}", flush=True)
    return accum


def split_even(items: list[dict], n_chunks: int) -> list[list[dict]]:
    chunks = [[] for _ in range(max(1, n_chunks))]
    for idx, item in enumerate(items):
        chunks[idx % len(chunks)].append(item)
    return [chunk for chunk in chunks if chunk]


def run_sweep(cells: list[dict], workers: int) -> dict:
    if workers <= 1:
        return sweep_cells(cells, worker_id=0)

    chunks = split_even(cells, workers)
    accum = {}
    print(
        f"parallel strict sweep: workers={len(chunks)} chunk_sizes={[len(c) for c in chunks]}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(sweep_cells, chunk, worker_id + 1, 10)
            for worker_id, chunk in enumerate(chunks)
        ]
        for future in as_completed(futures):
            part = future.result()
            merge_accum(accum, part)
            print(f"merged partial criteria={len(part)} total={len(accum)}", flush=True)
            del part
    return accum


def rank_full_rows(rows: list[dict]) -> None:
    for ds in base.DATASETS:
        ranked = sorted(
            enumerate(rows),
            key=lambda item: item[1][f"{ds}_mean_score"],
            reverse=True,
        )
        for rank, (idx, _) in enumerate(ranked, start=1):
            rows[idx][f"{ds}_rank"] = rank
    for row in rows:
        row["avg_rank_4ds"] = mean(row[f"{ds}_rank"] for ds in base.DATASETS)


def summarize(accum: dict, n_by_dataset: dict) -> tuple[list[dict], list[dict]]:
    all_rows = []
    full_rows = []
    for key, ds_stats in accum.items():
        row = {"key": "|".join(str(x) for x in key), **key_to_meta(key)}
        total_n = 0
        full = True
        for ds in base.DATASETS:
            st = ds_stats[ds]
            n = st["n"]
            total_n += n
            row[f"{ds}_n"] = n
            if n:
                row[f"{ds}_mean_score"] = st["score_sum"] / n
                row[f"{ds}_mean_drop"] = st["drop_sum"] / n
                row[f"{ds}_mean_rel_drop_pct"] = st["rel_drop_sum"] / n
                row[f"{ds}_mean_stop_epoch"] = st["stop_epoch_sum"] / n
                row[f"{ds}_after_post_warmup_pct"] = 100.0 * st["after_warmup"] / n
            else:
                for suffix in (
                    "mean_score",
                    "mean_drop",
                    "mean_rel_drop_pct",
                    "mean_stop_epoch",
                    "after_post_warmup_pct",
                ):
                    row[f"{ds}_{suffix}"] = math.nan
            if n != n_by_dataset[ds]:
                full = False
        row["total_n"] = total_n
        row["full_coverage"] = full
        row["mean_score_4ds"] = mean(row[f"{ds}_mean_score"] for ds in base.DATASETS)
        row["mean_drop_4ds"] = mean(row[f"{ds}_mean_drop"] for ds in base.DATASETS)
        row["mean_rel_drop_pct_4ds"] = mean(row[f"{ds}_mean_rel_drop_pct"] for ds in base.DATASETS)
        row["mean_stop_epoch_4ds"] = mean(row[f"{ds}_mean_stop_epoch"] for ds in base.DATASETS)
        row["mean_after_post_warmup_pct_4ds"] = mean(
            row[f"{ds}_after_post_warmup_pct"] for ds in base.DATASETS
        )
        all_rows.append(row)
        if full:
            full_rows.append(row)

    rank_full_rows(full_rows)
    full_rows.sort(key=lambda r: (r["avg_rank_4ds"], -r["mean_score_4ds"]))
    all_rows.sort(key=lambda r: (-r["total_n"], r["mean_drop_4ds"]))
    return full_rows, all_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = base.load_cells(Path("results/experiments"))
    n_by_dataset = defaultdict(int)
    warmups = defaultdict(int)
    oracle_by_dataset = defaultdict(list)
    for cell in cells:
        n_by_dataset[cell["dataset"]] += 1
        warmups[cell["warmup"]] += 1
        oracle_by_dataset[cell["dataset"]].append(cell["oracle_score"])

    source_audit = audit_history_keys(cells)
    accum = run_sweep(cells, max(1, int(args.workers)))
    full_rows, all_rows = summarize(accum, dict(n_by_dataset))

    write_csv(OUT_DIR / "leaderboard_full_coverage.csv", full_rows)
    write_csv(OUT_DIR / "leaderboard_all_coverage.csv", all_rows)

    warmup_rows = [r for r in full_rows if r["start_policy"] == "warmup"]
    warmup_rows.sort(key=lambda r: (r["avg_rank_4ds"], -r["mean_score_4ds"]))
    write_csv(OUT_DIR / "leaderboard_warmup_full_coverage.csv", warmup_rows)

    post_warmup_rows = [r for r in full_rows if r["start_policy"] == "post_warmup"]
    post_warmup_rows.sort(key=lambda r: (r["avg_rank_4ds"], -r["mean_score_4ds"]))
    write_csv(OUT_DIR / "leaderboard_post_warmup_full_coverage.csv", post_warmup_rows)

    summary = {
        "datasets": list(base.DATASETS),
        "n_cells_by_dataset": dict(n_by_dataset),
        "warmup_distribution": dict(sorted(warmups.items())),
        "mean_oracle_by_dataset": {ds: mean(vs) for ds, vs in oracle_by_dataset.items()},
        "included_train_scalar_candidate_keys": sorted(
            source_audit["categories"].get("included_train_scalar_candidate", [])
        ),
        "excluded_prefixes": EXCLUDED_PREFIXES,
        "excluded_exact": sorted(EXCLUDED_EXACT),
        "source_audit": source_audit,
        "n_criteria_total": len(all_rows),
        "n_criteria_full_coverage": len(full_rows),
        "n_criteria_warmup_full_coverage": len(warmup_rows),
        "n_criteria_post_warmup_full_coverage": len(post_warmup_rows),
        "top_post_warmup_by_avg_rank": post_warmup_rows[:20],
        "top_post_warmup_by_mean_score": sorted(
            post_warmup_rows, key=lambda r: r["mean_score_4ds"], reverse=True
        )[:20],
        "top_full_by_avg_rank": full_rows[:20],
        "top_full_by_mean_score": sorted(full_rows, key=lambda r: r["mean_score_4ds"], reverse=True)[:20],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_cells_by_dataset": summary["n_cells_by_dataset"],
                "warmup_distribution": summary["warmup_distribution"],
                "included_train_scalar_candidate_keys": summary["included_train_scalar_candidate_keys"],
                "n_criteria_full_coverage": summary["n_criteria_full_coverage"],
                "n_criteria_post_warmup_full_coverage": summary["n_criteria_post_warmup_full_coverage"],
                "out": str(OUT_DIR),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
