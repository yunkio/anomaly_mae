"""Sweep train-metric-only early stopping rules on current 4-dataset TSMAE cells.

Inputs:
  results/experiments/*/{training_histories.json, epoch_metrics.json, best_config.json}

Scope:
  Numeric experiment dirs only (271-329 style); legacy_* is excluded.
  Datasets: SWaT_excl22, PSM, WaDi_A1, WaDi_A2.

Early-stop inputs are restricted to train-time metrics:
  - scalar train_* history keys
  - train_feature_* per-feature histories reduced by simple aggregates
  - pairwise train-metric differences/ratios/relative gaps/absolute differences

Explicitly excluded:
  - epoch_* score/contribution histories (produced by test-eval callback)
  - batch profiling / epoch timing
  - pure schedule/ramp counters that are not data-derived
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np


DATASETS = {
    "SWaT_excl22": ("SWaT/A1A2_full", "SWaT/A1A2_excl22"),
    "PSM": ("PSM", "PSM"),
    "WaDi_A1": ("WaDi/A1", "WaDi/A1"),
    "WaDi_A2": ("WaDi/A2", "WaDi/A2"),
}

EXCLUDED_EXACT_KEYS = {
    "epoch",
    "batch_profiling",
    "epoch_timings",
    "train_grl_lambda",
    "train_grl_ramp_lambda",
    "train_scad_ramp",
}

PAIR_OPS = ("diff", "ratio", "relgap", "absdiff")
FEATURE_AGGS = ("mean", "std", "min", "max", "p90", "range")
EMA_ALPHAS = {
    "ema01": 0.1,
    "ema02": 0.2,
    "ema03": 0.3,
    "ema05": 0.5,
    "ema07": 0.7,
}
TRANSFORMS = ("raw", *EMA_ALPHAS.keys())
DIRECTION_MODES = ("auto", "force_max", "force_min")
RULES = ("standard", "peak_reversal")
PATIENCES = (3,)
THRESHOLDS = (("abs", 0.0), ("rel", 0.01))
ROLLBACKS = ("best_seen_before_stop",)
START_POLICIES = ("epoch50", "epoch100", "warmup")


@dataclass(frozen=True)
class Criterion:
    metric: str
    transform: str
    direction_mode: str
    rule: str
    patience: int
    threshold_type: str
    threshold_value: float
    rollback: str
    start_policy: str

    def key(self) -> str:
        return "|".join(
            [
                self.metric,
                self.transform,
                self.direction_mode,
                self.rule,
                str(self.patience),
                self.threshold_type,
                f"{self.threshold_value:g}",
                self.rollback,
                self.start_policy,
            ]
        )


def numeric_exp_num(path: Path) -> Optional[int]:
    try:
        return int(path.name.split("_", 1)[0])
    except ValueError:
        return None


def read_history(path: Path) -> dict:
    raw = json.loads(path.read_text())
    return raw[next(iter(raw))]


def read_epoch_metrics(path: Path) -> List[dict]:
    raw = json.loads(path.read_text())
    return raw.get("epochs", [])


def is_scalar(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def finite_ratio(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.isfinite(arr).sum() / arr.size)


def to_scalar_series(values: object, n_epochs: int) -> Optional[np.ndarray]:
    if not isinstance(values, list) or len(values) != n_epochs:
        return None
    out = np.full(n_epochs, np.nan, dtype=np.float64)
    for i, v in enumerate(values):
        if v is None:
            continue
        if not is_scalar(v):
            return None
        out[i] = float(v)
    if finite_ratio(out) < 0.2:
        return None
    return out


def to_feature_agg_series(values: object, n_epochs: int, agg: str) -> Optional[np.ndarray]:
    if not isinstance(values, list) or len(values) != n_epochs:
        return None
    out = np.full(n_epochs, np.nan, dtype=np.float64)
    for i, v in enumerate(values):
        if v is None:
            continue
        if not isinstance(v, list) or not v:
            return None
        try:
            arr = np.asarray(v, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        if agg == "mean":
            out[i] = float(np.mean(arr))
        elif agg == "std":
            out[i] = float(np.std(arr))
        elif agg == "min":
            out[i] = float(np.min(arr))
        elif agg == "max":
            out[i] = float(np.max(arr))
        elif agg == "p90":
            out[i] = float(np.percentile(arr, 90))
        elif agg == "range":
            out[i] = float(np.max(arr) - np.min(arr))
        else:
            raise ValueError(agg)
    if finite_ratio(out) < 0.2:
        return None
    return out


def include_train_key(key: str) -> bool:
    if key in EXCLUDED_EXACT_KEYS:
        return False
    if key.startswith("epoch_"):
        return False
    if not key.startswith("train_"):
        return False
    return True


def auto_direction(name: str) -> str:
    n = name.lower()
    if "absdiff" in n:
        return "min"
    if any(s in n for s in ("auc", "acc", "f1", "snr", "precision", "recall")):
        return "max"
    if any(s in n for s in ("gap", "ratio", "relgap", "separation", "margin", "health")):
        return "max"
    if any(s in n for s in ("loss", "error", "recon", "disc", "discrepancy", "norm", "grad")):
        return "min"
    return "min"


def build_base_metrics(history: dict) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    n_epochs = len(history.get("epoch", []))
    scalar: Dict[str, np.ndarray] = {}
    all_metrics: Dict[str, np.ndarray] = {}
    if n_epochs <= 0:
        return scalar, all_metrics

    for key, values in history.items():
        if not include_train_key(key):
            continue
        if key.startswith("train_feature_"):
            for agg in FEATURE_AGGS:
                s = to_feature_agg_series(values, n_epochs, agg)
                if s is not None:
                    all_metrics[f"{key}__{agg}"] = s
            continue
        s = to_scalar_series(values, n_epochs)
        if s is not None:
            scalar[key] = s
            all_metrics[key] = s
    return scalar, all_metrics


def pair_series(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    eps = 1e-8
    out = np.full_like(a, np.nan, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if op == "diff":
        out[valid] = a[valid] - b[valid]
    elif op == "ratio":
        out[valid] = a[valid] / (np.abs(b[valid]) + eps)
    elif op == "relgap":
        out[valid] = (a[valid] - b[valid]) / (np.abs(a[valid]) + np.abs(b[valid]) + eps)
    elif op == "absdiff":
        out[valid] = np.abs(a[valid] - b[valid])
    else:
        raise ValueError(op)
    return out


def build_metric_bank(history: dict) -> Dict[str, np.ndarray]:
    scalar, metrics = build_base_metrics(history)
    keys = sorted(scalar)
    for i, a_name in enumerate(keys):
        for b_name in keys[i + 1 :]:
            a = scalar[a_name]
            b = scalar[b_name]
            if finite_ratio(a) < 0.5 or finite_ratio(b) < 0.5:
                continue
            for op in PAIR_OPS:
                name = f"pair_{op}__{a_name}__{b_name}"
                s = pair_series(op, a, b)
                if finite_ratio(s) >= 0.5 and np.nanstd(s) > 1e-12:
                    metrics[name] = s
    return metrics


def transform_series(series: np.ndarray, transform: str) -> np.ndarray:
    s = series.astype(np.float64, copy=True)
    if transform == "raw":
        return s
    if transform in EMA_ALPHAS:
        alpha = EMA_ALPHAS[transform]
        out = np.full_like(s, np.nan)
        prev = np.nan
        for i, v in enumerate(s):
            if not np.isfinite(v):
                out[i] = prev
                continue
            prev = v if not np.isfinite(prev) else alpha * v + (1.0 - alpha) * prev
            out[i] = prev
        return out
    if transform == "slope5":
        out = np.full_like(s, np.nan)
        for i in range(5, len(s)):
            w = s[i - 5 : i + 1]
            if np.isfinite(w).sum() >= 4:
                out[i] = float(np.nanmean(np.diff(w)))
        return out
    raise ValueError(transform)


def start_epoch(policy: str, warmup: int) -> int:
    if policy.startswith("epoch"):
        return int(policy.replace("epoch", ""))
    if policy == "warmup":
        return max(1, warmup)
    if policy == "post_warmup":
        return max(1, warmup + 1)
    raise ValueError(policy)


def improvement(new: float, old: float, ttype: str, tval: float) -> bool:
    delta = new - old
    if ttype == "abs":
        return delta > tval
    return delta / max(abs(old), 1e-8) > tval


def significant_drop(peak: float, value: float, ttype: str, tval: float) -> bool:
    drop = peak - value
    if ttype == "abs":
        return drop > tval
    return drop / max(abs(peak), 1e-8) > tval


def eval_points(
    values: np.ndarray,
    score_epochs: Iterable[int],
    start: int,
    direction: str,
) -> List[Tuple[int, float]]:
    pts = []
    sign = -1.0 if direction == "min" else 1.0
    for e in score_epochs:
        if e < start or e < 1 or e > len(values):
            continue
        v = values[e - 1]
        if np.isfinite(v):
            pts.append((e, sign * float(v)))
    return pts


def es_standard(
    pts: List[Tuple[int, float]],
    patience: int,
    ttype: str,
    tval: float,
    rollback: str,
) -> Optional[Tuple[int, int]]:
    if not pts:
        return None
    best_e, best_v = pts[0]
    counter = 0
    stop_e = pts[-1][0]
    for e, v in pts[1:]:
        if improvement(v, best_v, ttype, tval):
            best_e, best_v = e, v
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                stop_e = e
                break
    return (best_e, best_e) if rollback == "best_seen_before_stop" else (stop_e, best_e)


def es_peak_reversal(
    pts: List[Tuple[int, float]],
    patience: int,
    ttype: str,
    tval: float,
    rollback: str,
) -> Optional[Tuple[int, int]]:
    if not pts:
        return None
    peak_e, peak_v = pts[0]
    drops = 0
    stop_e = pts[-1][0]
    for e, v in pts[1:]:
        if v > peak_v:
            peak_e, peak_v = e, v
            drops = 0
        elif significant_drop(peak_v, v, ttype, tval):
            drops += 1
            if drops >= patience:
                stop_e = e
                break
        else:
            drops = 0
    return (peak_e, peak_e) if rollback == "best_seen_before_stop" else (stop_e, peak_e)


def es_first_local_extreme(
    pts: List[Tuple[int, float]],
    patience: int,
    ttype: str,
    tval: float,
    rollback: str,
) -> Optional[Tuple[int, int]]:
    if len(pts) < patience + 2:
        return None
    for i in range(1, len(pts) - patience):
        e_cur, v_cur = pts[i]
        _, v_prev = pts[i - 1]
        if v_cur <= v_prev:
            continue
        ok = True
        for j in range(1, patience + 1):
            _, v_next = pts[i + j]
            if not significant_drop(v_cur, v_next, ttype, tval):
                ok = False
                break
        if ok:
            stop_e = pts[i + patience][0]
            return (e_cur, e_cur) if rollback == "best_seen_before_stop" else (stop_e, e_cur)
    return None


RULE_FN = {
    "standard": es_standard,
    "peak_reversal": es_peak_reversal,
    "first_local_extreme": es_first_local_extreme,
}


def resolve_direction(metric: str, mode: str) -> str:
    if mode == "auto":
        return auto_direction(metric)
    if mode == "force_max":
        return "max"
    if mode == "force_min":
        return "min"
    raise ValueError(mode)


def load_cells(root: Path) -> List[dict]:
    cells = []
    for exp_dir in sorted(root.iterdir()):
        exp_num = numeric_exp_num(exp_dir)
        if exp_num is None:
            continue
        for ds_name, (train_rel, score_rel) in DATASETS.items():
            train_dir = exp_dir / train_rel
            score_dir = exp_dir / score_rel
            hist_path = train_dir / "training_histories.json"
            score_path = score_dir / "epoch_metrics.json"
            cfg_path = train_dir / "best_config.json"
            if not (hist_path.exists() and score_path.exists() and cfg_path.exists()):
                continue
            score_epochs = read_epoch_metrics(score_path)
            score_epochs = [e for e in score_epochs if e.get("pak_auc_f1") is not None]
            if not score_epochs:
                continue
            oracle = max(score_epochs, key=lambda e: e["pak_auc_f1"])
            score_by_epoch = {int(e["epoch"]): float(e["pak_auc_f1"]) for e in score_epochs}
            cfg = json.loads(cfg_path.read_text())
            cells.append(
                {
                    "exp": exp_dir.name,
                    "exp_num": exp_num,
                    "dataset": ds_name,
                    "history_path": hist_path,
                    "score_path": score_path,
                    "config_path": cfg_path,
                    "warmup": int(cfg.get("teacher_only_warmup_epochs", 0) or 0),
                    "history": read_history(hist_path),
                    "score_epochs": sorted(score_by_epoch),
                    "score_by_epoch": score_by_epoch,
                    "oracle_epoch": int(oracle["epoch"]),
                    "oracle_score": float(oracle["pak_auc_f1"]),
                }
            )
    return cells


def init_stats():
    return {
        "n": 0,
        "score_sum": 0.0,
        "drop_sum": 0.0,
        "rel_drop_sum": 0.0,
        "stop_epoch_sum": 0.0,
        "peak_epoch_sum": 0.0,
    }


def update_stats(stats: dict, score: float, oracle: float, stop_epoch: int, peak_epoch: int) -> None:
    stats["n"] += 1
    stats["score_sum"] += score
    drop = oracle - score
    stats["drop_sum"] += drop
    stats["rel_drop_sum"] += 100.0 * drop / max(abs(oracle), 1e-8)
    stats["stop_epoch_sum"] += stop_epoch
    stats["peak_epoch_sum"] += peak_epoch


def criterion_tuple(
    metric: str,
    transform: str,
    direction_mode: str,
    rule: str,
    patience: int,
    threshold_type: str,
    threshold_value: float,
    rollback: str,
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
        rollback,
        start_policy,
    )


def criterion_meta_from_tuple(c: tuple) -> dict:
    return {
        "metric": c[0],
        "transform": c[1],
        "direction_mode": c[2],
        "rule": c[3],
        "patience": c[4],
        "threshold_type": c[5],
        "threshold_value": c[6],
        "rollback": c[7],
        "start_policy": c[8],
    }


def criterion_key(c: tuple) -> str:
    return "|".join(str(x) for x in c)


def sweep(cells: List[dict]) -> Tuple[Dict[tuple, dict], Dict[tuple, dict], Dict[str, int]]:
    n_by_dataset = defaultdict(int)
    for cell in cells:
        n_by_dataset[cell["dataset"]] += 1

    accum: Dict[tuple, dict] = {}
    criterion_meta: Dict[tuple, dict] = {}
    t0 = time.time()
    for idx, cell in enumerate(cells, start=1):
        metrics = build_metric_bank(cell["history"])
        if idx == 1 or idx % 20 == 0:
            print(
                f"  cell {idx:3d}/{len(cells)} {cell['dataset']:11s} {cell['exp']}: "
                f"{len(metrics)} candidate metric series, elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
        for metric_name, raw_values in metrics.items():
            if np.nanstd(raw_values) <= 1e-12:
                continue
            for transform in TRANSFORMS:
                values = transform_series(raw_values, transform)
                if finite_ratio(values) < 0.2 or np.nanstd(values) <= 1e-12:
                    continue
                for direction_mode in DIRECTION_MODES:
                    direction = resolve_direction(metric_name, direction_mode)
                    for start_policy in START_POLICIES:
                        start = start_epoch(start_policy, cell["warmup"])
                        pts = eval_points(values, cell["score_epochs"], start, direction)
                        if len(pts) < 3:
                            continue
                        for rule in RULES:
                            fn = RULE_FN[rule]
                            for patience in PATIENCES:
                                for threshold_type, threshold_value in THRESHOLDS:
                                    for rollback in ROLLBACKS:
                                        res = fn(pts, patience, threshold_type, threshold_value, rollback)
                                        if res is None:
                                            continue
                                        stop_e, peak_e = res
                                        score = cell["score_by_epoch"].get(stop_e)
                                        if score is None:
                                            continue
                                        key = criterion_tuple(
                                            metric_name,
                                            transform,
                                            direction_mode,
                                            rule,
                                            patience,
                                            threshold_type,
                                            threshold_value,
                                            rollback,
                                            start_policy,
                                        )
                                        if key not in accum:
                                            accum[key] = {ds: init_stats() for ds in DATASETS}
                                            criterion_meta[key] = criterion_meta_from_tuple(key)
                                        update_stats(
                                            accum[key][cell["dataset"]],
                                            score,
                                            cell["oracle_score"],
                                            stop_e,
                                            peak_e,
                                        )
    return accum, criterion_meta, dict(n_by_dataset)


def summarize(
    accum: Dict[tuple, dict],
    criterion_meta: Dict[tuple, dict],
    n_by_dataset: Dict[str, int],
) -> Tuple[List[dict], List[dict]]:
    full_rows = []
    coverage_rows = []
    for key, ds_stats in accum.items():
        row = {"key": criterion_key(key), **criterion_meta[key]}
        full = True
        total_n = 0
        for ds in DATASETS:
            st = ds_stats[ds]
            n = st["n"]
            total_n += n
            row[f"{ds}_n"] = n
            if n:
                row[f"{ds}_mean_score"] = st["score_sum"] / n
                row[f"{ds}_mean_drop"] = st["drop_sum"] / n
                row[f"{ds}_mean_rel_drop_pct"] = st["rel_drop_sum"] / n
                row[f"{ds}_mean_stop_epoch"] = st["stop_epoch_sum"] / n
                row[f"{ds}_mean_peak_epoch"] = st["peak_epoch_sum"] / n
            else:
                row[f"{ds}_mean_score"] = math.nan
                row[f"{ds}_mean_drop"] = math.nan
                row[f"{ds}_mean_rel_drop_pct"] = math.nan
                row[f"{ds}_mean_stop_epoch"] = math.nan
                row[f"{ds}_mean_peak_epoch"] = math.nan
            if n != n_by_dataset[ds]:
                full = False
        row["total_n"] = total_n
        row["full_coverage"] = full
        scores = [row[f"{ds}_mean_score"] for ds in DATASETS if math.isfinite(row[f"{ds}_mean_score"])]
        drops = [row[f"{ds}_mean_drop"] for ds in DATASETS if math.isfinite(row[f"{ds}_mean_drop"])]
        rel_drops = [
            row[f"{ds}_mean_rel_drop_pct"]
            for ds in DATASETS
            if math.isfinite(row[f"{ds}_mean_rel_drop_pct"])
        ]
        row["mean_score_4ds"] = mean(scores) if scores else math.nan
        row["mean_drop_4ds"] = mean(drops) if drops else math.nan
        row["mean_rel_drop_pct_4ds"] = mean(rel_drops) if rel_drops else math.nan
        coverage_rows.append(row)
        if full:
            full_rows.append(row)

    # Dataset-wise ranks among full-coverage criteria.
    for ds in DATASETS:
        ranked = sorted(
            full_rows,
            key=lambda r: (-r[f"{ds}_mean_score"], r[f"{ds}_mean_drop"], r["key"]),
        )
        for rank, row in enumerate(ranked, start=1):
            row[f"{ds}_rank"] = rank
    for row in full_rows:
        row["avg_rank_4ds"] = mean(row[f"{ds}_rank"] for ds in DATASETS)
    full_rows.sort(key=lambda r: (r["avg_rank_4ds"], -r["mean_score_4ds"], r["mean_drop_4ds"]))

    coverage_rows.sort(
        key=lambda r: (
            -r["total_n"],
            r["mean_drop_4ds"] if math.isfinite(r["mean_drop_4ds"]) else 1e9,
            -r["mean_score_4ds"] if math.isfinite(r["mean_score_4ds"]) else 1e9,
        )
    )
    return full_rows, coverage_rows


def metric_family(metric: str) -> str:
    if metric.startswith("pair_"):
        return "pairwise derived"
    if "__" in metric and metric.startswith("train_feature_"):
        return "per-feature aggregate"
    if metric.startswith("train_grl_"):
        return "GRL train diagnostic"
    if metric.startswith("train_scad_"):
        return "SCAD train diagnostic"
    if metric.startswith("train_fm_"):
        return "feature-matching train diagnostic"
    if "teacher_recon" in metric:
        return "teacher reconstruction train loss"
    if "student_recon" in metric:
        return "student reconstruction train diagnostic"
    if metric in ("train_loss", "train_rec_loss", "train_disc_loss"):
        return "core train loss"
    if metric in ("train_normal_loss", "train_anomaly_loss", "train_mean_discrepancy"):
        return "output-discrepancy train loss"
    return "other train scalar"


def write_csv(path: Path, rows: List[dict], limit: Optional[int] = None) -> None:
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_catalog(path: Path, cells: List[dict], full_rows: List[dict], coverage_rows: List[dict]) -> None:
    families = defaultdict(int)
    for row in coverage_rows:
        families[metric_family(row["metric"])] += 1
    lines = [
        "# Train-Metric Early Stopping Metric Catalog",
        "",
        "Scope: numeric experiment dirs under `results/experiments`, datasets SWaT_excl22, PSM, WaDi_A1, WaDi_A2.",
        "",
        "Excluded from criterion inputs: `epoch_*` score/contribution histories, `batch_profiling`, `epoch_timings`, and pure schedule/ramp counters.",
        "",
        "Metric families:",
        "- core train loss: `train_loss`, `train_rec_loss`, `train_disc_loss`; direct optimization loss traces.",
        "- output-discrepancy train loss: `train_normal_loss`, `train_anomaly_loss`, `train_mean_discrepancy`; student-teacher discrepancy pressure on train batches.",
        "- teacher reconstruction train loss: `train_teacher_recon_normal/anomaly`; label-split teacher reconstruction loss on train batches.",
        "- student reconstruction train diagnostic: `train_student_recon_normal/anomaly`; tracked student reconstruction quality by train label split.",
        "- feature-matching train diagnostic: `train_fm_loss`, `train_fm_adaptive_lambda`; hidden alignment loss and adaptive scale.",
        "- GRL train diagnostic: `train_grl_*`; train-label classifier/adversarial health metrics when GRL exists.",
        "- SCAD train diagnostic: `train_scad_*`; representation repulsion and transfer diagnostics when SCAD exists.",
        "- per-feature aggregate: `train_feature_*__mean/std/min/max/p90/range`; train feature-wise recon/disc summaries reduced to scalar traces.",
        "- pairwise derived: `pair_diff/ratio/relgap/absdiff__A__B`; exhaustive pair operations over scalar train metrics available in a cell.",
        "",
        f"Cells: {len(cells)} total.",
        "",
        "Criterion counts by family:",
    ]
    for fam, count in sorted(families.items()):
        lines.append(f"- {fam}: {count}")
    lines += [
        "",
        "Top full-coverage criteria:",
    ]
    for i, row in enumerate(full_rows[:20], start=1):
        lines.append(
            f"{i}. `{row['metric']}` transform={row['transform']} dir={row['direction_mode']} "
            f"rule={row['rule']} P={row['patience']} threshold={row['threshold_type']}:{row['threshold_value']} "
            f"rollback={row['rollback']} start={row['start_policy']} "
            f"avg_rank={row['avg_rank_4ds']:.2f} mean_drop={row['mean_drop_4ds']:.4f}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/experiments")
    ap.add_argument("--out", default="temp/early_stopping_train_metrics_4ds")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cells = load_cells(root)
    if not cells:
        raise SystemExit("No cells found")

    n_by_dataset = defaultdict(int)
    oracle_by_dataset = defaultdict(list)
    for cell in cells:
        n_by_dataset[cell["dataset"]] += 1
        oracle_by_dataset[cell["dataset"]].append(cell["oracle_score"])
    print("Target cells:")
    for ds in DATASETS:
        print(
            f"  {ds:11s}: cells={n_by_dataset[ds]:3d}, "
            f"mean_oracle={mean(oracle_by_dataset[ds]):.4f}"
        )

    accum, criterion_meta, n_by_dataset = sweep(cells)
    full_rows, coverage_rows = summarize(accum, criterion_meta, n_by_dataset)

    write_csv(out / "leaderboard_full_coverage.csv", full_rows)
    write_csv(out / "leaderboard_full_coverage_top200.csv", full_rows, limit=200)
    write_csv(out / "leaderboard_all_coverage_top500.csv", coverage_rows, limit=500)
    write_catalog(out / "metric_catalog.md", cells, full_rows, coverage_rows)

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": str(root),
        "datasets": list(DATASETS),
        "n_cells_by_dataset": dict(n_by_dataset),
        "mean_oracle_by_dataset": {ds: mean(vs) for ds, vs in oracle_by_dataset.items()},
        "n_criteria_total": len(coverage_rows),
        "n_criteria_full_coverage": len(full_rows),
        "top_full_coverage": full_rows[:25],
        "top_all_coverage": coverage_rows[:25],
        "elapsed_sec": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\nTop 20 full-coverage criteria:")
    for i, row in enumerate(full_rows[:20], start=1):
        print(
            f"{i:2d}. avg_rank={row['avg_rank_4ds']:7.2f} "
            f"mean_score={row['mean_score_4ds']:.4f} mean_drop={row['mean_drop_4ds']:.4f} "
            f"rel_drop={row['mean_rel_drop_pct_4ds']:.2f}% "
            f"metric={row['metric']} op={row['transform']} dir={row['direction_mode']} "
            f"rule={row['rule']} P={row['patience']} T={row['threshold_type']}:{row['threshold_value']} "
            f"rb={row['rollback']} start={row['start_policy']}"
        )
    print(f"\nWrote outputs to {out.resolve()}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
