"""Train-only dynamic stopping sweep for the teacher-only warm-up phase.

This analysis is separate from post-warmup early stopping.  It asks:

  Given only metrics available from the training split during teacher-only
  warm-up, which criterion can choose a warm-up end epoch close to the
  teacher-only oracle epoch?

Post-hoc teacher-only performance is read from ``teacher_pak_auc_f1`` where
available.  That score is used only for evaluation/ranking of a criterion; it is
never used as an early-stopping input.
"""

from __future__ import annotations

import csv
import argparse
import importlib.util
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np


BASE_SCRIPT = Path(__file__).with_name("early_stopping_train_metric_sweep_4ds.py")
spec = importlib.util.spec_from_file_location("early_stopping_base_for_warmup", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
assert spec.loader is not None
spec.loader.exec_module(base)


ROOT = Path("results/experiments")
OUT_DIR = Path("temp/warmup_teacher_only_dynamic_sweep")

MAIN_DATASETS = {
    "SWaT_excl22": ("SWaT/A1A2_full", "SWaT/A1A2_excl22"),
    "PSM": ("PSM", "PSM"),
    "WaDi_A1": ("WaDi/A1", "WaDi/A1"),
    "WaDi_A2": ("WaDi/A2", "WaDi/A2"),
}
FAMILY_NAMES = ("SMD", "MSL", "SMAP")

INTUITIVE_KEYS = (
    "train_loss",
    "train_rec_loss",
    "train_disc_loss",
    "train_normal_loss",
    "train_anomaly_loss",
    "train_anomaly_disc_forward",
    "train_mean_discrepancy",
    "train_teacher_recon_normal",
    "train_teacher_recon_anomaly",
    "train_student_recon_normal",
    "train_student_recon_anomaly",
    "train_recon_snr",
)

TRANSFORMS = ("raw", "ema01", "ema02", "ema03", "ema05", "ema07")
DIRECTION_MODES = ("auto", "force_max", "force_min")
RULES = ("standard", "peak_reversal")
PATIENCES = (2, 3, 5, 8, 10)
THRESHOLDS = (("abs", 0.0), ("rel", 0.005), ("rel", 0.01))
START_POLICIES = ("epoch5", "epoch10", "epoch20", "epoch50")
ROLLBACK = "best_seen_before_stop"
FIXED_EPOCHS = (50, 100, 150, 200, 250, 300)
EPS = 1e-8

PAPER_FRIENDLY_SPEC = {
    "metric": "pair_ratio__train_teacher_recon_anomaly__train_teacher_recon_normal",
    "transform": "ema01",
    "direction_mode": "force_max",
    "rule": "peak_reversal",
    "patience": 8,
    "threshold_type": "rel",
    "threshold_value": 0.01,
    "rollback": ROLLBACK,
    "start_policy": "epoch20",
}


@dataclass(frozen=True)
class Cell:
    exp: str
    exp_num: int
    family: str
    group: str
    entity: str
    train_rel: str
    score_rel: str
    history_path: str
    score_path: str
    config_path: str
    warmup: int
    history: dict
    score_epochs: tuple[int, ...]
    score_by_epoch: dict[int, float]
    oracle_epoch: int
    oracle_score: float
    fixed_warmup_epoch: int
    fixed_warmup_score: float


def numeric_exp_num(path: Path) -> int | None:
    match = re.match(r"^(\d+)_", path.name)
    return int(match.group(1)) if match else None


def read_history(path: Path) -> dict:
    raw = json.loads(path.read_text())
    return raw[next(iter(raw))]


def read_epochs(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    return raw.get("epochs", [])


def finite_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def teacher_score(row: dict) -> float | None:
    # Backfilled rows store recon-only teacher performance here.  For older rows
    # without this field, pre-warmup pak_auc_f1 is already recon-only after the
    # same gate, so it is an acceptable fallback for post-hoc evaluation.
    return finite_float(row.get("teacher_pak_auc_f1")) or finite_float(row.get("pak_auc_f1"))


def load_config(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def resolved_warmup(cfg: dict) -> int:
    warmup = int(cfg.get("teacher_only_warmup_epochs", 0) or 0)
    if warmup < 0:
        warmup = int(cfg.get("num_epochs", 500) or 500) // 2
    return warmup


def nearest_score_epoch(score_by_epoch: dict[int, float], target: int) -> tuple[int, float] | None:
    candidates = [epoch for epoch in score_by_epoch if epoch <= target]
    if not candidates:
        return None
    epoch = max(candidates)
    return epoch, score_by_epoch[epoch]


def make_cell(
    exp_dir: Path,
    family: str,
    group: str,
    entity: str,
    train_rel: str,
    score_rel: str,
) -> Cell | None:
    exp_num = numeric_exp_num(exp_dir)
    if exp_num is None or exp_num < 271:
        return None
    train_dir = exp_dir / train_rel
    score_dir = exp_dir / score_rel
    history_path = train_dir / "training_histories.json"
    score_path = score_dir / "epoch_metrics.json"
    config_path = train_dir / "best_config.json"
    if not (history_path.exists() and score_path.exists() and config_path.exists()):
        return None
    cfg = load_config(config_path)
    warmup = resolved_warmup(cfg)
    if warmup <= 0:
        return None
    score_by_epoch: dict[int, float] = {}
    for row in read_epochs(score_path):
        epoch = int(row.get("epoch", -1))
        if epoch <= 0 or epoch > warmup:
            continue
        score = teacher_score(row)
        if score is not None:
            score_by_epoch[epoch] = score
    if len(score_by_epoch) < 3:
        return None
    oracle_epoch, oracle_score = max(score_by_epoch.items(), key=lambda item: item[1])
    fixed = nearest_score_epoch(score_by_epoch, warmup)
    if fixed is None:
        return None
    fixed_epoch, fixed_score = fixed
    return Cell(
        exp=exp_dir.name,
        exp_num=exp_num,
        family=family,
        group=group,
        entity=entity,
        train_rel=train_rel,
        score_rel=score_rel,
        history_path=str(history_path),
        score_path=str(score_path),
        config_path=str(config_path),
        warmup=warmup,
        history=read_history(history_path),
        score_epochs=tuple(sorted(score_by_epoch)),
        score_by_epoch=score_by_epoch,
        oracle_epoch=int(oracle_epoch),
        oracle_score=float(oracle_score),
        fixed_warmup_epoch=int(fixed_epoch),
        fixed_warmup_score=float(fixed_score),
    )


def load_cells(root: Path, scope_filter: str = "all") -> list[Cell]:
    cells: list[Cell] = []
    seen: set[tuple[str, str, str]] = set()
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_num = numeric_exp_num(exp_dir)
        if exp_num is None or exp_num < 271:
            continue
        if scope_filter in ("all", "main4"):
            for ds, (train_rel, score_rel) in MAIN_DATASETS.items():
                cell = make_cell(exp_dir, "main4", ds, ds, train_rel, score_rel)
                if cell is not None:
                    key = (cell.exp, cell.family, cell.group)
                    if key not in seen:
                        cells.append(cell)
                        seen.add(key)
        for fam in FAMILY_NAMES:
            if scope_filter not in ("all", fam):
                continue
            fam_dir = exp_dir / fam
            if not fam_dir.is_dir():
                continue
            for ent_dir in sorted(p for p in fam_dir.iterdir() if p.is_dir()):
                rel = f"{fam}/{ent_dir.name}"
                cell = make_cell(exp_dir, fam, f"{fam}/{ent_dir.name}", ent_dir.name, rel, rel)
                if cell is not None:
                    key = (cell.exp, cell.family, cell.group)
                    if key not in seen:
                        cells.append(cell)
                        seen.add(key)
    return cells


def warmup_mask(series: np.ndarray, warmup: int) -> np.ndarray:
    end = min(len(series), warmup)
    out = np.full_like(series, np.nan, dtype=np.float64)
    out[:end] = series[:end]
    return out


def add_metric(metrics: dict[str, np.ndarray], name: str, series: np.ndarray, warmup: int) -> None:
    s = warmup_mask(series, warmup)
    if base.finite_ratio(s[:warmup]) >= 0.5 and np.nanstd(s[:warmup]) > 1e-12:
        metrics[name] = s


def build_metric_bank(history: dict, warmup: int) -> dict[str, np.ndarray]:
    n_epochs = len(history.get("epoch", []))
    scalar: dict[str, np.ndarray] = {}
    metrics: dict[str, np.ndarray] = {}
    if n_epochs <= 0:
        return metrics

    for key in INTUITIVE_KEYS:
        values = history.get(key)
        series = base.to_scalar_series(values, n_epochs)
        if series is None:
            continue
        s = warmup_mask(series, warmup)
        if base.finite_ratio(s[:warmup]) >= 0.5 and np.nanstd(s[:warmup]) > 1e-12:
            scalar[key] = s
            metrics[key] = s

    keys = sorted(scalar)
    for a_name, b_name in permutations(keys, 2):
        a = scalar[a_name]
        b = scalar[b_name]
        if base.finite_ratio(a[:warmup]) < 0.5 or base.finite_ratio(b[:warmup]) < 0.5:
            continue
        for op in ("ratio", "relgap"):
            series = base.pair_series(op, a, b)
            add_metric(metrics, f"pair_{op}__{a_name}__{b_name}", series, warmup)

    for a_name, b_name in combinations(keys, 2):
        a = scalar[a_name]
        b = scalar[b_name]
        for op in ("diff", "absdiff"):
            series = base.pair_series(op, a, b)
            add_metric(metrics, f"pair_{op}__{a_name}__{b_name}", series, warmup)

    if "train_teacher_recon_anomaly" in scalar and "train_teacher_recon_normal" in scalar:
        a = scalar["train_teacher_recon_anomaly"]
        n = scalar["train_teacher_recon_normal"]
        add_metric(metrics, "teacher_recon_gap__anomaly__normal", a - n, warmup)
        add_metric(metrics, "teacher_recon_ratio__anomaly__normal", a / (np.abs(n) + EPS), warmup)
        add_metric(metrics, "teacher_recon_relgap__anomaly__normal", base.pair_series("relgap", a, n), warmup)
    if "train_rec_loss" in scalar and "train_teacher_recon_normal" in scalar:
        r = scalar["train_rec_loss"]
        n = scalar["train_teacher_recon_normal"]
        add_metric(metrics, "normal_overall_recon_relgap__rec__normal", base.pair_series("relgap", r, n), warmup)
    return metrics


def pretty_metric(metric: str) -> str:
    if metric == "teacher_recon_relgap__anomaly__normal":
        return "relgap(train_teacher_recon_anomaly, train_teacher_recon_normal)"
    if metric == "teacher_recon_ratio__anomaly__normal":
        return "ratio(train_teacher_recon_anomaly, train_teacher_recon_normal)"
    if metric == "teacher_recon_gap__anomaly__normal":
        return "diff(train_teacher_recon_anomaly, train_teacher_recon_normal)"
    for prefix, label in (
        ("pair_ratio__", "ratio"),
        ("pair_relgap__", "relgap"),
        ("pair_diff__", "diff"),
        ("pair_absdiff__", "absdiff"),
    ):
        if metric.startswith(prefix):
            a, b = metric.removeprefix(prefix).split("__", 1)
            return f"{label}({a}, {b})"
    return metric


def auto_direction(metric: str) -> str:
    m = metric.lower()
    if metric in {
        "train_teacher_recon_anomaly",
        "teacher_recon_gap__anomaly__normal",
        "teacher_recon_ratio__anomaly__normal",
        "teacher_recon_relgap__anomaly__normal",
    }:
        return "max"
    if any(s in m for s in ("relgap", "ratio", "gap", "separation", "snr")):
        return "max"
    if "absdiff" in m:
        return "min"
    if any(s in m for s in ("loss", "recon", "discrepancy", "disc")):
        return "min"
    return "min"


def resolve_direction(metric: str, mode: str) -> str:
    if mode == "auto":
        return auto_direction(metric)
    if mode == "force_max":
        return "max"
    if mode == "force_min":
        return "min"
    raise ValueError(mode)


def start_epoch(policy: str) -> int:
    if policy.startswith("epoch"):
        return int(policy.removeprefix("epoch"))
    raise ValueError(policy)


def improvement(new: float, old: float, threshold_type: str, threshold_value: float) -> bool:
    delta = new - old
    if threshold_type == "abs":
        return delta > threshold_value
    return delta / max(abs(old), EPS) > threshold_value


def significant_drop(peak: float, value: float, threshold_type: str, threshold_value: float) -> bool:
    drop = peak - value
    if threshold_type == "abs":
        return drop > threshold_value
    return drop / max(abs(peak), EPS) > threshold_value


def eval_points(values: np.ndarray, score_epochs: Iterable[int], start: int, direction: str) -> list[tuple[int, float]]:
    sign = -1.0 if direction == "min" else 1.0
    pts = []
    for epoch in score_epochs:
        if epoch < start or epoch < 1 or epoch > len(values):
            continue
        value = values[epoch - 1]
        if np.isfinite(value):
            pts.append((int(epoch), sign * float(value)))
    return pts


def stop_standard(
    pts: list[tuple[int, float]],
    patience: int,
    threshold_type: str,
    threshold_value: float,
) -> tuple[int, int] | None:
    if not pts:
        return None
    best_epoch, best_value = pts[0]
    trigger_epoch = pts[-1][0]
    counter = 0
    for epoch, value in pts[1:]:
        if improvement(value, best_value, threshold_type, threshold_value):
            best_epoch, best_value = epoch, value
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                trigger_epoch = epoch
                break
    return best_epoch, trigger_epoch


def stop_peak_reversal(
    pts: list[tuple[int, float]],
    patience: int,
    threshold_type: str,
    threshold_value: float,
) -> tuple[int, int] | None:
    if not pts:
        return None
    peak_epoch, peak_value = pts[0]
    trigger_epoch = pts[-1][0]
    drops = 0
    for epoch, value in pts[1:]:
        if value > peak_value:
            peak_epoch, peak_value = epoch, value
            drops = 0
        elif significant_drop(peak_value, value, threshold_type, threshold_value):
            drops += 1
            if drops >= patience:
                trigger_epoch = epoch
                break
        else:
            drops = 0
    return peak_epoch, trigger_epoch


def stop_rule(
    rule: str,
    pts: list[tuple[int, float]],
    patience: int,
    threshold_type: str,
    threshold_value: float,
) -> tuple[int, int] | None:
    if rule == "standard":
        return stop_standard(pts, patience, threshold_type, threshold_value)
    if rule == "peak_reversal":
        return stop_peak_reversal(pts, patience, threshold_type, threshold_value)
    raise ValueError(rule)


def criterion_tuple(
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


def criterion_meta(key: tuple) -> dict:
    return {
        "metric": key[0],
        "metric_pretty": pretty_metric(key[0]),
        "transform": key[1],
        "direction_mode": key[2],
        "rule": key[3],
        "patience": key[4],
        "threshold_type": key[5],
        "threshold_value": key[6],
        "rollback": key[7],
        "start_policy": key[8],
    }


def key_from_spec(spec_dict: dict) -> tuple:
    return criterion_tuple(
        spec_dict["metric"],
        spec_dict["transform"],
        spec_dict["direction_mode"],
        spec_dict["rule"],
        int(spec_dict["patience"]),
        spec_dict["threshold_type"],
        float(spec_dict["threshold_value"]),
        spec_dict["start_policy"],
    )


def init_stats() -> dict:
    return {
        "n": 0,
        "score_sum": 0.0,
        "drop_sum": 0.0,
        "rel_drop_sum": 0.0,
        "selected_epoch_sum": 0.0,
        "trigger_epoch_sum": 0.0,
        "fixed_gap_sum": 0.0,
    }


def update_stats(stats: dict, cell: Cell, selected_epoch: int, trigger_epoch: int, score: float) -> None:
    stats["n"] += 1
    stats["score_sum"] += score
    drop = cell.oracle_score - score
    stats["drop_sum"] += drop
    stats["rel_drop_sum"] += 100.0 * drop / max(abs(cell.oracle_score), EPS)
    stats["selected_epoch_sum"] += selected_epoch
    stats["trigger_epoch_sum"] += trigger_epoch
    stats["fixed_gap_sum"] += score - cell.fixed_warmup_score


def rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return math.nan


def append_progress(progress_file: Path | None, payload: dict) -> None:
    if progress_file is None:
        return
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with progress_file.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sweep(
    cells: list[Cell],
    *,
    scope: str,
    progress_file: Path | None = None,
    progress_every: int = 10,
) -> dict[tuple, dict[str, dict]]:
    accum: dict[tuple, dict[str, dict]] = {}
    t0 = time.time()
    append_progress(
        progress_file,
        {
            "event": "start",
            "scope": scope,
            "total_cells": len(cells),
            "elapsed_sec": 0.0,
            "rss_mb": rss_mb(),
        },
    )
    for idx, cell in enumerate(cells, start=1):
        metrics = build_metric_bank(cell.history, cell.warmup)
        if idx == 1 or idx % progress_every == 0 or idx == len(cells):
            payload = {
                "event": "cell",
                "scope": scope,
                "current": idx,
                "total_cells": len(cells),
                "group": cell.group,
                "exp": cell.exp,
                "metric_series": len(metrics),
                "criteria_seen": len(accum),
                "elapsed_sec": round(time.time() - t0, 1),
                "rss_mb": round(rss_mb(), 1),
            }
            append_progress(progress_file, payload)
            print(
                f"[{scope}] cell {idx:4d}/{len(cells)} {cell.group:20s} "
                f"metrics={len(metrics):3d} criteria={len(accum):7d} "
                f"rss={payload['rss_mb']:.1f}MB elapsed={payload['elapsed_sec']:.1f}s",
                flush=True,
            )
        for metric, raw_values in metrics.items():
            for transform in TRANSFORMS:
                values = base.transform_series(raw_values, transform)
                if base.finite_ratio(values[: cell.warmup]) < 0.5 or np.nanstd(values[: cell.warmup]) <= 1e-12:
                    continue
                for direction_mode in DIRECTION_MODES:
                    direction = resolve_direction(metric, direction_mode)
                    for start_policy in START_POLICIES:
                        pts = eval_points(values, cell.score_epochs, start_epoch(start_policy), direction)
                        if len(pts) < 3:
                            continue
                        for rule in RULES:
                            for patience in PATIENCES:
                                for threshold_type, threshold_value in THRESHOLDS:
                                    res = stop_rule(rule, pts, patience, threshold_type, threshold_value)
                                    if res is None:
                                        continue
                                    selected_epoch, trigger_epoch = res
                                    score = cell.score_by_epoch.get(selected_epoch)
                                    if score is None:
                                        continue
                                    key = criterion_tuple(
                                        metric,
                                        transform,
                                        direction_mode,
                                        rule,
                                        patience,
                                        threshold_type,
                                        threshold_value,
                                        start_policy,
                                    )
                                    if key not in accum:
                                        accum[key] = {}
                                    if cell.group not in accum[key]:
                                        accum[key][cell.group] = init_stats()
                                    update_stats(accum[key][cell.group], cell, selected_epoch, trigger_epoch, score)
    append_progress(
        progress_file,
        {
            "event": "done",
            "scope": scope,
            "total_cells": len(cells),
            "criteria_seen": len(accum),
            "elapsed_sec": round(time.time() - t0, 1),
            "rss_mb": round(rss_mb(), 1),
        },
    )
    return accum


def summarize_scope(
    accum: dict[tuple, dict[str, dict]],
    groups: list[str],
    n_by_group: dict[str, int],
    scope_name: str,
) -> list[dict]:
    rows: list[dict] = []
    for key, stats_by_group in accum.items():
        full = True
        row = {"key": "|".join(str(x) for x in key), "scope": scope_name, **criterion_meta(key)}
        scores = []
        drops = []
        rel_drops = []
        selected_epochs = []
        trigger_epochs = []
        fixed_gaps = []
        total_n = 0
        for group in groups:
            stats = stats_by_group.get(group, init_stats())
            n = stats["n"]
            total_n += n
            row[f"{group}_n"] = n
            if n:
                row[f"{group}_mean_score"] = stats["score_sum"] / n
                row[f"{group}_mean_drop"] = stats["drop_sum"] / n
                row[f"{group}_mean_rel_drop_pct"] = stats["rel_drop_sum"] / n
                row[f"{group}_mean_selected_epoch"] = stats["selected_epoch_sum"] / n
                row[f"{group}_mean_trigger_epoch"] = stats["trigger_epoch_sum"] / n
                row[f"{group}_mean_delta_vs_fixed_warmup"] = stats["fixed_gap_sum"] / n
                scores.append(row[f"{group}_mean_score"])
                drops.append(row[f"{group}_mean_drop"])
                rel_drops.append(row[f"{group}_mean_rel_drop_pct"])
                selected_epochs.append(row[f"{group}_mean_selected_epoch"])
                trigger_epochs.append(row[f"{group}_mean_trigger_epoch"])
                fixed_gaps.append(row[f"{group}_mean_delta_vs_fixed_warmup"])
            else:
                for suffix in (
                    "mean_score",
                    "mean_drop",
                    "mean_rel_drop_pct",
                    "mean_selected_epoch",
                    "mean_trigger_epoch",
                    "mean_delta_vs_fixed_warmup",
                ):
                    row[f"{group}_{suffix}"] = math.nan
            if n != n_by_group.get(group, 0):
                full = False
        if not scores:
            continue
        row["total_n"] = total_n
        row["full_coverage"] = full
        row["mean_score"] = mean(scores)
        row["mean_drop"] = mean(drops)
        row["mean_rel_drop_pct"] = mean(rel_drops)
        row["mean_selected_epoch"] = mean(selected_epochs)
        row["mean_trigger_epoch"] = mean(trigger_epochs)
        row["mean_delta_vs_fixed_warmup"] = mean(fixed_gaps)
        rows.append(row)

    full_rows = [row for row in rows if row["full_coverage"]]
    rank_source = full_rows or rows
    for group in groups:
        sortable = [
            (idx, row)
            for idx, row in enumerate(rank_source)
            if math.isfinite(float(row.get(f"{group}_mean_score", math.nan)))
        ]
        sortable.sort(key=lambda item: item[1][f"{group}_mean_score"], reverse=True)
        for rank, (idx, _) in enumerate(sortable, start=1):
            rank_source[idx][f"{group}_rank"] = rank
    for row in rank_source:
        ranks = [row.get(f"{group}_rank") for group in groups if row.get(f"{group}_rank") is not None]
        row["avg_rank"] = mean(ranks) if ranks else math.nan
    rank_source.sort(key=lambda r: (r.get("avg_rank", math.inf), -r["mean_score"], r["mean_drop"]))
    rows.sort(key=lambda r: (not r["full_coverage"], r.get("avg_rank", math.inf), -r["mean_score"], r["mean_drop"]))
    return rows


def source_audit(cells: list[Cell]) -> dict:
    keys = defaultdict(lambda: {"present": 0, "scalar": 0, "nonconstant_warmup": 0})
    metric_counts = []
    score_key_counts = {"teacher_pak_auc_f1_rows": 0, "fallback_pak_auc_f1_rows": 0}
    for cell in cells:
        n_epochs = len(cell.history.get("epoch", []))
        for key in INTUITIVE_KEYS:
            if key in cell.history:
                keys[key]["present"] += 1
            series = base.to_scalar_series(cell.history.get(key), n_epochs)
            if series is not None:
                keys[key]["scalar"] += 1
                s = warmup_mask(series, cell.warmup)
                if base.finite_ratio(s[: cell.warmup]) >= 0.5 and np.nanstd(s[: cell.warmup]) > 1e-12:
                    keys[key]["nonconstant_warmup"] += 1
        metric_counts.append(len(build_metric_bank(cell.history, cell.warmup)))
        for row in read_epochs(Path(cell.score_path)):
            epoch = int(row.get("epoch", -1))
            if epoch <= 0 or epoch > cell.warmup:
                continue
            if finite_float(row.get("teacher_pak_auc_f1")) is not None:
                score_key_counts["teacher_pak_auc_f1_rows"] += 1
            elif finite_float(row.get("pak_auc_f1")) is not None:
                score_key_counts["fallback_pak_auc_f1_rows"] += 1
    return {
        "n_cells": len(cells),
        "keys": dict(sorted(keys.items())),
        "metric_bank_min": int(min(metric_counts)) if metric_counts else 0,
        "metric_bank_mean": float(mean(metric_counts)) if metric_counts else 0.0,
        "metric_bank_max": int(max(metric_counts)) if metric_counts else 0,
        **score_key_counts,
    }


def fixed_baselines(cells: list[Cell], groups: list[str], n_by_group: dict[str, int]) -> list[dict]:
    policies: list[tuple[str, int | None]] = [(f"epoch{e}", e) for e in FIXED_EPOCHS] + [("config_warmup", None)]
    rows = []
    for label, fixed_epoch in policies:
        row = {"policy": label}
        scores = []
        drops = []
        ns = []
        for group in groups:
            group_cells = [cell for cell in cells if cell.group == group]
            vals = []
            for cell in group_cells:
                target = cell.warmup if fixed_epoch is None else fixed_epoch
                if target > cell.warmup:
                    continue
                scored = nearest_score_epoch(cell.score_by_epoch, target)
                if scored is None:
                    continue
                epoch, score = scored
                vals.append((score, cell.oracle_score - score, epoch))
            row[f"{group}_n"] = len(vals)
            ns.append(len(vals))
            if vals:
                row[f"{group}_mean_score"] = mean(v[0] for v in vals)
                row[f"{group}_mean_drop"] = mean(v[1] for v in vals)
                row[f"{group}_mean_epoch"] = mean(v[2] for v in vals)
                scores.append(row[f"{group}_mean_score"])
                drops.append(row[f"{group}_mean_drop"])
            else:
                row[f"{group}_mean_score"] = math.nan
                row[f"{group}_mean_drop"] = math.nan
                row[f"{group}_mean_epoch"] = math.nan
        row["total_n"] = sum(ns)
        row["full_coverage"] = all(row.get(f"{group}_n", 0) == n_by_group[group] for group in groups)
        row["mean_score"] = mean(scores) if scores else math.nan
        row["mean_drop"] = mean(drops) if drops else math.nan
        rows.append(row)
    return rows


def oracle_summary(cells: list[Cell], groups: list[str]) -> dict:
    by_group = {}
    overfit_examples = []
    for group in groups:
        group_cells = [cell for cell in cells if cell.group == group]
        if not group_cells:
            continue
        drops = [cell.oracle_score - cell.fixed_warmup_score for cell in group_cells]
        by_group[group] = {
            "n": len(group_cells),
            "mean_oracle_score": mean(cell.oracle_score for cell in group_cells),
            "mean_oracle_epoch": mean(cell.oracle_epoch for cell in group_cells),
            "mean_fixed_warmup_score": mean(cell.fixed_warmup_score for cell in group_cells),
            "mean_fixed_warmup_epoch": mean(cell.fixed_warmup_epoch for cell in group_cells),
            "mean_fixed_drop": mean(drops),
            "overfit_drop_gt_0_01": sum(d > 0.01 for d in drops),
            "overfit_drop_gt_0_03": sum(d > 0.03 for d in drops),
            "overfit_drop_gt_0_05": sum(d > 0.05 for d in drops),
        }
        for cell, drop in sorted(zip(group_cells, drops), key=lambda item: item[1], reverse=True)[:5]:
            overfit_examples.append(
                {
                    "group": group,
                    "exp": cell.exp,
                    "oracle_score": cell.oracle_score,
                    "oracle_epoch": cell.oracle_epoch,
                    "fixed_score": cell.fixed_warmup_score,
                    "fixed_epoch": cell.fixed_warmup_epoch,
                    "drop": drop,
                }
            )
    overfit_examples.sort(key=lambda item: item["drop"], reverse=True)
    return {"by_group": by_group, "top_overfit_examples": overfit_examples[:25]}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cell_table_for_criterion(
    key: tuple,
    cell_results: dict[tuple, dict[str, dict]],
    cells: list[Cell],
    groups: list[str],
    max_rows: int | None = None,
) -> dict:
    by_exp_group = cell_results.get(key, {})
    exp_nums = sorted({cell.exp_num for cell in cells if cell.group in groups})
    exp_name = {cell.exp_num: cell.exp for cell in cells}
    ranks_by_group: dict[str, dict[str, int]] = {}
    for group in groups:
        vals = [v for v in by_exp_group.values() if v["group"] == group]
        vals.sort(key=lambda item: item["score"], reverse=True)
        ranks_by_group[group] = {f"{v['exp']}|{group}": rank for rank, v in enumerate(vals, start=1)}
    rows = []
    for exp_num in exp_nums:
        exp = exp_name[exp_num]
        ds_payload = {}
        ranks = []
        coverage = 0
        for group in groups:
            payload = by_exp_group.get(f"{exp}|{group}")
            if payload is None:
                ds_payload[group] = None
                continue
            rank = ranks_by_group[group].get(f"{exp}|{group}")
            ranks.append(rank)
            coverage += 1
            ds_payload[group] = {
                "score": payload["score"],
                "rank": rank,
                "selected_epoch": payload["selected_epoch"],
                "trigger_epoch": payload["trigger_epoch"],
                "oracle_epoch": payload["oracle_epoch"],
                "fixed_warmup_epoch": payload["fixed_warmup_epoch"],
            }
        rows.append(
            {
                "model": exp,
                "coverage": coverage,
                "avg_rank": mean(ranks) if ranks else None,
                "groups": ds_payload,
            }
        )
    rows.sort(key=lambda row: (row["avg_rank"] is None, row["avg_rank"] or 10**9, -row["coverage"]))
    if max_rows is not None:
        rows = rows[:max_rows]
    return {"criterion": criterion_meta(key), "groups": groups, "rows": rows}


def evaluate_selected_keys(cells: list[Cell], keys: set[tuple]) -> dict[tuple, dict[str, dict]]:
    results: dict[tuple, dict[str, dict]] = {key: {} for key in keys}
    if not keys:
        return results
    keys_by_metric: dict[str, list[tuple]] = defaultdict(list)
    for key in keys:
        keys_by_metric[key[0]].append(key)

    for cell in cells:
        metrics = build_metric_bank(cell.history, cell.warmup)
        for metric, metric_keys in keys_by_metric.items():
            raw_values = metrics.get(metric)
            if raw_values is None:
                continue
            for key in metric_keys:
                _, transform, direction_mode, rule, patience, threshold_type, threshold_value, _, start_policy = key
                values = base.transform_series(raw_values, transform)
                if base.finite_ratio(values[: cell.warmup]) < 0.5 or np.nanstd(values[: cell.warmup]) <= 1e-12:
                    continue
                direction = resolve_direction(metric, direction_mode)
                pts = eval_points(values, cell.score_epochs, start_epoch(start_policy), direction)
                if len(pts) < 3:
                    continue
                res = stop_rule(rule, pts, int(patience), threshold_type, float(threshold_value))
                if res is None:
                    continue
                selected_epoch, trigger_epoch = res
                score = cell.score_by_epoch.get(selected_epoch)
                if score is None:
                    continue
                results[key][f"{cell.exp}|{cell.group}"] = {
                    "exp": cell.exp,
                    "exp_num": cell.exp_num,
                    "family": cell.family,
                    "group": cell.group,
                    "entity": cell.entity,
                    "score": score,
                    "selected_epoch": selected_epoch,
                    "trigger_epoch": trigger_epoch,
                    "oracle_score": cell.oracle_score,
                    "oracle_epoch": cell.oracle_epoch,
                    "fixed_warmup_score": cell.fixed_warmup_score,
                    "fixed_warmup_epoch": cell.fixed_warmup_epoch,
                    "drop": cell.oracle_score - score,
                }
    return results


def family_top_tables(
    rows_by_scope: dict[str, list[dict]],
    cell_results: dict[tuple, dict[str, dict]],
    cells: list[Cell],
    scope_groups: dict[str, list[str]],
) -> dict:
    payload = {}
    for scope, rows in rows_by_scope.items():
        top = [row for row in rows if row.get("full_coverage")][:5] or rows[:5]
        payload[scope] = {
            "criteria": top,
            "tables": [
                cell_table_for_criterion(row_to_key(row), cell_results, cells, scope_groups[scope], max_rows=30)
                for row in top
            ],
        }
    return payload


def find_matching_row(rows: list[dict], key: tuple) -> dict | None:
    key_s = "|".join(str(x) for x in key)
    for row in rows:
        if row["key"] == key_s:
            return row
    return None


def row_to_key(row: dict) -> tuple:
    return criterion_tuple(
        row["metric"],
        row["transform"],
        row["direction_mode"],
        row["rule"],
        int(row["patience"]),
        row["threshold_type"],
        float(row["threshold_value"]),
        row["start_policy"],
    )


def output_name(base_name: str, scope: str) -> str:
    if scope == "all":
        return base_name
    stem, suffix = base_name.rsplit(".", 1)
    return f"{stem}_{scope}.{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("all", "main4", "SMD", "MSL", "SMAP"), default="all")
    parser.add_argument("--progress-file", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = load_cells(ROOT, args.scope)
    print(f"Loaded {len(cells)} teacher-only warmup cells for scope={args.scope}", flush=True)
    if not cells:
        raise SystemExit("No cells found")

    all_groups_by_scope = {
        "main4": list(MAIN_DATASETS),
        "SMD": sorted({cell.group for cell in cells if cell.family == "SMD"}),
        "MSL": sorted({cell.group for cell in cells if cell.family == "MSL"}),
        "SMAP": sorted({cell.group for cell in cells if cell.family == "SMAP"}),
    }
    if args.scope == "all":
        groups_by_scope = all_groups_by_scope
    else:
        groups_by_scope = {args.scope: all_groups_by_scope[args.scope]}
    cells_by_scope = {
        scope: [cell for cell in cells if cell.group in groups]
        for scope, groups in groups_by_scope.items()
    }
    n_by_group = defaultdict(int)
    for cell in cells:
        n_by_group[cell.group] += 1
    print("Cells by scope:", {k: len(v) for k, v in cells_by_scope.items()}, flush=True)

    audit = source_audit(cells)
    accum = sweep(
        cells,
        scope=args.scope,
        progress_file=args.progress_file,
        progress_every=max(1, args.progress_every),
    )

    rows_by_scope: dict[str, list[dict]] = {}
    fixed_by_scope: dict[str, list[dict]] = {}
    oracle_by_scope: dict[str, dict] = {}
    for scope, groups in groups_by_scope.items():
        if not groups:
            continue
        rows = summarize_scope(accum, groups, n_by_group, scope)
        rows_by_scope[scope] = rows
        write_csv(OUT_DIR / f"leaderboard_{scope}.csv", rows)
        fixed_by_scope[scope] = fixed_baselines(cells_by_scope[scope], groups, n_by_group)
        write_csv(OUT_DIR / f"fixed_baselines_{scope}.csv", fixed_by_scope[scope])
        oracle_by_scope[scope] = oracle_summary(cells_by_scope[scope], groups)

    pf_key = key_from_spec(PAPER_FRIENDLY_SPEC)
    pf_rows = {scope: find_matching_row(rows, pf_key) for scope, rows in rows_by_scope.items()}
    selected_keys: set[tuple] = {pf_key}
    for rows in rows_by_scope.values():
        selected = [row for row in rows if row.get("full_coverage")][:5] or rows[:5]
        selected_keys.update(row_to_key(row) for row in selected)
    cell_results = evaluate_selected_keys(cells, selected_keys)
    table_payload = family_top_tables(rows_by_scope, cell_results, cells, groups_by_scope)
    pf_tables = {
        scope: cell_table_for_criterion(pf_key, cell_results, cells, groups, max_rows=60 if scope == "main4" else 30)
        for scope, groups in groups_by_scope.items()
        if groups and pf_rows.get(scope) is not None
    }
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_cells_total": len(cells),
        "n_cells_by_scope": {scope: len(v) for scope, v in cells_by_scope.items()},
        "n_by_group": dict(sorted(n_by_group.items())),
        "source_audit": audit,
        "oracle_by_scope": oracle_by_scope,
        "fixed_by_scope": fixed_by_scope,
        "top_by_scope": {
            scope: ([row for row in rows if row.get("full_coverage")][:10] or rows[:10])
            for scope, rows in rows_by_scope.items()
        },
        "paper_friendly_spec": {**PAPER_FRIENDLY_SPEC, "metric_pretty": pretty_metric(PAPER_FRIENDLY_SPEC["metric"])},
        "paper_friendly_rows": pf_rows,
    }
    (OUT_DIR / output_name("summary.json", args.scope)).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True)
    )
    (OUT_DIR / output_name("top_tables.json", args.scope)).write_text(
        json.dumps(table_payload, indent=2, ensure_ascii=False, allow_nan=True)
    )
    (OUT_DIR / output_name("paper_friendly_tables.json", args.scope)).write_text(
        json.dumps(pf_tables, indent=2, ensure_ascii=False, allow_nan=True)
    )
    cells_light = [
        {
            "exp": cell.exp,
            "exp_num": cell.exp_num,
            "family": cell.family,
            "group": cell.group,
            "warmup": cell.warmup,
            "oracle_epoch": cell.oracle_epoch,
            "oracle_score": cell.oracle_score,
            "fixed_warmup_epoch": cell.fixed_warmup_epoch,
            "fixed_warmup_score": cell.fixed_warmup_score,
        }
        for cell in cells
    ]
    (OUT_DIR / output_name("cells.json", args.scope)).write_text(json.dumps(cells_light, indent=2, ensure_ascii=False))

    print("Wrote", OUT_DIR, flush=True)
    for scope, rows in rows_by_scope.items():
        full = [row for row in rows if row.get("full_coverage")]
        top = full[0] if full else rows[0]
        print(
            f"{scope}: full={len(full)} top={top['metric_pretty']} "
            f"score={top['mean_score']:.4f} drop={top['mean_drop']:.4f} "
            f"selected={top['mean_selected_epoch']:.1f}",
            flush=True,
        )
    if pf_rows.get("main4"):
        row = pf_rows["main4"]
        print(
            f"PF main4: score={row['mean_score']:.4f} drop={row['mean_drop']:.4f} "
            f"selected={row['mean_selected_epoch']:.1f} rank={row.get('avg_rank')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
