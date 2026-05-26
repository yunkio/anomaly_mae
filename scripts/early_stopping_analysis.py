"""Early Stopping Analysis on 271 baseline.

Goal: For each dataset, find the (metric, patience P, threshold T) combination that,
when used as an early-stopping criterion, yields the smallest performance loss
compared to oracle best_epoch (pak_auc_f1 selection).

Inputs (per dataset):
  - training_histories.json: 48 series, 500 values (per epoch)
  - epoch_metrics.json: 100 eval checkpoints (epoch 5, 10, ..., 500), 182 metrics each
  - Oracle metric: pak_auc_f1 (the same field the project uses as best_epoch_metric)

Algorithm:
  - Warmup: 250 epochs (no ES before)
  - Eval points: every 5 epochs (epochs 255, 260, ..., 500)
  - At each eval point, compare the metric value to the running best.
    If improvement > threshold T, reset patience counter; else increment.
    If patience counter >= P, STOP at current eval epoch.
  - Look up pak_auc_f1 at stop_epoch from epoch_metrics.
  - Compare to oracle best_epoch's pak_auc_f1 (max over all eval checkpoints).

Outputs:
  - JSON with all sweep results per (dataset, metric, P, T)
  - Best (metric, P, T) per dataset
  - Aggregate over SMD(15), Exathlon(6), and overall rank avg

Run:
  python scripts/early_stopping_analysis.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXP_ROOT = Path(
    "/home/ykio/notebooks/claude/results/experiments/"
    "271_20260508_094241_w500p10e4t3d2_dynamic_linear_minmax_k6"
)

# ---------------- Dataset path config ----------------
SMD_15_MACHINES = [
    "machine-1-2", "machine-1-7",
    "machine-2-1", "machine-2-2", "machine-2-3", "machine-2-4",
    "machine-2-6", "machine-2-7", "machine-2-9",
    "machine-3-1", "machine-3-2", "machine-3-3",
    "machine-3-6", "machine-3-8", "machine-3-9",
]
EXATHLON_APPS = ["app1", "app2", "app4", "app5", "app6", "app9"]

DATASETS = {
    # name → (training_history_path, scoring_epoch_metrics_path)
    # User Q2: history is FULL, scoring uses excl22 for SWaT
    "SWaT_excl22": ("SWaT/A1A2_full", "SWaT/A1A2_excl22"),
    "WaDi_A1": ("WaDi/A1", "WaDi/A1"),
    "WaDi_A2": ("WaDi/A2", "WaDi/A2"),
    "PSM": ("PSM", "PSM"),
}
for m in SMD_15_MACHINES:
    DATASETS[f"SMD_{m}"] = (f"SMD/{m}", f"SMD/{m}")
for a in EXATHLON_APPS:
    DATASETS[f"Exathlon_{a}"] = (f"Exathlon/{a}", f"Exathlon/{a}")

# ---------------- Hyperparameters ----------------
WARMUP_EPOCH = 250
EVAL_INTERVAL = 5  # epoch_metrics records at every 5 epochs
PATIENCE_GRID = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
# Threshold modes: ("type", value); abs = absolute delta, rel = relative delta
THRESHOLD_GRID = [
    ("abs", 0.0),
    ("abs", 0.001),
    ("rel", 0.001),
    ("rel", 0.01),
    ("rel", 0.05),
]

# ---------------- Metric direction ----------------
# direction: "max" = bigger is better (F1, AUC), "min" = smaller is better (loss)
def metric_direction(name: str) -> str:
    """Determine optimization direction from metric name."""
    n = name.lower()
    # losses, errors, recon = minimize
    if any(t in n for t in ["loss", "recon_", "_recon", "discrepancy", "_score"]):
        # special: "snr" prefers higher, "balanced_acc" higher, "f1" higher, "auc" higher
        if "snr" in n or "auc" in n or "_acc" in n or "_f1" in n or "precision" in n or "recall" in n:
            return "max"
        return "min"
    if any(t in n for t in ["acc", "auc", "f1", "precision", "recall", "snr"]):
        return "max"
    # default for ratios: ambiguous, treat as min (lower normal_loss good)
    if "ratio" in n:
        # ratio = anom/normal generally → higher = better separation
        return "max"
    if "separation" in n or "_diff" in n:
        return "max"  # bigger anom-normal gap is better
    if "threshold" in n:
        return "max"  # higher threshold often means better-separable scores
    return "min"


# ---------------- Loaders ----------------
def load_dataset(name: str, paths: tuple[str, str]) -> dict:
    """Load training_histories + epoch_metrics. Two paths: (training, scoring).

    Training: provides per-epoch loss/recon/grl series + inference metrics from
              the FULL evaluation set (used as ES monitor source).
    Scoring : provides the pak_auc_f1 lookup table used to report final
              performance at stop_epoch. For SWaT this points to the excl22
              directory; for all other datasets it equals the training path.
    """
    train_rel, score_rel = paths
    train_base = EXP_ROOT / train_rel
    score_base = EXP_ROOT / score_rel

    th_path = train_base / "training_histories.json"
    em_train_path = train_base / "epoch_metrics.json"
    em_score_path = score_base / "epoch_metrics.json"
    meta_path = train_base / "experiment_metadata.json"

    for p in (th_path, em_train_path, em_score_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing for {name}: {p}")

    with open(th_path) as f:
        th_raw = json.load(f)
    th = th_raw[list(th_raw.keys())[0]]  # fold index "0"

    with open(em_train_path) as f:
        em_train = json.load(f)["epochs"]

    if em_score_path == em_train_path:
        em_score = em_train  # same file
    else:
        with open(em_score_path) as f:
            em_score = json.load(f)["epochs"]

    meta = json.load(open(meta_path))
    timing = meta.get("timing", {})
    best_epoch_meta = timing.get("best_epoch")

    return {
        "th": th,
        "em": em_train,        # used for ES monitoring (em_* metrics)
        "em_score": em_score,  # used for final pak_auc_f1 lookup
        "meta": meta,
        "best_epoch_meta": best_epoch_meta,
    }


# ---------------- Candidate metrics ----------------
# Subset of training_histories metrics that are per-epoch scalars (skip lists, dicts, profiling)
TH_SCALAR_METRICS = [
    "train_loss",
    "train_rec_loss",
    "train_disc_loss",
    "train_normal_loss",
    "train_anomaly_loss",
    "train_fm_loss",
    "train_fm_adaptive_lambda",
    "train_grl_cls_loss",
    "train_grl_balanced_acc",
    "train_grl_anomaly_acc",
    "train_grl_normal_acc",
    "train_grl_effective_weight",
    "train_grl_lambda",
    "train_teacher_recon_normal",
    "train_teacher_recon_anomaly",
    "train_student_recon_normal",
    "train_student_recon_anomaly",
    "epoch_recon_score_normal",
    "epoch_recon_score_anomaly",
    "epoch_recon_score_disturbing",
    "epoch_disc_score_normal",
    "epoch_disc_score_anomaly",
    "epoch_disc_score_disturbing",
    "epoch_recon_ratio_normal",
    "epoch_recon_ratio_anomaly",
    "epoch_recon_ratio_disturbing",
    "epoch_disc_ratio_normal",
    "epoch_disc_ratio_anomaly",
    "epoch_disc_ratio_disturbing",
    "epoch_raw_recon_normal",
    "epoch_raw_recon_anomaly",
    "epoch_raw_disc_normal",
    "epoch_raw_disc_anomaly",
]

# Subset of epoch_metrics fields (eval-time, every 5 ep)
EM_SCALAR_METRICS = [
    # Primary AUC family
    "pak_auc_f1", "pak_auc_f1_raw", "pak_auc_prc_auc", "pak_auc_precision",
    "pak_auc_recall", "pak_auc_roc_auc",
    # Teacher version
    "teacher_pak_auc_f1", "teacher_pak_auc_prc_auc",
    # Base AUC
    "prc_auc", "roc_auc",
    "teacher_prc_auc",
    # F1 at optimal threshold
    "f1_score", "teacher_f1_t",
    "precision", "recall",
    # Disturbing region (SWaT-specific but populated everywhere)
    "disturbing_f1", "disturbing_roc_auc",
    # Discrepancy signal-to-noise
    "disc_snr",
    # PA-K @ specific levels
    "pa_0_f1", "pa_5_f1", "pa_10_f1", "pa_20_f1", "pa_30_f1", "pa_50_f1",
    "pa_0_prc_auc", "pa_5_prc_auc", "pa_10_prc_auc", "pa_30_prc_auc",
    "pa_50_prc_auc",
]


def extract_th_series(th: dict, name: str, n_epochs: int = 500) -> list[float] | None:
    """Extract a per-epoch series from training_histories (skip None / lists)."""
    if name not in th:
        return None
    series = th[name]
    if len(series) != n_epochs:
        return None
    out = []
    for v in series:
        if v is None or isinstance(v, (list, dict)):
            return None
        out.append(float(v))
    return out


def extract_em_series(em: list, name: str) -> tuple[list[float], list[int]] | None:
    """Extract metric series and corresponding epochs from epoch_metrics."""
    vals, eps = [], []
    for entry in em:
        if name not in entry:
            return None
        v = entry[name]
        if v is None:
            return None
        try:
            vals.append(float(v))
            eps.append(int(entry["epoch"]))
        except (TypeError, ValueError):
            return None
    return vals, eps


def build_derived_series(th: dict, em: list) -> dict[str, tuple[list[float], list[int], str]]:
    """Compute derived metrics. Returns {name: (values, epochs, direction)}."""
    out = {}
    # Derived from training_histories (every-epoch)
    epochs_all = list(range(1, 501))

    def th_series(name):
        return extract_th_series(th, name)

    teacher_n = th_series("train_teacher_recon_normal")
    teacher_a = th_series("train_teacher_recon_anomaly")
    student_n = th_series("train_student_recon_normal")
    student_a = th_series("train_student_recon_anomaly")
    recon_score_n = th_series("epoch_recon_score_normal")
    recon_score_a = th_series("epoch_recon_score_anomaly")
    disc_score_n = th_series("epoch_disc_score_normal")
    disc_score_a = th_series("epoch_disc_score_anomaly")

    def gap(a, b):
        if a is None or b is None:
            return None
        return [x - y for x, y in zip(a, b)]

    def ratio(a, b, eps=1e-8):
        if a is None or b is None:
            return None
        return [x / max(y, eps) for x, y in zip(a, b)]

    def separation(a, n, eps=1e-8):
        if a is None or n is None:
            return None
        return [(x - y) / max(abs(x) + abs(y), eps) for x, y in zip(a, n)]

    derived_th = {
        "deriv_teacher_anom_normal_gap": (gap(teacher_a, teacher_n), "max"),
        "deriv_teacher_anom_normal_ratio": (ratio(teacher_a, teacher_n), "max"),
        "deriv_teacher_anom_normal_separation": (separation(teacher_a, teacher_n), "max"),
        "deriv_student_anom_normal_gap": (gap(student_a, student_n), "max"),
        "deriv_student_anom_normal_ratio": (ratio(student_a, student_n), "max"),
        "deriv_student_anom_normal_separation": (separation(student_a, student_n), "max"),
        "deriv_recon_score_gap": (gap(recon_score_a, recon_score_n), "max"),
        "deriv_recon_score_separation": (separation(recon_score_a, recon_score_n), "max"),
        "deriv_disc_score_gap": (gap(disc_score_a, disc_score_n), "max"),
        "deriv_disc_score_separation": (separation(disc_score_a, disc_score_n), "max"),
    }
    for name, (vals, direction) in derived_th.items():
        if vals is None:
            continue
        out[name] = (vals, epochs_all, direction)
    return out


# ---------------- Early stopping core ----------------
@dataclass
class ESResult:
    stop_epoch: int
    best_seen_value: float
    n_evals: int


def early_stopping(
    series_values: list[float],
    series_epochs: list[int],
    direction: str,
    patience: int,
    thresh_type: str,
    thresh_value: float,
    warmup: int = WARMUP_EPOCH,
) -> ESResult:
    """Simulate early stopping over a metric series.

    series_values, series_epochs: parallel lists (sorted by epoch).
    direction: "max" or "min".
    Returns the stop_epoch (epoch at which we'd halt) and the best metric seen.
    If never triggered, stop_epoch = last epoch in series.
    """
    # Filter to evaluation points >= warmup AND at multiples of EVAL_INTERVAL
    eval_pts = [(v, e) for v, e in zip(series_values, series_epochs)
                if e >= warmup and e % EVAL_INTERVAL == 0]
    if not eval_pts:
        return ESResult(stop_epoch=series_epochs[-1], best_seen_value=series_values[-1], n_evals=0)

    # Initialize best from the first eval point at warmup
    best_value = eval_pts[0][0]
    best_epoch_local = eval_pts[0][1]
    counter = 0
    stop_epoch = eval_pts[-1][1]

    def is_improvement(new, current):
        if direction == "max":
            delta = new - current
        else:
            delta = current - new
        if thresh_type == "abs":
            return delta > thresh_value
        # rel
        denom = max(abs(current), 1e-8)
        return (delta / denom) > thresh_value

    for v, e in eval_pts[1:]:
        if is_improvement(v, best_value):
            best_value = v
            best_epoch_local = e
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                stop_epoch = e
                break

    return ESResult(stop_epoch=stop_epoch, best_seen_value=best_value, n_evals=len(eval_pts))


def lookup_em_at_epoch(em: list, epoch: int, key: str) -> float | None:
    """Find epoch_metrics entry at given epoch and return its key."""
    # exact match preferred
    for entry in em:
        if entry["epoch"] == epoch:
            return entry.get(key)
    # closest
    closest = min(em, key=lambda e: abs(e["epoch"] - epoch))
    return closest.get(key)


# ---------------- Main runner ----------------
def collect_metrics(ds: dict) -> dict[str, tuple[list[float], list[int], str]]:
    """Collect all candidate metrics for one dataset → {name: (values, epochs, direction)}.

    Uses ds['em'] (training/full eval) — NOT ds['em_score']. ES is decided on the
    FULL evaluation curve, scoring is reported via ds['em_score'].
    """
    th = ds["th"]
    em = ds["em"]
    metrics = {}

    # 1) training-history per-epoch scalars
    for name in TH_SCALAR_METRICS:
        vals = extract_th_series(th, name)
        if vals is None:
            continue
        epochs = list(range(1, len(vals) + 1))
        metrics[f"th_{name}"] = (vals, epochs, metric_direction(name))

    # 2) epoch-metrics every-5-ep scalars
    for name in EM_SCALAR_METRICS:
        r = extract_em_series(em, name)
        if r is None:
            continue
        vals, eps = r
        metrics[f"em_{name}"] = (vals, eps, metric_direction(name))

    # 3) derived
    derived = build_derived_series(th, em)
    metrics.update(derived)

    return metrics


def oracle_best(em: list, score_key: str = "pak_auc_f1") -> tuple[int, float]:
    """Find oracle best epoch by max pak_auc_f1."""
    best = max(em, key=lambda e: e.get(score_key, -1e18))
    return best["epoch"], best.get(score_key)


def sweep_dataset(ds_name: str, ds: dict, score_key: str = "pak_auc_f1") -> dict:
    """Run full sweep on one dataset.

    Decision metric series come from ds['em'] (full eval) and ds['th'] (training).
    Final pak_auc_f1 lookup at stop_epoch comes from ds['em_score']
    (= excl22 for SWaT, = same as em for others).
    """
    metrics = collect_metrics(ds)
    em_score = ds["em_score"]  # used for lookup

    # Oracle uses the SCORING table (excl22 for SWaT, full for others)
    oracle_ep, oracle_val = oracle_best(em_score, score_key)

    rows = []
    for mname, (vals, epochs, direction) in metrics.items():
        for P in PATIENCE_GRID:
            for ttype, tval in THRESHOLD_GRID:
                r = early_stopping(vals, epochs, direction, P, ttype, tval)
                lookup_val = lookup_em_at_epoch(em_score, r.stop_epoch, score_key)
                if lookup_val is None:
                    continue
                rows.append({
                    "metric": mname,
                    "direction": direction,
                    "patience": P,
                    "thresh_type": ttype,
                    "thresh_value": tval,
                    "stop_epoch": r.stop_epoch,
                    f"{score_key}_at_stop": float(lookup_val),
                    "n_evals": r.n_evals,
                })

    return {
        "dataset": ds_name,
        "oracle_epoch": oracle_ep,
        f"oracle_{score_key}": float(oracle_val) if oracle_val is not None else None,
        "n_metrics_checked": len(metrics),
        "n_rows": len(rows),
        "rows": rows,
    }


def main():
    print("Loading datasets...")
    datasets = {}
    for name, paths in DATASETS.items():
        try:
            datasets[name] = load_dataset(name, paths)
        except Exception as e:
            print(f"  WARN: failed to load {name}: {e}")
    print(f"Loaded {len(datasets)} datasets")

    print("Running sweep...")
    all_results = {}
    for i, (name, ds) in enumerate(datasets.items()):
        result = sweep_dataset(name, ds)
        all_results[name] = result
        print(f"  [{i+1}/{len(datasets)}] {name}: oracle ep={result['oracle_epoch']} "
              f"val={result['oracle_pak_auc_f1']:.4f} sweep_rows={result['n_rows']}")

    out_path = Path("/home/ykio/notebooks/claude/temp/early_stopping/sweep_raw.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=1)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved to {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
