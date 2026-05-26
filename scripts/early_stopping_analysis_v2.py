"""Early Stopping Analysis v2 — full metric pool.

Changes from v1:
1. ALL training_histories scalar keys (including train_mean_discrepancy after re-train; for now skip if missing).
2. ALL training_histories per-feature list keys reduced to scalars (mean/max/std/min over features).
3. ALL epoch_metrics scalar keys (176 keys including full PA-K table at K=0..100).
4. ALL epoch_metrics per-feature lists reduced to scalars.
5. Post-process variants: raw + ema03 + slope10 + slope20 + curvature10 + variance10 + sign_changes10.
6. User-proposed + brainstormed derived dynamics:
   - |Δteacher_recon_normal| / |Δstudent_recon_normal| (window=10)
   - Various recon-gap / score-separation / PA-K-curve dynamics.

No metric is excluded "due to time" — everything possible from the available
training_histories.json and epoch_metrics.json is run through the sweep.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean as stat_mean

import numpy as np

EXP_ROOT = Path(
    "/home/ykio/notebooks/claude/results/experiments/"
    "271_20260508_094241_w500p10e4t3d2_dynamic_linear_minmax_k6"
)

SMD_15_MACHINES = [
    "machine-1-2", "machine-1-7",
    "machine-2-1", "machine-2-2", "machine-2-3", "machine-2-4",
    "machine-2-6", "machine-2-7", "machine-2-9",
    "machine-3-1", "machine-3-2", "machine-3-3",
    "machine-3-6", "machine-3-8", "machine-3-9",
]
EXATHLON_APPS = ["app1", "app2", "app4", "app5", "app6", "app9"]

DATASETS = {
    "SWaT_excl22": ("SWaT/A1A2_full", "SWaT/A1A2_excl22"),
    "WaDi_A1": ("WaDi/A1", "WaDi/A1"),
    "WaDi_A2": ("WaDi/A2", "WaDi/A2"),
    "PSM": ("PSM", "PSM"),
}
for m in SMD_15_MACHINES:
    DATASETS[f"SMD_{m}"] = (f"SMD/{m}", f"SMD/{m}")
for a in EXATHLON_APPS:
    DATASETS[f"Exathlon_{a}"] = (f"Exathlon/{a}", f"Exathlon/{a}")

WARMUP_EPOCH = 250
EVAL_INTERVAL = 5
PATIENCE_GRID = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
THRESHOLD_GRID = [
    ("abs", 0.0),
    ("abs", 0.001),
    ("rel", 0.001),
    ("rel", 0.01),
    ("rel", 0.05),
]

# Keys that are not metrics (metadata)
EM_METADATA_KEYS = {"epoch", "_eval_time", "_inference_time",
                    "n_anomaly", "n_pure_normal", "n_disturbing_normal"}


def metric_direction(name: str) -> str:
    """max / min direction inference from name."""
    n = name.lower()
    # ratios / separations / acc / auc / f1 / snr / precision / recall / detection_rate
    if any(s in n for s in ("acc", "auc", "f1", "snr", "precision", "recall",
                             "detection_rate", "separation", "spread", "ratio")):
        return "max"
    # losses, scores, errors, residuals, lambdas (training progress: typically growing → max often;
    # for adaptive lambda we treat as `min` of |Δ| via dynamic derived; raw lambda we treat as "max"
    # since it grows). Be conservative.
    if any(s in n for s in ("loss", "discrepancy", "recon", "error", "_score",
                             "threshold", "magnitude", "norm", "_disc",
                             "_raw", "_time")):
        return "min"
    # default
    return "min"


# ---------------- Loaders ----------------

def load_dataset(name: str, paths) -> dict:
    train_rel, score_rel = paths
    train_base = EXP_ROOT / train_rel
    score_base = EXP_ROOT / score_rel
    th_path = train_base / "training_histories.json"
    em_train_path = train_base / "epoch_metrics.json"
    em_score_path = score_base / "epoch_metrics.json"

    with open(th_path) as f:
        th_raw = json.load(f)
    th = th_raw[list(th_raw.keys())[0]]

    with open(em_train_path) as f:
        em_train = json.load(f)["epochs"]

    if em_score_path == em_train_path:
        em_score = em_train
    else:
        with open(em_score_path) as f:
            em_score = json.load(f)["epochs"]

    return {"th": th, "em": em_train, "em_score": em_score}


# ---------------- Base series extraction ----------------

def is_scalar(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def extract_th_scalar_series(th, name) -> list[float] | None:
    if name not in th:
        return None
    s = th[name]
    if not isinstance(s, list) or len(s) != 500:
        return None
    out = []
    for v in s:
        if not is_scalar(v):
            return None
        out.append(float(v))
    return out


def extract_th_per_feature_reduced(th, name, reduce_fn) -> list[float] | None:
    """e.g., name='train_feature_recon_mean' (list[500] of list[features]).
    Apply reduce_fn over features for each epoch."""
    if name not in th:
        return None
    s = th[name]
    if not isinstance(s, list) or len(s) != 500:
        return None
    out = []
    for v in s:
        if v is None:
            return None
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            arr = np.asarray(v, dtype=float)
            try:
                out.append(float(reduce_fn(arr)))
            except (ValueError, FloatingPointError):
                return None
        else:
            return None
    return out


def extract_em_scalar_series(em, name) -> tuple[list[float], list[int]] | None:
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


def extract_em_per_feature_reduced(em, name, reduce_fn) -> tuple[list[float], list[int]] | None:
    vals, eps = [], []
    for entry in em:
        if name not in entry:
            return None
        v = entry[name]
        if v is None:
            return None
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            arr = np.asarray(v, dtype=float)
            try:
                vals.append(float(reduce_fn(arr)))
                eps.append(int(entry["epoch"]))
            except (ValueError, FloatingPointError):
                return None
        else:
            return None
    return vals, eps


# ---------------- Post-process operators ----------------

def op_raw(series):
    return series


def op_ema(series, alpha=0.3):
    if not series:
        return series
    out = [series[0]]
    for v in series[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def op_slope(series, window=10):
    """Recent slope = mean of last window's diff."""
    if len(series) < window + 1:
        return [0.0] * len(series)
    out = [0.0] * len(series)
    arr = np.asarray(series, dtype=float)
    diffs = np.diff(arr)
    # For position i, slope = mean(diffs[i-window:i])
    for i in range(window, len(series)):
        out[i] = float(np.mean(diffs[max(0, i - window):i]))
    return out


def op_curvature(series, window=10):
    """2nd derivative (mean over recent window)."""
    if len(series) < window + 2:
        return [0.0] * len(series)
    arr = np.asarray(series, dtype=float)
    d1 = np.diff(arr)
    d2 = np.diff(d1)
    out = [0.0] * len(series)
    for i in range(window + 1, len(series)):
        out[i] = float(np.mean(d2[max(0, i - window):i]))
    return out


def op_variance(series, window=10):
    if len(series) < window:
        return [0.0] * len(series)
    out = [0.0] * len(series)
    arr = np.asarray(series, dtype=float)
    for i in range(window, len(series)):
        out[i] = float(np.std(arr[i - window:i]))
    return out


def op_sign_changes(series, window=10):
    """Count of sign changes in recent diff window (oscillation)."""
    if len(series) < window + 1:
        return [0.0] * len(series)
    arr = np.asarray(series, dtype=float)
    diffs = np.diff(arr)
    signs = np.sign(diffs)
    out = [0.0] * len(series)
    for i in range(window, len(series)):
        seg = signs[max(0, i - window):i]
        # Count zero crossings (sign changes excluding zero)
        changes = int(np.sum(seg[1:] != seg[:-1]))
        out[i] = float(changes)
    return out


POST_OPS = [
    ("raw", lambda s: op_raw(s)),
    ("ema03", lambda s: op_ema(s, alpha=0.3)),
    ("slope10", lambda s: op_slope(s, window=10)),
    ("slope20", lambda s: op_slope(s, window=20)),
    ("curvature10", lambda s: op_curvature(s, window=10)),
    ("variance10", lambda s: op_variance(s, window=10)),
    ("sign_changes10", lambda s: op_sign_changes(s, window=10)),
]


# ---------------- Derived dynamic metrics ----------------

def safe_div(a, b, eps=1e-8):
    return [x / (abs(y) + eps) for x, y in zip(a, b)]


def safe_abs(s):
    return [abs(x) for x in s]


def safe_sub(a, b):
    return [x - y for x, y in zip(a, b)]


def safe_add_abs(a, b):
    return [abs(x) + abs(y) for x, y in zip(a, b)]


def windowed_abs_diff(series, window=10):
    """|x[t] - x[t-window]| series."""
    out = [0.0] * len(series)
    for i in range(window, len(series)):
        out[i] = abs(series[i] - series[i - window])
    return out


def windowed_diff(series, window=10):
    """x[t] - x[t-window] (signed)."""
    out = [0.0] * len(series)
    for i in range(window, len(series)):
        out[i] = series[i] - series[i - window]
    return out


def build_derived_series(th, em):
    """Construct user-proposed + brainstormed dynamic derived metrics.
    Returns dict {name: (values, epochs, direction)}.
    """
    out = {}
    epochs_all = list(range(1, 501))

    def th_get(name):
        return extract_th_scalar_series(th, name)

    teacher_n = th_get("train_teacher_recon_normal")
    teacher_a = th_get("train_teacher_recon_anomaly")
    student_n = th_get("train_student_recon_normal")
    student_a = th_get("train_student_recon_anomaly")
    recon_score_n = th_get("epoch_recon_score_normal")
    recon_score_a = th_get("epoch_recon_score_anomaly")
    recon_ratio_a = th_get("epoch_recon_ratio_anomaly")
    recon_ratio_n = th_get("epoch_recon_ratio_normal")
    disc_score_n = th_get("epoch_disc_score_normal")
    disc_score_a = th_get("epoch_disc_score_anomaly")
    train_loss = th_get("train_loss")
    train_rec_loss = th_get("train_rec_loss")
    train_anom_loss = th_get("train_anomaly_loss")
    train_norm_loss = th_get("train_normal_loss")
    fm_lambda = th_get("train_fm_adaptive_lambda")
    grl_lambda = th_get("train_grl_lambda")
    grl_eff_w = th_get("train_grl_effective_weight")
    grl_bal_acc = th_get("train_grl_balanced_acc")
    grl_anom_acc = th_get("train_grl_anomaly_acc")
    grl_norm_acc = th_get("train_grl_normal_acc")

    # ── Gaps, ratios, separations (static, every epoch) ──
    def add(name, vals, direction):
        if vals is None:
            return
        out[name] = (vals, epochs_all, direction)

    add("deriv_teacher_anom_normal_gap", safe_sub(teacher_a, teacher_n) if teacher_a and teacher_n else None, "max")
    add("deriv_teacher_anom_normal_ratio", safe_div(teacher_a, teacher_n) if teacher_a and teacher_n else None, "max")
    add("deriv_teacher_anom_normal_separation",
        [(a - n_) / max(abs(a) + abs(n_), 1e-8) for a, n_ in zip(teacher_a, teacher_n)] if teacher_a and teacher_n else None,
        "max")
    add("deriv_student_anom_normal_gap", safe_sub(student_a, student_n) if student_a and student_n else None, "max")
    add("deriv_student_anom_normal_ratio", safe_div(student_a, student_n) if student_a and student_n else None, "max")
    add("deriv_student_anom_normal_separation",
        [(a - n_) / max(abs(a) + abs(n_), 1e-8) for a, n_ in zip(student_a, student_n)] if student_a and student_n else None,
        "max")
    add("deriv_recon_score_gap", safe_sub(recon_score_a, recon_score_n) if recon_score_a and recon_score_n else None, "max")
    add("deriv_recon_score_separation",
        [(a - n_) / max(abs(a) + abs(n_), 1e-8) for a, n_ in zip(recon_score_a, recon_score_n)] if recon_score_a and recon_score_n else None,
        "max")
    add("deriv_disc_score_gap", safe_sub(disc_score_a, disc_score_n) if disc_score_a and disc_score_n else None, "max")
    add("deriv_disc_score_separation",
        [(a - n_) / max(abs(a) + abs(n_), 1e-8) for a, n_ in zip(disc_score_a, disc_score_n)] if disc_score_a and disc_score_n else None,
        "max")

    # ── Teacher-Student discrepancy proxies ──
    if teacher_n and student_n:
        add("deriv_TS_disagreement_normal", safe_sub(student_n, teacher_n), "min")
        add("deriv_TS_disagreement_normal_abs", safe_abs(safe_sub(student_n, teacher_n)), "min")
    if teacher_a and student_a:
        add("deriv_TS_disagreement_anomaly", safe_sub(student_a, teacher_a), "max")
        add("deriv_TS_disagreement_anomaly_abs", safe_abs(safe_sub(student_a, teacher_a)), "max")

    # Relative gap to total recon (scale-invariant version)
    if teacher_n and student_n and train_rec_loss:
        add("deriv_TS_disagreement_normal_relative",
            [abs(s - t) / max(abs(rl), 1e-8) for s, t, rl in zip(student_n, teacher_n, train_rec_loss)],
            "min")

    # ── USER-PROPOSED: |Δteacher_recon| / |Δstudent_recon| ──
    # Δ over window W = 5, 10, 20 — when ratio → 1, both decrease at same rate (stop signal).
    for W in (5, 10, 20):
        if teacher_n and student_n:
            dT = windowed_abs_diff(teacher_n, window=W)
            dS = windowed_abs_diff(student_n, window=W)
            # ratio (teacher/student) — when ≈ 1 means similar decrease rates
            r_ts_normal = [t / max(s, 1e-12) for t, s in zip(dT, dS)]
            add(f"deriv_dteacher_over_dstudent_normal_W{W}", r_ts_normal, "max")
            # also signed difference
            add(f"deriv_dteacher_minus_dstudent_normal_W{W}_abs",
                [abs(t - s) for t, s in zip(dT, dS)], "min")
        if teacher_a and student_a:
            dTa = windowed_abs_diff(teacher_a, window=W)
            dSa = windowed_abs_diff(student_a, window=W)
            r_ts_anom = [t / max(s, 1e-12) for t, s in zip(dTa, dSa)]
            add(f"deriv_dteacher_over_dstudent_anomaly_W{W}", r_ts_anom, "max")
            add(f"deriv_dteacher_minus_dstudent_anomaly_W{W}_abs",
                [abs(t - s) for t, s in zip(dTa, dSa)], "min")

        # Δ(teacher - student) — gap stability over window
        if teacher_n and student_n:
            gap = safe_sub(student_n, teacher_n)
            add(f"deriv_gap_TS_normal_dW{W}_abs", windowed_abs_diff(gap, window=W), "min")
        if teacher_a and student_a:
            gapa = safe_sub(student_a, teacher_a)
            add(f"deriv_gap_TS_anomaly_dW{W}_abs", windowed_abs_diff(gapa, window=W), "min")

        # Δ(anom - normal) recon score separation
        if recon_score_a and recon_score_n:
            sep = safe_sub(recon_score_a, recon_score_n)
            add(f"deriv_recon_score_separation_dW{W}", windowed_diff(sep, window=W), "max")
            add(f"deriv_recon_score_separation_dW{W}_abs", windowed_abs_diff(sep, window=W), "min")

        # Anomaly loss vs normal loss balance
        if train_anom_loss and train_norm_loss:
            ratio = [a / max(n_, 1e-12) for a, n_ in zip(train_anom_loss, train_norm_loss)]
            add(f"deriv_anom_normal_loss_ratio", ratio, "max")
            add(f"deriv_anom_normal_loss_ratio_dW{W}_abs", windowed_abs_diff(ratio, window=W), "min")

        # Adaptive lambda stabilization
        if fm_lambda:
            add(f"deriv_fm_lambda_dW{W}_abs", windowed_abs_diff(fm_lambda, window=W), "min")
        if grl_lambda:
            add(f"deriv_grl_lambda_dW{W}_abs", windowed_abs_diff(grl_lambda, window=W), "min")
        if grl_eff_w:
            add(f"deriv_grl_effective_weight_dW{W}_abs", windowed_abs_diff(grl_eff_w, window=W), "min")

        # GRL accuracy stabilization
        if grl_bal_acc:
            add(f"deriv_grl_balanced_acc_dW{W}_abs", windowed_abs_diff(grl_bal_acc, window=W), "min")

        # Recon loss decrease rate
        if train_rec_loss:
            add(f"deriv_train_rec_loss_dW{W}_abs", windowed_abs_diff(train_rec_loss, window=W), "min")
        if train_loss:
            add(f"deriv_train_loss_dW{W}_abs", windowed_abs_diff(train_loss, window=W), "min")

    # ── GRL classifier bias dynamics ──
    if grl_anom_acc and grl_norm_acc:
        add("deriv_grl_classifier_bias",
            [a - n_ for a, n_ in zip(grl_anom_acc, grl_norm_acc)],
            "max")
        add("deriv_grl_classifier_bias_abs",
            [abs(a - n_) for a, n_ in zip(grl_anom_acc, grl_norm_acc)],
            "max")

    # ── Disagreement scaled by training progress ──
    if teacher_n and student_n and train_rec_loss:
        scaled = [abs(s - t) / max(rl, 1e-8) for s, t, rl in zip(student_n, teacher_n, train_rec_loss)]
        add("deriv_TS_disagreement_normal_per_loss", scaled, "min")

    # ── EM-side derived: PA-K curve dynamics ──
    em_pa_keys_f1 = [f"pa_{k}_f1" for k in range(0, 105, 5)]
    em_pa_keys_prc = [f"pa_{k}_prc_auc" for k in range(0, 105, 5)]
    pa_series_f1 = []
    em_epochs = []
    for entry in em:
        row = []
        valid = True
        for k in em_pa_keys_f1:
            v = entry.get(k)
            if v is None:
                valid = False
                break
            row.append(float(v))
        if not valid:
            pa_series_f1 = []
            break
        pa_series_f1.append(row)
        em_epochs.append(int(entry["epoch"]))

    if pa_series_f1:
        # PA-K curve spread: pa_50_f1 - pa_0_f1
        spread_f1 = [r[10] - r[0] for r in pa_series_f1]  # idx 10 = pa_50
        add_em = lambda name, vals, d: out.update({name: (vals, em_epochs, d)})
        add_em("deriv_pa_K_curve_spread_50_0", spread_f1, "max")
        # PA-K curve mean over K
        mean_pa = [float(np.mean(r)) for r in pa_series_f1]
        add_em("deriv_pa_K_curve_mean_f1", mean_pa, "max")
        # PA-K curve area (trapezoid)
        area_pa = [float(np.trapz(r, dx=5)) for r in pa_series_f1]
        add_em("deriv_pa_K_curve_area_f1", area_pa, "max")
        # PA-K curve slope: |slope of pa_K_f1 over K| at each epoch
        slopes_curve = [float(np.mean(np.diff(r))) for r in pa_series_f1]
        add_em("deriv_pa_K_curve_avg_slope_over_K", slopes_curve, "min")  # near 0 means flat = good

    return out


# ---------------- Early stopping ----------------

@dataclass
class ESResult:
    stop_epoch: int


def early_stopping(values, epochs, direction, patience, thresh_type, thresh_value,
                   warmup=WARMUP_EPOCH) -> ESResult:
    eval_pts = [(v, e) for v, e in zip(values, epochs)
                if e >= warmup and e % EVAL_INTERVAL == 0]
    if not eval_pts:
        return ESResult(stop_epoch=epochs[-1])

    best_value = eval_pts[0][0]
    counter = 0
    stop_epoch = eval_pts[-1][1]

    def is_improvement(new, current):
        if direction == "max":
            delta = new - current
        else:
            delta = current - new
        if thresh_type == "abs":
            return delta > thresh_value
        denom = max(abs(current), 1e-8)
        return (delta / denom) > thresh_value

    for v, e in eval_pts[1:]:
        if is_improvement(v, best_value):
            best_value = v
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                stop_epoch = e
                break
    return ESResult(stop_epoch=stop_epoch)


def lookup_em_at_epoch(em, epoch, key):
    for entry in em:
        if entry["epoch"] == epoch:
            return entry.get(key)
    closest = min(em, key=lambda e: abs(e["epoch"] - epoch))
    return closest.get(key)


# ---------------- Collect metrics ----------------

def collect_all_metrics(ds):
    """Collect every plausible scalar series from th + em, plus derived dynamics.
    No metrics are excluded — we want the full pool.
    Returns: dict {name: (values, epochs, direction)}
    """
    metrics = {}
    th = ds["th"]
    em = ds["em"]

    # 1) training_histories scalar series (every key that's list[500] of scalars)
    for key in th.keys():
        s = extract_th_scalar_series(th, key)
        if s is not None:
            epochs = list(range(1, 501))
            metrics[f"th_{key}"] = (s, epochs, metric_direction(key))

    # 2) training_histories per-feature lists → reduced scalars
    for key in th.keys():
        if not key.startswith("train_feature_"):
            continue
        for red_name, red_fn in [("mean", np.mean), ("max", np.max),
                                  ("std", np.std), ("min", np.min)]:
            s = extract_th_per_feature_reduced(th, key, red_fn)
            if s is not None:
                epochs = list(range(1, 501))
                metrics[f"th_{key}__feat_{red_name}"] = (s, epochs, metric_direction(key))

    # 3) epoch_metrics scalar keys (all 176)
    if em:
        ep0 = em[0]
        for key in sorted(ep0.keys()):
            if key in EM_METADATA_KEYS:
                continue
            v = ep0[key]
            if isinstance(v, (int, float)):
                r = extract_em_scalar_series(em, key)
                if r is not None:
                    vals, eps = r
                    metrics[f"em_{key}"] = (vals, eps, metric_direction(key))

    # 4) epoch_metrics per-feature list keys → reduced
    if em:
        ep0 = em[0]
        for key in sorted(ep0.keys()):
            v = ep0[key]
            if isinstance(v, list):
                for red_name, red_fn in [("mean", np.mean), ("max", np.max),
                                          ("std", np.std), ("min", np.min)]:
                    r = extract_em_per_feature_reduced(em, key, red_fn)
                    if r is not None:
                        vals, eps = r
                        metrics[f"em_{key}__feat_{red_name}"] = (vals, eps, metric_direction(key))

    # 5) Derived dynamics (user-proposed + brainstormed)
    derived = build_derived_series(th, em)
    metrics.update(derived)

    return metrics


# ---------------- Sweep ----------------

def oracle_best(em, key="pak_auc_f1"):
    best = max(em, key=lambda e: e.get(key, -1e18))
    return best["epoch"], best.get(key)


def sweep_dataset(ds_name, ds, score_key="pak_auc_f1"):
    metrics = collect_all_metrics(ds)
    em_score = ds["em_score"]
    oracle_ep, oracle_val = oracle_best(em_score, score_key)

    rows = []
    for mname, (vals, epochs, direction) in metrics.items():
        # Apply post-process operators
        for op_name, op_fn in POST_OPS:
            try:
                series = op_fn(vals)
            except Exception:
                continue
            if not series or len(series) != len(vals):
                continue

            for P in PATIENCE_GRID:
                for ttype, tval in THRESHOLD_GRID:
                    r = early_stopping(series, epochs, direction, P, ttype, tval)
                    lookup_val = lookup_em_at_epoch(em_score, r.stop_epoch, score_key)
                    if lookup_val is None:
                        continue
                    rows.append({
                        "metric": mname,
                        "op": op_name,
                        "direction": direction,
                        "patience": P,
                        "thresh_type": ttype,
                        "thresh_value": tval,
                        "stop_epoch": r.stop_epoch,
                        "pak_auc_f1_at_stop": float(lookup_val),
                    })

    return {
        "dataset": ds_name,
        "oracle_epoch": oracle_ep,
        "oracle_pak_auc_f1": float(oracle_val) if oracle_val is not None else None,
        "n_metrics": len(metrics),
        "n_rows": len(rows),
        "rows": rows,
    }


def _worker(args):
    """Multiprocessing worker — loads ds, runs sweep, returns (name, result, rss_mb)."""
    import os
    import gc
    import psutil
    name, paths = args
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss / 1e6
    ds = load_dataset(name, paths)
    result = sweep_dataset(name, ds)
    del ds
    gc.collect()
    rss_after = proc.memory_info().rss / 1e6
    return name, result, rss_before, rss_after


def main():
    import psutil
    import multiprocessing as mp
    import time

    print(f"System: {mp.cpu_count()} CPUs, {psutil.virtual_memory().total / 1e9:.1f} GB RAM")
    print(f"Loading 1 dataset to probe metric pool size...")
    sample_ds = load_dataset("SWaT_excl22", DATASETS["SWaT_excl22"])
    probe = collect_all_metrics(sample_ds)
    n_metrics = len(probe)
    n_pt = len(PATIENCE_GRID) * len(THRESHOLD_GRID)
    n_ops = len(POST_OPS)
    expected_rows_per_ds = n_metrics * n_ops * n_pt
    print(f"  metric pool: {n_metrics}")
    print(f"  post-ops: {n_ops}")
    print(f"  P × T grid: {n_pt}")
    print(f"  expected rows per dataset: ~{expected_rows_per_ds:,}")
    print(f"  expected total rows: ~{expected_rows_per_ds * len(DATASETS):,}")
    del sample_ds, probe

    n_workers = 6  # conservative: 6 × ~1.5 GB = 9 GB peak (room within 33 GB)
    print(f"\nStarting sweep with {n_workers} parallel workers...")
    main_rss_start = psutil.Process(os.getpid()).memory_info().rss / 1e6
    print(f"  main process RSS at start: {main_rss_start:.0f} MB")

    items = list(DATASETS.items())
    all_results = {}
    t0 = time.time()

    with mp.Pool(processes=n_workers) as pool:
        for i, (name, result, rss_b, rss_a) in enumerate(
            pool.imap_unordered(_worker, items, chunksize=1)
        ):
            all_results[name] = result
            elapsed = time.time() - t0
            sys_mem_pct = psutil.virtual_memory().percent
            main_rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
            print(f"  [{i+1:2d}/{len(items)}] {name:30s}  "
                  f"rows={result['n_rows']:>7d}  oracle={result['oracle_pak_auc_f1']:.4f}  "
                  f"worker_RSS {rss_b:.0f}→{rss_a:.0f}MB  "
                  f"sys_mem={sys_mem_pct:.0f}%  main_RSS={main_rss:.0f}MB  "
                  f"t={elapsed:.0f}s")

    print(f"\nTotal time: {time.time() - t0:.1f}s")

    out_path = Path("/home/ykio/notebooks/claude/temp/early_stopping/sweep_raw_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing JSON to {out_path}...")
    with open(out_path, "w") as f:
        json.dump(all_results, f, separators=(",", ":"))
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    import os  # required for main()
    main()
