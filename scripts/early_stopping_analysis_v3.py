"""Early Stopping Analysis v3 — feedback-driven.

User feedback applied:
1. Label-based evaluation metrics REMOVED (em_pak_auc_*, em_pa_K_*, em_f1_*, em_prc_auc, em_roc_auc,
   em_precision*, em_recall*, em_disturbing_*, disc_snr, deriv_pa_K_curve_*).
2. Patience grid: 1, 2, 3 only.
3. New ES rule "peak_reversal" — max-so-far seen, then ↓ trigger (rollback to peak).
   Specifically designed for Type B signals (student_recon_anomaly, disc_score_anomaly, separation).
4. Direction mode: auto / force_max / force_min (3).
5. Rollback mode: stop_at_trigger / best_seen_before_stop (2).
6. Ops reduced to 5: raw, ema03, slope10, curvature10, sign_changes10.

Model objectives reflected in derived metrics:
- Type A (convergence): teacher_recon_normal plateau, train_rec_loss plateau.
- Type B (degradation): student_recon_anomaly peak↓, disc_score_anomaly peak↓, separation peak↓.
"""
from __future__ import annotations

import json
import os
import time
import gc
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from statistics import mean as stat_mean

import numpy as np
import psutil

EXP_ROOT = Path(
    "/home/ykio/notebooks/claude/results/experiments/"
    "271_20260508_094241_w500p10e4t3d2_dynamic_linear_minmax_k6"
)

SMD_15 = ["machine-1-2", "machine-1-7", "machine-2-1", "machine-2-2", "machine-2-3",
          "machine-2-4", "machine-2-6", "machine-2-7", "machine-2-9", "machine-3-1",
          "machine-3-2", "machine-3-3", "machine-3-6", "machine-3-8", "machine-3-9"]
EXATHLON_APPS = ["app1", "app2", "app4", "app5", "app6", "app9"]

DATASETS = {
    "SWaT_excl22": ("SWaT/A1A2_full", "SWaT/A1A2_excl22"),
    "WaDi_A1": ("WaDi/A1", "WaDi/A1"),
    "WaDi_A2": ("WaDi/A2", "WaDi/A2"),
    "PSM": ("PSM", "PSM"),
}
for m in SMD_15:
    DATASETS[f"SMD_{m}"] = (f"SMD/{m}", f"SMD/{m}")
for a in EXATHLON_APPS:
    DATASETS[f"Exathlon_{a}"] = (f"Exathlon/{a}", f"Exathlon/{a}")

WARMUP_EPOCH = 250
EVAL_INTERVAL = 5

PATIENCE_GRID = [1, 2, 3]
THRESHOLD_GRID = [("abs", 0.0), ("abs", 0.001),
                  ("rel", 0.001), ("rel", 0.01), ("rel", 0.05)]
DIRECTION_MODES = ["auto", "force_max", "force_min"]
ROLLBACK_MODES = ["stop_at_trigger", "best_seen_before_stop"]
ES_RULES = ["standard", "peak_reversal"]


# ---------------- Label-based key filter ----------------
LABEL_BASED_PATTERNS = [
    # em_* eval metrics ALL excluded (per user instruction)
    "em_pak_auc", "em_pa_", "em_teacher_", "em_f1_", "em_prc_auc", "em_roc_auc",
    "em_precision", "em_recall", "em_disturbing", "em_disc_snr", "em_optimal_threshold",
    "em_fm_loss", "em_grl_",
    # Derived label-based
    "deriv_pa_K_curve",
]


def is_label_based(metric_name: str) -> bool:
    for pat in LABEL_BASED_PATTERNS:
        if metric_name.startswith(pat) or f"_{pat}" in metric_name:
            return True
    return False


# ---------------- Direction inference ----------------
def auto_direction(name: str) -> str:
    """Direction for `auto` mode."""
    n = name.lower()
    if any(s in n for s in ("acc", "auc", "f1", "snr", "precision", "recall",
                             "detection_rate", "separation", "spread", "ratio", "gap")):
        return "max"
    if any(s in n for s in ("loss", "discrepancy", "recon", "error", "_score",
                             "threshold", "magnitude", "norm", "_disc", "_raw")):
        return "min"
    return "min"


# ---------------- Loaders ----------------
def load_dataset(name, paths):
    train_rel, score_rel = paths
    train_base = EXP_ROOT / train_rel
    score_base = EXP_ROOT / score_rel
    with open(train_base / "training_histories.json") as f:
        th = json.load(f)[list(json.load(open(train_base / "training_histories.json")).keys())[0]]
    # Re-open to avoid double load
    with open(train_base / "training_histories.json") as f:
        th_raw = json.load(f)
    th = th_raw[list(th_raw.keys())[0]]
    with open(train_base / "epoch_metrics.json") as f:
        em_train = json.load(f)["epochs"]
    if score_base != train_base:
        with open(score_base / "epoch_metrics.json") as f:
            em_score = json.load(f)["epochs"]
    else:
        em_score = em_train
    return {"th": th, "em": em_train, "em_score": em_score}


# ---------------- Base series extraction ----------------
def is_scalar(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def extract_th_scalar(th, name):
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


def extract_th_per_feature(th, name, reduce_fn):
    if name not in th:
        return None
    s = th[name]
    if not isinstance(s, list) or len(s) != 500:
        return None
    out = []
    for v in s:
        if v is None or not isinstance(v, list) or not v or not isinstance(v[0], (int, float)):
            return None
        arr = np.asarray(v, dtype=float)
        try:
            out.append(float(reduce_fn(arr)))
        except (ValueError, FloatingPointError):
            return None
    return out


# ---------------- Post-process ops (label-free reduced set) ----------------
def op_raw(s):
    return s


def op_ema(s, alpha=0.3):
    if not s:
        return s
    out = [s[0]]
    for v in s[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def op_slope(s, window=10):
    if len(s) < window + 1:
        return [0.0] * len(s)
    arr = np.asarray(s, dtype=float)
    diffs = np.diff(arr)
    out = [0.0] * len(s)
    for i in range(window, len(s)):
        out[i] = float(np.mean(diffs[max(0, i - window):i]))
    return out


def op_curvature(s, window=10):
    if len(s) < window + 2:
        return [0.0] * len(s)
    arr = np.asarray(s, dtype=float)
    d2 = np.diff(np.diff(arr))
    out = [0.0] * len(s)
    for i in range(window + 1, len(s)):
        out[i] = float(np.mean(d2[max(0, i - window):i]))
    return out


def op_sign_changes(s, window=10):
    if len(s) < window + 1:
        return [0.0] * len(s)
    arr = np.asarray(s, dtype=float)
    diffs = np.diff(arr)
    signs = np.sign(diffs)
    out = [0.0] * len(s)
    for i in range(window, len(s)):
        seg = signs[max(0, i - window):i]
        out[i] = float(int(np.sum(seg[1:] != seg[:-1])))
    return out


POST_OPS = [
    ("raw", op_raw),
    ("ema03", op_ema),
    ("slope10", op_slope),
    ("curvature10", op_curvature),
    ("sign_changes10", op_sign_changes),
]


# ---------------- Derived dynamics (label-free only) ----------------
def safe_div(a, b, eps=1e-8):
    return [x / (abs(y) + eps) for x, y in zip(a, b)]


def safe_sub(a, b):
    return [x - y for x, y in zip(a, b)]


def windowed_abs_diff(s, W=10):
    out = [0.0] * len(s)
    for i in range(W, len(s)):
        out[i] = abs(s[i] - s[i - W])
    return out


def build_derived(th):
    """Derived dynamics — label-free only.
    Includes user-proposed Δratio / Δdiff and Type B (peak reversal candidates).
    """
    out = {}
    eps = list(range(1, 501))

    g = lambda k: extract_th_scalar(th, k)

    teacher_n = g("train_teacher_recon_normal")
    teacher_a = g("train_teacher_recon_anomaly")
    student_n = g("train_student_recon_normal")
    student_a = g("train_student_recon_anomaly")
    recon_score_n = g("epoch_recon_score_normal")
    recon_score_a = g("epoch_recon_score_anomaly")
    disc_score_n = g("epoch_disc_score_normal")
    disc_score_a = g("epoch_disc_score_anomaly")
    train_loss = g("train_loss")
    train_rec_loss = g("train_rec_loss")
    train_anom_loss = g("train_anomaly_loss")
    train_norm_loss = g("train_normal_loss")
    fm_lambda = g("train_fm_adaptive_lambda")
    grl_lambda = g("train_grl_lambda")
    grl_eff_w = g("train_grl_effective_weight")
    grl_bal_acc = g("train_grl_balanced_acc")
    grl_anom_acc = g("train_grl_anomaly_acc")
    grl_norm_acc = g("train_grl_normal_acc")

    def add(name, vals, direction):
        if vals is None:
            return
        out[name] = (vals, eps, direction)

    # ── Static separations (max direction) ──
    if teacher_a and teacher_n:
        add("deriv_teacher_anom_normal_gap", safe_sub(teacher_a, teacher_n), "max")
        add("deriv_teacher_anom_normal_ratio", safe_div(teacher_a, teacher_n), "max")
        add("deriv_teacher_anom_normal_separation",
            [(a - n) / max(abs(a) + abs(n), 1e-8) for a, n in zip(teacher_a, teacher_n)], "max")
    if student_a and student_n:
        add("deriv_student_anom_normal_gap", safe_sub(student_a, student_n), "max")
        add("deriv_student_anom_normal_ratio", safe_div(student_a, student_n), "max")
        add("deriv_student_anom_normal_separation",
            [(a - n) / max(abs(a) + abs(n), 1e-8) for a, n in zip(student_a, student_n)], "max")
    if recon_score_a and recon_score_n:
        add("deriv_recon_score_gap", safe_sub(recon_score_a, recon_score_n), "max")
        add("deriv_recon_score_separation",
            [(a - n) / max(abs(a) + abs(n), 1e-8) for a, n in zip(recon_score_a, recon_score_n)], "max")
    if disc_score_a and disc_score_n:
        add("deriv_disc_score_gap", safe_sub(disc_score_a, disc_score_n), "max")
        add("deriv_disc_score_separation",
            [(a - n) / max(abs(a) + abs(n), 1e-8) for a, n in zip(disc_score_a, disc_score_n)], "max")

    # ── Teacher-Student disagreement (mean_discrepancy proxy) ──
    if teacher_n and student_n:
        add("deriv_TS_disagreement_normal_abs",
            [abs(s - t) for s, t in zip(student_n, teacher_n)], "min")
    if teacher_a and student_a:
        add("deriv_TS_disagreement_anomaly_abs",
            [abs(s - t) for s, t in zip(student_a, teacher_a)], "max")

    # ── Anomaly/Normal loss balance ──
    if train_anom_loss and train_norm_loss:
        add("deriv_anom_normal_loss_ratio",
            [a / max(n, 1e-12) for a, n in zip(train_anom_loss, train_norm_loss)], "max")

    # ── GRL classifier bias ──
    if grl_anom_acc and grl_norm_acc:
        add("deriv_grl_classifier_bias",
            [a - n for a, n in zip(grl_anom_acc, grl_norm_acc)], "max")
        add("deriv_grl_classifier_bias_abs",
            [abs(a - n) for a, n in zip(grl_anom_acc, grl_norm_acc)], "max")

    # ── User-proposed: Δratio / Δdiff with W=5,10,20 ──
    for W in (5, 10, 20):
        if teacher_n and student_n:
            dT = windowed_abs_diff(teacher_n, W)
            dS = windowed_abs_diff(student_n, W)
            add(f"deriv_dteacher_over_dstudent_normal_W{W}",
                [t / max(s, 1e-12) for t, s in zip(dT, dS)], "max")
            add(f"deriv_dteacher_minus_dstudent_normal_W{W}_abs",
                [abs(t - s) for t, s in zip(dT, dS)], "min")
        if teacher_a and student_a:
            dTa = windowed_abs_diff(teacher_a, W)
            dSa = windowed_abs_diff(student_a, W)
            add(f"deriv_dteacher_over_dstudent_anomaly_W{W}",
                [t / max(s, 1e-12) for t, s in zip(dTa, dSa)], "max")
            add(f"deriv_dteacher_minus_dstudent_anomaly_W{W}_abs",
                [abs(t - s) for t, s in zip(dTa, dSa)], "min")

        # Gap stability
        if teacher_n and student_n:
            gap = safe_sub(student_n, teacher_n)
            add(f"deriv_gap_TS_normal_dW{W}_abs", windowed_abs_diff(gap, W), "min")
        if teacher_a and student_a:
            gapa = safe_sub(student_a, teacher_a)
            add(f"deriv_gap_TS_anomaly_dW{W}_abs", windowed_abs_diff(gapa, W), "min")

        # Adaptive coefficient stabilization
        if fm_lambda:
            add(f"deriv_fm_lambda_dW{W}_abs", windowed_abs_diff(fm_lambda, W), "min")
        if grl_lambda:
            add(f"deriv_grl_lambda_dW{W}_abs", windowed_abs_diff(grl_lambda, W), "min")

    return out


# ---------------- Collect metrics (label-free only) ----------------
def collect_label_free_metrics(ds):
    th = ds["th"]
    metrics = {}

    # 1) training_histories scalar series (ALL list[500] scalars)
    for key in th.keys():
        s = extract_th_scalar(th, key)
        if s is not None:
            metrics[f"th_{key}"] = (s, list(range(1, 501)), auto_direction(key))

    # 2) training_histories per-feature reductions
    for key in th.keys():
        if not key.startswith("train_feature_"):
            continue
        for red_name, red_fn in [("mean", np.mean), ("max", np.max),
                                  ("std", np.std), ("min", np.min)]:
            s = extract_th_per_feature(th, key, red_fn)
            if s is not None:
                metrics[f"th_{key}__feat_{red_name}"] = (
                    s, list(range(1, 501)), auto_direction(key))

    # 3) Derived dynamics (label-free)
    metrics.update(build_derived(th))

    # Filter out any label-based that slipped through (paranoid)
    metrics = {k: v for k, v in metrics.items() if not is_label_based(k)}
    return metrics


# ---------------- ES Algorithms ----------------
def es_standard(values, epochs, direction, P, ttype, tval, rollback, warmup=WARMUP_EPOCH):
    """Standard best-based ES with patience+threshold.
    Returns (stop_epoch, best_epoch_seen)."""
    eval_pts = [(v, e) for v, e in zip(values, epochs)
                if e >= warmup and e % EVAL_INTERVAL == 0]
    if not eval_pts:
        return epochs[-1], epochs[-1]
    best_v, best_e = eval_pts[0]
    counter = 0
    stop_e = eval_pts[-1][1]

    def improved(new, cur):
        delta = (new - cur) if direction == "max" else (cur - new)
        if ttype == "abs":
            return delta > tval
        return (delta / max(abs(cur), 1e-8)) > tval

    for v, e in eval_pts[1:]:
        if improved(v, best_v):
            best_v = v
            best_e = e
            counter = 0
        else:
            counter += 1
            if counter >= P:
                stop_e = e
                break
    if rollback == "best_seen_before_stop":
        return best_e, best_e
    return stop_e, best_e


def es_peak_reversal(values, epochs, direction, P, ttype, tval, rollback, warmup=WARMUP_EPOCH):
    """Peak reversal: detect max-so-far, then if drop continues for P eval points → stop.
    Rollback applies the same way (best_seen returns the peak epoch).
    """
    eval_pts = [(v, e) for v, e in zip(values, epochs)
                if e >= warmup and e % EVAL_INTERVAL == 0]
    if not eval_pts:
        return epochs[-1], epochs[-1]

    if direction == "min":
        # Treat as valley reversal: negate values to reuse logic
        eval_pts = [(-v, e) for v, e in eval_pts]

    peak_v, peak_e = eval_pts[0]
    drop_count = 0
    stop_e = eval_pts[-1][1]

    for v, e in eval_pts[1:]:
        if v > peak_v:  # new peak
            peak_v = v
            peak_e = e
            drop_count = 0
        else:
            # Significant drop?
            drop = peak_v - v
            if ttype == "abs":
                significant = drop > tval
            else:  # rel
                significant = (drop / max(abs(peak_v), 1e-8)) > tval
            if significant:
                drop_count += 1
                if drop_count >= P:
                    stop_e = e
                    break
            else:
                drop_count = 0  # reset on negligible drop

    if rollback == "best_seen_before_stop":
        return peak_e, peak_e
    return stop_e, peak_e


# ---------------- Lookup ----------------
def lookup_em_at_epoch(em, epoch, key="pak_auc_f1"):
    for entry in em:
        if entry["epoch"] == epoch:
            return entry.get(key)
    closest = min(em, key=lambda e: abs(e["epoch"] - epoch))
    return closest.get(key)


# ---------------- Sweep ----------------
def sweep_dataset(ds_name, ds):
    metrics = collect_label_free_metrics(ds)
    em_score = ds["em_score"]
    oracle_v = max(em_score, key=lambda e: e.get("pak_auc_f1", -1e18))
    oracle_ep = oracle_v["epoch"]
    oracle_pak = oracle_v.get("pak_auc_f1")

    rows = []
    for mname, (vals, eps, auto_dir) in metrics.items():
        for op_name, op_fn in POST_OPS:
            try:
                series = op_fn(vals)
            except Exception:
                continue
            if not series or len(series) != len(vals):
                continue
            for dir_mode in DIRECTION_MODES:
                if dir_mode == "auto":
                    direction = auto_dir
                elif dir_mode == "force_max":
                    direction = "max"
                else:
                    direction = "min"
                for P in PATIENCE_GRID:
                    for ttype, tval in THRESHOLD_GRID:
                        for rollback in ROLLBACK_MODES:
                            for rule in ES_RULES:
                                if rule == "standard":
                                    stop_e, peak_e = es_standard(series, eps, direction,
                                                                 P, ttype, tval, rollback)
                                else:
                                    stop_e, peak_e = es_peak_reversal(series, eps, direction,
                                                                      P, ttype, tval, rollback)
                                lookup_val = lookup_em_at_epoch(em_score, stop_e)
                                if lookup_val is None:
                                    continue
                                rows.append({
                                    "metric": mname,
                                    "op": op_name,
                                    "dir": dir_mode,
                                    "rollback": rollback,
                                    "rule": rule,
                                    "P": P,
                                    "tt": ttype,
                                    "tv": tval,
                                    "se": stop_e,
                                    "v": float(lookup_val),
                                })
    return {
        "dataset": ds_name,
        "oracle_epoch": oracle_ep,
        "oracle_pak_auc_f1": float(oracle_pak) if oracle_pak is not None else None,
        "n_metrics": len(metrics),
        "n_rows": len(rows),
        "rows": rows,
    }


def _worker(args):
    name, paths = args
    proc = psutil.Process(os.getpid())
    rss_b = proc.memory_info().rss / 1e6
    ds = load_dataset(name, paths)
    result = sweep_dataset(name, ds)
    del ds
    gc.collect()
    rss_a = proc.memory_info().rss / 1e6
    return name, result, rss_b, rss_a


def main():
    print(f"System: {mp.cpu_count()} CPUs, {psutil.virtual_memory().total/1e9:.1f} GB RAM")
    print("Probing metric pool on SWaT...")
    probe_ds = load_dataset("SWaT_excl22", DATASETS["SWaT_excl22"])
    probe = collect_label_free_metrics(probe_ds)
    print(f"  label-free metric pool: {len(probe)}")

    grid_size = (len(POST_OPS) * len(DIRECTION_MODES) * len(PATIENCE_GRID)
                 * len(THRESHOLD_GRID) * len(ROLLBACK_MODES) * len(ES_RULES))
    print(f"  grid per metric: {grid_size}")
    print(f"  expected rows per dataset: ~{len(probe) * grid_size:,}")
    print(f"  total across 25 datasets: ~{len(probe) * grid_size * 25:,}")
    del probe_ds, probe

    n_workers = 6
    print(f"\nStarting sweep with {n_workers} workers...")
    main_rss_start = psutil.Process(os.getpid()).memory_info().rss / 1e6
    print(f"  main RSS start: {main_rss_start:.0f} MB")
    t0 = time.time()

    all_results = {}
    with mp.Pool(processes=n_workers) as pool:
        for i, (name, result, rss_b, rss_a) in enumerate(
            pool.imap_unordered(_worker, list(DATASETS.items()), chunksize=1)
        ):
            all_results[name] = result
            elapsed = time.time() - t0
            sys_pct = psutil.virtual_memory().percent
            main_rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
            print(f"  [{i+1:2d}/25] {name:30s} rows={result['n_rows']:>7d} "
                  f"oracle={result['oracle_pak_auc_f1']:.4f} "
                  f"worker_RSS {rss_b:.0f}→{rss_a:.0f}MB "
                  f"sys={sys_pct:.0f}% main_RSS={main_rss:.0f}MB t={elapsed:.0f}s")

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    out_path = Path("/home/ykio/notebooks/claude/temp/early_stopping/sweep_raw_v3.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("Writing JSON...")
    with open(out_path, "w") as f:
        json.dump(all_results, f, separators=(",", ":"))
    print(f"Saved {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
