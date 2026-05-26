"""Early Stopping Analysis v5 — second-peak detection 특화.

사용자 통찰: student_recon_anomaly는 warmup 직후 급락 후 짧은 시점에 second peak (hump)를 보임.
이 second peak가 oracle epoch과 가까운 dataset 다수.

기존 peak_reversal이 warmup 시점 (ep 250) 의 높은 값 (1.6) 을 initial peak로 잡아버려서
그 후의 작은 second peak (0.4-0.7) 를 못 잡음.

v5 4가지 새 접근:
A. Warmup ablation (warmup ∈ {250, 260, 270, 280}) — initial peak를 second peak 위치로 이동
B. peak_reversal_with_reset — 급락 (>50% in 1 step) 검출 시 peak_v reset
C. pre_warmup_baseline_spike — warmup 직전 baseline 위로 spike 검출 후 감소 detect
D. first_local_max — warmup 이후 처음 만나는 local max (3-point criterion)

v5 grid (메모리 절약):
- 93 metrics × 3 ops (raw, ema03, slope10)
- 3 dir (auto, force_max, force_min)
- 2 rollback (stop, best_seen)
- **5 ES rules** (standard, peak_reversal, peak_reversal_reset, baseline_spike, first_local_max)
- 2 P (1, 2)
- 4 thresholds (abs=0, abs=0.001, rel=0.001, rel=0.01)
- **4 warmups** (250, 260, 270, 280)
- = 93 × 3 × 3 × 2 × 5 × 2 × 4 × 4 = 535,680 per ds × 25 = ~13.4M rows
"""
from __future__ import annotations

import json
import os
import time
import gc
import multiprocessing as mp
from pathlib import Path
from statistics import mean as stat_mean

import numpy as np
import psutil

EXP_ROOT = Path(
    "/home/ykio/notebooks/claude/results/experiments/"
    "271_20260508_094241_w500p10e4t3d2_dynamic_linear_minmax_k6"
)

SMD_15 = ["machine-1-2","machine-1-7","machine-2-1","machine-2-2","machine-2-3","machine-2-4",
          "machine-2-6","machine-2-7","machine-2-9","machine-3-1","machine-3-2","machine-3-3",
          "machine-3-6","machine-3-8","machine-3-9"]
EXATHLON_APPS = ["app1","app2","app4","app5","app6","app9"]

DATASETS = {
    "SWaT_excl22": ("SWaT/A1A2_full", "SWaT/A1A2_excl22"),
    "WaDi_A1": ("WaDi/A1","WaDi/A1"),
    "WaDi_A2": ("WaDi/A2","WaDi/A2"),
    "PSM": ("PSM","PSM"),
}
for m in SMD_15:
    DATASETS[f"SMD_{m}"] = (f"SMD/{m}", f"SMD/{m}")
for a in EXATHLON_APPS:
    DATASETS[f"Exathlon_{a}"] = (f"Exathlon/{a}", f"Exathlon/{a}")

EVAL_INTERVAL = 5
PATIENCE_GRID = [1, 2]
THRESHOLD_GRID = [("abs", 0.0), ("abs", 0.001), ("rel", 0.001), ("rel", 0.01)]
DIRECTION_MODES = ["auto", "force_max", "force_min"]
ROLLBACK_MODES = ["stop_at_trigger", "best_seen_before_stop"]
WARMUP_GRID = [250, 260, 270, 280]
ES_RULES = ["standard", "peak_reversal", "peak_reversal_reset",
            "baseline_spike", "first_local_max", "post_drop_peak", "kth_peak_2"]

LABEL_BASED_PATTERNS = [
    "em_pak_auc", "em_pa_", "em_teacher_", "em_f1_", "em_prc_auc", "em_roc_auc",
    "em_precision", "em_recall", "em_disturbing", "em_disc_snr", "em_optimal_threshold",
    "em_fm_loss", "em_grl_", "deriv_pa_K_curve",
]

def is_label_based(name):
    return any(name.startswith(p) or f"_{p}" in name for p in LABEL_BASED_PATTERNS)


def auto_direction(name):
    n = name.lower()
    if any(s in n for s in ("acc", "auc", "f1", "snr", "precision", "recall",
                             "detection_rate", "separation", "spread", "ratio", "gap",
                             "product", "safety", "health", "indicator", "velocity",
                             "imbalance")):
        return "max"
    if any(s in n for s in ("loss", "discrepancy", "recon", "error", "_score",
                             "threshold", "magnitude", "norm", "_disc", "_raw")):
        return "min"
    return "min"


def load_dataset(name, paths):
    train_rel, score_rel = paths
    with open(EXP_ROOT / train_rel / "training_histories.json") as f:
        th_raw = json.load(f)
    th = th_raw[list(th_raw.keys())[0]]
    with open(EXP_ROOT / train_rel / "epoch_metrics.json") as f:
        em_train = json.load(f)["epochs"]
    if EXP_ROOT / score_rel == EXP_ROOT / train_rel:
        em_score = em_train
    else:
        with open(EXP_ROOT / score_rel / "epoch_metrics.json") as f:
            em_score = json.load(f)["epochs"]
    return {"th": th, "em": em_train, "em_score": em_score}


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


def extract_th_per_feature(th, name, fn):
    if name not in th:
        return None
    s = th[name]
    if not isinstance(s, list) or len(s) != 500:
        return None
    out = []
    for v in s:
        if v is None or not isinstance(v, list) or not v or not isinstance(v[0], (int, float)):
            return None
        try:
            out.append(float(fn(np.asarray(v, dtype=float))))
        except (ValueError, FloatingPointError):
            return None
    return out


def op_raw(s): return s
def op_ema(s, a=0.3):
    if not s: return s
    out = [s[0]]
    for v in s[1:]: out.append(a*v + (1-a)*out[-1])
    return out
def op_slope(s, w=10):
    if len(s) < w+1: return [0.0]*len(s)
    a = np.asarray(s, dtype=float); d = np.diff(a)
    out = [0.0]*len(s)
    for i in range(w, len(s)):
        out[i] = float(np.mean(d[max(0,i-w):i]))
    return out


POST_OPS = [("raw", op_raw), ("ema03", op_ema), ("slope10", op_slope)]


def sub(a, b): return [x - y for x, y in zip(a, b)]
def div(a, b, eps=1e-8): return [x / (abs(y) + eps) for x, y in zip(a, b)]
def w_abs_diff(s, W):
    out = [0.0] * len(s)
    for i in range(W, len(s)):
        out[i] = abs(s[i] - s[i - W])
    return out
def separation(a, n, eps=1e-8):
    return [(x - y) / max(abs(x) + abs(y), eps) for x, y in zip(a, n)]
def zscore(s):
    a = np.asarray(s, dtype=float)
    mu, sd = a.mean(), a.std()
    if sd < 1e-12: return [0.0] * len(s)
    return ((a - mu) / sd).tolist()


def build_derived(th):
    out = {}
    eps_all = list(range(1, 501))
    g = lambda k: extract_th_scalar(th, k)
    teacher_n = g("train_teacher_recon_normal"); teacher_a = g("train_teacher_recon_anomaly")
    student_n = g("train_student_recon_normal"); student_a = g("train_student_recon_anomaly")
    recon_n = g("epoch_recon_score_normal"); recon_a = g("epoch_recon_score_anomaly")
    disc_n = g("epoch_disc_score_normal"); disc_a = g("epoch_disc_score_anomaly")
    train_loss = g("train_loss"); train_rec_loss = g("train_rec_loss")
    train_anom_loss = g("train_anomaly_loss"); train_norm_loss = g("train_normal_loss")
    fm_lambda = g("train_fm_adaptive_lambda"); grl_lambda = g("train_grl_lambda")
    grl_a = g("train_grl_anomaly_acc"); grl_nrm = g("train_grl_normal_acc")

    def add(name, vals, direction):
        if vals is None: return
        out[name] = (vals, eps_all, direction)

    if teacher_a and teacher_n:
        add("deriv_teacher_anom_normal_gap", sub(teacher_a, teacher_n), "max")
        add("deriv_teacher_anom_normal_ratio", div(teacher_a, teacher_n), "max")
        add("deriv_teacher_anom_normal_separation", separation(teacher_a, teacher_n), "max")
    if student_a and student_n:
        add("deriv_student_anom_normal_gap", sub(student_a, student_n), "max")
        add("deriv_student_anom_normal_ratio", div(student_a, student_n), "max")
        add("deriv_student_anom_normal_separation", separation(student_a, student_n), "max")
    if recon_a and recon_n:
        add("deriv_recon_score_gap", sub(recon_a, recon_n), "max")
        add("deriv_recon_score_separation", separation(recon_a, recon_n), "max")
    if disc_a and disc_n:
        add("deriv_disc_score_gap", sub(disc_a, disc_n), "max")
        add("deriv_disc_score_separation", separation(disc_a, disc_n), "max")

    if teacher_n and student_n:
        add("deriv_TS_disagreement_normal_abs",
            [abs(s-t) for s, t in zip(student_n, teacher_n)], "min")
    if teacher_a and student_a:
        add("deriv_TS_disagreement_anomaly_abs",
            [abs(s-t) for s, t in zip(student_a, teacher_a)], "max")

    for W in (5, 10, 20):
        if teacher_n and student_n:
            dT = w_abs_diff(teacher_n, W); dS = w_abs_diff(student_n, W)
            add(f"deriv_dteacher_over_dstudent_normal_W{W}",
                [t / max(s, 1e-12) for t, s in zip(dT, dS)], "max")
            add(f"deriv_dteacher_minus_dstudent_normal_W{W}_abs",
                [abs(t - s) for t, s in zip(dT, dS)], "min")
        if teacher_a and student_a:
            dTa = w_abs_diff(teacher_a, W); dSa = w_abs_diff(student_a, W)
            add(f"deriv_dteacher_over_dstudent_anomaly_W{W}",
                [t / max(s, 1e-12) for t, s in zip(dTa, dSa)], "max")
            add(f"deriv_dteacher_minus_dstudent_anomaly_W{W}_abs",
                [abs(t - s) for t, s in zip(dTa, dSa)], "min")

    # v4 composite (top 5)
    if disc_a and recon_a and recon_n:
        recon_sep = separation(recon_a, recon_n)
        add("composite_disc_x_separation_product",
            [d*s for d, s in zip(disc_a, recon_sep)], "max")
    if student_a and teacher_a:
        add("composite_student_anom_over_teacher_anom_ratio",
            [s / max(t, 1e-12) for s, t in zip(student_a, teacher_a)], "max")
    if teacher_n and student_n:
        add("composite_learning_phase_indicator", sub(student_n, teacher_n), "max")
    return out


def collect_metrics(ds):
    th = ds["th"]
    metrics = {}
    for key in th.keys():
        s = extract_th_scalar(th, key)
        if s is not None:
            metrics[f"th_{key}"] = (s, list(range(1, 501)), auto_direction(key))
    for key in th.keys():
        if not key.startswith("train_feature_"): continue
        for n, fn in [("mean", np.mean), ("max", np.max), ("std", np.std), ("min", np.min)]:
            s = extract_th_per_feature(th, key, fn)
            if s is not None:
                metrics[f"th_{key}__feat_{n}"] = (s, list(range(1, 501)), auto_direction(key))
    metrics.update(build_derived(th))
    metrics = {k: v for k, v in metrics.items() if not is_label_based(k)}
    return metrics


# ============================================================
# ES Algorithms
# ============================================================

def _gather_eval_points(values, epochs, warmup, direction):
    ev = [(v, e) for v, e in zip(values, epochs)
          if e >= warmup and e % EVAL_INTERVAL == 0]
    if direction == "min":
        ev = [(-v, e) for v, e in ev]
    return ev


def es_standard(values, epochs, direction, P, ttype, tval, rollback, warmup):
    ev = _gather_eval_points(values, epochs, warmup, direction)
    if not ev: return epochs[-1], epochs[-1]
    best_v, best_e = ev[0]; counter = 0; stop_e = ev[-1][1]
    def improved(new, cur):
        delta = new - cur  # ev already sign-flipped if min
        if ttype == "abs": return delta > tval
        return (delta / max(abs(cur), 1e-8)) > tval
    for v, e in ev[1:]:
        if improved(v, best_v):
            best_v, best_e = v, e; counter = 0
        else:
            counter += 1
            if counter >= P:
                stop_e = e; break
    return (best_e, best_e) if rollback == "best_seen_before_stop" else (stop_e, best_e)


def es_peak_reversal(values, epochs, direction, P, ttype, tval, rollback, warmup):
    ev = _gather_eval_points(values, epochs, warmup, direction)
    if not ev: return epochs[-1], epochs[-1]
    peak_v, peak_e = ev[0]; drop_count = 0; stop_e = ev[-1][1]
    for v, e in ev[1:]:
        if v > peak_v:
            peak_v, peak_e = v, e; drop_count = 0
        else:
            drop = peak_v - v
            sig = (drop > tval) if ttype == "abs" else ((drop / max(abs(peak_v), 1e-8)) > tval)
            if sig:
                drop_count += 1
                if drop_count >= P:
                    stop_e = e; break
            else:
                drop_count = 0
    return (peak_e, peak_e) if rollback == "best_seen_before_stop" else (stop_e, peak_e)


def es_peak_reversal_reset(values, epochs, direction, P, ttype, tval, rollback, warmup,
                            big_drop_thr=0.5):
    """B. Big drop (>50%) detected → reset peak. 학습 단계 전환 후 second peak 검출.
    """
    ev = _gather_eval_points(values, epochs, warmup, direction)
    if not ev: return epochs[-1], epochs[-1]
    peak_v, peak_e = ev[0]; drop_count = 0; stop_e = ev[-1][1]
    prev_v = peak_v
    for v, e in ev[1:]:
        # Big drop detection (step-to-step)
        step_drop = (prev_v - v) / max(abs(prev_v), 1e-8)
        if step_drop > big_drop_thr:
            peak_v, peak_e = v, e
            drop_count = 0
            prev_v = v
            continue
        if v > peak_v:
            peak_v, peak_e = v, e; drop_count = 0
        else:
            drop = peak_v - v
            sig = (drop > tval) if ttype == "abs" else ((drop / max(abs(peak_v), 1e-8)) > tval)
            if sig:
                drop_count += 1
                if drop_count >= P:
                    stop_e = e; break
            else:
                drop_count = 0
        prev_v = v
    return (peak_e, peak_e) if rollback == "best_seen_before_stop" else (stop_e, peak_e)


def es_baseline_spike(values, epochs, direction, P, ttype, tval, rollback, warmup):
    """C. Pre-warmup baseline spike: baseline = mean(series[warmup-5:warmup]).
    Detect first spike above baseline+threshold, then peak_reversal from there.
    """
    # baseline: 5 ep before warmup
    baseline_ws = [v for v, e in zip(values, epochs) if warmup - 5 <= e < warmup]
    if not baseline_ws: return epochs[-1], epochs[-1]
    baseline = sum(baseline_ws) / len(baseline_ws)

    ev = _gather_eval_points(values, epochs, warmup, direction)
    if not ev: return epochs[-1], epochs[-1]
    if direction == "min":
        baseline = -baseline

    spike_started = False
    peak_v, peak_e = baseline, warmup
    drop_count = 0
    stop_e = ev[-1][1]

    for v, e in ev:
        rise = v - baseline
        rise_sig = (rise > tval) if ttype == "abs" else (rise / max(abs(baseline), 1e-8) > tval)
        if not spike_started:
            if rise_sig:
                spike_started = True
                peak_v, peak_e = v, e
        else:
            if v > peak_v:
                peak_v, peak_e = v, e; drop_count = 0
            else:
                drop = peak_v - v
                sig = (drop > tval) if ttype == "abs" else ((drop / max(abs(peak_v), 1e-8)) > tval)
                if sig:
                    drop_count += 1
                    if drop_count >= P:
                        stop_e = e; break
                else:
                    drop_count = 0
    if not spike_started:
        return ev[-1][1], ev[-1][1]
    return (peak_e, peak_e) if rollback == "best_seen_before_stop" else (stop_e, peak_e)


def es_first_local_max(values, epochs, direction, P, ttype, tval, rollback, warmup):
    """D. First local max: warmup 이후 처음 만나는 local max에서 stop.
    Local max criterion: v[i] > v[i-1] AND v[i+j] < v[i] for j in 1..P (significant)
    """
    ev = _gather_eval_points(values, epochs, warmup, direction)
    if len(ev) < P + 2: return ev[-1][1] if ev else epochs[-1], ev[-1][1] if ev else epochs[-1]
    for i in range(1, len(ev) - P):
        v_curr, e_curr = ev[i]
        v_prev, _ = ev[i-1]
        if v_curr <= v_prev: continue
        # Check all next P epochs are significantly below v_curr
        all_below = True
        for j in range(1, P + 1):
            v_next, _ = ev[i + j]
            drop = v_curr - v_next
            sig = (drop > tval) if ttype == "abs" else ((drop / max(abs(v_curr), 1e-8)) > tval)
            if not sig:
                all_below = False; break
        if all_below:
            peak_e = e_curr
            stop_e = ev[i + P][1]
            return (peak_e, peak_e) if rollback == "best_seen_before_stop" else (stop_e, peak_e)
    # No local max found — use last
    return ev[-1][1], ev[-1][1]


def es_post_drop_peak(values, epochs, direction, P, ttype, tval, rollback, warmup,
                       drop_thr=0.3):
    """★ Second-peak detector. 사용자 통찰:
    'warmup 직후 peak → 감소 → 다시 폭증(second peak) → 다시 감소' 패턴 잡기.

    1. warmup 이후 first peak tracking
    2. Significant drop (>30% from current peak) 검출 → 학습 단계 전환점
    3. 그 drop 이후 NEW peak tracking 시작 (current value를 second peak baseline)
    4. Second peak 검출 + 그 후 patience 동안 drop 지속 → stop, peak 시점 반환
    """
    ev = _gather_eval_points(values, epochs, warmup, direction)
    if len(ev) < 4:
        return ev[-1][1] if ev else epochs[-1], ev[-1][1] if ev else epochs[-1]

    state = "looking_for_drop"
    initial_peak = ev[0][0]
    second_peak_v = None
    second_peak_e = None
    drop_count = 0
    stop_e = ev[-1][1]

    for v, e in ev[1:]:
        if state == "looking_for_drop":
            if v > initial_peak:
                initial_peak = v
            else:
                rel_drop = (initial_peak - v) / max(abs(initial_peak), 1e-8)
                if rel_drop > drop_thr:
                    state = "tracking_second_peak"
                    second_peak_v = v
                    second_peak_e = e
                    drop_count = 0
        else:  # tracking_second_peak
            if v > second_peak_v:
                second_peak_v = v
                second_peak_e = e
                drop_count = 0
            else:
                drop = second_peak_v - v
                sig = (drop > tval) if ttype == "abs" else ((drop / max(abs(second_peak_v), 1e-8)) > tval)
                if sig:
                    drop_count += 1
                    if drop_count >= P:
                        stop_e = e
                        return (second_peak_e, second_peak_e) if rollback == "best_seen_before_stop" else (stop_e, second_peak_e)
                else:
                    drop_count = 0

    # If second peak was tracked but no confirmed drop, use last second peak
    if second_peak_e is not None:
        return (second_peak_e, second_peak_e) if rollback == "best_seen_before_stop" else (stop_e, second_peak_e)
    # If no drop ever detected, fall back
    return ev[-1][1], ev[-1][1]


def es_kth_peak(values, epochs, direction, P, ttype, tval, rollback, warmup, k=2):
    """k-th local maximum after warmup.
    Local max: v[i-1] < v[i] AND next P epochs all significantly below v[i].
    Iterate until k-th such max found.
    """
    ev = _gather_eval_points(values, epochs, warmup, direction)
    if len(ev) < P + 2:
        return ev[-1][1] if ev else epochs[-1], ev[-1][1] if ev else epochs[-1]
    peaks = []
    i = 1
    while i < len(ev) - P:
        v_curr, e_curr = ev[i]
        v_prev, _ = ev[i-1]
        if v_curr <= v_prev:
            i += 1
            continue
        all_below = True
        for j in range(1, P + 1):
            v_next, _ = ev[i + j]
            drop = v_curr - v_next
            sig = (drop > tval) if ttype == "abs" else ((drop / max(abs(v_curr), 1e-8)) > tval)
            if not sig:
                all_below = False
                break
        if all_below:
            peaks.append((v_curr, e_curr, ev[i + P][1]))
            if len(peaks) >= k:
                pv, pe, te = peaks[k-1]
                return (pe, pe) if rollback == "best_seen_before_stop" else (te, pe)
            i += P + 1
        else:
            i += 1
    if peaks:
        pv, pe, te = peaks[-1]
        return (pe, pe) if rollback == "best_seen_before_stop" else (te, pe)
    return ev[-1][1], ev[-1][1]


def es_kth_peak_2(values, epochs, direction, P, ttype, tval, rollback, warmup):
    return es_kth_peak(values, epochs, direction, P, ttype, tval, rollback, warmup, k=2)


ES_RULE_FN = {
    "standard": es_standard,
    "peak_reversal": es_peak_reversal,
    "peak_reversal_reset": es_peak_reversal_reset,
    "baseline_spike": es_baseline_spike,
    "first_local_max": es_first_local_max,
    "post_drop_peak": es_post_drop_peak,
    "kth_peak_2": es_kth_peak_2,
}


def lookup(em, ep, key="pak_auc_f1"):
    for entry in em:
        if entry["epoch"] == ep: return entry.get(key)
    cl = min(em, key=lambda e: abs(e["epoch"] - ep))
    return cl.get(key)


def sweep_dataset(name, ds):
    metrics = collect_metrics(ds)
    em_score = ds["em_score"]
    oracle = max(em_score, key=lambda e: e.get("pak_auc_f1", -1e18))
    rows = []
    for mname, (vals, eps, auto_dir) in metrics.items():
        for op_name, op_fn in POST_OPS:
            try: s = op_fn(vals)
            except Exception: continue
            if not s or len(s) != len(vals): continue
            for dir_mode in DIRECTION_MODES:
                direction = auto_dir if dir_mode == "auto" else (
                    "max" if dir_mode == "force_max" else "min")
                for warmup in WARMUP_GRID:
                    for rule in ES_RULES:
                        fn = ES_RULE_FN[rule]
                        for P in PATIENCE_GRID:
                            for tt, tv in THRESHOLD_GRID:
                                for rb in ROLLBACK_MODES:
                                    se, pe = fn(s, eps, direction, P, tt, tv, rb, warmup)
                                    v = lookup(em_score, se)
                                    if v is None: continue
                                    rows.append({
                                        "m": mname, "op": op_name, "d": dir_mode,
                                        "rb": rb, "rule": rule, "P": P,
                                        "tt": tt, "tv": tv, "w": warmup,
                                        "se": se, "v": float(v),
                                    })
    return {
        "dataset": name,
        "oracle_epoch": oracle["epoch"],
        "oracle_pak_auc_f1": float(oracle.get("pak_auc_f1")),
        "n_metrics": len(metrics),
        "n_rows": len(rows),
        "rows": rows,
    }


def _worker(args):
    name, paths = args
    proc = psutil.Process(os.getpid())
    rb = proc.memory_info().rss / 1e6
    ds = load_dataset(name, paths)
    r = sweep_dataset(name, ds)
    del ds; gc.collect()
    return name, r, rb, proc.memory_info().rss / 1e6


def main():
    print(f"System: {mp.cpu_count()} CPUs, {psutil.virtual_memory().total/1e9:.1f} GB RAM")
    print("Probing v5...")
    ds = load_dataset("SWaT_excl22", DATASETS["SWaT_excl22"])
    probe = collect_metrics(ds)
    print(f"  metrics: {len(probe)}")
    grid = (len(POST_OPS) * len(DIRECTION_MODES) * len(WARMUP_GRID) *
            len(ES_RULES) * len(PATIENCE_GRID) * len(THRESHOLD_GRID) * len(ROLLBACK_MODES))
    print(f"  grid per metric: {grid}")
    print(f"  expected rows per ds: ~{len(probe)*grid:,}")
    print(f"  total: ~{len(probe)*grid*25:,}")
    del ds, probe

    t0 = time.time()
    results = {}
    with mp.Pool(processes=6) as pool:
        for i, (name, r, rb, ra) in enumerate(
            pool.imap_unordered(_worker, list(DATASETS.items()), chunksize=1)
        ):
            results[name] = r
            sys_pct = psutil.virtual_memory().percent
            print(f"  [{i+1:2d}/25] {name:30s} rows={r['n_rows']:>8d} "
                  f"oracle={r['oracle_pak_auc_f1']:.4f} "
                  f"worker_RSS {rb:.0f}→{ra:.0f}MB sys={sys_pct:.0f}% t={time.time()-t0:.0f}s")
    print(f"\nTotal: {time.time()-t0:.1f}s")
    out = Path("/home/ykio/notebooks/claude/temp/early_stopping/sweep_raw_v5.json")
    with open(out, "w") as f:
        json.dump(results, f, separators=(",", ":"))
    print(f"Saved {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
