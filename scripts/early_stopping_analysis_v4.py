"""Early Stopping Analysis v4 — composite metrics from first-principles.

Self-Distilled MAE의 구성요소별 anomaly detection 목적을 재분석:
- Teacher (3-layer): 풍부한 capacity, 정상 학습. Phase 3에선 anomaly까지 학습 → BAD.
- Student (2-layer): 제한 capacity로 anomaly fit 불가가 핵심. anomaly recon ↑ 유지가 GOOD.
- Discrepancy = Teacher − Student: 정상 작게, 이상 크게. anomaly에서 peak ↓ = STOP.
- Separation (anom − normal recon): peak ↓ = STOP.

Phase 정의 (warmup=250 반영):
- Pre-warmup oracle (10/25, 40%): warmup 너무 김. 250 시점에 이미 phase 3.
- Just-after-warmup (9/25, 36%): 250-300 ep 안에 분리도 peak. peak_reversal 즉시 trigger 필요.
- Late phase 2 (6/25, 24%): 300+ ep 까지 분리도 ↑.

새 composite metrics (직접 고안):
A) `disc_x_separation_product` = disc_score_anomaly × recon_score_separation
   → 두 검출 신호 동시 peak. peak_reversal max.
B) `student_anom_over_teacher_anom_ratio` = student_recon_anomaly / teacher_recon_anomaly
   → 1로 수렴 = student가 teacher 따라잡음 = STOP. peak_reversal max.
C) `learning_phase_indicator` = teacher_recon_normal − student_recon_normal
   → 격차 ↓ = student over-distillation. peak_reversal max.
D) `composite_anomaly_separation` = z-score normalized sum of 3 separations
E) `type_a_x_type_b` = disc_score_anomaly / max(train_rec_loss, ε)
   → 학습 수렴도 × 분리력. peak_reversal max.
F) `student_anom_velocity` = -d(student_recon_anomaly)/dt rolling window
   → 양수면 student가 anomaly에 학습 시작 → 즉시 STOP.
G) `disc_anom_velocity` = d(disc_score_anomaly)/dt rolling
   → 음수면 검출력 손실 → STOP. force_max standard.
H) `unified_anomaly_health` = weighted sum of 5 normalized anomaly-favoring signals
I) `student_capacity_safety_margin` = min(student_anom_normal_sep, teacher_anom_normal_sep)
   → 약점 head 기준 분리도. peak_reversal max.
J) `anom_normal_loss_imbalance` = train_anomaly_loss × recon_score_separation
   → anomaly hardness × 분리도. peak_reversal max.
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


# ---------------- Post-process ----------------
def op_raw(s): return s
def op_ema(s, a=0.3):
    if not s: return s
    out = [s[0]]
    for v in s[1:]:
        out.append(a * v + (1 - a) * out[-1])
    return out
def op_slope(s, w=10):
    if len(s) < w + 1: return [0.0] * len(s)
    a = np.asarray(s, dtype=float)
    d = np.diff(a)
    out = [0.0] * len(s)
    for i in range(w, len(s)):
        out[i] = float(np.mean(d[max(0, i - w):i]))
    return out
def op_curvature(s, w=10):
    if len(s) < w + 2: return [0.0] * len(s)
    a = np.asarray(s, dtype=float)
    d2 = np.diff(np.diff(a))
    out = [0.0] * len(s)
    for i in range(w + 1, len(s)):
        out[i] = float(np.mean(d2[max(0, i - w):i]))
    return out
def op_sign_changes(s, w=10):
    if len(s) < w + 1: return [0.0] * len(s)
    a = np.asarray(s, dtype=float)
    signs = np.sign(np.diff(a))
    out = [0.0] * len(s)
    for i in range(w, len(s)):
        seg = signs[max(0, i - w):i]
        out[i] = float(int(np.sum(seg[1:] != seg[:-1])))
    return out


POST_OPS = [("raw", op_raw), ("ema03", op_ema), ("slope10", op_slope),
            ("curvature10", op_curvature), ("sign_changes10", op_sign_changes)]


# ---------------- Helpers ----------------
def sub(a, b): return [x - y for x, y in zip(a, b)]
def div(a, b, eps=1e-8): return [x / (abs(y) + eps) for x, y in zip(a, b)]
def mul(a, b): return [x * y for x, y in zip(a, b)]
def w_abs_diff(s, W):
    out = [0.0] * len(s)
    for i in range(W, len(s)):
        out[i] = abs(s[i] - s[i - W])
    return out
def w_diff(s, W):
    out = [0.0] * len(s)
    for i in range(W, len(s)):
        out[i] = s[i] - s[i - W]
    return out
def separation(a, n, eps=1e-8):
    return [(x - y) / max(abs(x) + abs(y), eps) for x, y in zip(a, n)]
def zscore(s):
    a = np.asarray(s, dtype=float)
    mu, sd = a.mean(), a.std()
    if sd < 1e-12:
        return [0.0] * len(s)
    return ((a - mu) / sd).tolist()


def build_derived(th):
    """v3 derived + v4 새 composite metrics."""
    out = {}
    eps_all = list(range(1, 501))
    g = lambda k: extract_th_scalar(th, k)

    teacher_n = g("train_teacher_recon_normal")
    teacher_a = g("train_teacher_recon_anomaly")
    student_n = g("train_student_recon_normal")
    student_a = g("train_student_recon_anomaly")
    recon_n = g("epoch_recon_score_normal")
    recon_a = g("epoch_recon_score_anomaly")
    disc_n = g("epoch_disc_score_normal")
    disc_a = g("epoch_disc_score_anomaly")
    train_loss = g("train_loss")
    train_rec_loss = g("train_rec_loss")
    train_anom_loss = g("train_anomaly_loss")
    train_norm_loss = g("train_normal_loss")
    fm_lambda = g("train_fm_adaptive_lambda")
    grl_lambda = g("train_grl_lambda")
    grl_bal = g("train_grl_balanced_acc")
    grl_a = g("train_grl_anomaly_acc")
    grl_nrm = g("train_grl_normal_acc")

    def add(name, vals, direction):
        if vals is None: return
        out[name] = (vals, eps_all, direction)

    # === v3 carry-over: 정상-이상 분리도 (static) ===
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

    # TS disagreement
    if teacher_n and student_n:
        add("deriv_TS_disagreement_normal_abs",
            [abs(s - t) for s, t in zip(student_n, teacher_n)], "min")
    if teacher_a and student_a:
        add("deriv_TS_disagreement_anomaly_abs",
            [abs(s - t) for s, t in zip(student_a, teacher_a)], "max")

    # User-proposed Δratio / Δdiff
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
        if teacher_n and student_n:
            add(f"deriv_gap_TS_normal_dW{W}_abs",
                w_abs_diff(sub(student_n, teacher_n), W), "min")
        if teacher_a and student_a:
            add(f"deriv_gap_TS_anomaly_dW{W}_abs",
                w_abs_diff(sub(student_a, teacher_a), W), "min")
        if fm_lambda:
            add(f"deriv_fm_lambda_dW{W}_abs", w_abs_diff(fm_lambda, W), "min")
        if grl_lambda:
            add(f"deriv_grl_lambda_dW{W}_abs", w_abs_diff(grl_lambda, W), "min")

    if train_anom_loss and train_norm_loss:
        add("deriv_anom_normal_loss_ratio",
            [a / max(n, 1e-12) for a, n in zip(train_anom_loss, train_norm_loss)], "max")
    if grl_a and grl_nrm:
        add("deriv_grl_classifier_bias", [a - n for a, n in zip(grl_a, grl_nrm)], "max")
        add("deriv_grl_classifier_bias_abs", [abs(a - n) for a, n in zip(grl_a, grl_nrm)], "max")

    # ================================================================
    # ★ v4 신규 COMPOSITE METRICS (직접 고안)
    # ================================================================

    # A. disc × separation product (두 검출 신호 동시 peak)
    if disc_a and recon_a and recon_n:
        recon_sep = separation(recon_a, recon_n)
        product = [d * s for d, s in zip(disc_a, recon_sep)]
        add("composite_disc_x_separation_product", product, "max")

    # B. student_a / teacher_a ratio (1로 수렴 = student over-distill)
    if student_a and teacher_a:
        ratio = [s / max(t, 1e-12) for s, t in zip(student_a, teacher_a)]
        add("composite_student_anom_over_teacher_anom_ratio", ratio, "max")

    # C. Learning phase indicator: teacher_n - student_n
    if teacher_n and student_n:
        # student가 더 큰 normal recon = 격차 큼 = phase 2, 격차 줄어들면 student over-distill
        # student_n - teacher_n 이 정상 — 격차가 max일 때 STOP
        add("composite_learning_phase_indicator", sub(student_n, teacher_n), "max")

    # D. Composite anomaly separation (z-score weighted sum of 3 separations)
    seps = []
    if disc_a and disc_n:
        seps.append(zscore(separation(disc_a, disc_n)))
    if recon_a and recon_n:
        seps.append(zscore(separation(recon_a, recon_n)))
    if student_a and student_n:
        seps.append(zscore(separation(student_a, student_n)))
    if len(seps) >= 2:
        n_seps = len(seps)
        composite = [sum(s[i] for s in seps) / n_seps for i in range(500)]
        add("composite_anomaly_separation_ensemble", composite, "max")

    # E. Type A × Type B: 학습 수렴도 (1/train_rec_loss) × 분리력 (disc_anomaly)
    if disc_a and train_rec_loss:
        composite = [d / max(rl, 1e-6) for d, rl in zip(disc_a, train_rec_loss)]
        add("composite_type_a_x_type_b", composite, "max")

    # F. Student anomaly velocity: -d(student_recon_anomaly)/dt
    # student가 anomaly에 학습되는 속도. 양수 = phase 3. force_min standard.
    if student_a:
        sa_slope = op_slope(student_a, w=10)
        add("composite_student_anom_velocity_negative",
            [-x for x in sa_slope], "max")  # max for "stay low"

    # G. Disc anomaly velocity: d(disc_score_anomaly)/dt
    # 양수 = 분리력 증가, 음수 = 검출력 손실. force_max standard.
    if disc_a:
        da_slope = op_slope(disc_a, w=10)
        add("composite_disc_anom_velocity", da_slope, "max")

    # H. Unified anomaly health score: z-score weighted sum of 5 metrics
    signals = []
    if disc_a:
        signals.append(zscore(disc_a))  # higher = good
    if recon_a and recon_n:
        signals.append(zscore(separation(recon_a, recon_n)))  # higher = good
    if student_a and teacher_a:
        # student_a - teacher_a: student가 anomaly에 fit 안 함 = 차이 큼 = good
        signals.append(zscore([s - t for s, t in zip(student_a, teacher_a)]))
    if train_anom_loss and train_norm_loss:
        signals.append(zscore(
            [a / max(n, 1e-12) for a, n in zip(train_anom_loss, train_norm_loss)]))
    if disc_a and disc_n:
        signals.append(zscore(separation(disc_a, disc_n)))
    if len(signals) >= 3:
        n_sig = len(signals)
        unified = [sum(s[i] for s in signals) / n_sig for i in range(500)]
        add("composite_unified_anomaly_health", unified, "max")

    # I. Student capacity safety margin: min(student_sep, teacher_sep)
    if student_a and student_n and teacher_a and teacher_n:
        ss = separation(student_a, student_n)
        ts = separation(teacher_a, teacher_n)
        margin = [min(s, t) for s, t in zip(ss, ts)]
        add("composite_student_capacity_safety_margin", margin, "max")

    # J. Anom-normal loss imbalance × separation
    if train_anom_loss and train_norm_loss and recon_a and recon_n:
        ratio = [a / max(n, 1e-12) for a, n in zip(train_anom_loss, train_norm_loss)]
        sep = separation(recon_a, recon_n)
        imb = mul(ratio, sep)
        add("composite_anom_normal_loss_imbalance_x_sep", imb, "max")

    return out


def collect_label_free(ds):
    th = ds["th"]
    metrics = {}
    for key in th.keys():
        s = extract_th_scalar(th, key)
        if s is not None:
            metrics[f"th_{key}"] = (s, list(range(1, 501)), auto_direction(key))
    for key in th.keys():
        if not key.startswith("train_feature_"):
            continue
        for n, fn in [("mean", np.mean), ("max", np.max), ("std", np.std), ("min", np.min)]:
            s = extract_th_per_feature(th, key, fn)
            if s is not None:
                metrics[f"th_{key}__feat_{n}"] = (s, list(range(1, 501)), auto_direction(key))
    metrics.update(build_derived(th))
    metrics = {k: v for k, v in metrics.items() if not is_label_based(k)}
    return metrics


# ---------------- ES Algorithms ----------------
def es_standard(values, epochs, direction, P, ttype, tval, rollback):
    ev = [(v, e) for v, e in zip(values, epochs)
          if e >= WARMUP_EPOCH and e % EVAL_INTERVAL == 0]
    if not ev:
        return epochs[-1], epochs[-1]
    best_v, best_e = ev[0]
    counter = 0
    stop_e = ev[-1][1]
    def improved(new, cur):
        delta = (new - cur) if direction == "max" else (cur - new)
        if ttype == "abs": return delta > tval
        return (delta / max(abs(cur), 1e-8)) > tval
    for v, e in ev[1:]:
        if improved(v, best_v):
            best_v, best_e = v, e
            counter = 0
        else:
            counter += 1
            if counter >= P:
                stop_e = e
                break
    return (best_e, best_e) if rollback == "best_seen_before_stop" else (stop_e, best_e)


def es_peak_reversal(values, epochs, direction, P, ttype, tval, rollback):
    ev = [(v, e) for v, e in zip(values, epochs)
          if e >= WARMUP_EPOCH and e % EVAL_INTERVAL == 0]
    if not ev:
        return epochs[-1], epochs[-1]
    if direction == "min":
        ev = [(-v, e) for v, e in ev]
    peak_v, peak_e = ev[0]
    drop_count = 0
    stop_e = ev[-1][1]
    for v, e in ev[1:]:
        if v > peak_v:
            peak_v, peak_e = v, e
            drop_count = 0
        else:
            drop = peak_v - v
            sig = (drop > tval) if ttype == "abs" else ((drop / max(abs(peak_v), 1e-8)) > tval)
            if sig:
                drop_count += 1
                if drop_count >= P:
                    stop_e = e
                    break
            else:
                drop_count = 0
    return (peak_e, peak_e) if rollback == "best_seen_before_stop" else (stop_e, peak_e)


def lookup(em, ep, key="pak_auc_f1"):
    for entry in em:
        if entry["epoch"] == ep:
            return entry.get(key)
    cl = min(em, key=lambda e: abs(e["epoch"] - ep))
    return cl.get(key)


def sweep_dataset(name, ds):
    metrics = collect_label_free(ds)
    em_score = ds["em_score"]
    oracle = max(em_score, key=lambda e: e.get("pak_auc_f1", -1e18))
    rows = []
    for mname, (vals, eps, auto_dir) in metrics.items():
        for op_name, op_fn in POST_OPS:
            try:
                s = op_fn(vals)
            except Exception:
                continue
            if not s or len(s) != len(vals):
                continue
            for dir_mode in DIRECTION_MODES:
                direction = auto_dir if dir_mode == "auto" else (
                    "max" if dir_mode == "force_max" else "min")
                for P in PATIENCE_GRID:
                    for tt, tv in THRESHOLD_GRID:
                        for rb in ROLLBACK_MODES:
                            for rule in ES_RULES:
                                fn = es_standard if rule == "standard" else es_peak_reversal
                                se, pe = fn(s, eps, direction, P, tt, tv, rb)
                                v = lookup(em_score, se)
                                if v is None: continue
                                rows.append({
                                    "metric": mname, "op": op_name, "dir": dir_mode,
                                    "rollback": rb, "rule": rule, "P": P,
                                    "tt": tt, "tv": tv, "se": se, "v": float(v),
                                })
    return {
        "dataset": name, "oracle_epoch": oracle["epoch"],
        "oracle_pak_auc_f1": float(oracle.get("pak_auc_f1")),
        "n_metrics": len(metrics), "n_rows": len(rows), "rows": rows,
    }


def _worker(args):
    name, paths = args
    proc = psutil.Process(os.getpid())
    rb = proc.memory_info().rss / 1e6
    ds = load_dataset(name, paths)
    r = sweep_dataset(name, ds)
    del ds
    gc.collect()
    return name, r, rb, proc.memory_info().rss / 1e6


def main():
    print(f"System: {mp.cpu_count()} CPUs, {psutil.virtual_memory().total/1e9:.1f} GB RAM")
    print("Probing v4 metric pool...")
    ds = load_dataset("SWaT_excl22", DATASETS["SWaT_excl22"])
    probe = collect_label_free(ds)
    composite_count = sum(1 for k in probe if k.startswith("composite_"))
    print(f"  total: {len(probe)}, composite (v4 new): {composite_count}")
    grid = (len(POST_OPS) * len(DIRECTION_MODES) * len(PATIENCE_GRID)
            * len(THRESHOLD_GRID) * len(ROLLBACK_MODES) * len(ES_RULES))
    print(f"  expected rows/ds: ~{len(probe)*grid:,}")
    del ds, probe

    t0 = time.time()
    results = {}
    with mp.Pool(processes=6) as pool:
        for i, (name, r, rb, ra) in enumerate(
            pool.imap_unordered(_worker, list(DATASETS.items()), chunksize=1)
        ):
            results[name] = r
            sys_pct = psutil.virtual_memory().percent
            print(f"  [{i+1:2d}/25] {name:30s} rows={r['n_rows']:>7d} "
                  f"oracle={r['oracle_pak_auc_f1']:.4f} "
                  f"worker_RSS {rb:.0f}→{ra:.0f}MB sys={sys_pct:.0f}% t={time.time()-t0:.0f}s")
    print(f"\nTotal: {time.time()-t0:.1f}s")
    out = Path("/home/ykio/notebooks/claude/temp/early_stopping/sweep_raw_v4.json")
    with open(out, "w") as f:
        json.dump(results, f, separators=(",", ":"))
    print(f"Saved {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
