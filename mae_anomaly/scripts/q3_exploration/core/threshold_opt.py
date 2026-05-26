"""
Threshold optimization utilities.

Currently pak_auc_f1 sweeps 200 thresholds linearly. 그러나 best F1을 직접 maximize하는
strategies가 있다:

- Per-percentile threshold sweep (high resolution)
- Logarithmic threshold sweep (focus on tail)
- Greedy F1 maximization (gradient-free)
- Per-dataset threshold scaling
"""
import numpy as np


def find_events(arr):
    """Binary array → list of (start, end)."""
    events = []
    in_e, st = False, None
    for i, v in enumerate(arr):
        if v == 1 and not in_e:
            st, in_e = i, True
        elif v == 0 and in_e:
            events.append((st, i))
            in_e = False
    if in_e:
        events.append((st, len(arr)))
    return events


def pak_f1_at_threshold(scores, labels, regions, threshold, k=0.5):
    """PA-K F1 at given threshold."""
    pred = scores > threshold

    # PA-K: if any K% of segment detected, label whole segment as detected
    tp = 0
    fn = 0
    detected_anomaly_pts = 0
    for r in regions:
        seg_len = r.end - r.start
        seg_pred = pred[r.start:r.end]
        if seg_pred.sum() >= k * seg_len:
            tp += seg_len
            detected_anomaly_pts += seg_len
        else:
            fn += seg_len

    # FP: predictions outside any region
    all_anom = np.zeros_like(pred, dtype=bool)
    for r in regions:
        all_anom[r.start:r.end] = True
    fp = int(((~all_anom) & pred).sum())

    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    p = tp / (tp + fp)
    r_ = tp / (tp + fn)
    return 2 * p * r_ / (p + r_) if (p + r_) > 0 else 0.0


def best_threshold_search(
    scores, labels, regions,
    n_log_thresholds=100, n_linear_thresholds=200, k=0.5,
):
    """Best F1 threshold search with combined log + linear grid."""
    s_min, s_max = scores.min(), scores.max()
    if s_max <= s_min:
        return s_min, 0.0

    # Linear grid
    linear_thrs = np.linspace(s_min, s_max, n_linear_thresholds)
    # Log grid on shifted positive scores
    shifted = scores - s_min + 1e-10
    log_thrs = np.exp(np.linspace(np.log(shifted.min() + 1e-10),
                                   np.log(shifted.max()), n_log_thresholds)) + s_min - 1e-10
    # Percentile-based grid (focus on tail)
    percentile_thrs = np.percentile(scores, np.linspace(50, 99.9, 50))

    all_thresholds = np.unique(np.concatenate([linear_thrs, log_thrs, percentile_thrs]))
    best_thr = all_thresholds[0]
    best_f1 = 0.0
    for thr in all_thresholds:
        f1 = pak_f1_at_threshold(scores, labels, regions, thr, k)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr), float(best_f1)


def adaptive_threshold(scores, labels, percentile_target=None):
    """Set threshold at percentile matching anomaly_ratio.
    e.g., if 5% data is anomaly, threshold = 95th percentile of scores."""
    if percentile_target is None:
        anomaly_ratio = labels.mean()
        percentile_target = 100 * (1 - anomaly_ratio)
    return float(np.percentile(scores, percentile_target))


def pak_auc_with_optimal_threshold(scores, labels, regions, eval_mask=None):
    """Best F1 across all thresholds (instead of AUC).
    Returns: best_f1 (single point optimization, not AUC integration)."""
    if eval_mask is not None:
        scores = scores[eval_mask]
        labels = labels[eval_mask]

    _, best_f1 = best_threshold_search(scores, labels, regions)
    return best_f1
