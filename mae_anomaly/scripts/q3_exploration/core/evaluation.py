"""
Evaluation utilities — pak_auc_f1 (re-implementation, self-contained).

Note: 정확히 mae_anomaly.evaluator.compute_pa_k_auc와 동일 result를 보장하지 않음.
본 모듈은 fast self-contained replacement (median threshold + percentile sweep).
정확한 reproduction이 필요한 경우 외부 evaluator import 권장.
"""
import sys
import numpy as np

# Use external evaluator for exact reproduction
sys.path.insert(0, '/home/ykio/notebooks/claude')
try:
    from mae_anomaly.evaluator import compute_pa_k_auc as _ext_compute_pa_k_auc
    EXTERNAL_EVAL_AVAILABLE = True
except ImportError:
    EXTERNAL_EVAL_AVAILABLE = False


def pak_auc_f1(scores, labels, regions, eval_mask=None, n_thresholds=200):
    """PA-K AUC F1 (matching mae_anomaly.evaluator)."""
    if not EXTERNAL_EVAL_AVAILABLE:
        return _fallback_pak(scores, labels, regions, eval_mask, n_thresholds)

    res = _ext_compute_pa_k_auc(
        point_scores=scores,
        point_labels=labels,
        anomaly_regions=regions,
        threshold=float(np.median(scores)),
        eval_mask=eval_mask,
        n_thresholds=n_thresholds,
    )
    return float(res.get('pak_auc_f1', 0))


def _fallback_pak(scores, labels, regions, eval_mask=None, n_thresholds=200):
    """Simplified PA-K AUC if external not available."""
    if eval_mask is not None:
        scores = scores[eval_mask]
        labels = labels[eval_mask]

    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    f1s = []
    for thr in thresholds:
        pred = scores > thr
        # PA-K: if any 50%+ of segment detected, label whole segment as detected
        tp = fp = fn = 0
        for r in regions:
            seg_pred = pred[r.start:r.end]
            if seg_pred.sum() >= 0.5 * (r.end - r.start):
                tp += (r.end - r.start)
            else:
                fn += (r.end - r.start)
        # FP: predictions outside any region
        all_anom = np.zeros_like(pred, dtype=bool)
        for r in regions:
            all_anom[r.start:r.end] = True
        fp = ((~all_anom) & pred).sum()
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        p = tp / (tp + fp)
        r_ = tp / (tp + fn)
        f1s.append(2 * p * r_ / (p + r_) if (p + r_) > 0 else 0.0)
    return float(np.mean(f1s))


def per_group_summary(deltas_dict, get_group_fn):
    """Compute per-group statistics from per-dataset deltas.

    Args:
        deltas_dict: {alias: float Δ}
        get_group_fn: alias → group_name

    Returns:
        {group: dict(n, mean_delta, wins, losses, catastrophic)}
    """
    groups = {}
    for alias, delta in deltas_dict.items():
        g = get_group_fn(alias)
        groups.setdefault(g, []).append(delta)
    summary = {}
    for g, deltas in groups.items():
        deltas = np.array(deltas)
        summary[g] = {
            'n': len(deltas),
            'mean_delta': float(deltas.mean()),
            'median_delta': float(np.median(deltas)),
            'wins': int((deltas > 0).sum()),
            'losses': int((deltas < 0).sum()),
            'ties': int((deltas == 0).sum()),
            'catastrophic': int((deltas < -0.05).sum()),
        }
    return summary


def wilcoxon_test(method_scores, baseline_scores, alternative='greater'):
    """Wilcoxon signed-rank test."""
    from scipy.stats import wilcoxon
    try:
        stat, p = wilcoxon(method_scores, baseline_scores, alternative=alternative)
        return float(p)
    except Exception:
        return 1.0
