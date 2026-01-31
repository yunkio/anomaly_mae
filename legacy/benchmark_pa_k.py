#!/usr/bin/env python3
"""
Benchmark PA%K ROC-AUC computation: vectorized vs non-vectorized
"""

import numpy as np
import time


def compute_pa_k_roc_auc_original(
    scores: np.ndarray,
    labels: np.ndarray,
    k_percent: int = 20,
    n_thresholds: int = 100
) -> float:
    """Original (non-vectorized) PA%K ROC-AUC computation."""
    if len(np.unique(labels)) < 2:
        return 0.0

    n = len(labels)
    n_positive = int(labels.sum())
    n_negative = n - n_positive

    if n_positive == 0 or n_negative == 0:
        return 0.0

    min_score, max_score = scores.min(), scores.max()
    thresholds = np.linspace(min_score - 0.01, max_score + 0.01, n_thresholds)

    tprs = []
    fprs = []

    for threshold in thresholds:
        predictions = (scores > threshold).astype(int)
        adjusted_preds = predictions.copy()

        # Apply PA%K adjustment
        i = 0
        while i < n:
            if labels[i] == 1:
                start = i
                while i < n and labels[i] == 1:
                    i += 1
                end = i

                segment_preds = predictions[start:end]
                detection_ratio = segment_preds.mean()

                if detection_ratio >= k_percent / 100:
                    adjusted_preds[start:end] = 1
                else:
                    adjusted_preds[start:end] = 0
            else:
                i += 1

        tp = (adjusted_preds & labels).sum()
        fp = (adjusted_preds & ~labels.astype(bool)).sum()

        tprs.append(tp / n_positive)
        fprs.append(fp / n_negative)

    sorted_indices = np.argsort(fprs)
    fprs = np.array(fprs)[sorted_indices]
    tprs = np.array(tprs)[sorted_indices]

    auc = np.trapz(tprs, fprs)
    return float(auc)


def _find_anomaly_segments(labels: np.ndarray) -> list:
    """Find contiguous anomaly segments in labels."""
    segments = []
    i = 0
    n = len(labels)
    while i < n:
        if labels[i] == 1:
            start = i
            while i < n and labels[i] == 1:
                i += 1
            segments.append((start, i))
        else:
            i += 1
    return segments


def compute_pa_k_roc_auc_vectorized(
    scores: np.ndarray,
    labels: np.ndarray,
    k_percent: int = 20,
    n_thresholds: int = 50,
    segments: list = None
) -> float:
    """Vectorized PA%K ROC-AUC computation."""
    if len(np.unique(labels)) < 2:
        return 0.0

    n = len(labels)
    n_positive = int(labels.sum())
    n_negative = n - n_positive

    if n_positive == 0 or n_negative == 0:
        return 0.0

    if segments is None:
        segments = _find_anomaly_segments(labels)

    if len(segments) == 0:
        return 0.0

    min_score, max_score = scores.min(), scores.max()
    thresholds = np.linspace(min_score - 0.01, max_score + 0.01, n_thresholds)

    # Vectorized: (n_thresholds, n_samples)
    all_preds = (scores[np.newaxis, :] > thresholds[:, np.newaxis]).astype(np.int8)

    k_ratio = k_percent / 100.0
    adjusted_preds = all_preds.copy()

    for start, end in segments:
        seg_len = end - start
        seg_sums = all_preds[:, start:end].sum(axis=1)
        detection_ratios = seg_sums / seg_len
        detected = detection_ratios >= k_ratio
        adjusted_preds[:, start:end] = detected[:, np.newaxis]

    labels_bool = labels.astype(bool)
    tp = (adjusted_preds & labels_bool).sum(axis=1)
    fp = (adjusted_preds & ~labels_bool).sum(axis=1)

    tprs = tp / n_positive
    fprs = fp / n_negative

    sorted_idx = np.argsort(fprs)
    auc = np.trapz(tprs[sorted_idx], fprs[sorted_idx])

    return float(auc)


def main():
    # Generate synthetic data (similar to test dataset size)
    np.random.seed(42)

    # Test with different data sizes
    test_cases = [
        ("Small (1K samples)", 1000, 100),
        ("Medium (10K samples)", 10000, 500),
        ("Large (80K samples)", 80000, 2000),
    ]

    print("=" * 70)
    print("PA%K ROC-AUC BENCHMARK: Original vs Vectorized")
    print("=" * 70)

    for name, n_samples, n_anomaly in test_cases:
        # Generate random scores and labels
        scores = np.random.randn(n_samples) * 0.1 + 0.5
        labels = np.zeros(n_samples, dtype=int)

        # Add anomaly segments
        n_segments = n_anomaly // 50
        for _ in range(n_segments):
            start = np.random.randint(0, n_samples - 50)
            end = min(start + 50, n_samples)
            labels[start:end] = 1
            scores[start:end] += 0.2  # Make anomalies have higher scores

        print(f"\n{name}")
        print(f"  Samples: {n_samples}, Anomalies: {labels.sum()}, Segments: {len(_find_anomaly_segments(labels))}")

        # Benchmark original (100 thresholds)
        t0 = time.time()
        auc_original = compute_pa_k_roc_auc_original(scores, labels, k_percent=20, n_thresholds=100)
        time_original = time.time() - t0

        # Benchmark vectorized (50 thresholds)
        t0 = time.time()
        auc_vectorized = compute_pa_k_roc_auc_vectorized(scores, labels, k_percent=20, n_thresholds=50)
        time_vectorized = time.time() - t0

        print(f"  Original (100 thresholds): {time_original:.4f}s, AUC={auc_original:.4f}")
        print(f"  Vectorized (50 thresholds): {time_vectorized:.4f}s, AUC={auc_vectorized:.4f}")
        print(f"  Speedup: {time_original / time_vectorized:.1f}x")
        print(f"  AUC difference: {abs(auc_original - auc_vectorized):.6f}")


if __name__ == '__main__':
    main()
