#!/usr/bin/env python
"""CPU-only tiny test for new minmax_range / minmax_clamp_min/max options.

Tests _minmax_per_feature with synthetic data:
  - 100 timesteps × 3 features
  - Train (0-70): uniform(0, 10) — normal range
  - Test (70-100): uniform(-5, 20) — outliers outside train range

Expected behavior:
  Mode A — '0_1' (default, legacy):
    feature_range=(0,1), clip=True, no clamp
    → both train+test tight-clipped to [0, 1]
    → ALL test outlier info lost (becomes 0 or 1)
  Mode B — 'neg1_1' (NPSR-style):
    feature_range=(-1,1), clip=False, clamp=(-4, 4)
    → train scaled to [-1, 1] exactly
    → test outliers preserved within [-4, 4]
    → train_max + (test_max - train_max) / range * 2 may exceed 1 but <= 4
"""
import sys
sys.path.insert(0, '/home/ykio/notebooks/claude')

import numpy as np
from mae_anomaly.dataset_sliding import _minmax_per_feature


def make_data(seed=42):
    rng = np.random.default_rng(seed)
    signals = np.zeros((100, 3), dtype=np.float64)
    signals[:70] = rng.uniform(0.0, 10.0, size=(70, 3))   # train: [0, 10]
    signals[70:] = rng.uniform(-5.0, 20.0, size=(30, 3))  # test: outliers
    return signals


def test_mode_a_default():
    """'0_1' mode — backward compat with prior behavior."""
    raw = make_data()
    train_min_raw = raw[:70].min(axis=0)
    train_max_raw = raw[:70].max(axis=0)

    out, smin, srange = _minmax_per_feature(
        raw, train_end=70,
        clip=True, feature_range=(0.0, 1.0),
        clamp_min=None, clamp_max=None,
    )
    # train portion: should be exactly [0, 1] (after clip)
    train_out = out[:70]
    test_out = out[70:]
    assert train_out.min() >= -1e-6, f"train_min < 0: {train_out.min()}"
    assert train_out.max() <= 1.0 + 1e-6, f"train_max > 1: {train_out.max()}"
    assert test_out.min() >= -1e-6, f"test_min < 0: {test_out.min()}"
    assert test_out.max() <= 1.0 + 1e-6, f"test_max > 1: {test_out.max()}"

    # Outlier info should be CLIPPED (some test values == 0.0 or == 1.0 due to clip)
    n_clipped = ((test_out == 0.0) | (test_out == 1.0)).sum()
    assert n_clipped > 0, "Expected some clipping but found none"
    # scaler stats match raw train
    assert np.allclose(smin, train_min_raw, atol=1e-5)
    assert np.allclose(srange, train_max_raw - train_min_raw, atol=1e-5)
    assert out.dtype == np.float32
    print(f"  ✓ Mode A '0_1': train [{train_out.min():.4f}, {train_out.max():.4f}], "
          f"test [{test_out.min():.4f}, {test_out.max():.4f}], "
          f"clipped cells={n_clipped}/{test_out.size}")


def test_mode_b_npsr_style():
    """'neg1_1' mode — NPSR-style: train [-1,1], test only clamped to [-4, 4]."""
    raw = make_data()
    out, smin, srange = _minmax_per_feature(
        raw, train_end=70,
        clip=False, feature_range=(-1.0, 1.0),
        clamp_min=-4.0, clamp_max=4.0,
    )
    train_out = out[:70]
    test_out = out[70:]

    # Train should be exactly [-1, 1] (mapped from [0, 1] via *2-1)
    assert abs(train_out.min() - (-1.0)) < 1e-5, f"train_min != -1.0: {train_out.min()}"
    assert abs(train_out.max() - 1.0) < 1e-5, f"train_max != 1.0: {train_out.max()}"

    # Test should be inside [-4, 4] but can exceed [-1, 1] (preserved outliers)
    assert test_out.min() >= -4.0 - 1e-5, f"test_min < -4: {test_out.min()}"
    assert test_out.max() <= 4.0 + 1e-5, f"test_max > 4: {test_out.max()}"
    # Some test values should be outside [-1, 1] (outliers preserved within clamp)
    n_outside_train_range = ((test_out < -1.0 - 1e-5) | (test_out > 1.0 + 1e-5)).sum()
    assert n_outside_train_range > 0, "Expected outliers outside [-1, 1] but found none"
    assert out.dtype == np.float32
    print(f"  ✓ Mode B 'neg1_1' + clamp ±4: train [{train_out.min():.4f}, {train_out.max():.4f}], "
          f"test [{test_out.min():.4f}, {test_out.max():.4f}], "
          f"outside [-1,1]: {n_outside_train_range}/{test_out.size}")


def test_constant_feature_protection():
    """Constant feature (range=0) should not produce NaN."""
    raw = np.ones((100, 3), dtype=np.float64)
    raw[:, 1] *= 5.0  # all 5.0
    raw[70:, 0] = 100.0  # only feature 0 has test outliers
    out, smin, srange = _minmax_per_feature(
        raw, train_end=70, clip=True, feature_range=(0.0, 1.0),
    )
    assert not np.isnan(out).any(), "NaN in output for constant feature"
    print(f"  ✓ Constant feature: srange={srange}, no NaN")


def test_neg11_without_clamp():
    """neg1_1 mode without clamp — test should freely exceed [-1, 1]."""
    raw = make_data()
    out, _, _ = _minmax_per_feature(
        raw, train_end=70,
        clip=False, feature_range=(-1.0, 1.0),
        clamp_min=None, clamp_max=None,
    )
    test_out = out[70:]
    # test extends beyond [-1, 1] without restriction
    assert test_out.min() < -1.0, f"test_min should < -1: {test_out.min()}"
    assert test_out.max() > 1.0, f"test_max should > 1: {test_out.max()}"
    print(f"  ✓ neg1_1 no clamp: test [{test_out.min():.4f}, {test_out.max():.4f}] (unrestricted)")


def test_shape_dtype_preserved():
    """Shape and dtype consistency."""
    raw = make_data().astype(np.float64)
    for mode_args in [
        dict(clip=True, feature_range=(0.0, 1.0)),
        dict(clip=False, feature_range=(-1.0, 1.0), clamp_min=-4.0, clamp_max=4.0),
    ]:
        out, smin, srange = _minmax_per_feature(raw, train_end=70, **mode_args)
        assert out.shape == raw.shape
        assert out.dtype == np.float32
        assert smin.dtype == np.float32
        assert srange.dtype == np.float32
    print(f"  ✓ Shape/dtype preserved across modes")


def test_sliding_window_dataset_integration():
    """End-to-end via SlidingWindowDataset constructor (covers config path)."""
    from mae_anomaly.dataset_sliding import SlidingWindowDataset
    rng = np.random.default_rng(0)
    n = 500
    signals = np.zeros((n, 3), dtype=np.float64)
    signals[:int(0.8*n)] = rng.uniform(0, 10, size=(int(0.8*n), 3))
    signals[int(0.8*n):] = rng.uniform(-5, 20, size=(n - int(0.8*n), 3))
    point_labels = np.zeros(n, dtype=np.float32)

    # Mode B via constructor
    ds_b = SlidingWindowDataset(
        signals=signals.copy(),
        point_labels=point_labels,
        anomaly_regions=[],
        window_size=100,
        stride=10,
        mask_last_n=10,
        split='test',
        train_ratio=0.8,
        normalize_mode='minmax',
        minmax_range='neg1_1',
        minmax_clamp_min=-4.0,
        minmax_clamp_max=4.0,
    )
    sig_b = ds_b.signals  # test portion
    assert sig_b.min() >= -4.0 - 1e-5
    assert sig_b.max() <= 4.0 + 1e-5
    n_outliers = ((sig_b < -1.0) | (sig_b > 1.0)).sum()
    assert n_outliers > 0

    # Mode A via constructor (backward compat)
    ds_a = SlidingWindowDataset(
        signals=signals.copy(),
        point_labels=point_labels,
        anomaly_regions=[],
        window_size=100,
        stride=10,
        mask_last_n=10,
        split='test',
        train_ratio=0.8,
        normalize_mode='minmax',  # minmax_range default '0_1'
    )
    sig_a = ds_a.signals
    assert sig_a.min() >= -1e-5
    assert sig_a.max() <= 1.0 + 1e-5
    print(f"  ✓ SlidingWindowDataset integration: "
          f"mode_a [{sig_a.min():.4f}, {sig_a.max():.4f}], "
          f"mode_b [{sig_b.min():.4f}, {sig_b.max():.4f}] (outliers={n_outliers})")


if __name__ == '__main__':
    print("=== CPU tiny test for minmax_range + clamp ===")
    test_mode_a_default()
    test_mode_b_npsr_style()
    test_constant_feature_protection()
    test_neg11_without_clamp()
    test_shape_dtype_preserved()
    test_sliding_window_dataset_integration()
    print("\n*** ALL TESTS PASSED ***")
