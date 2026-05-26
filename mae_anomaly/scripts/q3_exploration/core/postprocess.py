"""
Score post-processing utilities.

Provides various smoothing / transformation methods beyond gauss:
- median_filter1d
- bilateral_filter (edge-preserving)
- savitzky_golay
- asymmetric_smoothing (rising vs falling edge)
- z5_pyramid (5-scale geometric mean)
- nlm_sigmoid_transform
- multi_stride_aggregate
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter


def median_filter1d(score, window_size=15):
    """Median smoothing (robust to outliers)."""
    return median_filter(score, size=max(int(window_size), 3))


def bilateral_filter_1d(score, sigma_spatial=10, sigma_intensity=None):
    """1D bilateral filter (edge-preserving smoothing)."""
    if sigma_intensity is None:
        sigma_intensity = score.std() * 0.5

    n = len(score)
    # Build spatial weight kernel
    half_kernel = int(sigma_spatial * 3)
    if half_kernel < 1:
        return score.copy()

    output = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half_kernel)
        hi = min(n, i + half_kernel + 1)
        window = score[lo:hi]
        spatial_dist = np.arange(lo, hi) - i
        spatial_w = np.exp(-(spatial_dist ** 2) / (2 * sigma_spatial ** 2))
        intensity_dist = window - score[i]
        intensity_w = np.exp(-(intensity_dist ** 2) / (2 * sigma_intensity ** 2 + 1e-9))
        weights = spatial_w * intensity_w
        if weights.sum() > 1e-9:
            output[i] = (window * weights).sum() / weights.sum()
        else:
            output[i] = score[i]
    return output


def savitzky_golay_smooth(score, window_length=21, polyorder=3):
    """Savitzky-Golay filter (polynomial smoothing)."""
    window_length = max(int(window_length), polyorder + 1)
    if window_length % 2 == 0:
        window_length += 1
    if window_length >= len(score):
        return score.copy()
    return savgol_filter(score, window_length, polyorder)


def asymmetric_smooth(score, sigma_rising=10, sigma_falling=30):
    """Asymmetric Gaussian: different σ for rising vs falling edges.
    Anomaly가 점진적으로 발생하지만 sharp하게 종료될 수 있음 → 비대칭 smoothing."""
    diff = np.diff(score, prepend=score[0])
    smoothed = np.zeros_like(score, dtype=np.float64)

    # Use different sigma based on whether score is rising or falling
    rising_smoothed = gaussian_filter1d(score, sigma=max(sigma_rising, 0.5), mode='reflect')
    falling_smoothed = gaussian_filter1d(score, sigma=max(sigma_falling, 0.5), mode='reflect')

    # Blend: rising periods use rising_smoothed, else falling_smoothed
    rising_mask = (diff > 0).astype(np.float64)
    rising_mask = gaussian_filter1d(rising_mask, sigma=5, mode='reflect')  # smooth transition
    smoothed = rising_mask * rising_smoothed + (1 - rising_mask) * falling_smoothed
    return smoothed


def z5_pyramid(score, scales=(5, 25, 100, 400, 1600)):
    """5-scale geometric mean (1분기 Z5)."""
    s = np.maximum(score, 1e-10)
    log_s = np.log(s)
    smoothed = [gaussian_filter1d(log_s, sigma=max(W / 3.0, 0.5), mode='reflect')
                for W in scales]
    return np.exp(np.mean(np.stack(smoothed, axis=0), axis=0))


def nlm_sigmoid_transform(score, T_factor=2.0):
    """NLM-Surprisal: sigmoid((score - mean) / (T * std))."""
    centered = score - score.mean()
    T = T_factor * (score.std() + 1e-9)
    return 1.0 / (1.0 + np.exp(-np.clip(centered / T, -30, 30)))


def power_transform(score, power=0.5):
    """Power transform: x^power (positive shift)."""
    shifted = score - score.min() + 1e-10
    return np.power(shifted, power)


def rank_normalize(score):
    """Rank-based normalization (uniform distribution)."""
    n = len(score)
    ranks = np.argsort(np.argsort(score))
    return ranks / (n - 1) if n > 1 else score


def trimmed_mean_smooth(score, window_size=21, trim_ratio=0.1):
    """Trimmed mean smoothing (drop extremes within window)."""
    n = len(score)
    half = window_size // 2
    output = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = score[lo:hi]
        if len(window) > 0:
            k = int(len(window) * trim_ratio / 2)
            if k * 2 < len(window):
                sorted_w = np.sort(window)
                trimmed = sorted_w[k:len(window) - k] if k > 0 else sorted_w
                output[i] = trimmed.mean()
            else:
                output[i] = window.mean()
        else:
            output[i] = score[i]
    return output


def double_gaussian(score, sigma_short=5, sigma_long=50, weight_short=0.7):
    """Mixture of two gaussians at different scales."""
    short = gaussian_filter1d(score, sigma=sigma_short, mode='reflect')
    long = gaussian_filter1d(score, sigma=sigma_long, mode='reflect')
    return weight_short * short + (1 - weight_short) * long


def per_channel_zrank_combine(pt_r, pt_d, pt_f, weights=(0.5, 0.3, 0.2)):
    """Rank-normalize per channel + weighted z-sum."""
    rr = rank_normalize(pt_r)
    rd = rank_normalize(pt_d)
    rf = rank_normalize(pt_f)
    return weights[0] * rr + weights[1] * rd + weights[2] * rf
