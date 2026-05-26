"""
Unsupervised Median Segment Length Estimation.

본 module은 score sequence (labels 없이) 에서 anomaly의 segment length를
추정하는 6가지 method를 제공:

1. PeakRunEstimator: percentile threshold + connected run lengths
2. PeakWidthEstimator: scipy.signal.find_peaks + per-peak width estimation
3. AutocorrelationEstimator: ACF / FWHM에서 characteristic timescale 추출
4. WaveletEstimator: continuous wavelet transform power 분포로 dominant scale
5. KDEEstimator: KDE on score distribution + FWHM
6. ChangePointEstimator: 1st-derivative based segmentation

Each estimator returns:
- estimated_median_seg: float (timesteps)
- confidence: float (0-1, internal quality metric)

EnsembleEstimator: weighted combination of base estimators.
"""
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d


class BaseEstimator:
    """Estimator interface."""

    def estimate(self, score: np.ndarray) -> tuple:
        """Returns (median_seg_estimate, confidence)."""
        raise NotImplementedError


class PeakRunEstimator(BaseEstimator):
    """Method 1: percentile threshold + connected run lengths.

    Algorithm:
    1. Threshold = `percentile`-th percentile of score
    2. Binary mask (above threshold)
    3. Find connected runs (anomaly candidate segments)
    4. Return median of run lengths

    Confidence based on number of detected runs (more = higher confidence).
    """

    def __init__(self, percentile=90, min_run_length=2, max_runs_for_confidence=20):
        self.percentile = percentile
        self.min_run_length = min_run_length
        self.max_runs_for_confidence = max_runs_for_confidence

    def estimate(self, score):
        threshold = np.percentile(score, self.percentile)
        binary = score > threshold
        if binary.sum() == 0:
            return 10.0, 0.0

        diffs = np.diff(binary.astype(int))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1
        if binary[0]:
            starts = np.concatenate([[0], starts])
        if binary[-1]:
            ends = np.concatenate([ends, [len(binary)]])

        if len(starts) == 0:
            return 10.0, 0.0

        run_lengths = ends - starts
        valid_runs = run_lengths[run_lengths >= self.min_run_length]
        if len(valid_runs) == 0:
            return 10.0, 0.0

        median_run = float(np.median(valid_runs))
        confidence = min(1.0, len(valid_runs) / self.max_runs_for_confidence)
        return median_run, confidence


class PeakWidthEstimator(BaseEstimator):
    """Method 2: scipy.signal.find_peaks + peak widths.

    Algorithm:
    1. Smooth score lightly (σ=3) to denoise
    2. Find peaks above N percentile prominence
    3. Compute peak widths at half-max
    4. Median of peak widths
    """

    def __init__(self, smooth_sigma=3, prominence_percentile=90, rel_height=0.5):
        self.smooth_sigma = smooth_sigma
        self.prominence_percentile = prominence_percentile
        self.rel_height = rel_height

    def estimate(self, score):
        smoothed = gaussian_filter1d(score, sigma=self.smooth_sigma, mode='reflect')
        prominence_threshold = np.percentile(smoothed - smoothed.min(), self.prominence_percentile)
        peaks, properties = find_peaks(smoothed, prominence=prominence_threshold)
        if len(peaks) == 0:
            return 10.0, 0.0

        widths, _, _, _ = peak_widths(smoothed, peaks, rel_height=self.rel_height)
        if len(widths) == 0:
            return 10.0, 0.0

        median_width = float(np.median(widths))
        confidence = min(1.0, len(peaks) / 20.0)
        return median_width, confidence


class AutocorrelationEstimator(BaseEstimator):
    """Method 3: autocorrelation function characteristic timescale.

    Algorithm:
    1. Compute ACF on score sequence
    2. Find first zero-crossing or first decay to 1/e
    3. This timescale is approximately the characteristic segment width
    """

    def __init__(self, max_lag=500, target_decay=0.5):
        self.max_lag = max_lag
        self.target_decay = target_decay

    def estimate(self, score):
        s = score - score.mean()
        if s.std() < 1e-9:
            return 10.0, 0.0
        s = s / s.std()
        n = len(s)
        max_lag = min(self.max_lag, n // 3)

        # ACF via FFT (faster)
        f = np.fft.fft(s, n=2 * n)
        acf = np.real(np.fft.ifft(f * np.conj(f)))[:max_lag]
        acf /= acf[0]

        # Find first lag where ACF drops below target
        below = np.where(acf < self.target_decay)[0]
        if len(below) == 0:
            return float(max_lag), 0.5

        characteristic_lag = float(below[0])
        # Conf based on how monotonic decay is (less oscillation = higher conf)
        confidence = float(np.mean(np.diff(acf[:characteristic_lag + 1]) < 0))
        return characteristic_lag, confidence


class WaveletEstimator(BaseEstimator):
    """Method 4: continuous wavelet transform — dominant scale.

    Algorithm:
    1. CWT (Ricker wavelet) at multiple scales
    2. Sum power at each scale
    3. Find peak scale (excluding very small / very large)
    4. Convert wavelet scale to timestep equivalent
    """

    def __init__(self, scales=None):
        if scales is None:
            scales = np.array([2, 5, 10, 20, 50, 100, 200, 400, 800])
        self.scales = scales

    def estimate(self, score):
        try:
            from scipy.signal import cwt, ricker
        except ImportError:
            return 10.0, 0.0

        s = score - score.mean()
        if s.std() < 1e-9:
            return 10.0, 0.0
        s = s / s.std()

        try:
            cwtmatr = cwt(s, ricker, self.scales)
        except Exception:
            return 10.0, 0.0

        power_per_scale = np.mean(cwtmatr ** 2, axis=1)
        # Exclude edge scales
        valid_idx = slice(1, len(self.scales) - 1)
        if power_per_scale[valid_idx].size == 0:
            return 10.0, 0.0
        peak_idx = valid_idx.start + np.argmax(power_per_scale[valid_idx])
        dominant_scale = float(self.scales[peak_idx])

        # Confidence: peak prominence
        max_power = power_per_scale[peak_idx]
        median_power = np.median(power_per_scale)
        confidence = float(min(1.0, (max_power - median_power) / (median_power + 1e-9)))
        confidence = max(0.0, min(1.0, confidence))
        return dominant_scale, confidence


class KDEEstimator(BaseEstimator):
    """Method 5: KDE on score distribution — mode width."""

    def __init__(self, n_samples=5000):
        self.n_samples = n_samples

    def estimate(self, score):
        if len(score) > self.n_samples:
            idx = np.random.RandomState(0).choice(len(score), self.n_samples, replace=False)
            sample = score[idx]
        else:
            sample = score

        try:
            kde = gaussian_kde(sample, bw_method='silverman')
        except Exception:
            return 10.0, 0.0

        xs = np.linspace(sample.min(), sample.max(), 500)
        density = kde(xs)
        peak_idx = np.argmax(density)
        peak_val = density[peak_idx]
        half_max = peak_val / 2.0

        left_mask = density[:peak_idx] < half_max
        left_idx = np.where(left_mask)[0][-1] if left_mask.sum() > 0 else 0
        right_mask = density[peak_idx:] < half_max
        right_idx = (peak_idx + np.where(right_mask)[0][0]) if right_mask.sum() > 0 else len(density) - 1

        fwhm_score = xs[right_idx] - xs[left_idx]
        score_range = score.max() - score.min() + 1e-9
        fwhm_ratio = fwhm_score / score_range

        # Map FWHM ratio to timestep estimate (heuristic)
        estimated = 10.0 * (1 + fwhm_ratio * 5)
        estimated = max(min(estimated, 500.0), 1.0)

        # Confidence: peak sharpness (lower fwhm_ratio = higher conf in mode width)
        confidence = float(max(0.0, 1.0 - fwhm_ratio))
        return float(estimated), confidence


class ChangePointEstimator(BaseEstimator):
    """Method 6: 1st-derivative based segmentation.

    Algorithm:
    1. Smooth score then compute |1st derivative|
    2. Find change points (large derivatives)
    3. Distance between consecutive change points ~ segment scale
    """

    def __init__(self, smooth_sigma=10, derivative_percentile=95):
        self.smooth_sigma = smooth_sigma
        self.derivative_percentile = derivative_percentile

    def estimate(self, score):
        smoothed = gaussian_filter1d(score, sigma=self.smooth_sigma, mode='reflect')
        deriv = np.abs(np.diff(smoothed))
        if deriv.sum() < 1e-9:
            return 10.0, 0.0

        threshold = np.percentile(deriv, self.derivative_percentile)
        change_points = np.where(deriv > threshold)[0]

        if len(change_points) < 2:
            return 10.0, 0.0

        # Distances between consecutive change points
        gaps = np.diff(change_points)
        if len(gaps) == 0:
            return 10.0, 0.0

        # Filter very small gaps (noise)
        valid_gaps = gaps[gaps > 2]
        if len(valid_gaps) == 0:
            return 10.0, 0.0

        median_gap = float(np.median(valid_gaps))
        confidence = min(1.0, len(valid_gaps) / 10.0)
        return median_gap, confidence


class EnsembleEstimator(BaseEstimator):
    """Ensemble of base estimators with confidence-weighted combination."""

    def __init__(self, estimators=None, mode='weighted_geom_mean'):
        if estimators is None:
            estimators = [
                ('peak_run', PeakRunEstimator(percentile=90)),
                ('peak_width', PeakWidthEstimator()),
                ('autocorr', AutocorrelationEstimator()),
                ('kde', KDEEstimator()),
                ('change_point', ChangePointEstimator()),
            ]
        self.estimators = estimators
        self.mode = mode

    def estimate(self, score):
        results = {}
        for name, est in self.estimators:
            try:
                m, c = est.estimate(score)
                results[name] = (m, c)
            except Exception as e:
                results[name] = (10.0, 0.0)

        # Filter out zero-confidence estimates
        valid = [(m, c) for m, c in results.values() if c > 0.05]
        if not valid:
            return 10.0, 0.0, results

        estimates = np.array([m for m, _ in valid])
        confidences = np.array([c for _, c in valid])

        if self.mode == 'weighted_mean':
            combined = float((estimates * confidences).sum() / confidences.sum())
        elif self.mode == 'weighted_geom_mean':
            log_estimates = np.log(np.maximum(estimates, 0.5))
            combined = float(np.exp((log_estimates * confidences).sum() / confidences.sum()))
        elif self.mode == 'median':
            combined = float(np.median(estimates))
        elif self.mode == 'max_confidence':
            best_idx = np.argmax(confidences)
            combined = float(estimates[best_idx])
        else:
            combined = float(np.mean(estimates))

        total_confidence = float(min(1.0, confidences.mean()))
        return combined, total_confidence, results
