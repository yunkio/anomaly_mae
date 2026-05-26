"""
Time series modeling utilities for anomaly score processing.

Provides:
- AR model fitting + residual scoring (statsmodels AR via numpy)
- HMM-based state segmentation (lightweight implementation)
- Spectral subtraction (background spectrum removal)
"""
import numpy as np
from scipy.linalg import toeplitz


# ================== AR Model ==================

class ARScorer:
    """AR(p) model fit + residual as anomaly indicator.

    Algorithm:
    1. Fit AR(p) on score sequence (Yule-Walker estimation)
    2. Predict each point from past p
    3. Residual = actual - predicted
    4. Anomaly score = |residual| (or squared)
    """

    def __init__(self, p=5):
        self.p = p
        self.ar_coefs = None
        self.intercept = None
        self.residual_std = None

    def fit(self, x):
        """Yule-Walker AR estimation."""
        if len(x) <= self.p:
            self.ar_coefs = np.zeros(self.p)
            self.intercept = float(x.mean()) if len(x) > 0 else 0.0
            self.residual_std = 1.0
            return self

        x_centered = x - x.mean()

        # Compute autocorrelations
        n = len(x_centered)
        autocorr = np.zeros(self.p + 1)
        for k in range(self.p + 1):
            autocorr[k] = np.sum(x_centered[:n - k] * x_centered[k:]) / n
        autocorr /= autocorr[0]  # normalize

        # Yule-Walker: R * a = r
        R_mat = toeplitz(autocorr[:self.p])
        r_vec = autocorr[1:self.p + 1]
        try:
            a = np.linalg.solve(R_mat, r_vec)
        except Exception:
            a = np.zeros(self.p)

        self.ar_coefs = a
        self.intercept = float(x.mean() * (1 - a.sum()))

        # Compute residuals to estimate std
        preds = self._predict_in_sample(x)
        residuals = x[self.p:] - preds[self.p:]
        self.residual_std = float(residuals.std() + 1e-9)
        return self

    def _predict_in_sample(self, x):
        """Predict each point from past p (zero for initial)."""
        n = len(x)
        preds = np.zeros(n)
        preds[:self.p] = x[:self.p]  # initial: copy
        for t in range(self.p, n):
            preds[t] = self.intercept + np.sum(self.ar_coefs * x[t - self.p:t][::-1])
        return preds

    def residuals(self, x):
        """Return |residuals|."""
        preds = self._predict_in_sample(x)
        return np.abs(x - preds)

    def standardized_residuals(self, x):
        """|residuals| / σ."""
        return self.residuals(x) / (self.residual_std + 1e-9)


def ar_residual_score(score_sequence, p=5):
    """Functional interface: AR(p) residual anomaly score."""
    ar = ARScorer(p=p).fit(score_sequence)
    return ar.standardized_residuals(score_sequence)


# ================== Lightweight HMM for State Segmentation ==================

class GMM_HMM_Segmenter:
    """2-state HMM on score sequence (normal/anomaly).

    Uses sklearn GaussianMixture initialization + simple Viterbi
    (no full EM; we approximate via fixed emission distributions).

    Useful for getting a smooth binary state sequence from noisy scores.
    """

    def __init__(self, transition_prob_stay=0.95):
        """
        Args:
            transition_prob_stay: probability of staying in same state
        """
        self.transition_stay = transition_prob_stay
        self.normal_mean = None
        self.normal_std = None
        self.anomaly_mean = None
        self.anomaly_std = None

    def fit(self, scores):
        """Fit 2-mode GMM, identify which is anomaly."""
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
        try:
            gmm.fit(scores.reshape(-1, 1))
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            anom_idx = int(np.argmax(means))
            self.anomaly_mean = float(means[anom_idx])
            self.anomaly_std = float(stds[anom_idx])
            self.normal_mean = float(means[1 - anom_idx])
            self.normal_std = float(stds[1 - anom_idx])
        except Exception:
            self.normal_mean = float(scores.mean())
            self.normal_std = float(scores.std() + 1e-9)
            self.anomaly_mean = float(scores.mean() + 2 * scores.std())
            self.anomaly_std = float(scores.std() + 1e-9)
        return self

    def viterbi(self, scores):
        """Viterbi decoding to get most likely state sequence.

        Returns:
            states: (T,) array {0=normal, 1=anomaly}
            posteriors: (T, 2) state posteriors via forward-backward
        """
        T = len(scores)
        # Log emission probabilities
        from scipy.stats import norm
        log_emit_normal = norm.logpdf(scores, loc=self.normal_mean, scale=self.normal_std)
        log_emit_anomaly = norm.logpdf(scores, loc=self.anomaly_mean, scale=self.anomaly_std)
        log_emit = np.stack([log_emit_normal, log_emit_anomaly], axis=1)  # (T, 2)

        # Log transitions
        p_stay = self.transition_stay
        p_switch = 1.0 - p_stay
        log_trans = np.log(np.array([
            [p_stay, p_switch],   # from normal
            [p_switch, p_stay],   # from anomaly
        ]))

        # Initial: uniform
        log_init = np.log(np.array([0.5, 0.5]))

        # Viterbi forward
        log_delta = log_init + log_emit[0]  # (2,)
        psi = np.zeros((T, 2), dtype=int)
        for t in range(1, T):
            for j in range(2):
                vals = log_delta + log_trans[:, j]
                psi[t, j] = int(np.argmax(vals))
                log_delta[j] = np.max(vals) + log_emit[t, j]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = int(np.argmax(log_delta))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states


def hmm_state_anomaly_score(scores, transition_stay=0.95):
    """Functional interface: HMM state probability."""
    seg = GMM_HMM_Segmenter(transition_prob_stay=transition_stay).fit(scores)
    states = seg.viterbi(scores)
    # Convert states to continuous score (smoothed)
    return states.astype(np.float64)


# ================== Spectral Subtraction ==================

def spectral_subtract(score, percentile_baseline=50, smooth_baseline=True):
    """Spectral subtraction: subtract background spectrum.

    Algorithm:
    1. Compute FFT of score
    2. Estimate baseline spectrum (low percentile of power)
    3. Subtract baseline from spectrum
    4. IFFT to get clean signal
    """
    fft = np.fft.fft(score)
    power = np.abs(fft) ** 2

    if smooth_baseline:
        from scipy.ndimage import gaussian_filter1d
        smoothed_power = gaussian_filter1d(power, sigma=5)
    else:
        smoothed_power = power

    baseline = np.percentile(smoothed_power, percentile_baseline)

    # Subtract baseline (keep positive)
    new_power = np.maximum(power - baseline, 0.1 * power)

    # New magnitude
    magnitude = np.sqrt(new_power)
    new_fft = magnitude * np.exp(1j * np.angle(fft))

    cleaned = np.real(np.fft.ifft(new_fft))
    return cleaned


# ================== Recurrence-based features ==================

def state_persistence_score(score, threshold_percentile=85, window=21):
    """For each point, measure how persistent the high-score state is locally.

    Algorithm:
    1. Binary: x > threshold
    2. Local sum (window) → persistence ratio (0 to 1)
    """
    threshold = np.percentile(score, threshold_percentile)
    binary = (score > threshold).astype(np.float64)
    from scipy.ndimage import uniform_filter1d
    persistence = uniform_filter1d(binary, size=window, mode='reflect')
    return persistence
