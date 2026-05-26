"""
Probabilistic anomaly modeling utilities.

본 module은 다음 4 family의 probabilistic methods를 제공:

1. Bayesian Online Change Point Detection (BOCPD)
   - Adams & MacKay 2007의 algorithm
   - Constant hazard rate H(τ) = 1/λ
   - Time-varying hazard (rate increases with run length)
   - 다양한 prior (Gaussian, Student-t)

2. Extreme Value Theory (EVT)
   - Peak-Over-Threshold (POT) with GPD fit
   - Block maxima with GEV fit
   - POT-based anomaly probability (p-values)

3. Gaussian Mixture Model (GMM)
   - 2-component GMM (normal + anomaly modes)
   - Per-point posterior probability of anomaly mode

4. Conformal Prediction
   - Calibrated p-values from training distribution
   - Various non-conformity scores
"""
import numpy as np
from scipy.stats import norm, t as student_t, genpareto, genextreme
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ================== Bayesian Online Change Point Detection ==================

class BOCPD:
    """Bayesian Online Change Point Detection (Adams & MacKay 2007).

    For each time point t, maintains posterior over run length r (time since last CP).
    Update rule:
        P(r_t = 0 | x_{1:t}) = sum_{r_{t-1}} P(r_{t-1} | x_{1:t-1}) * H(r_{t-1}) * pi(x_t | r_{t-1})
        P(r_t = r_{t-1}+1 | x_{1:t}) = P(r_{t-1} | x_{1:t-1}) * (1 - H(r_{t-1})) * pi(x_t | r_{t-1})
    where:
        H(r): hazard function (probability of change point given run length)
        pi(x_t | r): predictive distribution given run length

    Anomaly score: posterior probability of change point at time t = P(r_t = 0 | x_{1:t})
    """

    def __init__(self, hazard_lambda=100, prior_mean=0.0, prior_var=1.0,
                 hazard_mode='constant', prior_dist='gaussian'):
        """
        Args:
            hazard_lambda: characteristic timescale (mean of geometric prior for run length)
            prior_mean, prior_var: prior for Gaussian observation distribution
            hazard_mode: 'constant', 'logistic', 'time_varying'
            prior_dist: 'gaussian', 'student_t'
        """
        self.hazard_lambda = hazard_lambda
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.hazard_mode = hazard_mode
        self.prior_dist = prior_dist

    def _hazard(self, run_lengths):
        """Hazard function H(r): probability of change point given current run length."""
        if self.hazard_mode == 'constant':
            return np.full_like(run_lengths, 1.0 / self.hazard_lambda, dtype=np.float64)
        elif self.hazard_mode == 'logistic':
            # Logistic: increases with run length
            return 1.0 / (1.0 + np.exp(-(run_lengths.astype(np.float64) - self.hazard_lambda) / 20.0))
        elif self.hazard_mode == 'time_varying':
            # Linear increase
            return np.minimum(run_lengths.astype(np.float64) / (self.hazard_lambda * 5), 0.5)
        else:
            raise ValueError(f"Unknown hazard mode: {self.hazard_mode}")

    def _predictive(self, x, mean_acc, var_acc, n_acc):
        """Predictive distribution P(x_t | run_length)."""
        # Online posterior update for Gaussian with unknown mean, known variance
        # Sequential mean: μ_n = (n0*μ0 + sum(x_i)) / (n0 + n)
        # Var of predictive: σ² * (1 + 1/(n+n0))
        if self.prior_dist == 'gaussian':
            n0 = 1.0  # prior strength
            posterior_mean = (n0 * self.prior_mean + mean_acc * n_acc) / (n0 + n_acc + 1e-9)
            posterior_var = self.prior_var * (1.0 + 1.0 / (n0 + n_acc + 1.0))
            return norm.logpdf(x, loc=posterior_mean, scale=np.sqrt(posterior_var))
        elif self.prior_dist == 'student_t':
            # Use Student-t for heavy-tailed predictive (more robust)
            df = max(n_acc + 1, 3.0)
            posterior_mean = (1.0 * self.prior_mean + mean_acc * n_acc) / (1.0 + n_acc + 1e-9)
            posterior_var = self.prior_var * (1.0 + 1.0 / (n_acc + 2.0))
            return student_t.logpdf(x, df=df, loc=posterior_mean,
                                    scale=np.sqrt(posterior_var))
        else:
            raise ValueError(f"Unknown prior dist: {self.prior_dist}")

    def run(self, signal):
        """Run BOCPD on 1D signal.

        Returns:
            cp_prob: (T,) probability of change point at each time
            run_length_posterior: (T, T) full posterior matrix (subsampled for memory)
        """
        T = len(signal)
        # Run length distribution: list of dicts {run_length: probability}
        # Use vectorized approach: P[r] at each step
        max_r = min(T, 3000)  # cap for memory

        # Initialize: P(r_0 = 0) = 1
        log_R = np.full(max_r + 1, -np.inf)
        log_R[0] = 0.0  # log(1.0)
        cp_prob = np.zeros(T)

        # Running sufficient statistics for each run length
        # For Gaussian predictive: mean accumulator, count
        mean_acc = np.zeros(max_r + 1)
        sumsq_acc = np.zeros(max_r + 1)
        n_acc = np.zeros(max_r + 1)

        # Standardize signal
        s_mean, s_std = signal.mean(), signal.std() + 1e-9
        signal_normalized = (signal - s_mean) / s_std

        for t in range(T):
            x = signal_normalized[t]

            # Compute predictive likelihood for each possible run length r
            log_pred = np.full(max_r + 1, -np.inf)
            for r in range(max_r + 1):
                if log_R[r] > -np.inf:
                    if r == 0:
                        # Use prior only
                        log_pred[r] = norm.logpdf(x, loc=self.prior_mean,
                                                   scale=np.sqrt(self.prior_var))
                    else:
                        # Use accumulated statistics
                        m = mean_acc[r] / max(n_acc[r], 1)
                        log_pred[r] = self._predictive(x, m, sumsq_acc[r], n_acc[r])

            # Vectorized: hazard function for valid r
            r_arr = np.arange(max_r + 1)
            valid_mask = log_R > -np.inf
            H = self._hazard(r_arr)

            # Growth probabilities
            log_growth = log_R + log_pred + np.log(np.maximum(1.0 - H, 1e-15))

            # Change point probability (at r=0)
            log_cp = np.logaddexp.reduce(log_R[valid_mask] +
                                          log_pred[valid_mask] +
                                          np.log(np.maximum(H[valid_mask], 1e-15)))

            # Build new log_R
            new_log_R = np.full(max_r + 1, -np.inf)
            new_log_R[0] = log_cp
            new_log_R[1:max_r + 1] = log_growth[:max_r]

            # Normalize
            log_norm = np.logaddexp.reduce(new_log_R[new_log_R > -np.inf])
            new_log_R -= log_norm

            log_R = new_log_R
            cp_prob[t] = np.exp(log_R[0])

            # Update sufficient statistics
            # Shift accumulators (r → r+1)
            new_mean_acc = np.zeros(max_r + 1)
            new_sumsq_acc = np.zeros(max_r + 1)
            new_n_acc = np.zeros(max_r + 1)
            new_mean_acc[1:] = (mean_acc[:max_r] * n_acc[:max_r] + x) / (n_acc[:max_r] + 1)
            new_sumsq_acc[1:] = sumsq_acc[:max_r] + (x - new_mean_acc[1:]) * (x - mean_acc[:max_r])
            new_n_acc[1:] = n_acc[:max_r] + 1
            mean_acc, sumsq_acc, n_acc = new_mean_acc, new_sumsq_acc, new_n_acc

        return cp_prob


def bocpd_fast(signal, hazard_lambda=100, prior_var=1.0):
    """Fast vectorized BOCPD using running statistics.

    Simplified version (constant hazard, Gaussian, no log overflow).
    Returns:
        cp_prob: (T,) probability of change point
    """
    T = len(signal)
    s_mean, s_std = signal.mean(), signal.std() + 1e-9
    x = (signal - s_mean) / s_std

    hazard = 1.0 / hazard_lambda
    max_r = min(T, 1500)

    # Initialize
    log_R = np.full(max_r + 1, -np.inf)
    log_R[0] = 0.0
    cp_prob = np.zeros(T)

    # Sufficient stats per run length
    # Predictive: N(0, prior_var) for r=0; otherwise empirical
    mean_acc = np.zeros(max_r + 2)
    n_acc = np.zeros(max_r + 2, dtype=np.float64)

    for t in range(T):
        xt = x[t]

        # Predictive likelihood per run length
        log_pred = np.full(max_r + 1, -1e30)
        valid_idx = np.where(log_R > -1e10)[0]
        for r in valid_idx:
            if r == 0 or n_acc[r] < 1:
                log_pred[r] = norm.logpdf(xt, loc=0.0, scale=np.sqrt(prior_var))
            else:
                # Posterior mean and variance with prior strength 1
                pm = (0.0 * 1.0 + mean_acc[r] * n_acc[r]) / (1.0 + n_acc[r])
                pv = prior_var * (1.0 + 1.0 / (n_acc[r] + 1.0))
                log_pred[r] = norm.logpdf(xt, loc=pm, scale=np.sqrt(pv))

        # Growth: r → r+1 with prob (1-H)
        log_growth = log_R + log_pred + np.log(max(1.0 - hazard, 1e-15))
        # CP: r → 0 with prob H
        log_cp_components = log_R[valid_idx] + log_pred[valid_idx] + np.log(max(hazard, 1e-15))
        log_cp = np.logaddexp.reduce(log_cp_components) if len(log_cp_components) > 0 else -1e30

        new_log_R = np.full(max_r + 1, -1e30)
        new_log_R[0] = log_cp
        new_log_R[1:max_r + 1] = log_growth[:max_r]

        # Normalize
        log_norm = np.logaddexp.reduce(new_log_R[new_log_R > -1e10])
        new_log_R -= log_norm

        log_R = new_log_R
        cp_prob[t] = np.exp(log_R[0])

        # Update sufficient stats (r → r+1)
        new_mean = np.zeros(max_r + 2)
        new_n = np.zeros(max_r + 2)
        new_mean[1:max_r + 2] = (mean_acc[:max_r + 1] * n_acc[:max_r + 1] + xt) / (n_acc[:max_r + 1] + 1)
        new_n[1:max_r + 2] = n_acc[:max_r + 1] + 1
        mean_acc, n_acc = new_mean, new_n

    return cp_prob


# ================== Extreme Value Theory (EVT) ==================

class POTAnomalyScore:
    """Peak-Over-Threshold based anomaly probability.

    Algorithm:
    1. Fit GPD (Generalized Pareto Distribution) to exceedances above threshold u
    2. For each point x, compute p-value = P(X > x | X > u)
    3. Anomaly score = -log(p-value) (high score = rare extreme)
    """

    def __init__(self, threshold_percentile=95):
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.shape = None  # ξ
        self.scale = None  # σ
        self.exceed_count = None

    def fit(self, train_scores):
        """Fit GPD on training set exceedances."""
        self.threshold = np.percentile(train_scores, self.threshold_percentile)
        exceedances = train_scores[train_scores > self.threshold] - self.threshold
        if len(exceedances) < 10:
            # Not enough exceedances
            self.shape = 0.0
            self.scale = 1.0
            self.exceed_count = 0
            return self

        try:
            # Fit GPD to exceedances (loc=0, scale, shape)
            shape, _, scale = genpareto.fit(exceedances, floc=0)
            self.shape = float(shape)
            self.scale = max(float(scale), 1e-9)
            self.exceed_count = len(exceedances)
        except Exception:
            self.shape = 0.0
            self.scale = float(np.std(exceedances) + 1e-9)
            self.exceed_count = len(exceedances)
        return self

    def transform(self, scores):
        """Compute -log(p-value) for each score."""
        result = np.zeros_like(scores, dtype=np.float64)
        if self.threshold is None:
            return result

        # For points below threshold: p-value ~ 1 (no signal)
        above_mask = scores > self.threshold
        below_pvalue = 1.0
        result[~above_mask] = -np.log(below_pvalue + 1e-15)

        # For points above threshold: compute GPD p-value
        if above_mask.sum() > 0:
            exceed = scores[above_mask] - self.threshold
            # Survival function
            sf = genpareto.sf(exceed, c=self.shape, loc=0, scale=self.scale)
            # Combine with exceedance rate
            p_values = sf * 0.05  # approximately base exceedance rate = 1 - threshold_percentile/100
            result[above_mask] = -np.log(np.maximum(p_values, 1e-15))

        return result

    def fit_transform(self, scores):
        return self.fit(scores).transform(scores)


def pot_anomaly_score(scores, threshold_percentile=95, train_scores=None):
    """Functional interface for POT anomaly score.
    If train_scores not provided, uses scores itself (self-fit)."""
    pot = POTAnomalyScore(threshold_percentile)
    if train_scores is None:
        return pot.fit_transform(scores)
    else:
        return pot.fit(train_scores).transform(scores)


# ================== Gaussian Mixture Model (GMM) ==================

def fit_gmm_2component(scores, random_state=42):
    """Fit 2-component GMM on scores (normal + anomaly modes).

    Returns:
        gmm: fitted GaussianMixture
        anomaly_component_idx: index of the component with HIGHER mean (anomaly assumption)
    """
    scores_2d = scores.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=random_state,
                          covariance_type='full', max_iter=200)
    try:
        gmm.fit(scores_2d)
    except Exception:
        return None, 0

    # Identify anomaly component as the one with higher mean
    anomaly_idx = int(np.argmax(gmm.means_.flatten()))
    return gmm, anomaly_idx


def gmm_anomaly_posterior(scores, train_scores=None, random_state=42):
    """Per-point posterior probability of anomaly component.

    Algorithm:
    1. Fit GMM on train_scores (or scores if None)
    2. For each point in scores, compute P(component_anomaly | x)
    """
    fit_data = train_scores if train_scores is not None else scores
    gmm, anomaly_idx = fit_gmm_2component(fit_data, random_state)
    if gmm is None:
        return np.zeros_like(scores)

    posterior = gmm.predict_proba(scores.reshape(-1, 1))[:, anomaly_idx]
    return posterior


def fit_gmm_n_components(scores, n_components=3, random_state=42):
    """Fit GMM with arbitrary n_components."""
    scores_2d = scores.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=random_state,
                          covariance_type='full', max_iter=200)
    try:
        gmm.fit(scores_2d)
    except Exception:
        return None
    return gmm


# ================== Conformal Prediction ==================

class ConformalCalibrator:
    """Conformal p-value calibration.

    Non-conformity score: how unusual is x compared to training distribution.
    p-value: P(rank of x's nonconformity > rank threshold)

    Standard approach: use rank-based calibration.
    """

    def __init__(self, nonconformity='rank', tail='upper'):
        """
        Args:
            nonconformity: 'rank', 'distance_from_median', 'percentile'
            tail: 'upper' (anomaly = high), 'lower', 'two_sided'
        """
        self.nonconformity = nonconformity
        self.tail = tail
        self.cal_scores = None

    def fit(self, cal_scores):
        """cal_scores: calibration set scores (assumed normal)."""
        self.cal_scores = np.array(cal_scores).copy()
        self.cal_scores.sort()
        return self

    def predict_pvalue(self, test_scores):
        """Compute conformal p-value for each test score.

        p_value(x) = (1 + #{s in cal : s >= x}) / (1 + n_cal)
        """
        if self.cal_scores is None or len(self.cal_scores) == 0:
            return np.ones_like(test_scores)

        if self.tail == 'upper':
            # p_value = fraction of cal_scores >= test_score
            n_cal = len(self.cal_scores)
            test_arr = np.atleast_1d(test_scores)
            p_values = np.zeros_like(test_arr, dtype=np.float64)
            # Use searchsorted (binary search)
            ranks = np.searchsorted(self.cal_scores, test_arr, side='left')
            # rank gives # of cal_scores < test. # of cal_scores >= test = n_cal - rank
            n_above = n_cal - ranks
            p_values = (1 + n_above) / (1 + n_cal)
            return p_values
        elif self.tail == 'two_sided':
            # Symmetric: use absolute deviation from median
            median = np.median(self.cal_scores)
            cal_abs = np.abs(self.cal_scores - median)
            cal_abs.sort()
            test_abs = np.abs(np.atleast_1d(test_scores) - median)
            n_cal = len(cal_abs)
            ranks = np.searchsorted(cal_abs, test_abs, side='left')
            n_above = n_cal - ranks
            return (1 + n_above) / (1 + n_cal)
        else:
            raise NotImplementedError(self.tail)

    def predict_neg_log_pvalue(self, test_scores):
        """-log(p-value): high score = unusual = anomaly."""
        p = self.predict_pvalue(test_scores)
        return -np.log(np.maximum(p, 1e-15))


def conformal_anomaly_score(scores, cal_scores=None, tail='upper'):
    """Conformal p-value transform.
    If cal_scores not provided, use bottom 80% of scores as calibration (assumed normal)."""
    if cal_scores is None:
        # Use lower percentile portion as calibration
        threshold = np.percentile(scores, 80)
        cal_scores = scores[scores <= threshold]

    calibrator = ConformalCalibrator(tail=tail).fit(cal_scores)
    return calibrator.predict_neg_log_pvalue(scores)
