"""
PCA Error Baseline — paper-faithful (QuoVadisTAD ICML 2024).

Reference: quovadis_tad/baselines/simple_baselines.py::PCAError
- pca_dim='auto' branches (paper line-by-line):
    univariate → 2
    n_features <= 50 → 10
    else → 30
    if dim > n_features: dim = min(max(2, n_features // 5), n_features)
- svd_solver='full' (paper default)
- transform: `np.abs(x - reconstructed)` returns per-feature error (n, f).

Score aggregation matches upstream evaluation path
`data_utils.normalise_scores(norm='median-iqr', smooth=True, smooth_window=5).max(axis=1)`
(see `quovadis_tad/dataset_utils/data_utils.py:47-72` and
`quovadis_tad/evaluation/evaluation_utils.py:21`). The model-internal call
produces 1D scores of shape (T_test,) by:
  1. PCA fit/project/inverse_transform → per-feature absolute error (n, f).
  2. Median-IQR per-sensor normalize with epsilon=1e-2.
  3. 5-box smoothing — first `smooth_window` rows stay 0 (upstream behavior).
  4. Max over sensors → 1D float32 vector.
The previous mean-over-features aggregation has been removed.
"""

import numpy as np
from scipy.stats import iqr
from sklearn.decomposition import PCA
from typing import Optional, Union


def _median_iqr_smooth(arr: np.ndarray, smooth_window: int = 5, epsilon: float = 1e-2) -> np.ndarray:
    """Port of `quovadis_tad/dataset_utils/data_utils.py:47-72` for norm='median-iqr', smooth=True.

    Per-column (sensor-axis) median-IQR normalization followed by a
    centered-ish box smoothing window. Mirrors upstream verbatim:
      - `n_err_mid = np.median(test_delta, axis=0)`
      - `n_err_iqr = iqr(test_delta, axis=0)`
      - `err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)`
      - For i in range(smooth_window, len(err_scores)):
            smoothed[i] = mean(err_scores[i - w : i + w - 1], axis=0)
      - First `smooth_window` rows remain zero.
    """
    n_err_mid = np.median(arr, axis=0)
    n_err_iqr = iqr(arr, axis=0)
    err_scores = (arr - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

    smoothed = np.zeros_like(err_scores)
    n = len(err_scores)
    for i in range(smooth_window, n):
        lo = i - smooth_window
        hi = min(i + smooth_window - 1, n)
        if hi > lo:
            smoothed[i] = np.mean(err_scores[lo:hi], axis=0)
    return smoothed


class PCAError:
    """Paper-faithful PCA reconstruction error baseline.

    Matches `quovadis_tad/baselines/simple_baselines.py::PCAError` line-by-line
    for fit and per-feature reconstruction error. Score aggregation matches
    upstream evaluation path (median-IQR + 5-box smooth + max-over-sensors).
    """

    def __init__(
        self,
        pca_dim: Union[int, str] = 'auto',
        svd_solver: str = 'full',
        univariate: bool = False,
    ):
        self.pca_dim = pca_dim
        self.svd_solver = svd_solver
        self.univariate = univariate
        self.pca: Optional[PCA] = None
        self.n_features: int = 0
        self.name = "PCAError"

        if pca_dim != 'auto':
            self.pca = PCA(n_components=int(pca_dim), svd_solver=svd_solver)

    def fit(self, train_data: np.ndarray) -> 'PCAError':
        self.n_features = train_data.shape[1]

        if self.pca_dim == 'auto':
            # Paper line-by-line:
            if self.univariate:
                dim = 2
            elif self.n_features <= 50:
                dim = 10
            else:
                dim = 30

            if dim > train_data.shape[1]:
                old_dim = dim
                dim = min(max(2, train_data.shape[1] // 5), train_data.shape[1])
                print(f'Adjusting estimated number of PCA components from {old_dim} to {dim}.')

            self.pca = PCA(n_components=dim, svd_solver=self.svd_solver)

        print(f"PCAError fitting (paper-faithful):")
        print(f"  n_features={self.n_features}, n_components={self.pca.n_components}, svd_solver={self.svd_solver}")
        self.pca.fit(train_data)
        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"  Explained variance: {explained_var:.2%}")
        return self

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """Returns 1D scores (T_test,) float32.

        Paper transform returns (n, f) per-feature absolute reconstruction error;
        upstream evaluation then applies
        `normalise_scores(norm='median-iqr', smooth=True, smooth_window=5).max(axis=1)`
        (`evaluation_utils.py:21`, `data_utils.py:47-72`). We do the full aggregation
        here so the pipeline contract (1D float32) is satisfied paper-faithfully.
        """
        if self.pca is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        latent = self.pca.transform(test_data)
        reconstructed = self.pca.inverse_transform(latent)
        per_feature_error = np.abs(test_data - reconstructed)        # (n, f) paper-faithful
        normed = _median_iqr_smooth(per_feature_error, smooth_window=5, epsilon=1e-2)
        scores = normed.max(axis=1).astype(np.float32)               # paper-faithful aggregation
        return scores

    def __repr__(self) -> str:
        return f"PCAError(pca_dim={self.pca_dim}, svd_solver={self.svd_solver})"
