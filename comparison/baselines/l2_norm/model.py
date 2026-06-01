"""
L2-Norm Baseline for Time Series Anomaly Detection

Based on QuoVadisTAD (arXiv:2405.02678):
- Computes L2 (Euclidean) norm of each timestamp's feature vector
- No training required
- Assumes anomalies have larger feature magnitudes

This is the simplest possible baseline - it just measures how "extreme"
the sensor values are at each timestamp.
"""

import numpy as np
from typing import Optional


class L2Norm:
    """
    L2-Norm baseline that uses vector magnitude as anomaly score.

    This baseline assumes that anomalous timestamps have larger
    feature magnitudes (further from the origin in feature space).
    """

    def __init__(self, ord: int = 2, normalize: bool = False):
        """
        Args:
            ord: Norm order (2 for L2/Euclidean, 1 for L1/Manhattan)
            normalize: If True, normalize scores to [0, 1] using training stats
        """
        self.ord = ord
        self.normalize = normalize
        self.train_mean: Optional[float] = None
        self.train_std: Optional[float] = None
        self.name = "L2Norm"

    def fit(self, train_data: np.ndarray) -> 'L2Norm':
        """
        Compute normalization statistics from training data (optional).

        Args:
            train_data: Training data (n_samples, n_features)

        Returns:
            self
        """
        if self.normalize:
            train_norms = np.linalg.norm(train_data, ord=self.ord, axis=1)
            self.train_mean = train_norms.mean()
            self.train_std = train_norms.std()
            print(f"L2Norm fitting (normalize=True):")
            print(f"  Training norm mean: {self.train_mean:.4f}")
            print(f"  Training norm std: {self.train_std:.4f}")
        else:
            print(f"L2Norm: No fitting needed (normalize=False)")

        return self

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores as L^ord norm.

        Args:
            test_data: Test data (n_samples, n_features)

        Returns:
            Anomaly scores (norms) for each sample
        """
        scores = np.linalg.norm(test_data, ord=self.ord, axis=1)

        if self.normalize and self.train_std is not None and self.train_std > 0:
            # Z-score normalization using training statistics
            scores = (scores - self.train_mean) / self.train_std

        return scores.astype(np.float32)

    def __repr__(self) -> str:
        return f"L2Norm(ord={self.ord}, normalize={self.normalize})"


if __name__ == "__main__":
    # Test the baseline
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.wadi_loader import WaDiLoader

    data_path = Path(__file__).parent.parent.parent / "dataset/WaDi/WADI.A1_9 Oct 2017/WADI_attackdata_preprocessed.csv"
    loader = WaDiLoader(data_path=str(data_path), train_ratio=0.5)
    loader.load()

    train_X, train_y = loader.get_train_data()
    test_X, test_y = loader.get_test_data()

    model = L2Norm(ord=2, normalize=True)
    model.fit(train_X)
    scores = model.predict(test_X)

    print(f"\nL2Norm Results:")
    print(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    print(f"  Score mean: {scores.mean():.6f}")
    print(f"  Score std: {scores.std():.6f}")

    # Check correlation with labels
    from scipy.stats import spearmanr
    corr, pval = spearmanr(scores, test_y)
    print(f"  Spearman correlation with labels: {corr:.4f} (p={pval:.2e})")
