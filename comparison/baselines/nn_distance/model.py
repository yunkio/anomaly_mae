"""
1-NN Distance Baseline for Time Series Anomaly Detection

Based on QuoVadisTAD (arXiv:2405.02678):
- Stores training data as reference
- For each test point, computes distance to nearest neighbor in training set
- Higher distance = more anomalous

This is a simple but effective baseline that captures the idea that
anomalies are far from normal data in feature space.
"""

import numpy as np
from typing import Optional
from sklearn.metrics import pairwise_distances


class NNDistance:
    """
    1-Nearest Neighbor Distance baseline.

    Anomaly score is the Euclidean distance to the nearest training sample.
    Higher scores indicate more anomalous samples.
    """

    def __init__(self, distance: str = 'euclidean', subsample: Optional[int] = None):
        """
        Args:
            distance: Distance metric ('euclidean', 'manhattan', 'cosine')
            subsample: If set, subsample training data to this many points
                       to reduce memory and computation
        """
        self.distance = distance
        self.subsample = subsample
        self.train_data: Optional[np.ndarray] = None
        self.name = "NNDistance"

    def fit(self, train_data: np.ndarray) -> 'NNDistance':
        """
        Store training data as reference.

        Args:
            train_data: Training data (n_samples, n_features)

        Returns:
            self
        """
        if self.subsample is not None and len(train_data) > self.subsample:
            # Random subsample to reduce memory
            np.random.seed(42)
            indices = np.random.choice(len(train_data), self.subsample, replace=False)
            self.train_data = train_data[indices].copy()
            print(f"NNDistance: Subsampled training data from {len(train_data)} to {self.subsample}")
        else:
            self.train_data = train_data.copy()

        print(f"NNDistance fitting:")
        print(f"  Training samples stored: {len(self.train_data):,}")
        print(f"  Features: {self.train_data.shape[1]}")
        print(f"  Distance metric: {self.distance}")

        return self

    def predict(self, test_data: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """
        Compute anomaly scores as distance to nearest training neighbor.

        Args:
            test_data: Test data (n_samples, n_features)
            batch_size: Process in batches to reduce memory.
                Default 1000 (was 5000, lowered 2026-05-26). ``pairwise_distances``
                allocates a ``batch_size × train_rows`` float32 array; at the
                old 5000 with paper-faithful WaDi train (1.29M rows after
                ``subsample: 10000`` was removed during paper-faithful work)
                this single allocation hits ~25 GB and OOM-kills the process
                on a 32 GB box once other prgs hold a few GB. 1000 caps the
                marginal allocation at ~5 GB.

        Returns:
            Anomaly scores (distances) for each sample
        """
        if self.train_data is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_samples = len(test_data)
        scores = np.zeros(n_samples, dtype=np.float32)

        # Process in batches to reduce memory
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = test_data[i:end_idx]

            # Compute distances to all training samples
            distances = pairwise_distances(
                batch,
                self.train_data,
                metric=self.distance
            )

            # Take minimum distance (nearest neighbor)
            scores[i:end_idx] = distances.min(axis=1)

            if (i // batch_size) % 10 == 0:
                print(f"  Processed {end_idx:,}/{n_samples:,} samples")

        return scores.astype(np.float32)

    def __repr__(self) -> str:
        n_train = len(self.train_data) if self.train_data is not None else 0
        return f"NNDistance(distance='{self.distance}', n_train={n_train})"


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

    # Use subsample to reduce computation time for testing
    model = NNDistance(distance='euclidean', subsample=10000)
    model.fit(train_X)

    # Test on subset
    scores = model.predict(test_X[:5000])

    print(f"\nNNDistance Results (on 5000 test samples):")
    print(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    print(f"  Score mean: {scores.mean():.6f}")
    print(f"  Score std: {scores.std():.6f}")
