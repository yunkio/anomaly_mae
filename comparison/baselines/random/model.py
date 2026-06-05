"""
Random Baseline — paper-faithful (QuoVadisTAD ICML 2024).

Reference: quovadis_tad/baselines/simple_baselines.py::Random
- Outputs binary integer {0, 1} per timestamp via np.random.randint(low=0, high=2).
- Paper purpose: expose F1+PA evaluation issues (paper Section 3).
- No training required.
"""

import numpy as np
from typing import Optional


class RandomBaseline:
    """Paper-faithful Random baseline (binary {0, 1} per timestep).

    Matches `quovadis_tad/baselines/simple_baselines.py::Random.transform`
    line-by-line: `np.random.randint(low=0, high=2, size=x.shape[0])`.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.name = "Random"

    def fit(self, train_data: np.ndarray) -> 'RandomBaseline':
        # Paper: pass (no training needed).
        return self

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        np.random.seed(self.seed)
        # Paper-faithful: integer 0 or 1 per timestep.
        scores = np.random.randint(low=0, high=2, size=test_data.shape[0])
        return scores.astype(np.float32)

    def __repr__(self) -> str:
        return f"RandomBaseline(seed={self.seed}, paper-faithful binary)"
