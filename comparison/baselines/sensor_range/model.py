"""
Sensor Range Deviation Baseline — paper-faithful (QuoVadisTAD ICML 2024).

Reference: quovadis_tad/baselines/simple_baselines.py::SensorRangeDeviation
- sensor_range=(0, 1) FIXED (eval_simple_baselines.py:36 calls with
  sensor_range=(0, 1)).
- transform: `(x < sensor_min) | (sensor_max < x)` → max(axis=1) (boolean).
- fit: train min/max only used if sensor_range is None at __init__.

Paper purpose: expose F1+PA evaluation issues — score is 0 when test stays in
[0, 1] (train fit minmax range), score is 1 when test exceeds [0, 1]
(i.e., out-of-distribution per train's min-max). Only meaningful when the
preceding min-max preprocessing does NOT clip test values.
"""

import numpy as np
from typing import Optional, Tuple, Union


class SensorRangeDeviation:
    """Paper-faithful Sensor Range Deviation baseline.

    Matches `quovadis_tad/baselines/simple_baselines.py::SensorRangeDeviation`
    line-by-line:
        outside_range = ((x < sensor_min) | (sensor_max < x)).astype(np.float32)
        any_sensor_outside_range = outside_range.max(axis=1)  # count_sensors=False
                                   outside_range.sum(axis=1)  # count_sensors=True
    """

    def __init__(
        self,
        sensor_range: Optional[Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]] = (0, 1),
        count_sensors: bool = False,
    ):
        self.sensor_range = sensor_range
        self.count_sensors = count_sensors
        self.name = "SensorRangeDeviation"

    def fit(self, train_data: np.ndarray) -> 'SensorRangeDeviation':
        # Paper line-by-line: if sensor_range is None, learn from train min/max.
        if self.sensor_range is None:
            self.sensor_range = (train_data.min(axis=0), train_data.max(axis=0))
        return self

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        sensor_min, sensor_max = self.sensor_range
        outside_range = ((test_data < sensor_min) | (sensor_max < test_data)).astype(np.float32)
        if self.count_sensors:
            scores = outside_range.sum(axis=1)
        else:
            scores = outside_range.max(axis=1)
        return scores.astype(np.float32)

    def __repr__(self) -> str:
        return f"SensorRangeDeviation(sensor_range={self.sensor_range}, count_sensors={self.count_sensors})"
