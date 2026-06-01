"""
Unified Data Loader for Baseline Comparison Experiments

Single loader that wraps MAE raw loaders + z-score normalization.
All data loading and preprocessing uses MAE code directly — no duplication.

Data pipeline (identical to MAE):
  1. Raw data loading     ← mae_anomaly.datasets.loaders
  2. Z-score (train-fit)  ← mae_anomaly.dataset_sliding._standardize_per_feature
  3. Train/test split
  4. Optional: normalonly variant (remove anomalies from train)

Supported datasets:
  - swat: SWaT A1+A2 (load_swat_a1a2_raw)
  - wadi: WaDi A1 or A2 attack-only (load_wadi_attack_raw)
  - wadi_14days: WaDi 14days+attack combined (load_wadi_14days_raw)
  - simulation: Synthetic data, complexity=False (load_simulation)
  - simulation_complex: Synthetic data, complexity=True (load_simulation_complex)
  - smd_block_split: SMD K-block interleaved split (load_smd_block_split)
  - smd_simple: SMD simple 50/50 split per machine (load_smd_simple)
  - psm: PSM Pooled Server Metrics, eBay (load_psm)

Supported variants:
  - None (standard): Full train/test as-is
  - 'normalonly': Remove anomaly regions from training data
"""

import sys
from pathlib import Path
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mae_anomaly.datasets.loaders import (
    load_swat_combined,
    load_wadi_attack_raw,
    load_wadi_14days_raw,
    load_simulation,
    load_simulation_complex,
    load_smd_block_split,
    load_smd_simple,
    load_psm,
    load_exathlon,
)
from mae_anomaly.dataset_sliding import (
    _standardize_per_feature,
    _minmax_per_feature,
    AnomalyRegion,
)
from mae_anomaly.evaluator import find_swat_largest_region


# ============================================================
# Data structures
# ============================================================

@dataclass
class NormalSegment:
    """Contiguous normal segment in training data (for normalonly variant)."""
    start: int       # inclusive (in concatenated normal-only space)
    end: int         # exclusive
    features: np.ndarray  # (length, n_features)

    @property
    def length(self) -> int:
        return self.end - self.start


# ============================================================
# Unified Loader
# ============================================================

class UnifiedLoader:
    """Unified data loader for all comparison experiments.

    Uses MAE raw loaders and z-score normalization directly.
    No duplicate code — changes to MAE pipeline automatically propagate here.
    """

    def __init__(
        self,
        dataset: str,
        scenario: str = None,
        variant: str = None,
        normalize_mode: str = 'zscore',
        # Simulation parameters
        total_length: int = 275000,
        train_ratio: float = 0.8,
        num_features: int = 8,
        interval_scale: float = 0.75,
        seed: int = 42,
        # Simulation normalonly uses complexity=True (legacy behavior)
        complexity: bool = None,
        # SMD block split parameters
        machine: str = None,
        k_blocks: int = 6,
        parity: int = 0,
        margin: int = 500,
        # Exathlon parameters
        app: int = None,
    ):
        """
        Args:
            dataset: One of 'swat', 'wadi', 'wadi_14days', 'simulation',
                     'simulation_complex', 'smd_block_split', 'smd_simple',
                     'psm', 'exathlon'
            scenario: 'A1' or 'A2' (for wadi/wadi_14days only)
            variant: None (standard) or 'normalonly'
            normalize_mode: 'zscore' (z-score standardization) or 'minmax' (min-max to [0,1])
            total_length: Simulation total timesteps
            train_ratio: Simulation train ratio
            num_features: Simulation feature count
            interval_scale: Simulation anomaly interval scale
            seed: Random seed
            complexity: Override complexity flag for simulation.
                        If None, derived from dataset name.
            machine: SMD machine ID (e.g. 'machine-1-1') for smd_block_split
            k_blocks: Number of blocks for SMD block split (default: 6)
            parity: Block assignment parity for SMD (0 or 1)
            margin: Safety margin for SMD block boundaries (default: 500)
            app: Exathlon application ID (1, 2, 4, 5, 6, 9) for exathlon
        """
        self.dataset = dataset
        self.scenario = scenario
        self.variant = variant
        self.normalize_mode = normalize_mode
        self.total_length_param = total_length
        self.train_ratio_param = train_ratio
        self.num_features_param = num_features
        self.interval_scale = interval_scale
        self.seed = seed
        self.complexity = complexity
        self.machine = machine
        self.k_blocks = k_blocks
        self.parity = parity
        self.margin = margin
        self.app = app

        # Populated by load()
        self.features: Optional[np.ndarray] = None    # z-score normalized
        self.labels: Optional[np.ndarray] = None
        self.anomaly_regions: List[AnomalyRegion] = []
        self.feature_names: List[str] = []
        self.train_ratio: float = 0.0
        self.data_info: Dict = {}
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None

        # Train/test arrays
        self.train_features: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None
        self.test_features: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None
        self.train_end: int = 0
        self.n_features: int = 0

        # NormalOnly state
        self.normal_segments: List[NormalSegment] = []
        self.original_train_length: int = 0

        # SWaT excl22 state (largest test anomaly region)
        self._largest_test_region: Optional[AnomalyRegion] = None

    def load(self) -> 'UnifiedLoader':
        """Load data using MAE raw loaders, apply z-score, split, apply variant."""

        # --- Step 1: Raw data loading (MAE loaders) ---
        features_raw, labels, anomaly_regions, feat_cols, train_ratio, data_info = \
            self._call_raw_loader()

        self.labels = labels.astype(np.int64)
        self.anomaly_regions = anomaly_regions
        self.feature_names = feat_cols if isinstance(feat_cols, list) else list(feat_cols)
        self.train_ratio = train_ratio
        self.data_info = data_info
        self.n_features = features_raw.shape[1]
        self.train_end = int(len(labels) * train_ratio)

        # --- Step 2: Normalization (train-only fit) ---
        # 'none': raw passed through; wrapper applies its own scaler (paper-faithful path
        # for RevIN-using SOTA models that assume zscore input — TimesNet/ModernTCN/
        # DCdetector/CATCH/TFMAE per upstream data_provider). Other 18 baselines stay on
        # minmax (Q1/Q3 policy); MAE pipeline is independent (own loaders).
        if self.normalize_mode == 'none':
            self.features = features_raw.astype(np.float32)
            # Identity sentinels (not used for inverse transform); keep names for
            # downstream code that may reference scaler attrs.
            self.scaler_min = None
            self.scaler_range = None
            print(f"\n  No normalization (raw data — wrapper applies its own scaler).")
        elif self.normalize_mode == 'minmax':
            self.features, self.scaler_min, self.scaler_range = _minmax_per_feature(
                features_raw.astype(np.float32), self.train_end
            )
            print(f"\n  Min-max normalization (train-only fit, clipped [0,1]):")
            print(f"    Train min range:  [{self.scaler_min.min():.4f}, {self.scaler_min.max():.4f}]")
            print(f"    Train range:      [{self.scaler_range.min():.4f}, {self.scaler_range.max():.4f}]")
        else:
            self.features, self.scaler_mean, self.scaler_std = _standardize_per_feature(
                features_raw.astype(np.float32), self.train_end
            )
            print(f"\n  Z-score normalization (train-only fit):")
            print(f"    Train mean range: [{self.scaler_mean.min():.4f}, {self.scaler_mean.max():.4f}]")
            print(f"    Train std range:  [{self.scaler_std.min():.4f}, {self.scaler_std.max():.4f}]")

        # --- Step 3: Train/test split ---
        self.train_features = self.features[:self.train_end]
        self.train_labels = self.labels[:self.train_end]
        self.test_features = self.features[self.train_end:]
        self.test_labels = self.labels[self.train_end:]
        self.original_train_length = self.train_end

        print(f"\n  Train/Test Split (ratio={train_ratio:.3f}):")
        print(f"    Train: {len(self.train_labels):,} samples, "
              f"anomaly ratio: {self.train_labels.mean():.2%}")
        print(f"    Test:  {len(self.test_labels):,} samples, "
              f"anomaly ratio: {self.test_labels.mean():.2%}")

        # --- Step 4: Variant ---
        if self.variant == 'normalonly':
            self._apply_normalonly()

        # --- SWaT: identify largest test anomaly region (excl22) ---
        if self.dataset == 'swat':
            self._identify_excl22_region()

        return self

    # ============================================================
    # Raw loader dispatch
    # ============================================================

    def _call_raw_loader(self):
        """Call the appropriate MAE raw loader."""
        if self.dataset == 'swat':
            return load_swat_combined()

        elif self.dataset == 'wadi':
            if self.scenario is None:
                raise ValueError("scenario required for wadi dataset ('A1' or 'A2')")
            return load_wadi_attack_raw(scenario=self.scenario, swap=False)

        elif self.dataset == 'wadi_14days':
            if self.scenario is None:
                raise ValueError("scenario required for wadi_14days dataset ('A1' or 'A2')")
            return load_wadi_14days_raw(scenario=self.scenario)

        elif self.dataset == 'simulation':
            return load_simulation(
                total_length=self.total_length_param,
                num_features=self.num_features_param,
                train_ratio=self.train_ratio_param,
                random_seed=self.seed,
            )

        elif self.dataset == 'simulation_complex':
            return load_simulation_complex(
                total_length=self.total_length_param,
                num_features=self.num_features_param,
                train_ratio=self.train_ratio_param,
                random_seed=self.seed,
            )

        elif self.dataset == 'smd_block_split':
            if self.machine is None:
                raise ValueError("machine required for smd_block_split dataset (e.g. 'machine-1-1')")
            return load_smd_block_split(
                machine=self.machine,
                k_blocks=self.k_blocks,
                parity=self.parity,
                margin=self.margin,
            )

        elif self.dataset == 'smd_simple':
            if self.machine is None:
                raise ValueError("machine required for smd_simple dataset (e.g. 'machine-1-1')")
            return load_smd_simple(machine=self.machine)

        elif self.dataset == 'psm':
            return load_psm()

        elif self.dataset == 'exathlon':
            if self.app is None:
                raise ValueError("app required for exathlon dataset (e.g. 1, 2, 4, 5, 6, 9)")
            return load_exathlon(app=self.app)

        else:
            raise ValueError(f"Unknown dataset: '{self.dataset}'. "
                             f"Available: swat, wadi, wadi_14days, simulation, "
                             f"simulation_complex, smd_block_split, smd_simple, psm, exathlon")

    # ============================================================
    # NormalOnly variant
    # ============================================================

    def _apply_normalonly(self):
        """Remove anomaly regions from training data, track segment boundaries."""
        print(f"\n  Applying NormalOnly variant:")

        # Find anomaly regions within training data
        train_anomaly_regions = []
        for region in self.anomaly_regions:
            if region.start < self.train_end:
                clipped_start = region.start
                clipped_end = min(region.end, self.train_end)
                if clipped_end > clipped_start:
                    train_anomaly_regions.append((clipped_start, clipped_end))

        train_anomaly_regions.sort(key=lambda x: x[0])

        # Extract contiguous normal segments
        segments = []
        concat_offset = 0
        prev_end = 0

        for anom_start, anom_end in train_anomaly_regions:
            if prev_end < anom_start:
                seg_len = anom_start - prev_end
                seg_features = self.features[prev_end:anom_start].copy()
                segments.append(NormalSegment(
                    start=concat_offset,
                    end=concat_offset + seg_len,
                    features=seg_features,
                ))
                concat_offset += seg_len
            prev_end = anom_end

        # Final segment after last anomaly
        if prev_end < self.train_end:
            seg_len = self.train_end - prev_end
            seg_features = self.features[prev_end:self.train_end].copy()
            segments.append(NormalSegment(
                start=concat_offset,
                end=concat_offset + seg_len,
                features=seg_features,
            ))

        self.normal_segments = segments

        # Replace train data with concatenated normal segments
        if segments:
            self.train_features = np.concatenate(
                [seg.features for seg in segments], axis=0
            )
            self.train_labels = np.zeros(len(self.train_features), dtype=np.int64)
        else:
            self.train_features = np.empty((0, self.n_features), dtype=np.float32)
            self.train_labels = np.empty(0, dtype=np.int64)

        removed = self.original_train_length - len(self.train_features)
        print(f"    Train anomaly regions: {len(train_anomaly_regions)}")
        print(f"    Normal segments: {len(segments)}")
        print(f"    Original train: {self.original_train_length:,} → "
              f"Normal-only: {len(self.train_features):,} "
              f"(removed {removed:,}, retention {len(self.train_features)/self.original_train_length:.2%})")

        if segments:
            seg_lens = [s.length for s in segments]
            print(f"    Segment lengths: min={min(seg_lens):,}, max={max(seg_lens):,}, "
                  f"mean={np.mean(seg_lens):,.0f}")

    # ============================================================
    # SWaT excl22 (largest test anomaly region exclusion)
    # ============================================================

    def _identify_excl22_region(self):
        """Identify the largest anomaly region in test set for excl22 evaluation."""
        test_regions = self.get_test_anomaly_regions()
        if test_regions:
            self._largest_test_region = find_swat_largest_region(test_regions)
            if self._largest_test_region:
                r_len = self._largest_test_region.end - self._largest_test_region.start
                test_anomaly = int(self.test_labels.sum())
                print(f"\n  Largest test region (excl22): "
                      f"[{self._largest_test_region.start:,}, {self._largest_test_region.end:,}) "
                      f"= {r_len:,} points ({r_len/test_anomaly:.1%} of test anomaly)")

    # ============================================================
    # Public interface
    # ============================================================

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get training data (features, labels).

        For normalonly: returns concatenated normal segments (no anomalies).
        For standard: returns full training split.
        """
        return self.train_features, self.train_labels

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test data (features, labels)."""
        return self.test_features, self.test_labels

    def get_test_anomaly_regions(self) -> List[AnomalyRegion]:
        """Get anomaly regions in test set (test-local coordinates).

        Returns MAE's AnomalyRegion objects with start/end adjusted to test-local.
        """
        test_regions = []
        test_len = len(self.test_labels)
        for r in self.anomaly_regions:
            if r.end > self.train_end and r.start < self.train_end + test_len:
                local_start = max(0, r.start - self.train_end)
                local_end = min(test_len, r.end - self.train_end)
                if local_end > local_start:
                    test_regions.append(AnomalyRegion(
                        start=local_start,
                        end=local_end,
                        anomaly_type=r.anomaly_type,
                    ))
        return test_regions

    @property
    def excl_region(self) -> 'Optional[AnomalyRegion]':
        """The largest test anomaly region (for excl22 evaluation), or None."""
        return self._largest_test_region

    def get_excl22_test_range(self) -> Tuple[int, int]:
        """Get the largest anomaly region range in test (excl22).

        Returns (start, end) in test-local coordinates, or (0, 0) if not applicable.
        """
        if self._largest_test_region is None:
            return 0, 0
        return self._largest_test_region.start, self._largest_test_region.end

    def has_excl22_region(self) -> bool:
        """Check if a large anomaly region exists in test set."""
        start, end = self.get_excl22_test_range()
        return (end - start) > 0

    def get_excl22_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get test data with largest region excluded (excl22).

        Returns:
            (excl_features, excl_labels, excl_mask)
            excl_mask: boolean mask on original test data (True = kept)
        """
        start, end = self.get_excl22_test_range()
        mask = np.ones(len(self.test_labels), dtype=bool)
        if end > start:
            mask[start:end] = False
        return self.test_features[mask], self.test_labels[mask], mask

    def get_dataset_info(self) -> dict:
        """Get summary information about the dataset."""
        test_regions = self.get_test_anomaly_regions()
        info = {
            "dataset": self.dataset,
            "scenario": self.scenario,
            "variant": self.variant,
            "total_length": len(self.labels),
            "n_features": self.n_features,
            "train_length": len(self.train_labels),
            "test_length": len(self.test_labels),
            "train_anomaly_ratio": float(self.train_labels.mean()) if len(self.train_labels) > 0 else 0.0,
            "test_anomaly_ratio": float(self.test_labels.mean()),
            "n_test_anomaly_regions": len(test_regions),
            "normalization": f"{self.normalize_mode} (train-only fit)",
            "data_source": "raw",
        }
        if self.variant == 'normalonly':
            info["original_train_length"] = self.original_train_length
            info["n_normal_segments"] = len(self.normal_segments)
        if self.dataset == 'swat' and self.has_excl22_region():
            excl_start, excl_end = self.get_excl22_test_range()
            info["excl22_test_start"] = excl_start
            info["excl22_test_end"] = excl_end
            info["excl22_length"] = excl_end - excl_start
        # Include raw loader info
        info.update(self.data_info)
        return info

    # ============================================================
    # NormalOnly: Segment-aware windowing for neural baselines
    # ============================================================

    def create_windows_from_segments(
        self,
        seq_len: int,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding windows from normal segments, respecting boundaries.

        Windows are created within each segment independently —
        no window spans across anomaly-removal boundaries.

        Only available for normalonly variant.

        Args:
            seq_len: Window length
            stride: Stride between windows

        Returns:
            (windows, targets)
            windows: (n_windows, seq_len, n_features)
            targets: (n_windows, n_features)
        """
        if self.variant != 'normalonly':
            raise RuntimeError("create_windows_from_segments() only for normalonly variant")

        all_windows = []
        all_targets = []

        for segment in self.normal_segments:
            seg_data = segment.features
            seg_len = len(seg_data)

            if seg_len < seq_len + 1:
                continue

            for i in range(0, seg_len - seq_len, stride):
                all_windows.append(seg_data[i:i + seq_len])
                all_targets.append(seg_data[i + seq_len])

        if all_windows:
            windows = np.array(all_windows, dtype=np.float32)
            targets = np.array(all_targets, dtype=np.float32)
        else:
            windows = np.zeros((0, seq_len, self.n_features), dtype=np.float32)
            targets = np.zeros((0, self.n_features), dtype=np.float32)

        return windows, targets


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test UnifiedLoader")
    parser.add_argument("--dataset", default="simulation",
                        choices=["swat", "wadi", "wadi_14days", "simulation", "simulation_complex"])
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--variant", default=None, choices=[None, "normalonly"])
    args = parser.parse_args()

    loader = UnifiedLoader(
        dataset=args.dataset,
        scenario=args.scenario,
        variant=args.variant,
    )
    loader.load()

    info = loader.get_dataset_info()
    print("\nDataset Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Verify z-score
    train_X, _ = loader.get_train_data()
    test_X, _ = loader.get_test_data()
    print(f"\nZ-score verification:")
    print(f"  Train mean: {train_X.mean(axis=0)[:3]}...")
    print(f"  Train std:  {train_X.std(axis=0)[:3]}...")
    print(f"  Test mean:  {test_X.mean(axis=0)[:3]}...")

    # Test regions
    test_regions = loader.get_test_anomaly_regions()
    print(f"\nTest anomaly regions: {len(test_regions)}")
    for i, r in enumerate(test_regions[:5]):
        print(f"  {i+1}. [{r.start}, {r.end}) len={r.end-r.start}")
    if len(test_regions) > 5:
        print(f"  ... and {len(test_regions) - 5} more")

    # NormalOnly windowing test
    if args.variant == 'normalonly':
        windows, targets = loader.create_windows_from_segments(seq_len=5)
        print(f"\nSegment-aware windows (seq_len=5): {len(windows):,}")
        print(f"  Window shape: {windows.shape}")
