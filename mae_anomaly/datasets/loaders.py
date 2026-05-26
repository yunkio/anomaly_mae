"""Dataset loaders for SWaT, WaDi, Simulation, and TEP datasets."""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional

from ..dataset_sliding import AnomalyRegion, SlidingWindowTimeSeriesGenerator, NormalDataComplexity

# Project root (mae_anomaly package root)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def clean_column_name(col: str) -> str:
    """Clean raw column names (remove Windows path prefix)."""
    if '\\' in col:
        parts = col.split('\\')
        return parts[-1]
    return col


def load_swat_combined():
    """Load SWaT A1 (normal) + A2 (attack) combined for training/testing.

    Train = A1 entire (all normal) + front 50% of A2
    Test = back 50% of A2

    Returns:
        signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info
    """
    data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'SWaT', 'SWaT.A1 & A2_Dec 2015')
    a1_path = os.path.join(data_dir, 'SWaT_A1_normal_raw.csv')
    a2_path = os.path.join(data_dir, 'SWaT_A2_attack_raw.csv')

    print(f"\n{'='*60}")
    print(f"Loading SWaT A1 (normal) + A2 (attack) data")
    print(f"{'='*60}")

    # Load A1 (normal)
    print(f"  Loading A1 normal: {a1_path}")
    df_a1 = pd.read_csv(a1_path)
    print(f"    Shape: {df_a1.shape}")

    # Load A2 (attack)
    print(f"  Loading A2 attack: {a2_path}")
    df_a2 = pd.read_csv(a2_path)
    df_a2.columns = df_a2.columns.str.strip()  # Raw CSV has leading spaces in some columns
    print(f"    Shape: {df_a2.shape}")

    # Feature columns (all except 'label')
    feature_cols = [c for c in df_a1.columns if c != 'label']

    # Extract data
    features_a1 = df_a1[feature_cols].values.astype(np.float32)
    labels_a1 = df_a1['label'].values.astype(np.int64)  # All 0 (normal)

    features_a2 = df_a2[feature_cols].values.astype(np.float32)
    labels_a2 = df_a2['label'].values.astype(np.int64)

    n_a1 = len(features_a1)
    n_a2 = len(features_a2)

    print(f"\n  A1: {n_a1:,} samples (all normal)")
    print(f"  A2: {n_a2:,} samples (normal={np.sum(labels_a2==0):,}, attack={np.sum(labels_a2==1):,})")

    # Concatenate A1 + A2
    all_features_raw = np.concatenate([features_a1, features_a2], axis=0)
    all_labels = np.concatenate([labels_a1, labels_a2], axis=0)
    n_total = n_a1 + n_a2
    print(f"  Combined: {n_total:,} samples, {len(feature_cols)} features")

    # Remove constant columns in combined data
    stds = np.std(all_features_raw, axis=0)
    constant_mask = stds == 0
    n_constant = np.sum(constant_mask)
    if n_constant > 0:
        print(f"  Removing {n_constant} constant columns")
        all_features_raw = all_features_raw[:, ~constant_mask]
        feature_cols = [f for f, m in zip(feature_cols, constant_mask) if not m]

    # Return raw (unnormalized) features.
    # Normalization is handled by SlidingWindowDataset (train-only fit z-score).
    all_features = all_features_raw.astype(np.float32)

    # Train/Test split: Train = A1 + front 50% A2, Test = back 50% A2
    train_len = n_a1 + n_a2 // 2
    train_ratio = train_len / n_total

    print(f"\n  Train/Test split:")
    print(f"    Train: {train_len:,} (A1 all + front 50% A2)")
    print(f"    Test:  {n_total - train_len:,} (back 50% A2)")
    print(f"    train_ratio: {train_ratio:.6f}")

    # Compute anomaly regions
    is_atk = (all_labels == 1).astype(int)
    diff = np.diff(is_atk, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    anomaly_regions = []
    for s, e in zip(starts, ends):
        anomaly_regions.append(AnomalyRegion(start=int(s), end=int(e), anomaly_type=1))

    # Split stats
    split_idx = int(n_total * train_ratio)
    train_labels = all_labels[:split_idx]
    test_labels = all_labels[split_idx:]

    # run_boundaries: A1 and A2 are from different recording periods
    run_boundaries = [n_a1]

    data_info = {
        'n_a1': n_a1,
        'n_a2': n_a2,
        'n_total': n_total,
        'n_features': all_features.shape[1],
        'train_len': train_len,
        'test_len': n_total - train_len,
        'train_ratio': train_ratio,
        'train_normal': int(np.sum(train_labels == 0)),
        'train_attack': int(np.sum(train_labels == 1)),
        'test_normal': int(np.sum(test_labels == 0)),
        'test_attack': int(np.sum(test_labels == 1)),
        'train_attack_ratio': float(np.mean(train_labels)),
        'test_attack_ratio': float(np.mean(test_labels)),
        'n_anomaly_regions_total': len(anomaly_regions),
        'run_boundaries': run_boundaries,
    }

    print(f"  Train anomaly ratio: {data_info['train_attack_ratio']:.2%}")
    print(f"  Test anomaly ratio: {data_info['test_attack_ratio']:.2%}")
    print(f"  Features: {data_info['n_features']}")
    print(f"  Anomaly regions: {len(anomaly_regions)}")
    print(f"  Run boundaries: {run_boundaries} (A1/A2 boundary)")

    return all_features, all_labels, anomaly_regions, feature_cols, train_ratio, data_info


def load_swat_combined_swap():
    """Load SWaT A1 (normal) + A2 (attack) with SWAPPED A2 halves.

    Train = A1 entire (all normal) + BACK 50% of A2
    Test = FRONT 50% of A2

    A2 is reordered as [back_half | front_half] before concatenation with A1,
    so the existing split logic (at n_a1 + n_a2//2) puts back_half in training
    and front_half in test.

    Returns:
        signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info
    """
    data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'SWaT', 'SWaT.A1 & A2_Dec 2015')
    a1_path = os.path.join(data_dir, 'SWaT_A1_normal_raw.csv')
    a2_path = os.path.join(data_dir, 'SWaT_A2_attack_raw.csv')

    print(f"\n{'='*60}")
    print(f"Loading SWaT A1 (normal) + A2 (attack) data [SWAP]")
    print(f"{'='*60}")

    # Load A1 (normal)
    print(f"  Loading A1 normal: {a1_path}")
    df_a1 = pd.read_csv(a1_path)
    print(f"    Shape: {df_a1.shape}")

    # Load A2 (attack)
    print(f"  Loading A2 attack: {a2_path}")
    df_a2 = pd.read_csv(a2_path)
    df_a2.columns = df_a2.columns.str.strip()  # Raw CSV has leading spaces in some columns
    print(f"    Shape: {df_a2.shape}")

    # Feature columns (all except 'label')
    feature_cols = [c for c in df_a1.columns if c != 'label']

    # Extract data
    features_a1 = df_a1[feature_cols].values.astype(np.float32)
    labels_a1 = df_a1['label'].values.astype(np.int64)  # All 0 (normal)

    features_a2 = df_a2[feature_cols].values.astype(np.float32)
    labels_a2 = df_a2['label'].values.astype(np.int64)

    n_a1 = len(features_a1)
    n_a2 = len(features_a2)
    mid_a2 = n_a2 // 2

    print(f"\n  A1: {n_a1:,} samples (all normal)")
    print(f"  A2: {n_a2:,} samples (normal={np.sum(labels_a2==0):,}, attack={np.sum(labels_a2==1):,})")

    # === SWAP: Reorder A2 as [back_half | front_half] ===
    print(f"\n  SWAP: Reordering A2 as [back {mid_a2:,}:{n_a2:,} | front 0:{mid_a2:,}]")
    features_a2_swapped = np.concatenate([features_a2[mid_a2:], features_a2[:mid_a2]], axis=0)
    labels_a2_swapped = np.concatenate([labels_a2[mid_a2:], labels_a2[:mid_a2]], axis=0)

    # Verify swap
    print(f"    A2 back half (→train): {n_a2 - mid_a2:,} samples, "
          f"attack={np.sum(labels_a2[mid_a2:]==1):,}")
    print(f"    A2 front half (→test): {mid_a2:,} samples, "
          f"attack={np.sum(labels_a2[:mid_a2]==1):,}")

    # Concatenate A1 + A2_swapped
    all_features_raw = np.concatenate([features_a1, features_a2_swapped], axis=0)
    all_labels = np.concatenate([labels_a1, labels_a2_swapped], axis=0)
    n_total = n_a1 + n_a2
    print(f"  Combined: {n_total:,} samples, {len(feature_cols)} features")

    # Remove constant columns in combined data
    stds = np.std(all_features_raw, axis=0)
    constant_mask = stds == 0
    n_constant = np.sum(constant_mask)
    if n_constant > 0:
        print(f"  Removing {n_constant} constant columns")
        all_features_raw = all_features_raw[:, ~constant_mask]
        feature_cols = [f for f, m in zip(feature_cols, constant_mask) if not m]

    # Return raw (unnormalized) features.
    # Normalization is handled by SlidingWindowDataset (train-only fit z-score).
    all_features = all_features_raw.astype(np.float32)

    # Train/Test split: same index as original (n_a1 + n_a2 // 2)
    # But now train has A1 + A2_back, test has A2_front
    train_len = n_a1 + mid_a2
    train_ratio = train_len / n_total

    print(f"\n  Train/Test split [SWAP]:")
    print(f"    Train: {train_len:,} (A1 all + BACK 50% A2)")
    print(f"    Test:  {n_total - train_len:,} (FRONT 50% A2)")
    print(f"    train_ratio: {train_ratio:.6f}")

    # Compute anomaly regions (on the swapped+concatenated data)
    is_atk = (all_labels == 1).astype(int)
    diff = np.diff(is_atk, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    anomaly_regions = []
    for s, e in zip(starts, ends):
        anomaly_regions.append(AnomalyRegion(start=int(s), end=int(e), anomaly_type=1))

    # Split stats
    split_idx = int(n_total * train_ratio)
    train_labels = all_labels[:split_idx]
    test_labels = all_labels[split_idx:]

    # run_boundaries: A1 and A2_swapped are from different recording periods.
    # Also A2_back and A2_front are non-contiguous within A2_swapped.
    run_boundaries = [n_a1, n_a1 + (n_a2 - mid_a2)]

    data_info = {
        'n_a1': n_a1,
        'n_a2': n_a2,
        'n_total': n_total,
        'n_features': all_features.shape[1],
        'train_len': train_len,
        'test_len': n_total - train_len,
        'train_ratio': train_ratio,
        'train_normal': int(np.sum(train_labels == 0)),
        'train_attack': int(np.sum(train_labels == 1)),
        'test_normal': int(np.sum(test_labels == 0)),
        'test_attack': int(np.sum(test_labels == 1)),
        'train_attack_ratio': float(np.mean(train_labels)),
        'test_attack_ratio': float(np.mean(test_labels)),
        'n_anomaly_regions_total': len(anomaly_regions),
        'swap': True,
        'r21_in_test': False,  # R#21 in back 50% of A2 → training
        'run_boundaries': run_boundaries,
    }

    print(f"  Train anomaly ratio: {data_info['train_attack_ratio']:.2%}")
    print(f"  Test anomaly ratio: {data_info['test_attack_ratio']:.2%}")
    print(f"  Features: {data_info['n_features']}")
    print(f"  Anomaly regions: {len(anomaly_regions)}")
    print(f"  Run boundaries: {run_boundaries} (A1/A2_back, A2_back/A2_front)")
    print(f"  R#21 in test: {data_info['r21_in_test']}")

    return all_features, all_labels, anomaly_regions, feature_cols, train_ratio, data_info


def load_wadi_14days_combined(scenario: str):
    """Load 14days + attack data combined for training/testing.

    Uses preprocessed attack data (already normalized with labels).
    Loads 14days raw data and extracts matching columns, then re-normalizes
    the combined dataset.

    Returns:
        signals: (N, num_features) normalized float32 array
        point_labels: (N,) int64 array (0=normal, 1=attack)
        anomaly_regions: List[AnomalyRegion]
        feature_names: List[str]
        train_ratio: float (proportion of data for training)
        data_info: dict with dataset statistics
    """
    if scenario == 'A1':
        data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'WaDi', 'WADI.A1_9 Oct 2017')
        days14_path = os.path.join(data_dir, 'WADI_14days.csv')
        attack_preprocessed_path = os.path.join(data_dir, 'WADI_attackdata_preprocessed.csv')
        days14_skiprows = 4  # Skip metadata lines
    else:  # A2
        data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'WaDi', 'WADI.A2_19 Nov 2019')
        days14_path = os.path.join(data_dir, 'WADI_14days_new.csv')
        attack_preprocessed_path = os.path.join(data_dir, 'WADI_attackdataLABLE_preprocessed.csv')
        days14_skiprows = 0  # No metadata lines

    print(f"\n{'='*60}")
    print(f"Loading WaDi {scenario} 14days + Attack data")
    print(f"{'='*60}")

    # Load preprocessed attack data (already has labels and normalization info)
    print(f"  Loading preprocessed attack: {attack_preprocessed_path}")
    df_attack = pd.read_csv(attack_preprocessed_path)
    print(f"    Shape: {df_attack.shape}")

    # Get feature columns from preprocessed attack data (all except 'label')
    attack_feature_cols = [c for c in df_attack.columns if c != 'label']
    attack_labels = df_attack['label'].values.astype(np.int64)

    # Note: preprocessed attack data may have its own scaling.
    # Final z-score normalization is applied by SlidingWindowDataset (train-only fit).
    # We need to load 14days raw and match these columns

    # Load 14days raw data
    print(f"  Loading 14days raw: {days14_path}")
    df_14days = pd.read_csv(days14_path, skiprows=days14_skiprows)
    print(f"    Shape: {df_14days.shape}")

    # Clean column names (remove Windows path prefix)
    df_14days.columns = [clean_column_name(c) for c in df_14days.columns]

    # Find matching columns between 14days and attack preprocessed
    days14_all_cols = list(df_14days.columns)
    matching_features = [c for c in attack_feature_cols if c in days14_all_cols]

    if len(matching_features) < len(attack_feature_cols):
        print(f"  WARNING: {len(attack_feature_cols) - len(matching_features)} features not found in 14days")
        print(f"    Using {len(matching_features)} matching features")
    else:
        print(f"  All {len(matching_features)} features matched")

    # Extract raw features from 14days
    features_14days_raw = df_14days[matching_features].values.astype(np.float32)
    labels_14days = np.zeros(len(features_14days_raw), dtype=np.int64)

    # Extract features from preprocessed attack (need to subset to matching features)
    features_attack = df_attack[matching_features].values.astype(np.float32)

    print(f"\n  14days: {len(features_14days_raw):,} samples (all normal)")
    print(f"  Attack: {len(features_attack):,} samples (normal={np.sum(attack_labels==0):,}, attack={np.sum(attack_labels==1):,})")

    # Concatenate 14days + attack
    # Note: Attack data is already normalized, but we'll re-normalize everything together
    # for consistency. This is the proper approach for fair comparison.
    all_features_raw = np.concatenate([features_14days_raw, features_attack], axis=0)
    all_labels = np.concatenate([labels_14days, attack_labels], axis=0)
    print(f"  Combined: {len(all_features_raw):,} samples")

    # Remove constant columns
    constant_mask = np.std(all_features_raw, axis=0) == 0
    n_constant = np.sum(constant_mask)
    if n_constant > 0:
        print(f"  Removing {n_constant} constant columns")
        all_features_raw = all_features_raw[:, ~constant_mask]
        matching_features = [f for f, m in zip(matching_features, constant_mask) if not m]

    # Handle NaN values
    nan_count = np.sum(np.isnan(all_features_raw))
    if nan_count > 0:
        print(f"  Handling {nan_count:,} NaN values (forward-fill + backward-fill)")
        df_temp = pd.DataFrame(all_features_raw)
        df_temp = df_temp.ffill().bfill()
        all_features_raw = df_temp.values.astype(np.float32)

    # Return raw (unnormalized) features.
    # Normalization is handled by SlidingWindowDataset (train-only fit z-score).
    all_features = all_features_raw.astype(np.float32)

    # Compute train_ratio: train = 14days + front 50% attack, test = back 50% attack
    n_14days = len(features_14days_raw)
    n_attack = len(features_attack)
    n_total = n_14days + n_attack

    # Split point: 14days + 50% of attack data
    train_len = n_14days + n_attack // 2
    train_ratio = train_len / n_total

    print(f"\n  Train/Test split:")
    print(f"    Train: {train_len:,} samples (14days + front 50% attack)")
    print(f"    Test:  {n_total - train_len:,} samples (back 50% attack)")
    print(f"    train_ratio: {train_ratio:.4f}")

    # Compute anomaly regions
    is_atk = (all_labels == 1).astype(int)
    diff = np.diff(is_atk, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    anomaly_regions = []
    for s, e in zip(starts, ends):
        anomaly_regions.append(AnomalyRegion(start=int(s), end=int(e), anomaly_type=1))

    print(f"  Anomaly regions: {len(anomaly_regions)}")

    # Compute split statistics
    split_idx = int(n_total * train_ratio)
    train_labels = all_labels[:split_idx]
    test_labels = all_labels[split_idx:]

    # run_boundaries: 14days and attack are from different recording periods
    run_boundaries = [n_14days]

    data_info = {
        'n_14days': n_14days,
        'n_attack': n_attack,
        'n_total': n_total,
        'n_features': all_features.shape[1],
        'train_len': train_len,
        'test_len': n_total - train_len,
        'train_ratio': train_ratio,
        'train_normal': int(np.sum(train_labels == 0)),
        'train_attack': int(np.sum(train_labels == 1)),
        'test_normal': int(np.sum(test_labels == 0)),
        'test_attack': int(np.sum(test_labels == 1)),
        'train_attack_ratio': float(np.mean(train_labels)),
        'test_attack_ratio': float(np.mean(test_labels)),
        'n_anomaly_regions_total': len(anomaly_regions),
        'run_boundaries': run_boundaries,
    }

    print(f"\n  Train anomaly ratio: {data_info['train_attack_ratio']:.2%}")
    print(f"  Test anomaly ratio: {data_info['test_attack_ratio']:.2%}")
    print(f"  Features: {data_info['n_features']}")
    print(f"  Run boundaries: {run_boundaries} (14days/attack boundary)")

    return all_features, all_labels, anomaly_regions, matching_features, train_ratio, data_info


# =============================================================================
# Background CPU Worker (same as run_wadi_ablation.py)
# =============================================================================


def load_wadi_a2(preprocessed_path: str):
    """Load preprocessed WaDi A2 dataset.

    Returns:
        signals: (N, num_features) float32 array (preprocessed CSV values)
        point_labels: (N,) int64 array (0=normal, 1=attack)
        anomaly_regions: List[AnomalyRegion]
        feature_names: List[str]
    """
    df = pd.read_csv(preprocessed_path)
    label_col = 'label'
    feature_cols = [c for c in df.columns if c != label_col]

    signals = df[feature_cols].values.astype(np.float32)
    point_labels = df[label_col].values.astype(np.int64)

    # Extract contiguous anomaly regions
    is_atk = (point_labels == 1).astype(int)
    diff = np.diff(is_atk, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    anomaly_regions = []
    for s, e in zip(starts, ends):
        anomaly_regions.append(AnomalyRegion(start=int(s), end=int(e), anomaly_type=1))

    return signals, point_labels, anomaly_regions, feature_cols


def load_wadi_attack_5050(preprocessed_path: str, swap: bool = False):
    """Load preprocessed WaDi attack dataset with 50:50 train/test split.

    Front 50% → train, back 50% → test (or swapped if swap=True).

    Returns:
        signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info
    """
    scenario = 'A1' if 'A1' in str(preprocessed_path) else 'A2'
    swap_str = ' [SWAP]' if swap else ''
    print(f"\n{'='*60}")
    print(f"Loading WaDi {scenario} attack data (50:50 split){swap_str}")
    print(f"{'='*60}")

    df = pd.read_csv(preprocessed_path)
    label_col = 'label'
    feature_cols = [c for c in df.columns if c != label_col]

    signals = df[feature_cols].values.astype(np.float32)
    point_labels = df[label_col].values.astype(np.int64)
    n_total = len(signals)
    mid = n_total // 2

    if swap:
        # Swap: back 50% → train, front 50% → test
        signals = np.concatenate([signals[mid:], signals[:mid]], axis=0)
        point_labels = np.concatenate([point_labels[mid:], point_labels[:mid]], axis=0)
        print(f"  SWAP: back half → train, front half → test")

    train_ratio = 0.5

    # Extract anomaly regions
    is_atk = (point_labels == 1).astype(int)
    diff = np.diff(is_atk, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    anomaly_regions = [AnomalyRegion(start=int(s), end=int(e), anomaly_type=1) for s, e in zip(starts, ends)]

    split_idx = int(n_total * train_ratio)
    train_labels = point_labels[:split_idx]
    test_labels = point_labels[split_idx:]

    data_info = {
        'n_total': n_total,
        'n_features': signals.shape[1],
        'train_len': split_idx,
        'test_len': n_total - split_idx,
        'train_ratio': train_ratio,
        'train_attack_ratio': float(np.mean(train_labels)),
        'test_attack_ratio': float(np.mean(test_labels)),
        'n_anomaly_regions_total': len(anomaly_regions),
        'swap': swap,
    }

    print(f"  Total: {n_total:,} samples, {signals.shape[1]} features")
    print(f"  Train: {split_idx:,}, anomaly ratio: {data_info['train_attack_ratio']:.2%}")
    print(f"  Test:  {n_total - split_idx:,}, anomaly ratio: {data_info['test_attack_ratio']:.2%}")
    print(f"  Anomaly regions: {len(anomaly_regions)}")

    return signals, point_labels, anomaly_regions, feature_cols, train_ratio, data_info


def load_simulation(
    total_length: int = 275000,
    num_features: int = 8,
    train_ratio: float = 0.8,
    random_seed: int = 42,
):
    """Generate simulation dataset (complexity=False).

    Returns:
        signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info
    """
    print(f"\n{'='*60}")
    print(f"Generating simulation dataset (complexity=False)")
    print(f"{'='*60}")

    complexity = NormalDataComplexity(enable_complexity=False)
    generator = SlidingWindowTimeSeriesGenerator(
        total_length=total_length,
        num_features=num_features,
        seed=random_seed,
        complexity=complexity,
    )

    signals, point_labels, anomaly_regions = generator.generate()
    feature_names = [f'feature_{i}' for i in range(num_features)]

    data_info = {
        'n_total': total_length,
        'n_features': num_features,
        'train_ratio': train_ratio,
        'train_len': int(total_length * train_ratio),
        'test_len': int(total_length * (1 - train_ratio)),
        'n_anomaly_regions_total': len(anomaly_regions),
        'dataset_type': 'simulation',
        'complexity': False,
    }

    print(f"  Generated: {total_length:,} samples, {num_features} features")
    print(f"  Anomaly regions: {len(anomaly_regions)}")
    print(f"  Train ratio: {train_ratio:.2%}")

    return signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info


def load_simulation_complex(
    total_length: int = 275000,
    num_features: int = 8,
    train_ratio: float = 0.8,
    random_seed: int = 42,
):
    """Generate simulation dataset with complexity=True.

    Returns:
        signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info
    """
    print(f"\n{'='*60}")
    print(f"Generating simulation dataset (complexity=True)")
    print(f"{'='*60}")

    complexity = NormalDataComplexity(enable_complexity=True)
    generator = SlidingWindowTimeSeriesGenerator(
        total_length=total_length,
        num_features=num_features,
        seed=random_seed,
        complexity=complexity,
    )

    signals, point_labels, anomaly_regions = generator.generate()
    feature_names = [f'feature_{i}' for i in range(num_features)]

    data_info = {
        'n_total': total_length,
        'n_features': num_features,
        'train_ratio': train_ratio,
        'train_len': int(total_length * train_ratio),
        'test_len': int(total_length * (1 - train_ratio)),
        'n_anomaly_regions_total': len(anomaly_regions),
        'dataset_type': 'simulation_complex',
        'complexity': True,
    }

    print(f"  Generated: {total_length:,} samples, {num_features} features")
    print(f"  Anomaly regions: {len(anomaly_regions)}")
    print(f"  Train ratio: {train_ratio:.2%}")

    return signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info


# =============================================================================
# TEP (Tennessee Eastman Process) Dataset Loader
# =============================================================================

# TEP fault type names (20 fault types)
TEP_FAULT_NAMES = [
    'normal',                          # 0: Normal operation
    'A/C feed ratio (stream 4)',       # 1: Step change
    'B composition (stream 4)',        # 2: Step change
    'D feed temperature (stream 2)',   # 3: Step change
    'Reactor cooling water inlet temp',# 4: Step change
    'Condenser cooling water inlet temp',# 5: Step change
    'A feed loss (stream 1)',          # 6: Step change
    'C header pressure loss (stream 4)',# 7: Step change
    'A,B,C composition (stream 4)',    # 8: Random variation
    'D feed temperature (stream 2)',   # 9: Random variation
    'C feed temperature (stream 4)',   # 10: Random variation
    'Reactor cooling water inlet temp',# 11: Random variation
    'Condenser cooling water inlet temp',# 12: Random variation
    'Reaction kinetics',               # 13: Slow drift
    'Reactor cooling water valve',     # 14: Sticking
    'Condenser cooling water valve',   # 15: Sticking
    'Unknown',                         # 16
    'Unknown',                         # 17
    'Unknown',                         # 18
    'Unknown',                         # 19
    'Unknown',                         # 20
]


def load_tep(
    fault_types: Optional[List[int]] = None,
    n_train_runs: int = 50,
    n_test_runs: int = 50,
    seed: int = 42,
):
    """Load TEP (Tennessee Eastman Process) dataset for anomaly detection.

    Constructs a continuous time series from independent simulation runs:
    - Training: fault-free testing runs (960 samples/run, all normal)
    - Testing: faulty testing runs (960 samples/run, fault onset at sample 160)

    Run boundaries are tracked to prevent sliding windows from crossing them.

    Args:
        fault_types: List of fault type numbers (1-20) to include. None = all 20.
        n_train_runs: Number of fault-free runs for training (max 500).
        n_test_runs: Number of faulty runs per fault type for testing (max 500).
        seed: Random seed for run selection.

    Returns:
        signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info
    """
    import pyreadr

    data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'TEP')
    rng = np.random.RandomState(seed)

    if fault_types is None:
        fault_types = list(range(1, 21))
    fault_types = sorted(fault_types)

    fault_types_str = ','.join(str(f) for f in fault_types)
    print(f"\n{'='*60}")
    print(f"Loading TEP dataset")
    print(f"  Fault types: {fault_types_str}")
    print(f"  Train runs: {n_train_runs}, Test runs/fault: {n_test_runs}")
    print(f"{'='*60}")

    # ---- Load fault-free testing data (for training) ----
    print(f"  Loading fault-free testing data...")
    ff_test_path = os.path.join(data_dir, 'TEP_FaultFree_Testing.RData')
    ff_data = pyreadr.read_r(ff_test_path)
    df_ff = list(ff_data.values())[0]

    # Identify feature columns (exclude metadata)
    meta_cols = {'faultNumber', 'simulationRun', 'sample'}
    feature_cols = [c for c in df_ff.columns if c not in meta_cols]

    # Select training runs
    available_train_runs = sorted(df_ff['simulationRun'].unique())
    n_train_runs = min(n_train_runs, len(available_train_runs))
    selected_train_runs = sorted(rng.choice(available_train_runs, size=n_train_runs, replace=False))
    print(f"    Selected {n_train_runs} fault-free runs (of {len(available_train_runs)} available)")

    # Extract training data: concatenate selected runs sequentially
    train_signals_list = []
    train_labels_list = []
    train_boundaries = []
    cumulative_len = 0

    for run_id in selected_train_runs:
        run_data = df_ff[df_ff['simulationRun'] == run_id].sort_values('sample')
        run_features = run_data[feature_cols].values.astype(np.float32)
        run_len = len(run_features)

        train_signals_list.append(run_features)
        train_labels_list.append(np.zeros(run_len, dtype=np.int64))

        cumulative_len += run_len
        train_boundaries.append(cumulative_len)

    train_signals = np.concatenate(train_signals_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    train_len = len(train_signals)
    # Remove last boundary (end of data, not an internal boundary)
    train_boundaries = train_boundaries[:-1]

    print(f"    Training: {train_len:,} samples ({n_train_runs} runs × ~960)")

    # Free fault-free data
    del df_ff, ff_data

    # ---- Load faulty testing data (for testing) ----
    print(f"  Loading faulty testing data...")
    faulty_test_path = os.path.join(data_dir, 'TEP_Faulty_Testing.RData')
    faulty_data = pyreadr.read_r(faulty_test_path)
    df_faulty = list(faulty_data.values())[0]

    # Fault onset: sample 160 (1-indexed) → index 159 (0-indexed)
    FAULT_ONSET_SAMPLE = 160  # 1-indexed, as per TEP description

    test_signals_list = []
    test_labels_list = []
    test_boundaries = []
    anomaly_regions = []
    test_cumulative_len = 0

    for fault_num in fault_types:
        df_fault = df_faulty[df_faulty['faultNumber'] == fault_num]
        available_test_runs = sorted(df_fault['simulationRun'].unique())
        n_select = min(n_test_runs, len(available_test_runs))
        selected_runs = sorted(rng.choice(available_test_runs, size=n_select, replace=False))

        for run_id in selected_runs:
            run_data = df_fault[df_fault['simulationRun'] == run_id].sort_values('sample')
            run_features = run_data[feature_cols].values.astype(np.float32)
            run_samples = run_data['sample'].values
            run_len = len(run_features)

            # Construct labels: normal before fault onset, anomaly after
            run_labels = np.zeros(run_len, dtype=np.int64)
            fault_onset_idx = np.searchsorted(run_samples, FAULT_ONSET_SAMPLE)
            run_labels[fault_onset_idx:] = 1

            # Anomaly region (offset by train_len + cumulative test position)
            region_start = train_len + test_cumulative_len + fault_onset_idx
            region_end = train_len + test_cumulative_len + run_len
            anomaly_regions.append(AnomalyRegion(
                start=int(region_start),
                end=int(region_end),
                anomaly_type=int(fault_num),
            ))

            test_signals_list.append(run_features)
            test_labels_list.append(run_labels)

            test_cumulative_len += run_len
            test_boundaries.append(train_len + test_cumulative_len)

    test_signals = np.concatenate(test_signals_list, axis=0)
    test_labels = np.concatenate(test_labels_list, axis=0)
    test_len = len(test_signals)
    # Remove last boundary
    test_boundaries = test_boundaries[:-1]

    print(f"    Testing: {test_len:,} samples "
          f"({len(fault_types)} faults × {n_test_runs} runs × ~960)")

    # Free faulty data
    del df_faulty, faulty_data

    # ---- Combine train + test ----
    all_signals_raw = np.concatenate([train_signals, test_signals], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    n_total = len(all_signals_raw)
    train_ratio = train_len / n_total

    # All run boundaries (internal only)
    run_boundaries = sorted(train_boundaries + test_boundaries)

    print(f"\n  Combined: {n_total:,} samples, {len(feature_cols)} features")
    print(f"  Run boundaries: {len(run_boundaries)}")

    # ---- Remove constant columns ----
    stds = np.std(all_signals_raw, axis=0)
    constant_mask = stds == 0
    n_constant = int(np.sum(constant_mask))
    if n_constant > 0:
        print(f"  Removing {n_constant} constant columns")
        all_signals_raw = all_signals_raw[:, ~constant_mask]
        feature_cols = [f for f, m in zip(feature_cols, constant_mask) if not m]

    # ---- Handle NaN ----
    nan_count = int(np.sum(np.isnan(all_signals_raw)))
    if nan_count > 0:
        print(f"  Handling {nan_count:,} NaN values (forward-fill + backward-fill)")
        df_temp = pd.DataFrame(all_signals_raw)
        df_temp = df_temp.ffill().bfill()
        all_signals_raw = df_temp.values.astype(np.float32)

    # Return raw (unnormalized) signals.
    # Normalization is handled by SlidingWindowDataset (train-only fit z-score).
    all_signals = all_signals_raw.astype(np.float32)

    # ---- Compute statistics ----
    split_idx = int(n_total * train_ratio)
    test_labels_split = all_labels[split_idx:]

    data_info = {
        'dataset_type': 'tep',
        'fault_types': fault_types,
        'n_train_runs': n_train_runs,
        'n_test_runs': n_test_runs,
        'n_total': n_total,
        'n_features': all_signals.shape[1],
        'train_len': train_len,
        'test_len': test_len,
        'train_ratio': train_ratio,
        'train_attack_ratio': 0.0,  # Training is all normal
        'test_attack_ratio': float(np.mean(test_labels_split)) if len(test_labels_split) > 0 else 0.0,
        'test_normal': int(np.sum(test_labels_split == 0)),
        'test_attack': int(np.sum(test_labels_split == 1)),
        'n_anomaly_regions_total': len(anomaly_regions),
        'run_boundaries': run_boundaries,
        'fault_onset_sample': FAULT_ONSET_SAMPLE,
    }

    print(f"\n  Train/Test split:")
    print(f"    Train: {train_len:,} samples (all normal)")
    print(f"    Test:  {test_len:,} samples")
    print(f"    train_ratio: {train_ratio:.4f}")
    print(f"    Test anomaly ratio: {data_info['test_attack_ratio']:.2%}")
    print(f"  Features: {data_info['n_features']}")
    print(f"  Anomaly regions: {len(anomaly_regions)}")

    return all_signals, all_labels, anomaly_regions, feature_cols, train_ratio, data_info


# =============================================================================
# SMD (Server Machine Dataset) Loader
# =============================================================================

# All 28 machine IDs in SMD
SMD_MACHINE_NAMES = [
    'machine-1-1', 'machine-1-2', 'machine-1-3', 'machine-1-4',
    'machine-1-5', 'machine-1-6', 'machine-1-7', 'machine-1-8',
    'machine-2-1', 'machine-2-2', 'machine-2-3', 'machine-2-4',
    'machine-2-5', 'machine-2-6', 'machine-2-7', 'machine-2-8',
    'machine-2-9',
    'machine-3-1', 'machine-3-2', 'machine-3-3', 'machine-3-4',
    'machine-3-5', 'machine-3-6', 'machine-3-7', 'machine-3-8',
    'machine-3-9', 'machine-3-10', 'machine-3-11',
]


def load_smd(
    machines: Optional[List[str]] = None,
):
    """Load SMD (Server Machine Dataset) for anomaly detection.

    Each machine has separate train (all normal) and test (with anomalies) files.
    Data is concatenated as [all_train | all_test] so train_ratio splits correctly.

    When multiple machines are loaded, run_boundaries prevent sliding windows
    from crossing machine boundaries.

    Args:
        machines: List of machine IDs to load (e.g. ['machine-1-1']).
                  None = all 28 machines.

    Returns:
        signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info
    """
    data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'SMD')

    if machines is None:
        machines = list(SMD_MACHINE_NAMES)
    machines = sorted(machines)

    print(f"\n{'='*60}")
    print(f"Loading SMD dataset")
    print(f"  Machines: {len(machines)} ({machines[0]}...{machines[-1]})" if len(machines) > 3
          else f"  Machines: {machines}")
    print(f"{'='*60}")

    # ---- Load all machines' train and test data separately ----
    all_train_signals = []
    all_test_signals = []
    all_test_labels = []
    train_lengths = []  # Per-machine train lengths
    test_lengths = []   # Per-machine test lengths

    num_features = None

    for machine_id in machines:
        train_path = os.path.join(data_dir, 'train', f'{machine_id}.txt')
        test_path = os.path.join(data_dir, 'test', f'{machine_id}.txt')
        label_path = os.path.join(data_dir, 'test_label', f'{machine_id}.txt')

        # Load train (comma-separated, no header)
        train_data = np.loadtxt(train_path, delimiter=',', dtype=np.float32)
        test_data = np.loadtxt(test_path, delimiter=',', dtype=np.float32)
        test_labels = np.loadtxt(label_path, dtype=np.int64)

        if num_features is None:
            num_features = train_data.shape[1]

        all_train_signals.append(train_data)
        all_test_signals.append(test_data)
        all_test_labels.append(test_labels)
        train_lengths.append(len(train_data))
        test_lengths.append(len(test_data))

    total_train = sum(train_lengths)
    total_test = sum(test_lengths)
    n_total = total_train + total_test

    print(f"  Loaded {len(machines)} machines, {num_features} features each")
    print(f"  Total train: {total_train:,} samples (all normal)")
    print(f"  Total test:  {total_test:,} samples")

    # ---- Concatenate: [all_train | all_test] ----
    # This ensures train_ratio splits correctly at the boundary
    train_concat = np.concatenate(all_train_signals, axis=0)
    test_concat = np.concatenate(all_test_signals, axis=0)
    test_labels_concat = np.concatenate(all_test_labels, axis=0)

    all_signals_raw = np.concatenate([train_concat, test_concat], axis=0)
    all_labels = np.concatenate([
        np.zeros(total_train, dtype=np.int64),
        test_labels_concat,
    ], axis=0)

    train_ratio = total_train / n_total

    # ---- Compute run boundaries ----
    # Train internal boundaries (between machines in train portion)
    run_boundaries = []
    cumulative = 0
    for i, tl in enumerate(train_lengths):
        cumulative += tl
        if i < len(train_lengths) - 1:  # Skip last (= total_train boundary = train/test split)
            run_boundaries.append(cumulative)

    # Test internal boundaries (between machines in test portion)
    cumulative = total_train
    for i, tl in enumerate(test_lengths):
        cumulative += tl
        if i < len(test_lengths) - 1:  # Skip last (= end of data)
            run_boundaries.append(cumulative)

    # ---- Compute anomaly regions (in test portion, offset by total_train) ----
    anomaly_regions = []
    test_offset = total_train
    for test_labels_machine in all_test_labels:
        is_atk = (test_labels_machine == 1).astype(int)
        diff = np.diff(is_atk, prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            anomaly_regions.append(AnomalyRegion(
                start=int(test_offset + s),
                end=int(test_offset + e),
                anomaly_type=1,
            ))
        test_offset += len(test_labels_machine)

    print(f"  Anomaly regions: {len(anomaly_regions)}")
    print(f"  Run boundaries: {len(run_boundaries)}")

    # ---- Feature names (anonymous) ----
    feature_names = [f'feature_{i}' for i in range(num_features)]

    # ---- Remove constant columns ----
    stds = np.std(all_signals_raw, axis=0)
    constant_mask = stds == 0
    n_constant = int(np.sum(constant_mask))
    if n_constant > 0:
        print(f"  Removing {n_constant} constant columns")
        all_signals_raw = all_signals_raw[:, ~constant_mask]
        feature_names = [f for f, m in zip(feature_names, constant_mask) if not m]
        num_features = len(feature_names)

    # ---- Handle NaN ----
    nan_count = int(np.sum(np.isnan(all_signals_raw)))
    if nan_count > 0:
        print(f"  Handling {nan_count:,} NaN values (forward-fill + backward-fill)")
        df_temp = pd.DataFrame(all_signals_raw)
        df_temp = df_temp.ffill().bfill()
        all_signals_raw = df_temp.values.astype(np.float32)

    # Return raw (unnormalized) signals.
    # Normalization is handled by SlidingWindowDataset (train-only fit z-score).
    all_signals = all_signals_raw.astype(np.float32)

    # ---- Compute statistics ----
    test_labels_split = all_labels[total_train:]

    data_info = {
        'dataset_type': 'smd',
        'machines': machines,
        'n_machines': len(machines),
        'n_total': n_total,
        'n_features': num_features,
        'train_len': total_train,
        'test_len': total_test,
        'train_ratio': train_ratio,
        'train_attack_ratio': 0.0,
        'test_attack_ratio': float(np.mean(test_labels_split)) if len(test_labels_split) > 0 else 0.0,
        'test_normal': int(np.sum(test_labels_split == 0)),
        'test_attack': int(np.sum(test_labels_split == 1)),
        'n_anomaly_regions_total': len(anomaly_regions),
        'run_boundaries': run_boundaries if run_boundaries else None,
    }

    print(f"\n  Train/Test split:")
    print(f"    Train: {total_train:,} samples (all normal)")
    print(f"    Test:  {total_test:,} samples")
    print(f"    train_ratio: {train_ratio:.4f}")
    print(f"    Test anomaly ratio: {data_info['test_attack_ratio']:.2%}")
    print(f"  Features: {num_features}")

    return all_signals, all_labels, anomaly_regions, feature_names, train_ratio, data_info


def _find_safe_cut_point(target: int, labels: np.ndarray, anomaly_regions_local: list, margin: int = 500) -> int:
    """Find the nearest safe cut point away from anomaly regions.

    A "safe" position is at least `margin` timestamps away from any anomaly region.
    If no such position exists, returns the position with maximum distance from
    the nearest anomaly.

    Args:
        target: Ideal cut position.
        labels: Binary label array for this machine's test data.
        anomaly_regions_local: List of (start, end) tuples in local coordinates.
        margin: Minimum distance from any anomaly region (default: window_size=500).

    Returns:
        Safe cut position index.
    """
    L = len(labels)

    def is_safe(pos):
        if pos <= 0 or pos >= L:
            return False
        for s, e in anomaly_regions_local:
            if s - margin <= pos <= e + margin:
                return False
        return True

    if is_safe(target):
        return target

    # Search outward from target
    for offset in range(1, L):
        for candidate in [target - offset, target + offset]:
            if 0 < candidate < L and is_safe(candidate):
                return candidate

    # Fallback: find position with maximum distance from any anomaly
    best_pos = target
    best_dist = 0
    search_start = max(1, target - L // 4)
    search_end = min(L - 1, target + L // 4)
    for pos in range(search_start, search_end):
        min_dist = L
        for s, e in anomaly_regions_local:
            if s <= pos <= e:
                min_dist = 0
                break
            dist = min(abs(pos - s), abs(pos - e))
            min_dist = min(min_dist, dist)
        if min_dist > best_dist:
            best_dist = min_dist
            best_pos = pos
    return best_pos


def _get_anomaly_regions_local(labels: np.ndarray) -> list:
    """Extract anomaly regions as (start, end) tuples from binary labels."""
    regions = []
    in_region = False
    start = 0
    for i, v in enumerate(labels):
        if v == 1 and not in_region:
            in_region = True
            start = i
        elif v == 0 and in_region:
            in_region = False
            regions.append((start, i - 1))
    if in_region:
        regions.append((start, len(labels) - 1))
    return regions


def load_smd_simple(machine: str):
    """Load a single SMD machine with simple train/test split.

    Same pattern as SWaT: Train = original train file (all normal) + front 50%
    of test file. Test = back 50% of test file.

    Args:
        machine: Machine ID (e.g. 'machine-1-1').

    Returns:
        signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info
    """
    data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'SMD')

    print(f"\n{'='*60}")
    print(f"Loading SMD simple split: {machine}")
    print(f"{'='*60}")

    # Load train file (all normal)
    train_path = os.path.join(data_dir, 'train', f'{machine}.txt')
    train_data = np.loadtxt(train_path, delimiter=',', dtype=np.float32)
    train_labels = np.zeros(len(train_data), dtype=np.int64)
    print(f"  Train file: {len(train_data):,} samples, {train_data.shape[1]} features (all normal)")

    # Load test file (contains normal + anomaly)
    test_path = os.path.join(data_dir, 'test', f'{machine}.txt')
    label_path = os.path.join(data_dir, 'test_label', f'{machine}.txt')
    test_data = np.loadtxt(test_path, delimiter=',', dtype=np.float32)
    test_labels = np.loadtxt(label_path, dtype=np.int64)
    print(f"  Test file:  {len(test_data):,} samples, {test_data.shape[1]} features")

    # Split test file 50/50
    test_split = len(test_data) // 2
    test_front = test_data[:test_split]
    test_front_labels = test_labels[:test_split]
    test_back = test_data[test_split:]
    test_back_labels = test_labels[test_split:]

    # Concatenate: [train_file | test_front_50%] + [test_back_50%]
    all_signals = np.concatenate([train_data, test_front, test_back], axis=0)
    all_labels = np.concatenate([train_labels, test_front_labels, test_back_labels], axis=0)
    num_features = all_signals.shape[1]

    total_train = len(train_data) + len(test_front)
    total_test = len(test_back)
    n_total = total_train + total_test
    train_ratio = total_train / n_total

    # Feature names
    feature_names = [f'feature_{i}' for i in range(num_features)]

    # Remove constant columns
    stds = np.std(all_signals, axis=0)
    constant_mask = stds == 0
    n_constant = int(np.sum(constant_mask))
    if n_constant > 0:
        print(f"  Removing {n_constant} constant columns")
        all_signals = all_signals[:, ~constant_mask]
        feature_names = [f for f, m in zip(feature_names, constant_mask) if not m]
        num_features = len(feature_names)

    # Handle NaN
    nan_count = int(np.sum(np.isnan(all_signals)))
    if nan_count > 0:
        print(f"  Handling {nan_count:,} NaN values (forward-fill + backward-fill)")
        df_temp = pd.DataFrame(all_signals)
        df_temp = df_temp.ffill().bfill()
        all_signals = df_temp.values.astype(np.float32)

    all_signals = all_signals.astype(np.float32)

    # Compute anomaly regions on concatenated data
    anomaly_regions = []
    is_atk = (all_labels == 1).astype(int)
    diff = np.diff(is_atk, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        anomaly_regions.append(AnomalyRegion(start=int(s), end=int(e), anomaly_type=1))

    # Statistics
    train_anom = int(np.sum(all_labels[:total_train]))
    test_anom = int(np.sum(all_labels[total_train:]))

    # run_boundaries: orig_train and test_front are from different recording periods.
    # Windows must not span this boundary.
    run_boundaries = [len(train_data)]

    data_info = {
        'dataset_type': 'smd_simple',
        'machine': machine,
        'n_total': n_total,
        'n_features': num_features,
        'train_len': total_train,
        'test_len': total_test,
        'train_ratio': train_ratio,
        'train_file_len': len(train_data),
        'test_file_len': len(test_data),
        'test_split_point': test_split,
        'train_attack_ratio': float(np.mean(all_labels[:total_train])) if total_train > 0 else 0.0,
        'test_attack_ratio': float(np.mean(all_labels[total_train:])) if total_test > 0 else 0.0,
        'run_boundaries': run_boundaries,
    }

    print(f"\n  Train/Test split:")
    print(f"    Train: {total_train:,} = {len(train_data):,}(orig train) + {len(test_front):,}(test front 50%)")
    print(f"      anomaly: {train_anom:,} ({data_info['train_attack_ratio']:.2%})")
    print(f"    Test:  {total_test:,} (test back 50%)")
    print(f"      anomaly: {test_anom:,} ({data_info['test_attack_ratio']:.2%})")
    print(f"    train_ratio: {train_ratio:.4f}")
    print(f"  Anomaly regions: {len(anomaly_regions)}")
    print(f"  Run boundaries: {run_boundaries} (orig_train / test_front boundary)")
    print(f"  Features: {num_features}")

    return all_signals, all_labels, anomaly_regions, feature_names, train_ratio, data_info


# =============================================================================
# Exathlon (Spark cluster anomaly benchmark — Jacob et al., VLDB 2021)
# =============================================================================

# Apps to use for evaluation. Apps 7, 8 are excluded:
#   - app7: no disturbed traces (test impossible)
#   - app8: no undisturbed traces (train impossible)
# TimeSeAD (Wagner 2023) additionally excludes apps 3 and 10 (distributional shift /
# short test set), but we follow the user's 6-app selection convention.
EXATHLON_APP_IDS = [1, 2, 4, 5, 6, 9]


def load_exathlon(app: int):
    """Load Exathlon dataset for a single application.

    Convention (user-defined):
      Train = all undisturbed traces concat + first floor(N_dist/2) disturbed traces
              (sorted by trace_id, the trailing integer in trace_name).
      Test  = remaining disturbed traces concat.

    Trace boundaries are recorded in `run_boundaries` so the sliding window cannot
    cross trace boundaries (each trace = one Spark application execution).

    Features: 19 FScustom features (preprocessed offline, see
              dataset/Exathlon/preprocess.py).

    Args:
        app: Application ID — must be in EXATHLON_APP_IDS.

    Returns:
        signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info
    """
    if app not in EXATHLON_APP_IDS:
        raise ValueError(
            f"app={app} not in EXATHLON_APP_IDS={EXATHLON_APP_IDS}. "
            f"Apps 7/8 are structurally invalid (no train or test data)."
        )

    data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'Exathlon', f'app{app}')
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"{data_dir} not found. Run `python dataset/Exathlon/preprocess.py` first."
        )

    print(f"\n{'='*60}")
    print(f"Loading Exathlon app{app}")
    print(f"{'='*60}")

    # ---- Discover trace files; classify undisturbed vs disturbed ----
    # Trace name format: {app}_{type}_{rate}_{trace_id}
    #   type 0 = undisturbed; type 1..5 = disturbed
    csv_paths = sorted(Path(data_dir).glob('*.csv'))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    undisturbed = []
    disturbed = []
    for p in csv_paths:
        name = p.stem
        parts = name.split('_')
        ttype = int(parts[1])
        tid = int(parts[-1])
        if ttype == 0:
            undisturbed.append((tid, p, name))
        else:
            disturbed.append((tid, p, name))

    # Sort by trace_id (the user's specified ordering)
    undisturbed.sort(key=lambda x: x[0])
    disturbed.sort(key=lambda x: x[0])

    n_dist = len(disturbed)
    n_dist_to_train = n_dist // 2  # floor
    train_dist = disturbed[:n_dist_to_train]
    test_dist = disturbed[n_dist_to_train:]

    print(f"  Undisturbed traces: {len(undisturbed)} → all to train")
    print(f"  Disturbed traces:   {n_dist} → {len(train_dist)} to train (first by id), "
          f"{len(test_dist)} to test")

    # ---- Load all traces; collect features + labels + boundaries ----
    # Train portion: undisturbed + first 50% disturbed (ordered)
    # Test portion:  remaining disturbed (ordered)
    train_features_list = []
    train_labels_list = []
    train_boundaries_internal = []  # offsets between concatenated train traces
    cum = 0
    feature_names = None
    for tid, p, name in undisturbed:
        df = pd.read_csv(p)
        if feature_names is None:
            feature_names = [c for c in df.columns if c not in ('t', 'label')]
        feats = df[feature_names].values.astype(np.float32)
        labels = df['label'].values.astype(np.int64)
        train_features_list.append(feats)
        train_labels_list.append(labels)
        cum += len(feats)
        train_boundaries_internal.append(cum)
    for tid, p, name in train_dist:
        df = pd.read_csv(p)
        feats = df[feature_names].values.astype(np.float32)
        labels = df['label'].values.astype(np.int64)
        train_features_list.append(feats)
        train_labels_list.append(labels)
        cum += len(feats)
        train_boundaries_internal.append(cum)
    train_boundaries_internal.pop()  # last one == total_train, drop (= train/test split point)

    test_features_list = []
    test_labels_list = []
    test_boundaries_internal_offsets = []  # relative to test start
    cum_test = 0
    for tid, p, name in test_dist:
        df = pd.read_csv(p)
        feats = df[feature_names].values.astype(np.float32)
        labels = df['label'].values.astype(np.int64)
        test_features_list.append(feats)
        test_labels_list.append(labels)
        cum_test += len(feats)
        test_boundaries_internal_offsets.append(cum_test)
    if test_boundaries_internal_offsets:
        test_boundaries_internal_offsets.pop()  # drop last (= end of data)

    train_features = np.concatenate(train_features_list, axis=0) if train_features_list else np.zeros((0, 19), dtype=np.float32)
    train_labels = np.concatenate(train_labels_list, axis=0) if train_labels_list else np.zeros(0, dtype=np.int64)
    test_features = np.concatenate(test_features_list, axis=0) if test_features_list else np.zeros((0, 19), dtype=np.float32)
    test_labels = np.concatenate(test_labels_list, axis=0) if test_labels_list else np.zeros(0, dtype=np.int64)

    total_train = len(train_features)
    total_test = len(test_features)
    n_total = total_train + total_test
    if n_total == 0:
        raise RuntimeError(f"Empty dataset for app{app}")

    all_signals = np.concatenate([train_features, test_features], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    train_ratio = total_train / n_total

    # ---- run_boundaries (absolute positions) ----
    # Trace boundaries inside train portion + trace boundaries inside test portion.
    # (train/test split point at total_train is the global boundary handled by train_ratio.)
    run_boundaries = list(train_boundaries_internal) + [
        total_train + ofs for ofs in test_boundaries_internal_offsets
    ]

    # ---- Handle NaN (after concat) ----
    nan_count = int(np.sum(np.isnan(all_signals)))
    if nan_count > 0:
        print(f"  Handling {nan_count:,} NaN values (forward-fill + backward-fill)")
        df_temp = pd.DataFrame(all_signals)
        df_temp = df_temp.ffill().bfill().fillna(0.0)
        all_signals = df_temp.values.astype(np.float32)

    # ---- Compute anomaly regions on the entire concatenated signal ----
    anomaly_regions = []
    is_atk = (all_labels == 1).astype(int)
    diff = np.diff(is_atk, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        anomaly_regions.append(AnomalyRegion(start=int(s), end=int(e), anomaly_type=1))

    # ---- Build data_info ----
    train_anom = int(np.sum(all_labels[:total_train]))
    test_anom = int(np.sum(all_labels[total_train:]))

    data_info = {
        'dataset_type': 'exathlon',
        'app': app,
        'n_undisturbed_traces': len(undisturbed),
        'n_disturbed_total': n_dist,
        'n_dist_to_train': n_dist_to_train,
        'n_dist_to_test': len(test_dist),
        'train_trace_names': [name for _, _, name in undisturbed] + [name for _, _, name in train_dist],
        'test_trace_names': [name for _, _, name in test_dist],
        'n_total': n_total,
        'n_features': all_signals.shape[1],
        'train_len': total_train,
        'test_len': total_test,
        'train_ratio': train_ratio,
        'train_attack_ratio': float(np.mean(all_labels[:total_train])) if total_train > 0 else 0.0,
        'test_attack_ratio': float(np.mean(all_labels[total_train:])) if total_test > 0 else 0.0,
        'test_normal': total_test - test_anom,
        'test_attack': test_anom,
        'n_anomaly_regions_total': len(anomaly_regions),
        'run_boundaries': run_boundaries if run_boundaries else None,
    }

    print(f"\n  Train/Test split:")
    print(f"    Train: {total_train:,} samples (anomaly: {train_anom:,} = {data_info['train_attack_ratio']:.2%})")
    print(f"    Test:  {total_test:,} samples (anomaly: {test_anom:,} = {data_info['test_attack_ratio']:.2%})")
    print(f"    train_ratio: {train_ratio:.4f}")
    print(f"  Anomaly regions: {len(anomaly_regions)}")
    print(f"  Run boundaries: {len(run_boundaries)} (trace boundaries within train + test)")
    print(f"  Features: {all_signals.shape[1]}")

    return all_signals, all_labels, anomaly_regions, feature_names, train_ratio, data_info


def load_psm():
    """Load PSM (Pooled Server Metrics, eBay) dataset with simple train/test split.

    Same pattern as SMD/SWaT: Train = original train file (all normal) + front 50%
    of test file. Test = back 50% of test file.

    PSM is a single contiguous server-monitoring stream from eBay
    (Abdulaal et al., KDD 2021). The original release ships three CSV files
    under `dataset/PSM/`:
        - train.csv:      timestamp_(min) + 25 anonymized server metrics, all normal
        - test.csv:       timestamp_(min) + 25 anonymized server metrics
        - test_label.csv: timestamp_(min) + label (0=normal, 1=anomaly)

    Returns:
        signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info
    """
    data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'PSM')

    print(f"\n{'='*60}")
    print(f"Loading PSM simple split")
    print(f"{'='*60}")

    # Load train file (all normal) — drop timestamp column
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    train_data = train_df.drop(columns=['timestamp_(min)']).values.astype(np.float32)
    train_labels = np.zeros(len(train_data), dtype=np.int64)
    print(f"  Train file: {len(train_data):,} samples, {train_data.shape[1]} features (all normal)")

    # Load test file + labels — drop timestamp column
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    label_df = pd.read_csv(os.path.join(data_dir, 'test_label.csv'))
    test_data = test_df.drop(columns=['timestamp_(min)']).values.astype(np.float32)
    test_labels = label_df['label'].values.astype(np.int64)
    print(f"  Test file:  {len(test_data):,} samples, {test_data.shape[1]} features")

    # Split test file 50/50
    test_split = len(test_data) // 2
    test_front = test_data[:test_split]
    test_front_labels = test_labels[:test_split]
    test_back = test_data[test_split:]
    test_back_labels = test_labels[test_split:]

    # Concatenate: [train_file | test_front_50%] + [test_back_50%]
    all_signals = np.concatenate([train_data, test_front, test_back], axis=0)
    all_labels = np.concatenate([train_labels, test_front_labels, test_back_labels], axis=0)
    num_features = all_signals.shape[1]

    total_train = len(train_data) + len(test_front)
    total_test = len(test_back)
    n_total = total_train + total_test
    train_ratio = total_train / n_total

    # Feature names (anonymized server metrics: feature_0 ~ feature_24)
    feature_names = [f'feature_{i}' for i in range(num_features)]

    # Remove constant columns
    stds = np.std(all_signals, axis=0)
    constant_mask = stds == 0
    n_constant = int(np.sum(constant_mask))
    if n_constant > 0:
        print(f"  Removing {n_constant} constant columns")
        all_signals = all_signals[:, ~constant_mask]
        feature_names = [f for f, m in zip(feature_names, constant_mask) if not m]
        num_features = len(feature_names)

    # Handle NaN
    nan_count = int(np.sum(np.isnan(all_signals)))
    if nan_count > 0:
        print(f"  Handling {nan_count:,} NaN values (forward-fill + backward-fill)")
        df_temp = pd.DataFrame(all_signals)
        df_temp = df_temp.ffill().bfill()
        all_signals = df_temp.values.astype(np.float32)

    all_signals = all_signals.astype(np.float32)

    # Compute anomaly regions on concatenated data
    anomaly_regions = []
    is_atk = (all_labels == 1).astype(int)
    diff = np.diff(is_atk, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        anomaly_regions.append(AnomalyRegion(start=int(s), end=int(e), anomaly_type=1))

    # Statistics
    train_anom = int(np.sum(all_labels[:total_train]))
    test_anom = int(np.sum(all_labels[total_train:]))

    # run_boundaries: orig_train and test_front are from different periods.
    # Windows must not span this boundary.
    run_boundaries = [len(train_data)]

    data_info = {
        'dataset_type': 'psm',
        'n_total': n_total,
        'n_features': num_features,
        'train_len': total_train,
        'test_len': total_test,
        'train_ratio': train_ratio,
        'train_file_len': len(train_data),
        'test_file_len': len(test_data),
        'test_split_point': test_split,
        'train_attack_ratio': float(np.mean(all_labels[:total_train])) if total_train > 0 else 0.0,
        'test_attack_ratio': float(np.mean(all_labels[total_train:])) if total_test > 0 else 0.0,
        'run_boundaries': run_boundaries,
    }

    print(f"\n  Train/Test split:")
    print(f"    Train: {total_train:,} = {len(train_data):,}(orig train) + {len(test_front):,}(test front 50%)")
    print(f"      anomaly: {train_anom:,} ({data_info['train_attack_ratio']:.2%})")
    print(f"    Test:  {total_test:,} (test back 50%)")
    print(f"      anomaly: {test_anom:,} ({data_info['test_attack_ratio']:.2%})")
    print(f"    train_ratio: {train_ratio:.4f}")
    print(f"  Anomaly regions: {len(anomaly_regions)}")
    print(f"  Run boundaries: {run_boundaries} (orig_train / test_front boundary)")
    print(f"  Features: {num_features}")

    return all_signals, all_labels, anomaly_regions, feature_names, train_ratio, data_info


def load_smd_block_split(
    machine: str,
    k_blocks: int = 6,
    parity: int = 0,
    margin: int = 500,
):
    """Load a single SMD machine with K-block interleaved train/test split.

    Instead of using the original train/test files (train=all normal, test=with anomalies),
    this loader splits only the TEST file into K blocks and alternates them between
    train and test partitions. This ensures anomaly regions are distributed evenly
    across both partitions (~50/50).

    Block boundaries are snapped away from anomaly regions (by at least `margin`
    timestamps) so that sliding windows at block edges always see normal data.

    The original TRAIN file (all normal) is NOT used — only the test file is split.

    Args:
        machine: Machine ID (e.g. 'machine-1-1').
        k_blocks: Number of blocks to split into (default: 6).
        parity: Which blocks go to train.
            0 = even-indexed blocks → train, odd → test (default).
            1 = odd-indexed blocks → train, even → test.
        margin: Minimum distance from anomaly regions for cut points (default: 500).

    Returns:
        signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info
    """
    data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'SMD')

    print(f"\n{'='*60}")
    print(f"Loading SMD block split: {machine}")
    print(f"  K={k_blocks}, parity={parity} ({'even' if parity == 0 else 'odd'}→train)")
    print(f"  margin={margin}")
    print(f"{'='*60}")

    # ---- Load test data only (contains both normal and anomaly) ----
    test_path = os.path.join(data_dir, 'test', f'{machine}.txt')
    label_path = os.path.join(data_dir, 'test_label', f'{machine}.txt')

    test_data = np.loadtxt(test_path, delimiter=',', dtype=np.float32)
    test_labels = np.loadtxt(label_path, dtype=np.int64)

    L = len(test_data)
    num_features = test_data.shape[1]

    print(f"  Test file: {L:,} samples, {num_features} features")

    # ---- Compute safe cut points ----
    anomaly_regions_local = _get_anomaly_regions_local(test_labels)

    cut_points = []
    for i in range(1, k_blocks):
        target = int(L * i / k_blocks)
        safe = _find_safe_cut_point(target, test_labels, anomaly_regions_local, margin)
        cut_points.append(safe)
    cut_points = sorted(set(cut_points))

    # Build block boundaries
    boundaries = [0] + cut_points + [L]
    blocks = []
    for i in range(len(boundaries) - 1):
        blocks.append((boundaries[i], boundaries[i + 1]))

    # ---- Assign blocks to train/test by parity ----
    train_blocks = [blocks[i] for i in range(len(blocks)) if i % 2 == parity]
    test_blocks = [blocks[i] for i in range(len(blocks)) if i % 2 != parity]

    print(f"  Blocks: {len(blocks)} (boundaries: {boundaries})")
    print(f"  Train blocks: {len(train_blocks)}, Test blocks: {len(test_blocks)}")

    # ---- Concatenate: [train_block_0 | train_block_1 | ...] [test_block_0 | test_block_1 | ...]
    train_signals_list = [test_data[s:e] for s, e in train_blocks]
    train_labels_list = [test_labels[s:e] for s, e in train_blocks]
    test_signals_list = [test_data[s:e] for s, e in test_blocks]
    test_labels_list = [test_labels[s:e] for s, e in test_blocks]

    train_signals = np.concatenate(train_signals_list, axis=0)
    train_labels_concat = np.concatenate(train_labels_list, axis=0)
    test_signals = np.concatenate(test_signals_list, axis=0)
    test_labels_concat = np.concatenate(test_labels_list, axis=0)

    total_train = len(train_signals)
    total_test = len(test_signals)
    n_total = total_train + total_test

    all_signals_raw = np.concatenate([train_signals, test_signals], axis=0)
    all_labels = np.concatenate([train_labels_concat, test_labels_concat], axis=0)

    train_ratio = total_train / n_total

    # ---- Compute run boundaries (between blocks within each partition) ----
    run_boundaries = []
    # Train internal boundaries
    cumulative = 0
    for i, (s, e) in enumerate(train_blocks):
        cumulative += (e - s)
        if i < len(train_blocks) - 1:
            run_boundaries.append(cumulative)

    # Test internal boundaries
    cumulative = total_train
    for i, (s, e) in enumerate(test_blocks):
        cumulative += (e - s)
        if i < len(test_blocks) - 1:
            run_boundaries.append(cumulative)

    # ---- Compute anomaly regions on concatenated data ----
    anomaly_regions = []
    is_atk = (all_labels == 1).astype(int)
    diff = np.diff(is_atk, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        anomaly_regions.append(AnomalyRegion(
            start=int(s),
            end=int(e),
            anomaly_type=1,
        ))

    # ---- Feature names ----
    feature_names = [f'feature_{i}' for i in range(num_features)]

    # ---- Remove constant columns ----
    stds = np.std(all_signals_raw, axis=0)
    constant_mask = stds == 0
    n_constant = int(np.sum(constant_mask))
    if n_constant > 0:
        print(f"  Removing {n_constant} constant columns")
        all_signals_raw = all_signals_raw[:, ~constant_mask]
        feature_names = [f for f, m in zip(feature_names, constant_mask) if not m]
        num_features = len(feature_names)

    # ---- Handle NaN ----
    nan_count = int(np.sum(np.isnan(all_signals_raw)))
    if nan_count > 0:
        print(f"  Handling {nan_count:,} NaN values (forward-fill + backward-fill)")
        df_temp = pd.DataFrame(all_signals_raw)
        df_temp = df_temp.ffill().bfill()
        all_signals_raw = df_temp.values.astype(np.float32)

    # Return raw (unnormalized) signals.
    # Normalization is handled by SlidingWindowDataset (train-only fit z-score).
    all_signals = all_signals_raw.astype(np.float32)

    # ---- Compute statistics ----
    train_anom_pts = int(np.sum(train_labels_concat))
    test_anom_pts = int(np.sum(test_labels_concat))

    data_info = {
        'dataset_type': 'smd_block_split',
        'machine': machine,
        'n_total': n_total,
        'n_features': num_features,
        'train_len': total_train,
        'test_len': total_test,
        'train_ratio': train_ratio,
        'train_attack_ratio': float(np.mean(train_labels_concat)) if total_train > 0 else 0.0,
        'test_attack_ratio': float(np.mean(test_labels_concat)) if total_test > 0 else 0.0,
        'test_normal': int(np.sum(test_labels_concat == 0)),
        'test_attack': test_anom_pts,
        'n_anomaly_regions_total': len(anomaly_regions),
        'run_boundaries': run_boundaries if run_boundaries else None,
        'k_blocks': k_blocks,
        'parity': parity,
        'margin': margin,
        'block_boundaries': boundaries,
        'train_blocks': train_blocks,
        'test_blocks': test_blocks,
    }

    print(f"\n  Train/Test split:")
    print(f"    Train: {total_train:,} samples (anomaly: {train_anom_pts:,}, {data_info['train_attack_ratio']:.2%})")
    print(f"    Test:  {total_test:,} samples (anomaly: {test_anom_pts:,}, {data_info['test_attack_ratio']:.2%})")
    print(f"    train_ratio: {train_ratio:.4f}")
    print(f"  Anomaly regions: {len(anomaly_regions)} (train partition)")
    test_anom_regions = [r for r in anomaly_regions if r.start >= total_train]
    print(f"  Anomaly regions: {len(test_anom_regions)} (test partition)")
    print(f"  Run boundaries: {len(run_boundaries)}")
    print(f"  Features: {num_features}")

    return all_signals, all_labels, anomaly_regions, feature_names, train_ratio, data_info


# =============================================================================
# Raw Dataset Loaders (no pre-normalization, all features preserved)
# =============================================================================

def _load_raw_csv(path: str):
    """Load a raw CSV file (features + label column).

    Returns:
        features: (N, F) float32 array
        labels: (N,) int64 array
        feature_names: list of str
    """
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != 'label']
    features = df[feature_cols].values.astype(np.float32)
    labels = df['label'].values.astype(np.int64)
    return features, labels, feature_cols


def _compute_anomaly_regions(labels: np.ndarray):
    """Extract contiguous anomaly regions from binary labels."""
    is_atk = (labels == 1).astype(int)
    diff = np.diff(is_atk, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [AnomalyRegion(start=int(s), end=int(e), anomaly_type=1)
            for s, e in zip(starts, ends)]


def load_swat_a1a2_raw(swap: bool = False):
    """Load SWaT A1+A2 from raw CSV (all features, no normalization).

    Train = A1 entire + front/back 50% of A2
    Test = back/front 50% of A2

    Args:
        swap: If True, back 50% A2 → train, front 50% → test.
    """
    data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'SWaT', 'SWaT.A1 & A2_Dec 2015')
    swap_str = ' [SWAP]' if swap else ''
    print(f"\n{'='*60}")
    print(f"Loading SWaT A1+A2 raw{swap_str}")
    print(f"{'='*60}")

    features_a1, labels_a1, feat_cols = _load_raw_csv(
        os.path.join(data_dir, 'SWaT_A1_normal_raw.csv'))
    features_a2, labels_a2, _ = _load_raw_csv(
        os.path.join(data_dir, 'SWaT_A2_attack_raw.csv'))

    n_a1, n_a2 = len(features_a1), len(features_a2)
    mid_a2 = n_a2 // 2

    print(f"  A1: {n_a1:,} (all normal), A2: {n_a2:,} "
          f"(attack={np.sum(labels_a2==1):,})")

    if swap:
        features_a2 = np.concatenate([features_a2[mid_a2:], features_a2[:mid_a2]])
        labels_a2 = np.concatenate([labels_a2[mid_a2:], labels_a2[:mid_a2]])

    all_features = np.concatenate([features_a1, features_a2], axis=0)
    all_labels = np.concatenate([labels_a1, labels_a2], axis=0)
    n_total = n_a1 + n_a2

    train_len = n_a1 + mid_a2
    train_ratio = train_len / n_total
    anomaly_regions = _compute_anomaly_regions(all_labels)

    split_idx = int(n_total * train_ratio)
    train_labels = all_labels[:split_idx]
    test_labels = all_labels[split_idx:]

    # run_boundaries: A1 and A2 are from different recording periods
    if swap:
        # [A1 | A2_back | A2_front]: boundaries at A1/A2_back and A2_back/A2_front
        run_boundaries = [n_a1, n_a1 + (n_a2 - mid_a2)]
    else:
        # [A1 | A2]: boundary at A1/A2
        run_boundaries = [n_a1]

    data_info = {
        'n_a1': n_a1, 'n_a2': n_a2, 'n_total': n_total,
        'n_features': all_features.shape[1],
        'train_len': train_len, 'test_len': n_total - train_len,
        'train_ratio': train_ratio,
        'train_normal': int(np.sum(train_labels == 0)),
        'train_attack': int(np.sum(train_labels == 1)),
        'test_normal': int(np.sum(test_labels == 0)),
        'test_attack': int(np.sum(test_labels == 1)),
        'train_attack_ratio': float(np.mean(train_labels)),
        'test_attack_ratio': float(np.mean(test_labels)),
        'n_anomaly_regions_total': len(anomaly_regions),
        'swap': swap, 'raw': True,
        'run_boundaries': run_boundaries,
    }

    print(f"  Combined: {n_total:,} x {all_features.shape[1]} features")
    print(f"  Train: {train_len:,}, Test: {n_total - train_len:,}")
    print(f"  Test anomaly ratio: {data_info['test_attack_ratio']:.2%}")
    print(f"  Anomaly regions: {len(anomaly_regions)}")
    print(f"  Run boundaries: {run_boundaries}")

    return all_features, all_labels, anomaly_regions, feat_cols, train_ratio, data_info


def load_swat_a4a5_raw():
    """Load SWaT A4+A5 from raw CSV (77 features, no normalization).

    50:50 train/test split.
    """
    data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'SWaT', 'SWaT.A4 & A5_Jul 2019')
    print(f"\n{'='*60}")
    print(f"Loading SWaT A4+A5 raw")
    print(f"{'='*60}")

    features, labels, feat_cols = _load_raw_csv(
        os.path.join(data_dir, 'SWaT_A4A5_raw.csv'))
    n_total = len(features)
    train_ratio = 0.5
    anomaly_regions = _compute_anomaly_regions(labels)

    split_idx = int(n_total * train_ratio)
    data_info = {
        'n_total': n_total, 'n_features': features.shape[1],
        'train_len': split_idx, 'test_len': n_total - split_idx,
        'train_ratio': train_ratio,
        'train_attack_ratio': float(np.mean(labels[:split_idx])),
        'test_attack_ratio': float(np.mean(labels[split_idx:])),
        'n_anomaly_regions_total': len(anomaly_regions),
        'raw': True,
    }

    print(f"  {n_total:,} x {features.shape[1]} features")
    print(f"  Test anomaly ratio: {data_info['test_attack_ratio']:.2%}")

    return features, labels, anomaly_regions, feat_cols, train_ratio, data_info


def load_swat_a6_raw():
    """Load SWaT A6 from raw CSV (81 features, no normalization).

    50:50 train/test split.
    """
    data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'SWaT', 'SWaT.A6_Dec 2019')
    print(f"\n{'='*60}")
    print(f"Loading SWaT A6 raw")
    print(f"{'='*60}")

    features, labels, feat_cols = _load_raw_csv(
        os.path.join(data_dir, 'SWaT_A6_raw.csv'))
    n_total = len(features)
    train_ratio = 0.5
    anomaly_regions = _compute_anomaly_regions(labels)

    split_idx = int(n_total * train_ratio)
    data_info = {
        'n_total': n_total, 'n_features': features.shape[1],
        'train_len': split_idx, 'test_len': n_total - split_idx,
        'train_ratio': train_ratio,
        'train_attack_ratio': float(np.mean(labels[:split_idx])),
        'test_attack_ratio': float(np.mean(labels[split_idx:])),
        'n_anomaly_regions_total': len(anomaly_regions),
        'raw': True,
    }

    print(f"  {n_total:,} x {features.shape[1]} features")
    print(f"  Test anomaly ratio: {data_info['test_attack_ratio']:.2%}")

    return features, labels, anomaly_regions, feat_cols, train_ratio, data_info


def load_wadi_attack_raw(scenario: str, swap: bool = False):
    """Load WaDi attack data from raw CSV (123 features, 50:50 split).

    Args:
        scenario: 'A1' or 'A2'
        swap: If True, back 50% → train, front 50% → test.
    """
    if scenario == 'A1':
        path = os.path.join(PROJECT_ROOT, 'dataset', 'WaDi',
                            'WADI.A1_9 Oct 2017', 'WADI_A1_attack_raw.csv')
    else:
        path = os.path.join(PROJECT_ROOT, 'dataset', 'WaDi',
                            'WADI.A2_19 Nov 2019', 'WADI_A2_attack_raw.csv')

    swap_str = ' [SWAP]' if swap else ''
    print(f"\n{'='*60}")
    print(f"Loading WaDi {scenario} attack raw (50:50){swap_str}")
    print(f"{'='*60}")

    features, labels, feat_cols = _load_raw_csv(path)
    n_total = len(features)
    mid = n_total // 2

    if swap:
        features = np.concatenate([features[mid:], features[:mid]], axis=0)
        labels = np.concatenate([labels[mid:], labels[:mid]], axis=0)

    train_ratio = 0.5
    anomaly_regions = _compute_anomaly_regions(labels)

    split_idx = int(n_total * train_ratio)
    data_info = {
        'n_total': n_total, 'n_features': features.shape[1],
        'train_len': split_idx, 'test_len': n_total - split_idx,
        'train_ratio': train_ratio,
        'train_attack_ratio': float(np.mean(labels[:split_idx])),
        'test_attack_ratio': float(np.mean(labels[split_idx:])),
        'n_anomaly_regions_total': len(anomaly_regions),
        'swap': swap, 'raw': True,
    }

    print(f"  {n_total:,} x {features.shape[1]} features")
    print(f"  Test anomaly ratio: {data_info['test_attack_ratio']:.2%}")

    return features, labels, anomaly_regions, feat_cols, train_ratio, data_info


def load_wadi_14days_raw(scenario: str):
    """Load WaDi 14days + attack combined from raw CSV.

    Train = 14days + front 50% attack, Test = back 50% attack.
    All features preserved (123), no normalization.
    """
    if scenario == 'A1':
        data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'WaDi', 'WADI.A1_9 Oct 2017')
        days14_path = os.path.join(data_dir, 'WADI_A1_14days_raw.csv')
        attack_path = os.path.join(data_dir, 'WADI_A1_attack_raw.csv')
    else:
        data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'WaDi', 'WADI.A2_19 Nov 2019')
        days14_path = os.path.join(data_dir, 'WADI_A2_14days_raw.csv')
        attack_path = os.path.join(data_dir, 'WADI_A2_attack_raw.csv')

    print(f"\n{'='*60}")
    print(f"Loading WaDi {scenario} 14days + attack raw")
    print(f"{'='*60}")

    features_14d, labels_14d, feat_cols = _load_raw_csv(days14_path)
    features_atk, labels_atk, _ = _load_raw_csv(attack_path)

    n_14d = len(features_14d)
    n_atk = len(features_atk)

    all_features = np.concatenate([features_14d, features_atk], axis=0)
    all_labels = np.concatenate([labels_14d, labels_atk], axis=0)
    n_total = n_14d + n_atk

    train_len = n_14d + n_atk // 2
    train_ratio = train_len / n_total
    anomaly_regions = _compute_anomaly_regions(all_labels)

    split_idx = int(n_total * train_ratio)
    train_labels = all_labels[:split_idx]
    test_labels = all_labels[split_idx:]

    # run_boundaries: 14days and attack are from different recording periods
    run_boundaries = [n_14d]

    data_info = {
        'n_14days': n_14d, 'n_attack': n_atk, 'n_total': n_total,
        'n_features': all_features.shape[1],
        'train_len': train_len, 'test_len': n_total - train_len,
        'train_ratio': train_ratio,
        'train_normal': int(np.sum(train_labels == 0)),
        'train_attack': int(np.sum(train_labels == 1)),
        'test_normal': int(np.sum(test_labels == 0)),
        'test_attack': int(np.sum(test_labels == 1)),
        'train_attack_ratio': float(np.mean(train_labels)),
        'test_attack_ratio': float(np.mean(test_labels)),
        'n_anomaly_regions_total': len(anomaly_regions),
        'run_boundaries': run_boundaries,
        'raw': True,
    }

    print(f"  14days: {n_14d:,}, Attack: {n_atk:,}")
    print(f"  Combined: {n_total:,} x {all_features.shape[1]} features")
    print(f"  Train: {train_len:,}, Test: {n_total - train_len:,}")
    print(f"  Test anomaly ratio: {data_info['test_attack_ratio']:.2%}")
    print(f"  Run boundaries: {run_boundaries} (14days/attack boundary)")

    return all_features, all_labels, anomaly_regions, feat_cols, train_ratio, data_info


# ============================================================
# SMAP / MSL (NASA Telemanom, Hundman et al., KDD 2018)
# ============================================================
#
# Source: Telemanom data.zip (NASA public domain). Original URL
#   https://s3-us-west-2.amazonaws.com/telemanom/data.zip
# is currently returning HTTP 403; the same archive is available via
# Wayback Machine snapshot (2022-10-16) at the same path. Channel labels
# from https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
# (synced into each spacecraft directory at integration time).
#
# Layout produced under dataset/{SMAP,MSL}/ (matches QuoVadis convention):
#   train/<chan_id>.npy   shape (T_train, F)  — all normal
#   test/<chan_id>.npy    shape (T_test,  F)  — contains anomalies per CSV
#   labeled_anomalies.csv (filtered to that spacecraft)
#
# Channel counts (after split):
#   SMAP: 54 unique channels (CSV has 55 rows because P-2 is duplicated
#         with two different anomaly_sequences annotations — we take the
#         UNION as the conservative label).
#   MSL:  27 channels.
#
# Feature dims (consistent across all channels of one spacecraft, differs
# between SMAP and MSL):
#   SMAP: 25 (1 telemetry + 24 binary command response)
#   MSL:  55 (1 telemetry + 54 binary command response)
#
# Split pattern (matches PSM / SWaT / WaDi 14days convention used elsewhere):
#   For each channel:
#     - train portion = orig_train (all normal) + front 50% of test
#     - test  portion = back 50% of test
#   The 50% cut is pushed outside any anomaly region by `margin` timestamps
#   (SMD `_find_safe_cut_point` is reused) so no anomaly straddles the boundary.
#   Channels are then concatenated in time, with `run_boundaries` marking both
#   the orig_train↔test_front junction inside each channel AND the inter-channel
#   joins. UnifiedLoader's `_apply_normalonly()` and the segment-aware SOTA
#   wrappers honour these boundaries — no synthetic adjacency is introduced.

def _load_smap_msl_combined(spacecraft: str, safe_cut_margin: int = 10):
    """Shared loader for SMAP/MSL with QuoVadis-style per-channel processing.

    Pattern (per channel):
      1. Load train.npy (all normal) and test.npy (may contain anomalies).
      2. Build per-timestep label from anomaly_sequences in labeled_anomalies.csv.
         (Inclusive ranges per Telemanom convention: indices [s, e] → labels[s:e+1].)
      3. Find a safe cut point at ~50% of the test stream (pushed outside any
         anomaly region by ±safe_cut_margin via SMD's `_find_safe_cut_point`).
      4. Split test into front (→ train portion) / back (→ test portion).
      5. Append to global accumulators.

    Channels are then time-concatenated. `run_boundaries` records every
    discontinuity so windowed training cannot accidentally connect
    non-adjacent timesteps.
    """
    import csv
    import ast

    data_dir = os.path.join(PROJECT_ROOT, 'dataset', spacecraft)
    labels_csv_path = os.path.join(data_dir, 'labeled_anomalies.csv')

    # Aggregate anomaly_sequences per channel (handle duplicate rows like SMAP P-2
    # by taking the UNION of all annotated anomaly intervals — conservative label).
    chan_to_seqs = {}
    with open(labels_csv_path) as f:
        for row in csv.DictReader(f):
            chan = row['chan_id']
            seqs = ast.literal_eval(row['anomaly_sequences'])  # list of [start, end]
            if chan in chan_to_seqs:
                chan_to_seqs[chan] = chan_to_seqs[chan] + seqs
            else:
                chan_to_seqs[chan] = list(seqs)

    channels = sorted(chan_to_seqs.keys())
    n_channels = len(channels)

    print(f"\n{'='*60}")
    print(f"Loading {spacecraft} (QuoVadis per-channel pattern)")
    print(f"{'='*60}")
    print(f"  Channels: {n_channels}")
    print(f"  Safe-cut margin: {safe_cut_margin} timestamps")

    train_chunks_sig, train_chunks_lab = [], []
    test_chunks_sig,  test_chunks_lab  = [], []
    channel_meta = []
    n_features = None

    for ch in channels:
        train_arr = np.load(os.path.join(data_dir, 'train', f'{ch}.npy')).astype(np.float32)
        test_arr  = np.load(os.path.join(data_dir, 'test',  f'{ch}.npy')).astype(np.float32)

        if n_features is None:
            n_features = train_arr.shape[1]
        elif train_arr.shape[1] != n_features or test_arr.shape[1] != n_features:
            raise ValueError(
                f"{spacecraft} channel {ch}: feature dim mismatch "
                f"(train={train_arr.shape[1]} test={test_arr.shape[1]} expected={n_features})"
            )

        # Per-timestep test labels from anomaly_sequences (inclusive ranges)
        test_labels = np.zeros(len(test_arr), dtype=np.int64)
        for s, e in chan_to_seqs[ch]:
            test_labels[max(0, s):min(len(test_arr), e + 1)] = 1
        regions_local = _get_anomaly_regions_local(test_labels)

        # Safe cut at ~50% of test, pushed outside anomaly regions ±margin
        target_cut = len(test_arr) // 2
        cut = _find_safe_cut_point(target_cut, test_labels, regions_local,
                                   margin=safe_cut_margin)
        cut = max(1, min(cut, len(test_arr) - 1))  # never empty front/back

        front_sig, back_sig = test_arr[:cut], test_arr[cut:]
        front_lab, back_lab = test_labels[:cut], test_labels[cut:]

        # train portion = orig_train + test_front_50%
        train_chunks_sig.append(train_arr)
        train_chunks_sig.append(front_sig)
        train_chunks_lab.append(np.zeros(len(train_arr), dtype=np.int64))
        train_chunks_lab.append(front_lab)

        # test portion = test_back_50%
        test_chunks_sig.append(back_sig)
        test_chunks_lab.append(back_lab)

        channel_meta.append({
            'channel': ch,
            'orig_train_len': int(len(train_arr)),
            'orig_test_len': int(len(test_arr)),
            'cut_target': int(target_cut),
            'cut_actual': int(cut),
            'cut_moved': bool(cut != target_cut),
            'train_portion_len': int(len(train_arr) + len(front_sig)),
            'test_portion_len': int(len(back_sig)),
            'n_anomaly_regions_total': int(len(regions_local)),
            'n_anomaly_regions_train_portion': int(len(_get_anomaly_regions_local(front_lab))),
            'n_anomaly_regions_test_portion': int(len(_get_anomaly_regions_local(back_lab))),
            'n_anomaly_points_train_portion': int(front_lab.sum()),
            'n_anomaly_points_test_portion': int(back_lab.sum()),
        })

    # Concatenate
    train_signals = np.concatenate(train_chunks_sig, axis=0)
    train_labels  = np.concatenate(train_chunks_lab, axis=0)
    test_signals  = np.concatenate(test_chunks_sig,  axis=0)
    test_labels_global = np.concatenate(test_chunks_lab,  axis=0)
    all_signals = np.concatenate([train_signals, test_signals], axis=0)
    all_labels  = np.concatenate([train_labels,  test_labels_global], axis=0)

    train_end = int(train_signals.shape[0])
    total_len = int(all_signals.shape[0])

    # ===== run_boundaries =====
    # In concat-array coordinate, mark every time-discontinuity so SlidingWindow
    # cannot bridge non-adjacent recordings. Three sources:
    #   (a) inside each channel: orig_train ↔ test_front junction
    #   (b) inter-channel boundary inside the train side
    #   (c) inter-channel boundary inside the test side
    run_boundaries = []
    cum = 0
    for meta in channel_meta:
        # (a) inside-channel orig_train↔test_front
        run_boundaries.append(cum + meta['orig_train_len'])
        # (b) end of this channel's train_portion (= start of next channel's train_portion)
        cum += meta['train_portion_len']
        run_boundaries.append(cum)
    # `cum` now equals train_end; that final entry is the train↔test boundary itself
    # (kept implicit — the UnifiedLoader splits at train_end anyway, but harmless to have).

    cum_test = train_end
    for meta in channel_meta[:-1]:
        cum_test += meta['test_portion_len']
        run_boundaries.append(cum_test)

    # Dedup + sort + filter to (0, total_len) open interval
    run_boundaries = sorted({int(b) for b in run_boundaries if 0 < int(b) < total_len})

    anomaly_regions = _compute_anomaly_regions(all_labels)

    train_ratio = train_end / total_len
    feature_names = [f'feat_{i}' for i in range(n_features)]

    data_info = {
        'dataset_type': spacecraft.lower(),
        'n_total': total_len,
        'n_features': int(n_features),
        'n_channels': int(n_channels),
        'channels': channels,
        'train_len': int(train_end),
        'test_len': int(total_len - train_end),
        'train_ratio': float(train_ratio),
        'train_normal': int((train_labels == 0).sum()),
        'train_attack': int((train_labels == 1).sum()),
        'test_normal':  int((test_labels_global == 0).sum()),
        'test_attack':  int((test_labels_global == 1).sum()),
        'train_attack_ratio': float(train_labels.mean()),
        'test_attack_ratio':  float(test_labels_global.mean()),
        'n_anomaly_regions_total': len(anomaly_regions),
        'run_boundaries': run_boundaries,
        'safe_cut_margin': int(safe_cut_margin),
        'channel_meta': channel_meta,
        'source': ('NASA Telemanom data.zip via Wayback Machine snapshot '
                   '2022-10-16 (https://web.archive.org/web/20221016205142/'
                   'http://s3-us-west-2.amazonaws.com/telemanom/data.zip)'),
    }

    print(f"\n  Train/Test split:")
    print(f"    Train: {train_end:,} ({len(channel_meta)}-channel orig_train + test_front_50%)")
    print(f"      anomaly: {data_info['train_attack']:,} ({data_info['train_attack_ratio']:.2%})")
    print(f"    Test:  {total_len - train_end:,} (test_back_50%)")
    print(f"      anomaly: {data_info['test_attack']:,} ({data_info['test_attack_ratio']:.2%})")
    print(f"    train_ratio: {train_ratio:.4f}")
    print(f"  Features: {n_features}")
    print(f"  Anomaly regions (global): {len(anomaly_regions)}")
    print(f"  Run boundaries: {len(run_boundaries)} discontinuities")
    n_moved = sum(1 for m in channel_meta if m['cut_moved'])
    print(f"  Safe-cut: {n_moved}/{len(channel_meta)} channels needed cut adjustment "
          f"(margin={safe_cut_margin})")

    return all_signals, all_labels, anomaly_regions, feature_names, train_ratio, data_info


def load_smap_combined():
    """Load SMAP (NASA Telemanom) — all 54 channels combined, QuoVadis pattern.

    Each channel split per PSM convention (orig_train + test_front_50% → train,
    test_back_50% → test), with the 50% cut pushed outside any anomaly region
    by ±10 timestamps. Channels are time-concatenated; every discontinuity is
    recorded in `data_info['run_boundaries']` for segment-aware windowing.
    """
    return _load_smap_msl_combined('SMAP', safe_cut_margin=10)


def load_msl_combined():
    """Load MSL (NASA Telemanom) — all 27 channels combined, QuoVadis pattern.

    Same processing as SMAP; only the spacecraft and feature dim (55, vs SMAP's 25)
    differ. P-2 is a SMAP channel and is absent from the MSL CSV/data, so no
    exclusion is needed at our end (QuoVadis explicitly excludes ``P-2_`` from
    MSL because their preprocessed mirror erroneously included it).
    """
    return _load_smap_msl_combined('MSL', safe_cut_margin=10)


# =============================================================================
# SMAP / MSL — Pattern B: per-channel (OmniAnomaly/Telemanom entity-level)
# =============================================================================
# Pattern A (above): all-channels concatenated into a single stream
#                    (Anomaly Transformer / TranAD style; channel boundaries
#                    via `run_boundaries`; cross-channel minmax fit).
# Pattern B (below): each channel processed independently, mirroring our
#                    existing per-machine SMD (`load_smd_simple`) and per-app
#                    Exathlon (`load_exathlon`) entries. UnifiedLoader fits
#                    the scaler on this single channel's train portion only,
#                    matching the entity-level convention used in OmniAnomaly
#                    (KDD 2019) and the original Telemanom paper (Hundman
#                    et al. KDD 2018).

# Channel inventories — discovered from `dataset/{SMAP,MSL}/labeled_anomalies.csv`
# at integration time (2026-05-26). Sorted lexicographically; SMAP P-2 has
# duplicate CSV rows that are UNION'd inside the loader.
SMAP_CHANNEL_NAMES = [
    'A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'A-6', 'A-7', 'A-8', 'A-9',
    'B-1',
    'D-1', 'D-11', 'D-12', 'D-13', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7', 'D-8', 'D-9',
    'E-1', 'E-10', 'E-11', 'E-12', 'E-13', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7', 'E-8', 'E-9',
    'F-1', 'F-2', 'F-3',
    'G-1', 'G-2', 'G-3', 'G-4', 'G-6', 'G-7',
    'P-1', 'P-2', 'P-3', 'P-4', 'P-7',
    'R-1',
    'S-1',
    'T-1', 'T-2', 'T-3',
]

MSL_CHANNEL_NAMES = [
    'C-1', 'C-2',
    'D-14', 'D-15', 'D-16',
    'F-4', 'F-5', 'F-7', 'F-8',
    'M-1', 'M-2', 'M-3', 'M-4', 'M-5', 'M-6', 'M-7',
    'P-10', 'P-11', 'P-14', 'P-15',
    'S-2',
    'T-12', 'T-13', 'T-4', 'T-5', 'T-8', 'T-9',
]


def _load_smap_msl_simple_single(spacecraft: str, channel: str, safe_cut_margin: int = 10):
    """Single-channel loader shared by SMAP/MSL Pattern B.

    Mirrors `load_smd_simple` (SMD per-machine entry) structure:
      train = orig_train.npy (all normal) + test_front_50% (safe-cut margin)
      test  = test_back_50%
      run_boundaries = [len(orig_train)]   # one intra-channel junction

    The UnifiedLoader fits min-max / z-score on this single channel's train
    portion only → per-channel scaler (entity-level), matching SMD/Exathlon
    convention and the Telemanom/OmniAnomaly per-entity tradition.

    Args:
        spacecraft: 'SMAP' or 'MSL'
        channel: Channel ID (e.g. 'A-1' for SMAP, 'C-1' for MSL)
        safe_cut_margin: Timestamps to keep between the 50% cut and any
                         anomaly region (uses SMD's `_find_safe_cut_point`).

    Returns:
        (signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info)
        — same 6-tuple contract as load_smd_simple.
    """
    import csv
    import ast

    data_dir = os.path.join(PROJECT_ROOT, 'dataset', spacecraft)
    labels_csv_path = os.path.join(data_dir, 'labeled_anomalies.csv')

    # Aggregate anomaly_sequences for this channel only (UNION across duplicate
    # CSV rows — relevant for SMAP P-2 which is annotated twice).
    chan_seqs = []
    found = False
    with open(labels_csv_path) as f:
        for row in csv.DictReader(f):
            if row['chan_id'] == channel:
                chan_seqs.extend(ast.literal_eval(row['anomaly_sequences']))
                found = True
    if not found:
        raise ValueError(
            f"{spacecraft} channel '{channel}' not found in {labels_csv_path}. "
            f"Use one of: SMAP_CHANNEL_NAMES (54) or MSL_CHANNEL_NAMES (27)."
        )

    train_arr = np.load(os.path.join(data_dir, 'train', f'{channel}.npy')).astype(np.float32)
    test_arr  = np.load(os.path.join(data_dir, 'test',  f'{channel}.npy')).astype(np.float32)
    if train_arr.shape[1] != test_arr.shape[1]:
        raise ValueError(
            f"{spacecraft} {channel}: feature dim mismatch "
            f"(train={train_arr.shape[1]} test={test_arr.shape[1]})"
        )
    n_features = int(train_arr.shape[1])

    print(f"\n{'='*60}")
    print(f"Loading {spacecraft} simple split (Pattern B): {channel}")
    print(f"{'='*60}")
    print(f"  Train file: {len(train_arr):,} samples, {n_features} features (all normal)")
    print(f"  Test  file: {len(test_arr):,} samples, {n_features} features")

    # Per-timestep test labels from anomaly_sequences (inclusive ranges)
    test_labels = np.zeros(len(test_arr), dtype=np.int64)
    for s, e in chan_seqs:
        test_labels[max(0, s):min(len(test_arr), e + 1)] = 1
    regions_local = _get_anomaly_regions_local(test_labels)

    # Safe cut at ~50% of test, pushed outside anomaly regions ±margin
    target_cut = len(test_arr) // 2
    cut = _find_safe_cut_point(target_cut, test_labels, regions_local,
                               margin=safe_cut_margin)
    cut = max(1, min(cut, len(test_arr) - 1))  # never empty front/back

    front_sig, back_sig = test_arr[:cut], test_arr[cut:]
    front_lab, back_lab = test_labels[:cut], test_labels[cut:]

    # Concatenate: [orig_train | test_front_50% | test_back_50%]
    all_signals = np.concatenate([train_arr, front_sig, back_sig], axis=0)
    all_labels = np.concatenate(
        [np.zeros(len(train_arr), dtype=np.int64), front_lab, back_lab], axis=0
    )

    total_train = int(len(train_arr) + len(front_sig))
    total_test = int(len(back_sig))
    n_total = total_train + total_test
    train_ratio = total_train / n_total

    anomaly_regions = _compute_anomaly_regions(all_labels)

    # One intra-channel junction: orig_train ↔ test_front
    run_boundaries = [int(len(train_arr))]

    feature_names = [f'feat_{i}' for i in range(n_features)]

    train_anom = int(np.sum(all_labels[:total_train]))
    test_anom = int(np.sum(all_labels[total_train:]))

    data_info = {
        'dataset_type': f'{spacecraft.lower()}_simple',
        'spacecraft': spacecraft,
        'channel': channel,
        'n_total': n_total,
        'n_features': n_features,
        'train_len': total_train,
        'test_len': total_test,
        'train_ratio': float(train_ratio),
        'train_file_len': int(len(train_arr)),
        'test_file_len': int(len(test_arr)),
        'test_split_point': int(cut),
        'safe_cut_target': int(target_cut),
        'safe_cut_actual': int(cut),
        'safe_cut_moved': bool(cut != target_cut),
        'safe_cut_margin': int(safe_cut_margin),
        'train_attack_ratio': float(np.mean(all_labels[:total_train])) if total_train > 0 else 0.0,
        'test_attack_ratio': float(np.mean(all_labels[total_train:])) if total_test > 0 else 0.0,
        'n_anomaly_regions_total': len(anomaly_regions),
        'run_boundaries': run_boundaries,
        'source': ('NASA Telemanom data.zip via Wayback Machine snapshot '
                   '2022-10-16 (https://web.archive.org/web/20221016205142/'
                   'http://s3-us-west-2.amazonaws.com/telemanom/data.zip)'),
    }

    print(f"\n  Train/Test split:")
    print(f"    Train: {total_train:,} = {len(train_arr):,}(orig train) + {len(front_sig):,}(test front 50%)")
    print(f"      anomaly: {train_anom:,} ({data_info['train_attack_ratio']:.2%})")
    print(f"    Test:  {total_test:,} (test back 50%)")
    print(f"      anomaly: {test_anom:,} ({data_info['test_attack_ratio']:.2%})")
    print(f"    train_ratio: {train_ratio:.4f}")
    print(f"  Anomaly regions: {len(anomaly_regions)}")
    print(f"  Run boundaries: {run_boundaries} (orig_train / test_front boundary)")
    print(f"  Safe-cut: {'moved' if data_info['safe_cut_moved'] else 'unchanged'} "
          f"(target={target_cut}, actual={cut}, margin={safe_cut_margin})")

    return all_signals, all_labels, anomaly_regions, feature_names, float(train_ratio), data_info


def load_smap_simple(channel: str):
    """Load a single SMAP channel (Pattern B, per-entity).

    SMD-style 50/50 split with safe-cut margin. UnifiedLoader fits the scaler
    on this channel's train portion only (per-channel scaler), matching the
    Telemanom (Hundman et al. KDD 2018) and OmniAnomaly (KDD 2019) per-entity
    convention. See also `load_smap_combined` for Pattern A (all-channels
    concat).

    Args:
        channel: One of `SMAP_CHANNEL_NAMES` (e.g. 'A-1', 'P-2', 'T-3').
    """
    return _load_smap_msl_simple_single('SMAP', channel, safe_cut_margin=10)


def load_msl_simple(channel: str):
    """Load a single MSL channel (Pattern B, per-entity).

    Same processing as `load_smap_simple`. Feature dim differs (MSL=55,
    SMAP=25). See `load_msl_combined` for Pattern A.

    Args:
        channel: One of `MSL_CHANNEL_NAMES` (e.g. 'C-1', 'M-1', 'T-12').
    """
    return _load_smap_msl_simple_single('MSL', channel, safe_cut_margin=10)


# Dataset Loader Registry
DATASET_LOADERS = {
    # SWaT (raw CSV, all features)
    'SWaT_A1A2': lambda: load_swat_a1a2_raw(swap=False),
    'SWaT_A1A2_swap': lambda: load_swat_a1a2_raw(swap=True),
    'SWaT_A4A5': load_swat_a4a5_raw,
    'SWaT_A6': load_swat_a6_raw,
    # Legacy SWaT preprocessed
    'swat_A1A2': load_swat_combined,
    'swat_A1A2_swap': load_swat_combined_swap,
    # WaDi (14days normal + attack raw)
    'WaDi_14days_A1': lambda: load_wadi_14days_raw('A1'),
    'WaDi_14days_A2': lambda: load_wadi_14days_raw('A2'),
    # Simulation
    'simulation': load_simulation,
    'simulation_complex': load_simulation_complex,
    # TEP dataset loaders
    'tep': lambda: load_tep(),
    # SMD dataset loaders
    'smd': lambda: load_smd(),
    # PSM dataset loader (Pooled Server Metrics, eBay)
    'PSM': load_psm,
}
# Add per-fault TEP loaders dynamically (tep_fault1 through tep_fault20)
for _fn in range(1, 21):
    DATASET_LOADERS[f'tep_fault{_fn}'] = (lambda fn=_fn: load_tep(fault_types=[fn]))
del _fn  # Clean up loop variable
# Add per-machine SMD loaders dynamically (smd_machine-1-1 through smd_machine-3-11)
for _mn in SMD_MACHINE_NAMES:
    DATASET_LOADERS[f'smd_{_mn}'] = (lambda mn=_mn: load_smd(machines=[mn]))
del _mn  # Clean up loop variable
# Add per-machine SMD K=6 block split loaders (parity 0 and 1) — legacy
for _mn in SMD_MACHINE_NAMES:
    DATASET_LOADERS[f'smd_k6_{_mn}'] = (lambda mn=_mn: load_smd_block_split(machine=mn, k_blocks=6, parity=0))
    DATASET_LOADERS[f'smd_k6_{_mn}_swap'] = (lambda mn=_mn: load_smd_block_split(machine=mn, k_blocks=6, parity=1))
del _mn  # Clean up loop variable
# Add per-machine SMD simple split loaders (train + front 50% test / back 50% test)
for _mn in SMD_MACHINE_NAMES:
    DATASET_LOADERS[f'smd_simple_{_mn}'] = (lambda mn=_mn: load_smd_simple(machine=mn))
del _mn  # Clean up loop variable
# Add per-app Exathlon loaders (apps 1, 2, 4, 5, 6, 9 — TimeSeAD 6-app convention)
for _ap in EXATHLON_APP_IDS:
    DATASET_LOADERS[f'exathlon_app{_ap}'] = (lambda ap=_ap: load_exathlon(app=ap))
del _ap  # Clean up loop variable


def get_dataset_loader(dataset_type: str):
    """Get dataset loader function by name.
    
    Args:
        dataset_type: One of the keys in DATASET_LOADERS
        
    Returns:
        Loader function that returns (signals, labels, regions, features, train_ratio, data_info)
    """
    if dataset_type not in DATASET_LOADERS:
        available = ', '.join(DATASET_LOADERS.keys())
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {available}")
    
    return DATASET_LOADERS[dataset_type]
