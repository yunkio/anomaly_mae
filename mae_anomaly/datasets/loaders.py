"""Dataset loaders for SWaT, WaDi, and Simulation datasets."""

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
    a1_path = os.path.join(data_dir, 'SWaT_A1_normal_preprocessed.csv')
    a2_path = os.path.join(data_dir, 'SWaT_A2_attack_preprocessed.csv')

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

    # Re-normalize combined data (min-max)
    print("  Re-normalizing combined data (min-max)...")
    mins = np.min(all_features_raw, axis=0, keepdims=True)
    maxs = np.max(all_features_raw, axis=0, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    all_features = ((all_features_raw - mins) / ranges).astype(np.float32)

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
    }

    print(f"  Train anomaly ratio: {data_info['train_attack_ratio']:.2%}")
    print(f"  Test anomaly ratio: {data_info['test_attack_ratio']:.2%}")
    print(f"  Features: {data_info['n_features']}")
    print(f"  Anomaly regions: {len(anomaly_regions)}")

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
    a1_path = os.path.join(data_dir, 'SWaT_A1_normal_preprocessed.csv')
    a2_path = os.path.join(data_dir, 'SWaT_A2_attack_preprocessed.csv')

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

    # Re-normalize combined data (min-max)
    print("  Re-normalizing combined data (min-max)...")
    mins = np.min(all_features_raw, axis=0, keepdims=True)
    maxs = np.max(all_features_raw, axis=0, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    all_features = ((all_features_raw - mins) / ranges).astype(np.float32)

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
    }

    print(f"  Train anomaly ratio: {data_info['train_attack_ratio']:.2%}")
    print(f"  Test anomaly ratio: {data_info['test_attack_ratio']:.2%}")
    print(f"  Features: {data_info['n_features']}")
    print(f"  Anomaly regions: {len(anomaly_regions)}")
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

    # Note: preprocessed attack data is already normalized to [0, 1]
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

    # Min-max normalization on combined data
    print("  Applying min-max normalization on combined data...")
    mins = np.min(all_features_raw, axis=0, keepdims=True)
    maxs = np.max(all_features_raw, axis=0, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # Avoid division by zero
    all_features = (all_features_raw - mins) / ranges

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
    }

    print(f"\n  Train anomaly ratio: {data_info['train_attack_ratio']:.2%}")
    print(f"  Test anomaly ratio: {data_info['test_attack_ratio']:.2%}")
    print(f"  Features: {data_info['n_features']}")

    return all_features, all_labels, anomaly_regions, matching_features, train_ratio, data_info


# =============================================================================
# Background CPU Worker (same as run_wadi_ablation.py)
# =============================================================================


def load_wadi_a2(preprocessed_path: str):
    """Load preprocessed WaDi A2 dataset.

    Returns:
        signals: (N, num_features) float32 array, normalized to [0,1]
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
        random_seed=random_seed,
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
        random_seed=random_seed,
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


# Dataset Loader Registry
_WADI_A1_PATH = str(PROJECT_ROOT / 'dataset' / 'WaDi' / 'WADI.A1_9 Oct 2017' / 'WADI_attackdata_preprocessed.csv')
_WADI_A2_PATH = str(PROJECT_ROOT / 'dataset' / 'WaDi' / 'WADI.A2_19 Nov 2019' / 'WADI_attackdataLABLE_preprocessed.csv')

DATASET_LOADERS = {
    'swat_A1A2': load_swat_combined,
    'swat_A1A2_swap': load_swat_combined_swap,
    'wadi_A1': lambda: load_wadi_attack_5050(_WADI_A1_PATH, swap=False),
    'wadi_A1_swap': lambda: load_wadi_attack_5050(_WADI_A1_PATH, swap=True),
    'wadi_A2': lambda: load_wadi_attack_5050(_WADI_A2_PATH, swap=False),
    'wadi_A2_swap': lambda: load_wadi_attack_5050(_WADI_A2_PATH, swap=True),
    'wadi_14days_A1': lambda: load_wadi_14days_combined('A1'),
    'wadi_14days_A2': lambda: load_wadi_14days_combined('A2'),
    'simulation': load_simulation,
    'simulation_complex': load_simulation_complex,
}


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
