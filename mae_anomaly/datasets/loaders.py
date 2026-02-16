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

    # ---- Min-max normalization ----
    print("  Applying min-max normalization...")
    mins = np.min(all_signals_raw, axis=0, keepdims=True)
    maxs = np.max(all_signals_raw, axis=0, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    all_signals = ((all_signals_raw - mins) / ranges).astype(np.float32)

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

    # ---- Min-max normalization ----
    print("  Applying min-max normalization...")
    mins = np.min(all_signals_raw, axis=0, keepdims=True)
    maxs = np.max(all_signals_raw, axis=0, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    all_signals = ((all_signals_raw - mins) / ranges).astype(np.float32)

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
    # TEP dataset loaders
    'tep': lambda: load_tep(),
    # SMD dataset loaders
    'smd': lambda: load_smd(),
}
# Add per-fault TEP loaders dynamically (tep_fault1 through tep_fault20)
for _fn in range(1, 21):
    DATASET_LOADERS[f'tep_fault{_fn}'] = (lambda fn=_fn: load_tep(fault_types=[fn]))
del _fn  # Clean up loop variable
# Add per-machine SMD loaders dynamically (smd_machine-1-1 through smd_machine-3-11)
for _mn in SMD_MACHINE_NAMES:
    DATASET_LOADERS[f'smd_{_mn}'] = (lambda mn=_mn: load_smd(machines=[mn]))
del _mn  # Clean up loop variable


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
