#!/usr/bin/env python
"""Convert raw SWaT/WaDi datasets to pipeline-ready CSV files.

Reads original (raw) xlsx/csv files and produces clean CSV files with:
- All original sensor features preserved (no feature removal)
- No normalization applied (pipeline handles per-feature z-score)
- String columns encoded (Active=1, Inactive=0)
- NaN values handled (forward-fill + backward-fill)
- Labels attached from appropriate sources

Output files are saved alongside the original data with '_raw.csv' suffix.

Usage:
    python scripts/prepare_raw_datasets.py                    # Convert all
    python scripts/prepare_raw_datasets.py --datasets swat_a1a2 swat_a4a5
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def encode_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode string columns: Active=1, Inactive=0.

    Any object-dtype column with Active/Inactive values is converted.
    Other string values become NaN (handled by subsequent ffill+bfill).
    """
    for col in df.columns:
        if df[col].dtype == object:
            mapping = {'Active': 1, 'Inactive': 0}
            df[col] = df[col].map(mapping)
    return df


def clean_column_name(col: str) -> str:
    """Clean raw column names: remove Windows path prefix, strip whitespace."""
    if '\\' in col:
        parts = col.split('\\')
        return parts[-1].strip()
    return col.strip()


def handle_nan(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Handle NaN values: drop all-NaN columns, then forward-fill + backward-fill."""
    # Drop columns that are entirely NaN (no usable data)
    all_nan_cols = list(df.columns[df.isna().all()])
    if all_nan_cols:
        print(f"  Dropping {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")
        df = df.drop(columns=all_nan_cols)

    nan_count = int(df.isna().sum().sum())
    if nan_count > 0:
        print(f"  Handling {nan_count:,} NaN values (ffill + bfill)")
        df = df.ffill().bfill()
        remaining = int(df.isna().sum().sum())
        if remaining > 0:
            print(f"  WARNING: {remaining:,} NaN values still remaining!")
    else:
        print(f"  No NaN values found")
    return df


def validate_output(df: pd.DataFrame, expected_rows: int, dataset_name: str):
    """Validate output CSV structure."""
    assert 'label' in df.columns, f"{dataset_name}: missing 'label' column"
    feature_cols = [c for c in df.columns if c != 'label']
    n_features = len(feature_cols)

    # Check row count
    if expected_rows is not None:
        assert len(df) == expected_rows, \
            f"{dataset_name}: row count mismatch: {len(df)} != {expected_rows}"

    # Check all features are numeric
    for col in feature_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), \
            f"{dataset_name}: non-numeric column '{col}', dtype={df[col].dtype}"

    # Check no remaining NaN in features
    nan_count = int(df[feature_cols].isna().sum().sum())
    assert nan_count == 0, f"{dataset_name}: {nan_count} NaN values remaining"

    # Label validation
    label_values = set(df['label'].unique())
    assert label_values <= {0, 1}, f"{dataset_name}: unexpected label values: {label_values}"

    n_normal = int((df['label'] == 0).sum())
    n_attack = int((df['label'] == 1).sum())
    print(f"  Validation passed: {len(df):,} rows x {n_features} features")
    print(f"    Label: normal={n_normal:,}, attack={n_attack:,}")


# =============================================================================
# SWaT A1 + A2
# =============================================================================

def prepare_swat_a1a2():
    """Convert SWaT A1 (normal) and A2 (attack) xlsx to raw CSV.

    Raw xlsx structure (both files):
      Row 0: Section headers (P1, P2, ...) or empty
      Row 1: Feature names (Timestamp, FIT101, ..., Normal/Attack)
      Row 2+: Data

    Output: SWaT_A1_normal_raw.csv, SWaT_A2_attack_raw.csv
    Each with 51 sensor features + 'label' column.
    """
    data_dir = PROJECT_ROOT / 'dataset' / 'SWaT' / 'SWaT.A1 & A2_Dec 2015'

    for name, xlsx_name, expected_rows in [
        ('SWaT A1 Normal', 'SWaT_Dataset_Normal_v1.xlsx', 495000),
        ('SWaT A2 Attack', 'SWaT_Dataset_Attack_v0.xlsx', 449919),
    ]:
        print(f"\n{'='*60}")
        print(f"  {name} (xlsx -> raw csv)")
        print(f"{'='*60}")

        xlsx_path = data_dir / xlsx_name
        print(f"  Reading: {xlsx_path.name}")
        # header=1: skip row 0 (section headers), use row 1 as column names
        df = pd.read_excel(xlsx_path, header=1)
        print(f"  Raw shape: {df.shape}")

        # Identify columns
        timestamp_col = df.columns[0]  # 'Timestamp'
        label_col_name = 'Normal/Attack'
        assert label_col_name in df.columns, \
            f"Expected '{label_col_name}' column, got: {list(df.columns[-3:])}"

        # Convert label: Normal=0, Attack/A ttack=1
        label_map = df[label_col_name].map({
            'Normal': 0, 'Attack': 1, 'A ttack': 1,  # "A ttack" is a typo in original xlsx
        })
        unmapped = label_map.isna().sum()
        if unmapped > 0:
            print(f"  WARNING: {unmapped} unmapped label values")
            raw_vals = df[label_col_name][label_map.isna()].unique()
            print(f"    Unmapped values: {raw_vals[:5]}")
        df['label'] = label_map.fillna(0).astype(int)

        # Drop timestamp and original label column
        df = df.drop(columns=[timestamp_col, label_col_name])
        print(f"  Dropped: '{timestamp_col}', '{label_col_name}'")
        print(f"  Features: {df.shape[1] - 1}")

        # Encode string columns if any (shouldn't be needed for A1/A2)
        string_cols = [c for c in df.columns if c != 'label' and df[c].dtype == object]
        if string_cols:
            print(f"  Encoding string columns: {string_cols}")
            df = encode_string_columns(df)

        # Handle NaN (may drop all-NaN columns)
        labels_backup = df['label'].copy()
        df = handle_nan(df.drop(columns=['label']), name)
        df['label'] = labels_backup

        # Validate
        validate_output(df, expected_rows=expected_rows, dataset_name=name)

        # Save
        suffix = 'A1_normal' if 'Normal' in name else 'A2_attack'
        out_path = data_dir / f'SWaT_{suffix}_raw.csv'
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path.name}")


# =============================================================================
# SWaT A4 + A5
# =============================================================================

def prepare_swat_a4a5():
    """Convert SWaT A4+A5 xlsx to raw CSV with labels from preprocessed.

    Raw xlsx structure:
      Row 0: Section headers (P1, P2, ...) - skip
      Row 1: Feature names (GMT +0, FIT 101, ...) - use as header
      Row 2: Value types (timestamp, value, ...) - skip
      Row 3+: Data

    String columns (Active/Inactive): LS 201, LS 202, LSL 203, LSLL 203,
      LS 401, LSH 601, LSH 602, LSH 603, LSL 601, LSL 602, LSL 603 (11 cols)

    Labels from preprocessed CSV (14,996 rows exact match).
    Output: SWaT_A4A5_raw.csv with 77 features + 'label'.
    """
    data_dir = PROJECT_ROOT / 'dataset' / 'SWaT' / 'SWaT.A4 & A5_Jul 2019'

    print(f"\n{'='*60}")
    print(f"  SWaT A4+A5 (xlsx -> raw csv)")
    print(f"{'='*60}")

    xlsx_path = data_dir / 'SWaT_dataset_Jul 19 v2.xlsx'
    print(f"  Reading: {xlsx_path.name}")
    # skiprows=[0, 2]: skip section header row and value-type row
    df = pd.read_excel(xlsx_path, skiprows=[0, 2])
    print(f"  Raw shape: {df.shape}")

    # Drop timestamp column (first: 'GMT +0')
    timestamp_col = df.columns[0]
    print(f"  Dropping timestamp: '{timestamp_col}'")
    df = df.drop(columns=[timestamp_col])

    # Encode string columns (Active/Inactive)
    string_cols = [c for c in df.columns if df[c].dtype == object]
    print(f"  String columns ({len(string_cols)}): {string_cols}")
    df = encode_string_columns(df)

    # Get labels from preprocessed
    preprocessed_path = data_dir / 'SWaT_A4A5_preprocessed.csv'
    print(f"  Loading labels from: {preprocessed_path.name}")
    labels = pd.read_csv(preprocessed_path, usecols=['label'])['label'].values

    assert len(df) == len(labels), \
        f"Row count mismatch: xlsx={len(df)}, preprocessed={len(labels)}"
    print(f"  Row count match: {len(df):,}")

    df['label'] = labels

    # Handle NaN (may drop all-NaN columns)
    labels_backup = df['label'].copy()
    df = handle_nan(df.drop(columns=['label']), 'SWaT A4A5')
    df['label'] = labels_backup

    # Validate
    validate_output(df, expected_rows=14996, dataset_name='SWaT A4A5')

    out_path = data_dir / 'SWaT_A4A5_raw.csv'
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name} ({len(df.columns)-1} features)")


# =============================================================================
# SWaT A6
# =============================================================================

def prepare_swat_a6():
    """Convert SWaT A6 xlsx to raw CSV with labels from preprocessed.

    Raw xlsx structure:
      Rows 0-8: Metadata notes (initial plant state)
      Row 9: Column names (t_stamp, P1_STATE, LIT101.Pv, ...)
      Row 10+: Data

    String columns: alarm columns (*.Alarm) with Active/Inactive values.
    Labels from preprocessed CSV (13,201 rows exact match).
    Output: SWaT_A6_raw.csv with 81 features + 'label'.
    """
    data_dir = PROJECT_ROOT / 'dataset' / 'SWaT' / 'SWaT.A6_Dec 2019'

    print(f"\n{'='*60}")
    print(f"  SWaT A6 (xlsx -> raw csv)")
    print(f"{'='*60}")

    xlsx_path = data_dir / 'Dec2019.xlsx'
    print(f"  Reading: {xlsx_path.name}")
    # header=9: rows 0-8 are metadata, row 9 has column names
    df = pd.read_excel(xlsx_path, header=9)
    print(f"  Raw shape: {df.shape}")

    # Drop timestamp column (first: 't_stamp')
    timestamp_col = df.columns[0]
    print(f"  Dropping timestamp: '{timestamp_col}'")
    df = df.drop(columns=[timestamp_col])

    # Encode string columns (alarm columns: Active/Inactive)
    string_cols = [c for c in df.columns if df[c].dtype == object]
    print(f"  String columns ({len(string_cols)}): {string_cols}")
    df = encode_string_columns(df)

    # Get labels from preprocessed
    preprocessed_path = data_dir / 'SWaT_A6_preprocessed.csv'
    print(f"  Loading labels from: {preprocessed_path.name}")
    labels = pd.read_csv(preprocessed_path, usecols=['label'])['label'].values

    assert len(df) == len(labels), \
        f"Row count mismatch: xlsx={len(df)}, preprocessed={len(labels)}"
    print(f"  Row count match: {len(df):,}")

    df['label'] = labels

    # Handle NaN (may drop all-NaN columns)
    labels_backup = df['label'].copy()
    df = handle_nan(df.drop(columns=['label']), 'SWaT A6')
    df['label'] = labels_backup

    # Validate
    validate_output(df, expected_rows=13201, dataset_name='SWaT A6')

    out_path = data_dir / 'SWaT_A6_raw.csv'
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name} ({len(df.columns)-1} features)")


# =============================================================================
# WaDi A1
# =============================================================================

def prepare_wadi_a1():
    """Convert WaDi A1 attack + 14days CSV to raw CSV.

    Attack CSV: 130 columns = 3 meta (Row, Date, Time) + 127 sensor features.
      Windows path column names cleaned. No label column.
      Labels from preprocessed CSV (172,801 rows exact match).

    14days CSV: 130 columns = 3 meta + 127 sensor features (skiprows=4).
      All normal (label=0).

    Output: WADI_A1_attack_raw.csv (127 features + label),
            WADI_A1_14days_raw.csv (127 features + label=0).
    """
    data_dir = PROJECT_ROOT / 'dataset' / 'WaDi' / 'WADI.A1_9 Oct 2017'

    # --- Attack data ---
    print(f"\n{'='*60}")
    print(f"  WaDi A1 Attack (csv -> raw csv)")
    print(f"{'='*60}")

    attack_path = data_dir / 'WADI_attackdata.csv'
    print(f"  Reading: {attack_path.name}")
    df_attack = pd.read_csv(attack_path)
    print(f"  Raw shape: {df_attack.shape}")

    # Clean column names (remove Windows path prefix)
    df_attack.columns = [clean_column_name(c) for c in df_attack.columns]

    # Drop meta columns
    meta_cols = [c for c in df_attack.columns if c in ('Row', 'Date', 'Time')]
    print(f"  Dropping meta columns: {meta_cols}")
    df_attack = df_attack.drop(columns=meta_cols)

    # Get labels from preprocessed
    preprocessed_path = data_dir / 'WADI_attackdata_preprocessed.csv'
    print(f"  Loading labels from: {preprocessed_path.name}")
    labels = pd.read_csv(preprocessed_path, usecols=['label'])['label'].values

    assert len(df_attack) == len(labels), \
        f"Row count mismatch: raw={len(df_attack)}, preprocessed={len(labels)}"
    print(f"  Row count match: {len(df_attack):,}")

    # Handle NaN (may drop all-NaN columns) — before adding label
    df_attack = handle_nan(df_attack, 'WaDi A1 attack')
    df_attack['label'] = labels

    # Validate
    validate_output(df_attack, expected_rows=172801, dataset_name='WaDi A1 attack')

    attack_feature_cols = [c for c in df_attack.columns if c != 'label']
    out_path = data_dir / 'WADI_A1_attack_raw.csv'
    df_attack.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name} ({len(attack_feature_cols)} features)")

    # --- 14days data ---
    print(f"\n{'='*60}")
    print(f"  WaDi A1 14days (csv -> raw csv)")
    print(f"{'='*60}")

    days14_path = data_dir / 'WADI_14days.csv'
    print(f"  Reading: {days14_path.name}")
    df_14days = pd.read_csv(days14_path, skiprows=4)
    print(f"  Raw shape: {df_14days.shape}")

    # Clean column names
    df_14days.columns = [clean_column_name(c) for c in df_14days.columns]

    # Drop meta columns
    meta_cols = [c for c in df_14days.columns if c in ('Row', 'Date', 'Time')]
    print(f"  Dropping meta columns: {meta_cols}")
    df_14days = df_14days.drop(columns=meta_cols)

    # Verify columns match attack data
    days14_features = list(df_14days.columns)
    if set(attack_feature_cols) == set(days14_features):
        print(f"  Columns match attack data ({len(attack_feature_cols)} features)")
    else:
        missing = set(attack_feature_cols) - set(days14_features)
        extra = set(days14_features) - set(attack_feature_cols)
        if missing:
            print(f"  WARNING: Missing in 14days: {missing}")
        if extra:
            print(f"  WARNING: Extra in 14days: {extra}")

    # Keep only columns that exist in attack data (after all-NaN removal)
    df_14days = df_14days[[c for c in attack_feature_cols if c in df_14days.columns]]

    # Handle NaN (may drop additional all-NaN columns)
    df_14days = handle_nan(df_14days, 'WaDi A1 14days')

    # All normal
    df_14days['label'] = 0

    # Validate
    validate_output(df_14days, expected_rows=None, dataset_name='WaDi A1 14days')

    out_path = data_dir / 'WADI_A1_14days_raw.csv'
    df_14days.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name} ({len(df_14days):,} rows)")


# =============================================================================
# WaDi A2
# =============================================================================

def prepare_wadi_a2():
    """Convert WaDi A2 attack + 14days CSV to raw CSV.

    Attack CSV (header=1): 131 columns = 3 meta + 127 features + 1 label.
      Row 0 has column indices (0,1,2,...), row 1 has actual names.
      Label: "Attack LABLE (1:No Attack, -1:Attack)" → 1=normal(0), -1=attack(1).

    14days CSV: 130 columns = 3 meta + 127 features. All normal.

    Output: WADI_A2_attack_raw.csv (127 features + label),
            WADI_A2_14days_raw.csv (127 features + label=0).
    """
    data_dir = PROJECT_ROOT / 'dataset' / 'WaDi' / 'WADI.A2_19 Nov 2019'

    # --- Attack data ---
    print(f"\n{'='*60}")
    print(f"  WaDi A2 Attack (csv -> raw csv)")
    print(f"{'='*60}")

    attack_path = data_dir / 'WADI_attackdataLABLE.csv'
    print(f"  Reading: {attack_path.name}")
    # header=1: row 0 has numeric indices, row 1 has actual column names
    df_attack = pd.read_csv(attack_path, header=1)
    print(f"  Raw shape: {df_attack.shape}")

    # Strip column name whitespace
    df_attack.columns = [c.strip() for c in df_attack.columns]

    # Extract label column
    label_col_name = 'Attack LABLE (1:No Attack, -1:Attack)'
    assert label_col_name in df_attack.columns, \
        f"Label column not found. Last 3 cols: {list(df_attack.columns[-3:])}"

    # Convert label: 1 (no attack) → 0, -1 (attack) → 1
    raw_labels = pd.to_numeric(df_attack[label_col_name], errors='coerce').values
    labels = np.where(raw_labels == -1, 1, 0).astype(int)
    n_attack = int(np.sum(labels == 1))
    n_normal = int(np.sum(labels == 0))
    print(f"  Label converted: normal={n_normal:,}, attack={n_attack:,}")

    # Cross-validate with preprocessed
    preprocessed_path = data_dir / 'WADI_attackdataLABLE_preprocessed.csv'
    pre_labels = pd.read_csv(preprocessed_path, usecols=['label'])['label'].values
    if len(pre_labels) == len(labels):
        label_match = np.sum(labels == pre_labels)
        print(f"  Label cross-validation: {label_match:,}/{len(labels):,} match "
              f"({label_match/len(labels)*100:.2f}%)")
    else:
        print(f"  WARNING: Row count differs from preprocessed "
              f"(raw={len(labels)}, pre={len(pre_labels)})")

    # Drop meta columns and label column
    meta_cols = [c for c in df_attack.columns if c in ('Row', 'Date', 'Time')]
    drop_cols = meta_cols + [label_col_name]
    print(f"  Dropping: {[c[:30] for c in drop_cols]}")
    df_attack = df_attack.drop(columns=drop_cols)

    # Handle NaN (may drop all-NaN columns) — before adding label
    df_attack = handle_nan(df_attack, 'WaDi A2 attack')
    df_attack['label'] = labels

    # Validate
    validate_output(df_attack, expected_rows=None, dataset_name='WaDi A2 attack')

    attack_feature_cols = [c for c in df_attack.columns if c != 'label']
    out_path = data_dir / 'WADI_A2_attack_raw.csv'
    df_attack.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name} ({len(attack_feature_cols)} features)")

    # --- 14days data ---
    print(f"\n{'='*60}")
    print(f"  WaDi A2 14days (csv -> raw csv)")
    print(f"{'='*60}")

    days14_path = data_dir / 'WADI_14days_new.csv'
    print(f"  Reading: {days14_path.name}")
    df_14days = pd.read_csv(days14_path)
    print(f"  Raw shape: {df_14days.shape}")

    # Strip column name whitespace
    df_14days.columns = [c.strip() for c in df_14days.columns]

    # Drop meta columns
    meta_cols = [c for c in df_14days.columns if c in ('Row', 'Date', 'Time')]
    print(f"  Dropping meta columns: {meta_cols}")
    df_14days = df_14days.drop(columns=meta_cols)

    # Verify columns match attack data
    days14_features = list(df_14days.columns)
    if set(attack_feature_cols) == set(days14_features):
        print(f"  Columns match attack data ({len(attack_feature_cols)} features)")
    else:
        missing = set(attack_feature_cols) - set(days14_features)
        extra = set(days14_features) - set(attack_feature_cols)
        if missing:
            print(f"  WARNING: Missing in 14days: {missing}")
        if extra:
            print(f"  WARNING: Extra in 14days: {extra}")

    # Keep only columns that exist in attack data (after all-NaN removal)
    df_14days = df_14days[[c for c in attack_feature_cols if c in df_14days.columns]]

    # Handle NaN (may drop additional all-NaN columns)
    df_14days = handle_nan(df_14days, 'WaDi A2 14days')

    # All normal
    df_14days['label'] = 0

    # Validate
    validate_output(df_14days, expected_rows=None, dataset_name='WaDi A2 14days')

    out_path = data_dir / 'WADI_A2_14days_raw.csv'
    df_14days.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name} ({len(df_14days):,} rows)")


# =============================================================================
# Main
# =============================================================================

DATASET_FUNCS = {
    'swat_a1a2': prepare_swat_a1a2,
    'swat_a4a5': prepare_swat_a4a5,
    'swat_a6': prepare_swat_a6,
    'wadi_a1': prepare_wadi_a1,
    'wadi_a2': prepare_wadi_a2,
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw SWaT/WaDi datasets to pipeline-ready CSV")
    parser.add_argument(
        '--datasets', nargs='*', default=['all'],
        choices=['all'] + list(DATASET_FUNCS.keys()),
        help="Which datasets to convert (default: all)")
    args = parser.parse_args()

    datasets = args.datasets
    if 'all' in datasets:
        datasets = list(DATASET_FUNCS.keys())

    print(f"Converting datasets: {datasets}\n")

    for ds in datasets:
        DATASET_FUNCS[ds]()

    print(f"\n{'='*60}")
    print(f"  ALL CONVERSIONS COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
