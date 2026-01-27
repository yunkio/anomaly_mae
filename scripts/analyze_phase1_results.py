#!/usr/bin/env python3
"""
Comprehensive analysis of ablation_phase1 results.
Aggregates all experiment metadata and performs deep analysis.
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_all_experiments(results_dir: str) -> pd.DataFrame:
    """Load all experiment metadata from the results directory."""
    results_path = Path(results_dir)

    all_data = []

    # Find all experiment directories
    for exp_dir in sorted(results_path.glob("*")):
        if not exp_dir.is_dir():
            continue

        metadata_file = exp_dir / "experiment_metadata.json"
        if not metadata_file.exists():
            print(f"Warning: No metadata found in {exp_dir.name}")
            continue

        with open(metadata_file, 'r') as f:
            data = json.load(f)

        # Flatten the structure
        row = {
            'experiment_name': data['experiment_name'],
            'scoring_mode': data.get('scoring_mode', 'unknown'),
            'inference_mode': data.get('inference_mode', 'unknown'),
            'train_time': data.get('train_time', 0),
            'inference_time': data.get('inference_time', 0),
        }

        # Add metrics
        if 'metrics' in data:
            for key, value in data['metrics'].items():
                row[f'metric_{key}'] = value

        # Add loss stats
        if 'loss_stats' in data:
            for key, value in data['loss_stats'].items():
                row[f'loss_{key}'] = value

        all_data.append(row)

    df = pd.DataFrame(all_data)

    # Rename columns to match user's requested format
    df = df.rename(columns={
        'metric_roc_auc': 'roc_auc',
        'metric_f1_score': 'f1_score',
        'metric_pa_20_roc_auc': 'PA%20_roc_auc',
        'metric_pa_20_f1': 'PA%20_f1_score',
        'metric_pa_50_roc_auc': 'PA%50_roc_auc',
        'metric_pa_50_f1': 'PA%50_f1_score',
        'metric_pa_80_roc_auc': 'PA%80_roc_auc',
        'metric_pa_80_f1': 'PA%80_f1_score',
        'loss_disc_ratio': 'disc_ratio_1',
        'loss_disc_ratio_disturbing': 'disc_ratio_2',
        'loss_recon_ratio': 't_ratio',
        'loss_disc_cohens_d_normal_vs_anomaly': 'disc_cohens_d_normal_vs_anomaly',
        'loss_disc_cohens_d_disturbing_vs_anomaly': 'disc_cohens_d_disturbing_vs_anomaly',
        'loss_recon_cohens_d_normal_vs_anomaly': 'recon_cohens_d_normal_vs_anomaly',
        'loss_recon_cohens_d_disturbing_vs_anomaly': 'recon_cohens_d_disturbing_vs_anomaly',
    })

    return df


def create_summary_tables(df: pd.DataFrame, output_dir: Path) -> Dict[str, pd.DataFrame]:
    """Create the three requested summary tables."""

    # Define columns to include in tables
    table_columns = [
        'experiment_name',
        'roc_auc',
        'f1_score',
        'PA%20_roc_auc',
        'PA%20_f1_score',
        'PA%50_roc_auc',
        'PA%50_f1_score',
        'PA%80_roc_auc',
        'PA%80_f1_score',
        'disc_ratio_1',
        'disc_ratio_2',
        't_ratio',
        'disc_cohens_d_normal_vs_anomaly',
        'disc_cohens_d_disturbing_vs_anomaly',
        'recon_cohens_d_normal_vs_anomaly',
        'recon_cohens_d_disturbing_vs_anomaly',
        'inference_mode',
        'scoring_mode'
    ]

    # Ensure all columns exist
    for col in table_columns:
        if col not in df.columns:
            print(f"Warning: Column {col} not found in dataframe")

    available_columns = [col for col in table_columns if col in df.columns]

    tables = {}

    # Table 1: Top 10 by ROC-AUC
    if 'roc_auc' in df.columns:
        top_roc = df.nlargest(10, 'roc_auc')[available_columns]
        tables['top_10_roc_auc'] = top_roc
        top_roc.to_csv(output_dir / 'table1_top10_roc_auc.csv', index=False)
        print(f"Table 1: Top 10 by ROC-AUC")
        print(top_roc.to_string())
        print("\n" + "="*100 + "\n")

    # Table 2: Top 10 by disc_ratio
    if 'disc_ratio_1' in df.columns:
        top_disc = df.nlargest(10, 'disc_ratio_1')[available_columns]
        tables['top_10_disc_ratio'] = top_disc
        top_disc.to_csv(output_dir / 'table2_top10_disc_ratio.csv', index=False)
        print(f"Table 2: Top 10 by Discrepancy Ratio")
        print(top_disc.to_string())
        print("\n" + "="*100 + "\n")

    # Table 3: Top 10 by t_ratio (teacher reconstruction ratio)
    if 't_ratio' in df.columns:
        top_t = df.nlargest(10, 't_ratio')[available_columns]
        tables['top_10_t_ratio'] = top_t
        top_t.to_csv(output_dir / 'table3_top10_t_ratio.csv', index=False)
        print(f"Table 3: Top 10 by T-Ratio (Teacher Reconstruction Ratio)")
        print(top_t.to_string())
        print("\n" + "="*100 + "\n")

    return tables


def analyze_parameter_effects(df: pd.DataFrame) -> Dict:
    """Analyze the effect of each parameter on performance metrics."""

    analysis = {}

    # Extract parameter variations from experiment names
    df['window_size'] = df['experiment_name'].str.extract(r'w(\d+)').astype(float).fillna(100)
    df['patch_size'] = df['experiment_name'].str.extract(r'p(\d+)').astype(float).fillna(10)
    df['d_model'] = df['experiment_name'].str.extract(r'd(\d+)').astype(float).fillna(256)
    df['num_heads'] = df['experiment_name'].str.extract(r'nhead(\d+)').astype(float).fillna(8)
    df['encoder_depth'] = df['experiment_name'].str.extract(r'encoder_(\d+)').astype(float).fillna(6)
    df['decoder_depth'] = df['experiment_name'].str.extract(r'decoder_(\d+)').astype(float).fillna(3)
    df['ffn_ratio'] = df['experiment_name'].str.extract(r'ffn_(\d+)').astype(float).fillna(2048)
    df['masking_ratio'] = df['experiment_name'].str.extract(r'mask_(\d+\.\d+)').astype(float).fillna(0.75)
    df['lambda_disc'] = df['experiment_name'].str.extract(r'lambda_(\d+\.\d+)').astype(float).fillna(2.0)
    df['k_value'] = df['experiment_name'].str.extract(r'k_(\d+\.\d+)').astype(float).fillna(2.0)

    # Extract teacher/student configuration
    df['teacher_ratio'] = df['experiment_name'].str.extract(r't(\d+)s\d+').astype(float)
    df['student_ratio'] = df['experiment_name'].str.extract(r't\d+s(\d+)').astype(float)

    # Extract mask timing (mask_after or mask_before)
    df['mask_timing'] = df['experiment_name'].str.extract(r'mask_(after|before)')[0]
    df['mask_timing'] = df['mask_timing'].fillna('unknown')

    analysis['parameter_summary'] = {
        'window_size': df.groupby('window_size').agg({
            'roc_auc': ['mean', 'std', 'count'],
            'disc_ratio_1': ['mean', 'std'],
            't_ratio': ['mean', 'std'],
            'disc_cohens_d_normal_vs_anomaly': ['mean', 'std']
        }).round(4),

        'd_model': df.groupby('d_model').agg({
            'roc_auc': ['mean', 'std', 'count'],
            'disc_ratio_1': ['mean', 'std'],
            't_ratio': ['mean', 'std']
        }).round(4),

        'masking_ratio': df.groupby('masking_ratio').agg({
            'roc_auc': ['mean', 'std', 'count'],
            'disc_ratio_1': ['mean', 'std'],
            't_ratio': ['mean', 'std']
        }).round(4),

        'mask_timing': df.groupby('mask_timing').agg({
            'roc_auc': ['mean', 'std', 'count'],
            'disc_ratio_1': ['mean', 'std'],
            't_ratio': ['mean', 'std'],
            'disc_cohens_d_normal_vs_anomaly': ['mean', 'std']
        }).round(4),

        'inference_mode': df.groupby('inference_mode').agg({
            'roc_auc': ['mean', 'std', 'count'],
            'disc_ratio_1': ['mean', 'std'],
            't_ratio': ['mean', 'std']
        }).round(4),

        'scoring_mode': df.groupby('scoring_mode').agg({
            'roc_auc': ['mean', 'std', 'count'],
            'disc_ratio_1': ['mean', 'std'],
            't_ratio': ['mean', 'std']
        }).round(4)
    }

    return analysis, df


def main():
    """Main analysis pipeline."""

    # Setup paths
    results_dir = "results/experiments/20260127_055709_phase1"
    output_dir = Path("docs/ablation_result")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*100)
    print("PHASE 1 ABLATION STUDY RESULTS ANALYSIS")
    print("="*100)
    print()

    # Load all experiments
    print("Loading experiment results...")
    df = load_all_experiments(results_dir)
    print(f"Loaded {len(df)} experiments")
    print()

    # Save full dataset
    df.to_csv(output_dir / 'all_experiments.csv', index=False)
    print(f"Full dataset saved to {output_dir / 'all_experiments.csv'}")
    print()

    # Create summary tables
    print("Creating summary tables...")
    tables = create_summary_tables(df, output_dir)
    print()

    # Analyze parameter effects
    print("Analyzing parameter effects...")
    analysis, df_with_params = analyze_parameter_effects(df)

    # Save analysis
    with open(output_dir / 'parameter_analysis.json', 'w') as f:
        # Convert dataframes to dict for JSON serialization
        analysis_json = {}
        for param, data in analysis['parameter_summary'].items():
            analysis_json[param] = data.to_dict()
        json.dump(analysis_json, f, indent=2)

    print(f"Parameter analysis saved to {output_dir / 'parameter_analysis.json'}")
    print()

    # Save enhanced dataset with extracted parameters
    df_with_params.to_csv(output_dir / 'all_experiments_with_params.csv', index=False)
    print(f"Enhanced dataset saved to {output_dir / 'all_experiments_with_params.csv'}")
    print()

    print("="*100)
    print("BASIC ANALYSIS COMPLETE")
    print("="*100)
    print()
    print("Next steps:")
    print("1. Review the three summary tables")
    print("2. Conduct deep analysis with the 10 focus areas")
    print("3. Generate insights and hypotheses")
    print("4. Plan phase 2 experiments")


if __name__ == "__main__":
    main()
