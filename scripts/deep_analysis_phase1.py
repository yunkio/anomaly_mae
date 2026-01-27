#!/usr/bin/env python3
"""
Deep analysis of Phase 1 ablation study results.
Addresses all 10 focus areas specified by the user.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load the aggregated experiment data."""
    data_path = Path("docs/ablation_result/all_experiments.csv")
    df = pd.DataFrame(pd.read_csv(data_path))

    # Extract parameters from experiment names
    df['window_size'] = df['experiment_name'].str.extract(r'w(\d+)').astype(float).fillna(100)
    df['patch_size'] = df['experiment_name'].str.extract(r'p(\d+)').astype(float).fillna(10)
    df['d_model'] = df['experiment_name'].str.extract(r'd_model_(\d+)|(?:^|_)d(\d+)(?:_|$)').bfill(axis=1)[0].astype(float).fillna(256)
    df['num_heads'] = df['experiment_name'].str.extract(r'nhead(\d+)').astype(float).fillna(8)
    df['encoder_depth'] = df['experiment_name'].str.extract(r'encoder_(\d+)').astype(float).fillna(6)
    df['decoder_depth'] = df['experiment_name'].str.extract(r'decoder_(\d+)').astype(float).fillna(3)
    df['ffn_dim'] = df['experiment_name'].str.extract(r'ffn_(\d+)').astype(float).fillna(2048)
    df['masking_ratio'] = df['experiment_name'].str.extract(r'mask_(\d+\.\d+)').astype(float).fillna(0.75)
    df['lambda_disc'] = df['experiment_name'].str.extract(r'lambda_(\d+\.\d+)').astype(float).fillna(2.0)
    df['k_value'] = df['experiment_name'].str.extract(r'k_(\d+\.\d+)').astype(float).fillna(2.0)
    df['dropout'] = df['experiment_name'].str.extract(r'dropout(\d+\.\d+)|dropout_(\d+\.\d+)').bfill(axis=1)[0].astype(float).fillna(0.1)
    df['teacher_ratio'] = df['experiment_name'].str.extract(r't(\d+)s\d+').astype(float)
    df['student_ratio'] = df['experiment_name'].str.extract(r't\d+s(\d+)').astype(float)
    df['mask_timing'] = df['experiment_name'].str.extract(r'mask_(after|before)')[0].fillna('after')

    # Create combined ratio columns
    df['ts_ratio'] = df['teacher_ratio'].astype(str) + 's' + df['student_ratio'].astype(str)
    df['ts_ratio'] = df['ts_ratio'].replace('nans nan', 'default')

    return df


def focus_area_1_high_disc_ratio_models(df: pd.DataFrame) -> Dict:
    """
    Focus Area 1: Analyze models with high discrepancy ratio.
    Find characteristics that maximize disc_ratio and disc_cohens_d_normal_vs_anomaly.
    """
    print("\n" + "="*100)
    print("FOCUS AREA 1: High Discrepancy Ratio Models")
    print("="*100)

    # Filter top performers by disc_cohens_d_normal_vs_anomaly
    top_disc_cohens = df.nlargest(50, 'disc_cohens_d_normal_vs_anomaly')

    analysis = {
        'summary': {},
        'parameter_effects': {}
    }

    # Analyze parameter distributions in high disc_cohens_d models
    print("\n--- Parameter Characteristics of High disc_cohens_d Models ---")

    params_to_analyze = ['d_model', 'masking_ratio', 'window_size', 'patch_size',
                         'encoder_depth', 'decoder_depth', 'mask_timing', 'dropout',
                         'lambda_disc', 'k_value', 'ffn_dim']

    for param in params_to_analyze:
        if param in top_disc_cohens.columns:
            value_counts = top_disc_cohens[param].value_counts()
            if len(value_counts) > 0:
                print(f"\n{param}:")
                print(value_counts.head(10))

                # Compare with overall distribution
                overall_mean = df[param].mean()
                top_mean = top_disc_cohens[param].mean()
                print(f"  Overall mean: {overall_mean:.3f}, Top models mean: {top_mean:.3f}")

                analysis['parameter_effects'][param] = {
                    'value_counts': value_counts.to_dict(),
                    'overall_mean': float(overall_mean) if not pd.isna(overall_mean) else None,
                    'top_mean': float(top_mean) if not pd.isna(top_mean) else None
                }

    # Correlation analysis
    print("\n--- Correlation with disc_cohens_d_normal_vs_anomaly ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['disc_cohens_d_normal_vs_anomaly'].sort_values(ascending=False)
    print(correlations.head(20))

    analysis['correlations'] = correlations.to_dict()

    # Identify winning combinations
    print("\n--- Top Combinations for High disc_cohens_d ---")
    top_5 = df.nlargest(5, 'disc_cohens_d_normal_vs_anomaly')
    print(top_5[['experiment_name', 'disc_cohens_d_normal_vs_anomaly', 'disc_ratio_1',
                 'roc_auc', 'd_model', 'masking_ratio', 'mask_timing']].to_string())

    analysis['top_combinations'] = top_5[['experiment_name', 'disc_cohens_d_normal_vs_anomaly',
                                          'disc_ratio_1', 'roc_auc']].to_dict('records')

    return analysis


def focus_area_2_high_disc_and_recon(df: pd.DataFrame) -> Dict:
    """
    Focus Area 2: Models with both high disc_cohens_d AND high recon_cohens_d.
    """
    print("\n" + "="*100)
    print("FOCUS AREA 2: Models with High Disc AND Recon Cohen's d")
    print("="*100)

    # Define threshold for "high"
    disc_threshold = df['disc_cohens_d_normal_vs_anomaly'].quantile(0.75)
    recon_threshold = df['recon_cohens_d_normal_vs_anomaly'].quantile(0.75)

    high_both = df[
        (df['disc_cohens_d_normal_vs_anomaly'] > disc_threshold) &
        (df['recon_cohens_d_normal_vs_anomaly'] > recon_threshold)
    ]

    print(f"\nFound {len(high_both)} models with both disc_cohens_d > {disc_threshold:.3f}")
    print(f"and recon_cohens_d > {recon_threshold:.3f}")

    # Analyze characteristics
    print("\n--- Characteristics of High Disc + Recon Models ---")
    print(f"Average ROC-AUC: {high_both['roc_auc'].mean():.4f}")
    print(f"Average disc_ratio: {high_both['disc_ratio_1'].mean():.4f}")
    print(f"Average t_ratio: {high_both['t_ratio'].mean():.4f}")
    print(f"Average PA%80 ROC-AUC: {high_both['PA%80_roc_auc'].mean():.4f}")

    # Parameter analysis
    print("\n--- Key Parameters ---")
    for param in ['d_model', 'masking_ratio', 'window_size', 'mask_timing', 'inference_mode']:
        if param in high_both.columns:
            print(f"\n{param}:")
            print(high_both[param].value_counts().head())

    # Top models
    print("\n--- Top 10 Models with High Disc + Recon ---")
    top_both = high_both.nlargest(10, 'roc_auc')
    print(top_both[['experiment_name', 'roc_auc', 'disc_cohens_d_normal_vs_anomaly',
                    'recon_cohens_d_normal_vs_anomaly', 'disc_ratio_1', 't_ratio']].to_string())

    analysis = {
        'count': len(high_both),
        'avg_roc_auc': float(high_both['roc_auc'].mean()),
        'avg_disc_ratio': float(high_both['disc_ratio_1'].mean()),
        'avg_t_ratio': float(high_both['t_ratio'].mean()),
        'top_models': top_both[['experiment_name', 'roc_auc', 'disc_cohens_d_normal_vs_anomaly',
                                'recon_cohens_d_normal_vs_anomaly']].to_dict('records')
    }

    return analysis, high_both


def focus_area_3_scoring_window_effects(df: pd.DataFrame, high_both_df: pd.DataFrame) -> Dict:
    """
    Focus Area 3: Effect of scoring mode and window size on high disc+recon models.
    """
    print("\n" + "="*100)
    print("FOCUS AREA 3: Scoring Mode and Window Size Effects")
    print("="*100)

    # Get base experiment names (without scoring/inference suffixes)
    analysis = {'scoring_effects': [], 'window_effects': []}

    # Analyze scoring mode effects for top models
    print("\n--- Scoring Mode Effects on Same Model ---")

    # Group by base experiment (same parameters, different scoring/inference)
    # Extract base name pattern
    base_models = {}
    for idx, row in df.iterrows():
        # Get experiment ID (number part)
        exp_id = row['experiment_name'].split('_')[0]
        if exp_id not in base_models:
            base_models[exp_id] = []
        base_models[exp_id].append(row)

    # Find models with multiple scoring variations
    multi_scoring = {}
    for exp_id, rows in base_models.items():
        if len(rows) >= 3:  # Has multiple scoring modes
            # Check if it's a high performer
            any_high = any(r['roc_auc'] > 0.94 for r in rows)
            if any_high:
                multi_scoring[exp_id] = rows

    print(f"\nFound {len(multi_scoring)} high-performing models with multiple scoring modes")

    # Analyze top cases
    for exp_id in list(multi_scoring.keys())[:5]:
        rows = multi_scoring[exp_id]
        print(f"\n{exp_id}:")
        for r in rows:
            print(f"  {r['scoring_mode']:10s} | inference: {r['inference_mode']:15s} | "
                  f"ROC-AUC: {r['roc_auc']:.4f} | disc_ratio: {r['disc_ratio_1']:.3f} | "
                  f"t_ratio: {r['t_ratio']:.3f}")

        analysis['scoring_effects'].append({
            'experiment_id': exp_id,
            'variations': [{
                'scoring_mode': r['scoring_mode'],
                'inference_mode': r['inference_mode'],
                'roc_auc': float(r['roc_auc']),
                'disc_ratio': float(r['disc_ratio_1']),
                't_ratio': float(r['t_ratio'])
            } for r in rows]
        })

    # Window size analysis
    print("\n--- Window Size Effects ---")
    for ws in [100, 500, 1000]:
        ws_data = df[df['window_size'] == ws]
        if len(ws_data) > 0:
            print(f"\nWindow Size {ws}:")
            print(f"  Count: {len(ws_data)}")
            print(f"  Avg ROC-AUC: {ws_data['roc_auc'].mean():.4f}")
            print(f"  Avg disc_ratio: {ws_data['disc_ratio_1'].mean():.4f}")
            print(f"  Avg t_ratio: {ws_data['t_ratio'].mean():.4f}")
            print(f"  Avg PA%80 ROC-AUC: {ws_data['PA%80_roc_auc'].mean():.4f}")

            analysis['window_effects'].append({
                'window_size': int(ws),
                'count': len(ws_data),
                'avg_roc_auc': float(ws_data['roc_auc'].mean()),
                'avg_disc_ratio': float(ws_data['disc_ratio_1'].mean()),
                'avg_t_ratio': float(ws_data['t_ratio'].mean())
            })

    return analysis


def focus_area_4_disturbing_normal_separation(df: pd.DataFrame) -> Dict:
    """
    Focus Area 4: Models that separate disturbing normal from anomaly well.
    """
    print("\n" + "="*100)
    print("FOCUS AREA 4: Disturbing Normal vs Anomaly Separation")
    print("="*100)

    # Find top performers on disc_cohens_d_disturbing_vs_anomaly
    top_disturbing = df.nlargest(30, 'disc_cohens_d_disturbing_vs_anomaly')

    print(f"\n--- Top Models for Disturbing vs Anomaly Separation ---")
    print(top_disturbing[['experiment_name', 'roc_auc', 'disc_cohens_d_disturbing_vs_anomaly',
                          'disc_cohens_d_normal_vs_anomaly', 'disc_ratio_1', 'disc_ratio_2']].head(15).to_string())

    # Parameter analysis
    print("\n--- Key Parameters for Disturbing Separation ---")
    for param in ['d_model', 'masking_ratio', 'mask_timing', 'window_size', 'patch_size']:
        if param in top_disturbing.columns:
            print(f"\n{param}:")
            print(top_disturbing[param].value_counts().head())

    # Correlation analysis
    print("\n--- Correlation with disc_cohens_d_disturbing_vs_anomaly ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['disc_cohens_d_disturbing_vs_anomaly'].sort_values(ascending=False)
    print(correlations.head(15))

    analysis = {
        'top_models': top_disturbing[['experiment_name', 'disc_cohens_d_disturbing_vs_anomaly',
                                      'roc_auc']].head(10).to_dict('records'),
        'correlations': correlations.to_dict()
    }

    return analysis


def focus_area_5_high_pa80_with_disc_ratio(df: pd.DataFrame) -> Dict:
    """
    Focus Area 5: Models with high PA%80 performance AND high disc_ratio.
    """
    print("\n" + "="*100)
    print("FOCUS AREA 5: High PA%80 with High Disc Ratio")
    print("="*100)

    # Define thresholds
    pa80_threshold = df['PA%80_roc_auc'].quantile(0.75)
    disc_ratio_threshold = df['disc_ratio_1'].quantile(0.75)

    high_both = df[
        (df['PA%80_roc_auc'] > pa80_threshold) &
        (df['disc_ratio_1'] > disc_ratio_threshold)
    ]

    print(f"\nFound {len(high_both)} models with PA%80 > {pa80_threshold:.3f} and disc_ratio > {disc_ratio_threshold:.3f}")

    if len(high_both) > 0:
        print("\n--- Top Models ---")
        top = high_both.nlargest(15, 'PA%80_roc_auc')
        print(top[['experiment_name', 'PA%80_roc_auc', 'disc_ratio_1', 'roc_auc',
                   'd_model', 'masking_ratio', 'mask_timing']].to_string())

        # Parameter analysis
        print("\n--- Key Parameters ---")
        for param in ['d_model', 'masking_ratio', 'mask_timing', 'window_size', 'encoder_depth', 'decoder_depth']:
            if param in high_both.columns:
                print(f"\n{param}:")
                print(high_both[param].value_counts())

    analysis = {
        'count': len(high_both),
        'top_models': high_both.nlargest(10, 'PA%80_roc_auc')[['experiment_name', 'PA%80_roc_auc',
                                                                 'disc_ratio_1', 'roc_auc']].to_dict('records') if len(high_both) > 0 else []
    }

    return analysis


def focus_area_6_window_depth_masking(df: pd.DataFrame) -> Dict:
    """
    Focus Area 6: Relationship between window size, model depth, and masking ratio.
    """
    print("\n" + "="*100)
    print("FOCUS AREA 6: Window Size, Model Depth, and Masking Ratio")
    print("="*100)

    analysis = {}

    # Analyze by window size
    for ws in [100, 500, 1000]:
        ws_data = df[df['window_size'] == ws]
        if len(ws_data) == 0:
            continue

        print(f"\n--- Window Size {ws} ---")

        # Group by decoder depth
        print("\nBy Decoder Depth:")
        decoder_groups = ws_data.groupby('decoder_depth').agg({
            'roc_auc': ['mean', 'std', 'count'],
            'disc_ratio_1': ['mean', 'std'],
            't_ratio': ['mean', 'std']
        }).round(4)
        print(decoder_groups)

        # Group by d_model
        print("\nBy d_model:")
        dmodel_groups = ws_data.groupby('d_model').agg({
            'roc_auc': ['mean', 'std', 'count'],
            'disc_ratio_1': ['mean', 'std']
        }).round(4)
        print(dmodel_groups)

        # Group by masking ratio
        print("\nBy Masking Ratio:")
        mask_groups = ws_data.groupby('masking_ratio').agg({
            'roc_auc': ['mean', 'std', 'count'],
            'disc_ratio_1': ['mean', 'std']
        }).round(4)
        print(mask_groups)

    return analysis


def focus_area_7_mask_after_optimization(df: pd.DataFrame) -> Dict:
    """
    Focus Area 7: Maximize disc_loss and t_ratio with mask_after=True.
    """
    print("\n" + "="*100)
    print("FOCUS AREA 7: Mask After Optimization")
    print("="*100)

    mask_after = df[df['mask_timing'] == 'after']

    print(f"\nTotal mask_after experiments: {len(mask_after)}")

    # Find models with high disc_ratio and t_ratio
    print("\n--- Models with High disc_ratio AND t_ratio ---")
    mask_after['combined_score'] = mask_after['disc_ratio_1'] * mask_after['t_ratio']
    top_combined = mask_after.nlargest(20, 'combined_score')

    print(top_combined[['experiment_name', 'roc_auc', 'disc_ratio_1', 't_ratio',
                        'combined_score', 'd_model', 'masking_ratio', 'lambda_disc']].to_string())

    # Parameter analysis for top performers
    print("\n--- Key Parameters for High disc + t_ratio ---")
    for param in ['d_model', 'masking_ratio', 'lambda_disc', 'decoder_depth', 'ffn_dim']:
        if param in top_combined.columns:
            print(f"\n{param}:")
            print(top_combined[param].value_counts())

    analysis = {
        'top_models': top_combined[['experiment_name', 'disc_ratio_1', 't_ratio',
                                    'combined_score', 'roc_auc']].to_dict('records')
    }

    return analysis


def focus_area_8_scoring_inference_sensitivity(df: pd.DataFrame) -> Dict:
    """
    Focus Area 8: Identify parameters sensitive to scoring/inference mode changes.
    """
    print("\n" + "="*100)
    print("FOCUS AREA 8: Scoring and Inference Mode Sensitivity")
    print("="*100)

    # Group experiments by base configuration (without scoring/inference)
    # Calculate performance variance across scoring modes

    analysis = {'high_variance_params': []}

    # Get unique experiment IDs
    df['exp_id'] = df['experiment_name'].str.extract(r'^(\d+)_')[0]

    # For each experiment ID, calculate variance across scoring/inference modes
    variance_data = []

    for exp_id in df['exp_id'].unique():
        if pd.isna(exp_id):
            continue

        exp_variants = df[df['exp_id'] == exp_id]

        if len(exp_variants) >= 3:  # Has multiple variants
            roc_std = exp_variants['roc_auc'].std()
            roc_mean = exp_variants['roc_auc'].mean()
            roc_range = exp_variants['roc_auc'].max() - exp_variants['roc_auc'].min()

            if roc_mean > 0.90:  # Only consider reasonably good models
                variance_data.append({
                    'exp_id': exp_id,
                    'exp_name': exp_variants.iloc[0]['experiment_name'].split('_mask')[0],
                    'count': len(exp_variants),
                    'roc_mean': roc_mean,
                    'roc_std': roc_std,
                    'roc_range': roc_range,
                    'disc_ratio_std': exp_variants['disc_ratio_1'].std(),
                    'sample_row': exp_variants.iloc[0]
                })

    variance_df = pd.DataFrame(variance_data)

    if len(variance_df) > 0:
        # Find high-variance cases
        print("\n--- High Variance Models (most sensitive to scoring/inference) ---")
        high_var = variance_df.nlargest(15, 'roc_range')
        print(high_var[['exp_name', 'roc_mean', 'roc_std', 'roc_range']].to_string())

        # Analyze parameters of high-variance models
        print("\n--- Parameters of High-Variance Models ---")
        for param in ['d_model', 'masking_ratio', 'window_size', 'decoder_depth']:
            if param in df.columns:
                param_vals = [row['sample_row'][param] for _, row in high_var.iterrows()]
                print(f"\n{param}: {pd.Series(param_vals).value_counts()}")

    return analysis


def focus_area_9_high_perf_with_disturbing_sep(df: pd.DataFrame) -> Dict:
    """
    Focus Area 9: Good anomaly detection + good disturbing normal separation.
    """
    print("\n" + "="*100)
    print("FOCUS AREA 9: High Performance with Disturbing Normal Separation")
    print("="*100)

    # Define thresholds
    roc_threshold = 0.945  # Top tier performance
    disturbing_threshold = df['disc_cohens_d_disturbing_vs_anomaly'].quantile(0.70)

    high_both = df[
        (df['roc_auc'] > roc_threshold) &
        (df['disc_cohens_d_disturbing_vs_anomaly'] > disturbing_threshold)
    ]

    print(f"\nFound {len(high_both)} models with ROC-AUC > {roc_threshold}")
    print(f"and disc_cohens_d_disturbing > {disturbing_threshold:.3f}")

    if len(high_both) > 0:
        print("\n--- Top Models ---")
        top = high_both.nlargest(15, 'roc_auc')
        print(top[['experiment_name', 'roc_auc', 'disc_cohens_d_disturbing_vs_anomaly',
                   'disc_ratio_2', 'd_model', 'window_size', 'mask_timing']].to_string())

        # Parameter analysis
        print("\n--- Key Parameters ---")
        for param in ['d_model', 'window_size', 'mask_timing', 'masking_ratio', 'patch_size']:
            if param in high_both.columns:
                print(f"\n{param}:")
                print(high_both[param].value_counts())

    analysis = {
        'count': len(high_both),
        'top_models': high_both.nlargest(10, 'roc_auc')[['experiment_name', 'roc_auc',
                                                          'disc_cohens_d_disturbing_vs_anomaly']].to_dict('records') if len(high_both) > 0 else []
    }

    return analysis


def focus_area_10_additional_insights(df: pd.DataFrame) -> Dict:
    """
    Focus Area 10: Additional insights not covered in 1-9.
    """
    print("\n" + "="*100)
    print("FOCUS AREA 10: Additional Insights")
    print("="*100)

    analysis = {}

    # 1. Inference mode comparison
    print("\n--- Inference Mode Performance ---")
    inf_mode_stats = df.groupby('inference_mode').agg({
        'roc_auc': ['mean', 'std', 'max', 'count'],
        'disc_ratio_1': ['mean', 'std'],
        't_ratio': ['mean', 'std']
    }).round(4)
    print(inf_mode_stats)

    # 2. Scoring mode comparison
    print("\n--- Scoring Mode Performance ---")
    score_mode_stats = df.groupby('scoring_mode').agg({
        'roc_auc': ['mean', 'std', 'max', 'count'],
        'disc_ratio_1': ['mean', 'std']
    }).round(4)
    print(score_mode_stats)

    # 3. mask_before vs mask_after
    print("\n--- Mask Timing Comparison ---")
    mask_timing_stats = df.groupby('mask_timing').agg({
        'roc_auc': ['mean', 'std', 'max', 'count'],
        'disc_ratio_1': ['mean', 'std'],
        't_ratio': ['mean', 'std'],
        'disc_cohens_d_normal_vs_anomaly': ['mean', 'std']
    }).round(4)
    print(mask_timing_stats)

    # 4. Patch size effects
    print("\n--- Patch Size Effects ---")
    patch_stats = df.groupby('patch_size').agg({
        'roc_auc': ['mean', 'std', 'count'],
        'disc_ratio_1': ['mean', 'std']
    }).round(4)
    print(patch_stats)

    # 5. Lambda discrepancy effects
    print("\n--- Lambda Discrepancy Effects ---")
    lambda_stats = df.groupby('lambda_disc').agg({
        'roc_auc': ['mean', 'std', 'count'],
        'disc_ratio_1': ['mean', 'std'],
        't_ratio': ['mean', 'std']
    }).round(4)
    print(lambda_stats)

    # 6. Teacher/Student ratio effects
    print("\n--- Teacher/Student Ratio Effects ---")
    ts_stats = df.groupby('ts_ratio').agg({
        'roc_auc': ['mean', 'std', 'count'],
        'disc_ratio_1': ['mean', 'std'],
        't_ratio': ['mean', 'std']
    }).round(4)
    print(ts_stats)

    # 7. Overall best combination identification
    print("\n--- Overall Best Models (Multi-Metric) ---")
    # Create composite score
    df['composite_score'] = (
        df['roc_auc'] * 0.4 +
        (df['disc_cohens_d_normal_vs_anomaly'] / df['disc_cohens_d_normal_vs_anomaly'].max()) * 0.2 +
        (df['recon_cohens_d_normal_vs_anomaly'] / df['recon_cohens_d_normal_vs_anomaly'].max()) * 0.2 +
        (df['PA%80_roc_auc'] / df['PA%80_roc_auc'].max()) * 0.2
    )

    top_composite = df.nlargest(20, 'composite_score')
    print(top_composite[['experiment_name', 'roc_auc', 'disc_cohens_d_normal_vs_anomaly',
                         'recon_cohens_d_normal_vs_anomaly', 'PA%80_roc_auc',
                         'composite_score']].to_string())

    return analysis


def main():
    """Run comprehensive deep analysis."""

    print("="*100)
    print("DEEP ANALYSIS OF PHASE 1 ABLATION STUDY")
    print("="*100)

    # Load data
    df = load_data()
    print(f"\nLoaded {len(df)} experiments")

    all_analysis = {}

    # Run all focus areas
    all_analysis['focus_1'] = focus_area_1_high_disc_ratio_models(df)

    all_analysis['focus_2'], high_both_df = focus_area_2_high_disc_and_recon(df)

    all_analysis['focus_3'] = focus_area_3_scoring_window_effects(df, high_both_df)

    all_analysis['focus_4'] = focus_area_4_disturbing_normal_separation(df)

    all_analysis['focus_5'] = focus_area_5_high_pa80_with_disc_ratio(df)

    all_analysis['focus_6'] = focus_area_6_window_depth_masking(df)

    all_analysis['focus_7'] = focus_area_7_mask_after_optimization(df)

    all_analysis['focus_8'] = focus_area_8_scoring_inference_sensitivity(df)

    all_analysis['focus_9'] = focus_area_9_high_perf_with_disturbing_sep(df)

    all_analysis['focus_10'] = focus_area_10_additional_insights(df)

    # Save analysis results
    output_dir = Path("docs/ablation_result")
    with open(output_dir / 'deep_analysis_results.json', 'w') as f:
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            return obj

        json.dump(all_analysis, f, indent=2, default=convert_types)

    print("\n" + "="*100)
    print("DEEP ANALYSIS COMPLETE")
    print("="*100)
    print(f"\nResults saved to {output_dir / 'deep_analysis_results.json'}")


if __name__ == "__main__":
    main()
