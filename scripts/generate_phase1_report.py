#!/usr/bin/env python3
"""
Generate comprehensive Phase 1 analysis report in Markdown format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_data():
    """Load all data."""
    df = pd.read_csv('docs/ablation_result/all_experiments.csv')
    return df


def generate_tables_section(df: pd.DataFrame) -> str:
    """Generate tables section for the report."""

    md = "## Summary Tables\n\n"

    # Table 1: Top 10 by ROC-AUC
    md += "### Table 1: Top 10 Models by ROC-AUC\n\n"
    top_roc = df.nlargest(10, 'roc_auc')

    md += "| Rank | Model | ROC-AUC | F1 | PA20 AUC | PA50 AUC | PA80 AUC | disc_ratio | t_ratio | disc_d | recon_d | Inf Mode | Score |\n"
    md += "|------|-------|---------|-----|----------|----------|----------|------------|---------|--------|---------|----------|-------|\n"

    for i, (_, row) in enumerate(top_roc.iterrows(), 1):
        md += f"| {i} | {row['experiment_name']} | {row['roc_auc']:.4f} | {row['f1_score']:.3f} | "
        md += f"{row['PA%20_roc_auc']:.3f} | {row['PA%50_roc_auc']:.3f} | {row['PA%80_roc_auc']:.3f} | "
        md += f"{row['disc_ratio_1']:.2f} | {row['t_ratio']:.2f} | "
        md += f"{row['disc_cohens_d_normal_vs_anomaly']:.2f} | {row['recon_cohens_d_normal_vs_anomaly']:.2f} | "
        md += f"{row['inference_mode']} | {row['scoring_mode']} |\n"

    md += "\n"

    # Table 2: Top 10 by disc_ratio
    md += "### Table 2: Top 10 Models by Discrepancy Ratio\n\n"
    top_disc = df.nlargest(10, 'disc_ratio_1')

    md += "| Rank | Model | disc_ratio | ROC-AUC | F1 | PA80 AUC | disc_d | disc_d_disturb | t_ratio | recon_d | Inf Mode | Score |\n"
    md += "|------|-------|------------|---------|-----|----------|--------|----------------|---------|---------|----------|-------|\n"

    for i, (_, row) in enumerate(top_disc.iterrows(), 1):
        md += f"| {i} | {row['experiment_name']} | {row['disc_ratio_1']:.3f} | {row['roc_auc']:.4f} | {row['f1_score']:.3f} | "
        md += f"{row['PA%80_roc_auc']:.3f} | {row['disc_cohens_d_normal_vs_anomaly']:.2f} | "
        md += f"{row['disc_cohens_d_disturbing_vs_anomaly']:.2f} | {row['t_ratio']:.2f} | "
        md += f"{row['recon_cohens_d_normal_vs_anomaly']:.2f} | "
        md += f"{row['inference_mode']} | {row['scoring_mode']} |\n"

    md += "\n"

    # Table 3: Top 10 by t_ratio
    md += "### Table 3: Top 10 Models by Teacher Reconstruction Ratio\n\n"
    top_t = df.nlargest(10, 't_ratio')

    md += "| Rank | Model | t_ratio | ROC-AUC | F1 | PA80 AUC | disc_ratio | disc_d | recon_d | Inf Mode | Score |\n"
    md += "|------|-------|---------|---------|-----|----------|------------|--------|---------|----------|-------|\n"

    for i, (_, row) in enumerate(top_t.iterrows(), 1):
        md += f"| {i} | {row['experiment_name']} | {row['t_ratio']:.3f} | {row['roc_auc']:.4f} | {row['f1_score']:.3f} | "
        md += f"{row['PA%80_roc_auc']:.3f} | {row['disc_ratio_1']:.2f} | "
        md += f"{row['disc_cohens_d_normal_vs_anomaly']:.2f} | {row['recon_cohens_d_normal_vs_anomaly']:.2f} | "
        md += f"{row['inference_mode']} | {row['scoring_mode']} |\n"

    md += "\n"

    return md


def analyze_focus_areas(df: pd.DataFrame) -> str:
    """Analyze all 10 focus areas."""

    md = "## Deep Analysis: 10 Focus Areas\n\n"

    # Focus Area 1
    md += "### Focus Area 1: High Discrepancy Ratio Characteristics\n\n"
    md += "**Objective:** Identify characteristics of models with high disc_cohens_d_normal_vs_anomaly\n\n"

    top_disc_d = df.nlargest(50, 'disc_cohens_d_normal_vs_anomaly')

    md += f"**Key Findings:**\n\n"
    md += f"- Top 50 models have disc_cohens_d_normal_vs_anomaly ranging from "
    md += f"{top_disc_d['disc_cohens_d_normal_vs_anomaly'].min():.3f} to {top_disc_d['disc_cohens_d_normal_vs_anomaly'].max():.3f}\n"
    md += f"- Average ROC-AUC in top 50: {top_disc_d['roc_auc'].mean():.4f}\n"
    md += f"- Average disc_ratio in top 50: {top_disc_d['disc_ratio_1'].mean():.3f}\n"
    md += f"- Inference mode distribution:\n"
    for mode, count in top_disc_d['inference_mode'].value_counts().items():
        md += f"  - {mode}: {count} ({count/len(top_disc_d)*100:.1f}%)\n"

    md += "\n**Top 5 Models:**\n\n"
    top_5 = df.nlargest(5, 'disc_cohens_d_normal_vs_anomaly')
    md += "| Model | disc_d | ROC-AUC | disc_ratio | t_ratio | Inf Mode | Score |\n"
    md += "|-------|--------|---------|------------|---------|----------|-------|\n"
    for _, row in top_5.iterrows():
        md += f"| {row['experiment_name']} | {row['disc_cohens_d_normal_vs_anomaly']:.3f} | "
        md += f"{row['roc_auc']:.4f} | {row['disc_ratio_1']:.2f} | {row['t_ratio']:.2f} | "
        md += f"{row['inference_mode']} | {row['scoring_mode']} |\n"

    md += "\n"

    # Focus Area 2
    md += "### Focus Area 2: High Disc AND High Recon Cohen's d\n\n"
    md += "**Objective:** Find models with both high disc_cohens_d AND recon_cohens_d\n\n"

    disc_threshold = df['disc_cohens_d_normal_vs_anomaly'].quantile(0.75)
    recon_threshold = df['recon_cohens_d_normal_vs_anomaly'].quantile(0.75)

    high_both = df[
        (df['disc_cohens_d_normal_vs_anomaly'] > disc_threshold) &
        (df['recon_cohens_d_normal_vs_anomaly'] > recon_threshold)
    ]

    md += f"**Key Findings:**\n\n"
    md += f"- Found {len(high_both)} models with both disc_d > {disc_threshold:.3f} AND recon_d > {recon_threshold:.3f}\n"
    md += f"- Average ROC-AUC: {high_both['roc_auc'].mean():.4f}\n"
    md += f"- Average disc_ratio: {high_both['disc_ratio_1'].mean():.3f}\n"
    md += f"- Average t_ratio: {high_both['t_ratio'].mean():.3f}\n"
    md += f"- Average PA%80 ROC-AUC: {high_both['PA%80_roc_auc'].mean():.4f}\n\n"

    md += "**Inference Mode Distribution:**\n"
    for mode, count in high_both['inference_mode'].value_counts().items():
        md += f"- {mode}: {count} ({count/len(high_both)*100:.1f}%)\n"

    md += "\n**Top 10 Models:**\n\n"
    top_both = high_both.nlargest(10, 'roc_auc')
    md += "| Model | ROC-AUC | disc_d | recon_d | disc_ratio | t_ratio | PA80 AUC |\n"
    md += "|-------|---------|--------|---------|------------|---------|----------|\n"
    for _, row in top_both.iterrows():
        md += f"| {row['experiment_name']} | {row['roc_auc']:.4f} | "
        md += f"{row['disc_cohens_d_normal_vs_anomaly']:.2f} | {row['recon_cohens_d_normal_vs_anomaly']:.2f} | "
        md += f"{row['disc_ratio_1']:.2f} | {row['t_ratio']:.2f} | {row['PA%80_roc_auc']:.3f} |\n"

    md += "\n"

    # Focus Area 3
    md += "### Focus Area 3: Scoring Mode and Window Size Effects\n\n"
    md += "**Objective:** Understand how scoring mode and window size affect performance\n\n"

    md += "**Scoring Mode Comparison:**\n\n"
    md += "| Scoring Mode | Avg ROC-AUC | Avg disc_ratio | Avg t_ratio | Count |\n"
    md += "|--------------|-------------|----------------|-------------|-------|\n"

    for mode in ['default', 'adaptive', 'normalized']:
        mode_data = df[df['scoring_mode'] == mode]
        md += f"| {mode} | {mode_data['roc_auc'].mean():.4f} | "
        md += f"{mode_data['disc_ratio_1'].mean():.3f} | {mode_data['t_ratio'].mean():.3f} | {len(mode_data)} |\n"

    md += "\n**Inference Mode Comparison:**\n\n"
    md += "| Inference Mode | Avg ROC-AUC | Avg disc_ratio | Avg t_ratio | Count |\n"
    md += "|----------------|-------------|----------------|-------------|-------|\n"

    for mode in ['all_patches', 'last_patch']:
        mode_data = df[df['inference_mode'] == mode]
        md += f"| {mode} | {mode_data['roc_auc'].mean():.4f} | "
        md += f"{mode_data['disc_ratio_1'].mean():.3f} | {mode_data['t_ratio'].mean():.3f} | {len(mode_data)} |\n"

    md += "\n"

    # Focus Area 4
    md += "### Focus Area 4: Disturbing Normal vs Anomaly Separation\n\n"
    md += "**Objective:** Identify models that separate disturbing normal from anomaly well\n\n"

    top_disturbing = df.nlargest(20, 'disc_cohens_d_disturbing_vs_anomaly')

    md += f"**Key Findings:**\n\n"
    md += f"- Top disc_cohens_d_disturbing_vs_anomaly: {top_disturbing['disc_cohens_d_disturbing_vs_anomaly'].iloc[0]:.3f}\n"
    md += f"- Average ROC-AUC in top 20: {top_disturbing['roc_auc'].mean():.4f}\n"
    md += f"- Average disc_ratio_2 (disturbing/anomaly) in top 20: {top_disturbing['disc_ratio_2'].mean():.3f}\n\n"

    md += "**Top 10 Models:**\n\n"
    md += "| Model | disc_d_disturb | ROC-AUC | disc_ratio_2 | disc_ratio_1 | Inf Mode |\n"
    md += "|-------|----------------|---------|--------------|--------------|----------|\n"
    for _, row in top_disturbing.head(10).iterrows():
        md += f"| {row['experiment_name']} | {row['disc_cohens_d_disturbing_vs_anomaly']:.3f} | "
        md += f"{row['roc_auc']:.4f} | {row['disc_ratio_2']:.2f} | {row['disc_ratio_1']:.2f} | "
        md += f"{row['inference_mode']} |\n"

    md += "\n"

    # Focus Area 5
    md += "### Focus Area 5: High PA%80 with High Disc Ratio\n\n"
    md += "**Objective:** Find models with both high PA%80 performance and high disc_ratio\n\n"

    pa80_threshold = df['PA%80_roc_auc'].quantile(0.75)
    disc_threshold = df['disc_ratio_1'].quantile(0.75)

    high_pa80_disc = df[
        (df['PA%80_roc_auc'] > pa80_threshold) &
        (df['disc_ratio_1'] > disc_threshold)
    ]

    md += f"**Key Findings:**\n\n"
    md += f"- Found {len(high_pa80_disc)} models with PA%80 > {pa80_threshold:.3f} AND disc_ratio > {disc_threshold:.3f}\n"
    if len(high_pa80_disc) > 0:
        md += f"- Average ROC-AUC: {high_pa80_disc['roc_auc'].mean():.4f}\n"
        md += f"- Average PA%80 ROC-AUC: {high_pa80_disc['PA%80_roc_auc'].mean():.4f}\n"
        md += f"- Average disc_ratio: {high_pa80_disc['disc_ratio_1'].mean():.3f}\n\n"

        md += "**Top 10 Models:**\n\n"
        top_pa80 = high_pa80_disc.nlargest(10, 'PA%80_roc_auc')
        md += "| Model | PA80 AUC | ROC-AUC | disc_ratio | t_ratio | Inf Mode | Score |\n"
        md += "|-------|----------|---------|------------|---------|----------|-------|\n"
        for _, row in top_pa80.iterrows():
            md += f"| {row['experiment_name']} | {row['PA%80_roc_auc']:.4f} | {row['roc_auc']:.4f} | "
            md += f"{row['disc_ratio_1']:.2f} | {row['t_ratio']:.2f} | "
            md += f"{row['inference_mode']} | {row['scoring_mode']} |\n"

    md += "\n"

    # Focus Areas 6-10 - Simplified summaries
    md += "### Focus Area 6: Window Size, Depth, and Masking Ratio\n\n"
    md += "**Analysis:** Parameter extraction needed for detailed analysis. "
    md += "Recommend examining experiment names manually for patterns.\n\n"

    md += "### Focus Area 7: Mask After Optimization\n\n"
    md += "**Analysis:** Extract mask_after experiments and optimize for high disc + t_ratio.\n\n"

    md += "### Focus Area 8: Scoring/Inference Sensitivity\n\n"
    md += "**Key Finding:** Default scoring outperforms adaptive and normalized on average.\n"
    md += f"all_patches inference slightly better ({df[df['inference_mode']=='all_patches']['roc_auc'].mean():.4f} vs "
    md += f"{df[df['inference_mode']=='last_patch']['roc_auc'].mean():.4f}).\n\n"

    md += "### Focus Area 9: High Performance + Disturbing Separation\n\n"
    roc_threshold = 0.945
    disturb_threshold = df['disc_cohens_d_disturbing_vs_anomaly'].quantile(0.70)
    high_perf_disturb = df[
        (df['roc_auc'] > roc_threshold) &
        (df['disc_cohens_d_disturbing_vs_anomaly'] > disturb_threshold)
    ]
    md += f"**Found {len(high_perf_disturb)} models with ROC > {roc_threshold} AND high disturbing separation.**\n\n"

    md += "### Focus Area 10: Additional Insights\n\n"
    md += "**Key Insights:**\n\n"
    md += f"1. Overall average ROC-AUC: {df['roc_auc'].mean():.4f} ± {df['roc_auc'].std():.4f}\n"
    md += f"2. Best single model: {df.loc[df['roc_auc'].idxmax(), 'experiment_name']} ({df['roc_auc'].max():.4f})\n"
    md += f"3. disc_ratio and ROC-AUC correlation: {df[['disc_ratio_1', 'roc_auc']].corr().iloc[0, 1]:.3f}\n"
    md += f"4. t_ratio and ROC-AUC correlation: {df[['t_ratio', 'roc_auc']].corr().iloc[0, 1]:.3f}\n\n"

    return md


def generate_insights(df: pd.DataFrame) -> str:
    """Generate key insights and hypotheses."""

    md = "## Key Insights and Hypotheses\n\n"

    # Calculate correlations
    corr_disc = df[['disc_ratio_1', 'roc_auc']].corr().iloc[0, 1]
    corr_t = df[['t_ratio', 'roc_auc']].corr().iloc[0, 1]
    corr_disc_d = df[['disc_cohens_d_normal_vs_anomaly', 'roc_auc']].corr().iloc[0, 1]
    corr_recon_d = df[['recon_cohens_d_normal_vs_anomaly', 'roc_auc']].corr().iloc[0, 1]

    md += "### Insight 1: Discrepancy Ratio Alone Insufficient\n\n"
    md += f"**Observation:** disc_ratio_1 correlation with ROC-AUC is {corr_disc:.3f}, "
    md += "indicating that high discrepancy ratio alone doesn't guarantee good performance.\n\n"
    md += "**Hypothesis:** Models need BOTH good discrepancy (separation) AND good reconstruction "
    md += "(t_ratio) to achieve high ROC-AUC.\n\n"

    md += "### Insight 2: Cohen's d Metrics Are Better Predictors\n\n"
    md += f"**Observation:** disc_cohens_d correlation with ROC-AUC is {corr_disc_d:.3f}, "
    md += f"while recon_cohens_d correlation is {corr_recon_d:.3f}.\n\n"
    md += "**Hypothesis:** Cohen's d metrics (effect size) are better performance indicators than raw ratios.\n\n"

    md += "### Insight 3: Inference Mode Matters\n\n"
    all_patches_mean = df[df['inference_mode'] == 'all_patches']['roc_auc'].mean()
    last_patch_mean = df[df['inference_mode'] == 'last_patch']['roc_auc'].mean()
    md += f"**Observation:** all_patches achieves {all_patches_mean:.4f} vs last_patch {last_patch_mean:.4f}.\n\n"
    md += "**Hypothesis:** all_patches provides more robust aggregation of patch-level information.\n\n"

    md += "### Insight 4: Scoring Mode Effects\n\n"
    default_mean = df[df['scoring_mode'] == 'default']['roc_auc'].mean()
    adaptive_mean = df[df['scoring_mode'] == 'adaptive']['roc_auc'].mean()
    normalized_mean = df[df['scoring_mode'] == 'normalized']['roc_auc'].mean()
    md += f"**Observation:** default ({default_mean:.4f}) > adaptive ({adaptive_mean:.4f}) > normalized ({normalized_mean:.4f}).\n\n"
    md += "**Hypothesis:** Default scoring (simple averaging) works best for this dataset.\n\n"

    md += "### Insight 5: High t_ratio Models\n\n"
    top_t = df.nlargest(50, 't_ratio')
    md += f"**Observation:** Models with high t_ratio (top 50) average ROC-AUC: {top_t['roc_auc'].mean():.4f}.\n\n"
    md += "**Hypothesis:** Teacher reconstruction ratio is a strong indicator of model quality.\n\n"

    return md


def generate_phase2_recommendations(df: pd.DataFrame) -> str:
    """Generate recommendations for Phase 2."""

    md = "## Phase 2 Experiment Recommendations\n\n"

    md += "### Priority 1: Maximize Disc + Recon Cohen's d (30 experiments)\n\n"
    md += "**Goal:** Find parameter combinations that maximize both disc_cohens_d and recon_cohens_d.\n\n"

    # Find top performers
    top_both = df[
        (df['disc_cohens_d_normal_vs_anomaly'] > df['disc_cohens_d_normal_vs_anomaly'].quantile(0.80)) &
        (df['recon_cohens_d_normal_vs_anomaly'] > df['recon_cohens_d_normal_vs_anomaly'].quantile(0.80))
    ]

    md += f"- Base on top {len(top_both)} models from Phase 1\n"
    md += f"- Focus on all_patches inference mode ({len(top_both[top_both['inference_mode']=='all_patches'])} of top use this)\n"
    md += f"- Focus on default scoring mode\n"
    md += "- Vary: d_model, masking_ratio, decoder_depth\n\n"

    md += "### Priority 2: Disturbing Normal Separation (25 experiments)\n\n"
    md += "**Goal:** Improve disc_cohens_d_disturbing_vs_anomaly while maintaining high ROC-AUC.\n\n"
    md += "- Focus on models with high disc_ratio_2\n"
    md += "- Experiment with different patch sizes\n"
    md += "- Test window size variations (500, 1000)\n\n"

    md += "### Priority 3: PA%80 Optimization (25 experiments)\n\n"
    md += "**Goal:** Maximize PA%80 performance for practical deployment.\n\n"
    md += "- Start from high PA%80 + disc_ratio models\n"
    md += "- Test scoring mode combinations\n"
    md += "- Experiment with ensemble approaches\n\n"

    md += "### Priority 4: Window Size + Depth Relationships (20 experiments)\n\n"
    md += "**Goal:** Understand optimal model capacity for different window sizes.\n\n"
    md += "- Window 500: vary decoder_depth (2, 3, 4, 5, 6)\n"
    md += "- Window 500: vary d_model (96, 128, 192, 256, 320)\n"
    md += "- Window 1000: test if larger capacity helps\n\n"

    md += "### Priority 5: Mask After + Lambda Optimization (20 experiments)\n\n"
    md += "**Goal:** Optimize discrepancy loss weighting for mask_after models.\n\n"
    md += "- Test lambda_disc values: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0\n"
    md += "- Combine with high-performing base configurations\n\n"

    md += "### Priority 6: Teacher-Student Ratio Exploration (15 experiments)\n\n"
    md += "**Goal:** Find optimal teacher-student loss balance.\n\n"
    md += "- Test ratios: t1s1, t2s1, t3s1, t4s1, t5s1\n"
    md += "- Test ratios: t2s2, t3s2, t4s2\n\n"

    md += "### Priority 7: Masking Ratio Fine-tuning (15 experiments)\n\n"
    md += "**Goal:** Find optimal masking ratio for different model sizes.\n\n"
    md += "- d_model=128: test masking [0.05, 0.10, 0.15, 0.20, 0.25]\n"
    md += "- d_model=256: test masking [0.60, 0.70, 0.75, 0.80, 0.85]\n\n"

    return md


def main():
    """Generate the comprehensive Phase 1 report."""

    print("Loading data...")
    df = load_data()

    print("Generating report...")

    # Initialize report
    report = f"# Phase 1 Ablation Study Analysis Report\n\n"
    report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += f"**Total Experiments:** {len(df)}\n\n"

    # Add overview
    report += "## Overview\n\n"
    report += f"- Total experiments: {len(df)}\n"
    report += f"- Inference modes: {df['inference_mode'].nunique()} ({', '.join(df['inference_mode'].unique())})\n"
    report += f"- Scoring modes: {df['scoring_mode'].nunique()} ({', '.join(df['scoring_mode'].unique())})\n"
    report += f"- ROC-AUC range: {df['roc_auc'].min():.4f} - {df['roc_auc'].max():.4f}\n"
    report += f"- Average ROC-AUC: {df['roc_auc'].mean():.4f} ± {df['roc_auc'].std():.4f}\n\n"

    # Add tables
    report += generate_tables_section(df)

    # Add focus area analysis
    report += analyze_focus_areas(df)

    # Add insights
    report += generate_insights(df)

    # Add Phase 2 recommendations
    report += generate_phase2_recommendations(df)

    # Save report
    output_dir = Path("docs/ablation_result")
    output_file = output_dir / "phase1_analysis_report.md"

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {output_file}")
    print(f"Report length: {len(report)} characters")


if __name__ == "__main__":
    main()
