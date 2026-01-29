"""
Stage 2 Visualizer - Full Training Results Visualizations

This module provides visualizations for:
- Quick search vs full training comparison
- Selection criterion analysis
- Learning curves
- Hyperparameter impact analysis
- Summary dashboard
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from .base import VIS_COLORS


class Stage2Visualizer:
    """Visualize Stage 2 (Full Training) results"""

    # Known metric columns (not hyperparameters)
    METRIC_COLUMNS = {
        'combination_id', 'roc_auc', 'f1_score', 'precision', 'recall',
        'disturbing_roc_auc', 'disturbing_f1', 'disturbing_precision', 'disturbing_recall',
        'quick_roc_auc', 'quick_f1', 'quick_disturbing_roc_auc',
        'roc_auc_improvement', 'selection_criterion', 'stage2_rank'
    }

    def __init__(self, full_df: pd.DataFrame, quick_df: pd.DataFrame,
                 histories: Dict, output_dir: str):
        self.full_df = full_df
        self.quick_df = quick_df
        self.histories = histories
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _get_hyperparam_columns(self) -> list:
        """Dynamically get hyperparameter columns from DataFrame"""
        return [c for c in self.full_df.columns if c not in self.METRIC_COLUMNS]

    def plot_quick_vs_full(self):
        """Compare quick search vs full training performance"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Scatter plot
        ax = axes[0]
        ax.scatter(self.full_df['quick_roc_auc'], self.full_df['roc_auc'],
                  alpha=0.6, s=50, c=VIS_COLORS['normal'])

        # Diagonal line
        min_val = min(self.full_df['quick_roc_auc'].min(), self.full_df['roc_auc'].min())
        max_val = max(self.full_df['quick_roc_auc'].max(), self.full_df['roc_auc'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')

        ax.set_xlabel('Quick Search ROC-AUC')
        ax.set_ylabel('Full Training ROC-AUC')
        ax.set_title('Quick vs Full Performance', fontsize=12, fontweight='bold')
        ax.legend()

        # Improvement distribution
        ax = axes[1]
        improvements = self.full_df['roc_auc'] - self.full_df['quick_roc_auc']
        ax.hist(improvements, bins=30, color=VIS_COLORS['teacher'], edgecolor=VIS_COLORS['baseline'], alpha=0.7)
        ax.axvline(improvements.mean(), color=VIS_COLORS['anomaly_region'], linestyle='--', lw=2,
                  label=f'Mean: {improvements.mean():.4f}')
        ax.axvline(0, color=VIS_COLORS['baseline'], linestyle='-', lw=1)
        ax.set_xlabel('ROC-AUC Improvement')
        ax.set_ylabel('Count')
        ax.set_title('Improvement Distribution', fontsize=12, fontweight='bold')
        ax.legend()

        # Top 10 comparison
        ax = axes[2]
        top_n = min(10, len(self.full_df))
        top_10_full = self.full_df.nlargest(top_n, 'roc_auc')

        # Check if roc_auc_improvement column exists
        if 'roc_auc_improvement' in self.full_df.columns:
            top_10_improve = self.full_df.nlargest(top_n, 'roc_auc_improvement')
        else:
            top_10_improve = top_10_full  # Fallback

        x = np.arange(top_n)
        width = 0.35

        ax.bar(x - width/2, top_10_full['roc_auc'].values, width,
              label='Top by Final ROC-AUC', color=VIS_COLORS['normal'])
        ax.bar(x + width/2, top_10_improve['roc_auc'].values, width,
              label='Top by Improvement', color=VIS_COLORS['anomaly'])

        ax.set_xlabel('Rank')
        ax.set_ylabel('Final ROC-AUC')
        ax.set_title(f'Top {top_n} by Different Criteria', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'#{i+1}' for i in range(top_n)])
        ax.legend()

        plt.suptitle('Quick Search vs Full Training Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'stage2_quick_vs_full.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - stage2_quick_vs_full.png")

    def plot_selection_criterion_analysis(self):
        """Analyze performance by selection criterion"""
        if 'selection_criterion' not in self.full_df.columns:
            print("  - Skipping selection criterion analysis (column not found)")
            return

        criteria_groups = self.full_df.groupby('selection_criterion')['roc_auc']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Box plot
        ax = axes[0]
        criteria_data = {name: group.values for name, group in criteria_groups}
        bp = ax.boxplot(criteria_data.values(), labels=criteria_data.keys(), patch_artist=True)

        colors = plt.cm.Set3(np.linspace(0, 1, len(criteria_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel('ROC-AUC')
        ax.set_title('Performance by Selection Criterion', fontsize=12, fontweight='bold')

        # Bar chart of counts and mean performance
        ax = axes[1]
        means = criteria_groups.mean()
        counts = criteria_groups.count()

        x = np.arange(len(means))
        width = 0.4

        ax2 = ax.twinx()

        bars1 = ax.bar(x - width/2, means, width, label='Mean ROC-AUC', color=VIS_COLORS['normal'])
        bars2 = ax2.bar(x + width/2, counts, width, label='Count', color=VIS_COLORS['anomaly'], alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(means.index, rotation=45, ha='right')
        ax.set_ylabel('Mean ROC-AUC', color=VIS_COLORS['normal'])
        ax2.set_ylabel('Count', color=VIS_COLORS['anomaly'])
        ax.set_title('Selection Criteria Statistics', fontsize=12, fontweight='bold')

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'stage2_selection_criterion.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - stage2_selection_criterion.png")

    def plot_learning_curves(self, top_k: int = 10):
        """Plot learning curves for top experiments"""
        if not self.histories:
            print("  - Skipping learning curves (no history data)")
            return

        # Get top k experiment IDs
        top_k_df = self.full_df.nlargest(top_k, 'roc_auc')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Training loss
        ax = axes[0]
        for _, row in top_k_df.iterrows():
            exp_id = str(int(row['combination_id']))
            if exp_id in self.histories:
                history = self.histories[exp_id]
                if 'train_loss' in history:
                    ax.plot(history['train_loss'], alpha=0.7,
                           label=f"#{int(row.get('stage2_rank', 0))} (ROC={row['roc_auc']:.3f})")

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss Curves (Top 10)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')

        # Validation loss if available
        ax = axes[1]
        has_val = False
        for _, row in top_k_df.iterrows():
            exp_id = str(int(row['combination_id']))
            if exp_id in self.histories:
                history = self.histories[exp_id]
                if 'val_loss' in history:
                    ax.plot(history['val_loss'], alpha=0.7,
                           label=f"#{int(row.get('stage2_rank', 0))}")
                    has_val = True

        if has_val:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Validation Loss')
            ax.set_title('Validation Loss Curves (Top 10)', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8, loc='upper right')
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No validation loss data available',
                   ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'learning_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - learning_curves.png")

    def plot_summary_dashboard(self):
        """Create Stage 2 summary dashboard"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Quick vs Full scatter
        ax = fig.add_subplot(gs[0, 0])
        ax.scatter(self.full_df['quick_roc_auc'], self.full_df['roc_auc'],
                  alpha=0.6, s=50, c=VIS_COLORS['normal'])
        min_val = min(self.full_df['quick_roc_auc'].min(), self.full_df['roc_auc'].min())
        max_val = max(self.full_df['quick_roc_auc'].max(), self.full_df['roc_auc'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        ax.set_xlabel('Quick Search ROC-AUC')
        ax.set_ylabel('Full Training ROC-AUC')
        ax.set_title('Quick vs Full Performance', fontweight='bold')

        # 2. Improvement histogram
        ax = fig.add_subplot(gs[0, 1])
        improvements = self.full_df['roc_auc'] - self.full_df['quick_roc_auc']
        ax.hist(improvements, bins=20, color=VIS_COLORS['teacher'], edgecolor=VIS_COLORS['baseline'], alpha=0.7)
        ax.axvline(improvements.mean(), color=VIS_COLORS['anomaly_region'], linestyle='--', lw=2)
        ax.axvline(0, color=VIS_COLORS['baseline'], linestyle='-', lw=1)
        ax.set_xlabel('ROC-AUC Improvement')
        ax.set_ylabel('Count')
        ax.set_title(f'Improvement (mean={improvements.mean():.4f})', fontweight='bold')

        # 3. Top 10 table
        ax = fig.add_subplot(gs[1, 0])
        ax.axis('off')
        top_10 = self.full_df.nlargest(10, 'roc_auc')

        table_data = []
        for i, (_, row) in enumerate(top_10.iterrows()):
            table_data.append([
                f'#{i+1}',
                f'{row["roc_auc"]:.4f}',
                f'{row["quick_roc_auc"]:.4f}',
                f'{row["roc_auc"] - row["quick_roc_auc"]:+.4f}',
                str(row.get('selection_criterion', 'N/A'))[:20]
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=['Rank', 'Final ROC', 'Quick ROC', 'Improve', 'Criterion'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.set_title('Top 10 Stage 2 Results', fontweight='bold', y=0.95)

        # 4. Statistics
        ax = fig.add_subplot(gs[1, 1])
        ax.axis('off')

        stats_text = f"""
        Stage 2 Summary

        Total Models Trained: {len(self.full_df)}

        Final ROC-AUC:
          Best:   {self.full_df['roc_auc'].max():.4f}
          Mean:   {self.full_df['roc_auc'].mean():.4f}
          Std:    {self.full_df['roc_auc'].std():.4f}

        Improvement from Quick Search:
          Mean:   {improvements.mean():+.4f}
          Max:    {improvements.max():+.4f}
          Min:    {improvements.min():+.4f}

        Models Improved: {(improvements > 0).sum()} / {len(improvements)}
        """

        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.suptitle('Stage 2 Summary Dashboard', fontsize=16, fontweight='bold', y=1.02)
        plt.savefig(os.path.join(self.output_dir, 'stage2_summary_dashboard.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - stage2_summary_dashboard.png")

    def plot_hyperparameter_impact(self, param: str, metric: str = 'roc_auc'):
        """Plot detailed impact of a single hyperparameter on Stage 2 results"""
        if param not in self.full_df.columns:
            print(f"  ! Skipping {param} impact (column not found)")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        groups = self.full_df.groupby(param)

        # 1. Box plot of final ROC-AUC
        ax = axes[0, 0]
        group_data = {str(k): v[metric].values for k, v in groups}
        bp = ax.boxplot(group_data.values(), labels=group_data.keys(), patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(group_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_xlabel(param)
        ax.set_ylabel(metric)
        ax.set_title(f'{param} vs {metric.upper()} (Full Training)', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # 2. Mean + Std comparison
        ax = axes[0, 1]
        means = groups[metric].mean()
        stds = groups[metric].std()
        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor=VIS_COLORS['baseline'])
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in means.index], rotation=45, ha='right')
        ax.set_xlabel(param)
        ax.set_ylabel(f'Mean {metric}')
        ax.set_title(f'{param} Mean +/- Std', fontweight='bold')

        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                   f'{mean:.4f}', ha='center', va='bottom', fontsize=8)

        # 3. Improvement analysis (Quick -> Full)
        ax = axes[1, 0]
        improvements = self.full_df.groupby(param).apply(
            lambda x: (x['roc_auc'] - x['quick_roc_auc']).mean()
        )
        colors_imp = [VIS_COLORS['true_positive'] if imp > 0 else VIS_COLORS['false_negative'] for imp in improvements]
        bars = ax.bar(range(len(improvements)), improvements, color=colors_imp, alpha=0.8, edgecolor=VIS_COLORS['baseline'])
        ax.set_xticks(range(len(improvements)))
        ax.set_xticklabels([str(k) for k in improvements.index], rotation=45, ha='right')
        ax.axhline(y=0, color=VIS_COLORS['baseline'], linestyle='-', lw=1)
        ax.set_xlabel(param)
        ax.set_ylabel('Mean Improvement')
        ax.set_title(f'{param} Mean Improvement (Quick -> Full)', fontweight='bold')

        for bar, imp in zip(bars, improvements):
            y_pos = bar.get_height() + 0.002 if imp >= 0 else bar.get_height() - 0.01
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{imp:+.4f}', ha='center', va='bottom' if imp >= 0 else 'top', fontsize=8)

        # 4. Statistics table
        ax = axes[1, 1]
        ax.axis('off')

        table_data = []
        for param_val in means.index:
            param_data = self.full_df[self.full_df[param] == param_val]
            table_data.append([
                str(param_val),
                f"{param_data[metric].mean():.4f}",
                f"{param_data[metric].std():.4f}",
                f"{param_data[metric].max():.4f}",
                f"{param_data[metric].min():.4f}",
                f"{len(param_data)}"
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=[param, 'Mean', 'Std', 'Max', 'Min', 'Count'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        ax.set_title(f'{param} Statistics', fontweight='bold', y=0.95)

        plt.suptitle(f'Hyperparameter Impact: {param}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save to hyperparameter-specific file
        plt.savefig(os.path.join(self.output_dir, f'hyperparam_{param}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  - hyperparam_{param}.png")

    def plot_all_hyperparameters(self):
        """Generate separate visualization for each hyperparameter"""
        # Dynamically get hyperparameters from DataFrame columns
        hyperparams = self._get_hyperparam_columns()

        for param in hyperparams:
            if param in self.full_df.columns:
                self.plot_hyperparameter_impact(param)

    def plot_hyperparameter_interactions(self):
        """Plot interactions between key hyperparameters"""
        # Dynamically generate interaction pairs from available hyperparameters
        hyperparams = self._get_hyperparam_columns()

        # Generate pairs (up to 6 pairs for 2x3 grid)
        from itertools import combinations
        all_pairs = list(combinations(hyperparams, 2))
        interactions = all_pairs[:6]  # Take first 6 pairs

        # Adjust grid size based on number of interactions
        n_interactions = len(interactions)
        n_cols = min(3, n_interactions)
        n_rows = (n_interactions + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
        if n_interactions == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, (param1, param2) in enumerate(interactions):
            ax = axes[idx // n_cols, idx % n_cols]

            if param1 not in self.full_df.columns or param2 not in self.full_df.columns:
                ax.text(0.5, 0.5, f'{param1} or {param2} not found',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{param1} x {param2}', fontweight='bold')
                continue

            # Create pivot table
            try:
                pivot = self.full_df.pivot_table(
                    values='roc_auc', index=param1, columns=param2, aggfunc='mean'
                )
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, cbar_kws={'label': 'ROC-AUC'})
                ax.set_title(f'{param1} x {param2}', fontweight='bold')
            except Exception as e:
                ax.text(0.5, 0.5, f'Cannot create heatmap:\n{str(e)[:30]}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{param1} x {param2}', fontweight='bold')

        plt.suptitle('Hyperparameter Interactions (Stage 2 ROC-AUC)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'hyperparameter_interactions.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - hyperparameter_interactions.png")

    def plot_best_config_summary(self):
        """Visualize the best configuration details"""
        best_row = self.full_df.loc[self.full_df['roc_auc'].idxmax()]

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')

        # Hyperparameter descriptions (fallback for unknown params)
        hyperparam_info = {
            'masking_ratio': ('Masking Ratio', 'Ratio of patches to mask'),
            'masking_strategy': ('Masking Strategy', 'patch or feature_wise'),
            'num_patches': ('Num Patches', 'Number of patches'),
            'margin_type': ('Margin Type', 'Loss calculation method'),
            'force_mask_anomaly': ('Force Mask Anomaly', 'Force masking anomaly regions'),
            'patch_level_loss': ('Patch Level Loss', 'Compute loss per patch'),
            'patchify_mode': ('Patchify Mode', 'CNN/Linear patch embedding'),
        }

        # Dynamically get hyperparameters from DataFrame
        hyperparams = []
        for param in self._get_hyperparam_columns():
            if param in hyperparam_info:
                name, desc = hyperparam_info[param]
            else:
                # Auto-generate name and description for unknown params
                name = param.replace('_', ' ').title()
                desc = 'Hyperparameter'
            hyperparams.append((param, name, desc))

        # Build summary text
        summary_lines = [
            "BEST MODEL CONFIGURATION",
            "=" * 60,
            "",
        ]

        for param, name, desc in hyperparams:
            if param in best_row.index:
                value = best_row[param]
                if isinstance(value, float):
                    value_str = f"{value:.4f}" if value < 1 else f"{value:.2f}"
                else:
                    value_str = str(value)
                summary_lines.append(f"  {name:<25} = {value_str:<12} ({desc})")

        summary_lines.append("")
        summary_lines.append("=" * 60)
        summary_lines.append("PERFORMANCE METRICS")
        summary_lines.append("=" * 60)
        summary_lines.append("")

        metrics = ['roc_auc', 'f1_score', 'precision', 'recall', 'quick_roc_auc', 'roc_auc_improvement']
        metric_names = ['ROC-AUC', 'F1-Score', 'Precision', 'Recall', 'Quick ROC-AUC', 'Improvement']

        for metric, name in zip(metrics, metric_names):
            if metric in best_row.index:
                value = best_row[metric]
                if pd.notna(value):
                    summary_lines.append(f"  {name:<25} = {value:.4f}")

        summary_text = "\n".join(summary_lines)

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        plt.savefig(os.path.join(self.output_dir, 'best_config_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_config_summary.png")

    def generate_all(self):
        """Generate all Stage 2 visualizations"""
        print("\n  Generating Stage 2 Visualizations...")
        self.plot_quick_vs_full()
        self.plot_selection_criterion_analysis()
        self.plot_learning_curves()
        self.plot_summary_dashboard()
        self.plot_all_hyperparameters()
        self.plot_hyperparameter_interactions()
        self.plot_best_config_summary()
