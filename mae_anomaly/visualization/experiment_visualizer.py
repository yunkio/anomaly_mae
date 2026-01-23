"""
Experiment Visualizer - Stage 1 (Quick Search) Results Visualizations

This module provides visualizations for:
- Parameter pair heatmaps
- Parallel coordinates plot
- Parameter importance box plots
- Top k configurations comparison
- Metric distributions and correlations
- Summary dashboard
"""

import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from .base import VIS_COLORS


class ExperimentVisualizer:
    """Visualize Stage 1 (Quick Search) results"""

    def __init__(self, results_df: pd.DataFrame, param_keys: List[str], output_dir: str):
        self.results_df = results_df
        # Filter param_keys to only include columns that exist in the DataFrame
        self.param_keys = [p for p in param_keys if p in results_df.columns]
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _get_categorical_params(self) -> List[str]:
        """Dynamically detect categorical parameters from param_keys"""
        cat_params = []
        for param in self.param_keys:
            if param in self.results_df.columns:
                # Check if column is object/string type or has few unique values
                dtype = self.results_df[param].dtype
                if dtype == 'object' or dtype == 'bool' or self.results_df[param].nunique() <= 5:
                    cat_params.append(param)
        return cat_params

    def plot_heatmaps(self, metric: str = 'roc_auc'):
        """Generate heatmaps for parameter pairs"""
        numeric_params = [p for p in self.param_keys if self.results_df[p].dtype in ['float64', 'int64']]

        if len(numeric_params) < 2:
            print("  - Skipping heatmaps (insufficient numeric parameters)")
            return

        n_params = len(numeric_params)
        n_plots = n_params * (n_params - 1) // 2
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_plots > 1 else [axes]

        idx = 0
        for i, param1 in enumerate(numeric_params):
            for j, param2 in enumerate(numeric_params):
                if i >= j:
                    continue

                pivot = self.results_df.pivot_table(
                    values=metric, index=param1, columns=param2, aggfunc='mean'
                )

                ax = axes[idx]
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, cbar_kws={'label': metric})
                ax.set_title(f'{param1} vs {param2}')
                idx += 1

        # Hide unused axes
        for ax in axes[idx:]:
            ax.axis('off')

        plt.suptitle(f'Parameter Pair Heatmaps ({metric.upper()})', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'heatmaps_{metric}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  - heatmaps_{metric}.png")

    def plot_parallel_coordinates(self):
        """Generate parallel coordinates plot"""
        df = self.results_df.copy()

        # Select numeric columns for parallel coordinates
        numeric_cols = [c for c in self.param_keys if df[c].dtype in ['float64', 'int64']]
        if len(numeric_cols) < 2:
            print("  - Skipping parallel coordinates (insufficient numeric parameters)")
            return

        # Normalize columns
        df_norm = df.copy()
        for col in numeric_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                df_norm[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                df_norm[col] = 0.5

        fig, ax = plt.subplots(figsize=(12, 6))

        # Color by ROC-AUC
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(df['roc_auc'].min(), df['roc_auc'].max())

        for idx, row in df_norm.iterrows():
            values = [row[col] for col in numeric_cols]
            color = cmap(norm(df.loc[idx, 'roc_auc']))
            ax.plot(range(len(numeric_cols)), values, color=color, alpha=0.3, lw=0.8)

        # Highlight top 10
        top_10_idx = df.nlargest(10, 'roc_auc').index
        for idx in top_10_idx:
            values = [df_norm.loc[idx, col] for col in numeric_cols]
            ax.plot(range(len(numeric_cols)), values, color=VIS_COLORS['anomaly_region'], alpha=0.8, lw=2)

        ax.set_xticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Parallel Coordinates (colored by ROC-AUC, red=top 10)', fontsize=12, fontweight='bold')

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='ROC-AUC')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parallel_coordinates.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - parallel_coordinates.png")

    def plot_parameter_importance(self):
        """Generate parameter importance box plots"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        axes = axes.flatten()

        for idx, param in enumerate(self.param_keys):
            if idx >= len(axes):
                break

            ax = axes[idx]
            groups = self.results_df.groupby(param)['roc_auc'].apply(list).to_dict()

            labels = list(groups.keys())
            data = list(groups.values())

            bp = ax.boxplot(data, labels=[str(l)[:10] for l in labels], patch_artist=True)

            # Color boxes
            colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            ax.set_title(f'{param}', fontsize=11, fontweight='bold')
            ax.set_ylabel('ROC-AUC')
            ax.tick_params(axis='x', rotation=45)

        # Hide unused axes
        for ax in axes[len(self.param_keys):]:
            ax.axis('off')

        plt.suptitle('Parameter Importance (ROC-AUC Distribution)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_importance.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - parameter_importance.png")

    def plot_top_k_comparison(self, k: int = 10):
        """Plot top k configurations comparison"""
        actual_k = min(k, len(self.results_df))
        top_k = self.results_df.nlargest(actual_k, 'roc_auc')

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Bar chart
        ax = axes[0]
        x = np.arange(actual_k)
        width = 0.35

        bars1 = ax.bar(x - width/2, top_k['roc_auc'], width, label='ROC-AUC', color=VIS_COLORS['normal'])
        bars2 = ax.bar(x + width/2, top_k['f1_score'], width, label='F1-Score', color=VIS_COLORS['anomaly'])

        ax.set_xlabel('Configuration Rank')
        ax.set_ylabel('Score')
        ax.set_title(f'Top {actual_k} Configurations', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'#{i+1}' for i in range(actual_k)])
        ax.legend()
        ax.set_ylim(0, 1.1)

        # Add values
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

        # Table
        ax = axes[1]
        ax.axis('off')

        table_data = []
        for i, (_, row) in enumerate(top_k.iterrows()):
            config_str = ', '.join([f"{p}={row[p]}" for p in self.param_keys[:4]])
            table_data.append([
                f'#{i+1}',
                f'{row["roc_auc"]:.4f}',
                f'{row["f1_score"]:.4f}',
                config_str[:50] + '...' if len(config_str) > 50 else config_str
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=['Rank', 'ROC-AUC', 'F1', 'Configuration'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.set_title(f'Top {k} Configuration Details', fontsize=12, fontweight='bold', y=0.95)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'top_k_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - top_k_comparison.png")

    def plot_metric_distributions(self):
        """Plot metric distributions"""
        metrics = ['roc_auc', 'f1_score', 'precision', 'recall']
        available_metrics = [m for m in metrics if m in self.results_df.columns]

        fig, axes = plt.subplots(1, len(available_metrics), figsize=(4*len(available_metrics), 4))
        if len(available_metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, available_metrics):
            data = self.results_df[metric].dropna()
            ax.hist(data, bins=30, color=VIS_COLORS['normal'], edgecolor='black', alpha=0.7)
            ax.axvline(data.mean(), color=VIS_COLORS['anomaly_region'], linestyle='--', lw=2, label=f'Mean: {data.mean():.4f}')
            ax.axvline(data.median(), color=VIS_COLORS['threshold'], linestyle='--', lw=2, label=f'Median: {data.median():.4f}')
            ax.set_title(metric.upper(), fontsize=11, fontweight='bold')
            ax.set_xlabel('Score')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)

        plt.suptitle('Metric Distributions', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metric_distributions.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - metric_distributions.png")

    def plot_metric_correlations(self):
        """Plot metric correlations"""
        metrics = ['roc_auc', 'f1_score', 'precision', 'recall']
        if 'disturbing_roc_auc' in self.results_df.columns:
            metrics.append('disturbing_roc_auc')

        available_metrics = [m for m in metrics if m in self.results_df.columns]
        corr = self.results_df[available_metrics].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                   vmin=-1, vmax=1, center=0)
        ax.set_title('Metric Correlations', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metric_correlations.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - metric_correlations.png")

    def plot_categorical_comparison(self, param: str, metric: str = 'roc_auc'):
        """Plot comparison for a categorical parameter"""
        if param not in self.results_df.columns:
            return

        groups = self.results_df.groupby(param)[metric]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Bar chart with error bars
        ax = axes[0]
        means = groups.mean()
        stds = groups.std()
        x = range(len(means))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=plt.cm.Set2(np.linspace(0, 1, len(means))))
        ax.set_xticks(x)
        ax.set_xticklabels(means.index, rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(f'{param} Mean {metric} (with std)', fontsize=11, fontweight='bold')

        # Box plot
        ax = axes[1]
        data = [self.results_df[self.results_df[param] == v][metric].values for v in means.index]
        bp = ax.boxplot(data, labels=means.index, patch_artist=True)
        for patch, color in zip(bp['boxes'], plt.cm.Set2(np.linspace(0, 1, len(means)))):
            patch.set_facecolor(color)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel(metric)
        ax.set_title(f'{param} Distribution', fontsize=11, fontweight='bold')

        # Violin plot
        ax = axes[2]
        positions = range(len(means.index))
        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
        ax.set_xticks(positions)
        ax.set_xticklabels(means.index, rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(f'{param} Violin Plot', fontsize=11, fontweight='bold')

        plt.suptitle(f'{param} Comparison ({metric})', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{param}_comparison_{metric}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  - {param}_comparison_{metric}.png")

    def plot_summary_dashboard(self):
        """Create a comprehensive summary dashboard"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. ROC-AUC distribution
        ax = fig.add_subplot(gs[0, 0])
        ax.hist(self.results_df['roc_auc'], bins=30, color=VIS_COLORS['normal'], edgecolor='black', alpha=0.7)
        ax.axvline(self.results_df['roc_auc'].mean(), color=VIS_COLORS['anomaly_region'], linestyle='--', lw=2)
        ax.set_title('ROC-AUC Distribution', fontweight='bold')
        ax.set_xlabel('ROC-AUC')
        ax.set_ylabel('Count')

        # 2. Top 10 bar chart
        ax = fig.add_subplot(gs[0, 1])
        top_n = min(10, len(self.results_df))
        top_10 = self.results_df.nlargest(top_n, 'roc_auc')
        ax.barh(range(top_n), top_10['roc_auc'].values[::-1], color=VIS_COLORS['teacher'])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f'#{i+1}' for i in range(top_n-1, -1, -1)])
        ax.set_xlabel('ROC-AUC')
        ax.set_title('Top 10 Configurations', fontweight='bold')

        # 3. Summary statistics
        ax = fig.add_subplot(gs[0, 2])
        ax.axis('off')
        stats_text = f"""
        Total Configurations: {len(self.results_df)}

        ROC-AUC:
          Best: {self.results_df['roc_auc'].max():.4f}
          Mean: {self.results_df['roc_auc'].mean():.4f}
          Std: {self.results_df['roc_auc'].std():.4f}

        F1-Score:
          Best: {self.results_df['f1_score'].max():.4f}
          Mean: {self.results_df['f1_score'].mean():.4f}
        """
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Summary Statistics', fontweight='bold')

        # 4-6. Parameter importance for first 3 numeric params
        numeric_params = [p for p in self.param_keys if self.results_df[p].dtype in ['float64', 'int64']][:3]
        for i, param in enumerate(numeric_params):
            ax = fig.add_subplot(gs[1, i])
            groups = self.results_df.groupby(param)['roc_auc'].apply(list).to_dict()
            bp = ax.boxplot(list(groups.values()), labels=[str(k) for k in groups.keys()], patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(VIS_COLORS['normal'])
            ax.set_title(f'{param}', fontweight='bold')
            ax.set_ylabel('ROC-AUC')

        # 7-9. Categorical params (dynamically detected)
        cat_params = self._get_categorical_params()[:3]  # Take first 3 for the grid
        cat_idx = 0
        for param in cat_params:
            if param in self.results_df.columns and cat_idx < 3:
                ax = fig.add_subplot(gs[2, cat_idx])
                groups = self.results_df.groupby(param)['roc_auc']
                means = groups.mean()
                stds = groups.std()
                colors = plt.cm.Set2(np.linspace(0, 1, len(means)))
                ax.bar(range(len(means)), means, yerr=stds, capsize=5, color=colors)
                ax.set_xticks(range(len(means)))
                ax.set_xticklabels(means.index, rotation=45, ha='right')
                ax.set_title(f'{param}', fontweight='bold')
                ax.set_ylabel('ROC-AUC')
                cat_idx += 1

        plt.suptitle('Stage 1 Summary Dashboard', fontsize=16, fontweight='bold', y=1.02)
        plt.savefig(os.path.join(self.output_dir, 'stage1_summary_dashboard.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - stage1_summary_dashboard.png")

    def generate_all(self):
        """Generate all Stage 1 visualizations"""
        print("\n  Generating Stage 1 Visualizations...")
        self.plot_heatmaps()
        self.plot_parallel_coordinates()
        self.plot_parameter_importance()
        self.plot_top_k_comparison()
        self.plot_metric_distributions()
        self.plot_metric_correlations()

        # Categorical comparisons (dynamically detected from param_keys)
        cat_params = self._get_categorical_params()
        for param in cat_params[:2]:  # Plot first 2 categorical params
            self.plot_categorical_comparison(param)

        self.plot_summary_dashboard()
