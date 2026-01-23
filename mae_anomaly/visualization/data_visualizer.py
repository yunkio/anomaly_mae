"""
Data Visualizer - Dataset and Anomaly Type Visualizations

This module provides visualizations for:
- Anomaly types and their characteristics
- Sample types (pure normal, disturbing normal, anomaly)
- Feature examples and correlations
- Dataset statistics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mae_anomaly import (
    Config, set_seed,
    SlidingWindowTimeSeriesGenerator, SlidingWindowDataset,
    NormalDataComplexity,
    ANOMALY_TYPE_NAMES, FEATURE_NAMES,
)
from mae_anomaly.dataset_sliding import ANOMALY_TYPE_CONFIGS

from .base import (
    get_anomaly_colors, get_anomaly_type_info,
    SAMPLE_TYPE_NAMES, SAMPLE_TYPE_COLORS, VIS_COLORS,
)


class DataVisualizer:
    """Visualize dataset characteristics and anomaly types"""

    def __init__(self, output_dir: str, config: Config = None):
        self.output_dir = output_dir
        self.config = config or Config()
        os.makedirs(output_dir, exist_ok=True)

    def plot_anomaly_types(self):
        """Visualize different anomaly types in the dataset"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Generate example sequences for each anomaly type
        np.random.seed(42)
        seq_length = 100
        t = np.linspace(0, 4*np.pi, seq_length)

        # 1. Normal - clean sinusoid
        normal = np.sin(t) + 0.1 * np.random.randn(seq_length)
        ax = axes[0, 0]
        ax.plot(t, normal, 'b-', lw=1.5)
        ax.set_title('Normal Pattern', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_ylim(-3, 3)

        # 2. Point anomaly - spike
        point_anomaly = normal.copy()
        point_anomaly[70:75] = 2.5
        ax = axes[0, 1]
        ax.plot(t, point_anomaly, 'b-', lw=1.5)
        ax.axvspan(t[70], t[74], alpha=0.3, color=VIS_COLORS['anomaly_region'])
        ax.scatter(t[70:75], point_anomaly[70:75], c='red', s=50, zorder=5)
        ax.set_title('Point Anomaly (Spike)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_ylim(-3, 3)

        # 3. Contextual anomaly - level shift
        contextual = normal.copy()
        contextual[60:] += 1.0
        ax = axes[0, 2]
        ax.plot(t, contextual, 'b-', lw=1.5)
        ax.axvspan(t[60], t[-1], alpha=0.3, color=VIS_COLORS['anomaly_region'])
        ax.set_title('Contextual Anomaly (Level Shift)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_ylim(-3, 3)

        # 4. Collective anomaly - unusual pattern
        collective = normal.copy()
        collective[50:80] = 0.5 * np.sin(3*t[50:80]) + np.sin(10*t[50:80]) * 0.3
        ax = axes[1, 0]
        ax.plot(t, collective, 'b-', lw=1.5)
        ax.axvspan(t[50], t[79], alpha=0.3, color=VIS_COLORS['anomaly_region'])
        ax.set_title('Collective Anomaly (Unusual Pattern)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_ylim(-3, 3)

        # 5. Frequency anomaly
        freq_anomaly = normal.copy()
        freq_anomaly[40:70] = np.sin(4*t[40:70]) + 0.1 * np.random.randn(30)
        ax = axes[1, 1]
        ax.plot(t, freq_anomaly, 'b-', lw=1.5)
        ax.axvspan(t[40], t[69], alpha=0.3, color=VIS_COLORS['anomaly_region'])
        ax.set_title('Frequency Anomaly', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_ylim(-3, 3)

        # 6. Trend anomaly
        trend_anomaly = normal.copy()
        trend_anomaly[50:] += np.linspace(0, 1.5, 50)
        ax = axes[1, 2]
        ax.plot(t, trend_anomaly, 'b-', lw=1.5)
        ax.axvspan(t[50], t[-1], alpha=0.3, color=VIS_COLORS['anomaly_region'])
        ax.set_title('Trend Anomaly (Drift)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_ylim(-3, 3)

        plt.suptitle('Types of Anomalies in Time Series Data', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'anomaly_types.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - anomaly_types.png")

    def plot_sample_types(self):
        """Visualize different sample types (normal, disturbing normal, anomaly)"""
        # Create sample dataset using sliding window
        set_seed(42)
        generator = SlidingWindowTimeSeriesGenerator(
            total_length=100000,  # Larger dataset for better diversity
            num_features=self.config.num_features,  # Use config value (must be >= 2 for point_spike)
            interval_scale=1.5,
            seed=42
        )
        signals, point_labels, anomaly_regions = generator.generate()
        dataset = SlidingWindowDataset(
            signals=signals,
            point_labels=point_labels,
            anomaly_regions=anomaly_regions,
            window_size=self.config.seq_length,
            stride=11,
            mask_last_n=10,
            split='test',
            train_ratio=0.5,
            seed=42
        )

        # Collect ALL samples by type first
        normal_samples_all = []
        disturbing_samples_all = []  # (seq, point_labels)
        anomaly_samples_all = []

        for i in range(len(dataset)):
            seq, label, point_labels, sample_type, _ = dataset[i]
            if sample_type == 0:  # pure normal
                normal_samples_all.append(seq[:, 0].numpy())
            elif sample_type == 1:  # disturbing normal
                disturbing_samples_all.append((seq[:, 0].numpy(), point_labels.numpy()))
            else:  # anomaly
                anomaly_samples_all.append(seq[:, 0].numpy())

        # Shuffle and select diverse samples (avoid stride-10 overlap)
        np.random.seed(42)

        # Select samples with sufficient spacing to show diversity
        def select_diverse(samples, n=5, min_spacing=10):
            """Select n diverse samples with minimum spacing in the original order"""
            if len(samples) <= n:
                return samples
            # Shuffle indices and pick
            indices = np.random.permutation(len(samples))[:n * min_spacing:min_spacing]
            if len(indices) < n:
                indices = np.random.choice(len(samples), n, replace=False)
            return [samples[i] for i in indices[:n]]

        normal_samples = select_diverse(normal_samples_all)
        disturbing_samples = select_diverse(disturbing_samples_all)
        anomaly_samples = select_diverse(anomaly_samples_all)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        x = np.arange(self.config.seq_length)

        # Normal samples
        ax = axes[0]
        for i, seq in enumerate(normal_samples[:5]):
            ax.plot(x, seq, alpha=0.7, label=f'Sample {i+1}')
        ax.set_title('Pure Normal Samples', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.axvspan(x[-10], x[-1], alpha=0.2, color=VIS_COLORS['masked_region'], label='Last Patch')
        ax.legend(fontsize=8)

        # Disturbing normal samples - show anomaly regions
        ax = axes[1]
        for i, (seq, point_labels) in enumerate(disturbing_samples[:5]):
            line, = ax.plot(x, seq, alpha=0.7, label=f'Sample {i+1}')
            # Find and highlight anomaly regions in this sample
            anomaly_mask = point_labels > 0
            if anomaly_mask.any():
                # Find contiguous anomaly regions
                diff = np.diff(np.concatenate([[0], anomaly_mask.astype(int), [0]]))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                for start, end in zip(starts, ends):
                    ax.axvspan(start, end - 1, alpha=0.15, color=line.get_color())
        ax.set_title('Disturbing Normal Samples\n(Anomaly in window, but last patch normal)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.axvspan(x[-10], x[-1], alpha=0.2, color=VIS_COLORS['normal_region'], label='Normal Last Patch')
        ax.legend(fontsize=8)

        # Anomaly samples
        ax = axes[2]
        for i, seq in enumerate(anomaly_samples[:5]):
            ax.plot(x, seq, alpha=0.7, label=f'Sample {i+1}')
        ax.set_title('Anomaly Samples\n(Anomaly in last patch)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.axvspan(x[-10], x[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'], label='Anomaly Last Patch')
        ax.legend(fontsize=8)

        plt.suptitle('Sample Types in Dataset', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sample_types.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - sample_types.png")

    def plot_feature_examples(self):
        """Visualize multivariate feature examples"""
        set_seed(42)
        generator = SlidingWindowTimeSeriesGenerator(
            total_length=50000,
            num_features=self.config.num_features,
            interval_scale=1.5,
            seed=42
        )
        signals, point_labels, anomaly_regions = generator.generate()
        dataset = SlidingWindowDataset(
            signals=signals,
            point_labels=point_labels,
            anomaly_regions=anomaly_regions,
            window_size=self.config.seq_length,
            stride=11,
            mask_last_n=10,
            split='test',
            train_ratio=0.5,
            seed=42
        )

        # Get one normal and one anomaly sample
        normal_seq = None
        anomaly_seq = None
        for i in range(len(dataset)):
            seq, label, _, _, _ = dataset[i]
            if label == 0 and normal_seq is None:
                normal_seq = seq.numpy()
            elif label == 1 and anomaly_seq is None:
                anomaly_seq = seq.numpy()
            if normal_seq is not None and anomaly_seq is not None:
                break

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        x = np.arange(self.config.seq_length)

        # Get feature names dynamically
        feature_names = FEATURE_NAMES[:self.config.num_features]

        # Normal sample features - show ALL features
        ax = axes[0]
        for f in range(self.config.num_features):
            ax.plot(x, normal_seq[:, f], alpha=0.8, label=feature_names[f])
        ax.axvspan(x[-10], x[-1], alpha=0.2, color=VIS_COLORS['masked_region'])
        ax.set_title('Normal Sample - All Features', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right', ncol=2, fontsize=8)

        # Anomaly sample features - show ALL features
        ax = axes[1]
        for f in range(self.config.num_features):
            ax.plot(x, anomaly_seq[:, f], alpha=0.8, label=feature_names[f])
        ax.axvspan(x[-10], x[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'])
        ax.set_title('Anomaly Sample - All Features', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right', ncol=2, fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_examples.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - feature_examples.png")

    def plot_dataset_statistics(self):
        """Visualize dataset statistics"""
        set_seed(42)
        generator = SlidingWindowTimeSeriesGenerator(
            total_length=200000,  # Larger for better statistics
            num_features=self.config.num_features,
            interval_scale=1.5,
            seed=42
        )
        signals, point_labels, anomaly_regions = generator.generate()
        dataset = SlidingWindowDataset(
            signals=signals,
            point_labels=point_labels,
            anomaly_regions=anomaly_regions,
            window_size=self.config.seq_length,
            stride=11,
            mask_last_n=10,
            split='test',
            train_ratio=0.5,
            target_counts={'pure_normal': 600, 'disturbing_normal': 150, 'anomaly': 250},
            seed=42
        )

        labels = []
        sample_types = []

        for i in range(len(dataset)):
            _, label, _, st, _ = dataset[i]
            labels.append(label)
            sample_types.append(st)

        labels = np.array(labels)
        sample_types = np.array(sample_types)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Label distribution
        ax = axes[0]
        label_counts = [np.sum(labels == 0), np.sum(labels == 1)]
        bars = ax.bar(['Normal', 'Anomaly'], label_counts, color=[VIS_COLORS['normal'], VIS_COLORS['anomaly']])
        ax.set_title('Label Distribution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count')
        for bar, count in zip(bars, label_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   f'{count}\n({count/len(labels)*100:.1f}%)', ha='center', va='bottom')

        # Sample type distribution
        ax = axes[1]
        type_counts = [np.sum(sample_types == 0), np.sum(sample_types == 1), np.sum(sample_types == 2)]
        type_labels = [SAMPLE_TYPE_NAMES[i] for i in range(3)]
        colors = [SAMPLE_TYPE_COLORS[i] for i in range(3)]
        bars = ax.bar(type_labels, type_counts, color=colors)
        ax.set_title('Sample Type Distribution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=15)
        for bar, count in zip(bars, type_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   f'{count}', ha='center', va='bottom')

        # Pie chart
        ax = axes[2]
        ax.pie(type_counts, labels=type_labels, colors=colors, autopct='%1.1f%%',
               startangle=90, explode=[0, 0.05, 0])
        ax.set_title('Sample Type Proportions', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dataset_statistics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - dataset_statistics.png")

    def plot_anomaly_generation_rules(self):
        """Visualize the rules for generating each anomaly type

        Dynamically generates visualizations based on ANOMALY_TYPE_NAMES.
        Uses actual data examples from the dataset instead of synthetic simulation.
        """
        # Get actual anomaly types (excluding 'normal')
        anomaly_types = [name for name in ANOMALY_TYPE_NAMES if name != 'normal']
        n_types = len(anomaly_types)

        # Calculate grid size
        n_cols = 3
        n_rows = (n_types + n_cols - 1) // n_cols
        if n_types == 7:  # Current case: 7 types
            n_rows, n_cols = 3, 3  # 3x3 grid with 2 empty

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

        # Generate dataset with actual anomalies
        set_seed(42)
        generator = SlidingWindowTimeSeriesGenerator(
            total_length=100000,
            num_features=self.config.num_features,
            interval_scale=1.5,
            seed=42
        )
        signals, point_labels, anomaly_regions = generator.generate()
        dataset = SlidingWindowDataset(
            signals=signals,
            point_labels=point_labels,
            anomaly_regions=anomaly_regions,
            window_size=self.config.seq_length,
            stride=11,
            mask_last_n=self.config.mask_last_n,
            split='test',
            train_ratio=0.5,
            seed=42
        )

        # Get anomaly type info
        anomaly_info = get_anomaly_type_info()
        colors = get_anomaly_colors()

        # Collect examples for each anomaly type
        type_examples = {atype: [] for atype in anomaly_types}

        for i in range(len(dataset)):
            seq, label, pt_labels, sample_type, anomaly_type_idx = dataset[i]
            if label == 1 and anomaly_type_idx > 0:  # Anomaly sample
                atype = ANOMALY_TYPE_NAMES[anomaly_type_idx]
                if atype in type_examples and len(type_examples[atype]) < 3:
                    type_examples[atype].append({
                        'seq': seq.numpy(),
                        'point_labels': pt_labels.numpy(),
                    })

            # Stop if we have enough examples
            if all(len(v) >= 3 for v in type_examples.values()):
                break

        # Plot each anomaly type
        for idx, atype in enumerate(anomaly_types):
            ax = axes[idx]
            x = np.arange(self.config.seq_length)

            # Get info for this type
            info = anomaly_info.get(atype, {})
            length_range = info.get('length_range', (10, 50))
            description = info.get('description', atype.replace('_', ' ').title())
            characteristics = info.get('characteristics', '')
            affected = info.get('affected_features', [])

            # Plot examples if available
            examples = type_examples.get(atype, [])
            if examples:
                for ex_idx, ex in enumerate(examples[:2]):  # Show up to 2 examples
                    seq = ex['seq']
                    pt_labels = ex['point_labels']

                    # Plot first feature
                    alpha = 0.8 if ex_idx == 0 else 0.4
                    ax.plot(x, seq[:, 0], alpha=alpha, lw=1.5,
                           color=colors[atype], label=f'Example {ex_idx+1}' if ex_idx == 0 else None)

                    # Highlight anomaly region
                    if ex_idx == 0:
                        anomaly_mask = pt_labels > 0
                        if anomaly_mask.any():
                            diff = np.diff(np.concatenate([[0], anomaly_mask.astype(int), [0]]))
                            starts = np.where(diff == 1)[0]
                            ends = np.where(diff == -1)[0]
                            for start, end in zip(starts, ends):
                                ax.axvspan(start, end, alpha=0.3, color=VIS_COLORS['anomaly_region'])
            else:
                # Fallback: show synthetic example
                np.random.seed(42 + idx)
                base = 0.5 + 0.3 * np.sin(x * 0.1) + 0.05 * np.random.randn(len(x))
                ax.plot(x, base, 'gray', alpha=0.5, lw=1, label='No example found')

            # Highlight last patch
            ax.axvspan(self.config.seq_length - self.config.mask_last_n, self.config.seq_length,
                      alpha=0.2, color=VIS_COLORS['masked_region'], label=f'Last Patch (n={self.config.mask_last_n})')

            # Title and labels
            ax.set_title(f'{description}\n({atype})', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend(fontsize=7, loc='upper left')

            # Description box
            desc_text = f"Duration: {length_range[0]}-{length_range[1]} steps\n{characteristics}"
            if affected:
                desc_text += f"\nAffected: {', '.join(affected[:3])}"
            ax.text(0.98, 0.02, desc_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Hide empty subplots
        for idx in range(n_types, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Anomaly Generation Rules ({n_types} types)\n'
                    f'All anomalies must overlap with last {self.config.mask_last_n} timesteps',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'anomaly_generation_rules.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - anomaly_generation_rules.png")

    def plot_feature_correlations(self):
        """Visualize correlations between features

        Uses FEATURE_NAMES dynamically.
        """
        set_seed(42)
        generator = SlidingWindowTimeSeriesGenerator(
            total_length=100000,
            num_features=self.config.num_features,
            interval_scale=1.5,
            seed=42
        )
        signals, point_labels, anomaly_regions = generator.generate()
        dataset = SlidingWindowDataset(
            signals=signals,
            point_labels=point_labels,
            anomaly_regions=anomaly_regions,
            window_size=self.config.seq_length,
            stride=11,
            mask_last_n=10,
            split='test',
            train_ratio=0.5,
            seed=42
        )

        # Collect all data
        all_data = []
        for i in range(len(dataset)):
            seq, _, _, _, _ = dataset[i]
            all_data.append(seq.numpy())
        all_data = np.array(all_data)  # (N, T, F)

        # Flatten to (N*T, F) for correlation
        flat_data = all_data.reshape(-1, self.config.num_features)

        # Use FEATURE_NAMES dynamically
        feature_names = FEATURE_NAMES[:self.config.num_features]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Correlation matrix
        ax = axes[0]
        corr_matrix = np.corrcoef(flat_data.T)
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax,
                   xticklabels=feature_names, yticklabels=feature_names,
                   vmin=-1, vmax=1, center=0)
        ax.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

        # 2. Pairwise scatter (first two features)
        ax = axes[1]
        sample_idx = np.random.choice(len(flat_data), min(5000, len(flat_data)), replace=False)
        ax.scatter(flat_data[sample_idx, 0], flat_data[sample_idx, 1], alpha=0.3, s=5)
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title(f'{feature_names[0]} vs {feature_names[1]} (corr={corr_matrix[0, 1]:.3f})',
                    fontsize=12, fontweight='bold')

        # 3. Feature description - dynamic based on FEATURE_NAMES
        ax = axes[2]
        ax.axis('off')

        # Build description text dynamically
        desc_lines = ["Feature Generation Rules (Server Monitoring Simulation)", ""]
        desc_lines.append("=" * 50)

        for i, fname in enumerate(feature_names):
            desc_lines.append(f"\n{fname} (Feature {i}):")
            if fname == 'CPU':
                desc_lines.append("  Base pattern: sinusoidal + noise")
            elif fname == 'Memory':
                desc_lines.append("  Correlated with CPU (slower variation)")
            elif fname == 'DiskIO':
                desc_lines.append("  Correlated with Memory (spiky)")
            elif fname == 'Network':
                desc_lines.append("  Bursty traffic pattern")
            elif fname == 'ResponseTime':
                desc_lines.append("  Correlated with CPU and Network")
            elif fname == 'ThreadCount':
                desc_lines.append("  Correlated with CPU")
            elif fname == 'ErrorRate':
                desc_lines.append("  Correlated with ResponseTime")
            elif fname == 'QueueLength':
                desc_lines.append("  Correlated with CPU, ThreadCount")
            else:
                desc_lines.append("  Generated feature")

        desc_lines.append("\n" + "=" * 50)
        desc_text = "\n".join(desc_lines)

        ax.text(0.05, 0.95, desc_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Feature Correlations and Generation Rules', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_correlations.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - feature_correlations.png")

    def plot_experiment_settings(self):
        """Visualize experiment settings for reproducibility"""
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')

        settings_text = f"""
Experiment Settings
====================

Data Configuration:
  - Sequence Length: {self.config.seq_length}
  - Number of Features: {self.config.num_features}
  - Feature Names: {', '.join(FEATURE_NAMES[:self.config.num_features])}
  - Anomaly Types: {len(ANOMALY_TYPE_NAMES) - 1} types
    ({', '.join(ANOMALY_TYPE_NAMES[1:])})

Sliding Window Dataset:
  - Total Length: {self.config.sliding_window_total_length:,}
  - Stride: {self.config.sliding_window_stride}
  - Mask Last N: {self.config.mask_last_n}
  - Anomaly Interval Scale: {self.config.anomaly_interval_scale}

Test Set Target Composition:
  - Pure Normal: {int(self.config.num_test_samples * self.config.test_ratio_pure_normal)}
  - Disturbing Normal: {int(self.config.num_test_samples * self.config.test_ratio_disturbing_normal)}
  - Anomaly: {int(self.config.num_test_samples * self.config.test_ratio_anomaly)}

Model Configuration:
  - Batch Size: {self.config.batch_size}
  - Learning Rate: {self.config.learning_rate}
  - d_model: {self.config.d_model}
  - Encoder Layers: {self.config.num_encoder_layers}
  - Attention Heads: {self.config.nhead}

Random Seeds:
  - Dataset: seed={self.config.random_seed}
        """

        ax.text(0.05, 0.95, settings_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

        plt.savefig(os.path.join(self.output_dir, 'experiment_settings.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - experiment_settings.png")

    def plot_normal_complexity_features(self):
        """Visualize the 6 normal data complexity features

        Shows how each complexity feature affects normal data patterns,
        comparing enabled vs disabled states.
        """
        fig = plt.figure(figsize=(20, 16))

        # Create grid: 3 rows x 2 cols for 6 features
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

        # Total length for demonstration
        demo_length = 20000
        np.random.seed(42)

        # Feature info for visualization
        complexity_features = [
            {
                'name': 'Regime Switching',
                'description': 'Different operational states with smooth transitions',
                'enable_key': 'enable_regime_switching',
                'characteristics': [
                    'Duration: 8,000-25,000 timesteps per regime',
                    'Transition: 1,500 ts (sigmoid smoothing)',
                    'Each regime has different base values & frequencies',
                ],
                'safety': 'Transitions >> anomaly duration (1500 vs 3-150 ts)',
            },
            {
                'name': 'Multi-Scale Periodicity',
                'description': 'Overlapping cycles at different frequencies',
                'enable_key': 'enable_multi_scale_periodicity',
                'characteristics': [
                    'Fast (hourly-like): freq 0.8-1.5',
                    'Medium (daily-like): freq 0.08-0.15',
                    'Slow (weekly-like): freq 0.015-0.03',
                ],
                'safety': 'Smooth sinusoids, total amplitude < 0.25',
            },
            {
                'name': 'Heteroscedastic Noise',
                'description': 'Load-dependent noise variance (busier = noisier)',
                'enable_key': 'enable_heteroscedastic_noise',
                'characteristics': [
                    'Formula: noise = 0.025 × (1 + 0.8 × load)',
                    'At low load (0.3): noise ≈ 0.031',
                    'At high load (0.5): noise ≈ 0.035',
                ],
                'safety': 'Symmetric noise, max 3σ ≈ 0.08 (anomaly adds 0.3+)',
            },
            {
                'name': 'Time-Varying Correlations',
                'description': 'Feature correlations that slowly change over time',
                'enable_key': 'enable_varying_correlations',
                'characteristics': [
                    'Period: 15,000 timesteps',
                    'Amplitude: ±0.08 variation',
                    'CPU-Memory corr varies 0.12-0.38',
                ],
                'safety': 'Very gradual (period = 15000 ts), always positive',
            },
            {
                'name': 'Bounded Drift (O-U Process)',
                'description': 'Mean-reverting random walk for baseline drift',
                'enable_key': 'enable_drift',
                'characteristics': [
                    'O-U: dx = -θx·dt + σ·dW',
                    'θ=0.002 (slow reversion), σ=0.025',
                    'Clipped to ±0.08 maximum drift',
                ],
                'safety': 'Bidirectional & mean-reverting (leak is monotonic)',
            },
            {
                'name': 'Normal Bumps',
                'description': 'Small, gradual load increases (batch jobs, traffic)',
                'enable_key': 'enable_normal_bumps',
                'characteristics': [
                    'Duration: 100-300 ts (spike: 10-25 ts)',
                    'Magnitude: max 0.10 (spike: 0.3-0.6)',
                    'Affects 1-2 features only (spike: 5+)',
                ],
                'safety': 'Gaussian shape, no error rate increase',
            },
        ]

        for idx, feat_info in enumerate(complexity_features):
            row = idx // 2
            col = idx % 2
            ax = fig.add_subplot(gs[row, col])

            # Generate data with only this feature enabled vs all disabled
            complexity_enabled = NormalDataComplexity(
                enable_complexity=True,
                enable_regime_switching=(feat_info['enable_key'] == 'enable_regime_switching'),
                enable_multi_scale_periodicity=(feat_info['enable_key'] == 'enable_multi_scale_periodicity'),
                enable_heteroscedastic_noise=(feat_info['enable_key'] == 'enable_heteroscedastic_noise'),
                enable_varying_correlations=(feat_info['enable_key'] == 'enable_varying_correlations'),
                enable_drift=(feat_info['enable_key'] == 'enable_drift'),
                enable_normal_bumps=(feat_info['enable_key'] == 'enable_normal_bumps'),
            )

            complexity_disabled = NormalDataComplexity(enable_complexity=False)

            # Generate both versions
            gen_enabled = SlidingWindowTimeSeriesGenerator(
                total_length=demo_length,
                num_features=self.config.num_features,
                complexity=complexity_enabled,
                seed=42
            )
            gen_disabled = SlidingWindowTimeSeriesGenerator(
                total_length=demo_length,
                num_features=self.config.num_features,
                complexity=complexity_disabled,
                seed=42
            )

            signals_enabled, _, _ = gen_enabled.generate()
            signals_disabled, _, _ = gen_disabled.generate()

            # Plot comparison (show first 5000 timesteps for visibility)
            show_length = min(5000, demo_length)
            x = np.arange(show_length)

            # Plot disabled (simple) in gray
            ax.plot(x, signals_disabled[:show_length, 0], 'gray', alpha=0.5,
                   lw=0.8, label='Simple (disabled)')

            # Plot enabled in color
            ax.plot(x, signals_enabled[:show_length, 0], 'C0', alpha=0.8,
                   lw=1.0, label=f'{feat_info["name"]} enabled')

            # Title and labels
            ax.set_title(f'{idx+1}. {feat_info["name"]}', fontsize=13, fontweight='bold')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('CPU Value')
            ax.legend(loc='upper right', fontsize=8)
            ax.set_xlim(0, show_length)

            # Description box
            desc_lines = [feat_info['description'], '']
            desc_lines.extend(feat_info['characteristics'])
            desc_lines.append('')
            desc_lines.append(f"Safety: {feat_info['safety']}")
            desc_text = '\n'.join(desc_lines)

            ax.text(0.02, 0.02, desc_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='bottom', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.suptitle('Normal Data Complexity Features\n'
                    'Each feature adds realistic variation WITHOUT resembling anomalies',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(os.path.join(self.output_dir, 'normal_complexity_features.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("  - normal_complexity_features.png")

    def plot_complexity_comparison(self):
        """Compare simple vs complex normal data with all features enabled"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        demo_length = 30000
        show_length = 10000

        # Generate both versions
        complexity_full = NormalDataComplexity(enable_complexity=True)
        complexity_none = NormalDataComplexity(enable_complexity=False)

        gen_complex = SlidingWindowTimeSeriesGenerator(
            total_length=demo_length,
            num_features=self.config.num_features,
            complexity=complexity_full,
            seed=42
        )
        gen_simple = SlidingWindowTimeSeriesGenerator(
            total_length=demo_length,
            num_features=self.config.num_features,
            complexity=complexity_none,
            seed=42
        )

        signals_complex, _, _ = gen_complex.generate()
        signals_simple, _, _ = gen_simple.generate()

        x = np.arange(show_length)
        feature_names = FEATURE_NAMES[:self.config.num_features]

        # 1. Simple normal (CPU feature)
        ax = axes[0, 0]
        ax.plot(x, signals_simple[:show_length, 0], 'C0', lw=0.8)
        ax.set_title('Simple Normal Data (CPU)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.set_ylim(0, 1)

        # 2. Complex normal (CPU feature)
        ax = axes[0, 1]
        ax.plot(x, signals_complex[:show_length, 0], 'C1', lw=0.8)
        ax.set_title('Complex Normal Data (CPU) - All Features Enabled', fontsize=12, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.set_ylim(0, 1)

        # 3. Simple - all features
        ax = axes[1, 0]
        for f in range(min(4, self.config.num_features)):
            ax.plot(x, signals_simple[:show_length, f], alpha=0.7, lw=0.8,
                   label=feature_names[f])
        ax.set_title('Simple Normal - Multi-Feature View', fontsize=12, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, 1)

        # 4. Complex - all features
        ax = axes[1, 1]
        for f in range(min(4, self.config.num_features)):
            ax.plot(x, signals_complex[:show_length, f], alpha=0.7, lw=0.8,
                   label=feature_names[f])
        ax.set_title('Complex Normal - Multi-Feature View', fontsize=12, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, 1)

        plt.suptitle('Simple vs Complex Normal Data Comparison\n'
                    'Complex data includes: Regime Switching, Multi-Scale Periodicity, '
                    'Heteroscedastic Noise, Time-Varying Correlations, O-U Drift, Normal Bumps',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'complexity_comparison.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("  - complexity_comparison.png")

    def plot_complexity_vs_anomaly(self):
        """Show why complexity features don't get confused with anomalies"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        demo_length = 20000

        # Generate complex normal data
        complexity = NormalDataComplexity(enable_complexity=True)
        gen = SlidingWindowTimeSeriesGenerator(
            total_length=demo_length,
            num_features=self.config.num_features,
            complexity=complexity,
            interval_scale=1.5,
            seed=42
        )
        signals, point_labels, anomaly_regions = gen.generate()

        # Find regions with specific characteristics
        x = np.arange(demo_length)

        # Row 1: Normal complexity features
        # 1. Regime transition (normal)
        ax = axes[0, 0]
        # Show a smooth transition region
        start = 8000
        length = 4000
        ax.plot(x[start:start+length], signals[start:start+length, 0], 'C0', lw=1)
        ax.fill_between(x[start:start+length], 0, 1, alpha=0.1, color=VIS_COLORS['normal_region'])
        ax.set_title('Normal: Regime Transition\n(Smooth, 1500+ ts)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.95, 'Gradual change over thousands of timesteps',
               transform=ax.transAxes, ha='center', va='top', fontsize=9,
               bbox=dict(facecolor='lightgreen', alpha=0.7))

        # 2. Normal bump (normal)
        ax = axes[0, 1]
        # Generate data with just bumps to find one
        bump_complexity = NormalDataComplexity(
            enable_complexity=True,
            enable_regime_switching=False,
            enable_multi_scale_periodicity=False,
            enable_heteroscedastic_noise=False,
            enable_varying_correlations=False,
            enable_drift=False,
            enable_normal_bumps=True,
        )
        gen_bump = SlidingWindowTimeSeriesGenerator(
            total_length=demo_length,
            num_features=self.config.num_features,
            complexity=bump_complexity,
            seed=42
        )
        signals_bump, _, _ = gen_bump.generate()
        start = 2000
        length = 800
        ax.plot(x[start:start+length], signals_bump[start:start+length, 0], 'C0', lw=1)
        ax.fill_between(x[start:start+length], 0, 1, alpha=0.1, color=VIS_COLORS['normal_region'])
        ax.set_title('Normal: Small Bump\n(Gaussian, 100-300 ts, mag<0.10)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.95, 'Smooth Gaussian shape, affects 1-2 features',
               transform=ax.transAxes, ha='center', va='top', fontsize=9,
               bbox=dict(facecolor='lightgreen', alpha=0.7))

        # 3. O-U drift (normal)
        ax = axes[0, 2]
        drift_complexity = NormalDataComplexity(
            enable_complexity=True,
            enable_regime_switching=False,
            enable_multi_scale_periodicity=False,
            enable_heteroscedastic_noise=False,
            enable_varying_correlations=False,
            enable_drift=True,
            enable_normal_bumps=False,
        )
        gen_drift = SlidingWindowTimeSeriesGenerator(
            total_length=demo_length,
            num_features=self.config.num_features,
            complexity=drift_complexity,
            seed=42
        )
        signals_drift, _, _ = gen_drift.generate()
        start = 0
        length = 5000
        ax.plot(x[start:start+length], signals_drift[start:start+length, 0], 'C0', lw=1)
        ax.fill_between(x[start:start+length], 0, 1, alpha=0.1, color=VIS_COLORS['normal_region'])
        ax.set_title('Normal: O-U Drift\n(Mean-reverting, ±0.08 max)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.95, 'Bidirectional, always returns to baseline',
               transform=ax.transAxes, ha='center', va='top', fontsize=9,
               bbox=dict(facecolor='lightgreen', alpha=0.7))

        # Row 2: Anomaly patterns (for comparison)
        # Find anomaly regions
        spike_region = None
        leak_region = None
        for region in anomaly_regions:
            if region.anomaly_type == 1 and spike_region is None:  # spike
                spike_region = region
            elif region.anomaly_type == 2 and leak_region is None:  # memory_leak
                leak_region = region
            if spike_region is not None and leak_region is not None:
                break

        # 4. Spike anomaly
        ax = axes[1, 0]
        if spike_region is not None:
            start = max(0, spike_region.start - 50)
            end = min(demo_length, spike_region.end + 50)
            ax.plot(x[start:end], signals[start:end, 0], 'C3', lw=1)
            ax.axvspan(spike_region.start, spike_region.end, alpha=0.3, color=VIS_COLORS['anomaly_region'])
        else:
            ax.text(0.5, 0.5, 'No spike anomaly found\nin this dataset',
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('ANOMALY: Spike\n(Sudden, 10-25 ts, mag 0.3-0.6)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.95, 'Sudden jump, affects 5+ features + error rate',
               transform=ax.transAxes, ha='center', va='top', fontsize=9,
               bbox=dict(facecolor='lightcoral', alpha=0.7))

        # 5. Memory leak anomaly
        ax = axes[1, 1]
        if leak_region is not None:
            start = max(0, leak_region.start - 20)
            end = min(demo_length, leak_region.end + 20)
            # Show Memory feature (index 1) for leak
            ax.plot(x[start:end], signals[start:end, 1], 'C3', lw=1)
            ax.axvspan(leak_region.start, leak_region.end, alpha=0.3, color=VIS_COLORS['anomaly_region'])
        else:
            ax.text(0.5, 0.5, 'No memory leak anomaly found\nin this dataset',
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('ANOMALY: Memory Leak\n(Monotonic increase, 80-150 ts)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Memory Value')
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.95, 'Only goes UP (never down), mag 0.3-0.5',
               transform=ax.transAxes, ha='center', va='top', fontsize=9,
               bbox=dict(facecolor='lightcoral', alpha=0.7))

        # 6. Summary comparison table
        ax = axes[1, 2]
        ax.axis('off')

        table_text = """
╔══════════════════════════════════════════════════════════════╗
║           COMPLEXITY vs ANOMALY COMPARISON                   ║
╠══════════════════════════════════════════════════════════════╣
║  Aspect          │  Normal Complexity  │  Anomaly            ║
╠══════════════════════════════════════════════════════════════╣
║  Duration        │  1000+ timesteps    │  3-150 timesteps    ║
║  Value range     │  [0.05, 0.70]       │  [0.70, 1.00]       ║
║  Direction       │  Bidirectional      │  Usually upward     ║
║  Transition      │  Smooth/gradual     │  Sudden             ║
║  Features        │  1-2 (bumps)        │  5+ simultaneous    ║
║  Error rate      │  NOT affected       │  Increases          ║
║  Mean-reverting  │  Yes (O-U drift)    │  No (leaks grow)    ║
╚══════════════════════════════════════════════════════════════╝

KEY INSIGHT: All complexity features are designed with
safety constraints that make them fundamentally different
from anomaly patterns.
"""
        ax.text(0.5, 0.5, table_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', horizontalalignment='center',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.suptitle('Normal Complexity Features vs Anomaly Patterns\n'
                    'Green = Normal (complexity), Red = Anomaly',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'complexity_vs_anomaly.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("  - complexity_vs_anomaly.png")

    def generate_all(self):
        """Generate all data visualizations"""
        print("\n  Generating Data Visualizations...")
        # Note: plot_anomaly_types() removed - redundant with plot_anomaly_generation_rules()
        self.plot_sample_types()
        self.plot_feature_examples()
        self.plot_dataset_statistics()
        self.plot_anomaly_generation_rules()
        self.plot_feature_correlations()
        self.plot_normal_complexity_features()
        self.plot_complexity_comparison()
        self.plot_complexity_vs_anomaly()
        self.plot_experiment_settings()
