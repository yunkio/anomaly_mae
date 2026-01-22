"""
Architecture Visualizer - Model Architecture Visualizations

This module provides visualizations for:
- Model pipeline overview
- Patchify modes comparison
- Masking process
- Self-distillation concept
- Margin types for loss
- Loss function components
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mae_anomaly import Config


class ArchitectureVisualizer:
    """Visualize model architecture and concepts"""

    def __init__(self, output_dir: str, config: Config = None):
        self.output_dir = output_dir
        self.config = config or Config()
        os.makedirs(output_dir, exist_ok=True)

    def plot_model_pipeline(self):
        """Visualize the overall model pipeline"""
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 8)
        ax.axis('off')

        # Define colors
        colors = {
            'input': '#3498DB',
            'encoder': '#27AE60',
            'teacher': '#E74C3C',
            'student': '#9B59B6',
            'loss': '#F39C12',
            'mask': '#95A5A6'
        }

        # Input
        rect = mpatches.FancyBboxPatch((0.5, 3), 2, 2, boxstyle="round,pad=0.05",
                                        facecolor=colors['input'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(1.5, 4, 'Input\nSequence\n(T, F)', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Patchify + Embed
        rect = mpatches.FancyBboxPatch((3.5, 3), 2, 2, boxstyle="round,pad=0.05",
                                        facecolor=colors['mask'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(4.5, 4, 'Patchify\n+\nEmbed', ha='center', va='center', fontsize=10, fontweight='bold')

        # Masking
        rect = mpatches.FancyBboxPatch((6.5, 3), 2, 2, boxstyle="round,pad=0.05",
                                        facecolor=colors['mask'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(7.5, 4, 'Random\nMasking', ha='center', va='center', fontsize=10, fontweight='bold')

        # Encoder
        rect = mpatches.FancyBboxPatch((9.5, 3), 2, 2, boxstyle="round,pad=0.05",
                                        facecolor=colors['encoder'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(10.5, 4, 'Shared\nEncoder', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Teacher Decoder (top)
        rect = mpatches.FancyBboxPatch((12.5, 5.5), 2, 2, boxstyle="round,pad=0.05",
                                        facecolor=colors['teacher'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(13.5, 6.5, 'Teacher\nDecoder', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Student Decoder (bottom)
        rect = mpatches.FancyBboxPatch((12.5, 0.5), 2, 2, boxstyle="round,pad=0.05",
                                        facecolor=colors['student'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(13.5, 1.5, 'Student\nDecoder', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Arrows
        arrow_style = dict(arrowstyle='->', color='black', lw=2)
        ax.annotate('', xy=(3.4, 4), xytext=(2.6, 4), arrowprops=arrow_style)
        ax.annotate('', xy=(6.4, 4), xytext=(5.6, 4), arrowprops=arrow_style)
        ax.annotate('', xy=(9.4, 4), xytext=(8.6, 4), arrowprops=arrow_style)

        # Split to teacher/student
        ax.annotate('', xy=(12.4, 6.5), xytext=(11.6, 4.5), arrowprops=arrow_style)
        ax.annotate('', xy=(12.4, 1.5), xytext=(11.6, 3.5), arrowprops=arrow_style)

        # Loss box
        rect = mpatches.FancyBboxPatch((15, 3), 1, 2, boxstyle="round,pad=0.05",
                                        facecolor=colors['loss'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(15.5, 4, 'Loss', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Arrows to loss
        ax.annotate('', xy=(14.9, 4.5), xytext=(14.6, 5.5), arrowprops=arrow_style)
        ax.annotate('', xy=(14.9, 3.5), xytext=(14.6, 2.5), arrowprops=arrow_style)

        # Labels
        ax.text(15.5, 5.3, 'Recon', ha='center', fontsize=8)
        ax.text(15.5, 2.7, 'Disc', ha='center', fontsize=8)

        # Title
        ax.set_title('Self-Distilled MAE Pipeline', fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_pipeline.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - model_pipeline.png")

    def plot_patchify_modes(self):
        """Visualize different patchify modes"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Sample data
        np.random.seed(42)
        seq_length = 100
        t = np.arange(seq_length)
        signal = np.sin(t * 0.1) + 0.5 * np.sin(t * 0.3) + 0.1 * np.random.randn(seq_length)

        num_patches = 10
        patch_size = seq_length // num_patches

        titles = ['CNN-First Mode', 'Patch-CNN Mode', 'Linear Mode']
        descriptions = [
            'CNN on full sequence\n(potential information leakage)',
            'Patchify first, then CNN per patch\n(no cross-patch leakage)',
            'Patchify then linear embedding\n(original MAE style)'
        ]
        colors = ['#4682B4', '#CD5C5C', '#228B22']

        for idx, (ax, title, desc, color) in enumerate(zip(axes, titles, descriptions, colors)):
            ax.plot(t, signal, 'b-', alpha=0.5, lw=1)

            # Draw patches
            for p in range(num_patches):
                start = p * patch_size
                end = (p + 1) * patch_size
                ax.axvspan(start, end, alpha=0.2 if p % 2 == 0 else 0.1, color=color)
                ax.axvline(x=start, color='gray', linestyle='--', alpha=0.5)

            ax.set_title(f'{title}\n{desc}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')

            # Add processing indicator
            if idx == 0:
                ax.annotate('', xy=(80, 1.5), xytext=(20, 1.5),
                           arrowprops=dict(arrowstyle='<->', color='red', lw=2))
                ax.text(50, 1.7, 'CNN sees all', ha='center', fontsize=9, color='red')
            elif idx == 1:
                ax.annotate('', xy=(patch_size-1, 1.5), xytext=(0, 1.5),
                           arrowprops=dict(arrowstyle='<->', color='green', lw=2))
                ax.text(patch_size/2, 1.7, 'CNN per patch', ha='center', fontsize=9, color='green')

        plt.suptitle('Patchify Mode Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'patchify_modes.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - patchify_modes.png")

    def plot_masking_visualization(self):
        """Visualize the masking process"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        np.random.seed(42)
        seq_length = 100
        t = np.arange(seq_length)
        signal = np.sin(t * 0.1) + 0.3 * np.cos(t * 0.2)

        num_patches = 10
        patch_size = seq_length // num_patches

        # 1. Original sequence
        ax = axes[0, 0]
        ax.plot(t, signal, 'b-', lw=2)
        ax.set_title('1. Original Sequence', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')

        # 2. Patchified
        ax = axes[0, 1]
        colors = plt.cm.tab10(np.arange(num_patches))
        for p in range(num_patches):
            start = p * patch_size
            end = (p + 1) * patch_size
            ax.plot(t[start:end], signal[start:end], color=colors[p], lw=2)
            ax.axvline(x=start, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'2. Patchified ({num_patches} patches)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')

        # 3. Masked
        ax = axes[1, 0]
        mask_ratio = 0.5
        n_masked = int(num_patches * mask_ratio)
        masked_patches = np.random.choice(num_patches, n_masked, replace=False)

        for p in range(num_patches):
            start = p * patch_size
            end = (p + 1) * patch_size
            if p in masked_patches:
                ax.axvspan(start, end, alpha=0.3, color='red')
                ax.plot(t[start:end], signal[start:end], 'r--', lw=1, alpha=0.5)
            else:
                ax.plot(t[start:end], signal[start:end], 'b-', lw=2)
            ax.axvline(x=start, color='gray', linestyle='--', alpha=0.5)

        ax.set_title(f'3. Masked ({mask_ratio*100:.0f}% masking ratio)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')

        # Legend
        patches = [mpatches.Patch(color='blue', label='Visible'),
                   mpatches.Patch(color='red', alpha=0.3, label='Masked')]
        ax.legend(handles=patches)

        # 4. Reconstruction target
        ax = axes[1, 1]
        for p in range(num_patches):
            start = p * patch_size
            end = (p + 1) * patch_size
            if p in masked_patches:
                ax.axvspan(start, end, alpha=0.3, color='red')
                ax.plot(t[start:end], signal[start:end], 'g-', lw=2, label='Target' if p == masked_patches[0] else '')
            else:
                ax.plot(t[start:end], signal[start:end], 'b-', lw=2, alpha=0.3)
            ax.axvline(x=start, color='gray', linestyle='--', alpha=0.5)

        ax.set_title('4. Reconstruction Target (masked regions)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')

        plt.suptitle('MAE Masking Process', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'masking_visualization.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - masking_visualization.png")

    def plot_self_distillation_concept(self):
        """Visualize the self-distillation concept"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Normal sample - teacher and student similar
        ax = axes[0]
        np.random.seed(42)
        t = np.arange(100)
        original = np.sin(t * 0.1) + 0.1 * np.random.randn(100)
        teacher_recon = original + 0.05 * np.random.randn(100)
        student_recon = original + 0.08 * np.random.randn(100)

        ax.plot(t, original, 'b-', lw=2, label='Original')
        ax.plot(t, teacher_recon, 'g--', lw=2, label='Teacher', alpha=0.8)
        ax.plot(t, student_recon, 'r:', lw=2, label='Student', alpha=0.8)

        # Highlight small discrepancy
        ax.fill_between(t, teacher_recon, student_recon, alpha=0.2, color='purple')

        ax.set_title('Normal Sample\n(Small Teacher-Student Discrepancy)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()

        # Right: Anomaly sample - teacher and student differ
        ax = axes[1]
        original_anomaly = np.sin(t * 0.1)
        original_anomaly[70:85] = 1.5  # Anomaly
        original_anomaly += 0.1 * np.random.randn(100)

        teacher_recon_a = original_anomaly.copy()
        teacher_recon_a[70:85] = np.sin(t[70:85] * 0.1) + 0.1  # Teacher reconstructs normal pattern

        student_recon_a = original_anomaly.copy()
        student_recon_a[70:85] = np.sin(t[70:85] * 0.1) + 0.5  # Student fails more

        ax.plot(t, original_anomaly, 'b-', lw=2, label='Original (Anomaly)')
        ax.plot(t, teacher_recon_a, 'g--', lw=2, label='Teacher', alpha=0.8)
        ax.plot(t, student_recon_a, 'r:', lw=2, label='Student', alpha=0.8)

        # Highlight large discrepancy
        ax.fill_between(t, teacher_recon_a, student_recon_a, alpha=0.3, color='purple')
        ax.axvspan(70, 85, alpha=0.2, color='red')

        ax.set_title('Anomaly Sample\n(Large Teacher-Student Discrepancy)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()

        plt.suptitle('Self-Distillation for Anomaly Detection\nTeacher (trained) vs Student (weaker decoder)',
                     fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'self_distillation_concept.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - self_distillation_concept.png")

    def plot_margin_types(self):
        """Visualize different margin types for discrepancy loss"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        x = np.linspace(-0.5, 2, 200)
        margin = 0.5

        # Hinge loss
        ax = axes[0]
        hinge = np.maximum(0, margin - x)
        ax.plot(x, hinge, 'b-', lw=2)
        ax.axvline(x=margin, color='r', linestyle='--', label=f'margin={margin}')
        ax.fill_between(x, 0, hinge, alpha=0.2)
        ax.set_title('Hinge Loss\nmax(0, m - d)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Discrepancy (d)')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_ylim(-0.1, 1.5)
        ax.grid(True, alpha=0.3)

        # Softplus loss
        ax = axes[1]
        softplus = np.log(1 + np.exp(margin - x))
        ax.plot(x, softplus, 'g-', lw=2)
        ax.axvline(x=margin, color='r', linestyle='--', label=f'margin={margin}')
        ax.fill_between(x, 0, softplus, alpha=0.2, color='green')
        ax.set_title('Softplus Loss\nlog(1 + exp(m - d))', fontsize=12, fontweight='bold')
        ax.set_xlabel('Discrepancy (d)')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_ylim(-0.1, 1.5)
        ax.grid(True, alpha=0.3)

        # Dynamic margin concept
        ax = axes[2]
        # Show multiple margins
        for m in [0.3, 0.5, 0.7, 1.0]:
            hinge = np.maximum(0, m - x)
            ax.plot(x, hinge, lw=1.5, label=f'm={m}', alpha=0.7)
        ax.set_title('Dynamic Margin\nAdaptive margin based on input', fontsize=12, fontweight='bold')
        ax.set_xlabel('Discrepancy (d)')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_ylim(-0.1, 1.5)
        ax.grid(True, alpha=0.3)

        plt.suptitle('Margin Types for Discrepancy Loss', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'margin_types.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - margin_types.png")

    def plot_loss_components(self):
        """Visualize the loss function components"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')

        # Title
        ax.text(0.5, 0.95, 'Loss Function Components', ha='center', va='top',
                fontsize=16, fontweight='bold', transform=ax.transAxes)

        # Total Loss formula
        ax.text(0.5, 0.85, r'$\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda \cdot \mathcal{L}_{disc}$',
                ha='center', va='center', fontsize=14, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Reconstruction Loss
        ax.text(0.25, 0.65, 'Reconstruction Loss', ha='center', va='center',
                fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.25, 0.55, r'$\mathcal{L}_{recon} = \frac{1}{|M|} \sum_{i \in M} ||\hat{x}_i^{teacher} - x_i||^2$',
                ha='center', va='center', fontsize=11, transform=ax.transAxes)
        ax.text(0.25, 0.45, 'Teacher reconstructs\nmasked patches',
                ha='center', va='center', fontsize=10, transform=ax.transAxes, style='italic')

        # Discrepancy Loss
        ax.text(0.75, 0.65, 'Discrepancy Loss', ha='center', va='center',
                fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.75, 0.55, r'$\mathcal{L}_{disc} = \max(0, m - ||\hat{x}^{teacher} - \hat{x}^{student}||)$',
                ha='center', va='center', fontsize=11, transform=ax.transAxes)
        ax.text(0.75, 0.45, 'Push Teacher-Student\ndiscrepancy above margin',
                ha='center', va='center', fontsize=10, transform=ax.transAxes, style='italic')

        # Intuition box
        rect = mpatches.FancyBboxPatch((0.1, 0.1), 0.8, 0.25, boxstyle="round,pad=0.02",
                                        facecolor='lightblue', edgecolor='black', linewidth=1,
                                        transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.5, 0.28, 'Intuition:', ha='center', va='center',
                fontsize=11, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.2, '- For Normal data: Both Teacher and Student reconstruct well\n'
                         '  -> Small discrepancy (good reconstruction, low disc loss)',
                ha='center', va='center', fontsize=10, transform=ax.transAxes)
        ax.text(0.5, 0.13, '- For Anomaly data: Teacher learns normal patterns better than Student\n'
                         '  -> Large discrepancy (anomaly signal!)',
                ha='center', va='center', fontsize=10, transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'loss_components.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - loss_components.png")

    def generate_all(self):
        """Generate all architecture visualizations"""
        print("\n  Generating Architecture Visualizations...")
        self.plot_model_pipeline()
        self.plot_patchify_modes()
        self.plot_masking_visualization()
        self.plot_self_distillation_concept()
        self.plot_margin_types()
        self.plot_loss_components()
