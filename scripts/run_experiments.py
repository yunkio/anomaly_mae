"""
Comprehensive Experiments for Self-Distilled MAE on Multivariate Time Series

Features:
- Grid search over parameter combinations
- Two-stage search: Quick screening → Full training on top combinations
- Results saving (no visualization - use visualize_all.py for that)
- Best model saving and analysis
"""

import sys
sys.path.insert(0, '/home/ykio/notebooks/claude')

import os
import json
import time
import itertools
from datetime import datetime
from copy import deepcopy
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import from mae_anomaly package
from mae_anomaly import (
    Config, set_seed,
    SlidingWindowTimeSeriesGenerator, SlidingWindowDataset,
    SelfDistilledMAEMultivariate, SelfDistillationLoss,
    Trainer, Evaluator, SLIDING_ANOMALY_TYPE_NAMES
)


# =============================================================================
# Default Parameter Grid
# =============================================================================

DEFAULT_PARAM_GRID = {
    # Masking (high impact)
    'masking_ratio': [0.4, 0.7],
    'masking_strategy': ['patch', 'feature_wise'],  # Masking strategy
    'num_patches': [10, 25, 50],

    # Discrepancy loss strategy
    'margin_type': ['hinge', 'softplus', 'dynamic'],

    # Anomaly masking strategy
    'force_mask_anomaly': [False, True],

    # Loss granularity
    'patch_level_loss': [True, False],

    # Patchify mode (CNN vs Linear embedding)
    'patchify_mode': ['cnn_first', 'patch_cnn', 'linear'],
}
# Note: margin=0.5, lambda_disc=0.5 are fixed (not in grid)
# Total combinations: 2*2*3*3*2*2*3 = 432


# =============================================================================
# Experiment Runner (Grid Search)
# =============================================================================

class ExperimentRunner:
    """
    Run comprehensive experiments with parameter grid search

    Two-stage approach:
    1. Quick search: Fast screening with reduced epochs/data
    2. Full search: Complete training on top combinations

    Results are saved to output_dir. Use visualize_all.py for visualization.
    """

    def __init__(
        self,
        param_grid: Dict[str, List] = None,
        base_config: Config = None,
        output_dir: str = 'results/experiments'
    ):
        self.param_grid = param_grid or DEFAULT_PARAM_GRID
        self.base_config = base_config or Config()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(output_dir, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)

        self.results = []
        self.full_results = []
        self.histories = {}
        self.best_model = None
        self.best_config = None
        self.best_metrics = None

        # Generate sliding window dataset once (shared across all experiments)
        self._generate_dataset()

        print(f"\nExperiment results will be saved to: {self.output_dir}")

    def _generate_dataset(self, quick_length: int = 200000, full_length: int = None):
        """Generate sliding window datasets for quick and full search

        Args:
            quick_length: Length of time series for quick search (default: 200,000)
                         With train_ratio=0.3 -> ~6,000 train, ~14,000 test
                         Test set sufficient for 1200:300:500 target composition
            full_length: Length for full search (default: config.sliding_window_total_length)
        """
        config = self.base_config
        full_length = full_length or config.sliding_window_total_length

        # Store train_ratio settings for quick/full search
        self.quick_train_ratio = 0.3  # 30% train, 70% test for quick search
        self.full_train_ratio = 0.5   # 50% train, 50% test for full search

        # Generate quick search time series (shorter for fast screening)
        print("Generating quick search dataset...", flush=True)
        quick_generator = SlidingWindowTimeSeriesGenerator(
            total_length=quick_length,
            num_features=config.num_features,
            interval_scale=config.anomaly_interval_scale,
            seed=config.random_seed
        )
        self.quick_signals, self.quick_point_labels, self.quick_anomaly_regions = quick_generator.generate()
        print(f"  Quick: {len(self.quick_signals):,} timesteps, {len(self.quick_anomaly_regions)} anomaly regions", flush=True)
        print(f"         train_ratio={self.quick_train_ratio} -> ~{int(quick_length/10*self.quick_train_ratio):,} train, ~{int(quick_length/10*(1-self.quick_train_ratio)):,} test", flush=True)

        # Print quick dataset statistics
        self._print_dataset_statistics(
            "Quick Dataset",
            self.quick_signals,
            self.quick_point_labels,
            self.quick_anomaly_regions,
            self.quick_train_ratio
        )

        # Generate full search time series (longer for thorough training)
        print("\nGenerating full search dataset...", flush=True)
        full_generator = SlidingWindowTimeSeriesGenerator(
            total_length=full_length,
            num_features=config.num_features,
            interval_scale=config.anomaly_interval_scale,
            seed=config.random_seed
        )
        self.signals, self.point_labels, self.anomaly_regions = full_generator.generate()
        print(f"  Full: {len(self.signals):,} timesteps, {len(self.anomaly_regions)} anomaly regions", flush=True)
        print(f"        train_ratio={self.full_train_ratio} -> ~{int(full_length/10*self.full_train_ratio):,} train, ~{int(full_length/10*(1-self.full_train_ratio)):,} test", flush=True)

        # Print full dataset statistics
        self._print_dataset_statistics(
            "Full Dataset",
            self.signals,
            self.point_labels,
            self.anomaly_regions,
            self.full_train_ratio
        )

    def _print_dataset_statistics(
        self,
        name: str,
        signals: np.ndarray,
        point_labels: np.ndarray,
        anomaly_regions: list,
        train_ratio: float
    ):
        """Print detailed statistics about the dataset

        Args:
            name: Dataset name (e.g., "Quick Dataset", "Full Dataset")
            signals: Time series signals
            point_labels: Point-level anomaly labels
            anomaly_regions: List of AnomalyRegion objects
            train_ratio: Train/test split ratio
        """
        config = self.base_config

        # Create temporary test dataset to get sample-level statistics
        temp_dataset = SlidingWindowDataset(
            signals=signals,
            point_labels=point_labels,
            anomaly_regions=anomaly_regions,
            window_size=config.seq_length,
            stride=config.sliding_window_stride,
            mask_last_n=config.mask_last_n,
            split='test',
            train_ratio=train_ratio,
            seed=config.random_seed
            # No target_counts - get raw distribution
        )

        # Get sample type distribution
        sample_types = temp_dataset.sample_type_labels
        n_pure_normal = (sample_types == 0).sum()
        n_disturbing_normal = (sample_types == 1).sum()
        n_anomaly = (sample_types == 2).sum()
        total = len(sample_types)

        print(f"\n  [{name} Statistics - Test Set (Raw)]")
        print(f"  Sample Types:")
        print(f"    - Pure Normal:       {n_pure_normal:,} ({100*n_pure_normal/total:.1f}%)")
        print(f"    - Disturbing Normal: {n_disturbing_normal:,} ({100*n_disturbing_normal/total:.1f}%)")
        print(f"    - Anomaly:           {n_anomaly:,} ({100*n_anomaly/total:.1f}%)")
        print(f"    - Total:             {total:,}")

        # Get anomaly type distribution (from anomaly_regions)
        anomaly_type_counts = {}
        for region in anomaly_regions:
            atype = region.anomaly_type
            anomaly_type_counts[atype] = anomaly_type_counts.get(atype, 0) + 1

        print(f"\n  Anomaly Types (region count):")
        for atype_idx in sorted(anomaly_type_counts.keys()):
            atype_name = SLIDING_ANOMALY_TYPE_NAMES[atype_idx] if atype_idx < len(SLIDING_ANOMALY_TYPE_NAMES) else f"type_{atype_idx}"
            count = anomaly_type_counts[atype_idx]
            print(f"    - {atype_name}: {count}")

        # Clean up temporary dataset
        del temp_dataset

    def generate_combinations(self) -> List[Dict]:
        """Generate all parameter combinations"""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _estimate_stage2_count(self) -> int:
        """Estimate the number of models for Stage 2 based on selection criteria"""
        n_per_value = 5

        # Phase 1: Count unique parameter values
        phase1_max = 0
        for key, values in self.param_grid.items():
            phase1_max += len(values) * n_per_value

        # Phase 2 & 3: Additional models (excluding duplicates)
        phase2_max = 10  # overall_roc_auc
        phase3_max = 5   # disturbing_roc_auc

        # Estimate with ~50% overlap between phases
        # Phase 1 typically has many overlaps within itself
        estimated = int(phase1_max * 0.6) + phase2_max + phase3_max

        return estimated

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable string"""
        if seconds < 60:
            return f"{seconds:.1f}초"
        elif seconds < 3600:
            mins = seconds / 60
            return f"{mins:.1f}분"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}시간 {mins}분"

    def _print_time_estimate(
        self,
        first_model_time: float,
        n_quick: int,
        n_full_est: int,
        quick_epochs: int,
        full_epochs: int,
        quick_length: int = 200000,
        full_length: int = None
    ):
        """Print estimated time based on first model training time"""
        full_length = full_length or self.base_config.sliding_window_total_length

        # Calculate actual train samples
        stride = self.base_config.sliding_window_stride
        quick_train_samples = int(quick_length * self.quick_train_ratio / stride)
        full_train_samples = int(full_length * self.full_train_ratio / stride)

        # Calculate scaling factors
        train_scale = full_train_samples / quick_train_samples
        epoch_scale = full_epochs / quick_epochs

        # Estimate times
        quick_remaining = first_model_time * (n_quick - 1)  # First one already done
        quick_total = first_model_time * n_quick

        full_model_time = first_model_time * train_scale * epoch_scale
        full_total = full_model_time * n_full_est

        total = quick_total + full_total

        # Print estimates
        print(f"\n>>> Estimated Time (based on 1st model: {first_model_time:.1f}s) <<<")
        print(f"  Quick Search: {self._format_time(quick_total)} ({n_quick} models × {first_model_time:.1f}s)")
        print(f"  Full Search:  {self._format_time(full_total)} (~{n_full_est} models × {full_model_time:.1f}s)")
        print(f"  Total:        {self._format_time(total)}")
        print(f"  (Quick remaining: {self._format_time(quick_remaining)})")
        print()

    def _create_config(self, params: Dict, quick: bool = False) -> Config:
        """Create config with given parameters"""
        config = deepcopy(self.base_config)

        for key, value in params.items():
            if hasattr(config, key):
                # Ensure integer types for parameters that require it
                int_params = [
                    'num_patches', 'num_epochs', 'batch_size', 'num_train_samples', 'num_test_samples',
                    'd_model', 'nhead', 'num_encoder_layers', 'num_teacher_decoder_layers',
                    'num_student_decoder_layers', 'dim_feedforward'
                ]
                if key in int_params:
                    value = int(value)
                setattr(config, key, value)

        if 'num_patches' in params:
            config.num_patches = int(config.num_patches)
            config.patch_size = int(config.seq_length // config.num_patches)
            config.mask_last_n = int(config.patch_size)

        # Fix margin and lambda_disc (not in grid search)
        config.margin = 0.5
        config.lambda_disc = 0.5

        if quick:
            config.num_epochs = 15
            config.num_train_samples = 500
            config.num_test_samples = 200

        return config

    def _run_single_experiment(
        self,
        config: Config,
        experiment_id: int,
        save_history: bool = False,
        use_quick_data: bool = False
    ) -> Dict:
        """Run a single experiment

        Args:
            config: Experiment configuration
            experiment_id: Unique experiment ID
            save_history: Whether to save training history
            use_quick_data: If True, use shorter quick_signals; else use full signals
        """
        set_seed(config.random_seed)

        # Select dataset and train_ratio based on quick/full mode
        if use_quick_data:
            signals = self.quick_signals
            point_labels = self.quick_point_labels
            anomaly_regions = self.quick_anomaly_regions
            train_ratio = self.quick_train_ratio  # 0.3 for quick search
        else:
            signals = self.signals
            point_labels = self.point_labels
            anomaly_regions = self.anomaly_regions
            train_ratio = self.full_train_ratio   # 0.5 for full search

        # Create train dataset from selected time series
        train_dataset = SlidingWindowDataset(
            signals=signals,
            point_labels=point_labels,
            anomaly_regions=anomaly_regions,
            window_size=config.seq_length,
            stride=config.sliding_window_stride,
            mask_last_n=config.mask_last_n,
            split='train',
            train_ratio=train_ratio,
            seed=config.random_seed
        )

        # Create test dataset with target counts
        test_dataset = SlidingWindowDataset(
            signals=signals,
            point_labels=point_labels,
            anomaly_regions=anomaly_regions,
            window_size=config.seq_length,
            stride=config.sliding_window_stride,
            mask_last_n=config.mask_last_n,
            split='test',
            train_ratio=train_ratio,
            target_counts={
                'pure_normal': config.test_target_pure_normal,
                'disturbing_normal': config.test_target_disturbing_normal,
                'anomaly': config.test_target_anomaly
            },
            seed=config.random_seed
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        model = SelfDistilledMAEMultivariate(config)
        trainer = Trainer(model, config, train_loader, test_loader, verbose=False)
        trainer.train()

        evaluator = Evaluator(model, config, test_loader)
        metrics = evaluator.evaluate()

        if save_history:
            self.histories[experiment_id] = trainer.history

        # Check if this is the best model
        if self.best_metrics is None or metrics['roc_auc'] > self.best_metrics['roc_auc']:
            self.best_model = model
            self.best_config = config
            self.best_metrics = metrics

        # Clean up
        del trainer
        if not save_history:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    def run_grid_search(
        self,
        quick_epochs: int = 15,
        quick_train: int = 1000,
        quick_test: int = 400,
        full_epochs: int = 100,
        full_train: int = 2000,
        full_test: int = 500,
        two_stage: bool = True
    ) -> pd.DataFrame:
        """
        Run grid search over parameter combinations

        Args:
            quick_epochs: Epochs for quick search
            quick_train: Training samples for quick search
            quick_test: Test samples for quick search
            full_epochs: Epochs for full search
            full_train: Training samples for full search
            full_test: Test samples for full search
            two_stage: If True, run quick search first then full search
        """
        combinations = self.generate_combinations()
        n_combinations = len(combinations)

        print("\n" + "="*80)
        print(" " * 20 + "GRID SEARCH EXPERIMENT")
        print("="*80)
        print(f"Parameter grid: {self.param_grid}")
        print(f"Total combinations: {n_combinations}")
        print(f"Two-stage search: {two_stage}")

        if two_stage:
            # Estimate Stage 2 model count
            estimated_stage2 = self._estimate_stage2_count()
            print(f"\n>>> Expected Stage 2 models: ~{estimated_stage2} (after deduplication) <<<")
        else:
            estimated_stage2 = 0

        # Stage 1: Quick Search
        print("\n" + "-"*80)
        print("STAGE 1: Quick Search")
        print(f"Epochs: {quick_epochs}, Train: {quick_train}, Test: {quick_test}")
        print("-"*80)

        self.base_config.num_epochs = quick_epochs
        self.base_config.num_train_samples = quick_train
        self.base_config.num_test_samples = quick_test

        self.results = []
        first_model_time = None

        for i, params in enumerate(tqdm(combinations, desc="Quick Search")):
            # Measure first model time for estimation
            if i == 0:
                start_time = time.time()

            config = self._create_config(params, quick=True)
            metrics = self._run_single_experiment(config, experiment_id=i,
                                                   use_quick_data=True)

            # After first model, print time estimate
            if i == 0:
                first_model_time = time.time() - start_time
                self._print_time_estimate(
                    first_model_time=first_model_time,
                    n_quick=n_combinations,
                    n_full_est=estimated_stage2 if two_stage else 0,
                    quick_epochs=quick_epochs,
                    full_epochs=full_epochs
                )

            result = {
                'combination_id': i,
                **params,
                **metrics
            }
            self.results.append(result)

            # Print every trial's performance
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            tqdm.write(f"  [{i+1}/{n_combinations}] {params_str}")
            tqdm.write(f"    -> ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")
            if 'disturbing_roc_auc' in metrics:
                tqdm.write(f"    -> Disturbing Normal - ROC-AUC: {metrics['disturbing_roc_auc']:.4f}, F1: {metrics['disturbing_f1']:.4f}")

            # Print best model every 20 trials and at the end
            if (i + 1) % 20 == 0 or (i + 1) == n_combinations:
                best_idx = max(range(len(self.results)), key=lambda x: self.results[x]['roc_auc'])
                best_r = self.results[best_idx]
                # Get best model's parameters
                best_params = {k: best_r[k] for k in self.param_grid.keys() if k in best_r}
                best_params_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
                tqdm.write(f"  *** BEST so far (#{best_idx+1}): ROC-AUC={best_r['roc_auc']:.4f}, F1={best_r['f1_score']:.4f} ***")
                tqdm.write(f"      Params: {best_params_str}")

        quick_df = pd.DataFrame(self.results)
        quick_df = quick_df.sort_values('roc_auc', ascending=False).reset_index(drop=True)

        print(f"\nQuick Search Complete!")
        print(f"Best ROC-AUC: {quick_df['roc_auc'].max():.4f}")

        # Save quick search results
        quick_df.to_csv(os.path.join(self.output_dir, 'quick_search_results.csv'), index=False)

        # Stage 2: Full Search (if two_stage)
        if two_stage:
            print("\n" + "-"*80)
            print("STAGE 2: Full Search (Diverse Selection)")
            print(f"Epochs: {full_epochs}, Train: {full_train}, Test: {full_test}")
            print("-"*80)

            # Use diverse selection method
            print("\nSelecting diverse candidates for Stage 2:")
            top_combinations = self._select_stage2_candidates(quick_df)
            actual_k = len(top_combinations)
            print(f"\n>>> Stage 2 will train {actual_k} models (from {n_combinations} Stage 1 combinations) <<<")
            param_keys = list(self.param_grid.keys())

            self.full_results = []

            for i, (_, data) in enumerate(tqdm(top_combinations.iterrows(), total=actual_k, desc="Full Search")):
                params = {k: data[k] for k in param_keys}

                config = self._create_config(params, quick=False)
                config.num_epochs = full_epochs
                config.num_train_samples = full_train
                config.num_test_samples = full_test

                metrics = self._run_single_experiment(config, experiment_id=int(data['combination_id']),
                                                       save_history=True, use_quick_data=False)

                # Enhanced result with all metrics (matching Stage 1 format)
                result = {
                    'stage2_rank': i + 1,
                    'combination_id': int(data['combination_id']),
                    'selection_criterion': data.get('selection_criterion', 'unknown'),
                    **params,
                    # Main metrics
                    'roc_auc': metrics['roc_auc'],
                    'f1_score': metrics['f1_score'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    # Disturbing normal metrics (if available)
                    'disturbing_roc_auc': metrics.get('disturbing_roc_auc', None),
                    'disturbing_f1': metrics.get('disturbing_f1', None),
                    'disturbing_precision': metrics.get('disturbing_precision', None),
                    'disturbing_recall': metrics.get('disturbing_recall', None),
                    # Quick search reference
                    'quick_roc_auc': data['roc_auc'],
                    'quick_f1': data.get('f1_score', None),
                    'quick_disturbing_roc_auc': data.get('disturbing_roc_auc', None),
                    # Improvement from quick to full
                    'roc_auc_improvement': metrics['roc_auc'] - data['roc_auc'],
                }
                self.full_results.append(result)

                # Enhanced per-trial output
                params_str = ", ".join(f"{k}={v}" for k, v in params.items())
                tqdm.write(f"  [{i+1}/{actual_k}] criterion={data.get('selection_criterion', 'N/A')}")
                tqdm.write(f"    Params: {params_str}")
                tqdm.write(f"    -> ROC-AUC: {metrics['roc_auc']:.4f} (quick: {data['roc_auc']:.4f}, Δ={metrics['roc_auc'] - data['roc_auc']:+.4f})")
                tqdm.write(f"    -> F1: {metrics['f1_score']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
                if 'disturbing_roc_auc' in metrics and metrics['disturbing_roc_auc'] is not None:
                    tqdm.write(f"    -> Disturbing Normal - ROC-AUC: {metrics['disturbing_roc_auc']:.4f}, F1: {metrics['disturbing_f1']:.4f}")

            full_df = pd.DataFrame(self.full_results)
            full_df = full_df.sort_values('roc_auc', ascending=False).reset_index(drop=True)

            print(f"\nFull Search Complete!")
            print(f"Best ROC-AUC: {full_df['roc_auc'].max():.4f}")
            print(f"Best F1-Score: {full_df['f1_score'].max():.4f}")
            if 'disturbing_roc_auc' in full_df.columns and full_df['disturbing_roc_auc'].notna().any():
                print(f"Best Disturbing ROC-AUC: {full_df['disturbing_roc_auc'].max():.4f}")

            # Print top 10 results summary
            print("\n--- Top 10 Stage 2 Results ---")
            for i, row in full_df.head(10).iterrows():
                print(f"  #{i+1}: ROC-AUC={row['roc_auc']:.4f}, F1={row['f1_score']:.4f}, "
                      f"criterion={row['selection_criterion']}")

            # Print per-parameter performance summary (including disturbing normal)
            self._print_parameter_performance_summary(full_df)

            # Save full search results
            full_df.to_csv(os.path.join(self.output_dir, 'full_search_results.csv'), index=False)

            final_df = full_df
        else:
            final_df = quick_df

        # Save all results
        self._save_all_results()

        return final_df

    def _select_stage2_candidates(self, quick_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select diverse candidates for Stage 2 training.

        Selection strategy:
        1. First, select top 10 from each parameter value (ensures coverage)
        2. Remove duplicates
        3. Add top 30 by overall ROC-AUC (skip already selected)
        4. Add top 20 by disturbing_roc_auc (skip already selected)

        Returns unique candidates with selection info.
        """
        selected_ids = set()
        selection_info = []

        def add_candidates(df_subset, criterion_name, max_count):
            """Add candidates from subset, skip duplicates."""
            added = 0
            for _, row in df_subset.iterrows():
                cid = int(row['combination_id'])
                if cid not in selected_ids:
                    selected_ids.add(cid)
                    selection_info.append({
                        'combination_id': cid,
                        'selection_criterion': criterion_name,
                        'criterion_rank': added + 1
                    })
                    added += 1
                    if added >= max_count:
                        break
            return added

        n_per_value = 5  # Top 5 per parameter value

        # 1. Select top 5 from each parameter value
        print("  [Phase 1] Selecting top 5 per parameter value:")

        # force_mask_anomaly: True and False
        if 'force_mask_anomaly' in quick_df.columns:
            for val in [True, False]:
                df_subset = quick_df[quick_df['force_mask_anomaly'] == val].sort_values('roc_auc', ascending=False)
                added = add_candidates(df_subset, f'force_mask_anomaly={val}', n_per_value)
                print(f"    force_mask_anomaly={val}: {added} models")

        # patch_level_loss: True and False
        if 'patch_level_loss' in quick_df.columns:
            for val in [True, False]:
                df_subset = quick_df[quick_df['patch_level_loss'] == val].sort_values('roc_auc', ascending=False)
                added = add_candidates(df_subset, f'patch_level_loss={val}', n_per_value)
                print(f"    patch_level_loss={val}: {added} models")

        # margin_type: hinge, softplus, dynamic
        if 'margin_type' in quick_df.columns:
            for val in ['hinge', 'softplus', 'dynamic']:
                df_subset = quick_df[quick_df['margin_type'] == val].sort_values('roc_auc', ascending=False)
                added = add_candidates(df_subset, f'margin_type={val}', n_per_value)
                print(f"    margin_type={val}: {added} models")

        # patchify_mode: cnn_first, patch_cnn, linear
        if 'patchify_mode' in quick_df.columns:
            for val in ['cnn_first', 'patch_cnn', 'linear']:
                df_subset = quick_df[quick_df['patchify_mode'] == val].sort_values('roc_auc', ascending=False)
                added = add_candidates(df_subset, f'patchify_mode={val}', n_per_value)
                print(f"    patchify_mode={val}: {added} models")

        # masking_strategy: patch, feature_wise
        if 'masking_strategy' in quick_df.columns:
            for val in ['patch', 'feature_wise']:
                df_subset = quick_df[quick_df['masking_strategy'] == val].sort_values('roc_auc', ascending=False)
                added = add_candidates(df_subset, f'masking_strategy={val}', n_per_value)
                print(f"    masking_strategy={val}: {added} models")

        # masking_ratio
        if 'masking_ratio' in quick_df.columns:
            for val in quick_df['masking_ratio'].unique():
                df_subset = quick_df[quick_df['masking_ratio'] == val].sort_values('roc_auc', ascending=False)
                added = add_candidates(df_subset, f'masking_ratio={val}', n_per_value)
                print(f"    masking_ratio={val}: {added} models")

        # num_patches
        if 'num_patches' in quick_df.columns:
            for val in quick_df['num_patches'].unique():
                df_subset = quick_df[quick_df['num_patches'] == val].sort_values('roc_auc', ascending=False)
                added = add_candidates(df_subset, f'num_patches={val}', n_per_value)
                print(f"    num_patches={val}: {added} models")

        print(f"  After Phase 1: {len(selected_ids)} unique models")

        # 2. Add top 10 by overall ROC-AUC (excluding already selected)
        print("  [Phase 2] Adding top by overall ROC-AUC (excluding Phase 1):")
        df_by_roc = quick_df.sort_values('roc_auc', ascending=False)
        added = add_candidates(df_by_roc, 'overall_roc_auc', 10)
        print(f"    Added {added} new models by overall ROC-AUC")

        # 3. Add top 5 by disturbing_roc_auc (excluding already selected)
        print("  [Phase 3] Adding top by disturbing ROC-AUC (excluding Phase 1, 2):")
        if 'disturbing_roc_auc' in quick_df.columns:
            df_by_disturbing = quick_df.sort_values('disturbing_roc_auc', ascending=False)
            added = add_candidates(df_by_disturbing, 'disturbing_roc_auc', 5)
            print(f"    Added {added} new models by disturbing ROC-AUC")

        # Create the selection dataframe
        selection_df = pd.DataFrame(selection_info)
        selected_df = quick_df[quick_df['combination_id'].isin(selected_ids)].copy()
        selected_df = selected_df.merge(selection_df, on='combination_id', how='left')

        # Sort by ROC-AUC but keep selection info
        selected_df = selected_df.sort_values('roc_auc', ascending=False).reset_index(drop=True)

        print(f"\n  Total unique models selected: {len(selected_df)}")

        return selected_df

    def _print_parameter_performance_summary(self, full_df: pd.DataFrame):
        """Print per-parameter performance summary including disturbing normal metrics."""
        print("\n" + "="*80)
        print(" " * 15 + "PARAMETER PERFORMANCE SUMMARY")
        print("="*80)

        # Parameters to analyze
        params_to_analyze = [
            ('force_mask_anomaly', [True, False]),
            ('patch_level_loss', [True, False]),
            ('margin_type', ['hinge', 'softplus', 'dynamic']),
            ('patchify_mode', ['cnn_first', 'patch_cnn', 'linear']),
            ('masking_strategy', ['patch', 'feature_wise']),
        ]

        # Add masking_ratio and num_patches if present
        if 'masking_ratio' in full_df.columns:
            params_to_analyze.append(('masking_ratio', sorted(full_df['masking_ratio'].unique())))
        if 'num_patches' in full_df.columns:
            params_to_analyze.append(('num_patches', sorted(full_df['num_patches'].unique())))

        for param_name, values in params_to_analyze:
            if param_name not in full_df.columns:
                continue

            print(f"\n--- {param_name} ---")
            print(f"{'Value':<20} {'Count':>6} {'ROC-AUC':>10} {'F1':>10} {'Dist.ROC':>10} {'Dist.F1':>10}")
            print("-" * 70)

            for val in values:
                subset = full_df[full_df[param_name] == val]
                if len(subset) == 0:
                    continue

                count = len(subset)
                mean_roc = subset['roc_auc'].mean()
                mean_f1 = subset['f1_score'].mean()

                # Disturbing normal metrics
                if 'disturbing_roc_auc' in subset.columns and subset['disturbing_roc_auc'].notna().any():
                    mean_dist_roc = subset['disturbing_roc_auc'].mean()
                    mean_dist_f1 = subset['disturbing_f1'].mean() if 'disturbing_f1' in subset.columns else float('nan')
                else:
                    mean_dist_roc = float('nan')
                    mean_dist_f1 = float('nan')

                val_str = str(val)[:20]
                dist_roc_str = f"{mean_dist_roc:.4f}" if not np.isnan(mean_dist_roc) else "N/A"
                dist_f1_str = f"{mean_dist_f1:.4f}" if not np.isnan(mean_dist_f1) else "N/A"

                print(f"{val_str:<20} {count:>6} {mean_roc:>10.4f} {mean_f1:>10.4f} {dist_roc_str:>10} {dist_f1_str:>10}")

        print("\n" + "="*80)

    def _save_all_results(self):
        """Save all experiment results"""
        # Save best config
        if self.best_config is not None:
            best_params = {}
            for key in self.param_grid.keys():
                best_params[key] = getattr(self.best_config, key)

            best_params['patch_size'] = self.best_config.patch_size
            best_params['mask_last_n'] = self.best_config.mask_last_n

            with open(os.path.join(self.output_dir, 'best_config.json'), 'w') as f:
                json.dump(best_params, f, indent=2)

        # Save best model
        if self.best_model is not None:
            torch.save({
                'model_state_dict': self.best_model.state_dict(),
                'config': asdict(self.best_config),
                'metrics': self.best_metrics
            }, os.path.join(self.output_dir, 'best_model.pt'))

            # Save detailed results for best model (anomaly type analysis)
            self._save_best_model_detailed_results()

        # Save training histories
        if self.histories:
            # Convert numpy arrays to lists for JSON serialization
            histories_serializable = {}
            for exp_id, history in self.histories.items():
                histories_serializable[str(exp_id)] = {
                    k: [float(v) if isinstance(v, (int, float, np.floating)) else v for v in vals]
                    for k, vals in history.items()
                }
            with open(os.path.join(self.output_dir, 'training_histories.json'), 'w') as f:
                json.dump(histories_serializable, f, indent=2)

        # Save experiment metadata
        metadata = {
            'param_grid': {k: [str(v) if isinstance(v, bool) else v for v in vals]
                          for k, vals in self.param_grid.items()},
            'total_combinations': len(self.generate_combinations()),
            'stage1_results_count': len(self.results),
            'stage2_results_count': len(self.full_results),
            'best_metrics': self.best_metrics,
            'timestamp': datetime.now().isoformat()
        }
        with open(os.path.join(self.output_dir, 'experiment_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"\nAll results saved to {self.output_dir}")
        print(f"  - quick_search_results.csv")
        if self.full_results:
            print(f"  - full_search_results.csv")
        print(f"  - best_model.pt")
        print(f"  - best_config.json")
        print(f"  - best_model_detailed.csv")
        print(f"  - anomaly_type_metrics.json")
        if self.histories:
            print(f"  - training_histories.json")
        print(f"  - experiment_metadata.json")

    def _save_best_model_detailed_results(self):
        """Save detailed results for best model including anomaly type analysis"""
        if self.best_model is None or self.best_config is None:
            return

        print("\nGenerating detailed results for best model...")

        # Create test dataset with same parameters used for final evaluation (full search)
        test_dataset = SlidingWindowDataset(
            signals=self.signals,
            point_labels=self.point_labels,
            anomaly_regions=self.anomaly_regions,
            window_size=self.best_config.seq_length,
            stride=self.best_config.sliding_window_stride,
            mask_last_n=self.best_config.mask_last_n,
            split='test',
            train_ratio=self.full_train_ratio,  # Use full search train_ratio
            target_counts={
                'pure_normal': self.best_config.test_target_pure_normal,
                'disturbing_normal': self.best_config.test_target_disturbing_normal,
                'anomaly': self.best_config.test_target_anomaly
            },
            seed=self.best_config.random_seed
        )
        test_loader = DataLoader(test_dataset, batch_size=self.best_config.batch_size, shuffle=False)

        # Create evaluator
        evaluator = Evaluator(self.best_model, self.best_config, test_loader)

        # Get detailed losses
        detailed_losses = evaluator.compute_detailed_losses()

        # Save detailed results as CSV
        detailed_df = pd.DataFrame({
            'reconstruction_loss': detailed_losses['reconstruction_loss'],
            'discrepancy_loss': detailed_losses['discrepancy_loss'],
            'total_loss': detailed_losses['total_loss'],
            'label': detailed_losses['labels'],
            'sample_type': detailed_losses['sample_types'],
            'anomaly_type': detailed_losses['anomaly_types'],
            'anomaly_type_name': [SLIDING_ANOMALY_TYPE_NAMES[int(at)] for at in detailed_losses['anomaly_types']]
        })
        detailed_df.to_csv(os.path.join(self.output_dir, 'best_model_detailed.csv'), index=False)

        # Get performance by anomaly type
        anomaly_type_metrics = evaluator.get_performance_by_anomaly_type()

        # Add summary statistics per anomaly type
        for atype_name in anomaly_type_metrics.keys():
            atype_mask = detailed_df['anomaly_type_name'] == atype_name
            if atype_mask.sum() > 0:
                anomaly_type_metrics[atype_name]['mean_reconstruction_loss'] = float(
                    detailed_df.loc[atype_mask, 'reconstruction_loss'].mean()
                )
                anomaly_type_metrics[atype_name]['std_reconstruction_loss'] = float(
                    detailed_df.loc[atype_mask, 'reconstruction_loss'].std()
                )
                anomaly_type_metrics[atype_name]['mean_discrepancy_loss'] = float(
                    detailed_df.loc[atype_mask, 'discrepancy_loss'].mean()
                )
                anomaly_type_metrics[atype_name]['std_discrepancy_loss'] = float(
                    detailed_df.loc[atype_mask, 'discrepancy_loss'].std()
                )

        # Save anomaly type metrics as JSON
        with open(os.path.join(self.output_dir, 'anomaly_type_metrics.json'), 'w') as f:
            json.dump(anomaly_type_metrics, f, indent=2)

        print(f"  - Saved best_model_detailed.csv ({len(detailed_df)} samples)")
        print(f"  - Saved anomaly_type_metrics.json ({len(anomaly_type_metrics)} types)")


# =============================================================================
# Main Function
# =============================================================================

def run_experiments(
    param_grid: Dict[str, List] = None,
    quick_epochs: int = 1,
    quick_train: int = 1000,
    quick_test: int = 400,
    full_epochs: int = 2,
    full_train: int = 2000,
    full_test: int = 500,
    two_stage: bool = True,
    output_dir: str = 'results/experiments'
) -> Tuple[pd.DataFrame, str]:
    """
    Run complete experiment pipeline (without visualization)

    Results are saved to output_dir/YYYYMMDD_HHMMSS/
    Use visualize_all.py to generate visualizations from saved results.

    Args:
        param_grid: Parameter grid to search
        quick_epochs: Epochs for quick search
        quick_train: Training samples for quick search
        quick_test: Test samples for quick search
        full_epochs: Epochs for full search
        full_train: Training samples for full search
        full_test: Test samples for full search
        two_stage: If True, run quick then full search
        output_dir: Directory to save results

    Returns:
        (results_df, output_dir_path)
    """
    print("\n" + "="*80)
    print(" " * 15 + "SELF-DISTILLED MAE EXPERIMENTS")
    print("="*80)

    # Initialize runner
    runner = ExperimentRunner(
        param_grid=param_grid,
        output_dir=output_dir
    )

    # Run grid search
    results_df = runner.run_grid_search(
        quick_epochs=quick_epochs,
        quick_train=quick_train,
        quick_test=quick_test,
        full_epochs=full_epochs,
        full_train=full_train,
        full_test=full_test,
        two_stage=two_stage
    )

    # Print best configuration
    print("\n" + "="*80)
    print(" " * 20 + "BEST CONFIGURATION")
    print("="*80)

    best_row = results_df.iloc[0]
    print("\nParameters:")
    for key in runner.param_grid.keys():
        print(f"  {key}: {best_row[key]}")

    if 'num_patches' in runner.param_grid:
        print(f"  patch_size: {100 // int(best_row['num_patches'])}")

    print("\nMetrics:")
    print(f"  ROC-AUC: {best_row['roc_auc']:.4f}")
    print(f"  F1-Score: {best_row['f1_score']:.4f}")
    print(f"  Precision: {best_row['precision']:.4f}")
    print(f"  Recall: {best_row['recall']:.4f}")

    print("\n" + "="*80)
    print(" " * 15 + "EXPERIMENTS COMPLETE!")
    print(f"Results saved to: {runner.output_dir}")
    print(f"\nTo generate visualizations, run:")
    print(f"  python scripts/visualize_all.py --experiment-dir {runner.output_dir}")
    print("="*80)

    return results_df, runner.output_dir


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Self-Distilled MAE experiments')
    parser.add_argument('--quick-epochs', type=int, default=1, help='Epochs for quick search')
    parser.add_argument('--quick-train', type=int, default=1000, help='Training samples for quick search')
    parser.add_argument('--quick-test', type=int, default=400, help='Test samples for quick search')
    parser.add_argument('--full-epochs', type=int, default=2, help='Epochs for full search')
    parser.add_argument('--full-train', type=int, default=2000, help='Training samples for full search')
    parser.add_argument('--full-test', type=int, default=500, help='Test samples for full search')
    parser.add_argument('--no-two-stage', action='store_true', help='Disable two-stage search')
    parser.add_argument('--output-dir', type=str, default='results/experiments', help='Output directory')
    args = parser.parse_args()

    run_experiments(
        quick_epochs=args.quick_epochs,
        quick_train=args.quick_train,
        quick_test=args.quick_test,
        full_epochs=args.full_epochs,
        full_train=args.full_train,
        full_test=args.full_test,
        two_stage=not args.no_two_stage,
        output_dir=args.output_dir
    )
