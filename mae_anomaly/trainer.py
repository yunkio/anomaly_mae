"""
Trainer for Self-Distilled MAE Anomaly Detection
"""

import time
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from typing import Dict
from tqdm import tqdm

from .loss import SelfDistillationLoss


class Trainer:
    """Trainer class for Self-Distilled MAE"""

    def __init__(
        self,
        model,
        config,
        train_loader: DataLoader,
        test_loader: DataLoader = None,
        verbose: bool = True
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.verbose = verbose

        self.criterion = SelfDistillationLoss(config)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )

        self.model = self.model.to(config.device)

        # Mixed Precision Training (AMP)
        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None

        self.history = {
            'train_loss': [],
            'train_rec_loss': [],
            'train_disc_loss': [],
            'train_normal_loss': [],
            'train_anomaly_loss': [],
            # Detailed metrics by sample type
            'train_teacher_recon_normal': [],
            'train_teacher_recon_anomaly': [],
            'train_student_recon_normal': [],
            'train_student_recon_anomaly': [],
            # Epoch-wise contribution ratios by sample type (test set)
            'epoch_recon_ratio_normal': [],
            'epoch_recon_ratio_disturbing': [],
            'epoch_recon_ratio_anomaly': [],
            'epoch_disc_ratio_normal': [],
            'epoch_disc_ratio_disturbing': [],
            'epoch_disc_ratio_anomaly': [],
            # Epoch-wise absolute scores by sample type (test set) - WEIGHTED
            'epoch_recon_score_normal': [],
            'epoch_recon_score_disturbing': [],
            'epoch_recon_score_anomaly': [],
            'epoch_disc_score_normal': [],
            'epoch_disc_score_disturbing': [],
            'epoch_disc_score_anomaly': [],
            # Epoch-wise RAW scores by sample type (test set) - UNWEIGHTED
            'epoch_raw_recon_normal': [],
            'epoch_raw_recon_disturbing': [],
            'epoch_raw_recon_anomaly': [],
            'epoch_raw_disc_normal': [],
            'epoch_raw_disc_disturbing': [],
            'epoch_raw_disc_anomaly': [],
            # Epoch-wise scores by anomaly type (test set)
            'epoch_anomaly_type_scores': [],  # List of dicts per epoch
            'epoch': []
        }

    def _compute_warmup_factor(self, epoch: int) -> float:
        if epoch < self.config.warmup_epochs:
            return epoch / self.config.warmup_epochs
        return 1.0

    def train_epoch(self, epoch: int, teacher_only: bool = False,
                    profile_batches: int = 0) -> Dict[str, float]:
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'discrepancy_loss': 0.0,
            'normal_loss': 0.0,
            'anomaly_loss': 0.0,
            'mean_discrepancy': 0.0,
            # Detailed metrics by sample type
            'teacher_recon_normal': 0.0,
            'teacher_recon_anomaly': 0.0,
            'student_recon_normal': 0.0,
            'student_recon_anomaly': 0.0,
        }

        warmup_factor = self._compute_warmup_factor(epoch)

        # Batch-level profiling: first N batches with cuda.synchronize() per component
        batch_profiles = [] if profile_batches > 0 else None

        # Epoch-level timing (sync only at epoch boundaries → ~1% overhead)
        torch.cuda.synchronize()
        t_epoch_start = time.time()
        t_forward_acc = 0.0
        t_backward_acc = 0.0

        iterator = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}',
                        disable=not self.verbose, leave=False)

        for batch_idx, batch in enumerate(iterator):
            do_profile = batch_profiles is not None and batch_idx < profile_batches

            # Support 3-tuple, 4-tuple, and 5-tuple returns from dataset
            if len(batch) == 5:
                sequences, window_labels, point_labels, sample_types, anomaly_types = batch
            elif len(batch) == 4:
                sequences, window_labels, point_labels, sample_types = batch
            else:
                sequences, window_labels, point_labels = batch

            # Data transfer to GPU
            if do_profile:
                torch.cuda.synchronize()
                t_bp_start = time.time()

            sequences = sequences.to(self.config.device)
            point_labels = point_labels.to(self.config.device)

            if do_profile:
                torch.cuda.synchronize()
                t_bp_data = time.time()

            # Forward pass with AMP
            t_fwd = time.time()
            with autocast('cuda', enabled=self.use_amp):
                teacher_output, student_output, mask = self.model(sequences, point_labels=point_labels)

                if do_profile:
                    torch.cuda.synchronize()
                    t_bp_model = time.time()

                loss, loss_dict = self.criterion(
                    teacher_output, student_output, sequences, mask, point_labels, warmup_factor,
                    teacher_only=teacher_only
                )

            if do_profile:
                torch.cuda.synchronize()
                t_bp_loss = time.time()

            # Backward pass with AMP
            t_bwd = time.time()
            t_forward_acc += t_bwd - t_fwd

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if do_profile:
                    torch.cuda.synchronize()
                    t_bp_backward = time.time()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if do_profile:
                    torch.cuda.synchronize()
                    t_bp_backward = time.time()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            if do_profile:
                torch.cuda.synchronize()
                t_bp_optim = time.time()
                batch_profiles.append({
                    'batch': batch_idx,
                    'data_to_gpu_ms': (t_bp_data - t_bp_start) * 1000,
                    'model_forward_ms': (t_bp_model - t_bp_data) * 1000,
                    'loss_compute_ms': (t_bp_loss - t_bp_model) * 1000,
                    'backward_ms': (t_bp_backward - t_bp_loss) * 1000,
                    'optimizer_step_ms': (t_bp_optim - t_bp_backward) * 1000,
                    'total_ms': (t_bp_optim - t_bp_start) * 1000,
                })

            t_backward_acc += time.time() - t_bwd

            for key in epoch_losses.keys():
                epoch_losses[key] += loss_dict[key]

        torch.cuda.synchronize()
        t_epoch_total = time.time() - t_epoch_start

        for key in epoch_losses.keys():
            epoch_losses[key] /= len(self.train_loader)

        # Attach timing (CPU wall clock; forward/backward are approximate due to CUDA async)
        epoch_losses['_timing'] = {
            'epoch_total': t_epoch_total,
            'forward_approx': t_forward_acc,
            'backward_approx': t_backward_acc,
            'n_batches': len(self.train_loader),
        }

        # Attach batch profiling if measured
        if batch_profiles:
            epoch_losses['_batch_profiling'] = batch_profiles

        return epoch_losses

    def _compute_test_contrib_ratios(self) -> Dict:
        """Compute contribution ratios and absolute scores by sample type on test set

        Returns:
            Dict with recon_ratio, disc_ratio, recon_score, disc_score for each sample type
            and anomaly_type_scores dict for each anomaly type
        """
        import numpy as np
        from mae_anomaly import SLIDING_ANOMALY_TYPE_NAMES

        empty_results = {
            'recon_ratio_normal': 0.0, 'disc_ratio_normal': 0.0,
            'recon_ratio_disturbing': 0.0, 'disc_ratio_disturbing': 0.0,
            'recon_ratio_anomaly': 0.0, 'disc_ratio_anomaly': 0.0,
            'recon_score_normal': 0.0, 'disc_score_normal': 0.0,
            'recon_score_disturbing': 0.0, 'disc_score_disturbing': 0.0,
            'recon_score_anomaly': 0.0, 'disc_score_anomaly': 0.0,
            'raw_recon_normal': 0.0, 'raw_recon_disturbing': 0.0, 'raw_recon_anomaly': 0.0,
            'raw_disc_normal': 0.0, 'raw_disc_disturbing': 0.0, 'raw_disc_anomaly': 0.0,
            'anomaly_type_scores': {}
        }

        if self.test_loader is None:
            return empty_results

        from mae_anomaly.evaluator import Evaluator

        # All-patches inference via Evaluator (correct inference mechanism)
        self.model.eval()
        evaluator = Evaluator(self.model, self.config, self.test_loader,
                              test_dataset=self.test_dataset if hasattr(self, 'test_dataset') else None)
        recon_patches, disc_patches, student_recon_patches, labels, sample_types_all, anomaly_types_all = \
            evaluator._compute_patch_scores_all_patches()
        del evaluator

        # Window-level scores (mean over patches)
        recon_all = recon_patches.mean(axis=1)
        disc_all = disc_patches.mean(axis=1)

        # Compute contributions based on scoring mode
        score_mode = self.config.anomaly_score_mode
        lambda_disc = self.config.lambda_disc

        if score_mode == 'adaptive':
            adaptive_lambda = recon_all.mean() / (disc_all.mean() + 1e-4)
            recon_contrib_all = recon_all
            disc_contrib_all = adaptive_lambda * disc_all
            total = recon_contrib_all + disc_contrib_all + 1e-4
            recon_ratio_all = recon_contrib_all / total
            disc_ratio_all = disc_contrib_all / total
        else:  # default
            recon_contrib_all = recon_all
            disc_contrib_all = lambda_disc * disc_all
            total = recon_contrib_all + disc_contrib_all + 1e-4
            recon_ratio_all = recon_contrib_all / total
            disc_ratio_all = disc_contrib_all / total

        # Compute mean ratios and absolute scores by sample type
        # sample_type: 0=pure_normal, 1=disturbing_normal, 2=anomaly
        results = {}
        for type_idx, type_name in [(0, 'normal'), (1, 'disturbing'), (2, 'anomaly')]:
            mask = (sample_types_all == type_idx)
            if mask.sum() > 0:
                results[f'recon_ratio_{type_name}'] = float(recon_ratio_all[mask].mean() * 100)  # as percentage
                results[f'disc_ratio_{type_name}'] = float(disc_ratio_all[mask].mean() * 100)
                results[f'recon_score_{type_name}'] = float(recon_contrib_all[mask].mean())
                results[f'disc_score_{type_name}'] = float(disc_contrib_all[mask].mean())
                # RAW (unweighted) values
                results[f'raw_recon_{type_name}'] = float(recon_all[mask].mean())
                results[f'raw_disc_{type_name}'] = float(disc_all[mask].mean())
            else:
                results[f'recon_ratio_{type_name}'] = 0.0
                results[f'disc_ratio_{type_name}'] = 0.0
                results[f'recon_score_{type_name}'] = 0.0
                results[f'disc_score_{type_name}'] = 0.0
                results[f'raw_recon_{type_name}'] = 0.0
                results[f'raw_disc_{type_name}'] = 0.0

        # Compute scores by anomaly type (only for anomaly samples, sample_type=2)
        anomaly_type_scores = {}
        anomaly_mask = (sample_types_all == 2)
        # Discover all anomaly types from actual data (supports >9 types, e.g. TEP)
        unique_atypes = sorted(set(int(x) for x in np.unique(anomaly_types_all[anomaly_mask]))) if anomaly_mask.any() else []
        for atype_idx in unique_atypes:
            if atype_idx < len(SLIDING_ANOMALY_TYPE_NAMES):
                atype_name = SLIDING_ANOMALY_TYPE_NAMES[atype_idx]
            else:
                atype_name = f'fault_{atype_idx}'
            atype_mask = anomaly_mask & (anomaly_types_all == atype_idx)
            if atype_mask.sum() > 0:
                anomaly_type_scores[atype_name] = {
                    'recon_score': float(recon_contrib_all[atype_mask].mean()),
                    'disc_score': float(disc_contrib_all[atype_mask].mean()),
                    'recon_ratio': float(recon_ratio_all[atype_mask].mean() * 100),
                    'disc_ratio': float(disc_ratio_all[atype_mask].mean() * 100),
                    'count': int(atype_mask.sum())
                }
        # Also add normal (sample_type=0) for comparison
        normal_mask = (sample_types_all == 0)
        if normal_mask.sum() > 0:
            anomaly_type_scores['normal'] = {
                'recon_score': float(recon_contrib_all[normal_mask].mean()),
                'disc_score': float(disc_contrib_all[normal_mask].mean()),
                'recon_ratio': float(recon_ratio_all[normal_mask].mean() * 100),
                'disc_ratio': float(disc_ratio_all[normal_mask].mean() * 100),
                'count': int(normal_mask.sum())
            }

        results['anomaly_type_scores'] = anomaly_type_scores

        self.model.train()  # Restore training mode
        return results

    def _print_batch_profiling(self, batch_profiles, epoch_timing):
        """Print batch profiling summary table immediately after epoch 0."""
        n = len(batch_profiles)
        n_batches = epoch_timing.get('n_batches', len(self.train_loader))
        epoch_total = epoch_timing.get('epoch_total', 0)

        components = ['data_to_gpu_ms', 'model_forward_ms', 'loss_compute_ms',
                      'backward_ms', 'optimizer_step_ms']
        labels = ['Data -> GPU', 'Model Forward', 'Loss Compute',
                  'Backward', 'Optimizer Step']

        total_sum = 0.0
        rows = []
        for comp, label in zip(components, labels):
            vals = [bp[comp] for bp in batch_profiles]
            total = sum(vals)
            total_sum += total
            rows.append((label, total, total / n, min(vals), max(vals)))

        avg_batch_ms = total_sum / n
        est_epoch_s = avg_batch_ms * n_batches / 1000
        remaining = self.config.num_epochs - 1
        est_remaining_s = est_epoch_s * remaining

        hdr = f"{'Component':<20} {'Total(ms)':>10} {'Avg(ms)':>10} {'Min':>10} {'Max':>10}"
        sep = '-' * len(hdr)
        print(f"\n  Batch Profiling ({n} batches, batch_size={self.config.batch_size})")
        print(f"  {sep}")
        print(f"  {hdr}")
        print(f"  {sep}")
        for label, total, avg, mn, mx in rows:
            print(f"  {label:<20} {total:>10.1f} {avg:>10.1f} {mn:>10.1f} {mx:>10.1f}")
        print(f"  {sep}")
        print(f"  {'TOTAL':<20} {total_sum:>10.1f} {avg_batch_ms:>10.1f}")
        print(f"  {sep}")
        print(f"  Epoch 1 actual: {epoch_total:.1f}s | "
              f"Est. per epoch (train only): {est_epoch_s:.1f}s | "
              f"Est. remaining ({remaining} epochs): {est_remaining_s:.0f}s ({est_remaining_s/60:.1f}min)\n")

    def train(self, epoch_callback=None, profile_n_batches: int = 0) -> Dict:
        """Train the model for num_epochs.

        Args:
            epoch_callback: Optional callable(epoch, model, history) invoked at end of each epoch.
                           Use for lightweight epoch-wise test evaluation.
            profile_n_batches: If > 0, profile first N batches of epoch 0 with per-component
                              cuda.synchronize() timing. Results stored in history['batch_profiling'].
        """
        teacher_warmup = self.config.teacher_only_warmup_epochs
        for epoch in range(self.config.num_epochs):
            # First N epochs are warm-up: train teacher only (no discrepancy/student loss)
            teacher_only = (epoch < teacher_warmup)
            # Profile only on epoch 0
            pb = profile_n_batches if epoch == 0 else 0
            epoch_losses = self.train_epoch(epoch, teacher_only=teacher_only, profile_batches=pb)
            self.scheduler.step()

            # Extract and record per-epoch timing from train_epoch
            epoch_timing = epoch_losses.pop('_timing', {})
            batch_profiling = epoch_losses.pop('_batch_profiling', None)
            if batch_profiling:
                self.history['batch_profiling'] = batch_profiling
                self._print_batch_profiling(batch_profiling, epoch_timing)

            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(epoch_losses['total_loss'])
            self.history['train_rec_loss'].append(epoch_losses['reconstruction_loss'])
            self.history['train_disc_loss'].append(epoch_losses['discrepancy_loss'])
            self.history['train_normal_loss'].append(epoch_losses['normal_loss'])
            self.history['train_anomaly_loss'].append(epoch_losses['anomaly_loss'])
            # Detailed metrics by sample type
            self.history['train_teacher_recon_normal'].append(epoch_losses['teacher_recon_normal'])
            self.history['train_teacher_recon_anomaly'].append(epoch_losses['teacher_recon_anomaly'])
            self.history['train_student_recon_normal'].append(epoch_losses['student_recon_normal'])
            self.history['train_student_recon_anomaly'].append(epoch_losses['student_recon_anomaly'])

            # Compute and record contribution ratios by sample type (on test set)
            eval_interval = self.config.eval_interval
            is_eval_epoch = ((epoch + 1) % eval_interval == 0) or (epoch == self.config.num_epochs - 1)
            t_contrib_start = time.time()
            if is_eval_epoch:
                contrib_ratios = self._compute_test_contrib_ratios()
                self._last_contrib_ratios = contrib_ratios
            else:
                contrib_ratios = getattr(self, '_last_contrib_ratios', self._compute_test_contrib_ratios())
            t_contrib = time.time() - t_contrib_start
            self.history['epoch_recon_ratio_normal'].append(contrib_ratios['recon_ratio_normal'])
            self.history['epoch_recon_ratio_disturbing'].append(contrib_ratios['recon_ratio_disturbing'])
            self.history['epoch_recon_ratio_anomaly'].append(contrib_ratios['recon_ratio_anomaly'])
            self.history['epoch_disc_ratio_normal'].append(contrib_ratios['disc_ratio_normal'])
            self.history['epoch_disc_ratio_disturbing'].append(contrib_ratios['disc_ratio_disturbing'])
            self.history['epoch_disc_ratio_anomaly'].append(contrib_ratios['disc_ratio_anomaly'])
            # Absolute scores by sample type (weighted)
            self.history['epoch_recon_score_normal'].append(contrib_ratios['recon_score_normal'])
            self.history['epoch_recon_score_disturbing'].append(contrib_ratios['recon_score_disturbing'])
            self.history['epoch_recon_score_anomaly'].append(contrib_ratios['recon_score_anomaly'])
            self.history['epoch_disc_score_normal'].append(contrib_ratios['disc_score_normal'])
            self.history['epoch_disc_score_disturbing'].append(contrib_ratios['disc_score_disturbing'])
            self.history['epoch_disc_score_anomaly'].append(contrib_ratios['disc_score_anomaly'])
            # RAW scores by sample type (unweighted)
            self.history['epoch_raw_recon_normal'].append(contrib_ratios['raw_recon_normal'])
            self.history['epoch_raw_recon_disturbing'].append(contrib_ratios['raw_recon_disturbing'])
            self.history['epoch_raw_recon_anomaly'].append(contrib_ratios['raw_recon_anomaly'])
            self.history['epoch_raw_disc_normal'].append(contrib_ratios['raw_disc_normal'])
            self.history['epoch_raw_disc_disturbing'].append(contrib_ratios['raw_disc_disturbing'])
            self.history['epoch_raw_disc_anomaly'].append(contrib_ratios['raw_disc_anomaly'])
            # Scores by anomaly type
            self.history['epoch_anomaly_type_scores'].append(contrib_ratios['anomaly_type_scores'])

            # Epoch callback (for epoch-wise test evaluation)
            t_callback_start = time.time()
            if epoch_callback is not None:
                epoch_callback(epoch, self.model, self.history)
            t_callback = time.time() - t_callback_start

            # Record per-epoch timing
            epoch_timing['contrib_ratios'] = t_contrib
            epoch_timing['callback'] = t_callback
            self.history.setdefault('epoch_timings', []).append(epoch_timing)

        return self.history
