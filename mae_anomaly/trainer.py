"""
Trainer for Self-Distilled MAE Anomaly Detection
"""

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
        self.use_amp = getattr(config, 'use_amp', False) and torch.cuda.is_available()
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

    def train_epoch(self, epoch: int, teacher_only: bool = False) -> Dict[str, float]:
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

        iterator = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}',
                        disable=not self.verbose, leave=False)

        for batch in iterator:
            # Support 3-tuple, 4-tuple, and 5-tuple returns from dataset
            if len(batch) == 5:
                sequences, last_patch_labels, point_labels, sample_types, anomaly_types = batch
            elif len(batch) == 4:
                sequences, last_patch_labels, point_labels, sample_types = batch
            else:
                sequences, last_patch_labels, point_labels = batch

            sequences = sequences.to(self.config.device)
            point_labels = point_labels.to(self.config.device)

            # Forward pass with AMP
            with autocast('cuda', enabled=self.use_amp):
                teacher_output, student_output, mask = self.model(sequences, point_labels=point_labels)
                loss, loss_dict = self.criterion(
                    teacher_output, student_output, sequences, mask, point_labels, warmup_factor,
                    teacher_only=teacher_only
                )

            # Backward pass with AMP
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            for key in epoch_losses.keys():
                epoch_losses[key] += loss_dict[key]

        for key in epoch_losses.keys():
            epoch_losses[key] /= len(self.train_loader)

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
            'anomaly_type_scores': {}
        }

        if self.test_loader is None:
            return empty_results

        self.model.eval()
        all_recon = []
        all_disc = []
        all_sample_types = []
        all_anomaly_types = []

        with torch.no_grad(), autocast('cuda', enabled=self.use_amp):
            for batch in self.test_loader:
                if len(batch) == 5:
                    sequences, last_patch_labels, point_labels, sample_types, anomaly_types = batch
                elif len(batch) == 4:
                    sequences, last_patch_labels, point_labels, sample_types = batch
                    anomaly_types = torch.zeros_like(last_patch_labels)
                else:
                    sequences, last_patch_labels, point_labels = batch
                    sample_types = torch.zeros_like(last_patch_labels)
                    anomaly_types = torch.zeros_like(last_patch_labels)

                sequences = sequences.to(self.config.device)
                batch_size, seq_length, num_features = sequences.shape

                # Create mask for last n positions (evaluation mode)
                mask = torch.ones(batch_size, seq_length, device=self.config.device)
                mask[:, -self.config.mask_last_n:] = 0

                teacher_output, student_output, _ = self.model(sequences, masking_ratio=0.0, mask=mask)

                # Compute raw reconstruction error and discrepancy
                recon_error = ((teacher_output - sequences) ** 2).mean(dim=2)
                discrepancy = ((teacher_output - student_output) ** 2).mean(dim=2)

                # Per-sample scores on masked positions
                masked_positions = (mask == 0)
                recon_scores = (recon_error * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-4)
                disc_scores = (discrepancy * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-4)

                all_recon.append(recon_scores.cpu().numpy())
                all_disc.append(disc_scores.cpu().numpy())
                all_sample_types.append(sample_types.numpy())
                all_anomaly_types.append(anomaly_types.numpy())

        recon_all = np.concatenate(all_recon)
        disc_all = np.concatenate(all_disc)
        sample_types_all = np.concatenate(all_sample_types)
        anomaly_types_all = np.concatenate(all_anomaly_types)

        # Compute contributions based on scoring mode
        score_mode = getattr(self.config, 'anomaly_score_mode', 'default')
        lambda_disc = getattr(self.config, 'lambda_disc', 0.5)

        if score_mode == 'normalized':
            recon_mean, recon_std = recon_all.mean(), recon_all.std() + 1e-4
            disc_mean, disc_std = disc_all.mean(), disc_all.std() + 1e-4
            recon_z = (recon_all - recon_mean) / recon_std
            disc_z = (disc_all - disc_mean) / disc_std

            # Min-shift for visualization: shift both to start from 0
            min_val = min(recon_z.min(), disc_z.min())
            recon_contrib_all = recon_z - min_val
            disc_contrib_all = disc_z - min_val

            # Contribution ratios based on shifted (non-negative) values
            total = recon_contrib_all + disc_contrib_all + 1e-4
            recon_ratio_all = recon_contrib_all / total
            disc_ratio_all = disc_contrib_all / total
        elif score_mode == 'adaptive':
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
        for atype_idx in range(len(SLIDING_ANOMALY_TYPE_NAMES)):
            atype_name = SLIDING_ANOMALY_TYPE_NAMES[atype_idx]
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

    def train(self) -> Dict:
        teacher_warmup = getattr(self.config, 'teacher_only_warmup_epochs', 1)
        for epoch in range(self.config.num_epochs):
            # First N epochs are warm-up: train teacher only (no discrepancy/student loss)
            teacher_only = (epoch < teacher_warmup)
            epoch_losses = self.train_epoch(epoch, teacher_only=teacher_only)
            self.scheduler.step()

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
            contrib_ratios = self._compute_test_contrib_ratios()
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

        return self.history
