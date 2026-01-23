"""
Trainer for Self-Distilled MAE Anomaly Detection
"""

import torch
from torch.utils.data import DataLoader
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

        self.history = {
            'train_loss': [],
            'train_rec_loss': [],
            'train_disc_loss': [],
            'train_student_recon_loss': [],
            'train_normal_loss': [],
            'train_anomaly_loss': [],
            # Detailed metrics by sample type
            'train_teacher_recon_normal': [],
            'train_teacher_recon_anomaly': [],
            'train_student_recon_normal': [],
            'train_student_recon_anomaly': [],
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
            'student_recon_loss': 0.0,
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

            teacher_output, student_output, mask = self.model(sequences, point_labels=point_labels)

            loss, loss_dict = self.criterion(
                teacher_output, student_output, sequences, mask, point_labels, warmup_factor,
                teacher_only=teacher_only
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for key in epoch_losses.keys():
                epoch_losses[key] += loss_dict[key]

        for key in epoch_losses.keys():
            epoch_losses[key] /= len(self.train_loader)

        return epoch_losses

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
            self.history['train_student_recon_loss'].append(epoch_losses['student_recon_loss'])
            self.history['train_normal_loss'].append(epoch_losses['normal_loss'])
            self.history['train_anomaly_loss'].append(epoch_losses['anomaly_loss'])
            # Detailed metrics by sample type
            self.history['train_teacher_recon_normal'].append(epoch_losses['teacher_recon_normal'])
            self.history['train_teacher_recon_anomaly'].append(epoch_losses['teacher_recon_anomaly'])
            self.history['train_student_recon_normal'].append(epoch_losses['student_recon_normal'])
            self.history['train_student_recon_anomaly'].append(epoch_losses['student_recon_anomaly'])

        return self.history
