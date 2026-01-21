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
            'epoch': []
        }

    def _compute_warmup_factor(self, epoch: int) -> float:
        if epoch < self.config.warmup_epochs:
            return epoch / self.config.warmup_epochs
        return 1.0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'discrepancy_loss': 0.0,
            'normal_loss': 0.0,
            'anomaly_loss': 0.0,
            'mean_discrepancy': 0.0
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
                teacher_output, student_output, sequences, mask, point_labels, warmup_factor
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
        for epoch in range(self.config.num_epochs):
            epoch_losses = self.train_epoch(epoch)
            self.scheduler.step()

            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(epoch_losses['total_loss'])
            self.history['train_rec_loss'].append(epoch_losses['reconstruction_loss'])
            self.history['train_disc_loss'].append(epoch_losses['discrepancy_loss'])

        return self.history
