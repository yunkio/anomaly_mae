"""
Evaluator for Self-Distilled MAE Anomaly Detection
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)

from .dataset import ANOMALY_TYPE_NAMES


class Evaluator:
    """Evaluator for anomaly detection"""

    def __init__(
        self,
        model,
        config,
        test_loader: DataLoader
    ):
        self.model = model
        self.config = config
        self.test_loader = test_loader
        self.model.eval()

    def compute_anomaly_scores(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute anomaly scores for all samples in test_loader

        Returns:
            scores: (n_samples,) anomaly scores
            labels: (n_samples,) true labels
            sample_types: (n_samples,) sample type indicators
            anomaly_types: (n_samples,) anomaly type indicators
        """
        all_scores = []
        all_labels = []
        all_sample_types = []
        all_anomaly_types = []

        with torch.no_grad():
            for batch in self.test_loader:
                # Support 3-tuple, 4-tuple, and 5-tuple returns
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

                # Create mask for last n positions
                mask = torch.ones(batch_size, seq_length, device=self.config.device)
                mask[:, -self.config.mask_last_n:] = 0

                teacher_output, student_output, _ = self.model(sequences, masking_ratio=0.0, mask=mask)

                # Anomaly score: reconstruction error + discrepancy
                recon_error = ((teacher_output - sequences) ** 2).mean(dim=2)
                discrepancy = ((teacher_output - student_output) ** 2).mean(dim=2)
                error = recon_error + self.config.lambda_disc * discrepancy

                # Compute scores on masked positions
                masked_positions = (mask == 0)
                scores = (error * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-8)

                all_scores.append(scores.cpu().numpy())
                all_labels.append(last_patch_labels.cpu().numpy())
                all_sample_types.append(sample_types.cpu().numpy())
                all_anomaly_types.append(anomaly_types.cpu().numpy())

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        sample_types = np.concatenate(all_sample_types)
        anomaly_types = np.concatenate(all_anomaly_types)

        return scores, labels, sample_types, anomaly_types

    def compute_detailed_losses(self) -> Dict[str, np.ndarray]:
        """Compute detailed losses for all samples in test_loader

        Returns:
            Dictionary containing:
                reconstruction_loss: (n_samples,) reconstruction loss per sample
                discrepancy_loss: (n_samples,) discrepancy loss per sample
                total_loss: (n_samples,) total loss per sample
                labels: (n_samples,) true labels
                sample_types: (n_samples,) sample type indicators
                anomaly_types: (n_samples,) anomaly type indicators
        """
        all_recon_loss = []
        all_disc_loss = []
        all_labels = []
        all_sample_types = []
        all_anomaly_types = []

        with torch.no_grad():
            for batch in self.test_loader:
                # Support 3-tuple, 4-tuple, and 5-tuple returns
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

                # Create mask for last n positions
                mask = torch.ones(batch_size, seq_length, device=self.config.device)
                mask[:, -self.config.mask_last_n:] = 0

                teacher_output, student_output, _ = self.model(sequences, masking_ratio=0.0, mask=mask)

                # Compute reconstruction loss (on masked positions)
                masked_positions = (mask == 0)
                recon_error = ((teacher_output - sequences) ** 2).mean(dim=2)
                recon_loss = (recon_error * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-8)

                # Compute discrepancy loss (on masked positions)
                disc_error = ((teacher_output - student_output) ** 2).mean(dim=2)
                disc_loss = (disc_error * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-8)

                all_recon_loss.append(recon_loss.cpu().numpy())
                all_disc_loss.append(disc_loss.cpu().numpy())
                all_labels.append(last_patch_labels.cpu().numpy())
                all_sample_types.append(sample_types.cpu().numpy())
                all_anomaly_types.append(anomaly_types.cpu().numpy())

        recon_loss = np.concatenate(all_recon_loss)
        disc_loss = np.concatenate(all_disc_loss)
        labels = np.concatenate(all_labels)
        sample_types = np.concatenate(all_sample_types)
        anomaly_types = np.concatenate(all_anomaly_types)

        return {
            'reconstruction_loss': recon_loss,
            'discrepancy_loss': disc_loss,
            'total_loss': recon_loss + self.config.lambda_disc * disc_loss,
            'labels': labels,
            'sample_types': sample_types,
            'anomaly_types': anomaly_types
        }

    def get_performance_by_anomaly_type(self) -> Dict[str, Dict[str, float]]:
        """Compute performance metrics for each anomaly type

        Returns:
            Dictionary with anomaly type names as keys, containing metrics for each type
        """
        scores, labels, sample_types, anomaly_types = self.compute_anomaly_scores()

        # Get optimal threshold from overall performance
        if len(np.unique(labels)) > 1:
            fpr, tpr, thresholds = roc_curve(labels, scores)
            optimal_idx = np.argmax(tpr - fpr)
            threshold = thresholds[optimal_idx]
        else:
            threshold = np.median(scores)

        results = {}

        for atype_idx, atype_name in enumerate(ANOMALY_TYPE_NAMES):
            type_mask = (anomaly_types == atype_idx)
            if type_mask.sum() == 0:
                continue

            type_scores = scores[type_mask]
            type_labels = labels[type_mask]
            type_predictions = (type_scores > threshold).astype(int)

            type_results = {
                'count': int(type_mask.sum()),
                'mean_score': float(type_scores.mean()),
                'std_score': float(type_scores.std()),
            }

            # Compute metrics if both classes exist
            if len(np.unique(type_labels)) > 1:
                type_results['roc_auc'] = float(roc_auc_score(type_labels, type_scores))
                type_results['precision'] = float(precision_score(type_labels, type_predictions, zero_division=0))
                type_results['recall'] = float(recall_score(type_labels, type_predictions, zero_division=0))
                type_results['f1_score'] = float(f1_score(type_labels, type_predictions, zero_division=0))
            else:
                # For anomaly types (all labels are 1), compute detection rate
                if type_labels.sum() == len(type_labels):  # All anomalies
                    type_results['detection_rate'] = float(type_predictions.mean())
                else:  # All normal
                    type_results['false_positive_rate'] = float(type_predictions.mean())

            results[atype_name] = type_results

        return results

    def evaluate(self) -> Dict[str, float]:
        """Evaluate and return metrics including disturbing normal performance"""
        scores, labels, sample_types, anomaly_types = self.compute_anomaly_scores()

        results = {}

        # Overall performance
        if len(np.unique(labels)) > 1:
            roc_auc = roc_auc_score(labels, scores)
            fpr, tpr, thresholds = roc_curve(labels, scores)
            optimal_idx = np.argmax(tpr - fpr)
            threshold = thresholds[optimal_idx]

            predictions = (scores > threshold).astype(int)
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            f1 = f1_score(labels, predictions, zero_division=0)

            results = {
                'roc_auc': roc_auc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'optimal_threshold': threshold
            }
        else:
            results = {
                'roc_auc': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'optimal_threshold': 0.0
            }

        # Disturbing normal performance
        # sample_type: 0=pure_normal, 1=disturbing_normal, 2=anomaly
        disturbing_mask = (sample_types == 0) | (sample_types == 1)
        if disturbing_mask.sum() > 0:
            disturbing_scores = scores[disturbing_mask]
            disturbing_labels = sample_types[disturbing_mask]

            if len(np.unique(disturbing_labels)) > 1:
                disturbing_roc_auc = roc_auc_score(disturbing_labels, disturbing_scores)
                d_fpr, d_tpr, d_thresholds = roc_curve(disturbing_labels, disturbing_scores)
                d_optimal_idx = np.argmax(d_tpr - d_fpr)
                d_threshold = d_thresholds[d_optimal_idx]

                d_predictions = (disturbing_scores > d_threshold).astype(int)
                disturbing_precision = precision_score(disturbing_labels, d_predictions, zero_division=0)
                disturbing_recall = recall_score(disturbing_labels, d_predictions, zero_division=0)
                disturbing_f1 = f1_score(disturbing_labels, d_predictions, zero_division=0)

                results['disturbing_roc_auc'] = disturbing_roc_auc
                results['disturbing_precision'] = disturbing_precision
                results['disturbing_recall'] = disturbing_recall
                results['disturbing_f1'] = disturbing_f1
                results['n_pure_normal'] = int((sample_types == 0).sum())
                results['n_disturbing_normal'] = int((sample_types == 1).sum())
                results['n_anomaly'] = int((sample_types == 2).sum())

        return results
