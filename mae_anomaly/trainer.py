"""
Trainer for Self-Distilled MAE Anomaly Detection
"""

import time
import math
import random
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from typing import Dict
from tqdm import tqdm

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from .loss import (
    SelfDistillationLoss,
    compute_discriminator_loss,
    compute_student_adversarial_loss,
    compute_adaptive_lambda,
)


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

        # --- Config validation ---
        # 1. teacher_only_warmup_epochs auto
        if config.teacher_only_warmup_epochs < 0:
            config.teacher_only_warmup_epochs = config.num_epochs // 2
            if verbose:
                print(f"  [Config] teacher_only_warmup_epochs = "
                      f"{config.teacher_only_warmup_epochs} (auto: num_epochs // 2)")

        # 2. freeze → warmup 강제 override
        if getattr(config, 'freeze_teacher_after_warmup', False):
            forced = config.num_epochs // 2
            if config.teacher_only_warmup_epochs != forced:
                print(f"  [Config] freeze_teacher_after_warmup=True: "
                      f"teacher_only_warmup_epochs {config.teacher_only_warmup_epochs} → {forced}")
                config.teacher_only_warmup_epochs = forced

        # 3. use_grl + use_discriminator 동시 금지
        if getattr(config, 'use_grl', False) and getattr(config, 'use_discriminator', False):
            raise ValueError(
                "use_grl과 use_discriminator는 동시에 True일 수 없습니다. "
                "둘 중 하나만 활성화하세요.")

        # 4. use_grl → patch_level_loss 필수
        if getattr(config, 'use_grl', False) and not config.patch_level_loss:
            raise ValueError(
                "use_grl=True는 patch_level_loss=True를 필요로 합니다. "
                "GRL classifier는 patch-level에서 동작합니다.")

        # 5. shared_mask_token + freeze 금지
        if getattr(config, 'shared_mask_token', False) and getattr(config, 'freeze_teacher_after_warmup', False):
            raise ValueError(
                "shared_mask_token=True와 freeze_teacher_after_warmup=True는 "
                "동시 사용 불가합니다.")

        # 6. freeze_encoder_only + freeze_teacher_after_warmup 동시 금지
        if getattr(config, 'freeze_encoder_only', False) and getattr(config, 'freeze_teacher_after_warmup', False):
            raise ValueError(
                "freeze_encoder_only=True와 freeze_teacher_after_warmup=True는 "
                "동시 사용 불가합니다. 하나만 선택하세요.")

        # 7. use_scad + use_grl 동시 금지 (mutually exclusive)
        if getattr(config, 'use_scad', False) and getattr(config, 'use_grl', False):
            raise ValueError(
                "use_scad과 use_grl은 동시에 True일 수 없습니다. "
                "둘 중 하나만 활성화하세요.")

        # 8. use_scad + use_discriminator 동시 금지 (mutually exclusive)
        if getattr(config, 'use_scad', False) and getattr(config, 'use_discriminator', False):
            raise ValueError(
                "use_scad과 use_discriminator는 동시에 True일 수 없습니다.")

        # 9. use_scad → patch_level_loss 필수
        if getattr(config, 'use_scad', False) and not config.patch_level_loss:
            raise ValueError(
                "use_scad=True는 patch_level_loss=True를 필요로 합니다. "
                "SCAD는 patch-level supervision으로 동작합니다.")

        # 10. seq_length / patch_size / num_patches 일관성 (defense-in-depth;
        #    make_config()에서도 검증되나 Trainer가 직접 호출되는 경로 보호)
        if config.seq_length % config.patch_size != 0:
            raise ValueError(
                f"seq_length ({config.seq_length}) must be divisible by "
                f"patch_size ({config.patch_size})."
            )
        if config.seq_length != config.patch_size * config.num_patches:
            raise ValueError(
                f"seq_length ({config.seq_length}) != "
                f"patch_size ({config.patch_size}) * num_patches "
                f"({config.num_patches})."
            )

        self.criterion = SelfDistillationLoss(config)

        # Bias/Norm weight decay separation (matching original MAE)
        decay_params = []
        no_decay_params = []
        # Exclude WDGRL critic from main optimizer (it has its own optimizer)
        _exclude_prefix = 'wasserstein_critic.' if (
            getattr(config, 'use_grl', False) and getattr(config, 'grl_mode', 'classifier') == 'wdgrl'
        ) else None
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if _exclude_prefix and name.startswith(_exclude_prefix):
                continue  # WDGRL critic managed by separate optimizer
            if param.ndim <= 1:  # bias, LayerNorm weight/bias, mask tokens
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # Separate GRL classifier params if grl_cls_lr_ratio != 1.0
        _grl_cls_lr_ratio = getattr(config, 'grl_cls_lr_ratio', 1.0)
        _use_grl_cls_sep = (
            getattr(config, 'use_grl', False)
            and getattr(config, 'grl_mode', 'classifier') == 'classifier'
            and _grl_cls_lr_ratio != 1.0
            and hasattr(model, 'anomaly_classifier')
        )
        param_groups = [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        if _use_grl_cls_sep:
            _cls_ids = set(id(p) for p in model.anomaly_classifier.parameters())
            _cls_decay = [p for p in decay_params if id(p) in _cls_ids]
            _cls_no_decay = [p for p in no_decay_params if id(p) in _cls_ids]
            param_groups[0] = {'params': [p for p in decay_params if id(p) not in _cls_ids],
                               'weight_decay': config.weight_decay}
            param_groups[1] = {'params': [p for p in no_decay_params if id(p) not in _cls_ids],
                               'weight_decay': 0.0}
            _cls_lr = config.learning_rate * _grl_cls_lr_ratio
            if _cls_decay:
                param_groups.append({'params': _cls_decay, 'weight_decay': config.weight_decay, 'lr': _cls_lr})
            if _cls_no_decay:
                param_groups.append({'params': _cls_no_decay, 'weight_decay': 0.0, 'lr': _cls_lr})

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.99),  # Compromise: 0.95 (original MAE, 75% masking) ↔ 0.999 (PyTorch default)
        )

        # LR warmup + cosine annealing (matching original MAE)
        lr_warmup_epochs = config.warmup_epochs  # Reuse anomaly loss warmup period for LR warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-4,  # Start from near-zero LR
            end_factor=1.0,
            total_iters=lr_warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(config.num_epochs - lr_warmup_epochs, 1),
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[lr_warmup_epochs],
        )

        self.model = self.model.to(config.device)

        # Epoch-level adaptive lambda (prev epoch average → current epoch fixed)
        self._prev_epoch_adv_lambda = 1.0   # Discriminator
        self._prev_epoch_fm_lambda = 1.0    # Feature Matching
        self._prev_epoch_grl_lambda = 1.0   # GRL

        # WDGRL critic (separate optimizer, alternating training)
        self.wdgrl_critic = None
        self.wdgrl_critic_optimizer = None
        if getattr(config, 'use_grl', False) and getattr(config, 'grl_mode', 'classifier') == 'wdgrl':
            self.wdgrl_critic = self.model.wasserstein_critic
            self.wdgrl_critic_optimizer = torch.optim.Adam(
                self.wdgrl_critic.parameters(),
                lr=getattr(config, 'wdgrl_critic_lr', 1e-4),
                betas=(0.5, 0.999),
            )

        # Mixed Precision Training (AMP)
        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Discriminator (Adversarial Realism)
        self.use_discriminator = config.use_discriminator
        self.discriminator = None
        self.d_optimizer = None
        if self.use_discriminator:
            from .model import PatchDiscriminator
            self.discriminator = PatchDiscriminator(
                num_features=config.num_features,
                patch_size=config.patch_size,
                channels=config.disc_channels,
            ).to(config.device)
            self.d_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=config.learning_rate * config.disc_lr_ratio,
                betas=(0.0, 0.99),  # TTUR: β1=0 for discriminator
                weight_decay=0.0,   # Spectral norm provides regularization
            )
            # D scheduler: CosineAnnealingLR from disc_warmup_epochs to end
            d_active_epochs = max(config.num_epochs - config.disc_warmup_epochs, 1)
            self.d_scheduler = CosineAnnealingLR(
                self.d_optimizer, T_max=d_active_epochs,
            )

        self.history = {
            'train_loss': [],
            'train_rec_loss': [],
            'train_disc_loss': [],
            'train_normal_loss': [],
            'train_anomaly_loss': [],
            'train_mean_discrepancy': [],
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
            'epoch': [],
            # Discriminator metrics (populated only when use_discriminator=True)
            'train_d_loss': [],
            'train_d_real_acc': [],
            'train_d_fake_acc': [],
            'train_adv_loss': [],
            'train_adaptive_lambda': [],
            # Feature-level per-epoch stats (training): List[List[float]] — (num_features,) per epoch
            'train_feature_disc_mean': [],
            'train_feature_disc_max': [],
            'train_feature_recon_mean': [],
            'train_feature_recon_max': [],
            # Feature matching loss (populated always, 0.0 when disabled)
            'train_fm_loss': [],
            # FM adaptive lambda (populated only when fm_adaptive_lambda=True)
            'train_fm_adaptive_lambda': [],
            # GRL metrics (populated only when use_grl=True)
            'train_grl_cls_loss': [],
            'train_grl_balanced_acc': [],
            'train_grl_anomaly_acc': [],
            'train_grl_normal_acc': [],
            'train_grl_lambda': [],
            'train_grl_effective_weight': [],  # lambda * grl_loss_weight = actual multiplier
            # SCAD metrics (populated only when use_scad=True) — mirror GRL pattern
            'train_scad_loss': [],
            'train_scad_n_anom': [],
            'train_scad_n_norm': [],
            'train_scad_z_separation': [],
            'train_scad_z_anom_var': [],
            'train_scad_z_norm_var': [],
            'train_scad_lambda': [],            # adaptive λ (raw value)
            'train_scad_adaptive_lambda': [],   # adaptive λ (clamped to [0, 10])
            'train_scad_ramp': [],              # sigmoid ramp ∈ [0, 1]
            'train_scad_effective_weight': [],  # lambda * ramp * scad_loss_weight
            'train_scad_grad_norm': [],         # ||∇_w L_SCAD||
            'train_scad_main_grad_norm': [],    # ||∇_w L_main||
        }

    def _compute_warmup_factor(self, epoch: int) -> float:
        """Anomaly loss warmup: teacher_only 종료 후 자동 ramp.

        warmup_length = max(teacher_only_warmup_epochs // 5, 2)
        """
        student_start = self.config.teacher_only_warmup_epochs
        warmup_length = max(student_start // 5, 2)
        student_epoch = epoch - student_start
        if student_epoch < 0:
            return 0.0  # teacher_only 기간
        if student_epoch < warmup_length:
            return (student_epoch + 1) / warmup_length
        return 1.0

    def _extract_patches(self, original, student_output, mask, point_labels):
        """Extract patch-level data for discriminator training.

        D trains on ALL masked patches (normal + anomaly).
        anomaly_patch_mask selects anomaly patches for student adversarial loss.

        Args:
            original: (B, seq_length, num_features)
            student_output: (B, seq_length, num_features)
            mask: (B, seq_length) 1=visible, 0=masked
            point_labels: (B, seq_length) 1=anomaly, 0=normal

        Returns:
            real_patches: (N, num_features, patch_size) Conv1d format
            fake_patches: (N, num_features, patch_size) with gradient through student
            anomaly_patch_mask: (N,) bool — True if patch has anomaly in masked region
        """
        B = original.size(0)
        ps = self.config.patch_size
        np_ = self.config.num_patches
        nf = original.size(-1)

        # Reshape to (B, num_patches, patch_size, num_features)
        orig_patches = original.reshape(B, np_, ps, nf)
        stud_patches = student_output.reshape(B, np_, ps, nf)
        mask_patches = mask.reshape(B, np_, ps)
        label_patches = point_labels.reshape(B, np_, ps)

        # Identify patches with any masked position
        patch_has_masked = ((1 - mask_patches).sum(dim=2) > 0)  # (B, np_)

        # Flatten and select masked patches
        orig_flat = orig_patches.reshape(-1, ps, nf)
        stud_flat = stud_patches.reshape(-1, ps, nf)
        selector = patch_has_masked.reshape(-1)

        # (N, num_features, patch_size) — Conv1d format
        real_patches = orig_flat[selector].transpose(1, 2)
        fake_patches = stud_flat[selector].transpose(1, 2)  # preserves grad

        # Anomaly status: only count anomaly in masked positions
        masked_labels = label_patches * (1 - mask_patches)
        patch_anomaly = (masked_labels.sum(dim=2) > 0)
        anomaly_patch_mask = patch_anomaly.reshape(-1)[selector]

        return real_patches, fake_patches, anomaly_patch_mask

    def train_epoch(self, epoch: int, teacher_only: bool = False,
                    profile_batches: int = 0) -> Dict[str, float]:
        self.model.train()

        # Frozen 모듈 eval 복원 (BN stats 오염 방지 + Dropout OFF 유지)
        if hasattr(self, '_frozen_eval_modules'):
            for name in self._frozen_eval_modules:
                module = getattr(self.model, name, None)
                if module is not None:
                    module.eval()
        if hasattr(self, '_frozen_encoder_modules'):
            for name in self._frozen_encoder_modules:
                module = getattr(self.model, name, None)
                if module is not None:
                    module.eval()

        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'discrepancy_loss': 0.0,
            'normal_loss': 0.0,
            'anomaly_loss': 0.0,
            'fm_loss': 0.0,
            'mean_discrepancy': 0.0,
            # Detailed metrics by sample type
            'teacher_recon_normal': 0.0,
            'teacher_recon_anomaly': 0.0,
            'student_recon_normal': 0.0,
            'student_recon_anomaly': 0.0,
        }
        if self.use_discriminator:
            epoch_losses.update({
                'd_loss': 0.0, 'd_real_acc': 0.0, 'd_fake_acc': 0.0,
                'adv_loss': 0.0, 'adaptive_lambda': 0.0,
            })
        if getattr(self.config, 'fm_adaptive_lambda', False):
            epoch_losses['fm_adaptive_lambda'] = 0.0
        if getattr(self.config, 'use_grl', False):
            epoch_losses.update({
                'grl_cls_loss': 0.0, 'grl_balanced_acc': 0.0, 'grl_lambda': 0.0,
                'grl_anomaly_acc': 0.0, 'grl_normal_acc': 0.0, 'grl_effective_weight': 0.0,
            })
        if getattr(self.config, 'use_scad', False):
            epoch_losses.update({
                'scad_loss': 0.0,
                'scad_n_anom': 0,
                'scad_n_norm': 0,
                'scad_z_separation': 0.0,
                'scad_z_anom_var': 0.0,
                'scad_z_norm_var': 0.0,
                'scad_lambda': 0.0,
                'scad_adaptive_lambda': 0.0,
                'scad_ramp': 0.0,
                'scad_effective_weight': 0.0,
                'scad_grad_norm': 0.0,
                'scad_main_grad_norm': 0.0,
            })

        warmup_factor = self._compute_warmup_factor(epoch)

        # Batch-level profiling: first N batches with cuda.synchronize() per component
        batch_profiles = [] if profile_batches > 0 else None

        # Feature-level stats accumulator (separate from scalar epoch_losses)
        _feature_accum = {
            'recon_mean': None, 'recon_max': None,
            'disc_mean': None, 'disc_max': None,
        }
        _feature_batch_count = 0

        # Epoch-level timing (sync only at epoch boundaries → ~1% overhead)
        torch.cuda.synchronize()
        t_epoch_start = time.time()
        t_forward_acc = 0.0
        t_backward_acc = 0.0

        iterator = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}',
                        disable=not self.verbose, leave=False)

        for batch_idx, batch in enumerate(iterator):
            do_profile = batch_profiles is not None and 0 < batch_idx <= profile_batches

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
            if do_profile:
                self.model._profiling = True
            # Masking ratio: anneal > random range > default
            _mr = getattr(self, '_annealed_masking_ratio', None)
            if _mr is None:
                _mr_min = getattr(self.config, 'masking_ratio_min', -1.0)
                _mr_max = getattr(self.config, 'masking_ratio_max', -1.0)
                _mr = random.uniform(_mr_min, _mr_max) if (_mr_min >= 0 and _mr_max >= 0) else None
            with autocast('cuda', enabled=self.use_amp):
                teacher_output, student_output, mask = self.model(sequences, masking_ratio=_mr, point_labels=point_labels)

                if do_profile:
                    self.model._profiling = False
                    torch.cuda.synchronize()
                    t_bp_model = time.time()
                    layer_timing = getattr(self.model, '_forward_timing', {})

                # GRL cls_logits from model attribute (None if use_grl=False or teacher_only)
                _grl_logits = getattr(self.model, '_grl_cls_logits', None)
                # Hidden states for FM loss
                _t_hidden = getattr(self.model, '_teacher_hidden', None)
                _s_hidden = getattr(self.model, '_student_hidden', None)
                # SCAD projection embedding (None if use_scad=False or teacher_only)
                _scad_z = getattr(self.model, '_scad_z', None)

                loss, loss_dict, loss_tensors = self.criterion(
                    teacher_output, student_output, sequences, mask, point_labels, warmup_factor,
                    teacher_only=teacher_only, grl_cls_logits=_grl_logits,
                    teacher_hidden=_t_hidden, student_hidden=_s_hidden,
                    scad_z=_scad_z
                )

            if do_profile:
                torch.cuda.synchronize()
                t_bp_loss = time.time()

            # --- Discriminator step (D → Student, TTUR order) ---
            if self.use_discriminator and epoch >= self.config.disc_warmup_epochs and not teacher_only:
                real_patches, fake_patches, anomaly_patch_mask = self._extract_patches(
                    sequences, student_output, mask, point_labels)

                # D training: real vs fake on ALL masked patches
                self.d_optimizer.zero_grad()
                with autocast('cuda', enabled=self.use_amp):
                    d_loss, d_real_acc, d_fake_acc = compute_discriminator_loss(
                        self.discriminator, real_patches, fake_patches)
                if self.scaler is not None:
                    self.scaler.scale(d_loss).backward()
                    self.scaler.unscale_(self.d_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                    self.scaler.step(self.d_optimizer)
                else:
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                    self.d_optimizer.step()

                # Student adversarial loss (anomaly patches only — fool D)
                adv_loss_val = 0.0
                lambda_adv_val = 0.0
                if anomaly_patch_mask.any():
                    anomaly_fake = fake_patches[anomaly_patch_mask]
                    with autocast('cuda', enabled=self.use_amp):
                        adv_loss = compute_student_adversarial_loss(
                            self.discriminator, anomaly_fake)

                        if self.config.adaptive_lambda:
                            last_weight = self.model.student_output_projection.weight
                            lambda_adv = compute_adaptive_lambda(
                                last_weight,
                                loss_tensors['normal_loss'],
                                loss_tensors['anomaly_disc_forward'],
                                adv_loss,
                            )
                        else:
                            lambda_adv = torch.tensor(1.0, device=self.config.device)

                        # Use prev-epoch lambda for stability (batch value logged for monitoring)
                        loss = loss + self.config.adv_loss_weight * self._prev_epoch_adv_lambda * adv_loss

                    adv_loss_val = adv_loss.item()
                    lambda_adv_val = lambda_adv.item() if isinstance(lambda_adv, torch.Tensor) else lambda_adv

                loss_dict['d_loss'] = d_loss.item()
                loss_dict['d_real_acc'] = d_real_acc
                loss_dict['d_fake_acc'] = d_fake_acc
                loss_dict['adv_loss'] = adv_loss_val
                loss_dict['adaptive_lambda'] = lambda_adv_val
            elif self.use_discriminator:
                # Before D warmup or during teacher_only: zero metrics
                loss_dict['d_loss'] = 0.0
                loss_dict['d_real_acc'] = 0.0
                loss_dict['d_fake_acc'] = 0.0
                loss_dict['adv_loss'] = 0.0
                loss_dict['adaptive_lambda'] = 0.0

            # --- FM adaptive lambda step ---
            if getattr(self.config, 'fm_adaptive_lambda', False) and not teacher_only and 'fm_loss' in loss_tensors:
                _fm_loss_tensor = loss_tensors['fm_loss']
                if _fm_loss_tensor.item() > 1e-8:
                    _last_w_fm = list(self.model.student_decoder.parameters())[-1]
                    with autocast('cuda', enabled=False):
                        _main_g_fm = torch.autograd.grad(loss.float(), _last_w_fm, retain_graph=True, allow_unused=True)[0]
                        _fm_g = torch.autograd.grad(_fm_loss_tensor.float(), _last_w_fm, retain_graph=True, allow_unused=True)[0]
                    if _main_g_fm is not None and _fm_g is not None:
                        _fm_lambda = (_main_g_fm.norm() / (_fm_g.norm() + 1e-4)).clamp(0.0, 10.0).detach()
                    else:
                        _fm_lambda = torch.tensor(1.0, device=loss.device)
                    _fm_w = getattr(self.config, 'fm_loss_weight', 1.0)
                    # Use prev-epoch lambda for stability (batch value logged for monitoring)
                    loss = loss + self._prev_epoch_fm_lambda * _fm_w * _fm_loss_tensor
                    loss_dict['fm_adaptive_lambda'] = _fm_lambda.item()
                else:
                    loss_dict['fm_adaptive_lambda'] = 0.0
            elif getattr(self.config, 'fm_adaptive_lambda', False):
                # teacher_only or FM inactive: zero metric (consistent with D/GRL fallback)
                loss_dict['fm_adaptive_lambda'] = 0.0

            # --- GRL / WDGRL step ---
            _grl_mode = getattr(self.config, 'grl_mode', 'classifier')
            if getattr(self.config, 'use_grl', False) and not teacher_only and _grl_mode == 'wdgrl':
                # === WDGRL: Wasserstein critic with gradient penalty ===
                from .loss import compute_wdgrl_gradient_penalty
                _student_hidden = getattr(self.model, '_student_hidden', None)
                if _student_hidden is not None:
                    # Reshape: (num_patches, batch, d_model) → (batch*num_patches, d_model)
                    _sh = _student_hidden.detach().transpose(0, 1).reshape(-1, _student_hidden.size(-1))
                    # Get patch labels for masked patches
                    _patch_mask_flat = loss_tensors.get('patch_has_masked', None)
                    _patch_anom_flat = loss_tensors.get('patch_has_anomaly', None)
                    if _patch_mask_flat is not None and _patch_anom_flat is not None:
                        _masked = _patch_mask_flat.bool().reshape(-1)
                        _anom = _patch_anom_flat.reshape(-1)
                        _sh_masked = _sh[_masked]
                        _targets_masked = _anom[_masked]
                        _pos_m = _targets_masked > 0.5
                        _neg_m = ~_pos_m
                        _f_anom = _sh_masked[_pos_m]
                        _f_norm = _sh_masked[_neg_m]

                        if _f_anom.size(0) > 0 and _f_norm.size(0) > 0:
                            # Phase 1: Update critic k times (student frozen, features detached)
                            _k = getattr(self.config, 'wdgrl_k_critic', 5)
                            _gp_w = getattr(self.config, 'wdgrl_gp_weight', 10.0)
                            for _ in range(_k):
                                _c_norm = self.wdgrl_critic(_f_norm)
                                _c_anom = self.wdgrl_critic(_f_anom)
                                _wd = _c_norm.mean() - _c_anom.mean()
                                _gp = compute_wdgrl_gradient_penalty(
                                    self.wdgrl_critic, _f_norm, _f_anom)
                                _critic_loss = -_wd + _gp_w * _gp  # Maximize WD
                                self.wdgrl_critic_optimizer.zero_grad()
                                _critic_loss.backward()
                                self.wdgrl_critic_optimizer.step()

                            # Phase 2: Compute WD for main loss (critic FROZEN, student has grad)
                            for _p in self.wdgrl_critic.parameters():
                                _p.requires_grad_(False)

                            _sh_grad = _student_hidden.transpose(0, 1).reshape(-1, _student_hidden.size(-1))
                            _sh_masked_grad = _sh_grad[_masked]
                            _f_anom_grad = _sh_masked_grad[_pos_m]
                            _f_norm_grad = _sh_masked_grad[_neg_m]
                            with torch.no_grad():
                                _c_norm_score = self.wdgrl_critic(_f_norm_grad).mean().item()
                                _c_anom_score = self.wdgrl_critic(_f_anom_grad).mean().item()
                            # Student minimizes WD (no GRL needed — direct minimization)
                            _wd_for_student = self.wdgrl_critic(_f_norm_grad).mean() - self.wdgrl_critic(_f_anom_grad).mean()
                            _grl_w = getattr(self.config, 'grl_loss_weight', 1.0)
                            loss = loss + _grl_w * _wd_for_student

                            # Unfreeze critic for next batch
                            for _p in self.wdgrl_critic.parameters():
                                _p.requires_grad_(True)

                            loss_dict['grl_cls_loss'] = _wd.item()
                            loss_dict['grl_lambda'] = _gp.item()
                            loss_dict['grl_effective_weight'] = _grl_w
                            # Use critic score difference as proxy for balanced_acc
                            loss_dict['grl_balanced_acc'] = abs(_c_norm_score - _c_anom_score)
                            loss_dict['grl_anomaly_acc'] = _c_anom_score
                            loss_dict['grl_normal_acc'] = _c_norm_score
                        else:
                            loss_dict.setdefault('grl_cls_loss', 0.0)
                            loss_dict.setdefault('grl_balanced_acc', 0.0)
                            loss_dict.setdefault('grl_anomaly_acc', 0.0)
                            loss_dict.setdefault('grl_normal_acc', 0.0)
                            loss_dict.setdefault('grl_lambda', 0.0)
                            loss_dict.setdefault('grl_effective_weight', 0.0)
                    else:
                        loss_dict.setdefault('grl_cls_loss', 0.0)
                        loss_dict.setdefault('grl_balanced_acc', 0.0)
                        loss_dict.setdefault('grl_anomaly_acc', 0.0)
                        loss_dict.setdefault('grl_normal_acc', 0.0)
                        loss_dict.setdefault('grl_lambda', 0.0)
                        loss_dict.setdefault('grl_effective_weight', 0.0)
                else:
                    loss_dict.setdefault('grl_cls_loss', 0.0)
                    loss_dict.setdefault('grl_balanced_acc', 0.0)
                    loss_dict.setdefault('grl_anomaly_acc', 0.0)
                    loss_dict.setdefault('grl_normal_acc', 0.0)
                    loss_dict.setdefault('grl_lambda', 0.0)
                    loss_dict.setdefault('grl_effective_weight', 0.0)

            elif getattr(self.config, 'use_grl', False) and not teacher_only and 'grl_cls_loss' in loss_tensors:
                # === Classifier mode (DANN-style GRL, default) ===
                _grl_cls_loss = loss_tensors['grl_cls_loss']
                _grl_w = getattr(self.config, 'grl_loss_weight', 1.0)

                if getattr(self.config, 'grl_adaptive_lambda', True):
                    # Adaptive scaling: GRL gradient ≈ main gradient (auto 1:1 balancing)
                    _last_w = list(self.model.student_decoder.parameters())[-1]
                    with autocast('cuda', enabled=False):
                        _main_g = torch.autograd.grad(loss.float(), _last_w, retain_graph=True, allow_unused=True)[0]
                        _grl_g = torch.autograd.grad(_grl_cls_loss.float(), _last_w, retain_graph=True, allow_unused=True)[0]
                    if _main_g is None or _grl_g is None:
                        _grl_lambda_adp = torch.tensor(1.0, device=loss.device)
                    else:
                        _grl_lambda_adp = (_main_g.norm() / (_grl_g.norm() + 1e-4)).clamp(0.0, 10.0).detach()

                    _grl_effective = self._prev_epoch_grl_lambda * _grl_w
                    loss = loss + _grl_effective * _grl_cls_loss
                    loss_dict['grl_lambda'] = _grl_lambda_adp.item()
                    loss_dict['grl_effective_weight'] = _grl_effective
                else:
                    # Fixed weight: no adaptive lambda, direct grl_loss_weight
                    _grl_lambda_adp = torch.tensor(1.0, device=loss.device)
                    loss = loss + _grl_w * _grl_cls_loss
                    loss_dict['grl_lambda'] = 1.0
                    loss_dict['grl_effective_weight'] = _grl_w
            elif getattr(self.config, 'use_grl', False):
                # teacher_only or GRL inactive: zero metrics
                loss_dict.setdefault('grl_cls_loss', 0.0)
                loss_dict.setdefault('grl_balanced_acc', 0.0)
                loss_dict.setdefault('grl_anomaly_acc', 0.0)
                loss_dict.setdefault('grl_normal_acc', 0.0)
                loss_dict.setdefault('grl_lambda', 0.0)
                loss_dict.setdefault('grl_effective_weight', 0.0)

            # --- SCAD step (replaces GRL when use_scad=True) ---
            if getattr(self.config, 'use_scad', False) and not teacher_only and 'scad_loss' in loss_tensors:
                _scad_loss_tensor = loss_tensors['scad_loss']
                _scad_loss_val = float(_scad_loss_tensor.item() if torch.is_tensor(_scad_loss_tensor) else _scad_loss_tensor)
                _scad_w = getattr(self.config, 'scad_loss_weight', 0.5)

                # Sigmoid / linear / none ramp-up
                _ramp_mode = getattr(self.config, 'scad_ramp_up', 'sigmoid')
                _twe = self.config.teacher_only_warmup_epochs
                if epoch < _twe:
                    _ramp = 0.0
                else:
                    _denom = max(1, self.config.num_epochs - _twe)
                    _p = (epoch - _twe) / _denom
                    if _ramp_mode == 'sigmoid':
                        _ramp = 2.0 / (1.0 + math.exp(-10.0 * _p)) - 1.0
                    elif _ramp_mode == 'linear':
                        _ramp = min(1.0, _p)
                    else:  # 'none' — full strength once warmup ends
                        _ramp = 1.0

                # Adaptive λ (VQGAN-style gradient balancing)
                _scad_grad_norm = 0.0
                _main_grad_norm = 0.0
                if getattr(self.config, 'scad_adaptive_lambda', True) and _scad_loss_val > 1e-12:
                    _last_w_scad = list(self.model.student_decoder.parameters())[-1]
                    with autocast('cuda', enabled=False):
                        _main_g_scad = torch.autograd.grad(
                            loss.float(), _last_w_scad,
                            retain_graph=True, allow_unused=True,
                        )[0]
                        _scad_g = torch.autograd.grad(
                            _scad_loss_tensor.float(), _last_w_scad,
                            retain_graph=True, allow_unused=True,
                        )[0]
                    if _main_g_scad is not None and _scad_g is not None:
                        _main_grad_norm = float(_main_g_scad.norm().item())
                        _scad_grad_norm = float(_scad_g.norm().item())
                        _scad_lambda = max(0.0, min(10.0, _main_grad_norm / (_scad_grad_norm + 1e-4)))
                    else:
                        _scad_lambda = 1.0
                else:
                    _scad_lambda = 1.0

                _scad_effective = _scad_lambda * _ramp * _scad_w
                loss = loss + _scad_effective * _scad_loss_tensor

                loss_dict['scad_lambda'] = _scad_lambda
                loss_dict['scad_adaptive_lambda'] = _scad_lambda
                loss_dict['scad_ramp'] = _ramp
                loss_dict['scad_effective_weight'] = _scad_effective
                loss_dict['scad_grad_norm'] = _scad_grad_norm
                loss_dict['scad_main_grad_norm'] = _main_grad_norm
            elif getattr(self.config, 'use_scad', False):
                # teacher_only or SCAD inactive: zero metrics for downstream logging
                loss_dict.setdefault('scad_loss', 0.0)
                loss_dict.setdefault('scad_n_anom', 0)
                loss_dict.setdefault('scad_n_norm', 0)
                loss_dict.setdefault('scad_z_separation', 0.0)
                loss_dict.setdefault('scad_z_anom_var', 0.0)
                loss_dict.setdefault('scad_z_norm_var', 0.0)
                loss_dict.setdefault('scad_lambda', 0.0)
                loss_dict.setdefault('scad_adaptive_lambda', 0.0)
                loss_dict.setdefault('scad_ramp', 0.0)
                loss_dict.setdefault('scad_effective_weight', 0.0)
                loss_dict.setdefault('scad_grad_norm', 0.0)
                loss_dict.setdefault('scad_main_grad_norm', 0.0)

            # Backward pass with AMP (loss now includes λ*adv_loss or λ*grl_cls_loss if applicable)
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
                bp_entry = {
                    'batch': batch_idx,
                    'data_to_gpu_ms': (t_bp_data - t_bp_start) * 1000,
                    'model_forward_ms': (t_bp_model - t_bp_data) * 1000,
                    'loss_compute_ms': (t_bp_loss - t_bp_model) * 1000,
                    'backward_ms': (t_bp_backward - t_bp_loss) * 1000,
                    'optimizer_step_ms': (t_bp_optim - t_bp_backward) * 1000,
                    'total_ms': (t_bp_optim - t_bp_start) * 1000,
                }
                if layer_timing:
                    bp_entry['layer_timing'] = layer_timing
                batch_profiles.append(bp_entry)

            t_backward_acc += time.time() - t_bwd

            for key in epoch_losses:
                epoch_losses[key] += loss_dict.get(key, 0.0)

            # Accumulate feature-level stats (ndarray, separate from scalar loop)
            fr = loss_dict.get('feature_recon_mean')
            if fr is not None:
                if _feature_accum['recon_mean'] is None:
                    import numpy as np
                    _feature_accum['recon_mean'] = fr.copy()
                    _feature_accum['recon_max'] = loss_dict['feature_recon_max'].copy()
                    fd = loss_dict.get('feature_disc_mean')
                    _feature_accum['disc_mean'] = fd.copy() if fd is not None else None
                    _feature_accum['disc_max'] = loss_dict['feature_disc_max'].copy() if fd is not None else None
                else:
                    _feature_accum['recon_mean'] += fr
                    np.maximum(_feature_accum['recon_max'], loss_dict['feature_recon_max'], out=_feature_accum['recon_max'])
                    fd = loss_dict.get('feature_disc_mean')
                    if fd is not None and _feature_accum['disc_mean'] is not None:
                        _feature_accum['disc_mean'] += fd
                        np.maximum(_feature_accum['disc_max'], loss_dict['feature_disc_max'], out=_feature_accum['disc_max'])
                _feature_batch_count += 1

        torch.cuda.synchronize()
        t_epoch_total = time.time() - t_epoch_start

        for key in epoch_losses.keys():
            epoch_losses[key] /= len(self.train_loader)

        # Feature-level epoch averages → epoch_losses (for history recording in train())
        if _feature_batch_count > 0 and _feature_accum['recon_mean'] is not None:
            epoch_losses['_feature_recon_mean'] = (_feature_accum['recon_mean'] / _feature_batch_count).tolist()
            epoch_losses['_feature_recon_max'] = _feature_accum['recon_max'].tolist()
            if _feature_accum['disc_mean'] is not None:
                epoch_losses['_feature_disc_mean'] = (_feature_accum['disc_mean'] / _feature_batch_count).tolist()
                epoch_losses['_feature_disc_max'] = _feature_accum['disc_max'].tolist()

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

    @staticmethod
    def _empty_contrib():
        """Return zero-valued contribution ratios dict."""
        return {
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

    def _print_batch_profiling(self, batch_profiles, epoch_timing):
        """Print batch profiling summary table immediately after epoch 0."""
        n = len(batch_profiles)
        n_batches = epoch_timing.get('n_batches', len(self.train_loader))
        epoch_total = epoch_timing.get('epoch_total', 0)

        components = ['data_to_gpu_ms', 'model_forward_ms', 'loss_compute_ms',
                      'backward_ms', 'optimizer_step_ms']
        labels = ['Data -> GPU', 'Model Forward', 'Loss Compute',
                  'Backward', 'Optimizer Step']

        # Layer-level components (nested inside model_forward)
        layer_components = ['embed_input_ms', 'masking_ms', 'encoder_ms',
                            'teacher_decoder_ms', 'student_decoder_ms']
        layer_labels = ['Embed (Patchify+CNN)', 'Masking', 'Encoder',
                        'Teacher Decoder', 'Student Decoder']
        has_layers = batch_profiles[0].get('layer_timing') is not None

        total_sum = 0.0
        rows = []
        layer_rows = []
        for comp, label in zip(components, labels):
            vals = [bp[comp] for bp in batch_profiles]
            total = sum(vals)
            total_sum += total
            rows.append((label, total, total / n, min(vals), max(vals)))

            # Collect layer breakdown for model_forward
            if comp == 'model_forward_ms' and has_layers:
                for lcomp, llabel in zip(layer_components, layer_labels):
                    lvals = [bp['layer_timing'][lcomp] for bp in batch_profiles]
                    ltotal = sum(lvals)
                    layer_rows.append((llabel, ltotal, ltotal / n, min(lvals), max(lvals)))

        avg_batch_ms = total_sum / n
        est_epoch_s = avg_batch_ms * n_batches / 1000
        remaining = self.config.num_epochs - 1
        est_remaining_s = est_epoch_s * remaining

        hdr = f"{'Component':<24} {'Total(ms)':>10} {'Avg(ms)':>10} {'Min':>10} {'Max':>10}"
        sep = '-' * len(hdr)
        print(f"\n  Batch Profiling ({n} batches, batch_size={self.config.batch_size}, batch 0 skipped)")
        print(f"  {sep}")
        print(f"  {hdr}")
        print(f"  {sep}")
        for i, (label, total, avg, mn, mx) in enumerate(rows):
            print(f"  {label:<24} {total:>10.1f} {avg:>10.1f} {mn:>10.1f} {mx:>10.1f}")
            # Print layer breakdown after Model Forward
            if i == 1 and layer_rows:
                for j, (llabel, ltotal, lavg, lmn, lmx) in enumerate(layer_rows):
                    prefix = '  \u2514\u2500 ' if j == len(layer_rows) - 1 else '  \u251c\u2500 '
                    print(f"  {prefix}{llabel:<20} {ltotal:>10.1f} {lavg:>10.1f} {lmn:>10.1f} {lmx:>10.1f}")
        print(f"  {sep}")
        print(f"  {'TOTAL':<24} {total_sum:>10.1f} {avg_batch_ms:>10.1f}")
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
        # Epoch offset: non-replacement random offsets within [0, stride)
        epoch_offset = getattr(self.config, 'epoch_offset', False)
        if epoch_offset:
            train_dataset = self.train_loader.dataset
            if hasattr(train_dataset, 'stride'):
                stride = train_dataset.stride
                import numpy as np
                offset_rng = np.random.RandomState(42)
                offset_pool = []  # Refilled each cycle
        for epoch in range(self.config.num_epochs):
            # Shift train window start positions each epoch (data augmentation)
            if epoch_offset and hasattr(train_dataset, 'set_epoch_offset'):
                if not offset_pool:
                    offset_pool = list(offset_rng.permutation(stride))
                train_dataset.set_epoch_offset(offset_pool.pop())

            # --- Teacher freeze (방법 C: eval + requires_grad_(False)) ---
            if (getattr(self.config, 'freeze_teacher_after_warmup', False) and
                    epoch == teacher_warmup and not hasattr(self, '_frozen_eval_modules')):
                import torch.nn as nn
                freeze_modules = ['encoder', 'shared_decoder', 'teacher_decoder',
                                  'teacher_output_projection',
                                  'patch_cnn', 'cnn_flatten_proj', 'cnn_projection',
                                  'patch_embed']
                freeze_params = ['teacher_mask_token']

                for name in freeze_modules:
                    module = getattr(self.model, name, None)
                    if module is not None:
                        module.eval()
                        for param in module.parameters():
                            param.requires_grad_(False)
                for name in freeze_params:
                    param = getattr(self.model, name, None)
                    if param is not None and isinstance(param, nn.Parameter):
                        param.requires_grad_(False)

                self._frozen_eval_modules = freeze_modules
                if self.verbose:
                    total = sum(p.numel() for p in self.model.parameters())
                    trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    print(f"  [Freeze] Teacher frozen (eval mode). "
                          f"Trainable: {trainable:,}/{total:,} params")

            # --- Encoder-only freeze ---
            if (getattr(self.config, 'freeze_encoder_only', False) and
                    epoch == teacher_warmup and not hasattr(self, '_frozen_encoder_modules')):
                import torch.nn as nn
                freeze_modules = ['encoder', 'patch_cnn', 'cnn_flatten_proj',
                                  'cnn_projection', 'patch_embed']
                for name in freeze_modules:
                    module = getattr(self.model, name, None)
                    if module is not None:
                        module.eval()
                        for param in module.parameters():
                            param.requires_grad_(False)

                self._frozen_encoder_modules = freeze_modules
                if self.verbose:
                    total = sum(p.numel() for p in self.model.parameters())
                    trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    print(f"  [Freeze] Encoder-only frozen (decoders still trainable). "
                          f"Trainable: {trainable:,}/{total:,} params")

            # GRL lambda: set BEFORE train_epoch so the current epoch uses the correct value
            if getattr(self.config, 'use_grl', False):
                import math
                _student_start = self.config.teacher_only_warmup_epochs
                _student_total = max(self.config.num_epochs - _student_start, 1)
                _student_epoch = epoch - _student_start  # 0-indexed within student phase
                _p = max(0.0, min((_student_epoch + 1) / _student_total, 1.0))
                if _student_epoch < 0:
                    self.model._grl_lambda = 0.0
                else:
                    self.model._grl_lambda = 2.0 / (1.0 + math.exp(-10.0 * _p)) - 1.0

            # --- Masking ratio annealing ---
            if getattr(self.config, 'masking_ratio_anneal', False) and epoch >= teacher_warmup:
                _anneal_progress = (epoch - teacher_warmup) / max(self.config.num_epochs - teacher_warmup - 1, 1)
                _anneal_target = 1.0 / self.config.num_patches  # 1 patch
                self._annealed_masking_ratio = (
                    self.config.masking_ratio * (1 - _anneal_progress) + _anneal_target * _anneal_progress
                )
                if self.verbose and epoch == teacher_warmup:
                    print(f"  [Anneal] masking_ratio: {self.config.masking_ratio:.3f} → "
                          f"{_anneal_target:.4f} over epochs {teacher_warmup}-{self.config.num_epochs-1}")
            else:
                self._annealed_masking_ratio = None

            # First N epochs are warm-up: train teacher only (no discrepancy/student loss)
            teacher_only = (epoch < teacher_warmup)
            # Profile only on epoch 0
            pb = profile_n_batches if epoch == 0 else 0
            epoch_losses = self.train_epoch(epoch, teacher_only=teacher_only, profile_batches=pb)
            self.scheduler.step()
            # D scheduler: step only after disc_warmup_epochs (when D is active)
            if self.use_discriminator and epoch >= self.config.disc_warmup_epochs:
                self.d_scheduler.step()

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
            self.history['train_mean_discrepancy'].append(epoch_losses.get('mean_discrepancy', 0.0))
            # Detailed metrics by sample type
            self.history['train_teacher_recon_normal'].append(epoch_losses['teacher_recon_normal'])
            self.history['train_teacher_recon_anomaly'].append(epoch_losses['teacher_recon_anomaly'])
            self.history['train_student_recon_normal'].append(epoch_losses['student_recon_normal'])
            self.history['train_student_recon_anomaly'].append(epoch_losses['student_recon_anomaly'])
            # Feature matching loss
            self.history['train_fm_loss'].append(epoch_losses['fm_loss'])
            # FM adaptive lambda
            if getattr(self.config, 'fm_adaptive_lambda', False):
                self.history['train_fm_adaptive_lambda'].append(epoch_losses.get('fm_adaptive_lambda', 0.0))
            # Feature-level stats (training)
            self.history['train_feature_recon_mean'].append(epoch_losses.pop('_feature_recon_mean', None))
            self.history['train_feature_recon_max'].append(epoch_losses.pop('_feature_recon_max', None))
            self.history['train_feature_disc_mean'].append(epoch_losses.pop('_feature_disc_mean', None))
            self.history['train_feature_disc_max'].append(epoch_losses.pop('_feature_disc_max', None))
            # Discriminator metrics
            if self.use_discriminator:
                self.history['train_d_loss'].append(epoch_losses['d_loss'])
                self.history['train_d_real_acc'].append(epoch_losses['d_real_acc'])
                self.history['train_d_fake_acc'].append(epoch_losses['d_fake_acc'])
                self.history['train_adv_loss'].append(epoch_losses['adv_loss'])
                self.history['train_adaptive_lambda'].append(epoch_losses['adaptive_lambda'])
            # GRL metrics + next epoch lambda update
            if getattr(self.config, 'use_grl', False):
                self.history['train_grl_cls_loss'].append(epoch_losses['grl_cls_loss'])
                self.history['train_grl_balanced_acc'].append(epoch_losses['grl_balanced_acc'])
                self.history['train_grl_anomaly_acc'].append(epoch_losses.get('grl_anomaly_acc', 0.0))
                self.history['train_grl_normal_acc'].append(epoch_losses.get('grl_normal_acc', 0.0))
                self.history['train_grl_lambda'].append(epoch_losses['grl_lambda'])
                self.history['train_grl_effective_weight'].append(epoch_losses.get('grl_effective_weight', 0.0))
                # _grl_lambda is now set BEFORE train_epoch (see above), no post-epoch update needed

            if getattr(self.config, 'use_scad', False):
                self.history['train_scad_loss'].append(epoch_losses.get('scad_loss', 0.0))
                self.history['train_scad_n_anom'].append(epoch_losses.get('scad_n_anom', 0))
                self.history['train_scad_n_norm'].append(epoch_losses.get('scad_n_norm', 0))
                self.history['train_scad_z_separation'].append(epoch_losses.get('scad_z_separation', 0.0))
                self.history['train_scad_z_anom_var'].append(epoch_losses.get('scad_z_anom_var', 0.0))
                self.history['train_scad_z_norm_var'].append(epoch_losses.get('scad_z_norm_var', 0.0))
                self.history['train_scad_lambda'].append(epoch_losses.get('scad_lambda', 0.0))
                self.history['train_scad_adaptive_lambda'].append(epoch_losses.get('scad_adaptive_lambda', 0.0))
                self.history['train_scad_ramp'].append(epoch_losses.get('scad_ramp', 0.0))
                self.history['train_scad_effective_weight'].append(epoch_losses.get('scad_effective_weight', 0.0))
                self.history['train_scad_grad_norm'].append(epoch_losses.get('scad_grad_norm', 0.0))
                self.history['train_scad_main_grad_norm'].append(epoch_losses.get('scad_main_grad_norm', 0.0))

            # --- Update prev-epoch adaptive lambdas (for next epoch) ---
            _adv_l = epoch_losses.get('adaptive_lambda', 0.0)
            if _adv_l > 0:
                self._prev_epoch_adv_lambda = _adv_l
            _fm_l = epoch_losses.get('fm_adaptive_lambda', 0.0)
            if _fm_l > 0:
                self._prev_epoch_fm_lambda = _fm_l
            _grl_l = epoch_losses.get('grl_lambda', 0.0)
            if _grl_l > 0:
                self._prev_epoch_grl_lambda = _grl_l

            # Epoch callback (epoch-wise test evaluation + contrib ratio computation)
            # Callback computes contrib ratios from its GPU inference data (no extra inference).
            # Sets self._pending_contrib on eval epochs; non-eval epochs use cached values.
            self._pending_contrib = None
            t_callback_start = time.time()
            if epoch_callback is not None:
                epoch_callback(epoch, self.model, self.history)
            t_callback = time.time() - t_callback_start

            # Record contribution ratios from callback (or cached / zeros)
            if self._pending_contrib is not None:
                contrib = self._pending_contrib
                self._last_contrib = contrib
                self._pending_contrib = None
            else:
                contrib = getattr(self, '_last_contrib', self._empty_contrib())
            for key_suffix in ['recon_ratio_normal', 'recon_ratio_disturbing', 'recon_ratio_anomaly',
                               'disc_ratio_normal', 'disc_ratio_disturbing', 'disc_ratio_anomaly',
                               'recon_score_normal', 'recon_score_disturbing', 'recon_score_anomaly',
                               'disc_score_normal', 'disc_score_disturbing', 'disc_score_anomaly',
                               'raw_recon_normal', 'raw_recon_disturbing', 'raw_recon_anomaly',
                               'raw_disc_normal', 'raw_disc_disturbing', 'raw_disc_anomaly']:
                self.history[f'epoch_{key_suffix}'].append(contrib[key_suffix])
            self.history['epoch_anomaly_type_scores'].append(contrib['anomaly_type_scores'])

            # Record per-epoch timing
            epoch_timing['callback'] = t_callback
            self.history.setdefault('epoch_timings', []).append(epoch_timing)

        return self.history
