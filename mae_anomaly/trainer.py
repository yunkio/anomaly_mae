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

        # 11. loss_balance_mode (단일 enum, 상호배타 — 한 값만 선택 가능하므로 구조적 보장).
        #     adaptive_lambda_legacy(기본)는 기존 grl_adaptive_lambda bool을 그대로 따른다.
        _VALID_LBM = {'adaptive_lambda_legacy', 'fixed', 'mse_norm_dann', 'relobralo', 'famo', 'uwso'}
        _lbm = getattr(config, 'loss_balance_mode', 'adaptive_lambda_legacy')
        if _lbm not in _VALID_LBM:
            raise ValueError(f"loss_balance_mode must be one of {_VALID_LBM}, got {_lbm!r}")
        if _lbm not in ('adaptive_lambda_legacy', 'fixed'):
            # NEW scale-matching modes only operate on the classifier-mode GRL BCE term.
            if not getattr(config, 'use_grl', False):
                raise ValueError(f"loss_balance_mode={_lbm!r}는 use_grl=True를 필요로 합니다.")
            if getattr(config, 'grl_mode', 'classifier') != 'classifier':
                raise ValueError(
                    f"loss_balance_mode={_lbm!r}는 grl_mode='classifier'에서만 동작합니다 "
                    f"(현재 {getattr(config, 'grl_mode', 'classifier')!r}; WDGRL은 자체 minimax 사용).")
            if getattr(config, 'use_scad', False):
                raise ValueError(f"loss_balance_mode={_lbm!r}와 use_scad=True는 동시 사용 불가합니다.")

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

        # fused=True (2026-05-29): single-GPU CUDA-fused AdamW kernel,
        # cuts optimizer step from ~4 ms to ~1.5 ms (-1.7% total batch time).
        # No accuracy impact; AdamW fused param verified in torch 2.4.1+cu118.
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.99),  # Compromise: 0.95 (original MAE, 75% masking) ↔ 0.999 (PyTorch default)
            fused=True,
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

        # loss_balance_mode state (2026-06-14; NEW Axis-A scale-matchers only —
        # legacy/fixed paths never read these, so default behavior is unaffected).
        self._lbm = getattr(config, 'loss_balance_mode', 'adaptive_lambda_legacy')
        self._mn_ema_bce = None            # mse_norm_dann: EMA of |grl_cls_loss|
        self._mn_ema_mse = None            # (symmetry/logging)
        self._rlb_lam = [1.0, 1.0]         # relobralo EMA weights [MSE, BCE]
        self._rlb_l = [1.0, 1.0]           # relobralo prev losses
        self._rlb_l0 = [1.0, 1.0]          # relobralo onset losses
        self._rlb_l0_captured = False
        self._rlb_steps_done = 0
        self._rlb_last_epoch = -1
        self._rlb_rng = random.Random(getattr(config, 'random_seed', 0) + 777)
        self._uwso_ema_mse = None          # uwso EMA losses (when uwso_ema_beta<1.0)
        self._uwso_ema_bce = None
        self._famo = None                  # famo lazy dict(w, opt, min, prev)

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
        amp_dtype_name = getattr(config, 'amp_dtype', 'fp16').lower()
        if amp_dtype_name == 'fp16':
            self.amp_dtype = torch.float16
        elif amp_dtype_name == 'bf16':
            self.amp_dtype = torch.bfloat16
            # bf16 requires Ampere (sm_80) or later. Reject silent fallback on older GPUs.
            if self.use_amp:
                _cc = torch.cuda.get_device_capability()
                if _cc < (8, 0):
                    raise RuntimeError(
                        f"amp_dtype='bf16' requires CUDA capability >= 8.0 (Ampere/Ada/Hopper); "
                        f"got sm_{_cc[0]}{_cc[1]}. Use amp_dtype='fp16' on older GPUs."
                    )
        else:
            raise ValueError(
                f"config.amp_dtype must be 'fp16' or 'bf16', got {amp_dtype_name!r}"
            )
        # GradScaler is only needed for fp16 (bf16 has the same exponent range as fp32 →
        # gradient underflow does not occur; using a scaler would be a no-op).
        self.scaler = (
            GradScaler('cuda')
            if (self.use_amp and self.amp_dtype is torch.float16)
            else None
        )
        # AMP visibility — print once at init so default-flip (2026-05-27: fp16 -> bf16) is observable.
        print(
            f"  AMP: use_amp={self.use_amp}, dtype={amp_dtype_name}, "
            f"scaler={'enabled' if self.scaler is not None else 'none'}"
        )

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
                fused=True,         # CUDA fused kernel (2026-05-29)
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
            # |normal_acc - anomaly_acc| degeneracy gap (2026-06-01): balanced_acc=0.5
            # is deceptive — it coexists with full degeneracy (normal=0, anomaly=1).
            # gap=0 → balanced; gap→1 → degenerate. Read WITH balanced_acc, not alone.
            'train_grl_acc_gap': [],
            'train_grl_lambda': [],
            'train_grl_effective_weight': [],  # lambda * grl_loss_weight = actual multiplier
            # SCAD metrics (populated only when use_scad=True) — mirror GRL pattern
            'train_scad_loss': [],
            'train_scad_n_anom': [],
            'train_scad_n_norm': [],
            'train_scad_z_separation': [],
            'train_scad_z_anom_var': [],
            'train_scad_z_norm_var': [],
            'train_scad_c_mean_sim': [],            # Form C: mean cos(z_a, z_u)
            'train_scad_c_active_pair_frac': [],    # Form C: frac of pairs with cos > gamma (loss-active)
            'train_scad_c_active_sim_mean': [],     # Form C: mean cos over active pairs
            'train_scad_c_gamma': [],               # Form C: gamma threshold (echo)
            'train_scad_c_n_anchor': [],            # Form C: # anomaly anchors
            'train_scad_c_n_u': [],                 # Form C: # U(background) negatives
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

    # ===== loss_balance_mode helpers (2026-06-14) — NEW Axis-A scale-matchers ONLY. =====
    # Never called for adaptive_lambda_legacy/fixed (default), so they cannot affect the
    # legacy path. All weights are computed from DETACHED magnitudes (stop-gradient);
    # gradient flows only through the trailing loss tensors. They never touch the
    # gradient-reversal ramp (model._grl_lambda) — no double-ramp.
    def _lbm_apply(self, mode, loss, grl_cls_loss, grl_w, loss_tensors, epoch, teacher_only):
        if mode == 'mse_norm_dann':
            return self._lbm_mse_norm_dann(loss, grl_cls_loss, grl_w, epoch)
        elif mode == 'relobralo':
            return self._lbm_relobralo(loss, grl_cls_loss, epoch)
        elif mode == 'uwso':
            return self._lbm_uwso(loss, grl_cls_loss)
        elif mode == 'famo':
            return self._lbm_famo(loss, grl_cls_loss)
        raise ValueError(f"unknown loss_balance_mode in _lbm_apply: {mode!r}")

    def _lbm_mse_norm_dann(self, loss, grl_cls_loss, grl_w, epoch):
        # BCE scale-normalization (divide BCE by EMA(|BCE|) -> stable O(1) so it is NOT
        # starved as the MSE numerator decays) + Ganin deterministic ramp (runaway-proof:
        # weight is a pure function of training progress, not of either gradient norm).
        cfg = self.config
        with torch.no_grad():
            bce_mag = float(grl_cls_loss.detach().float())
            mse_mag = float(loss.detach().float())
            b = cfg.mse_norm_ema_beta
            if self._mn_ema_bce is None:
                self._mn_ema_bce, self._mn_ema_mse = bce_mag, mse_mag
            else:
                self._mn_ema_bce = b * bce_mag + (1.0 - b) * self._mn_ema_bce
                self._mn_ema_mse = b * mse_mag + (1.0 - b) * self._mn_ema_mse
        eps = cfg.mse_norm_eps
        horizon = max(1, int(getattr(cfg, 'dann_ramp_horizon', 100)))
        p = (epoch - cfg.teacher_only_warmup_epochs) / horizon
        p = min(max(p, 0.0), 1.0)
        lam_p = 2.0 / (1.0 + math.exp(-cfg.dann_ramp_gamma * p)) - 1.0
        if cfg.mse_norm_log_variant:
            w = lam_p * grl_w
            loss = loss + w * torch.log(grl_cls_loss + eps)
        else:
            s_bce = 1.0 / (self._mn_ema_bce + eps)
            w = lam_p * grl_w * s_bce
            loss = loss + w * grl_cls_loss
        return loss, float(w)

    def _lbm_relobralo(self, loss, grl_cls_loss, epoch):
        # Bischof & Kraus 2110.09813: per-term softmax of loss RATIOS (vs prev step + vs onset),
        # random Bernoulli(rho) lookback, EMA(alpha) of weights. m=2 [MSE=loss, BCE=grl_cls].
        # recon+disc anchored at weight 1; only the BCE term gets the relative weight lam_bce/lam_mse.
        cfg = self.config
        T = cfg.relobralo_T; alpha = cfg.relobralo_alpha; rho_p = cfg.relobralo_rho; eps = cfg.relobralo_eps
        update_now = (epoch != self._rlb_last_epoch) if getattr(cfg, 'relobralo_update_freq', 'epoch') == 'epoch' else True
        with torch.no_grad():
            l_mse = float(loss.detach().float()); l_bce = float(grl_cls_loss.detach().float())
            if update_now:
                if not self._rlb_l0_captured:
                    self._rlb_l0 = [max(l_mse, eps), max(l_bce, eps)]; self._rlb_l0_captured = True

                def _sm2(ratios):
                    mx = max(ratios); e = [math.exp(r - mx) for r in ratios]; Z = sum(e)
                    return [2.0 * ei / Z for ei in e]  # *m=2 rescale (mean 1)
                cur = [max(l_mse, eps), max(l_bce, eps)]
                prev = [max(self._rlb_l[0], eps), max(self._rlb_l[1], eps)]
                onset = [max(self._rlb_l0[0], eps), max(self._rlb_l0[1], eps)]
                lamb_hat = _sm2([cur[0] / (T * prev[0]), cur[1] / (T * prev[1])])
                lamb0_hat = _sm2([cur[0] / (T * onset[0]), cur[1] / (T * onset[1])])
                if self._rlb_steps_done == 0:      # onset epoch: freeze weights=1, capture l0
                    a = 1.0
                elif self._rlb_steps_done == 1:    # 2nd: fresh ratio
                    a = 0.0
                else:
                    a = alpha
                rho_t = 1.0 if self._rlb_rng.random() < rho_p else 0.0
                self._rlb_lam = [rho_t * a * self._rlb_lam[i] + (1.0 - rho_t) * a * lamb0_hat[i]
                                 + (1.0 - a) * lamb_hat[i] for i in range(2)]
                self._rlb_l = [l_mse, l_bce]
                self._rlb_last_epoch = epoch
                self._rlb_steps_done += 1
        rel = self._rlb_lam[1] / max(self._rlb_lam[0], eps)
        loss = loss + rel * grl_cls_loss
        return loss, float(rel)

    def _lbm_uwso(self, loss, grl_cls_loss):
        # Kirchdorfer et al. 2408.07985 (IJCV 2025) Eq.4: tempered-softmax over (1/L) with
        # log-sum-exp stabilization; loss floors cap 1/L blow-up as MSE->0. recon anchored;
        # BCE gets relative weight w_bce/w_mse. sigma closed-form (not learned) -> no extra params.
        cfg = self.config; T = cfg.uwso_temperature
        with torch.no_grad():
            Lm = max(float(loss.detach().float()), cfg.uwso_loss_floor_mse)
            Lb = max(float(grl_cls_loss.detach().float()), cfg.uwso_loss_floor_bce)
            if cfg.uwso_ema_beta < 1.0:
                b = cfg.uwso_ema_beta
                if self._uwso_ema_mse is None:
                    self._uwso_ema_mse, self._uwso_ema_bce = Lm, Lb
                else:
                    self._uwso_ema_mse = b * Lm + (1.0 - b) * self._uwso_ema_mse
                    self._uwso_ema_bce = b * Lb + (1.0 - b) * self._uwso_ema_bce
                Lm, Lb = self._uwso_ema_mse, self._uwso_ema_bce
            a_m = (1.0 / Lm) / T; a_b = (1.0 / Lb) / T
            mx = max(a_m, a_b); e_m = math.exp(a_m - mx); e_b = math.exp(a_b - mx); Z = e_m + e_b
            w_m = e_m / Z; w_b = e_b / Z
        rel = w_b / max(w_m, 1e-12)
        loss = loss + rel * grl_cls_loss
        return loss, float(rel)

    def _lbm_famo(self, loss, grl_cls_loss):
        # Liu et al. NeurIPS 2023 (Cranial-XIX/FAMO) — log-loss simplex balancer. 2 tasks:
        # [task0 = recon+disc (loss), task1 = BCE (grl_cls_loss)]. Full MTL balancer:
        # it reweights BOTH tasks (recon NOT anchored here) via the log-combination, and
        # updates softmax logits w by the consecutive-step log-loss change (streaming/
        # next-batch approximation of the official post-step re-forward). O(1), no per-task grads.
        cfg = self.config
        if self._famo is None:
            dev = loss.device
            w = torch.zeros(2, device=dev, requires_grad=True)
            opt = torch.optim.Adam([w], lr=cfg.famo_w_lr, weight_decay=cfg.famo_gamma)
            self._famo = {'w': w, 'opt': opt, 'min': torch.zeros(2, device=dev), 'prev': None}
        f = self._famo
        with torch.no_grad():
            curr = torch.stack([loss.detach().float(), grl_cls_loss.detach().float()])
        if f['prev'] is not None:
            delta = (f['prev'] - f['min'] + 1e-8).log() - (curr - f['min'] + 1e-8).log()
            with torch.enable_grad():
                z_w = torch.softmax(f['w'], -1)
                d = torch.autograd.grad(z_w, f['w'], grad_outputs=delta.detach())[0]
            f['opt'].zero_grad(); f['w'].grad = d; f['opt'].step()
        f['prev'] = curr
        z = torch.softmax(f['w'], -1).detach()                 # detached weight (safe for model backward)
        D = torch.stack([loss, grl_cls_loss]) - f['min'] + 1e-8
        c = (z / D.detach()).sum().detach()
        weighted = (D.log() * z / c).sum()
        return weighted, float(z[1].item())

    def _lbm_state_dict(self):
        """Serialize loss_balance_mode runtime state for checkpoint resume.
        Plain dict (CPU tensors); legacy/fixed modes carry only inert defaults."""
        st = {
            'mode': self._lbm,
            'mn_ema_bce': self._mn_ema_bce, 'mn_ema_mse': self._mn_ema_mse,
            'rlb_lam': list(self._rlb_lam), 'rlb_l': list(self._rlb_l), 'rlb_l0': list(self._rlb_l0),
            'rlb_l0_captured': self._rlb_l0_captured, 'rlb_steps_done': self._rlb_steps_done,
            'rlb_last_epoch': self._rlb_last_epoch, 'rlb_rng': self._rlb_rng.getstate(),
            'uwso_ema_mse': self._uwso_ema_mse, 'uwso_ema_bce': self._uwso_ema_bce,
        }
        if self._famo is not None:
            st['famo'] = {
                'w': self._famo['w'].detach().cpu(),
                'opt': self._famo['opt'].state_dict(),
                'min': self._famo['min'].detach().cpu(),
                'prev': None if self._famo['prev'] is None else self._famo['prev'].detach().cpu(),
            }
        return st

    def _lbm_load_state_dict(self, st):
        """Restore loss_balance_mode state; no-op for absent/None (legacy ckpt back-compat)."""
        if not st:
            return
        self._mn_ema_bce = st.get('mn_ema_bce'); self._mn_ema_mse = st.get('mn_ema_mse')
        self._rlb_lam = list(st.get('rlb_lam', [1.0, 1.0]))
        self._rlb_l = list(st.get('rlb_l', [1.0, 1.0]))
        self._rlb_l0 = list(st.get('rlb_l0', [1.0, 1.0]))
        self._rlb_l0_captured = st.get('rlb_l0_captured', False)
        self._rlb_steps_done = st.get('rlb_steps_done', 0)
        self._rlb_last_epoch = st.get('rlb_last_epoch', -1)
        if st.get('rlb_rng') is not None:
            try:
                self._rlb_rng.setstate(st['rlb_rng'])
            except (TypeError, ValueError):
                pass
        self._uwso_ema_mse = st.get('uwso_ema_mse'); self._uwso_ema_bce = st.get('uwso_ema_bce')
        _f = st.get('famo')
        if _f is not None:
            dev = next(self.model.parameters()).device
            w = _f['w'].to(dev).detach().requires_grad_(True)
            opt = torch.optim.Adam([w], lr=self.config.famo_w_lr, weight_decay=self.config.famo_gamma)
            opt.load_state_dict(_f['opt'])
            self._famo = {'w': w, 'opt': opt, 'min': _f['min'].to(dev),
                          'prev': None if _f['prev'] is None else _f['prev'].to(dev)}

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

        # Gradient health stats. clip_grad_norm_ already runs every batch and returns the
        # pre-clip norm; we just capture its return value (no extra grad traversal).
        # On NaN/Inf the optimizer.step() is skipped — important under bf16 which has no
        # GradScaler safety net (fp16 path's GradScaler.step does its own NaN check).
        _grad_norm_sum = 0.0
        _grad_norm_max = 0.0
        _grad_norm_finite_n = 0
        _grad_nonfinite_n = 0

        # [신규 2026-06-01] teacher warmup early-stop 메트릭: train recon_snr 누적기.
        # per-sample teacher recon을 normal/anomaly로 나눠 epoch 단위 mean·std → SNR 계산.
        # GPU 0-dim tensor로 누적 후 epoch 끝에서 1회 .item() (배치별 sync 회피). OFF면 미사용.
        _es_on = getattr(self.config, 'use_teacher_warmup_early_stop', False)
        _es_sum_n = _es_sumsq_n = _es_cnt_n = 0.0
        _es_sum_a = _es_sumsq_a = _es_cnt_a = 0.0

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
            with autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                # 2026-05-29: propagate teacher_only so model can skip student
                # decoder / GRL classifier / SCAD head forward during warmup.
                # During warmup these are computed but their outputs feed only
                # into loss tensors that are gated out at loss.py:196 and at
                # trainer.py:597/620/704 — so the forward compute was wasted.
                # Evaluator and visualizer paths leave teacher_only at default
                # False, so they still get full student forward.
                teacher_output, student_output, mask = self.model(
                    sequences, masking_ratio=_mr, point_labels=point_labels,
                    teacher_only=teacher_only,
                )

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

                # [신규 2026-06-01] EMA teacher 출력(있으면) — student discrepancy 표적으로 전달.
                # _ema_active가 아니면 model이 None으로 두므로 loss는 기존(live teacher) 동작.
                _ema_t_out = getattr(self.model, '_ema_teacher_output', None)
                loss, loss_dict, loss_tensors = self.criterion(
                    teacher_output, student_output, sequences, mask, point_labels, warmup_factor,
                    teacher_only=teacher_only, grl_cls_logits=_grl_logits,
                    teacher_hidden=_t_hidden, student_hidden=_s_hidden,
                    scad_z=_scad_z, ema_teacher_output=_ema_t_out
                )

            # [신규 2026-06-01] early-stop용 train recon_snr 누적 (GPU 0-dim, no-grad).
            # warmup(teacher_only) 중에만 누적 — 메트릭은 warmup 동안만 소비되므로 post-warmup 낭비 방지.
            if _es_on and teacher_only and 'es_teacher_recon_per_sample' in loss_tensors:
                _r = loss_tensors['es_teacher_recon_per_sample']
                _isn = loss_tensors['es_is_normal_sample']
                _isa = loss_tensors['es_has_anomaly_sample']
                _es_sum_n = _es_sum_n + (_r * _isn).sum()
                _es_sumsq_n = _es_sumsq_n + (_r * _r * _isn).sum()
                _es_cnt_n = _es_cnt_n + _isn.sum()
                _es_sum_a = _es_sum_a + (_r * _isa).sum()
                _es_sumsq_a = _es_sumsq_a + (_r * _r * _isa).sum()
                _es_cnt_a = _es_cnt_a + _isa.sum()

            if do_profile:
                torch.cuda.synchronize()
                t_bp_loss = time.time()

            # --- Discriminator step (D → Student, TTUR order) ---
            if self.use_discriminator and epoch >= self.config.disc_warmup_epochs and not teacher_only:
                real_patches, fake_patches, anomaly_patch_mask = self._extract_patches(
                    sequences, student_output, mask, point_labels)

                # D training: real vs fake on ALL masked patches
                self.d_optimizer.zero_grad()
                with autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
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
                    with autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
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
                _lbm = getattr(self.config, 'loss_balance_mode', 'adaptive_lambda_legacy')

                if _lbm == 'adaptive_lambda_legacy':
                    # ===== LEGACY (byte-identical to pre-2026-06-14): do NOT modify these lines =====
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
                    # ===== END LEGACY =====
                elif _lbm == 'fixed':
                    # Explicit fixed weight (enum alias; fixed_grl_weight<0 → grl_loss_weight)
                    _fw = getattr(self.config, 'fixed_grl_weight', -1.0)
                    _eff = _grl_w if _fw < 0 else _fw
                    loss = loss + _eff * _grl_cls_loss
                    loss_dict['grl_lambda'] = 1.0
                    loss_dict['grl_effective_weight'] = float(_eff)
                else:
                    # New Axis-A scale-matching modes (mse_norm_dann / relobralo / uwso / famo).
                    # All isolated in _lbm_apply; loss_dict uses only existing keys (schema unchanged).
                    loss, _eff_log = self._lbm_apply(
                        _lbm, loss, _grl_cls_loss, _grl_w, loss_tensors, epoch, teacher_only)
                    loss_dict['grl_lambda'] = 1.0
                    loss_dict['grl_effective_weight'] = float(_eff_log)
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
                _gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)  # GradScaler.step internally skips on NaN/Inf
                self.scaler.update()
            else:
                loss.backward()
                if do_profile:
                    torch.cuda.synchronize()
                    t_bp_backward = time.time()
                _gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # No GradScaler (bf16/fp32) → we must guard step() ourselves.
                if torch.isfinite(_gnorm):
                    self.optimizer.step()
                else:
                    _grad_nonfinite_n += 1
                    print(
                        f"  [WARN] NaN/Inf grad_norm at epoch {epoch+1} batch {batch_idx}: "
                        f"{_gnorm.item():.3e} — optimizer.step skipped"
                    )

            # [신규 2026-06-01] EMA teacher 갱신 (use_teacher_output_ema=False면 내부 no-op).
            # 매 optimizer step 호출 → warmup 도중부터 누적되어 warmup 종료 시 평탄한 표적 제공.
            if getattr(self.config, 'use_teacher_output_ema', False):
                self.model.update_teacher_output_ema(
                    getattr(self.config, 'teacher_output_ema_momentum', 0.996))

            # Track grad health (no extra CUDA work; clip_grad_norm_ already computed norm)
            _g = _gnorm.item()
            if _g == _g and _g != float('inf') and _g != float('-inf'):  # fast finite check
                _grad_norm_sum += _g
                _grad_norm_finite_n += 1
                if _g > _grad_norm_max:
                    _grad_norm_max = _g

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

        # [신규 2026-06-01] early-stop용 epoch 단위 train recon_snr (division loop 뒤에서
        # '_' 접두 키로 추가 → 위 평균화에 영향 없음). 공식: (mean_a − mean_n)/(σ_a + σ_n + ε),
        # TEST recon_SNR(run_ablation.py:616)과 동일한 Cohen's-d 형 분리도를 train data로 계산.
        if _es_on and torch.is_tensor(_es_cnt_n):
            _cn = _es_cnt_n.item(); _ca = _es_cnt_a.item()
            if _cn > 0 and _ca > 0:
                _mn = _es_sum_n.item() / _cn
                _ma = _es_sum_a.item() / _ca
                _vn = max(_es_sumsq_n.item() / _cn - _mn * _mn, 0.0)
                _va = max(_es_sumsq_a.item() / _ca - _ma * _ma, 0.0)
                _sn = _vn ** 0.5; _sa = _va ** 0.5
                epoch_losses['_train_recon_snr'] = (_ma - _mn) / (_sa + _sn + 1e-8)
            else:
                # anomaly 또는 normal 표본이 epoch 내 전무 → SNR 미정의. None 표식.
                epoch_losses['_train_recon_snr'] = None

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

        # Gradient health summary (one log line per epoch + epoch_losses keys)
        epoch_losses['grad_norm_mean'] = (
            _grad_norm_sum / _grad_norm_finite_n if _grad_norm_finite_n > 0 else 0.0
        )
        epoch_losses['grad_norm_max'] = _grad_norm_max
        epoch_losses['grad_nonfinite_batches'] = _grad_nonfinite_n
        if _grad_nonfinite_n > 0:
            print(
                f"  [CRITICAL] epoch {epoch+1}: {_grad_nonfinite_n} batch(es) had NaN/Inf gradients "
                f"(steps skipped). grad_norm mean(finite)={epoch_losses['grad_norm_mean']:.3e} "
                f"max(finite)={_grad_norm_max:.3e}. Investigate before continuing."
            )

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

    def train(self, epoch_callback=None, profile_n_batches: int = 0,
              start_epoch: int = 0, pre_epoch_hook=None, post_epoch_callback=None) -> Dict:
        """Train the model for num_epochs.

        Args:
            epoch_callback: Optional callable(epoch, model, history) invoked MID-epoch
                           (after train_loss is recorded, before the per-epoch
                           contribution-ratio keys are appended). Use for lightweight
                           epoch-wise test evaluation that must read the model while it
                           still holds this epoch's weights.
            profile_n_batches: If > 0, profile first N batches of epoch 0 with per-component
                              cuda.synchronize() timing. Results stored in history['batch_profiling'].
            start_epoch: 0-indexed epoch to start from. Set > 0 on resume after loading
                              optimizer/scheduler/RNG state externally. Default 0 = fresh.
                              Added 2026-05-28 for crash-resume support.
            pre_epoch_hook: Optional callable(epoch_idx) invoked at the START of each epoch
                              BEFORE batches are iterated. Used by caller to reseed
                              DataLoader's explicit generator deterministically per-epoch
                              (so resume produces identical sample order). Default None.
            post_epoch_callback: Optional callable(epoch, history) invoked at the very END
                              of each epoch AFTER every per-epoch history key has been
                              appended. Use for checkpoint saving so the persisted
                              history is COMPLETE (all per-epoch arrays equal length) —
                              this is what makes resume-then-finish produce a consistent
                              history (2026-05-30 score-contribution off-by-one fix).
        """
        teacher_warmup = self.config.teacher_only_warmup_epochs

        # [신규 2026-06-01] === teacher warmup early-stop 상태 (train recon_snr plateau) ===
        # warmup 중 train recon_snr를 strict-max 추적하다 patience 초과 + min_epochs 도달 시,
        # best epoch의 model+optimizer+scheduler로 full revert + warmup 동적 종료.
        # teacher_only_warmup_epochs는 상한(early-stop은 단축만). default OFF면 전 구간 no-op.
        _es_enabled = getattr(self.config, 'use_teacher_warmup_early_stop', False)
        _es_patience = int(getattr(self.config, 'teacher_warmup_early_stop_patience', 10))
        _es_min_epochs = int(getattr(self.config, 'teacher_warmup_early_stop_min_epochs', 50))
        _es_best_snr = None
        _es_best_epoch = -1
        _es_best_snapshot = None
        _es_triggered = False
        import copy as _es_copy

        def _es_clone_state(_obj):
            """state_dict 재귀 deep-clone (tensor는 현재 device 유지; revert 안전)."""
            if torch.is_tensor(_obj):
                return _obj.detach().clone()
            if isinstance(_obj, dict):
                return {_k: _es_clone_state(_v) for _k, _v in _obj.items()}
            if isinstance(_obj, list):
                return [_es_clone_state(_v) for _v in _obj]
            if isinstance(_obj, tuple):
                return tuple(_es_clone_state(_v) for _v in _obj)
            return _es_copy.deepcopy(_obj)

        # Epoch offset: deterministic per-cycle permutation (was stateful pool — broke resume).
        # Same set of offsets per cycle as before; the assignment of position-within-cycle to
        # epoch is also deterministic by (cycle_idx, position). Replaced 2026-05-28.
        epoch_offset = getattr(self.config, 'epoch_offset', False)
        train_dataset = self.train_loader.dataset if epoch_offset else None
        stride = getattr(train_dataset, 'stride', None) if epoch_offset else None
        import numpy as np

        def _epoch_offset_for(epoch_idx, _stride):
            """Deterministic offset for given epoch index. Resume-safe."""
            cycle_idx = epoch_idx // _stride
            pos_in_cycle = epoch_idx % _stride
            _rng = np.random.RandomState(42 + cycle_idx)
            return int(_rng.permutation(_stride)[pos_in_cycle])

        for epoch in range(start_epoch, self.config.num_epochs):
            # 2026-06-11: periodic GC + CUDA cache release. Fast-epoch datasets
            # (e.g. MSL: 109 windows = 1 batch/epoch ~0.2s) outpace CPython's
            # cyclic GC, so reclaimable host memory (cyclic refs holding
            # d_model-sized tensors) bloats ~linearly → ~30GB host-RAM OOM by
            # ep~420 at d_model=768 (WaDi @768 survives only because its 60
            # batches/epoch give GC time to keep up). Every-10-epoch collect
            # bounds it (<=~650MB sawtooth); cost is negligible on large datasets
            # (epoch 8-21s) and tiny in absolute terms on fast ones. Results unchanged.
            if epoch % 10 == 0:
                import gc as _gc
                _gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # Pre-epoch hook (DataLoader generator reseed for resume-safe sample order)
            if pre_epoch_hook is not None:
                pre_epoch_hook(epoch)
            # Shift train window start positions each epoch (data augmentation)
            if epoch_offset and stride is not None and hasattr(train_dataset, 'set_epoch_offset'):
                train_dataset.set_epoch_offset(_epoch_offset_for(epoch, stride))

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
            # [신규 2026-06-01] EMA teacher 출력 표적 활성 게이트.
            #   조건: (1) flag ON  (2) post-warmup(teacher_only=False)  (3) teacher가 계속 학습(freeze=False).
            #   freeze 시에는 live teacher가 고정이므로 EMA가 무의미 → 비활성(상호배타). 비활성이면
            #   model.forward가 _ema_teacher_output=None을 두어 loss는 기존 live-teacher 동작 유지.
            self.model._ema_active = (
                getattr(self.config, 'use_teacher_output_ema', False)
                and (not teacher_only)
                and (not getattr(self.config, 'freeze_teacher_after_warmup', False))
            )
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
                # Degeneracy gap derived from the same stored (epoch-mean) accuracies →
                # exactly comparable to balanced_acc, works for every GRL variant without
                # touching their compute paths. (WGAN/WDGRL path leaves these as scores.)
                self.history['train_grl_acc_gap'].append(
                    abs(epoch_losses.get('grl_normal_acc', 0.0) - epoch_losses.get('grl_anomaly_acc', 0.0)))
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
                self.history['train_scad_c_mean_sim'].append(epoch_losses.get('scad_c_mean_sim', 0.0))
                self.history['train_scad_c_active_pair_frac'].append(epoch_losses.get('scad_c_active_pair_frac', 0.0))
                self.history['train_scad_c_active_sim_mean'].append(epoch_losses.get('scad_c_active_sim_mean', 0.0))
                self.history['train_scad_c_gamma'].append(epoch_losses.get('scad_c_gamma', 0.0))
                self.history['train_scad_c_n_anchor'].append(epoch_losses.get('scad_c_n_anchor', 0))
                self.history['train_scad_c_n_u'].append(epoch_losses.get('scad_c_n_u', 0))
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

            # [신규 2026-06-01] === teacher warmup early-stop: best 추적 + 트리거/revert ===
            # 위치: history append 후, epoch_callback(아래 GPU eval) 전. revert가 eval/checkpoint
            # 보다 먼저 일어나야 둘 다 reverted 가중치를 반영(record-consistency 보존).
            # warmup 중(teacher_only=True)에만 동작. 미트리거 상태에서만 평가.
            if _es_enabled and not _es_triggered and teacher_only:
                _snr = epoch_losses.get('_train_recon_snr', None)
                if _snr is not None:
                    if _es_best_snr is None or _snr > _es_best_snr:  # strict-max (no min_delta)
                        _es_best_snr = _snr
                        _es_best_epoch = epoch
                        # best epoch 상태 스냅샷 (scheduler.step 이후라 epoch-end 상태와 일치)
                        _es_best_snapshot = {
                            'model': _es_clone_state(self.model.state_dict()),
                            'optim': _es_clone_state(self.optimizer.state_dict()),
                            'sched': _es_clone_state(self.scheduler.state_dict()),
                            'epoch': epoch,
                        }
                    elif ((epoch - _es_best_epoch) >= _es_patience
                            and (epoch + 1) >= _es_min_epochs
                            and _es_best_snapshot is not None):
                        # === 트리거: best recon_snr epoch으로 full revert ===
                        # model state_dict는 EMA 모듈(ema_*)도 포함 → best epoch B의 '누적된'
                        # EMA가 그대로 복원됨(reset 호출 금지: 누적 평활을 버리면 사용자 의도 위배).
                        self.model.load_state_dict(_es_best_snapshot['model'])
                        self.optimizer.load_state_dict(_es_best_snapshot['optim'])
                        self.scheduler.load_state_dict(_es_best_snapshot['sched'])
                        _new_warmup = epoch + 1
                        teacher_warmup = _new_warmup                       # local: teacher_only/freeze/anneal
                        self.config.teacher_only_warmup_epochs = _new_warmup  # config: GRL _student_start/SCAD ramp
                        self._early_stopped_warmup_end = _new_warmup       # run_base가 checkpoint에 persist
                        _es_triggered = True
                        _es_best_snapshot = None                           # GPU 메모리 해제
                        if self.verbose:
                            print(f"  [WarmupEarlyStop] ep{epoch+1}: train recon_snr plateau "
                                  f"(best={_es_best_snr:.4f} @ ep{_es_best_epoch+1}, "
                                  f"patience={_es_patience}, min_epochs={_es_min_epochs}). "
                                  f"Full-reverted to best; warmup ends now → "
                                  f"teacher_only_warmup_epochs={_new_warmup}.")

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

            # Post-epoch hook: history for this epoch is now COMPLETE (epoch, train_*,
            # contribution ratios, anomaly-type scores, timings all appended). The
            # caller saves the checkpoint here so the persisted history is internally
            # consistent and a later resume produces equal-length per-epoch arrays
            # (2026-05-30 root-cause fix for the score-contribution off-by-one).
            if post_epoch_callback is not None:
                post_epoch_callback(epoch, self.history)

        return self.history
