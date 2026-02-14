"""Experiment utilities for configuration and execution."""

from mae_anomaly import Config


def make_config(overrides: dict) -> Config:
    """Create Config with defaults + overrides.

    Args:
        overrides: Dictionary of config parameters to override

    Returns:
        Config object with applied overrides
    """
    config = Config()
    # Defaults (from WaDi experiments)
    config.seq_length = 500
    config.patch_size = 5
    config.num_patches = 100
    config.d_model = 128
    config.nhead = 8
    config.num_encoder_layers = 2  # enc2 (optimal from ablation)
    config.num_teacher_decoder_layers = 4  # td4 (optimal from ablation)
    config.num_student_decoder_layers = 1  # sd1 (shallow for better discrepancy)
    config.dim_feedforward = 512
    config.cnn_channels = (64, 128)
    config.mask_after_encoder = True
    config.masking_ratio = 0.15
    config.dropout = 0.15
    config.learning_rate = 2e-3
    config.batch_size = 256
    config.num_epochs = 50
    config.warmup_epochs = 10
    config.teacher_only_warmup_epochs = 3
    config.anomaly_score_mode = 'adaptive'
    config.force_mask_anomaly = True
    config.use_amp = True
    config.device = 'cuda'

    # Apply overrides
    for k, v in overrides.items():
        if k == 'name':
            continue
        if hasattr(config, k):
            setattr(config, k, v)

    return config
