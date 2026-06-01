"""
Simulation Normal 50 Experiment
===============================

Best model (083_w500_p5_td2) trained with 50% label noise:
- 50% of training anomaly regions are relabeled as normal
- Test set uses original clean labels
- force_mask_anomaly=True (intentionally uses corrupted labels)
"""

PHASE_NAME = "simulation_normal_50"
PHASE_DESCRIPTION = "Best model with 50% anomaly label noise in training"

# Enable label noise in training
LABEL_NOISE_RATIO = 0.5  # Relabel 50% of anomaly regions as normal

# Settings from 083_w500_p5_td2 (best model)
BASE_CONFIG = {
    'seq_length': 500,
    'num_features': 8,
    'use_sliding_window_dataset': True,
    'sliding_window_total_length': 275000,
    'sliding_window_stride': 11,
    'sliding_window_test_stride': 1,
    'sliding_window_train_ratio': 0.8,
    'anomaly_interval_scale': 0.75,
    'd_model': 128,
    'nhead': 8,
    'num_encoder_layers': 1,
    'num_teacher_decoder_layers': 2,
    'num_student_decoder_layers': 2,
    'num_shared_decoder_layers': 0,
    'dim_feedforward': 512,
    'dropout': 0.15,
    'masking_ratio': 0.15,
    'num_patches': 100,
    'patch_size': 5,
    'patchify_mode': 'patch_cnn',
    'mask_after_encoder': True,
    'shared_mask_token': False,
    'cnn_channels': [64, 128],
    'margin': 0.5,
    'lambda_disc': 2.0,
    'margin_type': 'dynamic',
    'dynamic_margin_k': 2.0,
    'patch_level_loss': True,
    'anomaly_loss_weight': 2.0,
    'anomaly_score_mode': 'default',
    'batch_size': 256,
    'num_epochs': 50,
    'learning_rate': 0.002,
    'weight_decay': 1e-05,
    'warmup_epochs': 10,
    'teacher_only_warmup_epochs': 3,
    'use_amp': True,
    'point_aggregation_method': 'voting',
    'use_discrepancy_loss': True,
    'use_teacher': True,
    'use_student': True,
    'use_masking': True,
    'force_mask_anomaly': True,  # Intentionally uses corrupted labels
    'random_seed': 42,
    'device': 'cuda',
}

EXPERIMENTS = [
    {
        'name': '000_best_model_noisy50',
        **BASE_CONFIG,
    },
]

SCORING_MODES = ['adaptive']
MASK_SETTINGS = [True]  # mask_after_encoder=True
