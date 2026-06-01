"""
Phase 2 Best Model Re-run
=========================

Re-runs the best model (083_w500_p5_td2) with updated evaluator
to collect new metrics: F1_T, PA%K precision/recall.

Settings copied from best_config.json of experiment 083.
"""

PHASE_NAME = "phase2"
PHASE_DESCRIPTION = "Re-run best model with updated evaluator for F1_T, PA%K metrics"

# Settings from 083_w500_p5_td2_mask_after_adaptive_all/best_config.json
BASE_CONFIG = {
    'seq_length': 500,
    'num_features': 8,
    'num_train_samples': 10000,
    'num_test_samples': 2500,
    'train_anomaly_ratio': 0.05,
    'test_anomaly_ratio': 0.25,
    'use_sliding_window_dataset': True,
    'sliding_window_total_length': 275000,
    'sliding_window_stride': 11,
    'sliding_window_test_stride': 1,
    'sliding_window_train_ratio': 0.8,
    'anomaly_interval_scale': 0.75,
    'test_ratio_pure_normal': 0.65,
    'test_ratio_disturbing_normal': 0.15,
    'test_ratio_anomaly': 0.25,
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
    'force_mask_anomaly': True,
    'random_seed': 42,
    'device': 'cuda',
}

EXPERIMENTS = [
    {
        'name': '000_best_model',
        **BASE_CONFIG,
    },
]

# Only use adaptive scoring (as in original experiment)
SCORING_MODES = ['adaptive']

# Only mask_after_encoder=True (as in original experiment)
MASK_SETTINGS = [True]
