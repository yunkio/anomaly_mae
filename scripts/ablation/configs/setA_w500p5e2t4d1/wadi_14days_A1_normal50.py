"""Set A: WaDi 14days + A1 with 50% label noise"""

DATASET_TYPE = 'wadi_14days_A1'
PHASE_NAME = "wadi_14days_A1_normal50"
PHASE_DESCRIPTION = "WaDi 14days + A1 with 50% label noise"

LABEL_NOISE_RATIO = 0.5

BASE_CONFIG = {
    'seq_length': 500,
    'patch_size': 5,
    'num_patches': 100,
    'd_model': 128,
    'nhead': 8,
    'dim_feedforward': 512,
    'num_encoder_layers': 2,
    'num_teacher_decoder_layers': 4,
    'num_student_decoder_layers': 1,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'batch_size': 256,
    'eval_interval': 5,
    'cnn_kernel_size': 3,
    'patchify_mode': 'patch_cnn',
    'mask_after_encoder': True,
    'anomaly_score_mode': 'adaptive',
    'sliding_window_stride': 11,
    'sliding_window_test_stride': 1,
}

EXPERIMENTS = [
    {'name': 'default', 'config': {}},
]

SCORING_MODES = ['adaptive']
MASK_SETTINGS = [True]
