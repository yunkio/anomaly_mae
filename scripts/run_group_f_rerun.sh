#!/bin/bash
# Group F RERUN: Exp 169-172 only (GRL experiments, after gradient bug fix)
# Base = Exp 150 config (e4t3d1, 200ep, FM+OD, minmax, mk6, offset=True)

set -e
cd /home/ykio/notebooks/claude

BASE_OPTS="num_encoder_layers=4 num_teacher_decoder_layers=3 num_student_decoder_layers=1 dynamic_margin_k=6 normalize_mode=minmax use_feature_matching=True eval_disc_weight=1.0 eval_fm_weight=1.0 epoch_offset=True num_epochs=200 teacher_only_warmup_epochs=100"
DATASETS="simulation SWaT_A1A2 WaDi_A1 WaDi_A2 smd_all"

LOG_DIR="/home/ykio/notebooks/claude/logs"
mkdir -p "$LOG_DIR"

echo "$(date): Starting Group F RERUN (169-172, GRL fix applied)" | tee "$LOG_DIR/group_f_rerun.log"

# Exp 169: GRL+anomaly (sd2, grl=True, w=0.5, grl_disable_anomaly_loss=False)
echo "$(date): === Starting Exp 169 (GRL+anomaly) ===" | tee -a "$LOG_DIR/group_f_rerun.log"
python scripts/run_base_experiments.py --set C \
  --dataset $DATASETS \
  --config-override $BASE_OPTS num_student_decoder_layers=2 use_grl=True grl_loss_weight=0.5 grl_disable_anomaly_loss=False \
  --no-wait 2>&1 | tee "$LOG_DIR/exp169_rerun.log"
echo "$(date): === Exp 169 complete ===" | tee -a "$LOG_DIR/group_f_rerun.log"

# Exp 170: GRL+same_dir
echo "$(date): === Starting Exp 170 (GRL+same_dir) ===" | tee -a "$LOG_DIR/group_f_rerun.log"
python scripts/run_base_experiments.py --set C \
  --dataset $DATASETS \
  --config-override $BASE_OPTS num_student_decoder_layers=2 use_grl=True grl_loss_weight=0.5 grl_disable_anomaly_loss=False anomaly_loss_direction=minimize \
  --no-wait 2>&1 | tee "$LOG_DIR/exp170_rerun.log"
echo "$(date): === Exp 170 complete ===" | tee -a "$LOG_DIR/group_f_rerun.log"

# Exp 171: GRL+unmask
echo "$(date): === Starting Exp 171 (GRL+unmask) ===" | tee -a "$LOG_DIR/group_f_rerun.log"
python scripts/run_base_experiments.py --set C \
  --dataset $DATASETS \
  --config-override $BASE_OPTS num_student_decoder_layers=2 use_grl=True grl_loss_weight=0.5 force_mask_anomaly=False \
  --no-wait 2>&1 | tee "$LOG_DIR/exp171_rerun.log"
echo "$(date): === Exp 171 complete ===" | tee -a "$LOG_DIR/group_f_rerun.log"

# Exp 172: GRL+window_target
echo "$(date): === Starting Exp 172 (GRL+window_target) ===" | tee -a "$LOG_DIR/group_f_rerun.log"
python scripts/run_base_experiments.py --set C \
  --dataset $DATASETS \
  --config-override $BASE_OPTS num_student_decoder_layers=2 use_grl=True grl_loss_weight=0.5 grl_target_mode=window \
  --no-wait 2>&1 | tee "$LOG_DIR/exp172_rerun.log"
echo "$(date): === Exp 172 complete ===" | tee -a "$LOG_DIR/group_f_rerun.log"

echo "$(date): All Group F RERUN experiments complete!" | tee -a "$LOG_DIR/group_f_rerun.log"
