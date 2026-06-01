#!/bin/bash
# Group F RESUME: Exp 171 remaining SMD + Exp 172 full
set -e
cd /home/ykio/notebooks/claude

BASE_OPTS="num_encoder_layers=4 num_teacher_decoder_layers=3 num_student_decoder_layers=1 dynamic_margin_k=6 normalize_mode=minmax use_feature_matching=True eval_disc_weight=1.0 eval_fm_weight=1.0 epoch_offset=True num_epochs=200 teacher_only_warmup_epochs=100"
DATASETS="simulation SWaT_A1A2 WaDi_A1 WaDi_A2 smd_all"

LOG_DIR="/home/ykio/notebooks/claude/logs"
EXP171_DIR=$(ls -d results/experiments/171_*/ 2>/dev/null | head -1)

echo "$(date): Starting Group F RESUME (171 remaining + 172)" | tee -a "$LOG_DIR/group_f_rerun.log"

# Exp 171 RESUME: only remaining SMD machines (use --output-base to write into existing dir)
echo "$(date): === Resuming Exp 171 (GRL+unmask) remaining SMD ===" | tee -a "$LOG_DIR/group_f_rerun.log"
echo "  Using existing dir: $EXP171_DIR"
python scripts/run_base_experiments.py --set C \
  --dataset smd_machine-2-1 smd_machine-2-2 smd_machine-2-3 smd_machine-2-4 smd_machine-2-5 smd_machine-2-6 smd_machine-2-7 smd_machine-2-8 smd_machine-2-9 smd_machine-3-1 smd_machine-3-2 smd_machine-3-3 smd_machine-3-4 smd_machine-3-5 smd_machine-3-6 smd_machine-3-7 smd_machine-3-8 smd_machine-3-9 smd_machine-3-10 smd_machine-3-11 \
  --config-override $BASE_OPTS num_student_decoder_layers=2 use_grl=True grl_loss_weight=0.5 force_mask_anomaly=False \
  --output-base "$EXP171_DIR" \
  --no-wait 2>&1 | tee "$LOG_DIR/exp171_resume.log"
echo "$(date): === Exp 171 resume complete ===" | tee -a "$LOG_DIR/group_f_rerun.log"

# Exp 172: GRL+window_target (full run)
echo "$(date): === Starting Exp 172 (GRL+window_target) ===" | tee -a "$LOG_DIR/group_f_rerun.log"
python scripts/run_base_experiments.py --set C \
  --dataset $DATASETS \
  --config-override $BASE_OPTS num_student_decoder_layers=2 use_grl=True grl_loss_weight=0.5 grl_target_mode=window \
  --no-wait 2>&1 | tee "$LOG_DIR/exp172_rerun.log"
echo "$(date): === Exp 172 complete ===" | tee -a "$LOG_DIR/group_f_rerun.log"

echo "$(date): All Group F RESUME complete!" | tee -a "$LOG_DIR/group_f_rerun.log"
