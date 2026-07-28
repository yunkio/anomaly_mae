#!/bin/bash
# TEP Type-Gen Phase 2 → Noisy → LOFO pipeline (2026-06-23)
# Config: official=True, window=100 (p5/np20), num_epochs=30, warmup=15, batch=768,
#         official_keep_checkpoints=False (weights OFF), random_seed=42.
#         eval_interval not set → TEP default 3. MAE_SKIP_VUS auto-set by run_base.
# Order: Phase2(A/B/B0) → Noisy(labeled 80/50/25/10) → LOFO(excluded + _cont).
set -u
cd /home/ykio/notebooks/claude
PY=/home/ykio/anaconda3/envs/dc_vis/bin/python
RB="scripts/run_base_experiments.py"
ROOT=results/experiments/TEP_phase2_win100_ep30
LOG=/tmp/tep_phase2_pipeline.log
OV="official=True batch_size=768 official_keep_checkpoints=False random_seed=42 num_epochs=30 teacher_only_warmup_epochs=15 seq_length=100 patch_size=5 num_patches=20"

DSA="TEP_typegen_fstep TEP_typegen_frand TEP_typegen_fds TEP_typegen_funk"
NOISY="TEP_typegen_fstep_lab80 TEP_typegen_fstep_lab50 TEP_typegen_fstep_lab25 TEP_typegen_fstep_lab10 \
TEP_typegen_frand_lab80 TEP_typegen_frand_lab50 TEP_typegen_frand_lab25 TEP_typegen_frand_lab10 \
TEP_typegen_fds_lab80 TEP_typegen_fds_lab50 TEP_typegen_fds_lab25 TEP_typegen_fds_lab10 \
TEP_typegen_funk_lab80 TEP_typegen_funk_lab50 TEP_typegen_funk_lab25 TEP_typegen_funk_lab10"
LOFO="TEP_typegen_lofo_step TEP_typegen_lofo_rand TEP_typegen_lofo_ds TEP_typegen_lofo_unk \
TEP_typegen_lofo_step_cont TEP_typegen_lofo_rand_cont TEP_typegen_lofo_ds_cont TEP_typegen_lofo_unk_cont"

run_group() {  # $1=label $2=outsub $3=datasets $4=extra_override
  echo "########## [$(date '+%F %T')] START $1 ##########" | tee -a "$LOG"
  $PY "$RB" --set A --dataset $3 --output-base "$ROOT/$2" --config-override $OV $4 >> "$LOG" 2>&1
  local rc=$?
  echo "########## [$(date '+%F %T')] END $1 (exit $rc) ##########" | tee -a "$LOG"
}

echo "===== PIPELINE START $(date '+%F %T') =====" | tee -a "$LOG"
run_group "Phase2-A"   phase2_A  "$DSA"                  ""
run_group "Phase2-B"   phase2_B  "$DSA"                  "blind_train_labels=True"
run_group "Phase2-B0"  phase2_B0 "TEP_typegen_ffonly"    ""
run_group "Noisy"      noisy     "$NOISY"                ""
run_group "LOFO"       lofo      "$LOFO"                 ""
echo "===== PIPELINE DONE $(date '+%F %T') =====" | tee -a "$LOG"
