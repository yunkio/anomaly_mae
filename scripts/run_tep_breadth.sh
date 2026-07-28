#!/bin/bash
# TEP Label-BREADTH sweep (2026-06-24): TRAIN DATA FIXED (3 families present), vary #labeled families
# (k=0,1,2; k=3 == LOFO-A, reused). Tests "more labeled TYPES -> better unseen" free of within-type
# partial-label confound. Same config as phase2/LOFO (official, win100, ep30, warmup15, seed42, weights OFF).
set -u
cd /home/ykio/notebooks/claude
PY=/home/ykio/anaconda3/envs/dc_vis/bin/python
RB="scripts/run_base_experiments.py"
ROOT=results/experiments/TEP_phase2_win100_ep30
LOG=/tmp/tep_breadth.log
OV="official=True batch_size=768 official_keep_checkpoints=False random_seed=42 num_epochs=30 teacher_only_warmup_epochs=15 seq_length=100 patch_size=5 num_patches=20"

# held-out order: unk (pilot) -> ds -> rand -> step. Each: k0, k1(x3), k2(x3) = 7 runs.
UNK="TEP_typegen_breadth_unk_k0 TEP_typegen_breadth_unk_step TEP_typegen_breadth_unk_rand TEP_typegen_breadth_unk_ds TEP_typegen_breadth_unk_step-rand TEP_typegen_breadth_unk_step-ds TEP_typegen_breadth_unk_rand-ds"
DS="TEP_typegen_breadth_ds_k0 TEP_typegen_breadth_ds_step TEP_typegen_breadth_ds_rand TEP_typegen_breadth_ds_unk TEP_typegen_breadth_ds_step-rand TEP_typegen_breadth_ds_step-unk TEP_typegen_breadth_ds_rand-unk"
RAND="TEP_typegen_breadth_rand_k0 TEP_typegen_breadth_rand_step TEP_typegen_breadth_rand_ds TEP_typegen_breadth_rand_unk TEP_typegen_breadth_rand_step-ds TEP_typegen_breadth_rand_step-unk TEP_typegen_breadth_rand_ds-unk"
STEP="TEP_typegen_breadth_step_rand TEP_typegen_breadth_step_ds TEP_typegen_breadth_step_unk TEP_typegen_breadth_step_k0 TEP_typegen_breadth_step_rand-ds TEP_typegen_breadth_step_rand-unk TEP_typegen_breadth_step_ds-unk"

run_group() {  # $1=label $2=datasets
  echo "########## [$(date '+%F %T')] START breadth-$1 ##########" | tee -a "$LOG"
  $PY "$RB" --set A --dataset $2 --output-base "$ROOT/breadth" --config-override $OV >> "$LOG" 2>&1
  echo "########## [$(date '+%F %T')] END breadth-$1 (exit $?) ##########" | tee -a "$LOG"
}

echo "===== BREADTH START $(date '+%F %T') =====" | tee -a "$LOG"
run_group "unk"  "$UNK"
run_group "ds"   "$DS"
run_group "rand" "$RAND"
run_group "step" "$STEP"
echo "===== BREADTH DONE $(date '+%F %T') =====" | tee -a "$LOG"
