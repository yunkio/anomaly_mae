#!/bin/bash
# TEP Type-Gen Phase 2 — extra-seed runner for the 5-seed Table 4 (2026-07-22).
# Per seed: Phase2-A + Phase2-B only (B0 excluded — Table 4 밖; D는 A의 teacher_recon
# 파생으로 pak_fill이 처리) x the same 4 folds as run_tep_phase2_pipeline.sh, into a
# SEED-SUFFIXED root (results/experiments/TEP_phase2_win100_ep30_s{seed}) — the
# existing seed-42 root is never touched. After both groups finish, pak_fill.py
# (--root) and build_table4.py (--seed) run automatically, producing
# results/experiments/TEP_phase2_win100_ep30/table4_data_s{seed}.json
# (results_baseline.md generator convention).
#
# Config: identical to run_tep_phase2_pipeline.sh except random_seed.
# Resumable: run_base_experiments.py skips any dataset whose results dir already
# has experiment_metadata.json — re-invoking this script resumes where it stopped.
#
# Usage:
#   bash scripts/run_tep_seeds.sh 40          # one seed
#   bash scripts/run_tep_seeds.sh             # all remaining seeds: 40 41 43 44 (sequential)
#
# nohup (run only AFTER the current GPU chain finishes — 32 GPU runs total):
#   nohup bash scripts/run_tep_seeds.sh    > /tmp/tep_seeds_all.log 2>&1 &
#   nohup bash scripts/run_tep_seeds.sh 40 > /tmp/tep_seeds_s40.log 2>&1 &
set -u
cd /home/ykio/notebooks/claude
PY=/home/ykio/anaconda3/envs/dc_vis/bin/python
RB="scripts/run_base_experiments.py"
BASE_ROOT=results/experiments/TEP_phase2_win100_ep30
DSA="TEP_typegen_fstep TEP_typegen_frand TEP_typegen_fds TEP_typegen_funk"

run_group() {  # $1=label $2=root $3=outsub $4=datasets $5=extra_override $6=log $7=ov
  echo "########## [$(date '+%F %T')] START $1 ##########" | tee -a "$6"
  $PY "$RB" --set A --dataset $4 --output-base "$2/$3" --config-override $7 $5 >> "$6" 2>&1
  local rc=$?
  echo "########## [$(date '+%F %T')] END $1 (exit $rc) ##########" | tee -a "$6"
  return $rc
}

run_seed() {  # $1=seed
  local seed=$1
  local ROOT="${BASE_ROOT}_s${seed}"
  local LOG=/tmp/tep_phase2_s${seed}.log
  local OV="official=True batch_size=768 official_keep_checkpoints=False random_seed=${seed} num_epochs=30 teacher_only_warmup_epochs=15 seq_length=100 patch_size=5 num_patches=20"
  if [ "$seed" = "42" ]; then
    echo "REFUSE: seed 42 is the canonical run in $BASE_ROOT — not re-running." | tee -a "$LOG"
    return 1
  fi
  echo "===== SEED $seed START $(date '+%F %T') (root: $ROOT) =====" | tee -a "$LOG"
  run_group "s${seed}-Phase2-A" "$ROOT" phase2_A "$DSA" ""                        "$LOG" "$OV"
  run_group "s${seed}-Phase2-B" "$ROOT" phase2_B "$DSA" "blind_train_labels=True" "$LOG" "$OV"
  echo "----- [$(date '+%F %T')] aggregate: pak_fill --root $ROOT -----" | tee -a "$LOG"
  $PY scripts/TEP/pak_fill.py --root "$ROOT" >> "$LOG" 2>&1
  echo "----- [$(date '+%F %T')] aggregate: build_table4 --seed $seed -----" | tee -a "$LOG"
  $PY scripts/TEP/build_table4.py --seed "$seed" >> "$LOG" 2>&1
  echo "===== SEED $seed DONE $(date '+%F %T') =====" | tee -a "$LOG"
}

if [ $# -ge 1 ]; then
  SEEDS="$@"
else
  SEEDS="40 41 43 44"
fi
for s in $SEEDS; do
  run_seed "$s"
done
echo "ALL SEEDS DONE ($SEEDS)"
