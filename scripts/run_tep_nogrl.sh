#!/bin/bash
# TEP 'MAE w/o GRL' condition (2026-06-24): condition A (LASAD) but GRL DISABLED.
# use_grl=False removes the GRL adversarial push; anomaly_loss_weight=0.0 keeps the student's
# loss on labeled-anomaly points at ZERO (so the student learns ONLY normal reconstruction and
# simply ignores labeled anomalies). Everything else (teacher, encoder, force_mask_anomaly, recon,
# scoring, win100/ep30/warmup15/seed42) IDENTICAL to A. Isolates the GRL's contribution.
# Runs the 4 type-disjoint folds (one Table-4 'MAE w/o GRL' row). WAITS for the breadth queue first.
set -u
cd /home/ykio/notebooks/claude
PY=/home/ykio/anaconda3/envs/dc_vis/bin/python
RB="scripts/run_base_experiments.py"
ROOT=results/experiments/TEP_phase2_win100_ep30
LOG=/tmp/tep_nogrl.log
OV="official=True batch_size=768 official_keep_checkpoints=False random_seed=42 num_epochs=30 teacher_only_warmup_epochs=15 seq_length=100 patch_size=5 num_patches=20"
NOGRL_OV="use_grl=False anomaly_loss_weight=0.0"
DSA="TEP_typegen_fstep TEP_typegen_frand TEP_typegen_fds TEP_typegen_funk"

echo "[nogrl $(date '+%F %T')] breadth 큐 종료 대기..." | tee -a "$LOG"
while pgrep -f "run_tep_breadth.sh" >/dev/null 2>&1; do sleep 120; done
echo "[nogrl $(date '+%F %T')] breadth 종료 감지. BREADTH DONE marker: $(grep -c 'BREADTH DONE' /tmp/tep_breadth.log 2>/dev/null)" | tee -a "$LOG"

echo "########## [$(date '+%F %T')] START MAE-w/o-GRL (4 folds) ##########" | tee -a "$LOG"
$PY "$RB" --set A --dataset $DSA --output-base "$ROOT/phase2_nogrl" --config-override $OV $NOGRL_OV >> "$LOG" 2>&1
echo "########## [$(date '+%F %T')] END MAE-w/o-GRL (exit $?) ##########" | tee -a "$LOG"
echo "===== NOGRL DONE $(date '+%F %T') =====" | tee -a "$LOG"
