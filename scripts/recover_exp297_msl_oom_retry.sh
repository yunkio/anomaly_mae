#!/usr/bin/env bash
# Recover exp297 MSL (5 entities) at d_model=768 (dynamic, intended) despite the
# per-epoch host-RAM leak that OOMs ~ep420. Strategy: run each MSL in its own
# run_base process; on OOM (SIGKILL → nonzero rc), re-run — run_base resumes from
# latest_checkpoint (resume_version>=1) with FRESH process memory, so it finishes
# the remaining epochs + finalize under the RAM ceiling. ~1-2 attempts per MSL.
set -u
cd /home/ykio/notebooks/TSMAE
source /home/ykio/anaconda3/etc/profile.d/conda.sh && conda activate dc_vis
D=$(ls -d results/experiments/297_*dyn_dmodel)
# exp297 override, batch_size 512 (per user; harmless for MSL=109 windows=1 batch)
OV=$(python3 -c "import json;o=[e['config_override'] for e in json.load(open('configs/queue_dedup_renumbered_v5.json'))['experiments'] if e['exp_num']==297][0];print(o.replace('batch_size=1024','batch_size=512'))")
echo "=== exp297 MSL OOM-retry recovery START $(date '+%H:%M:%S') | dir=$(basename $D) ==="
for m in C-1 C-2 F-7 P-11 T-13; do
  meta="$D/MSL/$m/experiment_metadata.json"
  for attempt in 1 2 3 4 5; do
    if [ -f "$meta" ]; then echo ">>> MSL/$m DONE (attempt $attempt-1)"; break; fi
    echo ">>> MSL/$m attempt $attempt $(date '+%H:%M:%S') (resume if ckpt exists)"
    python -u scripts/run_base_experiments.py --set C --no-wait --output-base "$D" \
      --dataset "MSL_simple_$m" --config-override "$OV" >> "$D/MSL/_oom_retry_$m.log" 2>&1
    rc=$?
    echo ">>> MSL/$m attempt $attempt ended rc=$rc"
    if [ -f "$meta" ]; then echo ">>> MSL/$m FINALIZED"; break; fi
    sleep 3
  done
done
echo "=== MSL recovery DONE: $(find "$D/MSL" -name experiment_metadata.json|wc -l)/5 | $(date '+%H:%M:%S') ==="
