#!/bin/bash
# Finish the two TEP Table-3 protocol tracks in dependency order.
#
# The optional first argument is the PID of an already-running historical
# 30/15 w/o-GRL launcher.  That process is observed only; it is never signaled.
set -euo pipefail

cd /home/ykio/notebooks/claude
PY=/home/ykio/anaconda3/envs/dc_vis/bin/python
HIST_LOG=temp/baseline_experiment_run/tep_table3_nogrl_multiseed.log
PIPE_LOG=temp/baseline_experiment_run/tep_completion_pipeline.log
HIST_PID=${1:-}

mkdir -p "$(dirname "$PIPE_LOG")"
exec > >(tee -a "$PIPE_LOG") 2>&1

echo "===== TEP COMPLETION PIPELINE START $(date '+%F %T') ====="

if [ -n "$HIST_PID" ] && [ -r "/proc/$HIST_PID/stat" ]; then
  HIST_START=$(awk '{print $22}' "/proc/$HIST_PID/stat")
  echo "[$(date '+%F %T')] observing existing historical launcher pid=$HIST_PID"
  while [ -r "/proc/$HIST_PID/stat" ]; do
    CURRENT_START=$(awk '{print $22}' "/proc/$HIST_PID/stat" 2>/dev/null || true)
    [ "$CURRENT_START" = "$HIST_START" ] || break
    sleep 30
  done
fi

if ! grep -q 'TEP TABLE3 w/o-GRL 30/15 COMPLETE' "$HIST_LOG"; then
  echo "ERROR: historical launcher ended without its completion marker"
  exit 1
fi

echo "[$(date '+%F %T')] aggregate historical 30/15 VUS axes"
"$PY" scripts/TEP/build_vus_seed_axes.py --workers 16 \
  --selection historical-pak --boundary-mode legacy_concat
"$PY" comparison/build_results_md.py

echo "[$(date '+%F %T')] start paper-horizon 10/5 learned conditions"
bash scripts/run_tep_table3_10_5.sh

echo "[$(date '+%F %T')] aggregate 10/5 VUS axes with run-boundary reset"
"$PY" scripts/TEP/build_vus_seed_axes.py --workers 16 \
  --base-root results/experiments/TEP_table3_win100_ep10_warm5 \
  --selection final --boundary-mode run_reset
"$PY" comparison/build_results_tep_10_5.py

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("results/experiments/TEP_table3_win100_ep10_warm5")
for name, axis in (("table3_vus_fixed_seed.json", "fixed_model_seed"),
                   ("table3_vus_data_seed.json", "data_and_model_seed")):
    value = json.loads((root / name).read_text(encoding="utf-8"))
    assert value["axis"] == axis
    assert value["selection"] == "fixed final epoch 10; no test-side epoch selection"
    assert value["run_boundary_handling"] == "run_reset"
    assert len(value["runs"]) == 5
report = Path("comparison/results/experiments/results_tep_10_5.md")
assert report.exists() and report.stat().st_size > 10_000
print("FINAL VALIDATION: 10/5 axes and report complete")
PY

echo "===== TEP COMPLETION PIPELINE COMPLETE $(date '+%F %T') ====="
