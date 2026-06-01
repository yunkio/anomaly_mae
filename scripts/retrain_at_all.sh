#!/bin/bash
# Retrain ALL Anomaly Transformer experiments (16 runs across 4 queues)
# Purpose: AT wrapper now has save()/load() — retrain to persist model weights
# Note: SWaT/A1A2_excl22 is auto-generated from swat_a1a2 (has_excl22=True), not a separate run

PYTHON=/home/ykio/anaconda3/envs/dc_vis/bin/python
RUNNER=comparison/run_baseline.py

TOTAL=0
SUCCESS=0
FAILED=0

echo "============================================"
echo "  AT Retrain — All Queues (16 experiments)"
echo "  Start: $(date)"
echo "============================================"

run_one() {
    local EXP="$1" BASE="$2" NORM="$3" LABEL="$4"
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "--- [$TOTAL/16] $LABEL ---"
    echo "  Start: $(date)"

    $PYTHON $RUNNER \
        --experiment "$EXP" \
        --model "anomaly_transformer" \
        --output-base "$BASE" \
        --eval-interval 1 \
        --normalize-mode "$NORM" \
        --force

    if [ $? -eq 0 ]; then
        echo "  Result: SUCCESS"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  Result: FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
    echo "  End: $(date)"
}

# Q1: minmax
Q1="/home/ykio/notebooks/claude/comparison/results/experiments/1_20260312_041500_baseline_minmax"
run_one "simulation"       "$Q1" "minmax" "Q1/simulation/AT"
run_one "swat_a1a2"        "$Q1" "minmax" "Q1/SWaT/AT"
run_one "wadi_14days_A1"   "$Q1" "minmax" "Q1/WaDi_A1/AT"
run_one "wadi_14days_A2"   "$Q1" "minmax" "Q1/WaDi_A2/AT"

# Q2: zscore
Q2="/home/ykio/notebooks/claude/comparison/results/experiments/2_20260312_144923_baseline_zscore"
run_one "simulation"       "$Q2" "zscore" "Q2/simulation/AT"
run_one "swat_a1a2"        "$Q2" "zscore" "Q2/SWaT/AT"
run_one "wadi_14days_A1"   "$Q2" "zscore" "Q2/WaDi_A1/AT"
run_one "wadi_14days_A2"   "$Q2" "zscore" "Q2/WaDi_A2/AT"

# Q3: minmax normalonly
Q3="/home/ykio/notebooks/claude/comparison/results/experiments/3_20260312_203923_baseline_minmax_normalonly"
run_one "simulation_sim_normalonly"    "$Q3" "minmax" "Q3/simulation_no/AT"
run_one "swat_a1a2_normalonly"         "$Q3" "minmax" "Q3/SWaT_no/AT"
run_one "wadi_14days_A1_normalonly"    "$Q3" "minmax" "Q3/WaDi_A1_no/AT"
run_one "wadi_14days_A2_normalonly"    "$Q3" "minmax" "Q3/WaDi_A2_no/AT"

# Q4: zscore normalonly
Q4="/home/ykio/notebooks/claude/comparison/results/experiments/4_20260313_020446_baseline_zscore_normalonly"
run_one "simulation_sim_normalonly"    "$Q4" "zscore" "Q4/simulation_no/AT"
run_one "swat_a1a2_normalonly"         "$Q4" "zscore" "Q4/SWaT_no/AT"
run_one "wadi_14days_A1_normalonly"    "$Q4" "zscore" "Q4/WaDi_A1_no/AT"
run_one "wadi_14days_A2_normalonly"    "$Q4" "zscore" "Q4/WaDi_A2_no/AT"

echo ""
echo "============================================"
echo "  AT Retrain Complete"
echo "  End: $(date)"
echo "  Total: $TOTAL, Success: $SUCCESS, Failed: $FAILED"
echo "============================================"
