#!/bin/bash
# Resume interrupted experiments
# AT retrain: Q4/WaDi/A1, Q4/WaDi/A2 (2 remaining)
# Q3/Q4 neural: all 24 (mlp, mlpmixer, transformer × 4 datasets × 2 queues)

PYTHON=/home/ykio/anaconda3/envs/dc_vis/bin/python
RUNNER=comparison/run_baseline.py

TOTAL=0
SUCCESS=0
FAILED=0

echo "============================================"
echo "  Resume Remaining Experiments (26 total)"
echo "  Start: $(date)"
echo "============================================"

run_one() {
    local EXP="$1" MODEL="$2" BASE="$3" NORM="$4" LABEL="$5"
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "--- [$TOTAL/26] $LABEL ---"
    echo "  Start: $(date)"

    $PYTHON $RUNNER \
        --experiment "$EXP" \
        --model "$MODEL" \
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

# ---- AT retrain: Q4 WaDi remaining ----
Q4="/home/ykio/notebooks/claude/comparison/results/experiments/4_20260313_020446_baseline_zscore_normalonly"
run_one "wadi_14days_A1_normalonly" "anomaly_transformer" "$Q4" "zscore" "Q4/WaDi_A1/AT"
run_one "wadi_14days_A2_normalonly" "anomaly_transformer" "$Q4" "zscore" "Q4/WaDi_A2/AT"

# ---- Q3/Q4 neural: mlp, mlpmixer, transformer ----
Q3="/home/ykio/notebooks/claude/comparison/results/experiments/3_20260312_203923_baseline_minmax_normalonly"
MODELS="mlp mlpmixer transformer"

for EXP in simulation_sim_normalonly swat_a1a2_normalonly wadi_14days_A1_normalonly wadi_14days_A2_normalonly; do
    for M in $MODELS; do
        run_one "$EXP" "$M" "$Q3" "minmax" "Q3/$EXP/$M"
    done
done

for EXP in simulation_sim_normalonly swat_a1a2_normalonly wadi_14days_A1_normalonly wadi_14days_A2_normalonly; do
    for M in $MODELS; do
        run_one "$EXP" "$M" "$Q4" "zscore" "Q4/$EXP/$M"
    done
done

echo ""
echo "============================================"
echo "  Resume Complete"
echo "  End: $(date)"
echo "  Total: $TOTAL, Success: $SUCCESS, Failed: $FAILED"
echo "============================================"
