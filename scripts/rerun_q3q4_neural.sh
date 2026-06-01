#!/bin/bash
# Re-run Q3/Q4 neural baselines (mlp, mlpmixer, transformer)
# After unifying segment-aware into run_dl_baseline_with_epoch_eval
# These now get proper per-epoch evaluation instead of single final eval

PYTHON=/home/ykio/anaconda3/envs/dc_vis/bin/python
RUNNER=comparison/run_baseline.py
MODELS="mlp mlpmixer transformer"

TOTAL=0
SUCCESS=0
FAILED=0

echo "============================================"
echo "  Q3/Q4 Neural Re-run (24 experiments)"
echo "  Start: $(date)"
echo "============================================"

run_one() {
    local EXP="$1" MODEL="$2" BASE="$3" NORM="$4" LABEL="$5"
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "--- [$TOTAL/24] $LABEL ---"
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

# Q3: minmax normalonly
Q3="/home/ykio/notebooks/claude/comparison/results/experiments/3_20260312_203923_baseline_minmax_normalonly"
for EXP in simulation_sim_normalonly swat_a1a2_normalonly wadi_14days_A1_normalonly wadi_14days_A2_normalonly; do
    for M in $MODELS; do
        run_one "$EXP" "$M" "$Q3" "minmax" "Q3/$EXP/$M"
    done
done

# Q4: zscore normalonly
Q4="/home/ykio/notebooks/claude/comparison/results/experiments/4_20260313_020446_baseline_zscore_normalonly"
for EXP in simulation_sim_normalonly swat_a1a2_normalonly wadi_14days_A1_normalonly wadi_14days_A2_normalonly; do
    for M in $MODELS; do
        run_one "$EXP" "$M" "$Q4" "zscore" "Q4/$EXP/$M"
    done
done

echo ""
echo "============================================"
echo "  Q3/Q4 Neural Re-run Complete"
echo "  End: $(date)"
echo "  Total: $TOTAL, Success: $SUCCESS, Failed: $FAILED"
echo "============================================"
