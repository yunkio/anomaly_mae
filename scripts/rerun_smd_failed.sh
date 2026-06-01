#!/bin/bash
# Rerun failed SMD experiments with per-model process isolation
# After CUDA errors, some SOTA models fail for specific machine/parity combos
# This script runs each model as a separate process to prevent cascade failures

PYTHON=/home/ykio/anaconda3/envs/dc_vis/bin/python
RUNNER=comparison/run_baseline.py

MODELS=(random sensor_range pca_error l2_norm nn_distance mlp mlpmixer transformer gcn_lstm anomaly_transformer tranad usad dagmm gdn omnianomaly)

TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

run_single_model() {
    local EXP_KEY="$1" MODEL="$2" QDIR="$3" NORM="$4" LABEL="$5"
    local MODEL_DIR="$QDIR/$(python -c "from comparison.experiment_configs import get_experiment_config; print(get_experiment_config('$EXP_KEY')['results_dir_name'])" 2>/dev/null)/$MODEL"

    # Skip if already completed
    if [ -f "$MODEL_DIR/epoch_metrics.json" ] && [ "$FORCE" != "1" ]; then
        return 2  # skipped
    fi

    $PYTHON $RUNNER \
        --experiment "$EXP_KEY" \
        --model "$MODEL" \
        --output-base "$QDIR" \
        --eval-interval 1 \
        --normalize-mode "$NORM" \
        --neural-epochs 10 \
        --sota-epochs 10 \
        --at-epochs 10 \
        --force

    return $?
}

echo "============================================"
echo "  SMD Rerun Failed Models (per-model isolation)"
echo "  Start: $(date)"
echo "============================================"

# Define what to rerun: QUEUE EXP_KEY QDIR NORM
# Add entries for each failed machine/parity
# Auto-detect: scan all queues for missing SOTA models

QUEUE_DIRS=(
    "Q1:/home/ykio/notebooks/claude/comparison/results/experiments/1_20260312_041500_baseline_minmax:minmax:"
    "Q2:/home/ykio/notebooks/claude/comparison/results/experiments/2_20260312_144923_baseline_zscore:zscore:"
    "Q3:/home/ykio/notebooks/claude/comparison/results/experiments/3_20260312_203923_baseline_minmax_normalonly:minmax:_normalonly"
    "Q4:/home/ykio/notebooks/claude/comparison/results/experiments/4_20260313_020446_baseline_zscore_normalonly:zscore:_normalonly"
)

MACHINES=(
    machine-1-1 machine-1-2 machine-1-3 machine-1-4
    machine-1-5 machine-1-6 machine-1-7 machine-1-8
    machine-2-1 machine-2-2 machine-2-3 machine-2-4
    machine-2-5 machine-2-6 machine-2-7 machine-2-8
    machine-2-9
    machine-3-1 machine-3-2 machine-3-3 machine-3-4
    machine-3-5 machine-3-6 machine-3-7 machine-3-8
    machine-3-9 machine-3-10 machine-3-11
)

for QINFO in "${QUEUE_DIRS[@]}"; do
    IFS=: read -r QNAME QDIR NORM SUFFIX <<< "$QINFO"

    # Check if SMD directory exists
    if [ ! -d "$QDIR/SMD" ]; then
        continue
    fi

    for MACHINE in "${MACHINES[@]}"; do
        for PARITY in 0 1; do
            EXP_KEY="smd_${MACHINE}_p${PARITY}${SUFFIX}"
            MDIR="$QDIR/SMD/$MACHINE/parity_$PARITY"

            for MODEL in "${MODELS[@]}"; do
                if [ ! -f "$MDIR/$MODEL/epoch_metrics.json" ]; then
                    TOTAL=$((TOTAL + 1))
                    echo ""
                    echo "--- [$TOTAL] $QNAME/$MACHINE/p$PARITY/$MODEL ---"
                    echo "  Start: $(date)"

                    $PYTHON $RUNNER \
                        --experiment "$EXP_KEY" \
                        --model "$MODEL" \
                        --output-base "$QDIR" \
                        --eval-interval 1 \
                        --normalize-mode "$NORM" \
                        --neural-epochs 10 \
                        --sota-epochs 10 \
                        --at-epochs 10 \
                        --force

                    if [ $? -eq 0 ]; then
                        echo "  Result: SUCCESS"
                        SUCCESS=$((SUCCESS + 1))
                    else
                        echo "  Result: FAILED"
                        FAILED=$((FAILED + 1))
                    fi
                    echo "  End: $(date)"
                fi
            done
        done
    done
done

echo ""
echo "============================================"
echo "  Rerun Complete"
echo "  End: $(date)"
echo "  Total: $TOTAL, Success: $SUCCESS, Failed: $FAILED"
echo "============================================"
