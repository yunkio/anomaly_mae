#!/bin/bash
# Run SMD baseline experiments for Q3 (minmax_normalonly)
# 28 machines × 2 parities × 15 models = 840 experiments
# Uses load_smd_block_split() with normalonly variant

PYTHON=/home/ykio/anaconda3/envs/dc_vis/bin/python
RUNNER=comparison/run_baseline.py
Q3="/home/ykio/notebooks/claude/comparison/results/experiments/3_20260312_203923_baseline_minmax_normalonly"

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

TOTAL_RUNS=$((${#MACHINES[@]} * 2))  # machines × parities
CURRENT=0
SUCCESS=0
FAILED=0

echo "============================================"
echo "  SMD Q3 Baseline (${#MACHINES[@]} machines × 2 parities × 15 models)"
echo "  Total: $TOTAL_RUNS machine/parity runs"
echo "  Output: $Q3/SMD/"
echo "  Start: $(date)"
echo "============================================"

for MACHINE in "${MACHINES[@]}"; do
    for PARITY in 0 1; do
        CURRENT=$((CURRENT + 1))
        EXP_KEY="smd_${MACHINE}_p${PARITY}_normalonly"

        echo ""
        echo "--- [$CURRENT/$TOTAL_RUNS] $MACHINE/parity_$PARITY ---"
        echo "  Start: $(date)"

        $PYTHON $RUNNER \
            --experiment "$EXP_KEY" \
            --model all \
            --output-base "$Q3" \
            --eval-interval 1 \
            --normalize-mode minmax \
            --neural-epochs 10 \
            --sota-epochs 10 \
            --at-epochs 10 \
            --force

        if [ $? -eq 0 ]; then
            echo "  Result: SUCCESS"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "  Result: FAILED (exit code $?)"
            FAILED=$((FAILED + 1))
        fi
        echo "  End: $(date)"
    done
done

echo ""
echo "============================================"
echo "  SMD Q3 Complete"
echo "  End: $(date)"
echo "  Total: $CURRENT, Success: $SUCCESS, Failed: $FAILED"
echo "============================================"
