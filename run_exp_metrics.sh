#!/bin/bash
# Run experiments for first 10 questions per dataset, recording all four metrics.

set -o pipefail

cd /hpc/group/fanglab/xx102/Chain-of-Experts

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate old_coe

# ============================================
# CONFIGURATION
# ============================================

TIMEOUT="2h"

DATASETS=(
    "ComplexLP"
    "IndustryOR"
    "BWOR"
)

MODELS=(
    "google/gemini-2.5-flash"
)

ALGORITHMS=(
    "coe"
)

MAX_PROBLEMS=10
MAX_COLLABORATE_NUMS=3
MAX_TRIALS=3
ENABLE_REFLECTION=true
LOG_DIR="log"
PROBLEM_PATTERN='.*'

# ============================================
# EXECUTION
# ============================================

echo "=========================================="
echo "Chain-of-Experts Four-Metric Experiment Runner"
echo "=========================================="
echo "Datasets: ${DATASETS[*]}"
echo "Models: ${MODELS[*]}"
echo "Max problems per dataset: $MAX_PROBLEMS"
echo "=========================================="

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for ALGORITHM in "${ALGORITHMS[@]}"; do
            echo ""
            echo "=========================================="
            echo "  Dataset=$DATASET, Model=$MODEL, Algorithm=$ALGORITHM"
            echo "  Max problems: $MAX_PROBLEMS"
            echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "=========================================="

            CMD_ARGS="--dataset $DATASET --problem '$PROBLEM_PATTERN' --algorithm $ALGORITHM --model $MODEL --max_collaborate_nums $MAX_COLLABORATE_NUMS --max_trials $MAX_TRIALS --log_dir $LOG_DIR --max_problems $MAX_PROBLEMS"

            if [ "$ENABLE_REFLECTION" = true ]; then
                CMD_ARGS="$CMD_ARGS --enable_reflection"
            fi

            echo "Command: python run_exp_metrics.py $CMD_ARGS"

            START_TIME=$(date +%s)

            timeout --signal=SIGTERM --kill-after=60s "${TIMEOUT}" \
                bash -c "python run_exp_metrics.py $CMD_ARGS"

            exit_code=$?

            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))

            if [ $exit_code -eq 0 ]; then
                echo "[OK]      Completed in ${DURATION}s."
            elif [ $exit_code -eq 124 ]; then
                echo "[TIMEOUT] Killed after ${TIMEOUT} (${DURATION}s)."
            else
                echo "[ERROR]   Failed with exit code ${exit_code} in ${DURATION}s."
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "  All experiments done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
