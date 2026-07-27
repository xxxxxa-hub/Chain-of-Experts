#!/bin/bash
# Run multiple experiments sequentially with a timeout per experiment.
# If an experiment exceeds the timeout, it is killed and the next one starts.
# Partial results are preserved since the Python script writes incrementally.

#SBATCH --job-name=coe_multi
#SBATCH --output=logs/coe_%A_%a.out
#SBATCH --error=logs/coe_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -o pipefail

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================
# CONFIGURATION - Modify these arrays as needed
# ============================================

TIMEOUT="2h"  # Per-experiment timeout (e.g., 30m, 2h, 7200)

# Datasets to run
DATASETS=(
    # "BWOR"
    # "ComplexOR"
    # "IndustryOR"
    # "ComplexLP"
    "car_side_impact"
    # "EasyLP"
    # "LPWP"
    # "NL4OPT"
    # "NLP4LP"
)

# Models to test
MODELS=(
    # "gpt-4.1"
    # "o4-mini"
    # "gpt-5"
    # "o3"
    # "gpt-5.2"
    "google/gemini-2.5-flash"
    # "qwen/qwen3-30b-a3b-thinking-2507"
    # "google/gemini-3-flash-preview"
    # "gpt-3.5-turbo"
    # "gpt-4-turbo"
    # "gpt-4o"
    # "gpt-4o-mini"
    # "o1-mini"
    # "o3-mini"
)

# Algorithms to test
ALGORITHMS=(
    "coe"
    # "standard"
    # "cot"
    # "php"
)

# Other parameters
PROBLEM_PATTERN='^(?!.*infeasible).*$'  # Regex pattern to match problems (use ".*" for all)
MAX_COLLABORATE_NUMS=3
MAX_TRIALS=3
ENABLE_REFLECTION=true
LOG_DIR="log"

# Resume mode: set to an existing log directory path to skip already completed problems.
# Leave empty ("") to run all from scratch.
# Example: RESUME_DIR="log/ComplexLP_google/gemini-3-flash-preview"
RESUME_DIR=""

# ============================================
# EXECUTION
# ============================================

cd /hpc/group/fanglab/xx102/Chain-of-Experts

echo "=========================================="
echo "Chain-of-Experts Multi-Dataset Multi-Model Runner"
echo "=========================================="
echo "Datasets: ${DATASETS[*]}"
echo "Models: ${MODELS[*]}"
echo "Algorithms: ${ALGORITHMS[*]}"
echo "Problem pattern: $PROBLEM_PATTERN"
echo "Timeout per experiment: $TIMEOUT"
echo "=========================================="

# Track results
RESULTS_FILE="experiment_results_$(date +%Y%m%d_%H%M%S).txt"
echo "Experiment Results - $(date)" > "$RESULTS_FILE"
echo "==========================================" >> "$RESULTS_FILE"

# Count experiments
total=0
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for ALGORITHM in "${ALGORITHMS[@]}"; do
            ((total++))
        done
    done
done

passed=0
failed=0
timed_out=0
idx=0

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for ALGORITHM in "${ALGORITHMS[@]}"; do
            ((idx++))

            echo ""
            echo "=========================================="
            echo "  Experiment ${idx}/${total}"
            echo "  Dataset=$DATASET, Model=$MODEL, Algorithm=$ALGORITHM"
            echo "  Timeout: ${TIMEOUT}"
            echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "=========================================="

            # Build the command arguments
            CMD_ARGS="--dataset $DATASET --problem '$PROBLEM_PATTERN' --algorithm $ALGORITHM --model $MODEL --max_collaborate_nums $MAX_COLLABORATE_NUMS --max_trials $MAX_TRIALS --log_dir $LOG_DIR"

            # Add reflection flag if enabled
            if [ "$ENABLE_REFLECTION" = true ]; then
                CMD_ARGS="$CMD_ARGS --enable_reflection"
            fi

            # Add resume directory if set
            if [ -n "$RESUME_DIR" ]; then
                CMD_ARGS="$CMD_ARGS --resume_dir $RESUME_DIR"
            fi

            echo "Command: python run_exp.py $CMD_ARGS"

            START_TIME=$(date +%s)

            timeout --signal=SIGTERM --kill-after=60s "${TIMEOUT}" \
                bash -c "python run_exp.py $CMD_ARGS"

            exit_code=$?

            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))

            if [ $exit_code -eq 0 ]; then
                STATUS="SUCCESS"
                echo "[OK]      Experiment ${idx} completed successfully in ${DURATION}s."
                ((passed++))
            elif [ $exit_code -eq 124 ]; then
                STATUS="TIMEOUT (killed after ${TIMEOUT})"
                echo "[TIMEOUT] Experiment ${idx} killed after ${TIMEOUT} (${DURATION}s). Partial results saved."
                ((timed_out++))
            else
                STATUS="FAILED (exit code: $exit_code)"
                echo "[ERROR]   Experiment ${idx} failed with exit code ${exit_code} in ${DURATION}s."
                ((failed++))
            fi

            echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "Dataset=$DATASET, Model=$MODEL, Algorithm=$ALGORITHM, Duration=${DURATION}s, Status=$STATUS" >> "$RESULTS_FILE"
        done
    done
done

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  All experiments done"
echo "  Passed:    ${passed}/${total}"
echo "  Timed out: ${timed_out}/${total}"
echo "  Failed:    ${failed}/${total}"
echo "  Results saved to: $RESULTS_FILE"
echo "=========================================="
cat "$RESULTS_FILE"
