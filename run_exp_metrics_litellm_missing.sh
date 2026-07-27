#!/bin/bash
# Re-run missing IndustryOR questions (0, 1, 9) with litellm-based metrics

set -o pipefail

cd /hpc/group/fanglab/xx102/Chain-of-Experts

source $(conda info --base)/etc/profile.d/conda.sh
conda activate old_coe

echo "Re-running IndustryOR missing questions (0, 1, 9)..."
echo "Timestamp: $(date)"

python run_exp_metrics_litellm.py \
    --dataset IndustryOR \
    --problem '^(0|1|9)$' \
    --algorithm coe \
    --model google/gemini-2.5-flash \
    --max_collaborate_nums 3 \
    --max_trials 3 \
    --enable_reflection \
    --log_dir log \
    --max_problems 10 \
    --num_processes 3

echo "Done! Timestamp: $(date)"
