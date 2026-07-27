#!/usr/bin/env bash
#SBATCH --job-name=coe-gpt-gemini-30x10
#SBATCH --account=fanglab
#SBATCH --partition=fanglab
#SBATCH --array=0-29%10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/hpc/group/fanglab/xx102/Chain-of-Experts/slurm_logs/%x_%A_%a.out
#SBATCH --error=/hpc/group/fanglab/xx102/Chain-of-Experts/slurm_logs/%x_%A_%a.err

set -euo pipefail

repo=/hpc/group/fanglab/xx102/Chain-of-Experts
global_task_id=$SLURM_ARRAY_TASK_ID

# Interleave providers so task IDs 0-9 contain exactly five GPT and five
# Gemini experiments. Divide by two to map the 30 global tasks onto each
# provider's 15-task (three benchmarks × five runs) logical matrix.
if (( global_task_id % 2 == 0 )); then
  export MODEL_FAMILY=gpt
else
  export MODEL_FAMILY=gemini
fi
export SLURM_ARRAY_TASK_ID=$((global_task_id / 2))

echo "global_task_id=$global_task_id"
echo "model_family=$MODEL_FAMILY"
echo "provider_logical_task_id=$SLURM_ARRAY_TASK_ID"

exec bash "$repo/submit_gpt_gemini_3bench_5x.sbatch"
