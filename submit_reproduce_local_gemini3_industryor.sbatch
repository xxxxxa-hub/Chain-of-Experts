#!/usr/bin/env bash
#SBATCH --job-name=coe-local-gemini3-industryor
#SBATCH --account=fanglab
#SBATCH --partition=fanglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/hpc/group/fanglab/xx102/Chain-of-Experts/slurm_logs/%x_%j.out
#SBATCH --error=/hpc/group/fanglab/xx102/Chain-of-Experts/slurm_logs/%x_%j.err

set -euo pipefail

repo=/hpc/group/fanglab/xx102/Chain-of-Experts
cd "$repo"
mkdir -p slurm_logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate old_coe

python run_exp_metrics_litellm.py \
  --dataset IndustryOR \
  --problem '.*' \
  --algorithm coe \
  --model google/gemini-3-flash-preview \
  --enable_reflection \
  --max_collaborate_nums 3 \
  --max_trials 3 \
  --max_problems 100 \
  --num_processes 50 \
  --log_dir log
