#!/usr/bin/env bash
#SBATCH --job-name=coe-gemini3-3bench-5x
#SBATCH --account=fanglab
#SBATCH --partition=fanglab
#SBATCH --array=0-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/hpc/group/fanglab/xx102/Chain-of-Experts/slurm_logs/%x_%A_%a.out
#SBATCH --error=/hpc/group/fanglab/xx102/Chain-of-Experts/slurm_logs/%x_%A_%a.err

set -euo pipefail

repo=/hpc/group/fanglab/xx102/Chain-of-Experts
model=google/gemini-3-flash-preview
matrix_root="$repo/log/reruns_matrix/google_gemini-3-flash-preview/coe_reflection3_conductor10"
completed_industryor_run1="$repo/log/reruns/IndustryOR/google_gemini-3-flash-preview/coe/run_20260727T213339_344985Z_1204587_345020554"

benchmarks=(ComplexLP IndustryOR BWOR)
problem_counts=(211 100 82)

benchmark_index=$((SLURM_ARRAY_TASK_ID / 5))
run_index=$((SLURM_ARRAY_TASK_ID % 5 + 1))
benchmark=${benchmarks[$benchmark_index]}
problem_count=${problem_counts[$benchmark_index]}
run_name=$(printf 'run_%02d' "$run_index")
run_dir="$matrix_root/$benchmark/$run_name"

cd "$repo"
mkdir -p slurm_logs "$(dirname "$run_dir")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate old_coe

# Prevent numerical libraries inside generated programs from multiplying the
# eight allocated CPUs by the 50 API-oriented Python workers.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

run_is_complete() {
  local results_file=$1
  local expected_count=$2
  python - "$results_file" "$expected_count" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)

records = {}
try:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[str(record["problem"])] = record
except (OSError, KeyError, json.JSONDecodeError):
    raise SystemExit(1)

valid_results = {"ACCEPT", "WRONG_ANSWER", "COMPILE_ERROR", "RUNTIME_ERROR"}
complete = (
    len(records) == expected
    and all(record.get("result") in valid_results for record in records.values())
)
raise SystemExit(0 if complete else 1)
PY
}

# The already-completed IndustryOR experiment is logical run 1. Link it into
# the matrix instead of spending API calls to reproduce it.
if [[ "$benchmark" == IndustryOR && "$run_index" -eq 1 ]]; then
  if run_is_complete "$completed_industryor_run1/results.jsonl" "$problem_count"; then
    if [[ ! -e "$run_dir" && ! -L "$run_dir" ]]; then
      ln -s "$completed_industryor_run1" "$run_dir"
    fi
    echo "Reused completed IndustryOR run 1: $completed_industryor_run1"
    exit 0
  fi
  echo "Expected reusable run is incomplete: $completed_industryor_run1" >&2
  exit 1
fi

if run_is_complete "$run_dir/results.jsonl" "$problem_count"; then
  echo "Skipping complete run: $run_dir"
  exit 0
fi

runner_args=(
  --dataset "$benchmark"
  --problem '.*'
  --algorithm coe
  --model "$model"
  --enable_reflection
  --max_collaborate_nums 3
  --max_trials 3
  --max_problems "$problem_count"
  --num_processes 50
  --run_index "$run_index"
)

if [[ -d "$run_dir" ]]; then
  echo "Resuming incomplete run: $run_dir"
  runner_args+=(--resume_dir "$run_dir")
else
  echo "Starting new run: $run_dir"
  runner_args+=(--run_dir "$run_dir")
fi

python run_exp_metrics_litellm.py "${runner_args[@]}"

if ! run_is_complete "$run_dir/results.jsonl" "$problem_count"; then
  echo "Run exited without all $problem_count problem records: $run_dir" >&2
  exit 1
fi

echo "Completed: $benchmark independent run $run_index"
