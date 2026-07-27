#!/usr/bin/env bash
# Submit this file twice, once with MODEL_FAMILY=gpt and once with
# MODEL_FAMILY=gemini. Each provider runs one 50-worker experiment at a time;
# the two provider arrays may run concurrently.
#
#SBATCH --account=fanglab
#SBATCH --partition=fanglab
#SBATCH --array=0-14%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/hpc/group/fanglab/xx102/Chain-of-Experts/slurm_logs/%x_%A_%a.out
#SBATCH --error=/hpc/group/fanglab/xx102/Chain-of-Experts/slurm_logs/%x_%A_%a.err

set -euo pipefail

repo=/hpc/group/fanglab/xx102/Chain-of-Experts
model_family=${MODEL_FAMILY:?Submit with --export=ALL,MODEL_FAMILY=gpt or gemini}

benchmarks=(ComplexLP IndustryOR BWOR)
problem_counts=(211 100 82)
benchmark_index=$((SLURM_ARRAY_TASK_ID / 5))
run_index=$((SLURM_ARRAY_TASK_ID % 5 + 1))
benchmark=${benchmarks[$benchmark_index]}
problem_count=${problem_counts[$benchmark_index]}
run_name=$(printf 'run_%02d' "$run_index")

load_env_file() {
  local env_file=$1
  local line key value
  if [[ ! -f "$env_file" ]]; then
    echo "Environment file not found: $env_file" >&2
    exit 2
  fi
  # Parse simple KEY=VALUE dotenv entries without executing the file. This
  # also tolerates CRLF line endings in the existing repository .env.
  while IFS= read -r line || [[ -n "$line" ]]; do
    line=${line%$'\r'}
    [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
    key=${line%%=*}
    value=${line#*=}
    if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      echo "Invalid dotenv key in $env_file: $key" >&2
      exit 2
    fi
    printf -v "$key" '%s' "$value"
    export "$key"
  done < "$env_file"
}

case "$model_family" in
  gpt)
    model=gpt-5.2
    provider=openai
    safe_model=gpt-5.2
    openai_env_file=${OPENAI_ENV_FILE:-/hpc/group/fanglab/xx102/Chain-of-Experts_official_parallel/.env}
    load_env_file "$openai_env_file"
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "OPENAI_API_KEY is missing from $openai_env_file" >&2
      exit 2
    fi
    # The baseline code reads the OpenAI-compatible endpoint from these two
    # variables. Exporting them here routes GPT directly to OpenAI without
    # changing any CoE prompt or algorithm code.
    export OPENROUTER_API_KEY=$OPENAI_API_KEY
    export OPENROUTER_API_BASE=${OPENAI_API_BASE:-https://api.openai.com/v1}
    ;;
  gemini)
    model=google/gemini-3-flash-preview
    provider=openrouter
    safe_model=google_gemini-3-flash-preview
    openrouter_env_file=${OPENROUTER_ENV_FILE:-$repo/.env}
    load_env_file "$openrouter_env_file"
    if [[ -z "${OPENROUTER_API_KEY:-}" || -z "${OPENROUTER_API_BASE:-}" ]]; then
      echo "OPENROUTER_API_KEY/OPENROUTER_API_BASE missing from $openrouter_env_file" >&2
      exit 2
    fi
    ;;
  *)
    echo "Unknown MODEL_FAMILY=$model_family; expected gpt or gemini" >&2
    exit 2
    ;;
esac

matrix_root="$repo/log/reruns_matrix/$safe_model/coe_reflection3_conductor10"
run_dir="$matrix_root/$benchmark/$run_name"
completed_gemini_industryor_run1="$repo/log/reruns/IndustryOR/google_gemini-3-flash-preview/coe/run_20260727T213339_344985Z_1204587_345020554"
problem_pattern='.*'
workers=50

if [[ "${SMOKE:-0}" == 1 ]]; then
  benchmark=IndustryOR
  problem_count=1
  run_index=0
  problem_pattern='^0$'
  workers=1
  matrix_root="$repo/log/smoke_backends/$safe_model"
  run_dir="$matrix_root/smoke_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi

cd "$repo"
mkdir -p slurm_logs "$(dirname "$run_dir")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate old_coe

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "model_family=$model_family"
  echo "model=$model"
  echo "provider=$provider"
  echo "benchmark=$benchmark"
  echo "run_index=$run_index"
  echo "problem_count=$problem_count"
  echo "workers=$workers"
  echo "run_dir=$run_dir"
  exit 0
fi

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
retryable_markers = (
    "ratelimiterror", "rate limit", "error code: 429", "timeout",
    "timed out", "apiconnectionerror", "connection error",
    "error code: 502", "error code: 503", "error code: 504",
)

def is_retryable(record):
    if record.get("retryable") is True:
        return True
    error = str(record.get("error") or "").lower()
    return any(marker in error for marker in retryable_markers)

complete = (
    len(records) == expected
    and all(record.get("result") in valid_results for record in records.values())
    and not any(is_retryable(record) for record in records.values())
)
raise SystemExit(0 if complete else 1)
PY
}

# Reuse the validated historical Gemini IndustryOR run as run 1.
if [[ "$model_family" == gemini && "$benchmark" == IndustryOR && "$run_index" -eq 1 ]]; then
  if run_is_complete "$completed_gemini_industryor_run1/results.jsonl" "$problem_count"; then
    if [[ ! -e "$run_dir" && ! -L "$run_dir" ]]; then
      ln -s "$completed_gemini_industryor_run1" "$run_dir"
    fi
    echo "Reused completed Gemini IndustryOR run 1: $completed_gemini_industryor_run1"
    exit 0
  fi
  echo "Expected reusable run is incomplete: $completed_gemini_industryor_run1" >&2
  exit 1
fi

if run_is_complete "$run_dir/results.jsonl" "$problem_count"; then
  echo "Skipping complete run: $run_dir"
  exit 0
fi

runner_args=(
  --dataset "$benchmark"
  --problem "$problem_pattern"
  --algorithm coe
  --model "$model"
  --provider "$provider"
  --enable_reflection
  --max_collaborate_nums 3
  --max_trials 3
  --max_problems "$problem_count"
  --num_processes "$workers"
  --run_index "$run_index"
)

for pass_index in 1 2 3; do
  location_args=()
  if [[ -d "$run_dir" ]]; then
    echo "Pass $pass_index: resuming incomplete run: $run_dir"
    location_args+=(--resume_dir "$run_dir")
  else
    echo "Pass $pass_index: starting new run: $run_dir"
    location_args+=(--run_dir "$run_dir")
  fi

  python run_exp_metrics_litellm.py "${runner_args[@]}" "${location_args[@]}"

  if run_is_complete "$run_dir/results.jsonl" "$problem_count"; then
    echo "Completed: $model | $benchmark | independent run $run_index"
    exit 0
  fi

  if [[ "$pass_index" -lt 3 ]]; then
    echo "Transient or missing records remain; retrying after 90 seconds."
    sleep 90
  fi
done

echo "Run remains incomplete after three passes: $run_dir" >&2
exit 1
