#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

# Optional overrides:
#   NUM_PROCESSES=3 TIMEOUT_DURATION=12h ./run_gemini25_industryor_fresh.sh
#   ./run_gemini25_industryor_fresh.sh /path/to/new/output/root
NUM_PROCESSES="${NUM_PROCESSES:-5}"
TIMEOUT_DURATION="${TIMEOUT_DURATION:-8h}"
OUTPUT_ROOT="${1:-${SCRIPT_DIR}/log/fresh_gemini25_industryor_${RUN_ID}}"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "Refusing to overwrite existing output path: ${OUTPUT_ROOT}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
cd "${SCRIPT_DIR}"

COMMAND=(
    python run_exp_metrics_litellm.py
    --dataset IndustryOR
    --problem '.*'
    --algorithm coe
    --model google/gemini-2.5-flash
    --enable_reflection
    --max_collaborate_nums 3
    --max_trials 3
    --max_problems 0
    --num_processes "${NUM_PROCESSES}"
    --log_dir "${OUTPUT_ROOT}"
)

{
    echo "Started: $(date --iso-8601=seconds)"
    echo "Repository: ${SCRIPT_DIR}"
    echo "Output root: ${OUTPUT_ROOT}"
    echo "Parallel workers: ${NUM_PROCESSES}"
    echo "Overall timeout: ${TIMEOUT_DURATION}"
    printf 'Command:'
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
} | tee "${OUTPUT_ROOT}/run_metadata.txt"

timeout --signal=SIGTERM --kill-after=60s "${TIMEOUT_DURATION}" \
    "${COMMAND[@]}" 2>&1 | tee "${OUTPUT_ROOT}/console.log"

echo "Finished: $(date --iso-8601=seconds)"
echo "Results saved under: ${OUTPUT_ROOT}"
