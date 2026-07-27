# Local CoE baseline rerun

## Core contract

- Goal: reproduce one historical `/Chain-of-Experts` result without changing
  the CoE algorithm or overwriting any prior artifacts.
- Route: reproduce.
- Baseline id: `local-coe-gpt52-industryor`.
- Source snapshot: `5176f35` on `origin/main`.
- Task: Chain-of-Experts on all 100 IndustryOR problems.
- Model/settings: `gpt-5.2` through the repository's OpenRouter configuration,
  temperature 1, high reasoning, reflection enabled, 3 collaborations, 3
  trials, and the existing 10-token Conductor limit.
- Metric: `ACCEPT / 100`, with wrong-answer, compile-error, and runtime-error
  counts reported separately.
- Historical comparison target: approximately 63% accuracy in the existing
  recorded result.
- Entrypoint: `run_exp_metrics_litellm.py`.
- Output: a unique directory below
  `log/reruns/IndustryOR/gpt-5.2/coe/`, containing `metadata.json`,
  `results.jsonl`, generated programs, test logs, and a summary JSON.
- Acceptance: the run finishes with 100 recorded problems and the observed
  accuracy/failure mix can be compared directly with the historical result.
- Fallback: resume the exact run directory; never reuse or overwrite an older
  result directory.

## Execution path

- Working directory: `/hpc/group/fanglab/xx102/Chain-of-Experts`.
- Environment: existing `old_coe` conda environment and repository `.env`.
- Smoke test: problem `0`, one worker, otherwise identical settings.
- Main run: `submit_reproduce_local_gpt52_industryor.sbatch`.
- Durable scheduler logs: `slurm_logs/`.
- Fastest failure signals: missing API environment variables, import failure,
  or no new JSONL records after the initial API wave.

## Risks

- The historical implementation uses threaded workers and second-resolution
  reflection paths; this behavior is intentionally preserved for comparison.
- API-side stochasticity and provider changes can move accuracy even when the
  local code is identical.
- The runner path/metadata changes are external to the CoE algorithm.

## Revision log

| Time | Change | Reason | Impact |
|---|---|---|---|
| 2026-07-27 | Created isolated rerun path | Prevent overwrite | No algorithm change |
