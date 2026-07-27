# GPT-5.2 and Gemini 3 Flash Preview: three benchmarks × five runs

## Experiment contract

- Route: reproduce the local Chain-of-Experts baseline without changing its
  prompts, expert-selection behavior, reflection algorithm, or evaluator.
- Models/providers: `gpt-5.2` through the OpenAI API and
  `google/gemini-3-flash-preview` through OpenRouter.
- Benchmarks: ComplexLP (211 problems), IndustryOR (100), and BWOR (82).
- Repetitions: five independent API runs per model and benchmark (30 logical
  runs total).
- Settings: temperature 1, high reasoning, reflection enabled, three
  collaborations, three trials, and the existing 10-token Conductor limit.
- Parallelism: 50 Python workers per run. GPT and Gemini use separate Slurm
  arrays, each capped at one active task, so the two providers can run
  concurrently without launching two runs against the same provider.
- Metric: exact repository evaluator result, summarized as ACCEPT,
  WRONG_ANSWER, COMPILE_ERROR, and RUNTIME_ERROR.
- Existing evidence: the completed IndustryOR run with 75 ACCEPT,
  21 WRONG_ANSWER, and 4 RUNTIME_ERROR is reused as IndustryOR run 1.
- No post-hoc repair or LLM objective extraction is part of the metric.

## Artifact contract

New runs are stored under:

`log/reruns_matrix/<safe_model>/coe_reflection3_conductor10/<benchmark>/run_XX`

Each problem stores readable artifacts under:

```text
problems/<problem>/
  reflection/trial_XX/
    selected_experts.jsonl
    reducer_answer.txt
    generated_code.py
    evaluation_feedback.txt
    backward/step_XX_<expert>/
      raw_response.txt
      parsed_response.json
  final/
    original_answer.txt
    generated_code.py
    test_log.txt
```

If an unrecorded, partially written problem is resumed, its new artifacts go
under `problems/<problem>/attempts/<timestamp>/` instead of overwriting the
partial attempt. Legacy root-level answer/code/test-log files are retained as
compatibility copies.

## Execution and acceptance

- Entrypoint: `submit_gpt_gemini_3bench_5x.sbatch`, submitted once with
  `MODEL_FAMILY=gpt` and once with `MODEL_FAMILY=gemini`.
- Scheduler shape: two arrays of tasks 0–14 with `%1`; Gemini task ID 5 reuses
  the completed IndustryOR run 1 and exits without API calls.
- Resubmission behavior: complete runs are skipped; incomplete runs resume
  problems absent from `results.jsonl` plus transient 429/timeout/connection
  failures. Each task makes up to three passes with a 90-second cooldown.
- Acceptance: each logical run has exactly the benchmark's expected number of
  unique problem records and a valid four-way result for every problem.
- Durable scheduler logs: `slurm_logs/`.

## Risks

- API-side stochasticity and provider changes can vary results across runs.
- Provider-side RPM limits can change independently of the local API account;
  the observed OpenRouter response capped this model at 275 RPM.
- The artifact/import isolation changes are external bookkeeping and
  concurrency-safety changes; they do not intentionally alter CoE behavior.

## Revision log

| Time | Change | Reason | Impact |
|---|---|---|---|
| 2026-07-27 | Created isolated rerun branch and paths | Preserve old results | No algorithm change |
| 2026-07-27 | Completed first Gemini IndustryOR run | Establish reproducibility point | 75/100 ACCEPT |
| 2026-07-27 | Added per-problem/per-trial artifacts | Make failures auditable | Logging only |
| 2026-07-27 | Added 3×5 Slurm matrix | Run independent repetitions concurrently | 14 new runs |
| 2026-07-27 | Capped array concurrency after observed 429s | OpenRouter exposed a shared 275 RPM limit | Keep 50 workers within one run; retry transient failures |
