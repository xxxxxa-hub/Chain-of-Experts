# Gemini 3 Flash Preview matrix checklist

- [x] Benchmarks, sizes, model, settings, and metrics are explicit.
- [x] Completed IndustryOR run 1 is validated at 100 records and assigned to
  the matrix without rerunning it.
- [x] Artifact layout preserves raw parse-failure responses and final outputs.
- [x] Legacy artifact filenames remain available.
- [x] Threaded dynamic imports use problem-specific module names.
- [x] Resume skips every problem already recorded in JSONL, including errors.
- [x] Slurm array maps 15 tasks to three benchmarks × five runs.
- [x] Provider 429 failures are excluded from benchmark results and retried.
- [x] Array concurrency is capped at one run while retaining 50 workers within
  each run.
- [x] Syntax and bounded mocked artifact/import checks pass.
- [x] Changes are committed and pushed (`f8afa29`).
- [x] Initial array `50847605` was cancelled after OpenRouter exposed its
  shared 275 RPM limit.
- [x] Corrected throttled Slurm array is submitted (job `50847696`).
- [ ] All 15 logical runs have complete result cardinality.
- [ ] Aggregate per-benchmark mean/variation and failure composition are
  reported.
