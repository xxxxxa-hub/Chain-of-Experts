# GPT-5.2 and Gemini 3 Flash Preview matrix checklist

- [x] Benchmarks, sizes, models/providers, settings, and metrics are explicit.
- [x] Completed IndustryOR run 1 is validated at 100 records and assigned to
  the matrix without rerunning it.
- [x] Artifact layout preserves raw parse-failure responses and final outputs.
- [x] Legacy artifact filenames remain available.
- [x] Threaded dynamic imports use problem-specific module names.
- [x] Resume skips every problem already recorded in JSONL, including errors.
- [x] Interleaved sbatch maps one `0-29%10` array to 30 logical runs, with
  even GPT and odd Gemini tasks.
- [x] Provider 429 failures are excluded from benchmark results and retried.
- [x] Array concurrency is capped at ten runs; each run retains 50 workers.
- [x] Syntax and bounded mocked artifact/import checks pass.
- [x] Changes are committed and pushed (`f8afa29`).
- [x] Initial array `50847605` was cancelled after OpenRouter exposed its
  shared 275 RPM limit.
- [x] Corrected throttled Slurm array is submitted (job `50847696`).
- [x] GPT direct-OpenAI backend smoke test passes (job `50848715`).
- [x] Replacement 30-task `%10` GPT/Gemini array is submitted (job
  `50849200`); superseded jobs `50848942` and `50848943` were cancelled.
- [ ] All 30 logical runs have complete result cardinality.
- [ ] Aggregate per-benchmark mean/variation and failure composition are
  reported.
