# Gemini 3 Flash Preview matrix checklist

- [x] Benchmarks, sizes, model, settings, and metrics are explicit.
- [x] Completed IndustryOR run 1 is validated at 100 records and assigned to
  the matrix without rerunning it.
- [x] Artifact layout preserves raw parse-failure responses and final outputs.
- [x] Legacy artifact filenames remain available.
- [x] Threaded dynamic imports use problem-specific module names.
- [x] Resume skips every problem already recorded in JSONL, including errors.
- [x] Slurm array maps 15 tasks to three benchmarks × five runs.
- [x] Syntax and bounded mocked artifact/import checks pass.
- [ ] Changes are committed and pushed.
- [ ] Slurm array is submitted.
- [ ] All 15 logical runs have complete result cardinality.
- [ ] Aggregate per-benchmark mean/variation and failure composition are
  reported.
