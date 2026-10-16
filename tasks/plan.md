# Implementation Plan: spec 015 — fair benchmark suite

(Earlier plans for specs 011, 007, 012 and 014 are complete; see git history of this file.)

## Overview

Add COCO bbob-mixint and IOHprofiler PBO to the benchmark harness. Every run records an
anytime trace of precision. The report scores precision against targets, checks for head
starts, ranks the optimisers and applies the new gate. CI shards the work, and one local run at
CI scope produces the first committed report.

## Architecture decisions

- **New module `tests/bench/suites.py`** builds problems from `cocoex` and `ioh` and computes
  precision. `run.py` keeps the adapters, the Budget wrapper and the CLI. It grows a `--suite`
  switch, with `legacy` the old three problems.
- **One record per run** (JSON line): suite, function, D, instance, optimiser label, seed, the
  GeneOpt config, unique evaluations, seconds, error, raw best values, and the trace as
  `(evaluation, best raw value)` at each improvement. Precision is computed at **report time**,
  because PBO references (`max(stated, best observed)`) need all runs. Shards therefore just
  write records, and `--report-only DIR` merges them.
- **The GeneOpt configuration is a literal table** keyed by (suite, D): main row 1/D plus
  sensitivity rows 0.5/D and 2/D, with every operator parameter written out. Labels are
  `geneopt[mr=1/D]` etc.
- **Seeds:** the instance id is the seed (one run per optimiser and problem, as COCO does).
- **Tests needing `cocoex`/`ioh`** use `pytest.importorskip`. They run in `bench.yml`, which
  installs the bench requirements. Core CI stays numpy-only, and pure scoring tests run there.
- Figures are 016's. `cocopp` isn't added here.

## Task list

- [ ] Task 1: `suites.py`: problems, cardinalities, f_opt from COCO logs, IOH internal values
- [ ] Task 2: per-run records with anytime traces, GeneOpt config table, vectorised Nevergrad
- [ ] Task 3: scoring and report: targets, references, head-start check, ranks, Wilcoxon, ECDF area, gate
- [ ] Task 4: CLI (`--suite/--dims/--instances/--functions/--out-dir/--report-only`), sharded `bench.yml`
- [ ] Task 5: local run at CI scope, committed dated report; specs 007/013/015/README updated

### Checkpoint
- [ ] `pytest -q` (core env, and bench env with `--bench`), ruff, mypy
- [ ] Spec 015 acceptance boxes ticked, or explained where the run says otherwise
- [ ] Pushed; CI green on PR #1

## Risks
| Risk | Mitigation |
|---|---|
| Local CI-scope run is long (Optuna ≈ 9 CPU-h on PBO) | 12 processes; run in the background; per-run records survive interruption |
| A competitor crashes on a new suite | recorded as `error` in its record, reported, never aborts the run |
| The gate fails | it's a finding: reported as is, then I stop and ask before changing anything |
