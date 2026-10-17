# Spec 015 — Task List

Tests: `python -m pytest -q` (+ `--bench` in the bench env) · Lint: `ruff check . && ruff format --check . && mypy`

## Task 1: Suite problems
**Description:** `tests/bench/suites.py` builds a problem for (suite, function, D, instance). bbob-mixint: integer variables as-is, continuous ones on a 256-value grid, COCO observer logging under the output dir, f_opt read from the `.dat` header. PBO: the optimiser sees the transformed value (minimised), and the internal value is recorded per evaluation. The stated optimum is read from instance 1, with `inf` meaning unknown.
**Acceptance criteria:**
- [x] Cardinalities for D = 10 and 20 match spec 015's table
- [x] mixint: f − f_opt ≥ 0 for random points; f_opt parsed from the log equals `measured − (f − f_opt)` in the same log
- [x] PBO: recorded internal value equals `state.current_internal.y`; F18/F25 give unknown optimum
**Verification:** `pytest tests/test_bench_suites.py` in the bench env
**Dependencies:** None
**Files:** `tests/bench/suites.py`, `tests/test_bench_suites.py`, `tests/bench/requirements.txt`
**Scope:** M

## Task 2: Run records, GeneOpt config, Nevergrad
**Description:** `run_fair(...)` runs one optimiser on one suite problem and returns a record with the anytime trace and early best (first 50 evals). Literal `GENEOPT_CONFIG` table. Nevergrad uses `Choice(repetitions=n)` / integer `Array`.
**Acceptance criteria:**
- [x] Record trace is monotone and its last value is the run's best; `evaluations` ≤ budget
- [x] Config table equals factor / D exactly for every (suite, D); the adapter passes every operator parameter of `GeneticAlgorithm` explicitly (signature check)
- [x] Nevergrad finishes a PBO n = 64, 500-eval run in < 30 s
**Verification:** `pytest tests/test_bench*.py` (bench env)
**Dependencies:** Task 1
**Files:** `tests/bench/run.py`, `tests/test_bench.py`
**Scope:** M

## Task 3: Scoring and report
**Description:** From records: references, precision, evaluations-to-target per target, median final precision per (function, D), Friedman mean ranks, Wilcoxon GeneOpt main vs each, head-start flags, ECDF area, gate, config section, Markdown.
**Acceptance criteria:**
- [x] Hand-built records: evaluations-to-target, ECDF area and ranks equal hand-computed values
- [x] Gate: passes at ≥ 90% better-than-random pairs with higher ECDF area; fails otherwise; both-solved ties count as wins
- [x] PBO reference = max(stated, observed), and observed-reference problems are listed; head-start flags at the 1e−2 target
**Verification:** `pytest tests/test_bench.py` (pure Python + scipy; scipy skipped if absent)
**Dependencies:** Task 2
**Files:** `tests/bench/score.py`, `tests/test_bench_score.py`
**Scope:** M

## Task 4: CLI and CI sharding
**Description:** `run.py --suite {mixint,pbo,legacy}` with `--dims`, `--instances`, `--functions`, `--only`, `--out-dir`, `--report-only`, `--budget-scale`. `bench.yml`: matrix optimiser × suite × instance, artifacts, a report job that merges and gates.
**Acceptance criteria:**
- [x] `--bench` smoke test runs mixint f1 and PBO F19 at a tiny budget for two optimisers end-to-end and writes a report
- [x] `--report-only` on shard outputs gives the same report as one combined run
- [x] `bench.yml` passes `actionlint`-style review (valid YAML, matrix 70 jobs, report job needs all)
**Verification:** `pytest --bench`, YAML parse
**Dependencies:** Task 3
**Files:** `tests/bench/run.py`, `tests/test_bench.py`, `.github/workflows/bench.yml`
**Scope:** M

## Task 5: First report
**Description:** Run CI scope locally (mixint D10 i1–5, PBO n64 i1–5, all optimisers + 3 GeneOpt rows); commit the report as `docs/benchmarks/2026-10-16-fair-suite.md`; update specs.
**Acceptance criteria:**
- [x] Report committed with config section, ranks, head-start check and gate result
- [x] Specs 015 (status, acceptance), 007 pointer, 013 checklist, specs/README updated
**Verification:** report present, `pytest -q`
**Dependencies:** Task 4
**Files:** `docs/benchmarks/*.md`, `specs/*.md`
**Scope:** S (mostly wall time)
