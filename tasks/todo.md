# Spec 007 — Task List

Tests: `python -m pytest -q` · Core-only: `pytest -m "not plots"` in a numpy-only venv ·
Lint: `ruff check . && ruff format --check . && mypy geneopt` · Build: `python -m build` + `twine check`
(scratch venv)

## Task 1: Core vs plots test split
**Description:** Register a `plots` marker in pyproject. Mark every test that needs pandas, matplotlib or catboost (History.to_frame, evaluations, plots, the quickstart example). Add `hypothesis`, `ruff`, `mypy` and `pytest-cov` to the `dev` extra.
**Acceptance criteria:**
- [x] In a fresh venv with only `.[dev]`, `pytest -m "not plots"` passes, with no import errors and no skips caused by missing modules
- [x] With `.[dev,analysis]`, `pytest -m plots` passes, and the full `pytest` count equals the sum of the two
- [x] An unregistered marker is an error (`--strict-markers`)
**Verification:** both venv runs
**Dependencies:** None
**Files:** `pyproject.toml`, `tests/test_history.py`, `tests/test_plots.py`, `tests/test_examples.py`
**Scope:** S

## Task 2: Lint & types
**Description:** Configure ruff (lint + format, line length 110) and mypy for `geneopt` in pyproject. Add `geneopt/py.typed` and include it in the wheel. Fix every finding without suppressions; if a suppression is unavoidable, it gets an inline reason.
**Acceptance criteria:**
- [x] `ruff check .`, `ruff format --check .` and `mypy geneopt` exit 0
- [x] The wheel contains `geneopt/py.typed`
- [x] Test suite still green
**Verification:** scratch venv with `.[dev]`
**Dependencies:** 1
**Files:** `pyproject.toml`, `geneopt/*.py`, `geneopt/py.typed`, `tests/*.py`, `examples/**` (format only)
**Scope:** M

## Checkpoint A
- [x] Core suite green numpy-only. Lint and types clean

## Task 3: Ledger coverage + property tests
**Description:** `tests/test_ledger.py` parses the ledger IDs from `specs/README.md` and requires a `test_<ID>_*` for each one (ID with `-` → `_`). Rename existing tests that already cover an ID, and add missing ones (GC-2, 4, 5, 6, 8, 11, H-2, 3, 5, E-2, 3, 4, 5, A-1, 2, 3, 4, X-3). Hypothesis tests: best-so-far monotonic over seeds, cache round-trip of arbitrary JSON-able values, uniform initial sampling for sizes 2..300 (chi-square).
**Acceptance criteria:**
- [x] `test_ledger.py` passes, and fails if a ledger row is added without a test (checked by a negative self-test)
- [x] Every new test asserts the 2.0 behaviour that contradicts the verified 1.x behaviour in the ledger
- [x] The property tests run within the default hypothesis budget, in under 10 s total
**Verification:** `pytest -q`
**Dependencies:** 1
**Files:** `tests/test_ledger.py`, `tests/test_*.py`
**Scope:** M

## Task 4: Coverage bar
**Description:** Measure line coverage of `geneopt/cache.py` and `geneopt/core.py` from the core suite. Add tests for any uncovered branches that matter. Enforce the bar with `coverage report --include=geneopt/cache.py,geneopt/core.py --fail-under=90`.
**Acceptance criteria:**
- [ ] ≥ 90% line coverage on each of the two files
- [ ] The enforcement command exits non-zero when the bar is broken (checked once by lowering coverage locally, not committed)
**Verification:** scratch venv `pytest --cov`
**Dependencies:** 3
**Files:** `tests/*.py`, `pyproject.toml`
**Scope:** S

## Checkpoint B
- [ ] All ledger IDs tested. Coverage bar met

## Task 5: Competitor benchmark
**Description:** `tests/bench/run.py` covers the three problems from 003 (Holder 2-D 4096² / 2,000 unique evals; Rastrigin 10-D × 256 / 5,000; categorical 6 × 5 labels / 500, hit rate). It runs seven optimisers: GeneOpt, random search, PyGAD, DEAP, pymoo GA, Optuna TPE and Nevergrad NGOpt, each as a small adapter. A shared memoising budget wrapper stops each run after N unique evaluations (with a 50 × N total-call cap). Seeds 0…N-1. Pins go in `tests/bench/requirements.txt`. Output is a Markdown table per problem (median, IQR or hit rate, unique evals, wall time). Crashes show as `crashed: …`.
**Acceptance criteria:**
- [ ] `python tests/bench/run.py --seeds 30` prints all 7 × 3 results. `--only` and `--quick` work
- [ ] `tests/test_bench.py` (`-m bench`, deselected by default) runs `--quick`, and also unit-tests the budget wrapper (unique counting, stop, cap)
- [ ] Results + seed count + library versions recorded in `specs/007-tests-and-ci.md` § Benchmark results. 003 hard gate (beats random search everywhere) checked. **If it fails → stop and report**
**Verification:** full local run in a scratch venv
**Dependencies:** 2
**Files:** `tests/bench/run.py`, `tests/bench/requirements.txt`, `tests/test_bench.py`, `pyproject.toml` (marker), `specs/007-tests-and-ci.md`
**Scope:** M

## Task 6: CI workflow
**Description:** `.github/workflows/ci.yml` with jobs lint, test (3.10–3.13 × ubuntu/windows/macos, core only), test-plots (+ coverage gate), build (+twine check), publish (tag `v*`, trusted publishing, `id-token: write`, environment `pypi`), and bench (`workflow_dispatch` + PRs labelled `perf`, installing `tests/bench/requirements.txt` and posting the table as a PR comment).
**Acceptance criteria:**
- [ ] actionlint reports no errors
- [ ] Every command in the workflow was run locally on Linux. The test job passes on Python 3.10, 3.12 and 3.13 (uv venvs)
- [ ] Publish runs only on `v*` tags, after the other jobs, with least-privilege permissions
**Verification:** actionlint + local commands
**Dependencies:** 4, 5
**Files:** `.github/workflows/ci.yml`, `specs/007-tests-and-ci.md` (criteria wording per the approval decisions)
**Scope:** S

## Checkpoint C
- [ ] All spec 007 criteria met or explicitly re-scoped
