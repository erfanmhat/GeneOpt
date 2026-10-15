# Spec 012 — Task List

Tests: `python -m pytest -q` · Lint: `ruff check . && ruff format --check . && mypy`

## Task 1: `crossover_swap_rate` parameter, `_crossover` extracted
**Description:** Move crossover out of `_next_generation` into `_crossover(p1, p2, swap_rate, crossover_rate, rng)`. Add `crossover_swap_rate: float | None = None` to `GeneticAlgorithm`. `None` means 1/n_params, which is today's behaviour. Validate the range `[0, 1]`.
**Acceptance criteria:**
- [x] With `crossover_swap_rate=None`, a seeded run is identical to an explicit `1/n_params` and to the pre-change code (same Result and history)
- [x] `_crossover` swaps genes at the configured rate (measured within ±10%), never swaps when `crossover_rate=0`, and children always contain exactly the parents' genes
- [x] Out-of-range values raise `ValueError` naming `crossover_swap_rate`
**Verification:** `pytest -q`, lint, mypy
**Dependencies:** None
**Files:** `geneopt/core.py`, `tests/test_core.py`, `tests/test_api.py`
**Scope:** S

## Task 2: Measure swap rates
**Description:** Add `--geneopt-swap-rate` (repeatable) to `tests/bench/run.py`, so GeneOpt can run as several variants (`geneopt@1/n`, `geneopt@0.2`, `geneopt@0.5`). Run 30 seeds on all three problems and record the table in spec 012.
**Acceptance criteria:**
- [ ] Table for 1/n, 0.2 and 0.5 (30 seeds, all problems) recorded in spec 012
- [ ] Variant parsing unit-tested in the core suite
**Dependencies:** 1
**Files:** `tests/bench/run.py`, `tests/test_bench.py`, `specs/012-crossover-mixing.md`
**Scope:** S

## Task 3: Choose the default
**Description:** If one rate closes most of the Rastrigin gap without regressing Holder/categorical, make it the default. Otherwise revise spec 012 with the evidence and stop. Update README parameter table, CHANGELOG, spec 003 §1 and the 007 benchmark table.
**Acceptance criteria:**
- [ ] Spec 012 acceptance criteria met, or the spec revised with evidence
- [ ] Full suite, lint, CI green
**Dependencies:** 2
**Files:** `geneopt/core.py`, tests, `README.md`, `CHANGELOG.md`, `specs/003*`, `specs/007*`, `specs/012*`
**Scope:** M
