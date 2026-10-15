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
- [x] Table for 1/n, 0.2 and 0.5 (30 seeds, all problems) recorded in spec 012
- [x] Variant parsing unit-tested in the core suite
**Dependencies:** 1
**Files:** `tests/bench/run.py`, `tests/test_bench.py`, `specs/012-crossover-mixing.md`
**Scope:** S

## Task 3 (replaced): standard operators
**Decision (user, 2026-10-16):** replace mutation, selection and crossover with the most standard approach.
**Description:** Crossover becomes uniform: each gene of a crossing pair swaps with p = 0.5, and pairs cross with `crossover_rate`. Mutation becomes random resetting: each gene, with p = `mutation_rate`, gets a uniformly chosen *other* value. `mutation_rate` defaults to `None` = 1/n_params. Selection stays as tournament (k = 3) + elitism (1), which is already standard. Remove `adaptive_mutation`, `crossover_swap_rate`, creep (`CREEP_SIGMA`, `_is_ordered_numeric`), `MAX_MUTATION_RATE` and `_mutation_rate`.
**Acceptance criteria:**
- [x] `_crossover` swaps about 50% of the genes of crossing pairs, and none when `crossover_rate=0`
- [x] `_mutate` changes each gene with probability `mutation_rate` (±10%), always to a different value, never outside range
- [x] Default `mutation_rate` is 1/n_params. Removed parameters raise `TypeError`. Ledger tests E-2/E-3 updated to the new rule
- [x] Full suite, lint and mypy green. Gate (beats random search) still passes on a quick check
**Files:** `geneopt/core.py`, `tests/test_core.py`, `tests/test_api.py`, `tests/bench/run.py`, `tests/test_bench.py`
**Scope:** M

## Task 4: Benchmark and docs
**Description:** Full 30-seed benchmark with all seven optimisers. Replace the results in spec 007, close spec 012 with the outcome, and update spec 003 (operators), spec 005 (signature), spec 011 (`adaptive_mutation` never shipped, so there's no 1.x mapping), the README parameter table and the CHANGELOG.
**Acceptance criteria:**
- [x] Gate passes. **If not → stop and report**
- [x] Every Markdown file that mentions the old operators or removed parameters is updated (grep-checked)
- [x] Full suite, lint, CI green on PR #1
**Dependencies:** 3
**Scope:** M
