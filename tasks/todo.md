# Spec 014 — Task List

Tests: `python -m pytest -q` · Lint: `ruff check . && ruff format --check . && mypy`

## Task 1: Remove `initial_population_size`
**Description:** Delete the parameter, its validation, `_initial_population_size`, and use `population_size` for generation 0.
**Acceptance criteria:**
- [x] `GeneticAlgorithm(f, env, initial_population_size=10)` raises `TypeError`
- [x] Every generation in `history` has exactly `population_size` rows
- [x] `test_initial_population_size_applies_to_generation_zero_only` and its validation case removed; README row removed
**Verification:** `pytest -q`, lint, mypy
**Dependencies:** None
**Files:** `geneopt/core.py`, `tests/test_core.py`, `tests/test_api.py`, `README.md`
**Scope:** S

## Task 2: `mutation_rate` default 0.05
**Description:** Signature `mutation_rate: float = 0.05`; drop the 1/n formula; `None` fails validation. The bench adapter passes its value explicitly.
**Acceptance criteria:**
- [x] Omitting it gives a measured per-gene mutation frequency of 0.05 ± 10% for 2- and 50-parameter environments
- [x] `mutation_rate=None` raises `ValueError` naming the parameter
- [x] `test_E_3_*` asserts the configured rate is used unchanged during the run
**Verification:** `pytest -q`, lint, mypy
**Dependencies:** Task 1
**Files:** `geneopt/core.py`, `tests/test_core.py`, `tests/test_api.py`, `tests/bench/run.py`
**Scope:** S

## Task 3: `crossover_probability`
**Description:** Rename `crossover_rate` → `crossover_probability` (default 0.9) in the constructor and `_crossover`; docstring uses 014 C steps 1–3.
**Acceptance criteria:**
- [x] `crossover_rate=` raises `TypeError`; `crossover_probability` validated in [0, 1]
- [x] p_c = 1, parents all-0 / all-1: each child gene from A with frequency 0.5 ± 0.02; child 2 is always the complement of child 1
- [x] Long genomes: fraction of recombining pairs = p_c ± 0.03 for p_c ∈ {0.3, 0.9}
- [x] p_c = 0: children equal parents
**Verification:** `pytest -q`, lint, mypy
**Dependencies:** Task 2
**Files:** `geneopt/core.py`, `tests/test_core.py`, `tests/test_api.py`
**Scope:** S

## Task 4: Docs
**Description:** README rows and "How a child is made" from 014 verbatim; specs 003, 005, 007, 011 and CHANGELOG updated; README images regenerated from the quickstart.
**Acceptance criteria:**
- [x] README matches 014 § README text; README-embed and image tests pass
- [x] Grep check from the plan checkpoint is clean
- [x] 014 status → implemented, acceptance boxes ticked; specs/README map updated
**Verification:** `pytest -q`, lint, grep
**Dependencies:** Task 3
**Files:** `README.md`, `CHANGELOG.md`, `specs/003*.md`, `specs/005*.md`, `specs/007*.md`, `specs/011*.md`, `specs/014*.md`, `specs/README.md`, `docs/images/*.png`
**Scope:** M
