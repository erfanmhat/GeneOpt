# Implementation Plan: spec 007 — Tests, Benchmarks & CI

(The spec 011 / 2.0.0 plan is complete; see git history of this file.)

## Overview

Make the 2.0 test suite meet spec 007:
- a core suite that runs with numpy only, plus a separate plots suite
- lint and type checks
- a named regression test for every bug ID in the ledger
- property tests
- a coverage bar on the code that can give wrong answers
- a benchmark against other optimisation libraries (PyGAD, DEAP, pymoo, Optuna, Nevergrad) and
  random search
- a GitHub Actions workflow

## Architecture decisions

- **Marker split.** Tests that need pandas/matplotlib/catboost get `@pytest.mark.plots`. The core
  CI job runs `-m "not plots"` with only `.[dev]` installed. That's how "core, no plotting extras"
  gets proven rather than assumed.
- **Dev extra** gains `hypothesis`, `ruff`, `mypy`, `pytest-cov`. None of it is installed into the
  user's conda env. Verification uses throwaway venvs in the scratchpad (uv, with its cache and
  Python downloads kept in the scratchpad).
- **The benchmark is library-agnostic** through a budget wrapper around the objective. It memoises
  calls, records the best score among the first N *unique* calls, then raises a sentinel to stop
  the run (with a 50 × N total-call cap). Every library gets the same metric and the same free
  duplicates, and none of them needs a budget feature. Competitor pins live in
  `tests/bench/requirements.txt` (spec 007 § Benchmarks).
- **The ledger meta-test** parses the ID column of `specs/README.md` and fails if any ID has no
  `test_<ID>_*` function. That makes criterion 1 self-enforcing.

## Task list

### Phase 1: Suite structure
- [x] Task 1: `plots` marker; core suite passes with numpy + pytest only
- [ ] Task 2: ruff + mypy config, `py.typed`, findings fixed

### Checkpoint A
- [ ] Core suite green in a numpy-only venv. `ruff check`, `ruff format --check`, `mypy geneopt` clean

### Phase 2: Coverage of the ledger
- [ ] Task 3: Ledger meta-test + a named test for every bug ID + hypothesis property tests
- [ ] Task 4: Coverage ≥ 90% on `cache.py` and `core.py`, enforced

### Checkpoint B
- [ ] All 29 ledger IDs have tests. Coverage bar met

### Phase 3: Benchmarks & CI
- [ ] Task 5: Benchmark harness (`tests/bench/`) vs 5 libraries + random search, results recorded
- [ ] Task 6: `.github/workflows/ci.yml` (lint, test matrix, test-plots, build, publish, bench)

### Checkpoint C (complete)
- [ ] Workflow passes actionlint. Linux matrix (3.10 / 3.12 / 3.13) reproduced locally with uv

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| GeneOpt doesn't beat random search (003 hard gate) | High | That's a real finding. **Stop and report** with the numbers instead of tuning to pass |
| GeneOpt ranks low against TPE / NGOpt | Low | Expected on small budgets for model-based optimisers. Reported, not gated (003) |
| A competitor's adapter is unfair (wrong encoding, starved population) | Med | Fairness rules in 007. Each adapter is ≤ 30 lines and uses library defaults. Versions pinned and printed |
| Optuna TPE is slow at 5,000 trials × 30 seeds | Med | The harness takes `--seeds` and `--only`. The seed count actually run is reported with the results |
| Windows/macOS CI can't be checked locally | Med | Only Linux is checked locally. Windows/macOS stay unverified until the workflow runs on GitHub |

## Decisions (resolved)

1. Benchmark against other libraries instead of 1.x (user). Specs 003 and 007 rewritten.
2. 007 criterion "the suite fails on 1.0.10" replaced. It's impossible after the hard break, so
   the 1.x side is covered by the inline reproductions (spec 007 § Principle).
3. Test files keep their current names. Spec 007's list was updated to match.
4. No push. CI is validated with actionlint and a local Linux matrix. The GitHub run is pending a
   push. `origin` now points to the canonical `github.com/erfanmhat/GeneOpt` (GAPY was its old
   name, and GitHub redirects it).
