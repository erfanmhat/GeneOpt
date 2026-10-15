# 007 — Tests, Benchmarks & CI

**Status:** proposed · **Severity:** 🟡 medium · **Covers:** every ledger ID

## Principle

**Every bug ID in the ledger has a regression test**, named `test_<ID>_*` (`GC-7` →
`test_GC_7_*`), so `pytest -k GC_7` finds it. Each one asserts the 2.0 behaviour that contradicts
the 1.0.10 behaviour recorded in the ledger. The 1.0.10 side is already proven by the
reproductions inlined in specs 001–003 and 008, which were run against 1.0.10.

> The suite can't run against 1.0.10 itself. After the hard break (011) it imports `geneopt`,
> which 1.0.10 doesn't have, so every test would fail at import and prove nothing.

`tests/test_ledger.py` makes this self-enforcing: it parses the ID column of
[README.md](README.md) and fails if any ID has no matching test.

## Test files

```
tests/
  test_cache.py          # GenomeCache: GC-1 … GC-10
  test_history.py        # History                                   [plots]
  test_result.py         # Result: A-1 / 010
  test_core.py           # operators + GA loop: E-1 … E-6, GC-7, H-1, GC-3
  test_engine.py         # executor determinism, on_error, logging
  test_api.py            # validation, removed 1.x kwargs raise TypeError (011), no import side effects
  test_plots.py          # H-2 … H-5, smoke test of every plot          [plots]
  test_examples.py       # client X-1 / 008, server X-2 / 009, runnable examples
  test_repo.py           # 1.x files gone, README code runs, CHANGELOG
  test_ledger.py         # every ledger ID has a test
  bench/                 # competitor benchmark (not collected by default)
```

`[plots]` files and tests are marked `@pytest.mark.plots` because they need pandas or matplotlib.
The core suite (`-m "not plots"`) needs only numpy and the `dev` extra.

## Property tests (hypothesis)

`hypothesis` is in the `dev` extra.

- Best-so-far is monotonic for any seed and any `maximize`.
- Cache round-trip through disk for arbitrary JSON-able values (str, int, float, bool, `None`).
- Initial sampling is uniform for any parameter size in 2…300 (chi-square, p > 0.001, 20k draws).

## Benchmarks (`tests/bench/run.py`)

GeneOpt is compared against **other optimisation libraries** on the three problems from 003.

| Optimiser | Library | Kind |
|---|---|---|
| GeneOpt | this repo | genetic algorithm |
| Random search | numpy (in the harness) | baseline, sampling without a model |
| `pygad.GA` | PyGAD | genetic algorithm |
| `eaSimple` + tournament | DEAP | genetic algorithm (classic toolbox) |
| `GA` with integer variables | pymoo | genetic algorithm |
| `TPESampler` | Optuna | Bayesian / tree-structured Parzen estimator |
| `NGOpt` | Nevergrad | algorithm-selecting meta-optimiser |

**Fairness rules:**
1. **Same search space.** Every optimiser sees the problem as `n` integer indices in `[0, size)`,
   or as categorical choices where the library supports them (Optuna, Nevergrad).
2. **Same budget, counted in *unique* evaluations.** A shared wrapper memoises the objective, so
   re-evaluating a duplicate is free for every library (matching GeneOpt's cache). It records the
   best score among the first N unique calls, then raises a sentinel that stops the run. Each
   library is also capped at 50 × N total calls, so one that only re-proposes duplicates still
   terminates.
3. **Library defaults**, except population size: the GA libraries all get the same population (50)
   and are told to run "forever" (the budget wrapper stops them). No per-problem tuning for anyone.
4. **Seeds 0…29 for everyone.** Libraries that take a seed get it; the harness also seeds
   `random` and `numpy` globally before each run.
5. **Versions** are pinned in `tests/bench/requirements.txt` and printed in the report.

**Output:** a Markdown table per problem, with these columns:
- optimiser
- median best and IQR (Holder, Rastrigin), or hit rate of the optimum (categorical)
- median unique evaluations used
- median wall time

Crashes are shown as `crashed: <error>` instead of aborting the table.

**Gate:** see [003 § Benchmark gate](003-encoding-and-operators.md#benchmark-gate). GeneOpt must
beat random search on every problem; its rank among the other libraries is reported, not gated.

**Running:** `pip install -e . -r tests/bench/requirements.txt`, then
`python tests/bench/run.py --seeds 30`. The pins live in a requirements file, not a package
extra, because users installing `GeneOpt` should never get pinned competitor libraries.
`--quick` (2 seeds, 10% budget) is the pytest smoke test (`-m bench`, deselected by default).
CI runs the full benchmark on `workflow_dispatch` and on PRs labelled `perf`, and posts the table
as a PR comment.

## CI (`.github/workflows/ci.yml`)

| Job | Matrix | Steps |
|---|---|---|
| lint | 3.12 | `ruff check`, `ruff format --check`, `mypy geneopt` |
| test | 3.10 – 3.13 × {ubuntu, windows, macos} | `pip install -e .[dev]` → `pytest -m "not plots"` |
| test-plots | 3.12 ubuntu | `pip install -e .[dev,analysis]`, `pytest -m plots` + coverage gate |
| build | 3.12 | `python -m build`, `twine check dist/*` |
| publish | on tag `v*`, after all of the above | trusted publishing to PyPI (`id-token: write`, environment `pypi`) |
| bench | `workflow_dispatch`, PRs labelled `perf` | full benchmark, table posted as PR comment |

Windows matters because the disk cache writes files (path handling, file locking on append).

## Coverage bar

- `geneopt/cache.py`, `geneopt/core.py`: ≥ 90% line coverage, enforced with
  `coverage report --include=geneopt/cache.py,geneopt/core.py --fail-under=90`. These are the
  parts that can give wrong answers.
- `geneopt/plots.py`: smoke-tested only (each plot writes a non-empty PNG). No pixel tests.

## Acceptance criteria

- [ ] Every ledger ID in `specs/README.md` matches at least one `test_<ID>_*`, enforced by
      `tests/test_ledger.py`, which also fails on a ledger row without a test.
- [ ] The core suite passes with only `.[dev]` installed. `-m plots` passes with `.[dev,analysis]`.
- [ ] `ruff check`, `ruff format --check`, `mypy geneopt` are clean. The coverage bar is met.
- [ ] The benchmark produces the table for all seven optimisers × three problems, and the 003
      gate passes.
- [ ] The workflow passes actionlint. Every job's commands pass locally on Linux (Python 3.10,
      3.12, 3.13).
- [ ] CI is green on the full GitHub matrix and takes < 5 min. *(This needs a push. Verified on the
      first run on GitHub.)*
