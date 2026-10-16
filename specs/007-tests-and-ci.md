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

> **Superseded by [015](015-fair-benchmark-suite.md).** The three problems below are separable or
> single-instance, so they favour some libraries for structural reasons. Their results are kept
> as history. New claims and the gate use the 015 suite.

GeneOpt is compared against **other optimisation libraries** on the three problems from 003.

| Optimiser | Library | Kind |
|---|---|---|
| GeneOpt | this repo | genetic algorithm |
| Random search | numpy (in the harness) | baseline, sampling without a model |
| `pygad.GA` | PyGAD | genetic algorithm |
| `eaSimple` + tournament | DEAP | genetic algorithm (DEAP tutorial settings: two-point crossover, uniform-int mutation 5%, cxpb 0.5, mutpb 0.2, tournament 3) |
| `GA` with integer variables | pymoo | genetic algorithm (pymoo's documented integer recipe: SBX + PM with rounding repair) |
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
6. **No free wins from the problem layout.** Rastrigin is *shifted*: its optimum sits on fixed
   off-centre grid points. Plain Rastrigin's optimum is the centre of the box, and NGOpt starts at
   the centre, so in a first trial it "solved" the problem on its first guess. Holder's optima are
   at the corners already.
7. **The call cap applies to GeneOpt too.** GeneOpt answers repeats from its own cache, so they never
   reach the wrapper. It's bounded by the same number of proposals instead:
   `generations = 50 × N / population`.
8. **Budget-aware optimisers are told the real budget.** NGOpt picks its strategy from `budget`,
   so it gets the same number of unique evaluations as the wrapper allows (including in `--quick`).

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

### Benchmark results (2026-10-16, standard operators, local run, Linux, 8 jobs)

Seeds: 30 · budget scale: 1 · Python 3.12.14 · GeneOpt 2.0.0, pygad 3.8.1, deap 1.4.4, pymoo 0.6.2, optuna 5.0.0, nevergrad 1.0.12

#### holder (2-D, sizes 4096, budget 2000 unique evals)

| optimiser | median best [IQR] (lower is better) | unique evals | seconds |
|---|---|---|---|
| pymoo | -19.2085 [-19.21, -19.21] | 2000 | 0.25 |
| deap | -19.208 [-19.21, -19.21] | 2000 | 0.89 |
| optuna | -19.2071 [-19.21, -19.21] | 2000 | 72.44 |
| geneopt | -19.2036 [-19.21, -19.19] | 2000 | 0.04 |
| pygad | -19.1888 [-19.2, -19.16] | 2000 | 1.17 |
| nevergrad | -19.1313 [-19.18, -18.99] | 2000 | 2.53 |
| random | -19.1087 [-19.15, -19.03] | 2000 | 0.01 |

#### rastrigin (10-D, sizes 256, budget 5000 unique evals)

| optimiser | median best [IQR] (lower is better) | unique evals | seconds |
|---|---|---|---|
| pymoo | 0 [0, 0] | 5000 | 0.51 |
| deap | 0.636473 [0.6365, 0.9547] | 5000 | 0.83 |
| geneopt | 3.5935 [2.907, 4.331] | 5000 | 0.16 |
| nevergrad | 6.65427 [5.677, 8.43] | 5000 | 30.73 |
| pygad | 11.716 [9.09, 13.58] | 5000 | 0.34 |
| optuna | 39.1921 [31.81, 40.99] | 5000 | 225.49 |
| random | 89.7588 [82.56, 98.67] | 5000 | 0.08 |

#### categorical (6-D, sizes 5, budget 500 unique evals)

| optimiser | hit rate | unique evals | seconds |
|---|---|---|---|
| geneopt | 100% (median best -6) | 500 | 0.01 |
| deap | 100% (median best -6) | 270 | 0.41 |
| pymoo | 100% (median best -6) | 500 | 0.26 |
| optuna | 100% (median best -6) | 500 | 1.68 |
| pygad | 97% (median best -6) | 500 | 0.02 |
| nevergrad | 70% (median best -6) | 448 | 129.33 |
| random | 3% (median best -5) | 500 | 0.00 |

**Gate (beats random search everywhere):** PASS

**Reading the results:**
- **Gate passes.** GeneOpt beats random search on all three problems.
- **GeneOpt uses the standard operators** (003): uniform crossover and random-resetting
  mutation at 1/n_params, with tournament selection and elitism. The other six rows are identical
  to the first run (same seeds and versions), so the comparison is like-for-like.
- **Categorical:** GeneOpt ties for 1st with DEAP, pymoo and Optuna, at a 100% hit rate.
- **Rastrigin: 3rd** (median 3.59), behind pymoo (0) and DEAP (0.64). That's 43% better than the
  first 2.0 build's custom operators (6.28, spec 012).
- **Holder: 4th** (median −19.2036, against −19.2085 for pymoo). The first build placed 1st
  here. Random resetting has no small local steps, so it fine-tunes less on a fine 4096-point grid.
- **Wall time:** GeneOpt is the fastest library on every problem (0.01–0.16 s per run).
- DEAP stops at about 270 unique evaluations on the categorical problem. It converges and only
  re-proposes known points until the 50 × N call cap ends the run.

**First 2.0 build (custom operators), for comparison:** Holder −19.2085 (1st, tied), Rastrigin
6.28 (3rd), categorical 100% (1st, tied). The swap-rate measurements are in spec 012.

## CI (`.github/workflows/ci.yml`)

| Job | Matrix | Steps |
|---|---|---|
| lint | 3.12 | `ruff check`, `ruff format --check`, `mypy geneopt` |
| test | 3.10 – 3.13 × {ubuntu, windows, macos} | `pip install -e .[dev]` → `pytest -m "not plots"` |
| test-plots | 3.12 ubuntu | `pip install -e .[dev,analysis]`, `pytest -m plots` + coverage gate |
| build | 3.12 | `python -m build`, `twine check dist/*` |
| publish | on tag `v*`, after all of the above | trusted publishing to PyPI (`id-token: write`, environment `pypi`) |
| bench (`bench.yml`) | `workflow_dispatch`, PRs labelled `perf` | full benchmark, table posted as PR comment and job summary; fails on the 003 gate |

The benchmark lives in its own workflow, `bench.yml`, because it triggers on PR *label* events.
In `ci.yml` that trigger would re-run the whole matrix every time any label is added.

Windows matters because the disk cache writes files (path handling, file locking on append).
Subprocess tests must pass the parent environment through: Windows Python can't start without
`SYSTEMROOT`.

## Coverage bar

- `geneopt/cache.py`, `geneopt/core.py`: ≥ 90% line coverage, enforced with
  `coverage report --include=geneopt/cache.py,geneopt/core.py --fail-under=90`. These are the
  parts that can give wrong answers.
- `geneopt/plots.py`: smoke-tested only (each plot writes a non-empty PNG). No pixel tests.

## Acceptance criteria

- [x] Every ledger ID in `specs/README.md` matches at least one `test_<ID>_*`, enforced by
      `tests/test_ledger.py`, which also fails on a ledger row without a test.
- [x] The core suite passes with only `.[dev]` installed. `-m plots` passes with `.[dev,analysis]`.
- [x] `ruff check`, `ruff format --check`, `mypy geneopt` are clean. The coverage bar is met: core
      suite gives cache 99% / core 93%, full suite gives 100% / 99%.
- [x] The benchmark produces the table for all seven optimisers × three problems, and the 003
      gate passes (results above).
- [x] Both workflows pass actionlint. Every job's commands pass locally on Linux: lint; test on
      Python 3.10, 3.12 and 3.13; test-plots + coverage gate; build + twine check; bench.
      `publish` can only run on GitHub.
- [x] CI is green on the full GitHub matrix and takes < 5 min: run 38811579288 on PR #1 took
      1 min 40 s, with all 16 jobs passing and `publish` correctly skipped (no tag). The first run
      failed on all 8 Windows/macOS jobs: a test checked `(ROOT / "GeneOpt").exists()`, which is
      true on case-insensitive filesystems because `geneopt/` exists. Fixed by comparing exact
      directory names.
