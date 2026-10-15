# 007 — Tests, Benchmarks & CI

**Status:** proposed · **Severity:** 🟡 medium · **Covers:** every ledger ID

## Principle

**Every bug ID in the ledger gets a test that fails on `1.0.10`** before the fix lands
(Prove-It pattern). The test name contains the ID, so `pytest -k GC_7` finds it.

## Test files

```
tests/
  test_cache.py          # GC-1 … GC-11
  test_history.py        # H-1 … H-4
  test_operators.py      # E-1 … E-6
  test_engine.py         # GC-7, A-1, A-2, determinism, error policy
  test_api.py            # validation, removed 1.x kwargs raise TypeError (011), no import side effects
  test_result.py         # A-1 / 010: one Result type in every mode
  test_http_example.py   # X-1 / 008, X-2 / 009 against a stdlib http.server stub
  test_examples.py       # runs examples/quickstart.py + categorical.py with tiny budgets
  bench/run.py           # benchmark gate from 003 (not in the default pytest run)
```

The tests use the stdlib (`tmp_path`, `monkeypatch`). The only extra dependency is
`hypothesis`, used for the property tests marked ★.

## Key tests (sketches)

```python
def test_GC_1_no_relative_tolerance_hits(tmp_path):
    ga = make_ga({"x": [1_000_000.0, 1_000_005.0]}, cache_dir=tmp_path)
    calls = run_with_population(ga, [[0], [1]])
    assert calls == 2

def test_GC_3_changed_objective_gets_fresh_cache(tmp_path):
    env = {"x": np.linspace(0, 1, 8)}
    GeneticAlgorithm(lambda x: x,  env, cache_dir=tmp_path, generations=3).start()
    r = GeneticAlgorithm(lambda x: -x, env, cache_dir=tmp_path, generations=3).start()
    assert r.params["x"] == 0.0 and r.score == 0.0    # 1.0.10 returns x=1.0, score=1.0

def test_GC_7_duplicates_evaluated_once(executor):    # parametrised: None, ThreadPool(4)
    counter = Counter()
    ga = make_ga(objective=counting(counter), executor=executor)
    run_with_population(ga, [[3, 5]] * 20)
    assert sum(counter.values()) == 1

def test_H_1_history_has_every_individual():
    ga = make_ga(population_size=20, generations=15); ga.start()
    assert [len(s) for s in ga.history.scores] == [20] * 15

★ def test_best_so_far_monotonic(seed): ...
★ def test_cache_roundtrip_any_values(values: list[str | int | float | None]): ...
★ def test_decode_uniform(n: int in 2..300): chi-square p > 0.001 over 20k draws

def test_same_result_sequential_and_parallel():
    a = make_ga(seed=7).start(); b = make_ga(seed=7, executor=ThreadPoolExecutor(4)).start()
    assert replace(a, elapsed=0) == replace(b, elapsed=0)

@pytest.mark.parametrize("kw", ["r_cross", "number_of_population", "cache_type", "verbose", ...])
def test_removed_1x_kwargs_raise(kw):
    with pytest.raises(TypeError, match=kw):
        GeneticAlgorithm(f, env, **{kw: 1})

def test_import_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run([sys.executable, "-c", "import geneopt, sys; assert 'matplotlib' not in sys.modules"], check=True)
    assert list(tmp_path.iterdir()) == []
```

`run_with_population(ga, pop)` is the one test helper: it seeds generation 0 with an explicit
population (through `initial_population=` or a monkeypatched init). It's what made GC-7 easy
to reproduce.

## Benchmarks (`tests/bench/run.py`)

- Runs the three problems from 003 × 30 seeds against **both** the installed `GeneOpt==1.0.10`
  (in a separate venv) and the working tree.
- Prints a Markdown table: median best, IQR, unique evaluations, wall time.
- CI runs it on `workflow_dispatch` and on PRs labelled `perf`, then posts the table as a
  PR comment.

## CI (`.github/workflows/ci.yml`)

| Job | Matrix | Steps |
|---|---|---|
| lint | 3.12 | `ruff check`, `ruff format --check`, `mypy geneopt` |
| test | 3.10 – 3.13 × {ubuntu, windows, macos} | `pip install -e .[dev]` → `pytest` (core, no plotting extras) |
| test-plots | 3.12 ubuntu | `pip install -e .[dev,analysis]`, `MPLBACKEND=Agg pytest -m plots` |
| build | 3.12 | `python -m build`, `twine check dist/*` |
| publish | on tag `v*` | trusted publishing to PyPI |

Windows matters because the disk cache writes files (path handling, file locking on append).

## Coverage bar

- `geneopt/cache.py`, `geneopt/core.py`: ≥ 90% line coverage. These are the parts that can
  give wrong answers.
- `geneopt/plots.py`: smoke-tested only (each plot writes a non-empty PNG). No pixel tests.

## Acceptance criteria

- [ ] Every ledger ID in `specs/README.md` matches at least one `test_<ID>_*`.
- [ ] Checking out `1.0.10` and running the suite fails every `GC-*`/`H-*`/`E-*` test it can
      import (proves the tests detect the bugs).
- [ ] CI is green on the full matrix and takes < 5 min.
