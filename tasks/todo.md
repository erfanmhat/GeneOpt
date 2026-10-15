# GeneOpt 2.0.0 — Task List

Tests: `python -m pytest -q` · Build: `pip wheel . --no-deps -w <scratch>` (the `build` tool isn't installed. pip builds in isolation)

## Task 1: Packaging skeleton + repo hygiene
**Description:** Add a PEP 621 `pyproject.toml` (hatchling, numpy-only core, `[plots]`/`[analysis]`/`[dev]` extras, Python ≥ 3.10), a `geneopt/__init__.py` with `__version__ = "2.0.0"`, and `tests/`. Delete `setup.py`, `build/`, `dist/`, `GeneOpt.egg-info/` and all tracked `__pycache__/` from git. Extend `.gitignore`.
**Acceptance criteria:**
- [x] `git ls-files | grep -E '(^build/|^dist/|egg-info|__pycache__|\.pyc$|^setup\.py$)'` is empty
- [x] `import geneopt; geneopt.__version__ == "2.0.0"`, and importing it creates no files and doesn't import matplotlib/pandas
- [x] `python -m build` produces a wheel that contains `geneopt/`
**Verification:** `pytest tests/test_api.py` · `python -m build`
**Dependencies:** None
**Files:** `pyproject.toml`, `.gitignore`, `geneopt/__init__.py`, `tests/test_api.py`, deletions
**Scope:** S

## Task 2: GenomeCache
**Description:** `geneopt/cache.py`: `GenomeCache(path|None, fingerprint)` with dict storage keyed by index tuples. `fingerprint(env, fn, objective_version)`. Append-only JSONL (header line + one record per put, `.item()` for numpy). A truncated last line is skipped with a warning. Fingerprint mismatch → `CacheMismatchError`. 1.x CSV/JSON files are never opened. NaN scores rejected.
**Acceptance criteria:**
- [x] GC-1 (1_000_000 vs 1_000_005 distinct), GC-5 (params named score/generation), GC-6 (string values), GC-8 (np.int64) round-trip through disk
- [x] A changed env / objective body / `objective_version` → different fingerprint (GC-2/GC-3). Truncated tail loads with one warning (GC-10)
- [x] 10⁵ put+get < 1 s (GC-9)
**Verification:** `pytest tests/test_cache.py`
**Dependencies:** 1
**Files:** `geneopt/cache.py`, `tests/test_cache.py`
**Scope:** S

## Checkpoint A
- [x] `pytest` green, `python -m build` OK

## Task 3: History + Result
**Description:** `geneopt/history.py`: `History` (per-generation genomes/scores/hits arrays, `append`, `to_frame(env)` with lazy pandas). `Result` frozen dataclass in `geneopt/core.py` (010), with Python-native values via `_to_python`.
**Acceptance criteria:**
- [x] `History.to_frame` returns one row per individual per generation, with a `cache_hit` column
- [x] `json.dumps(dataclasses.asdict(Result(...)))` works for numpy values. `Result` isn't iterable
**Verification:** `pytest tests/test_history.py tests/test_result.py`
**Dependencies:** 1
**Files:** `geneopt/history.py`, `geneopt/core.py`, tests
**Scope:** S

## Task 4: GeneticAlgorithm — sequential run
**Description:** `GeneticAlgorithm(objective_func, environment, *, …)` with the 005 §1 signature minus `executor`/`on_error` (Task 5). Construction-time validation (005 §2). Integer genomes, uniform init, crossover by gene mask, single-gate mutation with creep for ordered numeric params, adaptive rate from normalised diversity, exact-k tournament, elitism, `sign` comparisons, early stopping `>=`, `target_score` checked per generation (003, 010). The loop goes through the cache with in-generation dedupe and appends to the history. `start()` returns `Result` and sets `ga.result`/`ga.history`.
**Acceptance criteria:**
- [x] Every 1.x kwarg in 011 raises `TypeError`. Bad inputs raise with the parameter name
- [x] 20 identical genomes → 1 objective call (GC-7). The history has pop rows for every generation (H-1). Init over 5 values is uniform (E-1)
- [x] Convergence: a 2-D bowl optimum and a categorical toy optimum are found within budget. Elitism → monotonic best. `population_size=101` is kept. Warm disk cache rerun → 0 calls, same result
**Verification:** `pytest tests/test_core.py tests/test_api.py`
**Dependencies:** 2, 3
**Files:** `geneopt/core.py`, `geneopt/__init__.py`, `tests/test_core.py`, `tests/test_api.py`
**Scope:** M

## Task 5: Executor, on_error, logging
**Description:** `executor=` (any `concurrent.futures.Executor`, `map` in submission order) and `on_error="raise"|"worst"|"retry"` + `max_retries` (004 §4). Stdlib `logging.getLogger("geneopt")`: INFO per generation and at the end, DEBUG per evaluation. No handlers are added.
**Acceptance criteria:**
- [ ] Same seed → `Result`/history identical with `executor=None` and `ThreadPoolExecutor(4)`
- [ ] `worst`: failures scored as worst, not cached, and warned. `raise`: rows evaluated before the failure are on disk
- [ ] A full run with logging at WARNING emits no records
**Verification:** `pytest tests/test_engine.py`
**Dependencies:** 4
**Files:** `geneopt/core.py`, `tests/test_engine.py`
**Scope:** S

## Checkpoint B
- [ ] `pytest` green. 011 criteria 1, 2, 4, 5 pass

## Task 6: Plots + evaluations()
**Description:** `geneopt/plots.py`: `diversity(genomes)` in [0,1], plus best-so-far, best params, diversity, gene-frequency heatmap, cache-hit rate and lazily-imported feature analysis (finite-only, numeric-only, catboost optional). `ga.plot(output_dir=None, dpi=150, show=False)` → `{name: path|Exception}`, with descriptive filenames under `{cache_dir}/{name}-plots/` or `./{name}-plots/`. `ga.evaluations()` → DataFrame of unique evaluated points.
**Acceptance criteria:**
- [ ] `diversity` is 0 for identical genomes and in [0,1] in general. Gene-frequency rows sum to 1
- [ ] `ga.plot()` writes non-empty PNGs with the expected names. A minimize run with `+inf` scores still produces the feature plots
- [ ] A missing catboost only drops the analysis plots, with an `ImportError` recorded in the result dict
**Verification:** `MPLBACKEND=Agg pytest tests/test_plots.py`
**Dependencies:** 4
**Files:** `geneopt/plots.py`, `geneopt/core.py`, `tests/test_plots.py`
**Scope:** M

## Task 7: Examples
**Description:** `examples/quickstart.py` (Holder table), `examples/categorical.py`, and `examples/distributed_http/{objective,server,client,run_servers}.py` per 008/009 (stdlib client with bounded retries and 0.0-safe; server with `debug=False` and signature-checked JSON params; `run_servers.py` using `sys.executable`). Move `image/` → `docs/images/`.
**Acceptance criteria:**
- [ ] Client against a stdlib stub: `0.0` returned after 1 request. 400 → `ObjectiveError` after 1 request. 503,503,200 → score after 3 requests
- [ ] Server param validation rejects missing/unknown params and decodes JSON values (tested as a pure function, no Flask needed)
- [ ] `quickstart.py` and `categorical.py` run in tests with tiny budgets. There's no `debug=True` anywhere
**Verification:** `pytest tests/test_examples.py`
**Dependencies:** 5
**Files:** `examples/**`, `tests/test_examples.py`, `docs/images/*` (moved)
**Scope:** M

## Task 8: Delete 1.x, README, CHANGELOG
**Description:** Delete `GeneOpt/`, `main.py`, `main2.py`, `server.py`, `RunNServers.py`. Rewrite the README for the 2.0 API (no before/after). Add a `CHANGELOG.md` 2.0.0 entry that links to spec 011.
**Acceptance criteria:**
- [ ] `import GeneOpt` → `ModuleNotFoundError`. `grep -rnE "DeprecationWarning|_compat|_LEGACY" geneopt/` is empty
- [ ] Every code block in the README is either an embedded example file or runs as written
- [ ] `python -m build && twine check dist/*` pass
**Verification:** full `pytest` · build · twine check
**Dependencies:** 6, 7
**Files:** deletions, `README.md`, `CHANGELOG.md`
**Scope:** M

## Checkpoint C
- [ ] All spec 011 acceptance criteria met. `pytest` green. Build clean
