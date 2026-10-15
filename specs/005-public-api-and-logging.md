# 005 — Public API, Validation & Logging

**Status:** proposed · **Severity:** 🟡 medium · **Fixes:** A-3 · **Related:** 010 (A-1), 011 (removals) · **Depends on:** 001, 004

## Problem

- **Bad input fails late or silently.** `environment=None` crashes at
  `for key in environment` with a `TypeError`. `r_cross=5` is accepted, an empty value list
  gives a 0-bit gene, `tournament_selection_number=0` makes selection return a random parent,
  and `comparison_type="maximize"` (a string) raises a bare `Exception` with a tuple message.
- **Required parameters are inconsistent.** `start_number_of_population` is required but
  `number_of_population` defaults to `None → 100` with a log line. Most users set both to the
  same value (every example does).
- **Return values depend on the mode** (A-1), and there's no way to get the run statistics
  except by reading the log.
- **Logging is a hand-rolled class with import-time side effects** (A-3):
  `Logger.log_path = "./cache/<unix-ts>.log"` is fixed when the module is imported, it always
  writes to `./cache/` whatever `cache_path` is, and failures are `print`ed. `verbose` is an
  int from -1 to 3 that's checked with `>=` in about 30 places.
- **Package surface:** `__init__.py` imports `GeneticAlgorithm` twice, re-exports
  `GeneticAlgorithmCache` from the wrong module, doesn't export `CachePath` (which is a
  constructor argument), and exports `to_datetime`/`collect_results`, which are internal.
- Module names are `PascalCase.py` (`GeneticAlgorithm.py`), which isn't PEP 8. The class
  names can stay.

## Design

### 1. Constructor

```python
GeneticAlgorithm(
    objective_func: Callable[..., float],
    environment: Mapping[str, Sequence[Any]],         # now required, positional
    *,
    optimizer_name: str = "geneopt",
    maximize: bool = True,                           # replaces comparison_type enum
    population_size: int = 100,
    initial_population_size: int | None = None,      # None → population_size
    generations: int = 100,
    crossover_rate: float = 0.9,
    mutation_rate: float | None = None,             # None → 1/n_params (003)
    tournament_size: int = 3,
    elitism: int = 1,
    early_stopping_rounds: int | None = None,
    target_score: float | None = None,
    cache_dir: str | PathLike | None = None,
    objective_version: str | None = None,
    executor: Executor | None = None,
    on_error: Literal["raise", "worst", "retry"] = "raise",
    max_retries: int = 2,
    seed: int | None = 0,
)
```

- Everything after `environment` is keyword-only, so arguments can't be passed in the wrong
  order by accident.
- Every 1.x parameter name is **removed** with no alias (011). Because of the keyword-only
  signature, passing one raises Python's own `TypeError: unexpected keyword argument`, with
  no compat code at all.

### 2. Validation at construction

All checks raise `ValueError`/`TypeError` with the parameter name and the offending value:

| Check | Rule |
|---|---|
| environment | non-empty mapping. Every value is a non-empty sequence (not a str). Keys are `str.isidentifier()` because they're passed as `**kwargs` |
| rates | `0 ≤ crossover_rate, mutation_rate ≤ 1` |
| sizes | `population_size ≥ 2`, `tournament_size ≥ 1`, `0 ≤ elitism < population_size` |
| generations | `≥ 1` |
| objective | `callable`. Optionally, `inspect.signature(fn).bind(**first_params)` is tried once, so a misspelled parameter name fails **before** generation 0 |

### 3. Result

`start()` returns a frozen `Result` dataclass, the same type in every mode. Its full contract,
including Python-native values and the removal of tuple unpacking, is in
[010](010-consistent-result-type.md). `ga.history` (002) and `ga.result` stay attached to the
instance after `start()`.

### 4. Logging: stdlib `logging`

```python
log = logging.getLogger("geneopt")
```

- No handlers, files or directories are created by the library. That's the application's job.
  This deletes the `Logger` class, `convertVarArgToString`, and the import-time path.
- Levels: generation summary → `INFO`, each evaluation → `DEBUG`, cache hit → `DEBUG`,
  swallowed errors (`on_error="worst"`) → `WARNING`.
- `verbose` is removed (011). Verbosity is set the standard way:
  `logging.getLogger("geneopt").setLevel(logging.DEBUG)`.
- Optional convenience: `geneopt.enable_console_logging(level="INFO")` (3 lines, for notebooks).
- Lazy formatting (`log.debug("eval %s -> %s", params, score)`), so DEBUG costs nothing when off.

### 5. Optional progress hook

`callback: Callable[[GenerationStats], bool | None] | None = None`. It's called once per
generation, and returning `True` stops the run. That covers tqdm bars, custom stopping,
checkpoint-to-cloud and live dashboards without the library owning any of them. It's one
`if` in the loop. Skip it if nobody asks.

### 6. Module layout

```
geneopt/
  __init__.py        # GeneticAlgorithm, Result, History, __version__
  core.py            # GeneticAlgorithm + operators     (~250 lines, from 684)
  cache.py           # GenomeCache + fingerprint        (~70)
  history.py         # History                           (~50)
  plots.py           # plot functions on History/cache  (~200, matplotlib imported lazily)
```

- Package import name `geneopt` (lowercase, PEP 8). The `GeneOpt` package is deleted, with no
  re-export shim (011). The PyPI distribution name stays `GeneOpt`.
- `matplotlib` (and `catboost` for the analysis plots) are imported inside `plots.py` functions, so `import geneopt` stays
  fast and works headless.
- Add a `py.typed` marker and full annotations.

## Acceptance criteria

- [ ] Every bad input in the validation table raises with the parameter name in the message.
- [ ] A misspelled environment key vs objective signature fails before any evaluation.
- [ ] `ga.start()` returns a `Result`, and `ga.start().n_evaluations` is an int (010).
- [ ] `import geneopt` creates no files or directories and does not import matplotlib
      (`"matplotlib" not in sys.modules`).
- [ ] With the application logging at WARNING, a full run prints nothing.
- [ ] Every 1.0.10 kwarg raises `TypeError` (011). `geneopt/` contains no `DeprecationWarning`.
