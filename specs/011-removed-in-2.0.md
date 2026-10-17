# 011 — Removed in 2.0.0 (hard break, no shims)

**Status:** proposed · **Severity:** 🟡 policy · **Applies to:** every other spec

## Decision

2.0.0 is a **clean break**. Every 1.x name below is deleted outright:

- no deprecation warnings
- no `_compat.py`
- no legacy-kwarg mapping
- no `GeneOpt` re-export package
- no tuple-unpacking shim
- no import of 1.x caches

**Why:** 1.x has wrong results baked into its cache files (GC-1/2/3), so silently continuing a
1.x workflow is the outcome to prevent. A loud `TypeError` on the first line is better than a
run that seems to work. The library is small, and keeping the shims would roughly double the API surface that has to be tested (007).

**How removal behaves:**

- Removed kwargs → Python's own `TypeError: __init__() got an unexpected keyword argument 'r_cross'`.
  The signature is keyword-only (005 §1), so no custom code is needed.
- Removed modules → `ModuleNotFoundError: No module named 'GeneOpt'`.

## Removal table

### Package & modules

| 1.x | 2.0.0 |
|---|---|
| `import GeneOpt` / `from GeneOpt import …` | `import geneopt` (the distribution name on PyPI stays `GeneOpt`) |
| `GeneOpt/GeneticAlgorithm.py` | `geneopt/core.py` |
| `GeneOpt/GeneticAlgorithmCache.py` | `geneopt/cache.py` (`GenomeCache`, internal) |
| `GeneOpt/GeneticAlgorithmCacheType.py` | — deleted |
| `GeneOpt/GeneticAlgorithmComparisonType.py` | — deleted |
| `GeneOpt/CachePath.py` | — deleted |
| `GeneOpt/Util.py` | — deleted (the remaining helpers become private functions in the modules that use them) |

### Constructor parameters

| 1.x parameter | 2.0.0 |
|---|---|
| `start_number_of_population` | *deleted; use `population_size`* (014) |
| `number_of_population` | `population_size` |
| `number_of_generations` | `generations` |
| `comparison_type=GeneticAlgorithmComparisonType.maximize/minimize` | `maximize=True/False` |
| `r_cross` | `crossover_probability` |
| `tournament_selection_number` | `tournament_size` |
| `cache_type=GeneticAlgorithmCacheType.Ram` | `cache_dir=None` |
| `cache_type=GeneticAlgorithmCacheType.HardDisk` | `cache_dir="./cache"` |
| `cache_path=CachePath.MyOS / Colab` | `cache_dir="./cache"` / `cache_dir="/content/gdrive/MyDrive"` |
| `number_of_workers` | `executor=ThreadPoolExecutor(n)` (004) |
| `server_start_port` | — deleted; port pool lives in `examples/distributed_http/client.py` (004 §3) |
| `verbose` | stdlib logging: `logging.getLogger("geneopt").setLevel(...)` (005 §4) |
| `plot_DPI` | `ga.plot(dpi=150)` |
| `environment` (keyword, default `None`) | `environment` — required, 2nd positional |

### Methods & attributes

| 1.x | 2.0.0 |
|---|---|
| `ga.normal_start()`, `ga.concurrent_start()` | `ga.start()` only |
| `ga.on_end(*result)` | — deleted (`Result` + one INFO log line, 010) |
| `best, score = ga.start()` | `r = ga.start(); r.params, r.score` (010) |
| `ga.plot_best_score()`, `plot_parameter_values_for_the_best_scores()`, `plot_population_diversity()`, `plot_gene_frequencies()`, `feature_plots()` | `ga.plot(output_dir=…, dpi=…)`, or individual functions in `geneopt.plots` taking a `History` (002) |
| `ga._cache.ram_cache_score` (used by plots and by users) | `ga.history.to_frame()` (population per generation) / `ga.evaluations()` (unique evaluated points as a DataFrame) |
| `cache.get_other("count_of_run_objective")` | `r.n_evaluations` |
| `cache.get_other("best")` | `ga.result` |

### Public helpers

| 1.x export | 2.0.0 |
|---|---|
| `Logger`, `Logger.log_m`, `Logger.log_e`, `Logger.set_log_path` | — deleted; use `logging` |
| `to_datetime` | — deleted |
| `collect_results` | — deleted (`list(...)`) |
| `calculate_diversity` | `geneopt.plots.diversity(genomes)`, normalised to [0, 1] (002 §2) |
| `GeneticAlgorithmCache` | — not exported (internal `GenomeCache`) |

### Behaviour & files

| 1.x | 2.0.0 |
|---|---|
| Binary bit-string genomes | Integer index genomes (003). Changing the encoding is breaking for anyone reading `ga._population` |
| `{cache}/{name}.csv` + `{name}.json` | `{cache_dir}/{name}-{fingerprint}.jsonl`. Old files are **ignored**, not read or migrated |
| Plots in `./cache/0.png … N.png` | `{output_dir}/best_score.png`, … (002 §4) |
| `./cache/<timestamp>.log` written on every import + run | nothing written; logging is the application's job |
| Odd population silently decremented | kept as given (003 §4) |
| `catboost`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn` required | only `numpy`; the rest via `[plots]` / `[analysis]` extras (006) |
| Python ≥ 3.6 | Python ≥ 3.10 |

### Repo files

| 1.x | 2.0.0 |
|---|---|
| `main.py`, `main2.py` | `examples/quickstart.py`, `examples/distributed_http/client.py` |
| `server.py`, `RunNServers.py` | `examples/distributed_http/server.py`, `run_servers.py` (009) |
| `setup.py` | — deleted (`pyproject.toml` only) |
| `build/`, `dist/`, `GeneOpt.egg-info/`, `__pycache__/` | — deleted from git |
| `image/` | `docs/images/` |

## Acceptance criteria

- [ ] `grep -rnE "DeprecationWarning|_compat|_LEGACY" geneopt/` returns nothing.
- [ ] Every 1.x kwarg in the parameter table raises `TypeError` when passed to `GeneticAlgorithm`
      (one parametrised test in `test_api.py`).
- [ ] `import GeneOpt` raises `ModuleNotFoundError` once 2.0.0 is installed.
- [ ] `geneopt.__all__ == ["GeneticAlgorithm", "Result", "History", "__version__"]`.
- [ ] A 1.x `{name}.csv` in `cache_dir` is never opened (verify with `monkeypatch` on `open`).
