# GeneOpt Quality Specs

These are the improvement specs for GeneOpt `1.0.10`. Every defect below was **reproduced
against the current code** (probe scripts are inlined in each spec) unless marked *(read)*.

## The headline bug

> **The evaluation cache is also used as the run's history, and it can't do both jobs.**

`GeneticAlgorithmCache.ram_cache_score` is a memo table (genome → score) **and** the only
record of what happened in each generation. A cache hit returns early (`GeneticAlgorithm.py:114-118`)
and writes nothing, so a genome shows up only in the generation where it was *first* evaluated.

In one 50-generation run with population 100 (5,000 individual slots), the "history" held
**105 rows**. Some generations had no rows at all. Every plot is computed from that data, so
the best-score curve, diversity, gene frequencies and feature analysis all describe the wrong
thing. On top of that, the cache can return the wrong score for a genome (false hits), it
returns stale scores across runs, and it can't store non-numeric parameters.
→ [001-genome-cache.md](001-genome-cache.md), [002-history-and-plots.md](002-history-and-plots.md)

## Spec map

| # | Spec | Theme | Severity |
|---|------|-------|----------|
| 001 | [Genome cache](001-genome-cache.md) | Exact keys, run fingerprint, dedupe, persistence | 🔴 critical |
| 002 | [History & plots](002-history-and-plots.md) | Separate history from the cache, fix every plot | 🔴 critical |
| 003 | [Encoding & operators](003-encoding-and-operators.md) | Decode bias, mutation maths, elitism, selection | 🟠 high |
| 004 | [Evaluation engine](004-evaluation-engine.md) | One loop, any executor, error policy | 🟠 high |
| 005 | [Public API & logging](005-public-api-and-logging.md) | Return types, validation, stdlib logging, paths | 🟡 medium |
| 006 | [Packaging & repo hygiene](006-packaging-and-repo.md) | pyproject, extras, committed build output, README | 🟡 medium |
| 007 | [Test & CI plan](007-tests-and-ci.md) | Regression tests for every bug ID, CI matrix | 🟡 medium |
| 008 | [HTTP client zero-score hang](008-http-client-zero-score-hang.md) | `0.0` score → silent infinite request loop | 🔴 critical |
| 009 | [Example server hardening](009-example-server-hardening.md) | `debug=True`, unchecked params, process leaks | 🟠 high |
| 010 | [Consistent result type](010-consistent-result-type.md) | `str` vs `dict` best params, one `Result` | 🟠 high |
| 011 | [Removed in 2.0.0](011-removed-in-2.0.md) | Hard break: every 1.x name deleted, no shims | 🟡 policy |

## Bug ledger

IDs are referenced across specs. "Verified" means it was reproduced with a script.

| ID | Where | Defect | Verified |
|----|-------|--------|----------|
| GC-1 | `GeneticAlgorithmCache.py:49` | `np.isclose` default `rtol=1e-5` → `x=1_000_005` hits cached `x=1_000_000` | ✅ |
| GC-2 | `GeneticAlgorithmCache.py:43-46` | Lookup only checks the query's columns, so after a parameter is dropped an old row matches on the remaining subset | ✅ |
| GC-3 | `GeneticAlgorithmCache.py:21-29` | HardDisk cache is keyed only by `optimizer_name`. A changed objective or environment silently gets old scores | ✅ |
| GC-4 | `GeneticAlgorithmCache.py:71-72` | `set_score` mutates the caller's dict (adds `score`, `generation`) | ✅ |
| GC-5 | `GeneticAlgorithmCache.py:71` | A parameter named `score`/`generation` is overwritten, so that genome never gets a cache hit | ✅ |
| GC-6 | `GeneticAlgorithmCache.py:25,49` | Categorical environments (`["relu","tanh"]`) crash: `DTypePromotionError` | ✅ |
| GC-7 | `GeneticAlgorithm.py:402-410` | Concurrent mode queues duplicate genomes in a generation: 20 identical genomes → 20 objective calls | ✅ |
| GC-8 | `GeneticAlgorithm.py:449` + `Util.py:217` | Integer environment + HardDisk → `TypeError: int64 is not JSON serializable` in `on_end` | ✅ |
| GC-9 | `GeneticAlgorithmCache.py:74`, `:41-49` | O(n) `concat` per insert plus a linear-scan lookup → O(n²) per run | *(read)* |
| GC-10 | `GeneticAlgorithmCache.py:84`, `Util.py:217` | The whole JSON is rewritten on every evaluation, non-atomically. A crash mid-write corrupts the file | *(read)* |
| GC-11 | `GeneticAlgorithmCache.py:33` | Always creates `./cache`, even when `cache_path=Colab` | *(read)* |
| H-1 | `GeneticAlgorithm.py:114-118` | Cache hits aren't recorded, so the history misses ~98% of individuals | ✅ |
| H-2 | `GeneticAlgorithm.py:551` | Diversity plot compares decoded floats *and the score/generation columns* | *(read)* |
| H-3 | `GeneticAlgorithm.py:565-569` | "Gene frequencies" are min-max-scaled parameter means, not gene frequencies. Constant column → NaN | *(read)* |
| H-4 | `GeneticAlgorithm.py:584` | Only `-inf` is dropped. Minimize runs keep `+inf` and break correlation/regression | *(read)* |
| H-5 | `GeneticAlgorithm.py:476-498` | `raise e` placed before `Logger.log_e(e)` makes the error handling dead code | *(read)* |
| E-1 | `GeneticAlgorithm.py:213-214` | Clamping overflow → last value over-sampled. 5 values: `e` drawn 50% of the time | ✅ |
| E-2 | `GeneticAlgorithm.py:313-316` | Two nested `rate` gates → effective per-bit rate is `rate²` (0.06 → 0.0036) | *(read)* |
| E-3 | `GeneticAlgorithm.py:321` | Rate is divided by *unnormalised* Hamming distance, so more diversity means *less* mutation, and the rate can exceed 1 | ✅ |
| E-4 | `GeneticAlgorithm.py:242-246` | Tournament draws k+1 candidates. If k > population it logs an error on every selection | *(read)* |
| E-5 | — | No elitism: the best genome can drop out of the population | *(read)* |
| E-6 | `GeneticAlgorithm.py:340` | `early_stopping_rounds=N` stops after N+1 rounds | *(read)* |
| A-1 | `GeneticAlgorithm.py:446` vs `:374` | Concurrent `start()` returns `str(params)`. Sequential returns a `dict` → 010 | ✅ |
| A-2 | `GeneticAlgorithm.py:179,189` | Concurrent run counter is read and incremented separately, so log numbers are duplicated or skipped → 010 | *(read)* |
| A-3 | `Util.py:180` | Logger picks its file path at import time and always writes to `./cache/` | *(read)* |
| A-4 | `GeneticAlgorithm.py:12` | `catboost` is a hard import, but the README says it's optional | *(read)* |
| X-1 | `main2.py:21`, README | `if res_json.get("data")` is falsy for a score of `0.0`, so the client loops forever with no sleep and no log → 008 | ✅ |
| X-2 | `server.py:66`, README | `debug=True`: Werkzeug debugger console (PIN-protected) + reloader doubles processes. The danger comes when users bind `0.0.0.0` for multi-machine use → 009 | *(read)* |
| X-3 | README | Says `RunNServer.py`, but the file is `RunNServers.py` → 009 | *(read)* |

## Suggested order

```
011 removal policy (decided up front: no shims anywhere)
 │
001 cache ──► 002 history ──► 004 engine ──► 010 result ──► 003 operators ──► 005 API ──► 006 packaging
                                  └──► 008 client + 009 server (examples)
   └──────────────────── 007 tests land alongside each step (red test first) ────────────────────┘
```

Everything ships as **2.0.0**, a hard break. Every 1.x parameter, module, enum and helper is
deleted, with no deprecation period (see [011](011-removed-in-2.0.md)). 003 changes search
behaviour, so ship it with before/after benchmark numbers (see 007).
