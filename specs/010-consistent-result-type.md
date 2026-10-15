# 010 — One Result Type for Every Run

**Status:** proposed · **Severity:** 🟠 high · **Fixes:** A-1, A-2 · **Depends on:** 004 · **Defines:** the `Result` from 005 §3

## Problem

What `start()` returns depends on the mode (verified):

```
sequential: dict ({'x': np.float64(1.0)}, np.float64(1.0))
concurrent: str  ("{'x': np.float64(1.0)}", np.float64(1.0))
```

`concurrent_start` ends with `self.on_end(str(self._decode(best_chromosome)), …)`
(`GeneticAlgorithm.py:446`), while `normal_start` passes the dict (`:374`). Effects:

- `best["x1"]` → `TypeError: string indices must be integers` in parallel mode only. A user's
  script works locally and then breaks when they switch to `number_of_workers=4`.
- The persisted `best` in `{name}.json` is a dict after one run and a string after another.
- `str(dict)` of numpy scalars gives `"np.float64(1.0)"` on NumPy 2, which can't be parsed back
  by `ast.literal_eval` or `json`. The parallel result can't be recovered in code.

Other inconsistencies at the same boundary:

| # | Where | Inconsistency |
|---|---|---|
| 10a | `:336` vs `:380-384` | Sequential seeds `best` by evaluating `population[0]`. Concurrent seeds it with `None`/`±inf`. If every score is `-inf` (maximize), concurrent returns `best=None` → `self._decode(None)` → `TypeError` |
| 10b | `:358-363` vs `:429-434` | Sequential checks `target_score` in the middle of a generation (later individuals are never evaluated). Concurrent checks after the whole generation |
| 10c | `:179` vs `:189` (A-2) | The concurrent counter is read once for the log line and incremented again from a second read, so log numbers repeat or skip under contention |
| 10d | return values | Plain numpy scalars (`np.float64`) and the decoded dict hold numpy types, so `json.dumps(best)` fails for int environments (GC-8) |

## Design

`start()` has one code path (004) and returns one type:

```python
@dataclass(frozen=True, slots=True)
class Result:
    params: dict[str, Any]       # decoded values, numpy scalars converted to Python (.item())
    score: float                 # Python float
    generation: int              # generation where `params` was first found
    n_evaluations: int           # unique objective calls in this run
    n_cache_hits: int            # individuals served from cache in this run
    elapsed: float               # seconds, perf_counter
    stopped_by: Literal["generations", "early_stopping", "target_score", "callback"]
```

- **No tuple unpacking.** `best, score = ga.start()` is removed in 2.0.0 (see 011). Use
  `r = ga.start(); r.params, r.score`.
- **Python-native values.** `params` values and `score` go through `_to_python`
  (`np.generic → .item()`, everything else unchanged), so `json.dumps(dataclasses.asdict(r))`
  always works (10d).
- **`best` is never `None`.** It's seeded from the first scored generation and not from a
  sentinel (10a). If every score is non-finite, the run still returns that individual, and
  `score` is `±inf`.
- **`target_score` is checked once per generation, after scoring** (10b), so the history (002)
  never has a half-scored generation.
- **Counters are only touched in the main thread**, after `_evaluate` returns (10c).
- `ga.result` holds the same object after `start()`. `on_end()` is removed (011). The
  end-of-run summary is one `INFO` log line built from `Result`.

## Acceptance criteria

- [ ] For `executor ∈ {None, ThreadPoolExecutor(4), ProcessPoolExecutor(2)}`, `start()` returns
      `Result`, `type(r.params) is dict`, and `type(r.score) is float`.
- [ ] Same seed → `r_seq == r_par` except for `elapsed` (compare with `dataclasses.replace(r, elapsed=0)`).
- [ ] `json.dumps(dataclasses.asdict(r))` succeeds for float, `np.arange`, and categorical
      string environments.
- [ ] An objective that always returns `-inf` (maximize) gives a `Result` with `score == -inf`
      and a valid `params`, not a crash.
- [ ] With `target_score` hit at individual 3 of 20, all 20 are still scored and present in
      `history.scores[g]`.
- [ ] `best, score = ga.start()` raises `TypeError` (not iterable). This is intentional (011).
