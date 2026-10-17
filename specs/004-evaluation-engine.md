# 004 — Evaluation Engine (one loop, any executor)

**Status:** proposed · **Severity:** 🟠 high · **Fixes:** GC-7 · **Related:** 008 (X-1), 010 (A-1, A-2) · **Depends on:** 001

## Problem

`normal_start` (43 lines) and `concurrent_start` (71 lines) are two copies of the same GA loop
that have drifted apart:

| Behaviour | `normal_start` | `concurrent_start` |
|---|---|---|
| Initial best | evaluates `population[0]` first | starts at `±inf`, `None` |
| Return value | `(dict, score)` | `(str(dict), score)` (A-1) |
| Target score check | mid-generation | after the whole generation |
| Duplicate genomes | cached after the 1st | **all evaluated** (GC-7) |
| Objective signature | `f(**params)` | `f(**params, port=…)` |
| Run counter | read+write | read, then a separate read+write (A-2) |

The concurrent path also has its own oddities:
- A `queue.Queue` (already thread-safe) is wrapped in an extra `threading.Lock`, with a
  `while not q.empty()` / `get_nowait` race that the lock is there to cover.
- `yield` happens **while holding the lock**. It only works because `collect_results` consumes
  the generator straight away in the same thread.
- One exception from the objective kills that worker's generator, and `future.result()`
  re-raises it, so a single failed HTTP call aborts the whole run.
- Parallelism is tied to "HTTP servers on consecutive ports". Local multiprocessing, Dask,
  Ray or a cluster all have to fake a `port` kwarg.

## Design

### 1. One loop

```python
for gen in range(G):
    keys    = [tuple(g) for g in pop]
    pending = {k: g for k, g in zip(keys, pop) if k not in cache}   # dedupe (GC-7)
    for k, s in zip(pending, self._evaluate(list(pending.values()))):
        cache.put(k, s)
    scores = np.array([cache.get(k) for k in keys])
    history.append(pop, scores, hits=[k not in pending for k in keys])
    … stop checks → elitism → selection → crossover → mutation …
```

The sequential and parallel paths differ only in `_evaluate`.

### 2. `_evaluate` takes a standard Executor

```python
GeneticAlgorithm(..., executor: concurrent.futures.Executor | None = None)
```

- `None` → plain list comprehension, in order, in-process.
- Any `Executor` (thread, process, `dask.distributed.Client.get_executor()`, `loky`…) →
  `executor.map(call, genomes)`. That's stdlib, with no new dependency and no new abstraction.
- Results come back in submission order, so the RNG stream doesn't depend on timing.
  **Same seed means the same result sequentially and in parallel** (acceptance test).
- The library never creates or shuts down a user-supplied executor.

### 3. Port-based workers become an example, not a feature

The current HTTP setup is a 15-line pattern in user code:

```python
ports = queue.Queue(); [ports.put(5000 + i) for i in range(n)]
def objective(**p):
    port = ports.get()
    try:    return http_score(port, p)
    finally: ports.put(port)

GeneticAlgorithm(objective, ..., executor=ThreadPoolExecutor(n))
```

- `number_of_workers` and `server_start_port` are **removed** (011), and passing them raises
  `TypeError`. The pattern above lives in `examples/distributed_http/client.py` (006, 008).

### 4. Error policy

`on_error: Literal["raise", "worst", "retry"] = "raise"`, `max_retries: int = 2`

| Policy | Behaviour |
|---|---|
| `raise` | Re-raise after **finishing the in-flight batch** and flushing the cache, so finished work is never lost |
| `worst` | Score as `-inf` (maximize) / `+inf` (minimize), log a warning with params + exception, **don't cache** |
| `retry` | Retry with backoff `0.5·2^n s` up to `max_retries`, then fall back to `raise` |

`KeyboardInterrupt` always propagates immediately, after the cache flush.

### 5. Counters & timing

- `self.n_evaluations`, `self.n_cache_hits` are plain ints, incremented in the main thread
  only, so A-2 can't happen.
- Timing uses `time.perf_counter()`. The `to_datetime(time.time(), unit="s")` dance
  (`Util.py:152`) is deleted.
- The "10^x times faster than Blind Algorithm" line compares against **unique evaluations in
  this run**, not the cumulative cross-run JSON counter. Guard `n_evaluations == 0`
  (fully warm cache), which is currently a division by zero in `on_end`.

### 6. The shipped client (X-1)

Covered in full by [008](008-http-client-zero-score-hang.md).

## Acceptance criteria

- [ ] `normal_start` / `concurrent_start` are replaced by a single loop. `start()` is the only
      public entry point.
- [ ] Same seed → identical `history` with `executor=None` and with `ThreadPoolExecutor(4)`.
- [ ] 20 identical genomes → 1 objective call with or without an executor.
- [ ] `on_error="worst"`: a run whose objective raises on 10% of inputs finishes, logs 10%
      warnings and never caches those keys.
- [ ] `on_error="raise"` with a disk cache: rows evaluated before the failure are on disk.
- [ ] `start()` returns the same `Result` type in every mode (010).
