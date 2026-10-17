# 001 — Genome Cache

**Status:** proposed · **Severity:** 🔴 critical · **Fixes:** GC-1 … GC-11 · **Pairs with:** 002

## Problem

The cache answers "have I already evaluated this parameter set?" by searching a pandas
DataFrame of **decoded float values** with an approximate comparison (`np.isclose`). That
choice causes most of the problems:

| Symptom | Root cause |
|---|---|
| Wrong score returned (GC-1) | `np.isclose(a, b, atol=1e-8)` keeps the default `rtol=1e-5`, so values within 0.001% are treated as equal |
| Stale score after env change (GC-2) | Lookup only checks columns in the query |
| Stale score after objective change (GC-3) | The disk file is named only by `optimizer_name` |
| Categorical params crash (GC-6) | Float comparison and `dtype=np.float64` on load |
| `score`/`generation` params break (GC-5) | Bookkeeping columns share a namespace with parameters |
| Caller's dict mutated (GC-4) | `data["score"] = score` |
| Duplicate work in parallel (GC-7) | Cache checked before dispatch, never during it |
| O(n²) run time (GC-9) | `concat` + linear scan per call |
| Corrupt JSON on crash (GC-10) | Full non-atomic rewrite per evaluation |

### Reproduction

```python
from GeneOpt.GeneticAlgorithmCache import GeneticAlgorithmCache
from GeneOpt.GeneticAlgorithmCacheType import GeneticAlgorithmCacheType as T
from GeneOpt.CachePath import CachePath

c = GeneticAlgorithmCache("p", T.Ram, CachePath.MyOS)
c.set_score({"x": 1_000_000.0}, 1.0, 0)
c.get_score({"x": 1_000_005.0})          # -> 1.0   (GC-1, expected None)

c.set_score({"x1": 1.0, "x2": 2.0, "x3": 3.0}, 42.0, 0)
c.get_score({"x1": 1.0, "x2": 2.0})      # -> 42.0  (GC-2, expected None)

d = {"x": 1.0}; c.set_score(d, 9.0, 0); d  # -> {'x':1.0,'score':9.0,'generation':0} (GC-4)
```

HardDisk staleness (GC-3): run `maximize x` and then `maximize -x` with the same
`optimizer_name`. The second run returns `x=1.0, score=1.0`, which is the first run's answer.

## Design

### 1. Key on gene indices, not decoded values

A genome already *is* a tuple of indices into the environment lists. Using that as the key:

- makes equality exact (no tolerance, GC-1 gone)
- works for any value type: strings, objects, `None` (GC-6 gone)
- turns lookup into an O(1) `dict` hit (GC-9 gone)
- keeps parameter names out of the key namespace (GC-5 gone)

```python
Key = tuple[int, ...]          # one index per parameter, in sorted-parameter order

class GenomeCache:
    def __init__(self, path: Path | None, fingerprint: str): ...
    def get(self, key: Key) -> float | None
    def put(self, key: Key, score: float) -> None
    def __contains__(self, key: Key) -> bool
    def __len__(self) -> int
```

That's the whole interface. No generation and no "other" bag: run metadata lives on the
optimizer (005) and per-generation data lives in the history (002).

> Two bit patterns that clamp to the same index (E-1) correctly share one key today. Spec 003
> removes the clamping entirely, but the cache doesn't depend on that.

### 2. Run fingerprint, so stale scores can't leak (GC-2, GC-3)

A persisted score is only valid for the same **search space** and the same **objective**.

```python
fingerprint = sha256(json.dumps({
    "env": {name: [repr(v) for v in values] for name, values in sorted(env.items())},
    "objective": f"{fn.__module__}.{fn.__qualname__}",
    "objective_code": code_digest(fn.__code__) if has_code else None,  # see below
    "objective_version": objective_version,     # new optional user param, str
}, sort_keys=True)).hexdigest()[:16]
```

- Cache file: `{cache_dir}/{optimizer_name}-{fingerprint}.jsonl`. A changed environment or
  function body gets a fresh file automatically. Old files stay on disk and nothing is deleted.
- `code_digest` hashes `co_code`, `co_names` and `co_consts`, and recurses into nested code objects
  instead of using their `repr`. The `repr` of a code object embeds its memory address, so a naive
  `repr(co_consts)` gives a different fingerprint on every run whenever the objective contains a
  lambda or inner function. `co_names` is needed because `np.sin(x)` and `np.cos(x)` compile to
  the same `co_code` and differ only in the names they reference.
- `co_code` only covers the function's own body, not functions it calls. For those cases,
  `objective_version="2"` is the documented manual switch.
- Bytecode differs between Python versions, so upgrading Python means a cold cache. That's a
  harmless miss, never a wrong hit, and it's documented.
- Functions without `__code__` (partials, callable objects, the HTTP wrapper) fall back to
  qualname plus `objective_version`, with a one-time log warning that edits aren't detected.

### 3. On-disk format: append-only JSONL

```jsonl
{"geneopt_cache": 2, "fingerprint": "9b1c…", "params": ["x1","x2"], "created": "2026-10-15T12:00:00Z"}
{"k": [812, 3301], "s": -19.2085, "v": [-6.03, 6.12]}
{"k": [4095, 17],  "s": -0.4410,  "v": [10.0, -9.92]}
```

- Line 1 is the header. If its `fingerprint` doesn't match on load, raise
  `CacheMismatchError` (this can only happen if the file was renamed by hand).
- `v` (decoded values) is only there so people can read the file. It's serialised with
  `default=_jsonable` (`np.generic → .item()`, anything else → `repr`), which fixes GC-8.
- One `write()` + `flush()` per record, done by appending. No full rewrite, so GC-10 goes away.
- On load, a truncated last line (crash mid-write) is skipped with a warning **and cut from the
  file**. Otherwise the next append would join the partial line and corrupt a middle record.
  Every other line must parse. An empty file (crash while writing the header) is re-initialised.
- Why JSONL and not CSV/pickle/sqlite: it's stdlib, append-safe, mixed-type, readable with
  `pandas.read_json(lines=True)`, and has no schema migrations. sqlite is the upgrade path if
  caches go past ~10⁷ rows.

### 4. Dedupe within a generation (GC-7)

The evaluation engine (004) builds `pending = {key: genome}` for the cache misses of a
generation. Using a dict *is* the dedupe: each unique key is evaluated once, then the result is
fanned back to every individual that carries it. Sequential and parallel modes use the same
code path, so neither can regress separately.

### 5. Scores and failures

- `score` must be a real number (`float(score)` must succeed). `NaN` is rejected with a clear
  error naming the parameters, because `NaN` makes every comparison False and quietly freezes
  selection.
- `±inf` is allowed (it's a valid "worst" value).
- Exceptions are **not** cached by default (they may be transient). See 004 `on_error`.

### 6. What gets deleted

- `GeneticAlgorithmCacheType` → replaced by `cache_dir: str | PathLike | None = None`.
  `None` keeps the cache in memory only.
- `CachePath` enum (`MyOS`/`Colab`) → replaced by the same `cache_dir`. The Colab entry was
  just a hard-coded path.
- `ram_cache_other`, `get_other`/`set_other` → the counter becomes an attribute, `best` goes
  into the result (005).
- `read_json_file`, `write_json_file`, `is_file_exist`, `is_folder_exist`, `read/write_pickle_file`,
  `copy_and_replace_file` in `Util.py` → unused after this change (`pathlib` + `json` cover the rest).

Expected size: `GenomeCache` ≈ 50 lines, down from 117.

## Breaking changes (2.0.0, no shims, see 011)

- `cache_type=`, `cache_path=`, `GeneticAlgorithmCacheType` and `CachePath` are **removed**.
  Passing them raises `TypeError`. Use `cache_dir`.
- Old `{name}.csv` / `{name}.json` caches are **ignored**: never opened, never imported. They
  have no fingerprint and so can't be trusted (GC-3). Release notes say so explicitly.

## Acceptance criteria

- [ ] GC-1: values `1_000_000` and `1_000_005` are distinct keys.
- [ ] GC-2/GC-3: changing env values, adding/removing a parameter, or editing the objective
      body gives a different cache file, and the second run makes ≥1 objective call.
- [ ] GC-4: the dict passed to the objective is never mutated afterwards.
- [ ] GC-5: an environment with parameters named `score` and `generation` round-trips.
- [ ] GC-6: `{"act": ["relu", "tanh", "gelu"], "lr": [1e-3, 1e-2]}` runs with a disk cache,
      and a reload gets hits.
- [ ] GC-7: a population of 20 identical genomes makes exactly **1** objective call, in both modes.
- [ ] GC-8: `np.arange` / `np.int64` environments persist without errors.
- [ ] GC-10: truncating the last byte of a cache file still loads and warns once.
- [ ] GC-9: 10⁵ `put` + 10⁵ `get` take < 1 s.
- [ ] Rerunning the same seed + fingerprint against a warm disk cache makes **0** objective calls
      and returns the same best result.
