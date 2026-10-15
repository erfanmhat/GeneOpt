# Implementation Plan: GeneOpt 2.0.0 (spec 011, hard break)

## Overview

Spec 011 deletes every 1.x name and points each one at its 2.0.0 replacement. So implementing
011 means building those replacements: the `geneopt` package with `GenomeCache` (001),
`History` (002), integer genomes and fixed operators (003), one evaluation loop with an
executor (004), keyword-only API and stdlib logging (005), packaging (006), the example
client and server (008/009) and `Result` (010). Then the old `GeneOpt/` package and the 1.x
repo files are deleted.

The new package is built **next to** the old one, so every task leaves a working tree. The old
package is deleted only in the final task.

## Architecture decisions

- **Package `geneopt/`** with `core.py`, `cache.py`, `history.py`, `plots.py`, `__init__.py`.
  `__all__ = ["GeneticAlgorithm", "Result", "History", "__version__"]`.
- **Version lives in `geneopt/__init__.py`** (`__version__ = "2.0.0"`), read by hatchling via
  `[tool.hatch.version]`. It works uninstalled, so tests don't need `pip install`. This deviates
  from 006's `importlib.metadata` note, which needs an install to work.
- **Tests run uninstalled:** `[tool.pytest.ini_options] pythonpath = ["."]`. Nothing gets
  installed into the user's conda env.
- **No `hypothesis`:** property tests are plain loops over seeds, so there's no new dependency.
- **pandas, matplotlib, seaborn, catboost are imported lazily** inside the functions that need
  them, so `import geneopt` pulls in numpy only.
- **No shims anywhere.** Removed kwargs fail through the keyword-only signature.

## Task list

### Phase 1: Foundation
- [x] Task 1: Packaging skeleton + repo hygiene
- [x] Task 2: `GenomeCache` (exact keys, fingerprint, JSONL)

### Checkpoint A
- [x] `pytest` green, wheel builds

### Phase 2: Core
- [ ] Task 3: `History` + `Result` types
- [ ] Task 4: `GeneticAlgorithm` sequential run (integer genomes, operators, validation)
- [ ] Task 5: Executor + `on_error` policy + logging

### Checkpoint B
- [ ] Same seed → identical `Result` with and without executor. 011 kwarg tests green

### Phase 3: Surface
- [ ] Task 6: Plots + `evaluations()` / `to_frame()`
- [ ] Task 7: Examples (quickstart, categorical, distributed_http)
- [ ] Task 8: Delete 1.x package and files, rewrite README, CHANGELOG

### Checkpoint C (complete)
- [ ] All 011 acceptance criteria pass. `pytest` green. `python -m build` + `twine check` clean

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Integer genomes + new operators change search quality | Med | Task 4 includes a convergence test (the known optimum of a 2-D bowl and of a categorical toy is found within budget). The full 30-seed benchmark against 1.0.10 (003 gate) is **deferred** and listed as follow-up |
| Plot code is hard to test meaningfully | Low | Smoke tests only: each plot writes a non-empty PNG with the `Agg` backend (007) |
| `geneopt/` and `GeneOpt/` differ only by case | Low on Linux, High on macOS/Windows checkouts | Task 8 deletes `GeneOpt/`. Until then, this only matters on case-insensitive filesystems, and the work happens on Linux |
| Flask isn't installed, so `server.py` can't be run in tests | Low | The server's validation is a pure function, tested directly. The client is tested against a stdlib `http.server` stub (008) |

## Out of scope (follow-ups)

- CI workflow, trusted publishing (007 / 006 §4)
- 30-seed benchmark harness vs 1.0.10 (003 gate)
- `callback` progress hook (005 §5, marked optional)
