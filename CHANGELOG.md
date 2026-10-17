# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.0]

A rewrite and a hard break: every 1.x parameter, module and helper is removed, without
compatibility shims. The full list is in [specs/011-removed-in-2.0.md](specs/011-removed-in-2.0.md).
The bugs fixed are listed in [specs/README.md](specs/README.md).

### Changed
- Package renamed to `geneopt`, with a keyword-only API. `start()` returns a `Result`.
- Genomes are integer value indices, not bit strings. Initial sampling is uniform, mutation is a
  single per-gene probability. The standard operators are used: tournament selection (exactly `k`
  candidates) with elitism, uniform crossover (`crossover_probability`, default 0.9, fair 50/50
  gene mixing), and random-resetting mutation (`mutation_rate`, default 0.05; `"auto"` uses 1 / number of
  parameters).
- `start_number_of_population` is removed without a replacement: every generation has
  `population_size` individuals.
- Cache keyed by exact gene indices, persisted as append-only JSONL named by a fingerprint of the
  environment and objective code. Stale or wrong cache hits are no longer possible.
- One evaluation loop: per-generation dedupe, any `concurrent.futures` executor, `on_error` policy.
- Plots are built from a full per-generation `History` and written with descriptive names.
- Logging uses the standard `geneopt` logger, with no files written.
- `History` stores genomes as `int32` (half the memory) and can be saved and loaded:
  `history.save(path)` / `History.load(path)`, a compressed `.npz` without pickle.
- Core dependency is numpy only. Plots and analysis are extras. Python 3.10+.

### Added
- `callback`: called after each generation with a `GenerationStats`; returning `True` stops the run
  (`stopped_by="callback"`).
- A benchmark suite against other libraries on COCO bbob-mixint and IOHprofiler PBO, with a
  committed report (`docs/benchmarks/`).

### Removed
- 1.x caches (`{name}.csv`/`{name}.json`) are ignored, not migrated.
- `scikit-learn` and `seaborn` dependencies.

## [1.0.10]

Last 1.x release.
