# 002 — Run History & Plots

**Status:** proposed · **Severity:** 🔴 critical · **Fixes:** H-1 … H-5, A-4 · **Depends on:** 001

## Problem

All plots read `self._cache.ram_cache_score`, which only holds genomes that were *evaluated*,
tagged with the generation of their *first* evaluation (H-1).

```
seed=3, pop=100, gens=50 → 5000 individual slots → 105 rows in the "history"
seed=1, pop=20,  gens=15 → rows per generation: {0:20, 1:2, 2:1, 6:1, 7:11, 10:1, 11:1}
```

So in practice:

| Plot | What it claims | What it actually shows |
|---|---|---|
| Best score over generations | best individual per generation | best *newly discovered* genome. Generations with no new genome are missing, and the curve isn't monotonic even though the GA's best is |
| Best parameter over generations | params of that best | same gaps, padded with `None` |
| Population diversity | diversity of the population | Hamming distance over float **values + score + generation** of new genomes (H-2) |
| Gene frequencies | allele frequency per gene | min-max-scaled mean parameter value. A constant column → NaN (H-3) |
| Feature analysis | landscape analysis | OK in spirit (unique evaluated points), but keeps `+inf` in minimize runs (H-4) |

## Design

### 1. History is its own small thing

```python
@dataclass
class History:
    environment: dict           # name -> values, used to decode genomes
    maximize: bool              # direction, needed for best-so-far
    genomes: list[np.ndarray]   # per generation: (pop, n_params) int index array
    scores:  list[np.ndarray]   # per generation: (pop,) float
    hits:    list[np.ndarray]   # per generation: (pop,) bool, served from cache
```

- The engine (004) appends one entry per generation **after** scoring, so cache hits are
  recorded like any other individual. That fixes H-1.
- Memory: `gens × pop × n_params × 8 B`. 1,000 × 1,000 × 20 = 160 MB is the ceiling to
  document. *(built 2026-10-17)* Genomes are stored as `int32`, which halves it. `gene_frequency_matrix`
  widens to `int64` before multiplying, so values above ~33.5M per parameter can't overflow.
- `to_frame()` → long DataFrame `(generation, individual, <param values…>, score, cache_hit)`
  for anyone who wants pandas. It's built on demand, never stored.
- *(built 2026-10-17)* `save(path)` → `np.savez_compressed`, without pickle (the environment is
  stored as JSON), and `History.load(path)`. It's opt-in, so runs don't write files unless asked.

The evaluated-points table that feature analysis needs is just the cache (001), exposed as
`ga.evaluations()` → DataFrame `(<param values…>, score)`. It's a different question ("what does the
landscape look like?") from the history ("what did the population do?"), so it has a
different source.

### 2. Plot definitions (exact)

| Plot | Formula |
|---|---|
| Best-so-far | `np.minimum.accumulate` / `np.maximum.accumulate` over per-generation best. Monotonic by construction. Also draw per-generation best and median as thin lines |
| Best params | params of the best-so-far genome at each generation (step plot) |
| Diversity | mean pairwise **gene** disagreement over `genomes[g]`, normalised to `[0, 1]` (divide by `n_params`). Fully converged = 0 |
| Gene frequencies | for each parameter, a heatmap `generation × value-index` of the share of the population holding that index. This is what a gene-frequency plot actually means. For params with > 64 values, bin the indices into 64 buckets |
| Cache-hit rate | new: share of `hits[g]` per generation. Cheap, and the clearest convergence signal |

### 3. Feature analysis cleanup

- Drop `±inf` **and** NaN (`np.isfinite`), which fixes H-4.
- Use only numeric parameters. Skip categorical ones in correlation/regression with a log
  note instead of crashing.
- Move it to the `GeneOpt[analysis]` extra: `catboost` and `scikit-learn` get imported inside
  the function, and if they're missing you get an `ImportError` explaining
  `pip install GeneOpt[analysis]` (A-4). `r2_score` is one line of numpy:
  `1 - ((y - p)**2).sum() / ((y - y.mean())**2).sum()`, so sklearn isn't needed for it.
- Pairwise scatter loops are O(p²) plots. Cap them at the top 8 parameters by importance
  and log what was skipped.

### 4. Output

- `plot(output_dir=None, dpi=150)`. The default is `{cache_dir}/{optimizer_name}-plots/`, or
  the current directory if there's no cache dir. No more hard-coded `./cache/`.
- File names describe the plot: `best_score.png`, `param_x1.png`, `diversity.png`,
  `gene_freq_x1.png`, `corr_with_score.png` … (not `0.png … 37.png`, which overwrite across runs).
- `dpi` defaults to 150 (`plot_DPI` is removed, see 011). 500 DPI makes ~5 MB PNGs and is slow for dozens of plots.
- The per-plot methods `plot_best_score()`, `plot_parameter_values_for_the_best_scores()`,
  `plot_population_diversity()`, `plot_gene_frequencies()` and `feature_plots()` are removed from
  `GeneticAlgorithm` (011). They become functions in `geneopt.plots` that take a `History`.
- Each plot runs in its own `try`. A failure is logged with the plot name and the rest
  continue. `plot()` returns `{name: path | Exception}`. Remove the `raise e` + dead
  `log_e` (H-5).
- Use matplotlib's object-oriented `matplotlib.figure.Figure` without pyplot. The 1.x
  pyplot-global style leaks state between plots (seaborn `jointplot` makes its own figure, so the
  first `plt.title` lands on the wrong axes). A bare `Figure` also needs no GUI backend.
- Everything is drawn with matplotlib alone, so seaborn isn't a dependency. The KDE joint plot
  is dropped.

## Acceptance criteria

- [ ] A 50-gen × 100-pop run gives `len(history.genomes) == 50`, each with 100 rows,
      no matter the cache hit rate.
- [ ] The best-so-far curve is monotonic (property test across seeds).
- [ ] Diversity is in `[0, 1]`, is 0 for a population of identical genomes, and is > 0.4 for a
      random initial population with ≥ 8 values per param.
- [ ] Gene-frequency rows sum to 1 for each parameter and generation.
- [ ] A minimize run where some objective calls return `+inf` produces every feature plot.
- [ ] `import GeneOpt` works without `catboost`/`sklearn` installed, and `plot()` without them
      still produces the core (non-analysis) plots.
- [ ] Two runs with different `optimizer_name` in the same directory don't overwrite each
      other's plots.
