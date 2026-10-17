# 016 — Benchmark Figures: ECDF, Performance Profile, Speed–Success Pareto

**Status:** implemented (2026-10-17); figures in the [2026-10-16 report](../docs/benchmarks/2026-10-16-fair-suite.md#figures) · **Severity:** 🟡 medium (reporting, not search) ·
**Depends on:** 015 (the suite and its logs) · **Stack:** matplotlib + pandas (as `geneopt/plots.py`)

## Objective

Ranking tables (015) say *who* finished first, but not *how fast*, *how reliably* or *at what
cost*. This spec adds three figures to the benchmark report. The figures are for the owner and
for anyone reading a published benchmark, so they must work in a paper.

| Figure | Question it answers |
|---|---|
| **ECDF** of runtimes to targets | How much of the suite does each optimiser solve, as the budget grows? (the COCO/IOH standard) |
| **Dolan–Moré performance profile** | How often is each optimiser the fastest, or within a factor τ of the fastest? (robustness) |
| **Speed vs success Pareto scatter** | What does each optimiser's reliability cost, in seconds or evaluations? |

Success means that each figure, built from one tidy DataFrame, is correct against values computed
by hand. It also has to stay readable with 7–10 optimisers, and it must never hide missing runs.

## Decisions (owner-confirmed)

1. **The code lives in the benchmark harness, not the library:** a new `tests/bench/plots.py`.
   These figures compare libraries, which isn't something a GeneOpt user calls. The library's
   own `geneopt/plots.py` stays about one run's History.
2. **Long format, one row per (run, target).** The columns proposed in the request (`algorithm`,
   `function_id`, `evaluations_to_target`, `success_bool`, `elapsed_time`) hold one target per
   row. An ECDF needs *several* targets per run, and a profile needs each run identified, so the
   schema below adds `dimension`, `instance` and `target`.
3. **Runtime is counted in evaluations, not seconds,** for the ECDF and the profile (COCO
   convention: unique objective calls, as the 015 Budget wrapper counts them). Seconds are only
   on the Pareto x-axis, because seconds depend on the machine.
4. **The figures complement `cocopp` and IOHanalyzer; they don't replace them.** 015 still
   requires COCO's own ECDFs for bbob-mixint. These figures put both suites and all
   optimisers in one consistent view.

## Input schema: `runs` DataFrame

| column | dtype | meaning |
|---|---|---|
| `algorithm` | str | row label, as in the 015 report (`geneopt`, `geneopt[mr=0.01]`, `pymoo`, …) |
| `function_id` | str | suite-qualified, e.g. `bbob-mixint/f15`, `pbo/F24` |
| `dimension` | int | D or n |
| `instance` | int | instance id within the suite |
| `target` | float | precision level. Smaller is harder. It is a gap to the optimum (see *Producing the data*) |
| `evaluations_to_target` | float | unique evaluations when the target was first reached. **NaN = not reached** |
| `success_bool` | bool | `evaluations_to_target` is not NaN. Optional: derived when absent |
| `elapsed_time` | float | wall seconds for the whole run (the same on every target row of a run). NaN allowed |
| `evaluations` | int | unique evaluations the run used in total (≤ budget) |

A **problem** is `(function_id, dimension, instance)`. A **run** is `(algorithm, problem)`. 015
uses one run per (optimiser, problem), because instances replace repeated seeds (COCO practice).

### Producing the data (harness side, 015)
- **bbob-mixint:** f_opt isn't exposed by `cocoex`. Read the observer's logs with `cocopp`
  (`DataSet.detEvals(targets)` gives evaluations to each target, NaN if not reached).
  Targets are COCO's standard `10^{2, 1.8, …, −8}` (51 levels) on f − f_opt.
- **PBO:** `ioh` exposes the optimum (`problem.optimum.y`, checked on F24 instance 2). The target
  is the relative gap `(y_opt − y_best) / |y_opt|` ∈ {0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0}, where
  0 means the optimum itself. Hit times come from the Budget wrapper's best-so-far trace.
- One helper, `runs_frame(...)`, builds the DataFrame. The plot functions never touch suite APIs.

## The three functions

```python
def plot_ecdf(runs: pd.DataFrame, *, per_dimension: bool = True) -> Figure: ...
def plot_performance_profile(runs: pd.DataFrame, *, target: float | None = None,
                             tau_max: float | None = None) -> Figure: ...
def plot_pareto(runs: pd.DataFrame, *, target: float | None = None,
                cost: Literal["elapsed_time", "evaluations"] = "elapsed_time") -> Figure: ...
def palette(algorithms: Iterable[str]) -> dict[str, Style]: ...   # shared by all three
```

### 1. ECDF (runtime distribution over targets)
- **x:** budget `b` in evaluations / dimension (COCO convention, so 10-D and 20-D share an axis),
  log10 scale, from 1 to the largest budget in the data.
- **y:** for each algorithm, the fraction of all (problem, target) pairs with
  `evaluations_to_target ≤ b · dimension`. The denominator is the full set of pairs in the frame,
  the union over all algorithms, so a missing run counts as unsolved (see *Missing data*).
- Drawn as a step function, with a `×` marker where each algorithm's budget ends. Nothing is
  extrapolated past it.
- `per_dimension=True` draws one panel per dimension, side by side with a shared y-axis.
  `False` pools all dimensions into a single panel.
- The y-axis runs from 0 to 1. The legend is ordered by final ECDF value, best first (COCO style).

### 2. Dolan–Moré performance profile (Dolan & Moré, 2002)
- Fix one `target` (default **1e−2**, as for the Pareto scatter; it is printed in the axis label).
- For problem p and algorithm s, t_{p,s} = `evaluations_to_target`, and ∞ if not reached.
  r_{p,s} = t_{p,s} / min_s t_{p,s}.
- **x:** τ ≥ 1, log2 scale, up to `tau_max` (default: 2 × the largest finite ratio).
  **y:** ρ_s(τ) = fraction of problems with r_{p,s} ≤ τ. Drawn as a step chart.
- ρ_s(1) is the share of problems where s was (jointly) fastest. The right end is the share it
  solved at all.
- Problems **no** algorithm solved have no ratio. They're excluded, and the count appears in the
  caption ("12 of 150 problems unsolved by all, excluded"), never dropped silently.

### 3. Speed vs success Pareto scatter
- One point per algorithm.
  - **x:** mean `elapsed_time` per run (or mean `evaluations` with `cost="evaluations"`), log scale.
  - **y:** success rate, the share of problems on which the algorithm reached `target` (default
    **1e−2**: f − f_opt ≤ 0.01 on COCO, relative gap ≤ 1% on PBO). Requiring the exact optimum
    at a 100·D budget would put every library near 0% and erase the comparison.
  - 95% Wilson interval as a vertical error bar.
- **Pareto frontier:** the points no other point beats on both axes (lower cost *and* higher
  success, ties count as not beaten). They're connected left to right by a dashed line and
  labelled with their names. Dominated points get a smaller, lighter label.
- Every point carries its name. With ≤ 10 points, labels are nudged apart, with no extra
  dependency.

### Shared style: `palette`
- **Hue = library, line style = variant** (revised while building: 9 report rows would otherwise
  need generated hues, and the dataviz rules forbid cycling categorical colours).
  - **Fixed hue order:** each library has a fixed hue that doesn't depend on which libraries are
    present.
  - **The palette:** Okabe–Ito, with Tol's wine (#882255) replacing yellow, which failed the
    lightness band.
  - **Validated by script, not by eye:** all-pairs colour-blind ΔE is at least 7.6. That's in the
    6–8 band, so every library also has its own marker.
  - **Random search** is the neutral grey baseline.
  - **Variants:** a library's main row is solid, its other rows dashed or dotted. GeneOpt's three
    rows share its hue.
- `geneopt*` rows are drawn thicker and on top, since GeneOpt is the subject of the report.
- Output: each function returns a `matplotlib.figure.Figure` (OO API, no pyplot state, like
  `geneopt/plots.py`). The harness saves each as **PDF (vector) and PNG (200 dpi)** under
  `bench-results/<suite>/figures/`, and the Markdown report embeds the PNGs.

## Missing and incomplete data

| Situation | Behaviour |
|---|---|
| a required column is missing, the frame is empty, or `algorithm`/`function_id` is NaN | `ValueError` naming the column(s) |
| `evaluations_to_target` NaN | not reached. The row is kept: it's data, not noise |
| `success_bool` absent | derived from `evaluations_to_target` |
| `success_bool` disagrees with `evaluations_to_target` | `ValueError` with the first bad row |
| an algorithm lacks some runs another has (incomplete grid) | counted as unsolved in all three figures. Legend label gets `(n/N runs)` and a `logging` warning lists the gaps |
| `elapsed_time` NaN for some runs | Pareto mean uses the remaining runs, and the point is drawn hollow. All NaN: the algorithm is left out of the time Pareto, with a warning |
| a problem unsolved by every algorithm | profile: excluded and counted in the caption. ECDF: stays in the denominator |
| duplicate (run, target) rows | `ValueError`, since that means a harness bug |

## Commands

```bash
# figures for an existing results directory (no re-run)
python tests/bench/run.py --suite mixint --plots-only bench-results/mixint
# full run + report + figures
python tests/bench/run.py --suite pbo --dims 64 --instances 1-5
# tests (plots marker, fast, no competitors needed)
python -m pytest tests/test_bench_plots.py -m plots
ruff check . && ruff format --check . && mypy geneopt tests/bench
```

## Project structure

```
tests/bench/plots.py        → palette, plot_ecdf, plot_performance_profile, plot_pareto, runs_frame
tests/bench/run.py          → calls them after a suite run; --plots-only
tests/test_bench_plots.py   → hand-computed fixtures (no competitors, no suites installed)
tests/bench/requirements.txt→ + matplotlib, pandas, cocopp (pinned)
specs/016-benchmark-plots.md
```

## Code style

Match `geneopt/plots.py`: OO `Figure` API, module constants for magic numbers, short docstrings,
and computation kept apart from drawing so it can be tested without images:

```python
def ecdf_curve(runs: pd.DataFrame, algorithm: str) -> tuple[np.ndarray, np.ndarray]:
    """(budgets per dimension, fraction of (problem, target) pairs solved) as a step function."""
    pairs = len(runs.drop_duplicates(["function_id", "dimension", "instance", "target"]))
    own = runs[runs.algorithm == algorithm]
    hits = np.sort((own.evaluations_to_target / own.dimension).dropna().to_numpy())
    return hits, np.arange(1, len(hits) + 1) / pairs
```

Every figure has a pure `*_curve`/`*_points` function underneath, and the `plot_*` function only
draws what it returns.

## Testing strategy

pytest under the existing `plots` marker. Small hand-built frames, with values worked out in the
test's comments:
- **ECDF:** 2 algorithms × 2 problems × 2 targets. Check the step heights and positions exactly.
  A missing run lowers the curve and adds the `(n/N runs)` label.
- **Profile:** a 3 × 3 table of t_{p,s} with known ratios. Check ρ_s(1), ρ_s(2) and ρ_s(∞). A
  problem nobody solved is excluded and counted.
- **Pareto:** 4 points, of which 2 are dominated. The frontier is exactly the other two, in x
  order. A tie case and an all-NaN time case are included.
- **Palette:** stable under reordering the input, distinct for 10 algorithms, `geneopt*`
  emphasised.
- **Every row of *Missing data*** has one test.
- **Smoke test:** each `plot_*` returns a Figure that saves to PDF and PNG in `tmp_path`.
- Coverage ≥ 90% on `tests/bench/plots.py`, the same gate as `geneopt/`.

## Boundaries
- **Always:** keep calculation separate from drawing; return `Figure`; test against hand-computed
  values; keep the colour mapping stable; show missing data instead of dropping it.
- **Ask first:** adding a dependency beyond matplotlib, pandas and cocopp (for example seaborn or
  adjustText); moving any of this into the `geneopt` package; changing the target sets.
- **Never:** extrapolate an ECDF past an algorithm's budget; compare raw PBO values across
  instances (015 B); tune a plot to favour GeneOpt (axis cut-offs and default target are rules,
  not picked per report).

## Success criteria
- [x] The three functions and `palette` exist in `tests/bench/plots.py` with the signatures above
      (plus a `width` argument, which sets the font size).
- [x] Hand-computed tests for each figure and for every *Missing data* row pass. Coverage is 100%.
- [x] `run.py --suite … ` writes PDF + PNG for all three figures and embeds them in the report.
      `--plots-only` rebuilds them from saved results.
- [x] On the 015 bbob-mixint run, our pooled ECDF agrees with `cocopp`'s within one step for every
      algorithm: at most 1.2 percentage points, against a step of 1/51. Where they differ, COCO's
      raw `.dat` log agrees with ours; see the report. This check also exposed a COCO log bug in
      the harness, now fixed.
- [x] Figures are readable in greyscale and at single-column width (3.5 in), checked by rendering:
      fonts scale with `width`, the legend sits below the axes, every library has its own marker
      on its lines, and Pareto labels are placed to avoid markers, error bars and each other.

## Resolved questions (owner, 2026-10-16)
1. **ECDF x-axis:** evaluations / D (COCO/BBOB). Otherwise high dimensions, which need far more
   evaluations, would dominate a pooled ECDF.
2. **Default target for the profile and the Pareto scatter:** 1e−2, not the global optimum. If
   1e−2 isn't among the frame's targets, `ValueError` lists the available ones.
3. **Location:** `tests/bench/plots.py`. These are internal validation tools. A public
   `geneopt.bench` would grow the user API and pull benchmark dependencies onto users.
