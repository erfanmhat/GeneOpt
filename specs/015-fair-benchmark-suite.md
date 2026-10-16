# 015 — A Fair Benchmark Suite

**Status:** proposed, prototype tested · **Severity:** 🟠 high (it decides every operator change) ·
**Replaces:** the three problems of 007 § Benchmarks · **Pairs with:** 014

## Why the current suite is unfair

The 007 suite (Holder 2-D, shifted Rastrigin 10-D, a 6 × 5 categorical toy) was built quickly.
Measured against what established benchmarking practice requires, it has these flaws:

| Flaw | Evidence |
|---|---|
| **Separable problems.** Shifted Rastrigin can be solved one variable at a time, which per-variable operators (pymoo's SBX/PM, uniform crossover) exploit | Separable version (30 seeds): pymoo **0**, DEAP 0.64, GeneOpt 3.59. **Rotated** version (10 seeds): pymoo 19.2, GeneOpt 29.8, DEAP 32.8, random 92.1. The "solved" result comes from the problem's structure, not general ability |
| **One fixed instance.** A single hand-picked shift per problem | Any operator can overfit one optimum location |
| **Head starts are possible.** An optimum at a special point (the centre) | NGOpt starts at the centre and "solved" plain Rastrigin on its first guess (found in 007, fixed by shifting) |
| **No discrimination.** The categorical toy is separable (count the matches) | Four of seven optimisers hit 100%; the problem can't rank them |
| **Tiny dimension.** Holder is 2-D on a 4096² grid | It mainly tests fine-tuning on a grid, not search |
| **Final-best only.** One number per run | It says nothing about *how fast* an optimiser gets there (anytime performance) |

pymoo does **not** start at the solution. Its best after generation 0 is 134.6, and it first
reaches 0 at evaluation 3,247. The unfairness is in the problem, not the library.

## Principles (from COCO / IOHprofiler practice)

1. **Randomised instances.** Every function has many instances, each with its own random shift
   and rotation (or bit-flip/XOR transform). Results are aggregated over instances. No instance
   is hand-picked.
2. **Non-separable and deceptive problems.** These include rotated, ill-conditioned, multimodal,
   epistatic and deceptive landscapes, not only sums of independent terms.
3. **Several dimensions, with budgets proportional to dimension** (evaluations = k · D), so
   scaling is visible.
4. **Anytime metrics.** Record (evaluations, best-so-far) for every run. Report the fraction of
   (function, instance, target) pairs solved as a function of budget (empirical cumulative
   distribution, *ECDF*), the standard COCO/IOH measure, plus final-value ranks.
5. **Same representation for everyone.** Every optimiser sees the same discrete value lists. No
   optimiser gets a special initial point. Where a library seeds at the centre (NGOpt), the
   randomised optimum makes that harmless, and it is checked (principle 7).
6. **Disclosed configuration, calibrated by rule, never tuned on results.** Competitor libraries
   use their own defaults plus a shared population size. GeneOpt doesn't rely on any default:
   every operator parameter is passed explicitly, and `mutation_rate` is set per problem by the
   rule in *GeneOpt configuration*. The rule depends only on what is known before a run (suite,
   D, value counts), never on results. Every value used is listed in the report.
7. **Head-start check.** Each run records its best value within the first 50 evaluations. If any
   optimiser's early best is already within the final target, the problem is flagged and
   investigated before results are used.

## The suite

### A. COCO `bbob-mixint` (mixed-integer BBOB)

From the [COCO platform](https://coco-platform.org/), installed as `coco-experiment`, Python
module `cocoex`, version 2.8.2.
- **24 functions** from the BBOB families: separable, moderately conditioned, ill-conditioned,
  multimodal with global structure, multimodal with weak structure. Instances have random shifts
  and rotations.
- In each problem 80% of the variables are integer. Checked in dimension 10: 8 integer variables
  with cardinalities 2, 2, 4, 4, 8, 8, 16, 16, plus 2 continuous variables in [−5, 5].
- **GeneOpt adaptation:** the continuous variables become a shared grid of 256 evenly spaced
  values in their bounds. Every optimiser uses the same grid, and the grid is part of the
  problem definition.
- **Dimensions** 10 and 20. **Instances** 1–15. **Budget** 100 · D evaluations (a long-run
  setting of 1000 · D is optional, see Runtime).
- **Scoring:** COCO's own observer logs every run. `cocopp` produces ECDFs over the standard
  targets (f − f_opt ∈ 10^{2 … −8}). The true f_opt isn't exposed through the public API, so
  COCO's tooling is the only correct way to score precision. Our Markdown summary adds per-function
  median ranks.

### B. IOHprofiler PBO (pseudo-Boolean)

From [IOHprofiler](https://iohprofiler.github.io/), package `ioh`, version 0.3.22. These are
binary problems, where each gene has 2 values:

| ID | Problem | Why it's included |
|---|---|---|
| F18 | Low Autocorrelation Binary Sequences | hard, rugged, real-world origin |
| F19 | Ising model (ring) | strong variable interactions |
| F22 | Maximum independent vertex set | constrained structure |
| F23 | N-Queens | constraint-heavy, plateau-rich |
| F24 | Concatenated trap | **deceptive**: gradient-like signals point away from the optimum |
| F25 | NK landscape | tunable epistasis |
| F4–F10 | W-model OneMax variants | controlled neutrality, epistasis and ruggedness |

- **Dimensions** 64 and 100. **Instances** 1–5. **Budget** 100 · n evaluations.
- **Scoring must use IOH's logger and normalised targets**, not raw values. PBO instances 2 and
  above apply objective transformations (scaling and offset). In the prototype, raw values were
  positive although the maximisation problem had been negated, and on F25 every optimiser,
  random included, landed within 0.2 of the same transformed value. Raw medians across
  transformed instances aren't comparable. Use the IOH `Analyzer` logger and IOHanalyzer-style
  fixed-budget and fixed-target summaries.

### Optional later: C. real categorical problems
Tabular hyperparameter benchmarks (for example YAHPO Gym or HPOBench lookup tables) are realistic
mixed categorical spaces. They're out of scope for the first version and listed here so nobody
re-invents a synthetic "categorical toy".

## GeneOpt configuration (exact)

### Calibration rule for `mutation_rate`
**Main row: μ = 1/D**, the same as `mutation_rate="auto"` (014 D). Every child gets one mutation
on average, on every problem. Two **sensitivity rows** at μ = 0.5/D and μ = 2/D show how much
the result depends on this choice. They're reported, not picked from afterwards.

How the three inputs enter:
- **Dimension D:** directly, through 1/D. That's the standard 1/L rule, and for bit strings the
  classical result that 1/n is the best fixed rate on linear functions (Mühlenbein 1992; Droste,
  Jansen & Wegener 2002).
- **Benchmark type:** PBO genes are binary, so random resetting is a bit flip and μ = 1/n is
  exactly the textbook bit-flip rate. bbob-mixint genes have 2–256 values. The same per-gene
  rule applies, because random resetting always changes a mutated gene to a *different* value.
- **Environment size (values per gene):** it changes *where* a mutation lands, not *how many*
  genes change. The expected number of changed genes per child is μ · D for every cardinality.
  So cardinality doesn't enter the rate. A rule that scaled μ with cardinality would be a tuning
  knob without a published basis, so this suite doesn't use one.

### Values per problem
All runs: `population_size=50` (shared with every GA library), `crossover_probability=0.9`,
`tournament_size=3`, `elitism=1`, `seed` = the run's seed, `maximize=False`. The run stops when
the harness budget of unique evaluations is used up. `generations` is only an upper bound, which
caps total proposals at 50 × the budget, as for every library (007).

| Suite | D (n) | values per gene | budget (unique evals) | `mutation_rate` main (1/D) | sensitivity 0.5/D | sensitivity 2/D |
|---|---|---|---|---|---|---|
| bbob-mixint | 10 | 2, 2, 4, 4, 8, 8, 16, 16, 256, 256 | 1,000 | **0.1** | 0.05 | 0.2 |
| bbob-mixint | 20 | 2 ×4, 4 ×4, 8 ×4, 16 ×4, 256 ×4 | 2,000 | **0.05** | 0.025 | 0.1 |
| PBO | 64 | 2 | 6,400 | **0.015625** | 0.0078125 | 0.03125 |
| PBO | 100 | 2 | 10,000 | **0.01** | 0.005 | 0.02 |

Report row labels: `geneopt[mr=1/D]`, `geneopt[mr=0.5/D]` and `geneopt[mr=2/D]`. The library
default of 0.05 isn't run as a row of its own. It coincides with the main row at D = 20 anyway.

The 20-D cardinalities are COCO's (each integer cardinality repeated over its share of the
variables); the harness reads them from `cocoex` and prints the list it actually got in the
report, so this table is checked rather than trusted.

## Prototype test (2026-10-16)

A throwaway script ran the existing harness adapters on part of the suite:
- **Problems:** bbob-mixint f1 sphere, f10 rotated ellipsoid, f15 rotated Rastrigin, f21
  Gallagher 101 peaks and f24 Lunacek bi-Rastrigin (D = 10, instances 1–5, 1,000 evaluations),
  plus PBO F19, F24 and F25 (n = 64, instances 1–5, 2,000 evaluations).
- **Optimisers:** all seven. GeneOpt used the then-current default `mutation_rate` = 1/n, which
  equals the main row of *GeneOpt configuration*.
- **Cost:** 80 min wall time on 8 processes (about 8 CPU-hours, mostly Optuna and Nevergrad).

Median best over 5 instances (lower is better; bbob-mixint rows only, since the PBO rows used
raw transformed values, see B). In brackets: the best value within the first 50 evaluations.

| problem | GeneOpt | random | PyGAD | DEAP | pymoo | Optuna | Nevergrad |
|---|---|---|---|---|---|---|---|
| f1 sphere | −24.74 [−5.34] | −11.11 [−5.34] | −24.74 [−8.36] | **−25.25** [2.88] | −25.24 [−8.47] | −11.03 [−6.95] | −25.02 [−12.8] |
| f10 ellipsoid | 25.21 [988] | 89.06 [988] | 24.35 [768] | 13.77 [416] | 25.79 [492] | **10.29** [333] | 44.72 [295] |
| f15 Rastrigin (rotated) | 8.319 [24.1] | 15.3 [24.1] | 11.58 [20.4] | 7.201 [18.6] | **6.363** [18.1] | 9.633 [13.5] | 8.385 [13.2] |
| f21 Gallagher | −0.1822 [47.3] | 20.69 [47.3] | 11.53 [34.6] | 7.946 [39.6] | **−1.6** [63.1] | 5.366 [23.7] | 0.4075 [44.6] |
| f24 Lunacek | 15.5 [21.3] | 19.94 [21.3] | 16.97 [25.5] | 15.52 [23.3] | **13.6** [23] | 14.67 [19.7] | 14.55 [16] |

Mean rank over all 8 prototype problems (1 = best): pymoo 2.12, DEAP 2.25, **GeneOpt 2.62**,
Optuna 3.62, Nevergrad 4.25, PyGAD 5.38, random 6.88.

**What the prototype shows:**
- **No head starts.** Every early best (bracketed) is far from the final value.
- **The suite discriminates without one library dominating.** Optuna wins the ill-conditioned
  ellipsoid, pymoo the multimodal functions, DEAP the sphere. That's the expected pattern on a
  fair suite, unlike pymoo's flawless 0 on separable Rastrigin.
- **GeneOpt is 3rd overall and close to the leaders,** 2nd on Gallagher. That's a realistic
  baseline for the 014 changes to be measured against.
- **PBO needs proper scoring** before its results mean anything (see B).

## Runtime policy
- Optuna's TPE cost grows roughly quadratically with trial count, and NGOpt is slow on 64+ binary
  variables. At full size (24 functions × 2 dimensions × 15 instances, plus PBO) a single
  machine needs many hours.
- **CI (`bench.yml`, opt-in):** bbob-mixint D = 10, instances 1–5, budget 100 · D, plus PBO
  n = 64, instances 1–5. All seven optimisers run, sharded across a job matrix (one job per
  optimiser) so no job exceeds GitHub's 6-hour limit.
- **Full run:** locally or on a larger runner, with results committed as a dated report.

## Gate (replaces 003's)
- **Hard:** for the main GeneOpt row (μ = 1/D), its median final value beats random search on
  **≥ 90% of (function, dimension) pairs** in each suite, and its ECDF area is above random's.
- **Reported, not gated:** mean rank (Friedman) across all problems, pairwise Wilcoxon
  signed-rank tests of GeneOpt against each library, and the ECDF plots. A rank regression
  between releases is investigated.

## Harness changes
- `tests/bench/run.py` gets `--suite {mixint,pbo,legacy}`. `legacy` (today's three problems)
  stays only as the `--quick` adapter smoke test and is no longer used for any claim.
- Each run writes COCO/IOH logs under `bench-results/<suite>/<optimiser>/`. A summary step builds
  the Markdown report (ranks, head-start check, gate) and, when `cocopp` and IOHanalyzer exports
  are available, the ECDF figures.
- The GeneOpt adapter takes its whole configuration from one table in `run.py`, keyed by suite
  and D, with every operator parameter written out. A unit test checks that the adapter passes
  each operator parameter of `GeneticAlgorithm` (`mutation_rate`, `crossover_probability`,
  `tournament_size`, `elitism`, `population_size`) explicitly. It also checks that the table
  matches *Values per problem* above.
- The report starts with a configuration section: the exact GeneOpt values per problem, the
  cardinalities read from each suite, and the library versions.
- `tests/bench/requirements.txt` adds `coco-experiment` and `ioh` (pinned) and `scipy` (for the
  statistical tests).

## Acceptance criteria
- [ ] `python tests/bench/run.py --suite mixint --dims 10 --instances 1-5` runs all seven
      optimisers and writes COCO logs, plus a Markdown report with ranks and the head-start check.
- [ ] `--suite pbo` scores with IOH's logger. The report shows normalised fixed-budget results,
      and no raw transformed values are compared across instances.
- [ ] The head-start check flags nothing, or every flag is explained in the report.
- [ ] The new gate passes for the main GeneOpt row (μ = 1/D). The 0.5/D and 2/D rows are
      reported alongside.
- [ ] The report lists every GeneOpt parameter value for every (suite, D), and they match
      *Values per problem*. No GeneOpt run depends on a library default.
- [ ] `bench.yml` shards by optimiser, and each shard finishes within GitHub's limits.
- [ ] Spec 007's benchmark section points here. The legacy results stay as history only.
