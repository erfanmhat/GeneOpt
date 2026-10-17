# Fair benchmark suite: first run (2026-10-16)

The first run of the [spec 015](../../specs/015-fair-benchmark-suite.md) suite at CI scope:
COCO **bbob-mixint** (D = 10, all 24 functions) and IOHprofiler **PBO** (n = 64, 13 functions),
instances 1–5, a budget of 100 · D unique evaluations, all seven optimisers, plus GeneOpt's two
sensitivity rows. The *Configuration* and *Results* sections are generated from the run records by
`tests/bench/run.py --report-only`; sections marked *analysis* are not.

## Summary

- **The gate passes on both suites.** The main GeneOpt row (mutation rate 1/D) beats random
  search on 24/24 bbob-mixint functions and 13/13 PBO functions, with a larger ECDF area. It's
  closer to the optimum than random on 112/120 and 65/65 problem instances.
- **No head starts.** No optimiser reached the 1e−2 target within its first 50 evaluations.
- **Mean rank (1 = best):**

  | Suite | Order |
  |---|---|
  | bbob-mixint | DEAP 2.75, pymoo 2.79, GeneOpt 0.5/D 3.00, **GeneOpt 1/D 4.42**, Nevergrad 5.42, Optuna 5.54, PyGAD 6.00, GeneOpt 2/D 6.33, random 8.75 |
  | PBO | GeneOpt 0.5/D 2.54, **GeneOpt 1/D 3.00**, DEAP 3.27, GeneOpt 2/D 3.81, pymoo 5.04, Optuna 5.35, Nevergrad 5.69, PyGAD 7.42, random 8.88 |

- **Where GeneOpt stands:**
  - **PBO:** GeneOpt is the strongest library. It has the best ECDF area, 0.413 for 1/D and
    0.445 for 0.5/D, against DEAP's 0.409.
  - **bbob-mixint:** it's mid-field. DEAP and pymoo are clearly ahead, especially on smooth
    landscapes, where GeneOpt lacks fine local steps (spec 012).
- **Mutation rate:** 0.5/D beats 1/D on bbob-mixint's separable and ill-conditioned functions,
  but not on PBO, so no default changes (*analysis* below).
- **Comparability caveat:** pymoo and Nevergrad often stop before their budget is spent (see
  *Budget used*). Their results on those functions come from fewer evaluations.

## Setup

- **Machine:** run locally, as all full benchmark runs are (spec 007): 12 cores (Intel
  i5-11400H), Linux, Python 3.12.14, 12 worker processes. Wall time: about 6 min for
  bbob-mixint, 4 h for PBO. Most of it is Optuna, plus BLAS thread oversubscription that was
  fixed after this run (`764df6e`).
- **Code:** branch `v2.0.0`. The runs used `cdce24c`. Later commits change only the report,
  thread settings and tests, not the runs.
- **Seeds:** the instance id. One run per (optimiser, problem), as in COCO.
- **Reproducibility (*analysis*):** four cells were rerun afterwards, from different suites and
  GeneOpt rows. Their traces, final values and references were identical:
  - bbob-mixint f15 i1 at 1/D;
  - bbob-mixint f21 i3 at 0.5/D;
  - PBO F24 i2 at 1/D;
  - PBO F18 i4 at 2/D.
- **Reproduce:**
  ```bash
  pip install -e . -r tests/bench/requirements.txt
  python tests/bench/run.py --suite mixint --out-dir bench-results --out mixint.md
  python tests/bench/run.py --suite pbo --out-dir bench-results --out pbo.md
  ```

## How to read the tables

- **Precision** is the distance to the optimum. On bbob-mixint it's f − f_opt, with f_opt from
  COCO's own logs. On PBO it's the relative gap to the reference. 0 means the optimum was found.
- **ECDF area** is the anytime score: the mean share of the log-budget axis spent with each
  (problem, target) pair already solved. 1 means every target was hit at once.
- **Success** means the precision reached the 1e−2 target.
- **Budget used** below 100% means the optimiser stopped early (see *Budget used*).
- **Pairwise tests** compare the main GeneOpt row with each optimiser on every problem instance.
  Both tests are scale-free, so the functions with the largest values don't decide the result:
  - the exact sign test;
  - the Wilcoxon signed-rank test on log10 precision, floored at 1e−9.

## bbob-mixint, D = 10

### Configuration

Competitors: library defaults plus population 50. GeneOpt: every parameter explicit.

| suite | D | values per gene | budget | row | GeneOpt parameters |
|---|---|---|---|---|---|
| mixint | 10 | 2 ×2, 4 ×2, 8 ×2, 16 ×2, 256 ×2 | 1000 | geneopt[mr=0.5/D] | `mutation_rate=0.05`, `population_size=50`, `crossover_probability=0.9`, `tournament_size=3`, `elitism=1` |
| mixint | 10 | 2 ×2, 4 ×2, 8 ×2, 16 ×2, 256 ×2 | 1000 | geneopt[mr=1/D] | `mutation_rate=0.1`, `population_size=50`, `crossover_probability=0.9`, `tournament_size=3`, `elitism=1` |
| mixint | 10 | 2 ×2, 4 ×2, 8 ×2, 16 ×2, 256 ×2 | 1000 | geneopt[mr=2/D] | `mutation_rate=0.2`, `population_size=50`, `crossover_probability=0.9`, `tournament_size=3`, `elitism=1` |

### Results

Median final precision over instances (lower is better; best in bold).

| function | D | deap | geneopt[mr=0.5/D] | geneopt[mr=1/D] | geneopt[mr=2/D] | nevergrad | optuna | pygad | pymoo | random |
|---|---|---|---|---|---|---|---|---|---|---|
| f1 | 10 | **0.00164** | 0.0216 | 0.391 | 1.69 | 0.228 | 7.11 | 1.54 | 0.00667 | 11.9 |
| f2 | 10 | **0.471** | 1.14 | 8.86 | 12 | 6.68 | 3.23 | 7.44 | 0.79 | 133 |
| f3 | 10 | 0.201 | 0.238 | 0.632 | 1.91 | 0.842 | 2.37 | 1.22 | **0.0592** | 9.92 |
| f4 | 10 | 0.208 | 0.414 | 1.34 | 2.88 | 1.65 | 2.97 | 2.85 | **0.187** | 11.3 |
| f5 | 10 | 0.304 | 1.09 | 3.09 | 8.21 | 27.4 | 3.44 | 9.1 | **0** | 50.5 |
| f6 | 10 | 0.332 | **0.233** | 0.384 | 0.656 | 0.56 | 0.617 | 0.643 | 0.427 | 20.9 |
| f7 | 10 | **7.3** | 10.4 | 10.8 | 10.2 | 20.2 | 12.1 | 12.1 | 7.88 | 54.2 |
| f8 | 10 | **0.233** | 0.267 | 1.13 | 4.22 | 0.84 | 10.4 | 3.4 | 0.233 | 27.3 |
| f9 | 10 | 0.808 | 1.84 | 1.77 | 3.16 | **0.76** | 1.39 | 2.54 | 1.53 | 15.1 |
| f10 | 10 | 14.3 | 15.3 | 25.3 | 36 | 44.9 | **10.5** | 24.3 | 25.7 | 89.5 |
| f11 | 10 | 0.843 | 0.444 | **0.41** | 0.439 | 0.693 | 0.563 | 0.619 | 0.441 | 0.68 |
| f12 | 10 | **0.19** | 0.547 | 18.6 | 128 | 34.8 | 682 | 120 | 1.18 | 1.12e+03 |
| f13 | 10 | 1.4 | 3.39 | 17.4 | 18.3 | 30.1 | 32.6 | 20.1 | **0.69** | 53.6 |
| f14 | 10 | 0.0255 | 0.051 | 0.329 | 1.26 | 0.285 | 4.25 | 0.925 | **0.0101** | 5.51 |
| f15 | 10 | 5.26 | 6.58 | 6.93 | 8.01 | 7.02 | 6.39 | 8.34 | **3.82** | 13.9 |
| f16 | 10 | 11.2 | 8.92 | 14.8 | 21.1 | 11.3 | **6.42** | 17.6 | 8.91 | 17.2 |
| f17 | 10 | 7.42 | 5.99 | 10 | 13 | 34.2 | **0.639** | 17.6 | 12.9 | 43.3 |
| f18 | 10 | 7.51 | **3.66** | 6.17 | 10.8 | 7.62 | 6.49 | 4.48 | 4.88 | 17.3 |
| f19 | 10 | 48 | 43.4 | 41.6 | 47.1 | 42.3 | 45.7 | **38.4** | 42 | 61 |
| f20 | 10 | **0.0258** | 0.0406 | 0.116 | 0.166 | 0.121 | 0.155 | 0.215 | 0.173 | 11.2 |
| f21 | 10 | 0.942 | **0.0124** | 0.842 | 0.734 | 2 | 6.97 | 1.66 | 0.141 | 22.3 |
| f22 | 10 | 4.28 | 6.27 | 9.09 | 10.4 | 7.31 | 22.5 | 9.49 | **4.27** | 39.8 |
| f23 | 10 | **18.8** | 23.6 | 25.2 | 19.8 | 22 | 26.3 | 24.8 | 29.4 | 25.8 |
| f24 | 10 | 7.24 | 6.72 | 7.62 | 8.17 | 7.3 | 6.51 | 8.16 | **4.27** | 10.8 |
| **mean rank** | | 2.75 | 3.00 | 4.42 | 6.33 | 5.42 | 5.54 | 6.00 | 2.79 | 8.75 |
| **ECDF area** | | 0.087 | 0.075 | 0.068 | 0.063 | 0.087 | 0.093 | 0.066 | 0.074 | 0.049 |
| **success at 0.01** | | 8% | 4% | 1% | 0% | 2% | 5% | 0% | 11% | 0% |
| **budget used (median)** | | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

geneopt[mr=1/D] against each optimiser, per problem instance: exact sign test, and Wilcoxon signed-rank on log10 precision (both scale-free; wins = closer to the optimum):

| optimiser | wins | ties | losses | sign test p | Wilcoxon (log) p |
|---|---|---|---|---|---|
| deap | 29 | 0 | 91 | 1.22e-08 | 1.82e-11 |
| geneopt[mr=0.5/D] | 24 | 1 | 95 | 3.47e-11 | 5.4e-12 |
| geneopt[mr=2/D] | 91 | 1 | 28 | 5.8e-09 | 9.18e-10 |
| nevergrad | 68 | 1 | 51 | 0.142 | 0.0532 |
| optuna | 70 | 0 | 50 | 0.0824 | 0.0498 |
| pygad | 86 | 0 | 34 | 2.28e-06 | 4.35e-08 |
| pymoo | 30 | 0 | 90 | 3.77e-08 | 1.18e-10 |
| random | 112 | 0 | 8 | 1.36e-24 | 6.25e-20 |

### Head-start check

No run reached the 0.01 target within its first 50 evaluations.

**Gate (main GeneOpt row beats random on ≥ 90% of pairs, higher ECDF area):** PASS

## PBO, n = 64

### Results

Median final precision over instances (lower is better; best in bold).

| function | D | deap | geneopt[mr=0.5/D] | geneopt[mr=1/D] | geneopt[mr=2/D] | nevergrad | optuna | pygad | pymoo | random |
|---|---|---|---|---|---|---|---|---|---|---|
| F4 | 64 | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | 0.188 |
| F5 | 64 | **0** | **0** | **0** | **0** | 0.0175 | 0.0175 | 0.105 | **0** | 0.263 |
| F6 | 64 | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | 0.0952 |
| F7 | 64 | 0.0625 | **0.0469** | 0.0781 | 0.141 | 0.188 | 0.109 | 0.234 | 0.125 | 0.281 |
| F8 | 64 | **0** | **0** | **0** | **0** | 0.0303 | 0.0606 | 0.121 | 0.0303 | 0.273 |
| F9 | 64 | **0** | **0** | **0** | **0** | 0.0156 | 0.0469 | 0.109 | **0** | 0.266 |
| F10 | 64 | 0.25 | **0.0156** | **0.0156** | **0.0156** | **0.0156** | 0.0938 | 0.0938 | **0.0156** | 0.25 |
| F18 | 64 | 0.056 | **0** | 0.064 | 0.169 | 0.371 | 0.233 | 0.454 | 0.25 | 0.436 |
| F19 | 64 | **0.0625** | 0.0938 | **0.0625** | 0.0938 | **0.0625** | 0.0938 | 0.188 | 0.125 | 0.281 |
| F22 | 64 | 0.125 | **0.0625** | 0.125 | 0.156 | 0.344 | 0.156 | 0.438 | 0.281 | 14.3 |
| F23 | 64 | 0.125 | **0** | 0.125 | 0.125 | 1.25 | 0.25 | 6 | 0.375 | 33.8 |
| F24 | 64 | **0** | 0.0345 | 0.0357 | 0.0357 | 0.145 | 0.0357 | 0.182 | 0.0727 | 0.327 |
| F25 | 64 | 0.00822 | **0** | **0** | 0.0068 | 0.0603 | 0.0307 | 0.156 | 0.0437 | 0.28 |
| **mean rank** | | 3.27 | 2.54 | 3.00 | 3.81 | 5.69 | 5.35 | 7.42 | 5.04 | 8.88 |
| **ECDF area** | | 0.409 | 0.445 | 0.413 | 0.361 | 0.248 | 0.338 | 0.210 | 0.349 | 0.127 |
| **success at 0.01** | | 49% | 57% | 52% | 43% | 15% | 20% | 14% | 29% | 0% |
| **budget used (median)** | | 100% | 100% | 100% | 100% | 57% | 100% | 100% | 38% | 100% |

geneopt[mr=1/D] against each optimiser, per problem instance: exact sign test, and Wilcoxon signed-rank on log10 precision (both scale-free; wins = closer to the optimum):

| optimiser | wins | ties | losses | sign test p | Wilcoxon (log) p |
|---|---|---|---|---|---|
| deap | 19 | 33 | 13 | 0.377 | 0.123 |
| geneopt[mr=0.5/D] | 10 | 41 | 14 | 0.541 | 0.137 |
| geneopt[mr=2/D] | 26 | 36 | 3 | 1.52e-05 | 7.23e-05 |
| nevergrad | 51 | 14 | 0 | 8.88e-16 | 5.11e-10 |
| optuna | 44 | 16 | 5 | 7.6e-09 | 9.41e-08 |
| pygad | 56 | 9 | 0 | 2.78e-17 | 7.52e-11 |
| pymoo | 38 | 26 | 1 | 1.46e-10 | 6.59e-08 |
| random | 65 | 0 | 0 | 5.42e-20 | 2.39e-12 |

### Head-start check

No run reached the 0.01 target within its first 50 evaluations.

**Gate (main GeneOpt row beats random on ≥ 90% of pairs, higher ECDF area):** PASS

The configuration is the same as above, with n = 64. Every gene has 2 values, the budget is
6,400, and the mutation rates are 1/64, 0.5/64 and 2/64.

**References.** IOH states no optimum for F18 (LABS) or F25 (NK). Its stated F24 optimum at
n = 64 (8.75) is beaten: the all-zeros string scores 10.35. For these three functions, precision
is measured against the best value any optimiser found on that instance. So a 0 there means
"best found in this experiment", not "proven optimum".

## Budget used (*analysis*)

Median share of the 6,400-evaluation budget actually used, per function (PBO). All other
optimisers, and every optimiser on bbob-mixint, used 100%.

| optimiser | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F18 | F19 | F22 | F23 | F24 | F25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nevergrad | 100% | 81% | 100% | 57% | 93% | 100% | 100% | 56% | 52% | 52% | 50% | 58% | 54% |
| pymoo | 100% | 17% | 100% | 73% | 29% | 31% | 100% | 58% | 22% | 36% | 62% | 32% | 23% |

- **Nevergrad** keeps re-proposing points it has already seen. Spec 007's call cap (50 × budget
  calls) then ends the run. On F4, 2 of its 5 runs stopped at about 52%. The median is 100%,
  because the other three solved F4 first.
- **pymoo** stops by itself. With duplicate elimination on, it can't produce new offspring once
  the population has converged.
- Both rules are spec 007's and are kept. Their numbers on functions below 100% come from fewer
  evaluations, so they aren't comparable at equal budget. Their ranks there are lower bounds on
  what they would reach with the full budget.

## Mutation rate 0.5/D against 1/D (*analysis*)

**Method.**
- **Pairs:** each (function, instance) gives one pair of final precisions.
- **Pooling:** pairs are pooled per function group (BBOB's five groups; two PBO groups).
- **Correction:** Holm, across the groups of a suite.
- **Tests:** the same two scale-free tests as above.
- **Why not per function:** with 5 instances, a per-function test can't go below p = 0.0625.
- **Effect sizes**, on the log scale:
  - the median log10 ratio, where −0.3 means 0.5/D gets about 2× closer to the optimum;
  - the rank-biserial correlation, where +1 means 0.5/D is better on every pair.

**bbob-mixint:**

| group | pairs | 0.5/D better | 1/D better | ties | sign test p (Holm) | Wilcoxon on log10 precision p (Holm) | median log10 ratio | rank-biserial |
|---|---|---|---|---|---|---|---|---|
| separable f1–f5 | 25 | 23 | 1 | 1 | 1.5e-05 | 1.1e-05 | -0.61 | +0.95 |
| moderate f6–f9 | 20 | 15 | 5 | 0 | 0.12 | 0.083 | -0.15 | +0.45 |
| ill-conditioned f10–f14 | 25 | 23 | 2 | 0 | 7.8e-05 | 0.00011 | -0.32 | +0.87 |
| multimodal f15–f19 | 25 | 16 | 9 | 0 | 0.23 | 0.048 | -0.10 | +0.53 |
| weak structure f20–f24 | 25 | 18 | 7 | 0 | 0.12 | 0.048 | -0.15 | +0.54 |
| all (mixint) | 120 | 95 | 24 | 1 | 3.5e-11 (uncorrected) | 5.4e-12 (uncorrected) | -0.22 | +0.73 |

**PBO:**

| group | pairs | 0.5/D better | 1/D better | ties | sign test p (Holm) | Wilcoxon on log10 precision p (Holm) | median log10 ratio | rank-biserial |
|---|---|---|---|---|---|---|---|---|
| W-model F4–F10 | 35 | 3 | 2 | 30 | 1 | 0.75 | +0.00 | -0.20 |
| LABS, Ising, MIVS, N-Queens, trap, NK (F18, F19, F22–F25) | 30 | 11 | 8 | 11 | 1 | 0.15 | +0.00 | +0.46 |
| all (pbo) | 65 | 14 | 10 | 41 | 0.54 (uncorrected) | 0.14 (uncorrected) | +0.00 | +0.35 |

- **bbob-mixint, separable (f1–f5) and ill-conditioned (f10–f14):** 0.5/D is clearly better.
  - It wins 23 of 25 pairs in both groups.
  - Both tests give Holm-corrected p ≤ 1.1e−4.
  - The rank-biserial correlation is +0.87 to +0.95.
  - Precision is 2–4× closer to the optimum.
- **bbob-mixint, other groups:** the multimodal and weak-structure groups are borderline under
  the log test only (p 0.048). The moderate group isn't significant. No claim is made for them.
- **PBO:** no difference. Most pairs are ties (both rows solve the problem), and neither test
  is significant.
- **Conclusion:** the effect is real but specific to smooth, separable or ill-conditioned
  landscapes at a short budget. That fits spec 012: GeneOpt has no fine local steps. It doesn't
  justify changing the default or the 1/D rule. The rows were fixed before the run, and no rate
  is picked from these results after the fact.
