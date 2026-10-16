# Fair benchmark suite: first run (2026-10-16)

The first run of the [spec 015](../../specs/015-fair-benchmark-suite.md) suite at CI scope:
COCO **bbob-mixint** (D = 10, all 24 functions) and IOHprofiler **PBO** (n = 64, 13 functions),
instances 1–5, a budget of 100 · D unique evaluations, all seven optimisers, plus GeneOpt's two
sensitivity rows. Everything below is generated from the run records by
`tests/bench/run.py --report-only`, except the sections marked *analysis*.

- **Machine:** run locally, as all full benchmark runs are (spec 007): 12 cores (Intel
  i5-11400H), Linux, Python 3.12.14, 12 worker processes.
- **Code:** branch `v2.0.0`; bbob-mixint ran at `cdce24c`. Later commits change only the report
  layout and thread settings, not the runs.
- **Seeds:** the instance id. One run per (optimiser, problem), as in COCO.
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
- **Budget used** below 100% means the optimiser stopped early. It kept proposing points it had
  already seen until the duplicate-call cap ended the run (spec 007's rule).

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

geneopt[mr=1/D] against each optimiser, per problem instance (Wilcoxon signed-rank):

| optimiser | wins | ties | losses | p |
|---|---|---|---|---|
| deap | 29 | 0 | 91 | 4.26e-07 |
| geneopt[mr=0.5/D] | 24 | 1 | 95 | 5.2e-08 |
| geneopt[mr=2/D] | 91 | 1 | 28 | 2.1e-08 |
| nevergrad | 68 | 1 | 51 | 0.0173 |
| optuna | 70 | 0 | 50 | 0.0173 |
| pygad | 86 | 0 | 34 | 4.81e-05 |
| pymoo | 30 | 0 | 90 | 1.85e-07 |
| random | 112 | 0 | 8 | 3.86e-19 |

### Head-start check

No run reached the 0.01 target within its first 50 evaluations.

**Gate (main GeneOpt row beats random on ≥ 90% of pairs, higher ECDF area):** PASS

### Mutation rate 0.5/D against 1/D (*analysis*)

The 0.5/D row ranks 3rd overall (mean rank 3.00), against 4.42 for the main 1/D row.

**Method.**
- **Pairs:** each (function, instance) gives one pair of final precisions.
- **Scale-free tests:** pairs are pooled per BBOB function group. Raw precisions differ by
  orders of magnitude between functions, so a Wilcoxon test on raw differences would be decided
  by a few functions. Both tests used here ignore that scale:
  - the exact **sign test** (which row is closer to the optimum);
  - the **Wilcoxon signed-rank test on log10 precision**, floored at 1e−9 so that exact hits
    stay finite.
- **Correction:** Holm, across the 5 groups.
- **Why not per function:** with 5 instances, a per-function test can't go below p = 0.0625.
- **Effect sizes**, on the log scale:
  - the median log10 ratio, where −0.3 means 0.5/D gets about 2× closer to the optimum;
  - the rank-biserial correlation, where +1 means 0.5/D is better on every pair.

| group | pairs | 0.5/D better | 1/D better | ties | sign test p (Holm) | Wilcoxon on log10 precision p (Holm) | median log10 ratio | rank-biserial |
|---|---|---|---|---|---|---|---|---|
| separable f1–f5 | 25 | 23 | 1 | 1 | 1.5e-05 | 1.1e-05 | -0.61 | +0.95 |
| moderate f6–f9 | 20 | 15 | 5 | 0 | 0.12 | 0.083 | -0.15 | +0.45 |
| ill-conditioned f10–f14 | 25 | 23 | 2 | 0 | 7.8e-05 | 0.00011 | -0.32 | +0.87 |
| multimodal f15–f19 | 25 | 16 | 9 | 0 | 0.23 | 0.048 | -0.10 | +0.53 |
| weak structure f20–f24 | 25 | 18 | 7 | 0 | 0.12 | 0.048 | -0.15 | +0.54 |
| all (mixint) | 120 | 95 | 24 | 1 | 3.5e-11 (uncorrected) | 5.4e-12 (uncorrected) | -0.22 | +0.73 |

- **Separable (f1–f5) and ill-conditioned (f10–f14):** 0.5/D is clearly better.
  - It wins 23 of 25 pairs in both groups.
  - Both tests give Holm-corrected p ≤ 1.1e−4.
  - The rank-biserial correlation is +0.87 to +0.95.
  - Precision is 2–4× closer to the optimum.
- **Multimodal and weak structure:** the log-Wilcoxon test is borderline (p 0.048 after
  correction), but the sign test isn't significant (p 0.12–0.23). No claim is made here.
- **Moderate:** not significant under either test.
- **Reading:** at this short budget (1,000 evaluations), one mutation per child disturbs too
  much on smooth landscapes. That fits spec 012's finding that GeneOpt lacks fine local steps.
- **Status:** not yet a reason to change anything. It has to hold on PBO first (same tests,
  below). The rows were fixed before the run, and no rate is picked from these results after
  the fact.

## PBO, n = 64

*Pending: the local run is in progress.*
