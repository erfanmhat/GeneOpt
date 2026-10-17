# 012 — Crossover Mixes Too Little on Many-Parameter Problems

**Status:** resolved: standard operators adopted (see Resolution) · **Severity:** 🟠 high (search quality) · **Found by:** the 007 benchmark ·
**Depends on:** 003

## Problem

On the 10-D shifted Rastrigin benchmark (5,000 unique evaluations, 30 seeds), GeneOpt places
**3rd of 7**, and the gap to the leaders is real (the IQRs don't overlap):

| optimiser | median best [IQR] |
|---|---|
| pymoo GA | 0 [0, 0] |
| DEAP | 0.64 [0.64, 0.95] |
| **GeneOpt** | **6.28 [5.42, 7.59]** |
| Nevergrad NGOpt | 6.65 [5.68, 8.43] |
| PyGAD | 11.7 [9.09, 13.6] |

On the 2-D Holder table and the 6-parameter categorical problem, GeneOpt ties for 1st. The
weakness grows with the number of parameters.

### Likely cause

Crossover (003 §1) swaps **each gene with probability `1 / n_params`**:

```python
swap = rng.random(p1.shape) < 1.0 / p1.shape[1]
```

So two parents exchange **about one gene per mating**, whatever the problem size. On a separable
10-D problem, a good value found in one individual rarely meets good values found elsewhere.
pymoo's SBX and DEAP's two-point crossover recombine many genes at once. The 1/n rate was
inherited from 1.x (`1.0 / len(self._environment_param_n_bits)`) and was never measured.

This is a hypothesis. Step 1 below checks it before anything changes.

## Design

1. **Check the cause first.** Re-run the benchmark with the swap rate at 1/n, 0.2 and 0.5
   (uniform crossover). Keep everything else fixed. If 0.5 doesn't close most of the gap, the
   cause is elsewhere (likely candidates: mutation creep size, elitism count), and this spec is
   revised before any code changes.
2. **If confirmed:** add `crossover_swap_rate: float = 0.5` (the per-gene swap probability) to the
   constructor, validated in `[0, 1]`. Default to whichever rate wins the benchmark without
   losing Holder or categorical.
3. **The 1/n behaviour stays reachable** as `crossover_swap_rate=1/n_params`. It isn't the default.

## Measurement (step 1, 2026-10-16)

30 seeds per cell, same harness, budgets and seeds as the 007 benchmark
(`python tests/bench/run.py --only geneopt,random --geneopt-swap-rate 1/n --geneopt-swap-rate 0.2
--geneopt-swap-rate 0.5`):

| swap rate | Rastrigin median [IQR] | Holder median | categorical hit rate |
|---|---|---|---|
| 1/n (current default) | 6.28 [5.42, 7.59] | −19.2085 | 100% |
| 0.2 | 5.54 [4.76, 6.47] | −19.2085 | 100% |
| **0.5** | **4.90 [3.36, 5.93]** | −19.2085 | 100% |
| *for reference: DEAP / pymoo (007)* | *0.64 / 0* | | |

**Finding: crossover is a minor factor, not the main cause.**
- A 0.5 swap rate improves Rastrigin by about 22% and costs nothing on Holder or categorical.
- It closes only about a quarter of the gap to DEAP (6.28 → 4.90, against 0.64).
- Under step 1 the main cause is elsewhere. The default is **not** changed until it's found.

### Next hypotheses (for a revised step 2)

| Hypothesis | Why it fits | Experiment |
|---|---|---|
| Creep step too coarse | `CREEP_SIGMA = 0.1` × 256 values ≈ 26 index steps ≈ 1.0 in x. That's the spacing between Rastrigin's local minima, so creep keeps jumping between basins instead of refining within one. A median of ~5 is consistent with ~5 dimensions stuck one basin away | `CREEP_SIGMA` ∈ {0.1, 0.03, 0.01}, or a mix of large and small steps |
| Adaptive mutation too disruptive near convergence | the rate rises to 0.5 per gene as diversity falls, scrambling good individuals | `adaptive_mutation=False` at 0.05 and 0.1 |
| Selection pressure / elitism | 1 elite out of 50 with tournament 3 | `elitism` ∈ {1, 5}, `tournament_size` ∈ {3, 5} |

Each experiment is a harness variant, measured exactly like the table above. A default changes only
on evidence that holds across all three problems.

## Resolution (2026-10-16)

The owner decided to replace mutation, selection and crossover with the **most standard
approach**, rather than tune the custom operators:

| Operator | First 2.0 build | Now (003) |
|---|---|---|
| Selection | tournament (k = 3) + 1 elite | unchanged; already the standard form |
| Crossover | each gene swapped with p = 1/n_params | **uniform crossover**, each gene swapped with p = 0.5 |
| Mutation | creep for ordered numbers; adaptive rate up to 0.5 | **random resetting** at a fixed `mutation_rate`, default 1/n_params |

Removed: `adaptive_mutation`, the creep step, and the short-lived `crossover_swap_rate` parameter.

**Outcome** (30 seeds, full results in 007):

| Problem | Custom operators | Standard operators | Leader |
|---|---|---|---|
| Rastrigin (10-D) | 6.28 (3rd) | **3.59** (3rd) | pymoo 0 |
| Holder (2-D) | −19.2085 (1st, tied) | −19.2036 (4th) | pymoo −19.2085 |
| Categorical | 100% (1st, tied) | 100% (1st, tied) | — |

The standard operators recombine better, so Rastrigin improves by 43%. They fine-tune less,
because random resetting has no small local steps, so Holder drops from 1st to 4th. The gap to
pymoo/DEAP on Rastrigin remains. The likely next step is still the one in *Next hypotheses*:
local refinement, for example an optional creep or a small-step mutation, measured the same way.
That's for a later release.

## Release impact

Changing the default changes search behaviour, so the same seed gives a different run:
- **If this lands before 2.0.0 is published** (013), it's just part of 2.0.0.
- **If it lands after,** it's a 2.1.0 with a CHANGELOG note.

## Acceptance criteria

- [x] Benchmark table for swap rates 1/n, 0.2 and 0.5 recorded here (30 seeds).
- [x] The cause was shown to be mostly elsewhere (step 1), and the spec was revised. The owner
      chose standard operators instead of further tuning (Resolution).
- [x] Spec 003 gate still passes. All tests pass.
- [ ] *Not met, accepted:* Rastrigin is not yet at DEAP's level (3.59 vs 0.64), and Holder
      dropped from 1st to 4th. Tracked as the local-refinement follow-up.
