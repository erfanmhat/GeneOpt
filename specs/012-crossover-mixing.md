# 012 — Crossover Mixes Too Little on Many-Parameter Problems

**Status:** proposed · **Severity:** 🟠 high (search quality) · **Found by:** the 007 benchmark ·
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

## Release impact

Changing the default changes search behaviour, so the same seed gives a different run:
- **If this lands before 2.0.0 is published** (013), it's just part of 2.0.0.
- **If it lands after,** it's a 2.1.0 with a CHANGELOG note.

## Acceptance criteria

- [ ] Benchmark table for swap rates 1/n, 0.2 and 0.5 recorded here (30 seeds).
- [ ] With the new default, Rastrigin median best is no worse than DEAP's (0.64) or the cause is
      shown to be elsewhere and this spec revised.
- [ ] Holder and categorical results don't get worse (median and hit rate within the 30-seed IQR).
- [ ] Spec 003 gate still passes. All existing tests pass. A validation test covers the new parameter.
