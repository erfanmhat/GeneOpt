# 003 — Encoding & Genetic Operators

**Status:** proposed · **Severity:** 🟠 high · **Fixes:** E-1 … E-6 · **Depends on:** 004 (single loop)

## Problem

### E-1: decode bias

Each parameter gets `ceil(log2(n))` bits, and decoded values ≥ `n` are clamped to `n-1`
(`GeneticAlgorithm.py:213-214`). For non-power-of-two sizes, the last value takes all the
overflow:

```
env = {"v": list("abcde")}  # 5 values → 3 bits → 8 codes, 4 of which map to "e"
8000 random draws → {'a': 976, 'b': 1002, 'c': 1010, 'd': 1027, 'e': 3985}
```

The initial population is therefore biased toward the upper boundary of every non-power-of-two
range. For bounded searches, that edge is often the worst region.

### E-2 / E-3: mutation is much weaker than configured, and adapts the wrong way

```python
if rng.random() < rate:              # gate per parameter
    for bit in block:
        if rng.random() < rate:      # gate per bit  → effective per-bit p = rate²
```

- `mutation_rate=0.06` actually flips a given bit with p = **0.0036**.
- `adaptive = rate / diversity`, where `diversity` is the *mean Hamming distance in bits*
  (0 … n_bits). With 24 bits and a fresh population, diversity ≈ 12, so the rate becomes
  0.005 and then 0.005² per bit. Near convergence (diversity → 0.1) the rate can go **above 1**.
- Result: the population collapses within a few generations. One measured run made
  **105 objective calls across 5,000 slots**, so 98% of individuals were clones of already
  evaluated genomes.

### Other operator issues

- **E-4** `_selection` takes one random index and then `k` more without replacement, so it's
  a tournament of k+1. When `k > len(population)` the `choice` raises, gets logged as an error,
  and is retried with replacement, once per selection (hundreds of log lines per generation).
  `_selection` and `_selection_population_scores` are the same function written twice.
- **E-5** No elitism. With `r_cross=0.99` and mutation, the best-ever genome usually leaves
  the population. The run's "best" only survives because it's tracked outside the population.
- **E-6** `early_stopping_rounds < (gen - last_best)` stops one round late. It should be `>=`.
- Odd population sizes are silently decremented (`100 → 100`, `101 → 100`) with no log.

## Design

### 1. Integer genomes: drop the bit strings

The binary layer adds bias (E-1), Hamming cliffs (index 127→128 flips 8 bits), wasted codes and
a 30-line decoder. The crossover *already* swaps whole parameter blocks, so the GA effectively
treats parameters as atomic genes. Make that explicit:

```python
genome: np.ndarray[int]                 # shape (n_params,), genome[i] ∈ [0, len(env[i]))
decode = lambda g: {k: env[k][i] for k, i in zip(keys, g)}
init   = rng.integers(0, sizes, size=(pop, n_params))   # uniform, no bias (E-1 gone)
```

- **Crossover** (unchanged semantics): for each gene, swap between children with
  `p = 1 / n_params`, gated by `r_cross`. This is vectorised as a boolean mask.
- **Mutation**, per gene with probability `mutation_rate` (one gate, which fixes E-2):
  - *ordered numeric* parameters (detected: every value is a real number and the list is
    monotonic): **creep**, `i ± round(N(0, σ·n))`, clipped, with `σ = 0.1` by default. This
    gives small local steps, which is what the bit flips were trying to do.
  - *anything else*: resample uniformly from the other `n-1` indices.
- The cache key (001) is `tuple(genome)` with no conversion.
- `_init_environment_decode`, `_n_bits`, `_environment_param_n_bits` and the bit loops are
  deleted.

> Binary encoding is **removed** in 2.0.0, with no option to keep it (011). The "minimum fix"
> (`(n-1).bit_length()`, Gray code, `index % n`) still biases sizes that aren't powers of two,
> so it isn't worth carrying.

### 2. Adaptive mutation that adapts the right way

```python
d = normalised_diversity(pop)               # [0, 1], see 002
ceiling = max(MAX_MUTATION_RATE, base)    # MAX_MUTATION_RATE = 0.5, a module constant
rate = base + (ceiling - base) * (1 - d)   # low diversity → more mutation
```

- `adaptive_mutation: bool = True`. The ceiling is a constant, not a parameter. Nobody has asked
  to tune it.
- Clamped to `[base, ceiling]` by construction, so it can't exceed 1.

### 3. Selection: one function, exact tournament

```python
def tournament(scores, k, rng, better):   # returns index
    idx = rng.choice(len(scores), size=k, replace=k > len(scores))
    return idx[best_of(scores[idx], better)]
```

- Exactly `k` candidates. With-replacement only when it's required, with no exception or log.
- One `rng.choice` per tournament, then the winners are reduced with a single `argmax` over the
  `(m, k)` candidate matrix. Drawing *without replacement* per row has no single-call numpy
  equivalent, and m ≈ population size, so the loop costs about a millisecond.

### 4. Elitism

- `elitism: int = 1`. The top-`e` unique genomes are copied unchanged into the next
  generation, and the remaining `pop - e` slots are filled by selection, crossover and mutation.
- If `pop - e` is odd, the last crossover child is dropped. Population size is never silently
  changed, so the odd-size decrement goes away.

### 5. Comparisons

- Replace `_maximize`/`_minimize` + `_comparison_func` with `sign = +1 | -1` and work on
  `sign * score` internally. Every "better" check becomes `>`. That removes about 15 branches
  of the form `if maximize … elif minimize … else` (some of which are unreachable `else: 0`).
- E-6: stop when `gen - last_improvement >= early_stopping_rounds`.
- `target_score`: stop when `sign*score >= sign*target`, checked once per generation after
  scoring, not in the middle of a generation.

## Benchmark gate

These changes alter search behaviour, so they ship with numbers. The comparison is against
**other optimisation libraries**, not against 1.x. The harness, the competitors and the fairness
rules are in [007 § Benchmarks](007-tests-and-ci.md#benchmarks-testsbenchrunpy).

| Problem | Budget | Metric |
|---|---|---|
| Holder table (2-D, 4096²) | 2,000 unique evals | median best over 30 seeds |
| Rastrigin (10-D, 256 values each) | 5,000 unique evals | median best over 30 seeds |
| Categorical toy (6 params × 5 labels, known optimum) | 500 unique evals | hit rate of optimum |

**Gate (hard):** GeneOpt beats random search on every problem, by median best or by hit rate.
A GA that can't beat random sampling at an equal budget is broken.

**Reported, not gated:** GeneOpt's rank among the other libraries. A regression in rank between
releases is investigated, but it doesn't block a release on its own. Model-based optimisers
(TPE, NGOpt) are expected to win on small budgets.

## Acceptance criteria

- [ ] 8,000 random initial genes over 5 values: each value 1,600 ± 150.
- [ ] With `adaptive_mutation=False`, measured per-gene mutation frequency ≈ `mutation_rate` (±10%).
- [ ] Adaptive rate stays within `[base, max(0.5, base)]` for any population.
- [ ] Tournament with `k > pop` logs nothing and returns a valid index.
- [ ] With `elitism ≥ 1`, best-of-population is monotonic across generations.
- [ ] `early_stopping_rounds=3` stops exactly 3 generations after the last improvement.
- [ ] `population_size=101` → 101 individuals per generation.
- [ ] Benchmark gate passes (beats random search on all three problems).
