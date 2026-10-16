# 014 — Operator Parameters: Explicit and Standard

**Status:** proposed · **Severity:** 🟠 high (API + correctness of documentation) ·
**Depends on:** 003, 005 · **Pairs with:** 015 (benchmarks) · **Release:** before 2.0.0 is published

Three changes requested by the owner (2026-10-16), each a separate section with its own
acceptance criteria. All three change the public signature. 2.0.0 isn't published yet, so they
land inside 2.0.0 with no deprecation (011).

---

## A. Remove `initial_population_size`

### Problem
`initial_population_size` makes generation 0 a different size from every later generation.
It isn't part of the standard GA, and none of the compared libraries (PyGAD, DEAP, pymoo) has it.
It also costs more than it gives:
- **History** rows per generation are no longer constant, which complicates every plot and test (002).
- **Selection pressure** in generation 0 differs from the rest of the run, because the tournament
  size is the same but the population isn't.
- **Budget reasoning** gets harder ("generations × population" is no longer the evaluation count).
- It's a 1.x leftover (`start_number_of_population`) that was renamed instead of questioned.

### Design
- Delete the parameter. `population_size` applies to every generation, including generation 0.
- It's keyword-only, so passing it raises Python's own `TypeError` (011 policy, no shim).
- Delete `test_initial_population_size_applies_to_generation_zero_only` and its validation case.
- Docs: drop the README parameter row. Spec 005's signature loses the line. Spec 011's mapping
  row becomes `start_number_of_population` → *deleted; use `population_size`*.

### Acceptance criteria
- [ ] `GeneticAlgorithm(f, env, initial_population_size=10)` raises `TypeError`.
- [ ] Every generation in `history` has exactly `population_size` rows (existing H-1 test, now
      without exceptions).
- [ ] No mention of `initial_population_size` remains in `geneopt/`, README, CHANGELOG or specs,
      except 011's removal row and this spec.

---

## B. `mutation_rate` is required

### Problem
`mutation_rate=None` currently means 1/n_params. That's a formula hidden behind a default: two
problems with different parameter counts silently get different mutation strengths, and the
README can't state one number for "the default".

### Decision (owner)
**Required, no default.** Every call passes `mutation_rate` explicitly.

### Design
- Signature: `mutation_rate: float`, keyword-only, no default. Omitting it raises
  `TypeError: __init__() missing 1 required keyword-only argument: 'mutation_rate'`.
- Validation is unchanged: `0 ≤ mutation_rate ≤ 1`, and a `ValueError` names the parameter.
- **Meaning, stated once and identically in the docstring, the README and spec 003:**
  > *`mutation_rate`: the probability, per gene and per child, that random-resetting mutation
  > replaces that gene's value with a uniformly chosen **different** value.*
- The README gives guidance, not a default: "per-gene rates between 0.01 and 0.1 are common, and
  the right value depends on the problem (see the benchmark in 015)". Every example passes an
  explicit value.
- All tests, examples and the benchmark adapter pass `mutation_rate` explicitly. The benchmark's
  choice is disclosed in 015 rather than hidden in GeneOpt.

### Acceptance criteria
- [ ] Constructing without `mutation_rate` raises `TypeError` naming it.
- [ ] `mutation_rate=None` raises (`TypeError` or `ValueError`), so no implicit formula is left.
- [ ] The README, quickstart, categorical example and HTTP client all pass an explicit value.
      The README-embed test still passes.
- [ ] `test_E_3_*` asserts that the configured rate is used unchanged, with no default.

---

## C. Uniform crossover: one standard definition, standard names

### Problem
The implementation is correct, but its description isn't usable. The README says
*"probability that a pair of parents performs uniform crossover (each gene swapped with
p = 0.5)"*, which packs two different probabilities into one sentence. The name `crossover_rate`
doesn't say which of the two it controls.

The standard operator involves **two** distinct probabilities:

| Symbol | Name | Applies to | GeneOpt |
|---|---|---|---|
| p_c | **crossover probability** | a *pair* of parents: do they recombine at all? | parameter |
| p_s | **mixing ratio** (swap probability) | each *gene* of a recombining pair | fixed at **0.5** |

### Standard definition (Syswerda, 1989: *Uniform crossover in genetic algorithms*)

Given two selected parents A and B:
1. With probability **p_c**, the pair recombines. Otherwise the children are exact copies of A
   and B, and they still go through mutation.
2. If they recombine, then **independently for every gene i**:
   - child 1 takes gene i from A with probability 1/2, otherwise from B
   - child 2 takes the gene child 1 didn't take
   - so the two children are complementary
3. Both children then go through mutation.

**Why p_s is 0.5 and not a parameter.** 0.5 is the only value at which neither parent is
favoured. Any other value makes child 1 systematically closer to whichever parent was listed
first, so the result would depend on parent order, which carries no meaning. This is the textbook
operator, and it's what DEAP's `cxUniform(indpb=0.5)` does. The current code
(`swap = rng.random(shape) < 0.5`, then `np.where`) is exactly this definition. Only names, docs
and tests change.

### Design
- **Rename** `crossover_rate` → `crossover_probability`. Default **0.9**, the most common p_c in
  GA practice (the textbook range is 0.6–1.0). Validated in `[0, 1]`.
- The docstring of `_crossover` and the README row use the wording of steps 1–3.
- **Pairing:** parents come from independent tournaments, and consecutive winners are paired.
  The same individual may be drawn twice. That's standard (DEAP does the same), and such a pair
  just produces two clones. This is documented, not changed.
- Spec 003 §1 and the README "Parameters" table are updated to use p_c / p_s terminology.

### Open decision
- [ ] Should `crossover_probability` also be **required**, for symmetry with `mutation_rate`?
      Default if undecided: keep **0.9**. This is the one value the spec doesn't fix.

### Acceptance criteria
- [ ] `crossover_rate` raises `TypeError`; `crossover_probability` is accepted and validated.
- [ ] A statistical test with p_c = 1 and parents A = 0…, B = 1…: each child gene comes from A with
      frequency 0.5 ± 0.02, and child 2 is always the complement of child 1.
- [ ] A statistical test with long genomes: the fraction of pairs that recombine is p_c ± 0.03, for
      p_c ∈ {0.3, 0.9}.
- [ ] With p_c = 0, children equal their parents before mutation.
- [ ] The README describes p_c and p_s in separate sentences, matching steps 1–3. Any remaining
      wording that mixes the two (grep for "pair of parents performs") is gone.

---

## Order of work
1. A (smallest, no interactions).
2. B, together with updating every call site, including the benchmark adapter, whose value is
   chosen in 015.
3. C (rename + docs + statistical tests).
4. Re-run the 015 suite once, after all three.
