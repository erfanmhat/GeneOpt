# 014 — Operator Parameters: Explicit and Standard

**Status:** implemented (2026-10-16) · **Severity:** 🟠 high (API + correctness of documentation) ·
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
- [x] `GeneticAlgorithm(f, env, initial_population_size=10)` raises `TypeError`.
- [x] Every generation in `history` has exactly `population_size` rows (existing H-1 test, now
      without exceptions).
- [x] No mention of `initial_population_size` remains in `geneopt/`, README, CHANGELOG or specs,
      except 011's removal row and this spec.

---

## B. `mutation_rate`: a fixed default of 0.05

### Problem
`mutation_rate=None` currently means 1/n_params. That's a formula hidden behind a default: two
problems with different parameter counts silently get different mutation strengths, and the
README can't state one number for "the default".

### Decision (owner, revised 2026-10-16)
**Default `mutation_rate=0.05`**, a plain number with no formula behind it. This replaces the
earlier "required, no default" decision.

### Design
- Signature: `mutation_rate: float = 0.05`, keyword-only. No `None`, no formula.
- Validation is unchanged: `0 ≤ mutation_rate ≤ 1`, and a `ValueError` names the parameter.
- **Meaning, stated once and identically in the docstring, the README and spec 003:**
  > *`mutation_rate`: the probability, per gene and per child, that random-resetting mutation
  > replaces that gene's value with a uniformly chosen **different** value.*
- 0.05 sits inside the common 0.01–0.1 range. On a 20-parameter problem it mutates one gene per
  child on average; on 100 parameters, five. The README says so, so users with many parameters
  know when to lower it.
- The benchmark adapter (015) passes its value explicitly and discloses it.

### Acceptance criteria
- [x] Omitting `mutation_rate` gives a measured per-gene mutation frequency of 0.05 ± 10%, for
      both a 2-parameter and a 50-parameter environment.
- [x] `mutation_rate=None` raises `ValueError` naming the parameter, so no implicit formula is left.
- [x] No mention of `1/n_params` as a mutation default remains in `geneopt/`, README or spec 003.
- [x] `test_E_3_*` asserts that the configured rate is used unchanged during the run.

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

### Decision (owner, 2026-10-16)
`crossover_probability` keeps its default of **0.9**. It isn't required.

### Acceptance criteria
- [x] `crossover_rate` raises `TypeError`; `crossover_probability` is accepted and validated.
- [x] A statistical test with p_c = 1 and parents A = 0…, B = 1…: each child gene comes from A with
      frequency 0.5 ± 0.02, and child 2 is always the complement of child 1.
- [x] A statistical test with long genomes: the fraction of pairs that recombine is p_c ± 0.03, for
      p_c ∈ {0.3, 0.9}.
- [x] With p_c = 0, children equal their parents before mutation.
- [x] The README rows below replace the current `crossover_rate` and `mutation_rate` rows word
      for word, and the `initial_population_size` row is deleted (A). Any wording that mixes p_c
      and p_s (grep for "pair of parents performs") is gone.

---

## D. `mutation_rate="auto"`: 1/D on request

### Problem
A fixed 0.05 doesn't scale with the number of parameters D. For D = 2 it gives 0.1 mutations per
child, so the search stalls once the population converges: the 2-parameter bowl test stopped one
grid step from the optimum. For D = 100 it gives five mutations per child, which disrupts good
individuals.

### Decision (owner, 2026-10-16)
Keep 0.05 as the default and add **`mutation_rate="auto"`**, which means **1/D**: one mutation
per child on average, whatever D is (the standard 1/L rule). It is opt-in, not a hidden default.

### Design
- Signature: `mutation_rate: float | Literal["auto"] = 0.05`.
- `"auto"` resolves once in the constructor to `1 / len(environment)` and stays fixed for the run.
  With D = 1 that's 1.0, so every child is mutated.
- Only the exact string `"auto"` is accepted. Anything else that isn't a real number in [0, 1]
  (including `None`, `True`, `"Auto"`, `"1/n"`) raises `ValueError` naming the parameter.

### Acceptance criteria
- [x] `"auto"` gives a measured per-gene mutation frequency of 1/D ± 10% for D = 2 and D = 50.
- [x] `None`, `True`, `"Auto"` and `"1/n"` raise `ValueError` naming `mutation_rate`.
- [x] README row and "How a child is made" mention `"auto"` (text below).

---

## README text (exact)

Parameters table rows:

| parameter | default | meaning |
|---|---|---|
| `crossover_probability` | `0.9` | chance that two selected parents recombine. If they don't, the children start as copies of the parents. |
| `mutation_rate` | `0.05` | chance, for each gene of each child, that the gene is replaced by a different random value from its list. `"auto"` uses 1 / (number of parameters). |

Paragraph under the table, headed **How a child is made**:

> 1. **Select** two parents, each the winner of a tournament of `tournament_size` random
>    individuals.
> 2. **Recombine** with probability `crossover_probability`: for every gene separately, a fair
>    coin decides which parent child 1 takes it from, and child 2 takes it from the other parent.
>    The coin is always fair (50/50) so neither parent is favoured; this isn't a setting.
> 3. **Mutate** every gene of both children with probability `mutation_rate`.
>
> With the default 0.05, a child with 20 parameters gets one mutation on average, and one with
> 100 parameters gets five. For about one mutation per child whatever the number of parameters,
> pass `mutation_rate="auto"`, which uses 1 / (number of parameters).

---

## Order of work
1. A (smallest, no interactions).
2. B, together with updating every call site, including the benchmark adapter, whose value is
   chosen in 015.
3. C (rename + docs + statistical tests).
4. Re-run the 015 suite once, after all three.
