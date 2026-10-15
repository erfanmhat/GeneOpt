# Implementation Plan: spec 012 — crossover mixing

(Earlier plans for specs 011 and 007 are complete; see git history of this file.)

## Overview

The 007 benchmark put GeneOpt 3rd on 10-D Rastrigin. The suspected cause is that crossover swaps
each gene with p = 1/n_params, so two parents exchange about one gene per mating. Spec 012 says
to **measure first, then change the default**.

## Architecture decisions

- Crossover moves into a pure function, `_crossover(p1, p2, swap_rate, crossover_rate, rng)`,
  like `_mutate`, so its mixing rate can be tested directly.
- New constructor parameter `crossover_swap_rate: float | None`. `None` means 1/n_params, which is
  exactly today's behaviour, so Task 1 changes no runs.
- The benchmark harness gets `--geneopt-swap-rate` so rates can be compared without editing code.

## Task list

- [x] Task 1: `crossover_swap_rate` parameter + `_crossover` extraction (no behaviour change)
- [x] Task 2: Benchmark 1/n vs 0.2 vs 0.5 on all three problems (30 seeds), results in spec 012
- [ ] Task 3: Set the default to the winning rate (or revise spec 012 if the cause is elsewhere); update README, CHANGELOG, benchmark table

### Checkpoint
- [ ] Spec 012 acceptance criteria met. Full suite + lint green. CI green on PR #1

## Risks

| Risk | Mitigation |
|---|---|
| A higher swap rate helps Rastrigin but hurts Holder/categorical | Task 2 measures all three. Task 3 picks a default only if none regresses beyond the 30-seed IQR |
| The cause is elsewhere (mutation, elitism) | Spec 012 step 1: revise the spec and stop. Don't tune blindly |
| Changing the default changes every seeded run | 2.0.0 isn't published yet, so it ships inside 2.0.0 (spec 012 § Release impact) |
