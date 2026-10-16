# Implementation Plan: spec 014 — operator parameters

(Earlier plans for specs 011, 007 and 012 are complete; see git history of this file.)

## Overview

Spec 014 is the only pending spec with no unmet prerequisite (015 measures 014's defaults, and 016
plots 015's results). It makes three changes to the public signature, inside 2.0.0, with no shims:
A removes `initial_population_size`; B sets `mutation_rate` to a fixed default of 0.05; C renames
`crossover_rate` → `crossover_probability` (default 0.9) and documents uniform crossover as p_c
plus a fixed p_s = 0.5.

## Architecture decisions

- No operator logic changes. `_crossover` is already Syswerda's uniform crossover; only its
  parameter name, docstring and tests change (014 C).
- Removed or renamed keywords raise Python's own `TypeError` (keyword-only signature, 011 policy).
- `mutation_rate=None` raises `ValueError` through the existing `_check`, so no formula survives.
- The README text is pasted from 014 § *README text (exact)*, not rewritten.
- The legacy benchmark is **not** re-run: its problems are retired by 015, which runs once after
  014 lands. The README images come from the quickstart, whose run changes with the new defaults,
  so they're regenerated in Task 4.

## Task list

- [ ] Task 1: Remove `initial_population_size` (014 A)
- [ ] Task 2: `mutation_rate` default 0.05, `None` rejected (014 B)
- [ ] Task 3: `crossover_rate` → `crossover_probability`, statistical crossover tests (014 C)
- [ ] Task 4: README "How a child is made", specs 003/005/007/011, CHANGELOG, regenerate images

### Checkpoint
- [ ] `pytest -q` (all markers), `ruff check . && ruff format --check . && mypy`, coverage gate
- [ ] 014's acceptance boxes ticked; grep shows no `initial_population_size`, `crossover_rate`,
      `1/n_params`-as-default or "pair of parents performs" outside 011's removal rows and 012/014
- [ ] Pushed; CI green on PR #1

## Risks
| Risk | Mitigation |
|---|---|
| Statistical tests flaky | fixed seeds, tolerances from 014 (±0.02, ±0.03) with large samples |
| Seeded test expectations depend on the old 1/n default | update them to the new default, never loosen asserts |
