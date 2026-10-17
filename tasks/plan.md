# Implementation Plan: everything before the 2.0.0 release

(Earlier plans for specs 011, 007, 012, 014 and 015 are complete; see git history of this file.)

Owner instruction (2026-10-17): finish all open items before releasing 2.0.0. The release itself
(merge PR #1, push the `v2.0.0` tag, which publishes to PyPI) still waits for the owner's explicit go.

## Task list

- [x] Task 1: Housekeeping. Refresh the specs/README status rows and spec 013, and add
      `bench-results/`, `exdata/` and `.coverage` to `.gitignore` (the PR #1 description moves to Task 8)
- [x] Task 2: Test the objective-without-signature fallback in `core.py`
- [x] Task 3: `History` stores genomes as `int32` and gets `save(path)` / `History.load(path)` (spec 002 §1)
- [x] Task 4: `callback` progress hook with `GenerationStats`; returning True stops the run (spec 005 §5)
- [x] Task 5: Spec 016 figures: `tests/bench/plots.py` (palette, ECDF, performance profile,
      Pareto), with hand-computed tests and every *Missing data* rule
- [x] Task 6: Spec 016 harness: `run.py` writes PDF + PNG and embeds them; `--plots-only DIR`
- [x] Task 7: Spec 016 checks. Cross-check our ECDF data against `cocopp` on the 015 run;
      greyscale and 3.5 in readability; figures added to the 2026-10-16 report
- [x] Task 8: Docs. README, CHANGELOG, spec statuses (002, 005, 016, README map); push; CI green

### Not in this plan: needs an owner decision
- Local refinement (spec 012). It would add a non-standard operator, which reverses the
  "standard operators" decision. Options go to the owner after Task 8.

### Checkpoint
- [x] `pytest -q` (core, plots and bench envs), ruff, mypy, coverage gate
- [x] PR #1 CI green; no benchmark run on GitHub Actions (local only)
