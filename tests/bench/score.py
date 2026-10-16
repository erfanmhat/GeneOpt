"""Scoring and report for fair-suite run records (spec 015).

A record is one optimiser on one problem (see ``run.run_fair``). Precision is computed here, not
at run time, because a PBO reference can be the best value *any* optimiser found.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from suites import TARGETS

SUCCESS_TARGET = 1e-2  # success rates, head-start check and the 016 figures (owner decision)
GATE_SHARE = 0.9  # the main GeneOpt row must beat random search on this share of (function, D) pairs
MAIN_ROW = "geneopt[mr=1/D]"
SUITE_NAMES = {"mixint": "bbob-mixint", "pbo": "pbo"}


def problem(r) -> tuple:
    return r["suite"], r["function"], r["dim"], r["instance"]


def function_id(r) -> str:
    return f"{SUITE_NAMES[r['suite']]}/{'f' if r['suite'] == 'mixint' else 'F'}{r['function']}"


def where(r) -> str:
    return f"{function_id(r)} D={r['dim']} i{r['instance']}"


def precision(suite: str, value: float, ref: float) -> float:
    """f - f_opt for bbob-mixint; relative gap (ref - y) / |ref| for PBO (maximised)."""
    if suite == "mixint":
        return max(0.0, value - ref)
    gap = (ref - value) / abs(ref) if ref else ref - value
    return 0.0 if gap < 1e-12 else gap  # PBO values are sums of fractions: snap rounding noise to a hit


def references(records) -> dict[tuple, tuple[float, bool]]:
    """Problem -> (reference, came from observed values). PBO: max(stated optimum, best found)."""
    out: dict[tuple, tuple[float, bool]] = {}
    by_problem = defaultdict(list)
    for r in records:
        by_problem[problem(r)].append(r)
    for key, rs in by_problem.items():
        stated = next((r["reference"] for r in rs if r["reference"] is not None), None)
        if key[0] == "mixint":
            if stated is not None:
                out[key] = (stated, False)
            continue
        found = [r["best"] for r in rs if r["best"] is not None]
        best = max(found) if found else None
        if stated is None and best is None:
            continue
        if stated is None or (best is not None and best > stated):
            out[key] = (best, True)
        else:
            out[key] = (stated, False)
    return out


def evaluations_to_targets(r, ref: float) -> dict[float, int | None]:
    hits: dict[float, int | None] = dict.fromkeys(TARGETS[r["suite"]])
    for evaluation, value in r["trace"]:
        p = precision(r["suite"], value, ref)
        for t in hits:
            if hits[t] is None and p <= t:
                hits[t] = evaluation
    return hits


def runs_rows(records) -> list[dict]:
    """Long format, one row per (run, target): the input of the 016 figures."""
    refs = references(records)
    rows = []
    for r in records:
        if problem(r) not in refs:
            continue
        for target, hit in evaluations_to_targets(r, refs[problem(r)][0]).items():
            rows.append({
                "algorithm": r["optimiser"], "function_id": function_id(r),
                "suite": r["suite"], "dimension": r["dim"], "instance": r["instance"], "target": target,
                "evaluations_to_target": hit, "success_bool": hit is not None,
                "elapsed_time": r["seconds"], "evaluations": r["evaluations"], "budget": r["budget"],
            })  # fmt: skip
    return rows


def ecdf_area(rows) -> float:
    """Mean share of the log-budget axis [D, budget] after each (run, target) hit; 0 if missed."""
    areas = []
    for row in rows:
        hit, budget, dim = row["evaluations_to_target"], row["budget"], row["dimension"]
        if hit is None:
            areas.append(0.0)
        else:
            areas.append(min(1.0, max(0.0, math.log(budget / max(hit, dim)) / math.log(budget / dim))))
    return sum(areas) / len(areas) if areas else 0.0


def ranks(values: dict[str, float]) -> dict[str, float]:
    """1 = lowest value; ties share the average rank."""
    order = sorted(values, key=values.__getitem__)
    out, i = {}, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def score(records) -> dict:
    records = list(records)
    refs = references(records)
    optimisers = sorted({r["optimiser"] for r in records})
    finals: dict = defaultdict(
        lambda: defaultdict(dict)
    )  # suite -> (function, D) -> optimiser -> [precision]
    per_instance: dict = defaultdict(dict)  # problem -> optimiser -> precision
    for key in sorted({problem(r) for r in records}):
        for o in optimisers:
            per_instance[key][o] = math.inf  # missing or crashed: unsolved
    for r in records:
        key = problem(r)
        if key in refs and r["best"] is not None:
            per_instance[key][r["optimiser"]] = precision(r["suite"], r["best"], refs[key][0])
    for (suite, function, dim, _), by_opt in per_instance.items():
        for o, p in by_opt.items():
            finals[suite][(function, dim)].setdefault(o, []).append(p)
    final = {
        suite: {
            pair: {o: statistics.median(ps) for o, ps in by_opt.items()} for pair, by_opt in pairs.items()
        }
        for suite, pairs in finals.items()
    }
    rows = runs_rows(records)
    area, success, used = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for suite in final:
        for o in optimisers:
            mine = [x for x in rows if x["suite"] == suite and x["algorithm"] == o]
            area[suite][o] = ecdf_area(mine)
            at = [x for x in mine if x["target"] == success_target(suite)]
            success[suite][o] = sum(x["success_bool"] for x in at) / len(at) if at else 0.0
            runs = [r for r in records if r["suite"] == suite and r["optimiser"] == o]
            # below 1: the run stalled (only re-proposed known points until the call cap) or crashed
            used[suite][o] = statistics.median(r["evaluations"] / r["budget"] for r in runs) if runs else 0.0
    mean_rank = {
        suite: {o: statistics.mean(ranks(medians)[o] for medians in pairs.values()) for o in optimisers}
        for suite, pairs in final.items()
    }
    head_starts = [
        r for r in records
        if problem(r) in refs and r["early_best"] is not None
        and precision(r["suite"], r["early_best"], refs[problem(r)][0]) <= SUCCESS_TARGET
    ]  # fmt: skip
    return {
        "optimisers": optimisers, "references": refs, "final": final, "per_instance": per_instance,
        "ecdf_area": dict(area), "success": dict(success), "budget_used": dict(used), "mean_rank": mean_rank,
        "head_starts": head_starts, "crashes": [r for r in records if r["error"]],
    }  # fmt: skip


def success_target(suite: str) -> float:
    """The suite's own target value closest to SUCCESS_TARGET (COCO's grid is 10^(2 - 0.2k))."""
    return min(TARGETS[suite], key=lambda t: abs(t - SUCCESS_TARGET))


def gate(s: dict) -> list[str]:
    """Spec 015 hard gate, per suite: the main row beats random search on >= 90% of (function, D)
    pairs (both solved counts as a win) and has a larger ECDF area."""
    failures = []
    for suite, pairs in s["final"].items():
        if MAIN_ROW not in s["optimisers"] or "random" not in s["optimisers"]:
            continue
        wins = sum(m[MAIN_ROW] < m["random"] or m[MAIN_ROW] == m["random"] == 0 for m in pairs.values())
        if wins < GATE_SHARE * len(pairs):
            failures.append(
                f"{suite}: beats random on {wins}/{len(pairs)} (function, D) pairs, needs {GATE_SHARE:.0%}"
            )
        g, r = s["ecdf_area"][suite][MAIN_ROW], s["ecdf_area"][suite]["random"]
        if not g > r:
            failures.append(f"{suite}: ECDF area {g:.3f} is not above random's {r:.3f}")
    return failures


def wilcoxon(s: dict, suite: str) -> list[tuple[str, int, int, int, float | None]]:
    """Main row vs each optimiser over problem instances: (optimiser, wins, ties, losses, p)."""
    try:
        from scipy.stats import wilcoxon as test
    except ImportError:  # p-values need the bench requirements; counts don't
        test = None
    out = []
    keys = [k for k in s["per_instance"] if k[0] == suite]
    for o in s["optimisers"]:
        if o == MAIN_ROW:
            continue
        pairs = [(s["per_instance"][k][MAIN_ROW], s["per_instance"][k][o]) for k in keys]
        wins = sum(a < b for a, b in pairs)
        ties = sum(a == b for a, b in pairs)
        finite = [(a, b) for a, b in pairs if math.isfinite(a) and math.isfinite(b) and a != b]
        p = (
            float(test([a for a, _ in finite], [b for _, b in finite]).pvalue)
            if test and len(finite) >= 5
            else None
        )
        out.append((o, wins, ties, len(pairs) - wins - ties, p))
    return out


def compact(sizes) -> str:
    """[2, 2, 4, 4, 256] -> '2 ×2, 4 ×2, 256'."""
    groups: list[list[int]] = []
    for size in sizes:
        if groups and groups[-1][0] == size:
            groups[-1][1] += 1
        else:
            groups.append([size, 1])
    return ", ".join(f"{v} ×{n}" if n > 1 else str(v) for v, n in groups)


def _fmt(v: float) -> str:
    return "∞" if math.isinf(v) else f"{v:.3g}"


def report(records, header: str = "") -> str:
    records = sorted(records, key=lambda r: (problem(r), r["optimiser"]))  # same report however sharded
    s = score(records)
    opts = s["optimisers"]
    lines = [header, ""] if header else []
    lines += [
        "## Configuration",
        "",
        "Competitors: library defaults plus population 50. GeneOpt: every parameter explicit.",
        "",
    ]
    lines += [
        "| suite | D | values per gene | budget | row | GeneOpt parameters |",
        "|---|---|---|---|---|---|",
    ]
    seen = set()
    for r in sorted(records, key=lambda r: (r["suite"], r["dim"], r["optimiser"])):
        key = (r["suite"], r["dim"], r["optimiser"])
        if r["config"] and key not in seen:
            seen.add(key)
            params = ", ".join(f"`{k}={v}`" for k, v in r["config"].items())
            sizes = compact(r["sizes"])
            lines.append(
                f"| {r['suite']} | {r['dim']} | {sizes} | {r['budget']} | {r['optimiser']} | {params} |"
            )
    observed = sorted(k for k, (_, obs) in s["references"].items() if obs)
    if observed:
        lines += ["", "PBO references taken from the best value found (no stated optimum, or it was beaten): "
                  + ", ".join(f"F{f} n={d} i{i}" for _, f, d, i in observed)]  # fmt: skip
    for suite, pairs in s["final"].items():
        lines += [
            "",
            f"## {suite}",
            "",
            "Median final precision over instances (lower is better; best in bold).",
            "",
        ]
        lines += ["| function | D | " + " | ".join(opts) + " |", "|---|---|" + "---|" * len(opts)]
        for (function, dim), med in sorted(pairs.items()):
            best = min(med.values())
            cells = [f"**{_fmt(med[o])}**" if med[o] == best else _fmt(med[o]) for o in opts]
            lines.append(
                f"| {'f' if suite == 'mixint' else 'F'}{function} | {dim} | " + " | ".join(cells) + " |"
            )
        lines.append(
            "| **mean rank** | | " + " | ".join(f"{s['mean_rank'][suite][o]:.2f}" for o in opts) + " |"
        )
        lines.append(
            "| **ECDF area** | | " + " | ".join(f"{s['ecdf_area'][suite][o]:.3f}" for o in opts) + " |"
        )
        lines.append(f"| **success at {success_target(suite):.2g}** | | "
                     + " | ".join(f"{s['success'][suite][o]:.0%}" for o in opts) + " |")  # fmt: skip
        lines.append(
            "| **budget used (median)** | | "
            + " | ".join(f"{s['budget_used'][suite][o]:.0%}" for o in opts)
            + " |"
        )
        if MAIN_ROW in opts:
            lines += [
                "",
                f"{MAIN_ROW} against each optimiser, per problem instance (Wilcoxon signed-rank):",
                "",
            ]
            lines += ["| optimiser | wins | ties | losses | p |", "|---|---|---|---|---|"]
            for o, w, t, lo, p in wilcoxon(s, suite):
                lines.append(f"| {o} | {w} | {t} | {lo} | {'—' if p is None else f'{p:.3g}'} |")
    lines += ["", "## Head-start check", ""]
    if s["head_starts"]:
        lines += [
            f"- FLAG: {r['optimiser']} on {where(r)} reached {SUCCESS_TARGET:g} in its first 50 evaluations"
            for r in s["head_starts"]
        ]
    else:
        lines.append(f"No run reached the {SUCCESS_TARGET:g} target within its first 50 evaluations.")
    if s["crashes"]:
        lines += ["", "## Crashes", ""] + [
            f"- {r['optimiser']} on {where(r)}: {r['error']}" for r in s["crashes"]
        ]
    failures = gate(s)
    lines += ["", "**Gate (main GeneOpt row beats random on ≥ 90% of pairs, higher ECDF area):** "
              + ("PASS" if not failures else "FAIL: " + "; ".join(failures))]  # fmt: skip
    return "\n".join(lines)
