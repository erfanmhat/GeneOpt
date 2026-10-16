"""Benchmark GeneOpt against other optimisation libraries (spec 007 § Benchmarks).

    pip install -e . -r tests/bench/requirements.txt
    python tests/bench/run.py --seeds 30 [--only geneopt,random] [--problems holder] [--quick] [--out r.md]

Every optimiser searches the same integer index space under the same budget of *unique*
evaluations. Repeats are memoised and free for everyone. Each run stops after N unique
evaluations, or after 50 x N total calls for an optimiser that only re-proposes known points.
All problems are minimised.
"""

from __future__ import annotations

import argparse
import functools
import math
import os
import random
import statistics
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

POPULATION = 50  # shared by every GA library (fairness rule 3)
CALL_CAP = 50  # total calls allowed per unique evaluation
EARLY = 50  # head-start check: best value within the first EARLY evaluations (spec 015 rule 7)

# Spec 015 § GeneOpt configuration: every operator parameter is explicit, none is a library default.
GENEOPT_OPERATORS = {
    "population_size": POPULATION,
    "crossover_probability": 0.9,
    "tournament_size": 3,
    "elitism": 1,
}
GENEOPT_MUTATION = {  # (suite, D) -> row -> mutation_rate; main row 1/D, sensitivity rows 0.5/D and 2/D
    ("mixint", 10): {"1/D": 0.1, "0.5/D": 0.05, "2/D": 0.2},
    ("mixint", 20): {"1/D": 0.05, "0.5/D": 0.025, "2/D": 0.1},
    ("pbo", 64): {"1/D": 0.015625, "0.5/D": 0.0078125, "2/D": 0.03125},
    ("pbo", 100): {"1/D": 0.01, "0.5/D": 0.005, "2/D": 0.02},
}
MAIN_ROW = "geneopt[mr=1/D]"


class Exhausted(BaseException):
    """Budget spent. BaseException so a library's ``except Exception`` cannot swallow it."""


class Budget:
    """Memoising objective wrapper that stops the run after ``n_unique`` distinct points."""

    def __init__(self, fn: Callable[[tuple[int, ...]], float], n_unique: int, max_calls: int):
        self.fn, self.n_unique, self.max_calls = fn, n_unique, max_calls
        self.memo: dict[tuple[int, ...], float] = {}
        self.calls = 0
        self.best = math.inf

    def __call__(self, index) -> float:
        key = tuple(int(i) for i in index)
        self.calls += 1
        if self.calls > self.max_calls:
            raise Exhausted
        if key in self.memo:
            return self.memo[key]
        if len(self.memo) >= self.n_unique:
            raise Exhausted
        value = float(self.fn(key))
        self.memo[key] = value
        self.best = min(self.best, value)
        return value


# --- problems -----------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    name: str
    sizes: tuple[int, ...]
    budget: int
    categorical: bool  # unordered choices: libraries that support it get categorical params
    target: float | None = None  # known optimum; reported as hit rate instead of median best

    def objective(self, index: tuple[int, ...]) -> float:
        return OBJECTIVES[self.name](index)


HOLDER_GRID = np.linspace(-10, 10, 4096)
RASTRIGIN_GRID = np.linspace(-5.12, 5.12, 256)
# Shifted Rastrigin: the optimum sits on off-centre grid points. Plain Rastrigin's optimum is the
# centre of the box, which hands a free win to any optimiser that starts at the centre (NGOpt does).
RASTRIGIN_OPTIMUM = (37, 201, 90, 150, 12, 240, 64, 180, 110, 220)
RASTRIGIN_SHIFT = RASTRIGIN_GRID[list(RASTRIGIN_OPTIMUM)]
CATEGORICAL_TARGET = (2, 1, 3, 0, 4, 2)  # labels "c b d a e c"


def holder(index):
    x1, x2 = HOLDER_GRID[index[0]], HOLDER_GRID[index[1]]
    return -abs(math.sin(x1) * math.cos(x2) * math.exp(abs(1 - math.sqrt(x1 * x1 + x2 * x2) / math.pi)))


def rastrigin(index):
    x = RASTRIGIN_GRID[list(index)] - RASTRIGIN_SHIFT
    return float(10 * len(x) + np.sum(x * x - 10 * np.cos(2 * np.pi * x)))


def categorical(index):
    return -float(sum(i == t for i, t in zip(index, CATEGORICAL_TARGET, strict=True)))


OBJECTIVES = {"holder": holder, "rastrigin": rastrigin, "categorical": categorical}
PROBLEMS = {
    "holder": Problem("holder", (4096, 4096), 2000, categorical=False),
    "rastrigin": Problem("rastrigin", (256,) * 10, 5000, categorical=False),
    "categorical": Problem("categorical", (5,) * 6, 500, categorical=True, target=-6.0),
}


# --- optimisers: each searches index vectors and calls f until Exhausted ---------


def run_geneopt(p: Problem, f: Budget, seed: int, mutation_rate: float | None = None) -> None:
    import geneopt

    names = [f"p{i}" for i in range(len(p.sizes))]
    env = {n: list(range(s)) for n, s in zip(names, p.sizes, strict=True)}
    ga = geneopt.GeneticAlgorithm(
        lambda **kw: f(tuple(kw[n] for n in names)),
        env,
        maximize=False,
        mutation_rate=1 / len(p.sizes) if mutation_rate is None else mutation_rate,
        **GENEOPT_OPERATORS,
        # GeneOpt answers repeats from its own cache, so they never reach the call cap; bound it by
        # the same number of proposals the cap allows everyone else (CALL_CAP x N individuals)
        generations=max(1, CALL_CAP * f.n_unique // POPULATION),
        seed=seed,
    )
    ga.start()


def run_random(p: Problem, f: Budget, seed: int) -> None:
    rng = np.random.default_rng(seed)
    while True:
        f(rng.integers(0, p.sizes))


def run_pygad(p: Problem, f: Budget, seed: int) -> None:
    import pygad

    ga = pygad.GA(
        num_generations=10**9,
        num_parents_mating=POPULATION // 2,
        sol_per_pop=POPULATION,
        num_genes=len(p.sizes),
        gene_space=[list(range(s)) for s in p.sizes],
        gene_type=int,
        fitness_func=lambda _ga, solution, _idx: -f(solution),  # pygad maximises
        random_seed=seed,
        suppress_warnings=True,
    )
    ga.run()


def run_deap(p: Problem, f: Budget, seed: int) -> None:
    from deap import algorithms, base, creator, tools

    if not hasattr(creator, "BenchIndividual"):
        creator.create("BenchFitness", base.Fitness, weights=(-1.0,))
        creator.create("BenchIndividual", list, fitness=creator.BenchFitness)
    rng = random.Random(seed)
    random.seed(seed)  # DEAP's operators use the global random module
    tb = base.Toolbox()
    tb.register(
        "individual",
        lambda: creator.BenchIndividual(rng.randrange(s) for s in p.sizes),
    )
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate", lambda ind: (f(ind),))
    tb.register("mate", tools.cxTwoPoint)
    tb.register(
        "mutate", tools.mutUniformInt, low=[0] * len(p.sizes), up=[s - 1 for s in p.sizes], indpb=0.05
    )
    tb.register("select", tools.selTournament, tournsize=3)
    algorithms.eaSimple(tb.population(n=POPULATION), tb, cxpb=0.5, mutpb=0.2, ngen=10**9, verbose=False)


def run_pymoo(p: Problem, f: Budget, seed: int) -> None:
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.optimize import minimize

    class Wrapped(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(p.sizes), n_obj=1, xl=0, xu=np.array(p.sizes) - 1, vtype=int)

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = f(x)

    # pymoo's documented recipe for integer variables
    algorithm = GA(
        pop_size=POPULATION,
        sampling=IntegerRandomSampling(),
        crossover=SBX(vtype=float, repair=RoundingRepair()),
        mutation=PM(vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )
    minimize(Wrapped(), algorithm, ("n_eval", 10**12), seed=seed, verbose=False)


def run_optuna(p: Problem, f: Budget, seed: int) -> None:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        if p.categorical:
            index = [trial.suggest_categorical(f"p{i}", list(range(s))) for i, s in enumerate(p.sizes)]
        else:
            index = [trial.suggest_int(f"p{i}", 0, s - 1) for i, s in enumerate(p.sizes)]
        return f(index)

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=10**9)


def run_nevergrad(p: Problem, f: Budget, seed: int) -> None:
    import nevergrad as ng

    # vectorised parametrisations: same search space, and far faster than a Tuple of n parameters
    if p.categorical and len(set(p.sizes)) == 1:
        param = ng.p.Choice(list(range(p.sizes[0])), repetitions=len(p.sizes))
    elif p.categorical:
        param = ng.p.Tuple(*[ng.p.Choice(list(range(s))) for s in p.sizes])
    else:
        upper = np.array(p.sizes) - 1
        param = ng.p.Array(init=upper / 2, lower=0, upper=upper).set_integer_casting()
    param.random_state = np.random.RandomState(seed)
    optimizer = ng.optimizers.NGOpt(parametrization=param, budget=f.n_unique)
    while True:
        candidate = optimizer.ask()
        optimizer.tell(candidate, f(candidate.value))


OPTIMISERS: dict[str, Callable[[Problem, Budget, int], None]] = {
    "geneopt": run_geneopt,
    "random": run_random,
    "pygad": run_pygad,
    "deap": run_deap,
    "pymoo": run_pymoo,
    "optuna": run_optuna,
    "nevergrad": run_nevergrad,
}


def optimisers(suite: str, dim: int) -> dict[str, Callable[[Problem, Budget, int], None]]:
    """The competitors plus one GeneOpt row per mutation rate of spec 015's table."""
    rows = {
        f"geneopt[mr={row}]": functools.partial(run_geneopt, mutation_rate=rate)
        for row, rate in GENEOPT_MUTATION[(suite, dim)].items()
    }
    return rows | {name: fn for name, fn in OPTIMISERS.items() if name != "geneopt"}


DISTRIBUTIONS = {"geneopt": "GeneOpt", "pygad": "pygad", "deap": "deap", "pymoo": "pymoo",
                 "optuna": "optuna", "nevergrad": "nevergrad"}  # fmt: skip


# --- running & reporting ----------------------------------------------------------


def run_one(problem_name: str, optimiser: str, seed: int, budget_scale: float) -> dict:
    p = PROBLEMS[problem_name]
    n = max(10, int(p.budget * budget_scale))
    f = Budget(p.objective, n, CALL_CAP * n)
    random.seed(seed)
    np.random.seed(seed)
    started = time.perf_counter()
    error = None
    try:
        OPTIMISERS[optimiser](p, f, seed)
    except Exhausted:
        pass
    except Exception as e:  # reported as "crashed", never aborts the table
        error = f"{type(e).__name__}: {e}"[:120]
    return {
        "problem": problem_name,
        "optimiser": optimiser,
        "seed": seed,
        "best": f.best,
        "unique": len(f.memo),
        "seconds": time.perf_counter() - started,
        "error": error,
    }


def run_fair(
    suite: str, function: int, dim: int, instance: int, label: str, out_dir: Path, budget_scale: float = 1.0
) -> dict:
    """One optimiser on one suite problem (spec 015). The instance id is the seed."""
    import suites

    sp = suites.BUILDERS[suite](
        function, dim, instance, Path(out_dir) / suite / "logs", label.replace("/", "_")
    )
    budget = max(10, int(100 * dim * budget_scale))
    f = Budget(sp.objective, budget, CALL_CAP * budget)
    p = Problem(f"{suite}-f{function}", sp.sizes, budget, categorical=suite == "pbo")
    run = optimisers(suite, dim)[label]
    random.seed(instance)
    np.random.seed(instance)
    started = time.perf_counter()
    error = None
    try:
        run(p, f, instance)
    except Exhausted:
        pass
    except Exception as e:  # recorded, never aborts the suite
        error = f"{type(e).__name__}: {e}"[:120]
    seconds = time.perf_counter() - started
    better = max if sp.maximize else min
    trace, best = [], None
    for i, value in enumerate(sp.scored, 1):
        if best is None or better(value, best) != best:
            best = value
            trace.append((i, value))
    config = None
    if isinstance(run, functools.partial):
        config = {"mutation_rate": run.keywords["mutation_rate"], **GENEOPT_OPERATORS}
    return {
        "suite": suite, "function": function, "dim": dim, "instance": instance, "optimiser": label,
        "seed": instance, "budget": budget, "evaluations": len(sp.scored), "seconds": seconds,
        "error": error, "maximize": sp.maximize, "reference": sp.finish(), "best": best,
        "early_best": better(sp.scored[:EARLY]) if sp.scored else None, "trace": trace, "config": config,
        "sizes": list(sp.sizes),
    }  # fmt: skip


def summarise(rows: list[dict], problem: Problem) -> dict:
    ok = [r for r in rows if r["error"] is None]
    if not ok:
        return {"crashed": rows[0]["error"]}
    best = [r["best"] for r in ok]
    q1, q3 = np.percentile(best, [25, 75])
    summary = {
        "median": statistics.median(best),
        "iqr": (q1, q3),
        "unique": statistics.median(r["unique"] for r in ok),
        "seconds": statistics.median(r["seconds"] for r in ok),
        "runs": len(ok),
        "crashed": next((r["error"] for r in rows if r["error"]), None),
    }
    if problem.target is not None:
        summary["hit_rate"] = sum(b <= problem.target for b in best) / len(best)
    return summary


def gate(results: dict[str, dict[str, dict]]) -> list[str]:
    """Spec 003 hard gate: every GeneOpt variant beats random search on every problem."""
    failures = []
    for name, by_opt in results.items():
        r = by_opt.get("random")
        if not r or "median" not in r:
            continue
        for label, g in by_opt.items():
            if not label.startswith("geneopt") or "median" not in g:
                continue
            if PROBLEMS[name].target is not None:
                better = g["hit_rate"] > r["hit_rate"] or g["hit_rate"] == 1.0
                tie_won = g["hit_rate"] == r["hit_rate"] and g["median"] < r["median"]
                if not (better or tie_won):
                    failures.append(
                        f"{name} {label}: hit rate {g['hit_rate']:.0%} (median {g['median']:.4g}) vs random "
                        f"{r['hit_rate']:.0%} (median {r['median']:.4g})"
                    )
            elif not g["median"] < r["median"]:
                failures.append(f"{name} {label}: median {g['median']:.6g} vs random {r['median']:.6g}")
    return failures


def version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "not installed"


def report(results: dict[str, dict[str, dict]], seeds: int, scale: float) -> str:
    lines = [f"Seeds: {seeds} · budget scale: {scale:g} · Python {sys.version.split()[0]} · "
             + ", ".join(f"{d} {version(d)}" for d in DISTRIBUTIONS.values()), ""]  # fmt: skip
    for name, by_opt in results.items():
        p = PROBLEMS[name]
        lines += [
            f"### {name} ({len(p.sizes)}-D, sizes {p.sizes[0]}, budget {int(p.budget * scale)} unique evals)",
            "",
        ]
        metric = "hit rate" if p.target is not None else "median best [IQR] (lower is better)"
        lines += [f"| optimiser | {metric} | unique evals | seconds |", "|---|---|---|---|"]

        def sort_key(item, p=p):
            s = item[1]
            if "median" not in s:
                return (2, 0.0)
            return (0, -s["hit_rate"], s["median"]) if p.target is not None else (0, s["median"])

        for opt, s in sorted(by_opt.items(), key=sort_key):
            if "median" not in s:
                lines.append(f"| {opt} | crashed: {s['crashed']} | | |")
                continue
            value = (
                f"{s['hit_rate']:.0%} (median best {s['median']:.4g})"
                if p.target is not None
                else f"{s['median']:.6g} [{s['iqr'][0]:.4g}, {s['iqr'][1]:.4g}]"
            )
            note = f" ({s['runs']}/{seeds} runs; crashed: {s['crashed']})" if s["crashed"] else ""
            lines.append(f"| {opt} | {value}{note} | {s['unique']:.0f} | {s['seconds']:.2f} |")
        lines.append("")
    failures = gate(results)
    lines.append(
        "**Gate (beats random search everywhere):** "
        + ("PASS" if not failures else "FAIL: " + "; ".join(failures))
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--only", default=",".join(OPTIMISERS), help="comma-separated optimisers")
    ap.add_argument("--problems", default=",".join(PROBLEMS), help="comma-separated problems")
    ap.add_argument("--quick", action="store_true", help="2 seeds, 10%% budget: smoke test")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", help="also write the Markdown report to this file")
    args = ap.parse_args(argv)
    seeds, scale = (2, 0.1) if args.quick else (args.seeds, 1.0)
    optimisers, problems = args.only.split(","), args.problems.split(",")
    unknown = (set(optimisers) - set(OPTIMISERS)) | (set(problems) - set(PROBLEMS))
    if unknown:
        ap.error(f"unknown: {sorted(unknown)}")

    tasks = [(p, o, s, scale) for p in problems for o in optimisers for s in range(seeds)]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(run_one, *zip(*tasks, strict=True)))
    results = {
        p: {
            o: summarise([r for r in rows if r["problem"] == p and r["optimiser"] == o], PROBLEMS[p])
            for o in optimisers
        }
        for p in problems
    }
    text = report(results, seeds, scale)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    return 1 if gate(results) else 0


if __name__ == "__main__":
    sys.exit(main())
