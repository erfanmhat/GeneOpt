import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "bench"))
import run as bench  # noqa: E402


def test_budget_counts_unique_points_and_memoises():
    calls = []
    f = bench.Budget(lambda k: calls.append(k) or float(sum(k)), n_unique=3, max_calls=100)
    assert f((1, 2)) == 3.0 and f([1, 2]) == 3.0  # repeat is free and served from the memo
    f((0, 0))
    assert len(calls) == 2 and f.best == 0.0 and len(f.memo) == 2


def test_budget_stops_after_n_unique():
    f = bench.Budget(lambda k: 1.0, n_unique=2, max_calls=100)
    f((0,))
    f((1,))
    with pytest.raises(bench.Exhausted):
        f((2,))
    assert f((0,)) == 1.0  # known points stay available until the call cap


def test_budget_call_cap_stops_duplicate_only_optimisers():
    f = bench.Budget(lambda k: 1.0, n_unique=10, max_calls=5)
    for _ in range(5):
        f((0,))
    with pytest.raises(bench.Exhausted):
        f((0,))


def test_exhausted_is_not_swallowed_by_except_exception():
    assert not issubclass(bench.Exhausted, Exception)


def test_problems_have_their_known_optima():
    assert bench.categorical(bench.CATEGORICAL_TARGET) == -6.0
    assert bench.PROBLEMS["categorical"].target == -6.0
    grid = bench.HOLDER_GRID
    near = [int(abs(grid - v).argmin()) + d for v in (8.05502, 9.66459) for d in range(-2, 3)]
    best_holder = min(bench.holder((i, j)) for i in near[:5] for j in near[5:])
    assert best_holder == pytest.approx(-19.2085, abs=2e-3)  # grid optimum next to the analytic one
    assert bench.rastrigin(bench.RASTRIGIN_OPTIMUM) == pytest.approx(0.0, abs=1e-9)
    assert bench.rastrigin((127,) * 10) > 50  # the centre of the box is far from the optimum


def test_gate_requires_beating_random_search():
    good = {"holder": {"geneopt": {"median": -19.0}, "random": {"median": -15.0}}}
    tie = {"holder": {"geneopt": {"median": -15.0}, "random": {"median": -15.0}}}
    assert bench.gate(good) == [] and bench.gate(tie) != []
    cat = {
        "categorical": {"geneopt": {"median": -6, "hit_rate": 0.9}, "random": {"median": -4, "hit_rate": 0.1}}
    }
    assert bench.gate(cat) == []
    tie_better_median = {
        "categorical": {"geneopt": {"median": -5, "hit_rate": 0.0}, "random": {"median": -4, "hit_rate": 0.0}}
    }
    tie_same_median = {
        "categorical": {"geneopt": {"median": -4, "hit_rate": 0.0}, "random": {"median": -4, "hit_rate": 0.0}}
    }
    assert bench.gate(tie_better_median) == [] and bench.gate(tie_same_median) != []


@pytest.mark.bench
def test_quick_benchmark_runs_every_optimiser(capsys):
    code = bench.main(["--quick", "--jobs", "2"])
    out = capsys.readouterr().out
    for name in bench.OPTIMISERS:
        assert f"| {name} |" in out
    assert "crashed" not in out
    assert "**Gate" in out and code in (0, 1)  # smoke test only: the gate is judged on full runs


# --- spec 015: GeneOpt configuration -------------------------------------------

OPERATOR_PARAMS = {"population_size", "crossover_probability", "mutation_rate", "tournament_size", "elitism"}


def test_geneopt_mutation_rates_follow_the_1_over_d_rule_and_spec_015():
    spec = (Path(__file__).resolve().parents[1] / "specs" / "015-fair-benchmark-suite.md").read_text()
    names = {"bbob-mixint": "mixint", "PBO": "pbo"}
    rows = {}
    for line in spec.splitlines():
        cells = [c.strip(" *") for c in line.strip("|").split("|")]
        if len(cells) == 7 and cells[0] in names and cells[1].isdigit():
            rows[(names[cells[0]], int(cells[1]))] = {
                "1/D": float(cells[4]), "0.5/D": float(cells[5]), "2/D": float(cells[6])
            }  # fmt: skip
    assert rows == bench.GENEOPT_MUTATION
    for (_, dim), rates in bench.GENEOPT_MUTATION.items():
        assert rates == {"1/D": 1 / dim, "0.5/D": 0.5 / dim, "2/D": 2 / dim}


def test_geneopt_adapter_passes_every_operator_parameter(monkeypatch):
    import inspect

    import geneopt

    assert OPERATOR_PARAMS <= set(inspect.signature(geneopt.GeneticAlgorithm).parameters)
    seen = {}

    class Recorder:
        def __init__(self, objective, env, **kwargs):
            seen.update(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(geneopt, "GeneticAlgorithm", Recorder)
    p = bench.Problem("x", (2,) * 4, 10, categorical=True)
    bench.run_geneopt(p, bench.Budget(lambda k: 0.0, 10, 100), seed=1, mutation_rate=0.25)
    assert OPERATOR_PARAMS <= set(seen) and seen["mutation_rate"] == 0.25


def test_fair_suite_optimisers_have_three_geneopt_rows():
    labels = list(bench.optimisers("pbo", 64))
    assert [x for x in labels if x.startswith("geneopt")] == [
        "geneopt[mr=1/D]",
        "geneopt[mr=0.5/D]",
        "geneopt[mr=2/D]",
    ]
    assert set(labels) - {x for x in labels if x.startswith("geneopt")} == set(bench.OPTIMISERS) - {"geneopt"}
