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
