import builtins
from collections import Counter

import numpy as np
import pytest

from geneopt import GeneticAlgorithm
from geneopt.core import _mutate, _tournament, diversity


def counting(counter, fn):
    def objective(**params):
        counter[tuple(params.items())] += 1
        return fn(**params)
    return objective


def test_GC_7_identical_genomes_are_evaluated_once():
    calls = Counter()
    ga = GeneticAlgorithm(counting(calls, lambda x, y: x + y), {"x": [1.0], "y": [2.0]},
                          population_size=20, generations=3)
    ga.start()
    assert sum(calls.values()) == 1


def test_H_1_history_has_every_individual_of_every_generation():
    ga = GeneticAlgorithm(lambda x, y: -(x * x + y * y), {"x": np.linspace(-1, 1, 16), "y": np.linspace(-1, 1, 16)},
                          population_size=20, generations=15, seed=1)
    ga.start()
    assert [len(s) for s in ga.history.scores] == [20] * 15
    assert ga.history.genomes[3].shape == (20, 2)


def test_E_1_initial_population_is_uniform():
    ga = GeneticAlgorithm(lambda v: 0.0, {"v": list("abcde")}, population_size=8000, generations=1, seed=0)
    ga.start()
    counts = np.bincount(ga.history.genomes[0][:, 0], minlength=5)
    assert all(1450 <= c <= 1750 for c in counts), counts


def test_odd_population_size_is_kept():
    ga = GeneticAlgorithm(lambda x: x, {"x": list(range(50))}, population_size=101, generations=3)
    ga.start()
    assert [len(s) for s in ga.history.scores] == [101, 101, 101]


def test_initial_population_size_applies_to_generation_zero_only():
    ga = GeneticAlgorithm(lambda x: x, {"x": list(range(50))}, population_size=10,
                          initial_population_size=30, generations=3)
    ga.start()
    assert [len(s) for s in ga.history.scores] == [30, 10, 10]


@pytest.mark.parametrize("seed", range(5))
def test_elitism_keeps_best_of_population_monotonic(seed):
    ga = GeneticAlgorithm(lambda x, y: -abs(x - 37) - abs(y - 3), {"x": list(range(100)), "y": list(range(10))},
                          population_size=12, generations=25, elitism=1, seed=seed)
    ga.start()
    best = [s.max() for s in ga.history.scores]
    assert best == sorted(best)


@pytest.mark.parametrize("seed", range(5))
def test_finds_bowl_optimum(seed):
    env = {"x": np.linspace(-5, 5, 101), "y": np.linspace(-5, 5, 101)}
    r = GeneticAlgorithm(lambda x, y: x * x + y * y, env, maximize=False,
                         population_size=40, generations=60, seed=seed).start()
    assert r.score < 1e-9 and r.params == {"x": 0.0, "y": 0.0}


@pytest.mark.parametrize("seed", range(5))
def test_finds_categorical_optimum_within_budget(seed):
    env = {f"p{i}": list("abcde") for i in range(6)}
    r = GeneticAlgorithm(lambda **p: sum(v == "c" for v in p.values()), env,
                         population_size=30, generations=60, target_score=6, seed=seed).start()
    assert r.stopped_by == "target_score" and r.n_evaluations <= 500


def test_E_6_early_stopping_is_exact():
    r = None
    ga = GeneticAlgorithm(lambda x: 1.0, {"x": list(range(10))}, population_size=4, generations=50,
                          early_stopping_rounds=3)
    r = ga.start()
    assert r.stopped_by == "early_stopping" and len(ga.history) == 4  # gen 0 + 3 without improvement


def test_target_score_stops_after_scoring_the_whole_generation():
    ga = GeneticAlgorithm(lambda x: x, {"x": list(range(10))}, population_size=20, generations=50,
                          target_score=5)
    r = ga.start()
    assert r.stopped_by == "target_score" and r.score >= 5
    assert len(ga.history.scores[-1]) == 20


def test_all_worst_scores_still_give_a_result():
    r = GeneticAlgorithm(lambda x: float("-inf"), {"x": [1, 2, 3]}, population_size=4, generations=3).start()
    assert r.score == float("-inf") and r.params["x"] in (1, 2, 3)


def test_result_counters():
    ga = GeneticAlgorithm(lambda x: 0.0, {"x": [1]}, population_size=4, generations=3)
    r = ga.start()
    assert r.n_evaluations == 1 and r.n_cache_hits == 8  # gens 1 and 2 are fully cached
    assert ga.result is r and r.stopped_by == "generations"


def test_same_seed_same_result_and_start_is_repeatable():
    env = {"x": list(range(64)), "y": list(range(64))}
    make = lambda: GeneticAlgorithm(lambda x, y: -(x - 20) ** 2 - (y - 40) ** 2, env, population_size=10,
                                    generations=10, seed=7)
    ga = make()
    a, b = ga.start(), ga.start()
    c = make().start()
    assert (a.params, a.score) == (b.params, b.score) == (c.params, c.score)


def test_warm_disk_cache_makes_zero_calls_and_same_result(tmp_path):
    env = {"x": list(range(32)), "y": list(range(32))}
    calls = Counter()
    obj = counting(calls, lambda x, y: -abs(x - 9) - abs(y - 21))
    first = GeneticAlgorithm(obj, env, cache_dir=tmp_path, generations=8, population_size=10).start()
    calls.clear()
    second = GeneticAlgorithm(obj, env, cache_dir=tmp_path, generations=8, population_size=10).start()
    assert sum(calls.values()) == 0
    assert (first.params, first.score) == (second.params, second.score)
    assert second.n_evaluations == 0


def test_GC_3_changed_objective_gets_fresh_cache(tmp_path):
    env = {"x": list(np.linspace(0, 1, 8))}
    GeneticAlgorithm(lambda x: x, env, cache_dir=tmp_path, generations=3, population_size=8).start()
    r = GeneticAlgorithm(lambda x: -x, env, cache_dir=tmp_path, generations=3, population_size=8).start()
    assert r.params["x"] == 0.0 and r.score == 0.0


def test_1x_cache_files_are_never_opened(tmp_path, monkeypatch):
    (tmp_path / "op.csv").write_text("x,score,generation\n1.0,99.0,0\n")
    (tmp_path / "op.json").write_text("{}")
    opened = []
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: opened.append(str(f)) or real_open(f, *a, **k))
    GeneticAlgorithm(lambda x: x, {"x": [1.0, 2.0]}, optimizer_name="op", cache_dir=tmp_path,
                     generations=2, population_size=4).start()
    assert not [f for f in opened if f.endswith((".csv", ".json"))]


def test_categorical_disk_cache_reloads_with_hits(tmp_path):
    env = {"act": ["relu", "tanh", "gelu"], "lr": [1e-3, 1e-2]}
    score = lambda act, lr: {"relu": 1, "tanh": 2, "gelu": 3}[act] + lr
    GeneticAlgorithm(score, env, cache_dir=tmp_path, generations=5, population_size=6).start()
    r = GeneticAlgorithm(score, env, cache_dir=tmp_path, generations=5, population_size=6).start()
    assert r.n_evaluations == 0 and r.params == {"act": "gelu", "lr": 1e-2}


def test_objective_receives_a_fresh_dict_each_call():
    seen = []
    GeneticAlgorithm(lambda **p: seen.append(p) or 0.0, {"x": [1, 2]}, generations=2, population_size=4).start()
    assert all(set(p) == {"x"} for p in seen)  # GC-4: no score/generation keys injected


# --- operators ---------------------------------------------------------------

def test_mutation_rate_is_per_gene_probability():
    rng = np.random.default_rng(0)
    pop = np.zeros((10_000, 5), dtype=int)
    out = _mutate(pop.copy(), 0.2, np.array([10] * 5), np.array([False] * 5), rng)
    assert 0.18 <= (out != pop).mean() <= 0.22


def test_mutation_creeps_locally_for_ordered_numeric_params():
    rng = np.random.default_rng(0)
    pop = np.full((10_000, 1), 500)
    out = _mutate(pop.copy(), 1.0, np.array([1000]), np.array([True]), rng)
    step = np.abs(out - pop)
    assert (step > 0).all() and np.median(step) < 150


def test_tournament_larger_than_population_is_fine(caplog):
    rng = np.random.default_rng(0)
    winners = _tournament(np.array([1.0, 5.0, 3.0]), k=10, m=50, rng=rng)
    assert winners.shape == (50,) and set(winners) <= {0, 1, 2}
    assert caplog.records == []


def test_tournament_draws_exactly_k_distinct_candidates():
    rng = np.random.default_rng(0)
    scores = np.arange(100.0)
    winners = _tournament(scores, k=100, m=20, rng=rng)  # k == pop: every candidate seen
    assert (winners == 99).all()


def test_diversity_bounds():
    assert diversity(np.zeros((10, 3), dtype=int)) == 0.0
    rng = np.random.default_rng(0)
    d = diversity(rng.integers(0, 8, size=(200, 4)))
    assert 0.4 < d <= 1.0
    assert diversity(np.array([[0, 1]])) == 0.0
