import builtins
from collections import Counter

import numpy as np
import pytest

from geneopt import GeneticAlgorithm
from geneopt.core import _crossover, _mutate, _tournament, diversity


def counting(counter, fn):
    def objective(**params):
        counter[tuple(params.items())] += 1
        return fn(**params)

    return objective


def test_GC_7_identical_genomes_are_evaluated_once():
    calls = Counter()
    ga = GeneticAlgorithm(
        counting(calls, lambda x, y: x + y), {"x": [1.0], "y": [2.0]}, population_size=20, generations=3
    )
    ga.start()
    assert sum(calls.values()) == 1


def test_H_1_history_has_every_individual_of_every_generation():
    ga = GeneticAlgorithm(
        lambda x, y: -(x * x + y * y),
        {"x": np.linspace(-1, 1, 16), "y": np.linspace(-1, 1, 16)},
        population_size=20,
        generations=15,
        seed=1,
    )
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


@pytest.mark.parametrize("seed", range(5))
def test_E_5_elitism_keeps_best_of_population_monotonic(seed):
    ga = GeneticAlgorithm(
        lambda x, y: -abs(x - 37) - abs(y - 3),
        {"x": list(range(100)), "y": list(range(10))},
        population_size=12,
        generations=25,
        elitism=1,
        seed=seed,
    )
    ga.start()
    best = [s.max() for s in ga.history.scores]
    assert best == sorted(best)


@pytest.mark.parametrize("seed", range(5))
def test_finds_bowl_optimum(seed):
    env = {"x": np.linspace(-5, 5, 101), "y": np.linspace(-5, 5, 101)}
    r = GeneticAlgorithm(
        lambda x, y: x * x + y * y,
        env,
        maximize=False,
        population_size=40,
        generations=60,
        mutation_rate="auto",  # 1/D = 0.5: 0.05 gives 0.1 mutations per child at D = 2 and stalls
        seed=seed,
    ).start()
    assert r.score < 1e-9 and r.params == {"x": 0.0, "y": 0.0}


@pytest.mark.parametrize("seed", range(5))
def test_finds_categorical_optimum_within_budget(seed):
    env = {f"p{i}": list("abcde") for i in range(6)}
    r = GeneticAlgorithm(
        lambda **p: sum(v == "c" for v in p.values()),
        env,
        population_size=30,
        generations=60,
        target_score=6,
        seed=seed,
    ).start()
    assert r.stopped_by == "target_score" and r.n_evaluations <= 500


def test_E_6_early_stopping_is_exact():
    r = None
    ga = GeneticAlgorithm(
        lambda x: 1.0, {"x": list(range(10))}, population_size=4, generations=50, early_stopping_rounds=3
    )
    r = ga.start()
    assert r.stopped_by == "early_stopping" and len(ga.history) == 4  # gen 0 + 3 without improvement


def test_target_score_stops_after_scoring_the_whole_generation():
    ga = GeneticAlgorithm(
        lambda x: x, {"x": list(range(10))}, population_size=20, generations=50, target_score=5
    )
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

    def make():
        return GeneticAlgorithm(
            lambda x, y: -((x - 20) ** 2) - (y - 40) ** 2, env, population_size=10, generations=10, seed=7
        )

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
    GeneticAlgorithm(
        lambda x: x,
        {"x": [1.0, 2.0]},
        optimizer_name="op",
        cache_dir=tmp_path,
        generations=2,
        population_size=4,
    ).start()
    assert not [f for f in opened if f.endswith((".csv", ".json"))]


def test_GC_6_categorical_disk_cache_reloads_with_hits(tmp_path):
    env = {"act": ["relu", "tanh", "gelu"], "lr": [1e-3, 1e-2]}

    def score(act, lr):
        return {"relu": 1, "tanh": 2, "gelu": 3}[act] + lr

    GeneticAlgorithm(score, env, cache_dir=tmp_path, generations=5, population_size=6).start()
    r = GeneticAlgorithm(score, env, cache_dir=tmp_path, generations=5, population_size=6).start()
    assert r.n_evaluations == 0 and r.params == {"act": "gelu", "lr": 1e-2}


def test_GC_4_objective_dict_is_not_mutated_after_the_call():
    seen = []
    GeneticAlgorithm(
        lambda **p: seen.append(p) or 0.0, {"x": [1, 2]}, generations=2, population_size=4
    ).start()
    assert all(set(p) == {"x"} for p in seen)  # GC-4: no score/generation keys injected


# --- operators ---------------------------------------------------------------


def test_E_2_mutation_rate_is_a_single_per_gene_probability():
    # 1.x gated twice (per parameter, then per bit), so 0.06 behaved like 0.0036
    rng = np.random.default_rng(0)
    pop = np.zeros((10_000, 5), dtype=int)
    out = _mutate(pop.copy(), 0.2, np.array([10] * 5), rng)
    assert 0.18 <= (out != pop).mean() <= 0.22


def test_random_resetting_picks_a_different_value_in_range():
    rng = np.random.default_rng(0)
    sizes = np.array([2, 3, 1000])
    pop = np.column_stack([rng.integers(0, s, 5_000) for s in sizes])
    out = _mutate(pop.copy(), 1.0, sizes, rng)
    assert (out != pop).all() and (out >= 0).all() and (out < sizes).all()
    counts = np.bincount(out[:, 2], minlength=1000)  # uniform over the other values, no creep
    assert counts.max() < 25


def test_parameters_with_one_value_never_mutate():
    rng = np.random.default_rng(0)
    out = _mutate(np.zeros((100, 1), dtype=int), 1.0, np.array([1]), rng)
    assert (out == 0).all()


def test_E_4_tournament_larger_than_population_is_silent(caplog):
    rng = np.random.default_rng(0)
    winners = _tournament(np.array([1.0, 5.0, 3.0]), k=10, m=50, rng=rng)
    assert winners.shape == (50,) and set(winners) <= {0, 1, 2}
    assert caplog.records == []


def test_E_4_tournament_draws_exactly_k_distinct_candidates():
    rng = np.random.default_rng(0)
    scores = np.arange(100.0)
    winners = _tournament(scores, k=100, m=20, rng=rng)  # k == pop: every candidate seen
    assert (winners == 99).all()


def test_H_2_diversity_is_normalised_gene_disagreement():
    assert diversity(np.zeros((10, 3), dtype=int)) == 0.0
    rng = np.random.default_rng(0)
    d = diversity(rng.integers(0, 8, size=(200, 4)))
    assert 0.4 < d <= 1.0
    assert diversity(np.array([[0, 1]])) == 0.0


# --- ledger regressions ------------------------------------------------------


def total(**params):
    return float(sum(params.values()))


def test_GC_2_dropping_a_parameter_gets_a_fresh_cache(tmp_path):
    # 1.x matched an old {x1, x2, x3} row on the remaining {x1, x2} columns and returned its score
    three = {"x1": [1, 2], "x2": [3, 4], "x3": [5, 6]}
    GeneticAlgorithm(total, three, cache_dir=tmp_path, generations=2, population_size=4).start()
    r = GeneticAlgorithm(
        total, {"x1": [1, 2], "x2": [3, 4]}, cache_dir=tmp_path, generations=2, population_size=4
    ).start()
    assert r.n_evaluations > 0 and r.score <= 6
    assert len(list(tmp_path.glob("*.jsonl"))) == 2


def test_GC_5_parameters_named_score_and_generation(tmp_path):
    # 1.x overwrote these parameters with its own bookkeeping columns, so they never hit the cache
    def objective(score, generation):
        return score - generation

    env = {"score": [1, 2, 3], "generation": [0, 1]}
    GeneticAlgorithm(objective, env, cache_dir=tmp_path, generations=4, population_size=6).start()
    r = GeneticAlgorithm(objective, env, cache_dir=tmp_path, generations=4, population_size=6).start()
    assert r.n_evaluations == 0
    assert r.params == {"score": 3, "generation": 0} and r.score == 3


def test_GC_8_integer_numpy_environment_persists(tmp_path):
    # 1.x crashed in on_end: "Object of type int64 is not JSON serializable"
    env = {"n": np.arange(8)}
    GeneticAlgorithm(lambda n: n * 1, env, cache_dir=tmp_path, generations=3, population_size=4).start()
    r = GeneticAlgorithm(lambda n: n * 1, env, cache_dir=tmp_path, generations=3, population_size=4).start()
    assert r.n_evaluations == 0 and type(r.params["n"]) is int


def test_GC_11_only_cache_dir_is_created(tmp_path, monkeypatch):
    # 1.x always created ./cache, wherever the cache was really kept
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    store = tmp_path / "store" / "deep"
    GeneticAlgorithm(total, {"x": [1, 2]}, cache_dir=store, generations=2, population_size=4).start()
    assert list(cwd.iterdir()) == []
    assert len(list(store.glob("*.jsonl"))) == 1


@pytest.mark.parametrize("n_params", [2, 50])
@pytest.mark.parametrize("configured", [None, 0.1, "auto"])
def test_E_3_mutation_rate_is_fixed_and_defaults_to_0_05(monkeypatch, n_params, configured):
    # 1.x divided the rate by the raw Hamming distance (rates above 1 near convergence).
    # 2.0: one constant per-gene rate, 0.05 unless set; "auto" is 1 / n_params (spec 014 D).
    import geneopt.core

    expected = {None: 0.05, "auto": 1 / n_params}.get(configured, configured)
    seen, changed, total_genes = set(), 0, 0

    def spy(genomes, rate, sizes, rng):
        nonlocal changed, total_genes
        before = genomes.copy()
        out = mutate(genomes, rate, sizes, rng)
        seen.add(rate)
        changed += (out != before).sum()
        total_genes += before.size
        return out

    mutate = geneopt.core._mutate
    monkeypatch.setattr(geneopt.core, "_mutate", spy)
    env = {f"p{i}": list(range(10)) for i in range(n_params)}
    kwargs = {} if configured is None else {"mutation_rate": configured}
    GeneticAlgorithm(total, env, population_size=200, generations=40, **kwargs).start()
    assert seen == {expected}
    assert changed / total_genes == pytest.approx(expected, rel=0.1)


@pytest.mark.parametrize("bad", [None, "Auto", "1/n", True])
def test_mutation_rate_other_than_a_number_or_auto_is_rejected(bad):
    with pytest.raises(ValueError, match="mutation_rate"):
        GeneticAlgorithm(total, {"x": [1, 2]}, mutation_rate=bad)


# --- crossover: uniform, Syswerda 1989 (spec 014 C) ---------------------------


def test_uniform_crossover_takes_each_gene_from_either_parent_with_p_one_half():
    rng = np.random.default_rng(0)
    p1, p2 = np.zeros((5_000, 10), dtype=int), np.ones((5_000, 10), dtype=int)
    c1, c2 = _crossover(p1, p2, crossover_probability=1.0, rng=rng)
    assert abs(c1.mean() - 0.5) < 0.02  # c1 holds a 1 exactly where it took the gene from p2
    assert (c2 == 1 - c1).all()  # child 2 takes the gene child 1 didn't


@pytest.mark.parametrize("p_c", [0.3, 0.9])
def test_share_of_recombining_pairs_is_crossover_probability(p_c):
    rng = np.random.default_rng(0)
    p1, p2 = np.zeros((5_000, 64), dtype=int), np.ones((5_000, 64), dtype=int)
    c1, _ = _crossover(p1, p2, crossover_probability=p_c, rng=rng)
    recombined = c1.any(axis=1)  # a recombining 64-gene pair is all-zero with p = 2^-64
    assert abs(recombined.mean() - p_c) < 0.03


def test_crossover_probability_zero_copies_the_parents():
    rng = np.random.default_rng(0)
    p1, p2 = rng.integers(0, 9, (100, 4)), rng.integers(0, 9, (100, 4))
    c1, c2 = _crossover(p1, p2, crossover_probability=0.0, rng=rng)
    assert (c1 == p1).all() and (c2 == p2).all()


def test_crossover_children_hold_exactly_the_parents_genes():
    rng = np.random.default_rng(1)
    p1, p2 = rng.integers(0, 50, (200, 7)), rng.integers(0, 50, (200, 7))
    c1, c2 = _crossover(p1, p2, crossover_probability=0.9, rng=rng)
    assert ((c1 == p1) | (c1 == p2)).all() and ((c1 + c2) == (p1 + p2)).all()


def test_crossover_probability_is_validated():
    with pytest.raises(ValueError, match="crossover_probability"):
        GeneticAlgorithm(total, {"x": [1, 2]}, crossover_probability=1.5)


@pytest.mark.parametrize(
    "kwarg", ["adaptive_mutation", "crossover_swap_rate", "initial_population_size", "crossover_rate"]
)
def test_non_standard_operator_options_are_gone(kwarg):
    with pytest.raises(TypeError, match=kwarg):
        GeneticAlgorithm(total, {"x": [1, 2]}, **{kwarg: 0.5})
