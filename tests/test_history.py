import numpy as np
import pytest

from geneopt.history import History


@pytest.mark.plots  # to_frame needs pandas
def test_to_frame_has_one_row_per_individual_per_generation():
    history = History({"act": ["relu", "tanh"], "lr": [0.1, 0.01, 0.001]})
    history.append(np.array([[0, 2], [1, 0]]), np.array([1.0, 2.0]), np.array([False, True]))
    history.append(np.array([[1, 1], [1, 1]]), np.array([3.0, 3.0]), np.array([False, True]))

    frame = history.to_frame()

    assert len(history) == 2
    assert list(frame.columns) == ["generation", "individual", "act", "lr", "score", "cache_hit"]
    assert frame["generation"].tolist() == [0, 0, 1, 1]
    assert frame.iloc[0][["act", "lr"]].tolist() == ["relu", 0.001]
    assert frame["cache_hit"].tolist() == [False, True, False, True]


def test_append_copies_so_later_mutation_does_not_leak():
    history = History({"x": [0, 1]})
    genomes = np.array([[0], [1]])
    history.append(genomes, np.array([0.0, 1.0]), np.array([False, False]))
    genomes[0, 0] = 1
    assert history.genomes[0][0, 0] == 0


# --- spec 002 §1: int32 storage, save / load ------------------------------------


def test_genomes_are_stored_as_int32():
    history = History({"x": [1, 2, 3]})
    history.append(np.array([[2], [0]], dtype=np.int64), [1.0, 2.0], [False, True])
    assert history.genomes[0].dtype == np.int32 and history.genomes[0].tolist() == [[2], [0]]


def test_gene_frequencies_do_not_overflow_int32_for_huge_parameters():
    from geneopt.plots import gene_frequency_matrix

    history = History({"x": range(2**26)})  # index * 64 bins exceeds 2**31
    history.append([[2**26 - 1], [0]], [1.0, 2.0], [False, False])
    assert gene_frequency_matrix(history, "x")[0].tolist() == [0.5] + [0.0] * 62 + [0.5]


def test_save_and_load_round_trip(tmp_path):
    env = {"lr": [np.float64(0.1), 0.01], "kind": ["a", "b", "c"], "n": np.arange(4)}
    history = History(env, maximize=False)
    history.append([[0, 2, 3], [1, 0, 0]], [0.5, np.inf], [False, True])
    history.append([[1, 1, 1], [0, 2, 3]], [0.25, 0.5], [False, True])
    path = history.save(tmp_path / "run")
    assert path == tmp_path / "run.npz" and path.exists()
    loaded = History.load(path)
    assert loaded.environment == {"lr": [0.1, 0.01], "kind": ["a", "b", "c"], "n": [0, 1, 2, 3]}
    assert loaded.maximize is False and len(loaded) == 2
    for a, b in zip(
        loaded.genomes + loaded.scores + loaded.hits,
        history.genomes + history.scores + history.hits,
        strict=True,
    ):
        assert a.dtype == b.dtype and np.array_equal(a, b)


def test_save_an_empty_history(tmp_path):
    loaded = History.load(History({"x": [1, 2]}).save(tmp_path / "empty.npz"))
    assert len(loaded) == 0 and loaded.environment == {"x": [1, 2]}


def test_save_rejects_values_json_cannot_hold(tmp_path):
    history = History({"model": [object(), object()]})
    with pytest.raises(TypeError, match="model"):
        history.save(tmp_path / "x.npz")


def test_a_finished_run_saves_and_loads(tmp_path):
    from geneopt import GeneticAlgorithm

    ga = GeneticAlgorithm(
        lambda x, y: x + len(y), {"x": [1, 2, 3], "y": ["a", "bb"]}, generations=4, population_size=6
    )
    ga.start()
    loaded = History.load(ga.history.save(tmp_path / "h.npz"))
    assert len(loaded) == 4 and loaded.environment == ga.history.environment
    assert all(np.array_equal(a, b) for a, b in zip(loaded.scores, ga.history.scores, strict=True))
