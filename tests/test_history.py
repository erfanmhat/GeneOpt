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
