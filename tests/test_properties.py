"""Property tests (spec 007). Deterministic: derandomized examples, seeded GA runs."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")  # in the dev extra; skipped only outside it
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from geneopt import GeneticAlgorithm  # noqa: E402
from geneopt.cache import GenomeCache  # noqa: E402

FAST = settings(max_examples=25, deadline=None, derandomize=True)


@FAST
@given(seed=st.integers(0, 10_000), maximize=st.booleans())
def test_best_is_monotonic_and_result_is_the_overall_best(seed, maximize):
    env = {"x": list(range(30)), "y": list(range(30))}
    ga = GeneticAlgorithm(
        lambda x, y: (x - 11) ** 2 - (y - 17) ** 2,
        env,
        maximize=maximize,
        population_size=8,
        generations=12,
        elitism=1,
        seed=seed,
    )
    result = ga.start()
    sign = 1 if maximize else -1
    per_generation = [sign * s.max() if maximize else sign * s.min() for s in ga.history.scores]
    assert per_generation == sorted(per_generation)  # elitism: the population never gets worse
    overall = max(per_generation)
    assert sign * result.score == overall
    assert per_generation[result.generation] == overall  # found in the generation it reports


json_values = st.one_of(
    st.text(),  # includes newlines and quotes, which must not break the JSONL framing
    st.integers(-(2**70), 2**70),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
)


@FAST
@given(values=st.lists(json_values, min_size=1, max_size=6), score=st.floats(allow_nan=False))
def test_cache_round_trips_arbitrary_json_values(values, score):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.jsonl"
        key = tuple(range(len(values)))
        GenomeCache(path, "fp").put(key, score, values=values)
        assert GenomeCache(path, "fp").get(key) == score
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["v"] == values


@FAST
@given(values=st.lists(st.text(min_size=1), min_size=2, max_size=8, unique=True))
def test_text_environment_rerun_is_fully_cached(values):
    with tempfile.TemporaryDirectory() as tmp:
        env = {"v": values}

        def length(v):
            return len(v)

        first = GeneticAlgorithm(length, env, cache_dir=tmp, generations=3, population_size=6).start()
        again = GeneticAlgorithm(length, env, cache_dir=tmp, generations=3, population_size=6).start()
        assert again.n_evaluations == 0 and again.params == first.params


def chi2_critical(dof: int, z: float = 4.75) -> float:
    """Upper chi-square quantile via Wilson-Hilferty; z=4.75 is p ~ 1e-6, so a fair sampler
    essentially never fails, while E-1-style bias (one value at ~50%) fails by orders of magnitude."""
    return dof * (1 - 2 / (9 * dof) + z * math.sqrt(2 / (9 * dof))) ** 3


@FAST
@given(size=st.integers(2, 300))
def test_initial_sampling_is_uniform_for_any_size(size):
    draws = 20_000
    ga = GeneticAlgorithm(
        lambda v: 0.0, {"v": list(range(size))}, population_size=draws, generations=1, seed=size
    )
    ga.start()
    counts = np.bincount(ga.history.genomes[0][:, 0], minlength=size)
    expected = draws / size
    chi2 = ((counts - expected) ** 2 / expected).sum()
    assert chi2 < chi2_critical(size - 1)
