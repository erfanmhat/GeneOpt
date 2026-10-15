import dataclasses
import json
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pytest

from geneopt import GeneticAlgorithm

ENV = {"x": list(range(40)), "y": list(range(40))}


def peak(x, y):
    return -abs(x - 13) - abs(y - 29)


def run(**kwargs):
    ga = GeneticAlgorithm(peak, ENV, population_size=16, generations=12, seed=7, **kwargs)
    return ga, ga.start()


def same_run(a, b):
    (ga_a, r_a), (ga_b, r_b) = a, b
    assert dataclasses.replace(r_a, elapsed=0) == dataclasses.replace(r_b, elapsed=0)
    for name in ("genomes", "scores", "hits"):
        assert all(np.array_equal(x, y) for x, y in zip(getattr(ga_a.history, name), getattr(ga_b.history, name)))


def test_thread_pool_gives_identical_run():
    with ThreadPoolExecutor(4) as pool:
        same_run(run(), run(executor=pool))


def test_process_pool_gives_identical_run():
    with ProcessPoolExecutor(2) as pool:
        same_run(run(), run(executor=pool))


def flaky(x, y):
    if x % 10 == 0:
        raise RuntimeError(f"bad x={x}")
    return float(x + y)


def test_on_error_worst_scores_failures_as_worst_and_never_caches_them(tmp_path, caplog):
    ga = GeneticAlgorithm(flaky, ENV, population_size=16, generations=6, on_error="worst", cache_dir=tmp_path)
    with caplog.at_level(logging.WARNING, logger="geneopt"):
        result = ga.start()
    failures = [r for r in caplog.records if "bad x=" in r.getMessage()]
    assert failures
    assert result.params["x"] % 10 != 0
    records = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()[1:]]
    assert records and all(ENV["x"][r["k"][0]] % 10 != 0 for r in records)
    worst_rows = [s for gen in ga.history.scores for s in gen if s == -np.inf]
    assert worst_rows


def test_on_error_worst_minimize_uses_plus_inf():
    ga = GeneticAlgorithm(flaky, ENV, population_size=16, generations=3, maximize=False, on_error="worst")
    ga.start()
    assert np.inf in np.concatenate(ga.history.scores)


def make_fail_after(n):
    calls = []

    def objective(x, y):
        calls.append(1)
        if len(calls) > n:
            raise RuntimeError("boom")
        return float(x)
    return objective


def test_on_error_raise_keeps_finished_work_on_disk(tmp_path):
    ga = GeneticAlgorithm(make_fail_after(5), ENV, population_size=16, generations=3, cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        ga.start()
    lines = next(tmp_path.glob("*.jsonl")).read_text().splitlines()
    assert len(lines) == 1 + 5  # header + the five successful evaluations


def test_on_error_raise_with_executor_saves_the_rest_of_the_batch(tmp_path):
    import threading
    lock, calls = threading.Lock(), []

    def objective(x, y):
        with lock:
            calls.append((x, y))
            first = len(calls) == 1
        if first:
            raise RuntimeError("boom")
        return float(x)

    with ThreadPoolExecutor(4) as pool:
        ga = GeneticAlgorithm(objective, ENV, population_size=16, generations=3, cache_dir=tmp_path,
                              executor=pool)
        with pytest.raises(RuntimeError, match="boom"):
            ga.start()
    lines = next(tmp_path.glob("*.jsonl")).read_text().splitlines()
    assert len(lines) - 1 == len(calls) - 1  # every call of the batch except the failing one


def test_on_error_retry_recovers_from_transient_failures(monkeypatch):
    monkeypatch.setattr("geneopt.core.time.sleep", lambda s: None)
    attempts = {}

    def objective(x, y):
        attempts[(x, y)] = attempts.get((x, y), 0) + 1
        if attempts[(x, y)] <= 2:
            raise ConnectionError("transient")
        return float(x)

    ga = GeneticAlgorithm(objective, ENV, population_size=8, generations=2, on_error="retry", max_retries=2)
    ga.start()
    assert set(attempts.values()) == {3}


def test_on_error_retry_gives_up_after_max_retries(monkeypatch):
    sleeps = []
    monkeypatch.setattr("geneopt.core.time.sleep", sleeps.append)

    def objective(x, y):
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        GeneticAlgorithm(objective, ENV, population_size=4, generations=1, on_error="retry", max_retries=2).start()
    assert sleeps[:2] == [0.5, 1.0]


@pytest.mark.parametrize("kwargs, name", [
    ({"on_error": "ignore"}, "on_error"),
    ({"max_retries": -1}, "max_retries"),
    ({"executor": object()}, "executor"),
])
def test_invalid_engine_options(kwargs, name):
    with pytest.raises((ValueError, TypeError), match=name):
        GeneticAlgorithm(peak, ENV, **kwargs)


def test_silent_at_warning_level(caplog):
    with caplog.at_level(logging.WARNING, logger="geneopt"):
        run()
    assert caplog.records == []


def test_info_logs_one_line_per_generation_plus_summary(caplog):
    with caplog.at_level(logging.INFO, logger="geneopt"):
        ga, result = run()
    info = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(info) == len(ga.history) + 1
    assert "evaluations" in info[-1].getMessage()


def test_library_adds_no_log_handlers():
    run()
    assert logging.getLogger("geneopt").handlers == []


def test_huge_search_space_summary_does_not_overflow(caplog):
    env = {f"p{i}": list(range(10)) for i in range(400)}
    with caplog.at_level(logging.INFO, logger="geneopt"):
        GeneticAlgorithm(lambda **p: 0.0, env, population_size=2, generations=1).start()
    assert "1" + "0" * 400 in caplog.records[-1].getMessage()
