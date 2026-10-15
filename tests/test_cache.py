import json
import logging
import time

import numpy as np
import pytest

from geneopt.cache import CacheMismatchError, GenomeCache, fingerprint


def test_GC_9_memory_cache_is_exact_dict_lookup():
    cache = GenomeCache()
    cache.put((1, 2), 3.5)
    assert cache.get((1, 2)) == 3.5
    assert cache.get((1, 3)) is None
    assert (1, 2) in cache and len(cache) == 1


def test_GC_1_close_values_are_distinct_keys(tmp_path):
    # keys are indices, so 1_000_000 and 1_000_005 (indices 0 and 1) can never collide
    path = tmp_path / "c.jsonl"
    cache = GenomeCache(path, "fp")
    cache.put((0,), 1.0, values=[1_000_000.0])
    assert cache.get((1,)) is None


def test_values_round_trip_through_disk_including_numpy_and_strings(tmp_path):
    # GC-5 (param names irrelevant to keys), GC-6 (strings), GC-8 (np.int64)
    path = tmp_path / "c.jsonl"
    cache = GenomeCache(path, "fp", params=["score", "generation", "act"])
    cache.put((np.int64(1), 0, 2), np.float64(-4.25), values=[np.int64(7), np.float64(0.5), "relu"])
    reloaded = GenomeCache(path, "fp", params=["score", "generation", "act"])
    assert reloaded.get((1, 0, 2)) == -4.25
    assert json.loads(path.read_text().splitlines()[1])["v"] == [7, 0.5, "relu"]


def test_inf_allowed_nan_rejected():
    cache = GenomeCache()
    cache.put((0,), float("inf"))
    with pytest.raises(ValueError, match="NaN"):
        cache.put((1,), float("nan"))


def test_fingerprint_mismatch_raises(tmp_path):
    path = tmp_path / "c.jsonl"
    GenomeCache(path, "aaa").put((0,), 1.0)
    with pytest.raises(CacheMismatchError):
        GenomeCache(path, "bbb")


def test_GC_10_truncated_tail_is_skipped_once_and_file_repaired(tmp_path, caplog):
    path = tmp_path / "c.jsonl"
    cache = GenomeCache(path, "fp")
    cache.put((0,), 1.0)
    cache.put((1,), 2.0)
    path.write_bytes(path.read_bytes()[:-5])  # crash mid-write of the last record
    with caplog.at_level(logging.WARNING, logger="geneopt"):
        reloaded = GenomeCache(path, "fp")
    assert reloaded.get((0,)) == 1.0 and reloaded.get((1,)) is None
    assert sum("truncated" in r.message for r in caplog.records) == 1
    reloaded.put((2,), 3.0)  # appending after repair must not corrupt the file
    assert GenomeCache(path, "fp").get((2,)) == 3.0


def test_corrupt_middle_line_raises(tmp_path):
    path = tmp_path / "c.jsonl"
    cache = GenomeCache(path, "fp")
    cache.put((0,), 1.0)
    cache.put((1,), 2.0)
    lines = path.read_text().splitlines()
    lines[1] = "{broken"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        GenomeCache(path, "fp")


def test_GC_9_hundred_thousand_puts_and_gets_under_a_second():
    cache = GenomeCache()
    start = time.perf_counter()
    for i in range(100_000):
        cache.put((i, i), float(i))
    for i in range(100_000):
        cache.get((i, i))
    assert time.perf_counter() - start < 1.0


ENV = {"x": [0.0, 0.5, 1.0]}


def obj_a(x):
    return x


def obj_b(x):
    return -x


def test_GC_3_fingerprint_changes_with_env_objective_and_version():
    base = fingerprint(ENV, obj_a)
    assert base == fingerprint({"x": [0.0, 0.5, 1.0]}, obj_a)  # stable
    assert base != fingerprint({"x": [0.0, 0.5, 2.0]}, obj_a)  # values changed
    assert base != fingerprint({"x": [0.0, 0.5, 1.0], "y": [1]}, obj_a)  # GC-2: param added
    assert base != fingerprint(ENV, obj_b)  # objective body changed
    assert base != fingerprint(ENV, obj_a, objective_version="2")


def test_fingerprint_sees_called_function_names():
    def f(x):
        return np.sin(x)

    def g(x):
        return np.cos(x)

    assert fingerprint(ENV, f) != fingerprint(ENV, g)


def test_fingerprint_is_stable_across_processes_for_nested_code():
    # repr() of a nested code object contains a memory address; the digest must not
    import subprocess
    import sys

    code = (
        "from geneopt.cache import fingerprint\n"
        "def f(x):\n    return (lambda y: y + 1)(x)\n"
        "print(fingerprint({'x': [1, 2]}, f))"
    )
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
        for _ in range(2)
    }
    assert len(outs) == 1


def test_fingerprint_handles_callables_without_code():
    import functools

    fp = fingerprint(ENV, functools.partial(obj_a))
    assert isinstance(fp, str) and len(fp) == 16


def test_empty_file_from_crash_during_header_is_reinitialised(tmp_path):
    path = tmp_path / "c.jsonl"
    path.touch()
    cache = GenomeCache(path, "fp")
    cache.put((0,), 1.0)
    assert GenomeCache(path, "fp").get((0,)) == 1.0
