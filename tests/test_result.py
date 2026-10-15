import dataclasses
import json

import numpy as np
import pytest

from geneopt.core import Result


def make_result():
    return Result(params={"n": np.int64(3), "x": np.float64(0.5), "act": "relu"}, score=np.float64(-1.5),
                  generation=4, n_evaluations=10, n_cache_hits=2, elapsed=0.1, stopped_by="generations")


def test_values_are_python_native_and_json_serialisable():
    result = make_result()
    assert type(result.params["n"]) is int and type(result.score) is float
    json.dumps(dataclasses.asdict(result))


def test_result_is_frozen_and_not_tuple_unpackable():
    result = make_result()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.score = 0.0
    with pytest.raises(TypeError):
        best, score = result
