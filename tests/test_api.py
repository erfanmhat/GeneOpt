import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import geneopt
from geneopt import GeneticAlgorithm, History, Result


def test_version():
    assert geneopt.__version__ == "2.0.0"


def run_python(code: str, cwd) -> None:
    # keep the rest of the environment: Windows Python cannot start without SYSTEMROOT
    env = {**os.environ, "PYTHONPATH": str(Path(geneopt.__file__).parent.parent)}
    subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, check=True)


def test_A_4_import_pulls_in_no_optional_dependency(tmp_path):
    # 1.x imported catboost (and sklearn, seaborn, pandas, matplotlib) at module level
    run_python(
        "import sys, geneopt\n"
        "heavy = {'matplotlib', 'pandas', 'seaborn', 'catboost', 'sklearn'} & set(sys.modules)\n"
        "assert not heavy, heavy\n",
        tmp_path,
    )


def test_A_3_import_and_run_write_no_files_without_cache_dir(tmp_path):
    # 1.x created ./cache/<timestamp>.log at import time and appended to it on every log call
    run_python(
        "import logging\n"
        "logging.basicConfig(level=logging.DEBUG)\n"
        "from geneopt import GeneticAlgorithm\n"
        "GeneticAlgorithm(lambda x: x, {'x': [1, 2, 3]}, generations=3, population_size=4).start()\n",
        tmp_path,
    )
    assert list(tmp_path.iterdir()) == []


ENV = {"x": [0.0, 1.0, 2.0]}


def objective(x):
    return x


def test_public_names():
    assert geneopt.__all__ == ["GeneticAlgorithm", "Result", "History", "__version__"]
    assert geneopt.Result is Result and geneopt.History is History


@pytest.mark.parametrize(
    "kwarg",
    [
        "start_number_of_population",
        "number_of_population",
        "number_of_generations",
        "comparison_type",
        "r_cross",
        "tournament_selection_number",
        "cache_type",
        "cache_path",
        "number_of_workers",
        "server_start_port",
        "verbose",
        "plot_DPI",
    ],
)
def test_removed_1x_kwargs_raise_type_error(kwarg):
    with pytest.raises(TypeError, match=kwarg):
        GeneticAlgorithm(objective, ENV, **{kwarg: 1})


def test_environment_is_required_positional():
    with pytest.raises(TypeError):
        GeneticAlgorithm(objective)
    with pytest.raises(TypeError):
        GeneticAlgorithm(objective, ENV, "name")  # everything after environment is keyword-only


@pytest.mark.parametrize(
    "kwargs, name",
    [
        ({"crossover_rate": 1.5}, "crossover_rate"),
        ({"mutation_rate": -0.1}, "mutation_rate"),
        ({"population_size": 1}, "population_size"),
        ({"initial_population_size": 1}, "initial_population_size"),
        ({"tournament_size": 0}, "tournament_size"),
        ({"elitism": 5, "population_size": 5}, "elitism"),
        ({"generations": 0}, "generations"),
        ({"early_stopping_rounds": 0}, "early_stopping_rounds"),
        ({"mutation_rate": 1.5}, "mutation_rate"),
    ],
)
def test_invalid_numbers_raise_value_error_naming_the_parameter(kwargs, name):
    with pytest.raises(ValueError, match=name):
        GeneticAlgorithm(objective, ENV, **kwargs)


@pytest.mark.parametrize("environment", [{}, {"x": []}, {"x": "abc"}, {"not valid": [1]}, None])
def test_invalid_environment_raises(environment):
    with pytest.raises((ValueError, TypeError), match="environment"):
        GeneticAlgorithm(objective, environment)


def test_objective_must_be_callable():
    with pytest.raises(TypeError, match="objective_func"):
        GeneticAlgorithm(42, ENV)


def test_misspelled_parameter_fails_before_any_evaluation():
    calls = []

    def f(x1, x2):
        calls.append(1)
        return 0.0

    with pytest.raises(TypeError, match="environment"):
        GeneticAlgorithm(f, {"x1": [0], "x_2": [0]})
    assert calls == []


def test_numpy_arrays_are_accepted_as_value_lists():
    r = GeneticAlgorithm(objective, {"x": np.linspace(0, 1, 5)}, generations=2, population_size=4).start()
    assert isinstance(r, Result)
