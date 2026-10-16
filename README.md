# GeneOpt

A genetic algorithm for searching **discrete** spaces: numeric grids, categorical choices, or a mix.
It caches every evaluation (in memory or on disk), can evaluate a generation in parallel on any
`concurrent.futures` executor, and plots what the population did.

```bash
pip install GeneOpt                # core: numpy only
pip install "GeneOpt[plots]"       # + matplotlib, pandas  (ga.plot(), ga.evaluations())
pip install "GeneOpt[analysis]"    # + catboost            (feature-importance plots)
```

Requires Python 3.10+.

## Quickstart

Each environment entry lists the values a parameter may take. The objective receives one value
per parameter as keyword arguments and returns a number.

```python
"""Minimise the Holder table function on a 4096 x 4096 grid."""

import logging

import numpy as np

from geneopt import GeneticAlgorithm


def holder_table(x1, x2):
    return -abs(np.sin(x1) * np.cos(x2) * np.exp(abs(1 - np.sqrt(x1**2 + x2**2) / np.pi)))


def main(generations=100, population_size=100):
    environment = {
        "x1": np.linspace(-10, 10, 2**12),
        "x2": np.linspace(-10, 10, 2**12),
    }
    ga = GeneticAlgorithm(
        holder_table,
        environment,
        optimizer_name="holder_table",
        maximize=False,
        population_size=population_size,
        generations=generations,
        tournament_size=4,
        cache_dir="./cache",
        seed=42,
    )
    result = ga.start()
    print(f"{result.params} = {result.score}")
    ga.plot()  # PNGs in ./cache/holder_table-plots/
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

`start()` returns a `Result`:

| Field | Meaning |
|---|---|
| `params` | best parameters found (plain Python values) |
| `score` | their score |
| `generation` | generation in which they were found |
| `n_evaluations` | objective calls made in this run (cache misses) |
| `n_cache_hits` | individuals served from the cache |
| `elapsed` | wall time in seconds |
| `stopped_by` | `"generations"`, `"early_stopping"` or `"target_score"` |

After `start()`, `ga.history` holds every generation's genomes, scores and cache hits
(`ga.history.to_frame()` gives a long DataFrame), and `ga.evaluations()` gives the unique
evaluated points.

## Categorical and mixed spaces

Values can be anything: strings, ints, floats, objects. GeneOpt uses the standard operators for
this kind of search: tournament selection with elitism, uniform crossover, and random-resetting
mutation, where a mutated parameter takes a uniformly chosen other value.

```python
"""Search a mixed categorical / numeric space, e.g. model hyperparameters."""

import logging
import math

from geneopt import GeneticAlgorithm


def validation_score(optimizer, activation, layers, learning_rate, batch_size):
    """A stand-in for training a model; replace with real training + validation."""
    score = {"sgd": 0.80, "adam": 0.88, "rmsprop": 0.85}[optimizer]
    score += {"relu": 0.02, "tanh": 0.0, "gelu": 0.03}[activation]
    score -= 0.01 * abs(layers - 3)
    score -= 0.02 * abs(math.log10(learning_rate) + 3)
    score -= 0.005 * abs(math.log2(batch_size) - 6)
    return score


def main(generations=30, population_size=30):
    environment = {
        "optimizer": ["sgd", "adam", "rmsprop"],
        "activation": ["relu", "tanh", "gelu"],
        "layers": [1, 2, 3, 4, 5, 6],
        "learning_rate": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "batch_size": [16, 32, 64, 128, 256],
    }
    ga = GeneticAlgorithm(
        validation_score,
        environment,
        optimizer_name="hparams",
        population_size=population_size,
        generations=generations,
        seed=0,
    )
    result = ga.start()
    print(
        f"best: {result.params} -> {result.score:.3f} "
        f"({result.n_evaluations} of {3 * 3 * 6 * 5 * 5} configurations evaluated)"
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `objective_func` | required | called as `objective_func(**params)`, returns a number. `NaN` is rejected; `±inf` is allowed |
| `environment` | required | `{name: values}`; names must be valid identifiers |
| `optimizer_name` | `"geneopt"` | names the cache file and the plot directory |
| `maximize` | `True` | `False` to minimise |
| `population_size` | `100` | individuals per generation |
| `generations` | `100` | maximum number of generations |
| `crossover_rate` | `0.9` | probability that a pair of parents performs uniform crossover (each gene swapped with p = 0.5) |
| `mutation_rate` | `1/n_params` | per-gene probability of resetting to a random other value; the default gives one mutation per child on average |
| `tournament_size` | `3` | candidates per tournament selection |
| `elitism` | `1` | best distinct genomes copied unchanged into the next generation |
| `early_stopping_rounds` | `None` | stop after this many generations without improvement |
| `target_score` | `None` | stop once the best score reaches this value |
| `cache_dir` | `None` | directory for the on-disk cache; `None` keeps it in memory |
| `objective_version` | `None` | bump to invalidate the disk cache when the objective changes in ways the code hash can't see |
| `executor` | `None` | a `concurrent.futures.Executor` to evaluate each generation in parallel |
| `on_error` | `"raise"` | `"raise"`, `"worst"` (score failures as the worst value, uncached) or `"retry"` |
| `max_retries` | `2` | extra attempts per evaluation with `on_error="retry"` (backoff 0.5 s, 1 s, 2 s, ...) |
| `seed` | `0` | random seed; the same seed gives the same run with or without an executor |

Everything after `environment` is keyword-only.

## Caching

Every genome is evaluated at most once per run, and duplicates inside a generation are evaluated
once. With `cache_dir`, scores are appended to `{cache_dir}/{optimizer_name}-{fingerprint}.jsonl`
and reused by later runs. The fingerprint covers the environment's values and the objective's code,
so changing either starts a fresh cache instead of returning stale scores. The code hash only sees
the objective's own body; if it calls other functions that change, pass a new `objective_version`.

## Parallel evaluation

Pass any executor. Results are consumed in submission order, so the run is identical to a
sequential one with the same seed. Use a `ProcessPoolExecutor` for CPU-bound Python objectives
(the objective must then be importable, i.e. defined at module level).

```python
from concurrent.futures import ThreadPoolExecutor

from geneopt import GeneticAlgorithm


def sphere(x, y):
    return x * x + y * y


environment = {"x": list(range(-50, 51)), "y": list(range(-50, 51))}
with ThreadPoolExecutor(max_workers=4) as pool:
    result = GeneticAlgorithm(
        sphere, environment, maximize=False, generations=20, population_size=20, executor=pool
    ).start()
print(result.params, result.score)
```

### Objective servers over HTTP

`examples/distributed_http/` runs the objective in separate server processes (or machines) and
lets each executor thread borrow a free server:

```bash
cd examples/distributed_http
pip install flask
python run_servers.py 4        # servers on ports 5000-5003
python client.py 4             # the genetic algorithm, 4 evaluations at a time
```

The servers listen on `127.0.0.1` with debug mode off. To spread them over several machines, run
`server.py` behind a WSGI server on a private network; the endpoint has no authentication.

## Logging

GeneOpt logs to the `geneopt` logger and installs no handlers: one INFO line per generation and a
summary, DEBUG per evaluation, WARNING for failures scored as worst.

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("geneopt").setLevel(logging.DEBUG)
```

## Plots

`ga.plot(output_dir=None, dpi=150)` writes PNGs to `{cache_dir}/{optimizer_name}-plots/` (or
`./{optimizer_name}-plots/`) and returns `{name: path or exception}`; one failing plot never stops
the others. Images below are from the quickstart.

| | |
|---|---|
| ![Best score](https://raw.githubusercontent.com/erfanmhat/GeneOpt/master/docs/images/best_score.png) | ![Best x1](https://raw.githubusercontent.com/erfanmhat/GeneOpt/master/docs/images/param_x1.png) |
| Best so far, generation best and median | Best value of each parameter |
| ![Diversity](https://raw.githubusercontent.com/erfanmhat/GeneOpt/master/docs/images/diversity.png) | ![Gene frequencies](https://raw.githubusercontent.com/erfanmhat/GeneOpt/master/docs/images/gene_freq_x1.png) |
| Mean gene disagreement in the population | Share of the population holding each value |
| ![Cache hits](https://raw.githubusercontent.com/erfanmhat/GeneOpt/master/docs/images/cache_hit_rate.png) | ![Correlation with score](https://raw.githubusercontent.com/erfanmhat/GeneOpt/master/docs/images/corr_with_score.png) |
| Share of each generation served from cache | Spearman correlation of numeric params with score |
| ![Correlation matrix](https://raw.githubusercontent.com/erfanmhat/GeneOpt/master/docs/images/corr_matrix.png) | ![Predicted vs actual](https://raw.githubusercontent.com/erfanmhat/GeneOpt/master/docs/images/predicted_vs_actual.png) |
| Correlation between parameters | CatBoost fit of score on parameters (`[analysis]`) |
| ![Feature importance](https://raw.githubusercontent.com/erfanmhat/GeneOpt/master/docs/images/feature_importance.png) | ![Histogram](https://raw.githubusercontent.com/erfanmhat/GeneOpt/master/docs/images/hist_x1.png) |
| CatBoost feature importance (`[analysis]`) | Distribution of evaluated values |
| ![Regression](https://raw.githubusercontent.com/erfanmhat/GeneOpt/master/docs/images/regression_x2.png) | ![Pair scatter](https://raw.githubusercontent.com/erfanmhat/GeneOpt/master/docs/images/scatter_x1_x2.png) |
| Parameter vs score with a linear fit | Parameter pairs coloured by score |

Feature plots use the unique evaluated points with finite scores and numeric parameters only.

## Upgrading from 1.x

2.0.0 is a hard break: every 1.x parameter, module and helper was removed, without
compatibility shims. 1.x cache files are ignored. The complete list is in
[specs/011-removed-in-2.0.md](specs/011-removed-in-2.0.md).

## License

MIT
