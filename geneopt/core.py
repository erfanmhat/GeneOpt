"""The genetic algorithm."""

import inspect
import logging
import math
import numbers
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .cache import GenomeCache, _py, fingerprint
from .history import History

log = logging.getLogger("geneopt")

MAX_MUTATION_RATE = 0.5  # ceiling of the adaptive rate (reached when the population has fully converged)
CREEP_SIGMA = 0.1  # std of a creep step, as a fraction of the parameter's value count


StopReason = Literal["generations", "early_stopping", "target_score"]


@dataclass(frozen=True, slots=True)
class Result:
    params: dict[str, Any]
    score: float
    generation: int
    n_evaluations: int
    n_cache_hits: int
    elapsed: float
    stopped_by: StopReason

    def __post_init__(self):
        object.__setattr__(self, "params", {k: _py(v) for k, v in self.params.items()})
        object.__setattr__(self, "score", float(self.score))


def diversity(genomes) -> float:
    """Mean pairwise gene disagreement in [0, 1]; 0 means every genome is identical."""
    genomes = np.asarray(genomes)
    n = len(genomes)
    if n < 2:
        return 0.0
    disagree = [
        (n * n - (np.unique(col, return_counts=True)[1] ** 2).sum()) / (n * (n - 1)) for col in genomes.T
    ]
    return float(np.mean(disagree))


def _mutation_rate(base: float, diversity_value: float, adaptive: bool) -> float:
    """Per-gene mutation probability: ``base``, rising linearly to ``max(MAX_MUTATION_RATE, base)``
    as the population converges (diversity 1 -> 0)."""
    if not adaptive:
        return base
    ceiling = max(MAX_MUTATION_RATE, base)
    return base + (ceiling - base) * (1 - diversity_value)


def _tournament(signed_scores, k: int, m: int, rng) -> np.ndarray:
    """Indices of ``m`` tournament winners, each the best of exactly ``k`` candidates."""
    n = len(signed_scores)
    candidates = np.array([rng.choice(n, size=k, replace=k > n) for _ in range(m)])
    return candidates[np.arange(m), np.argmax(signed_scores[candidates], axis=1)]


def _mutate(genomes, rate: float, sizes, ordered, rng) -> np.ndarray:
    """Each gene mutates with probability ``rate``: a local creep for ordered numeric
    parameters, a uniform pick among the other values otherwise."""
    hit = rng.random(genomes.shape) < rate
    for j, (size, is_ordered) in enumerate(zip(sizes, ordered, strict=True)):
        rows = np.flatnonzero(hit[:, j])
        if size < 2 or not len(rows):
            continue
        current = genomes[rows, j]
        if is_ordered:
            step = np.rint(rng.normal(0, CREEP_SIGMA * size, len(rows))).astype(int)
            step[step == 0] = rng.choice([-1, 1], size=(step == 0).sum())
            genomes[rows, j] = np.clip(current + step, 0, size - 1)
        else:
            other = rng.integers(0, size - 1, len(rows))
            genomes[rows, j] = other + (other >= current)
    return genomes


def _score(objective, params: dict, retries: int) -> float:
    """One objective call with ``retries`` extra attempts (exponential backoff).
    Module-level so process pools can pickle it."""
    for attempt in range(retries):
        try:
            return objective(**params)
        except Exception:
            time.sleep(0.5 * 2**attempt)
    return objective(**params)  # last attempt: its exception propagates


def _is_ordered_numeric(values) -> bool:
    if len(values) < 3 or not all(isinstance(v, numbers.Real) and not isinstance(v, bool) for v in values):
        return False
    diffs = np.diff(np.asarray(values, dtype=float))
    return bool((diffs >= 0).all() or (diffs <= 0).all())


def _check_environment(environment) -> dict[str, list]:
    if not isinstance(environment, Mapping) or not environment:
        raise TypeError(f"environment must be a non-empty mapping of name -> values, got {environment!r}")
    env = {}
    for name in sorted(environment):
        values = environment[name]
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(
                f"environment key {name!r} must be a valid Python identifier "
                "(it is passed to the objective as a keyword argument)"
            )
        if isinstance(values, (str, bytes)) or not hasattr(values, "__len__") or not len(values):
            raise ValueError(f"environment[{name!r}] must be a non-empty sequence of values, got {values!r}")
        env[name] = list(values)
    return env


def _check(name: str, value, ok: bool, rule: str) -> None:
    if not ok:
        raise ValueError(f"{name} must be {rule}, got {value!r}")


class GeneticAlgorithm:
    def __init__(
        self,
        objective_func: Callable[..., float],
        environment: Mapping[str, Any],
        *,
        optimizer_name: str = "geneopt",
        maximize: bool = True,
        population_size: int = 100,
        initial_population_size: int | None = None,
        generations: int = 100,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.05,
        adaptive_mutation: bool = True,
        tournament_size: int = 3,
        elitism: int = 1,
        early_stopping_rounds: int | None = None,
        target_score: float | None = None,
        cache_dir: str | Path | None = None,
        objective_version: str | None = None,
        executor: Executor | None = None,
        on_error: Literal["raise", "worst", "retry"] = "raise",
        max_retries: int = 2,
        seed: int | None = 0,
    ) -> None:
        if not callable(objective_func):
            raise TypeError(f"objective_func must be callable, got {objective_func!r}")
        env = _check_environment(environment)
        _check("crossover_rate", crossover_rate, 0 <= crossover_rate <= 1, "in [0, 1]")
        _check("mutation_rate", mutation_rate, 0 <= mutation_rate <= 1, "in [0, 1]")
        _check("population_size", population_size, population_size >= 2, ">= 2")
        _check(
            "initial_population_size",
            initial_population_size,
            initial_population_size is None or initial_population_size >= 2,
            ">= 2",
        )
        _check("tournament_size", tournament_size, tournament_size >= 1, ">= 1")
        _check("elitism", elitism, 0 <= elitism < population_size, "in [0, population_size)")
        _check("generations", generations, generations >= 1, ">= 1")
        _check(
            "early_stopping_rounds",
            early_stopping_rounds,
            early_stopping_rounds is None or early_stopping_rounds >= 1,
            ">= 1",
        )
        _check("on_error", on_error, on_error in ("raise", "worst", "retry"), "'raise', 'worst' or 'retry'")
        _check("max_retries", max_retries, max_retries >= 0, ">= 0")
        if executor is not None and not callable(getattr(executor, "submit", None)):
            raise TypeError(f"executor must be a concurrent.futures.Executor, got {executor!r}")
        try:
            signature = inspect.signature(objective_func)
        except (TypeError, ValueError):  # some builtins/C callables have no signature
            signature = None
        if signature is not None:
            try:
                signature.bind(**{name: values[0] for name, values in env.items()})
            except TypeError as e:
                raise TypeError(
                    f"objective_func cannot be called with the environment's parameters {list(env)}: {e}"
                ) from None

        self._objective = objective_func
        self._env = env
        self._names = list(env)
        self._sizes = np.array([len(v) for v in env.values()])
        self._ordered = np.array([_is_ordered_numeric(v) for v in env.values()])
        self._sign = 1.0 if maximize else -1.0
        self._population_size = population_size
        self._initial_population_size = initial_population_size or population_size
        self._generations = generations
        self._crossover_rate = crossover_rate
        self._mutation_rate = mutation_rate
        self._adaptive_mutation = adaptive_mutation
        self._tournament_size = tournament_size
        self._elitism = elitism
        self._early_stopping_rounds = early_stopping_rounds
        self._target_score = target_score
        self._seed = seed
        self._executor = executor
        self._on_error = on_error
        self._retries = max_retries if on_error == "retry" else 0
        self.optimizer_name = optimizer_name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        fp = fingerprint(env, objective_func, objective_version)
        path = self.cache_dir / f"{optimizer_name}-{fp}.jsonl" if self.cache_dir is not None else None
        self._cache = GenomeCache(path, fp, params=self._names)
        self.history: History | None = None
        self.result: Result | None = None

    def _decode(self, key) -> dict[str, Any]:
        return {name: self._env[name][i] for name, i in zip(self._names, key, strict=True)}

    def _evaluate(self, keys):
        """Yield ``(key, score | exception)`` in submission order, so results never depend on timing."""
        if self._executor is None:
            for key in keys:
                try:
                    yield key, _score(self._objective, self._decode(key), self._retries)
                except Exception as e:
                    yield key, e
            return
        futures = [
            self._executor.submit(_score, self._objective, self._decode(k), self._retries) for k in keys
        ]
        for key, future in zip(keys, futures, strict=True):
            error = future.exception()
            yield key, error if error is not None else future.result()

    def _score_pending(self, pending) -> dict:
        """Evaluate and cache ``pending``. Every finished score is cached before an error is
        re-raised. Under ``on_error="worst"`` failures get the worst score and are not cached."""
        failed, first_error = {}, None
        for key, outcome in self._evaluate(pending):
            if not isinstance(outcome, Exception):
                log.debug("eval %s -> %s", self._decode(key), outcome)
                self._cache.put(key, outcome, values=self._decode(key).values())
            elif self._on_error == "worst":
                log.warning("objective failed for %s, scored as worst: %r", self._decode(key), outcome)
                failed[key] = -self._sign * np.inf
            else:
                first_error = first_error or outcome
                if self._executor is None:
                    break
        if first_error is not None:
            raise first_error
        return failed

    def _next_generation(self, population, signed, rng) -> np.ndarray:
        rate = _mutation_rate(self._mutation_rate, diversity(population), self._adaptive_mutation)

        elites: list[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()
        for i in np.argsort(-signed, kind="stable"):
            if len(elites) == self._elitism:
                break
            key = tuple(population[i])
            if key not in seen:
                seen.add(key)
                elites.append(population[i])

        n_children = self._population_size - len(elites)
        n_pairs = (n_children + 1) // 2
        parents = population[_tournament(signed, self._tournament_size, 2 * n_pairs, rng)]
        p1, p2 = parents[:n_pairs], parents[n_pairs:]
        swap = rng.random(p1.shape) < 1.0 / p1.shape[1]
        swap &= (rng.random(n_pairs) < self._crossover_rate)[:, None]
        children = np.vstack([np.where(swap, p2, p1), np.where(swap, p1, p2)])[:n_children]
        children = _mutate(children, rate, self._sizes, self._ordered, rng)
        return np.vstack([np.array(elites).reshape(-1, len(self._names)), children])

    def evaluations(self):
        """Unique evaluated points as a DataFrame ``(<params...>, score)`` (needs pandas)."""
        import pandas as pd

        rows = [[*self._decode(key).values(), score] for key, score in self._cache.items()]
        return pd.DataFrame(rows, columns=[*self._names, "score"])

    def plot(self, output_dir: str | Path | None = None, dpi: int = 150) -> dict:
        """Write every plot as PNG; returns ``{name: path | Exception}`` (one failure never stops the rest).
        Default directory: ``{cache_dir}/{optimizer_name}-plots`` or ``./{optimizer_name}-plots``."""
        if self.history is None:
            raise RuntimeError("call start() before plot()")
        from .plots import plot_features, plot_history

        if output_dir is None:
            output_dir = (self.cache_dir or Path(".")) / f"{self.optimizer_name}-plots"
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        results = plot_history(self.history, output_dir, dpi)
        results.update(plot_features(self.evaluations(), output_dir, dpi))
        for name, outcome in results.items():
            if isinstance(outcome, Exception):
                log.warning("plot %s failed: %r", name, outcome)
        return results

    def start(self) -> Result:
        started = time.perf_counter()
        rng = np.random.default_rng(self._seed)
        self.history = History(self._env, maximize=self._sign > 0)
        n_evaluations = n_cache_hits = 0
        best_key, best_signed, best_generation, last_improvement = None, -np.inf, 0, 0
        stopped_by: StopReason = "generations"
        population = rng.integers(0, self._sizes, size=(self._initial_population_size, len(self._names)))

        for generation in range(self._generations):
            keys = [tuple(int(i) for i in genome) for genome in population]
            pending = list(dict.fromkeys(k for k in keys if k not in self._cache))  # dedupe (GC-7)
            failed = self._score_pending(pending)
            n_evaluations += len(pending)
            new = set(pending)
            hits = [k not in new for k in keys]
            n_cache_hits += sum(hits)
            scores = np.array([failed[k] if k in failed else self._cache.get(k) for k in keys])
            self.history.append(population, scores, hits)

            signed = self._sign * scores
            i = int(np.argmax(signed))
            if best_key is None or signed[i] > best_signed:
                best_key, best_signed, best_generation = keys[i], signed[i], generation
                last_improvement = generation
            log.info(
                "generation %d/%d: best %s, %d new evaluations, %d cache hits",
                generation + 1,
                self._generations,
                self._sign * best_signed,
                len(pending),
                sum(hits),
            )

            if self._target_score is not None and best_signed >= self._sign * self._target_score:
                stopped_by = "target_score"
                break
            if (
                self._early_stopping_rounds is not None
                and generation - last_improvement >= self._early_stopping_rounds
            ):
                stopped_by = "early_stopping"
                break
            if generation < self._generations - 1:
                population = self._next_generation(population, signed, rng)

        self.result = Result(
            params=self._decode(best_key),
            score=self._sign * best_signed,
            generation=best_generation,
            n_evaluations=n_evaluations,
            n_cache_hits=n_cache_hits,
            elapsed=time.perf_counter() - started,
            stopped_by=stopped_by,
        )
        space = math.prod(int(s) for s in self._sizes)
        log.info(
            "done (%s): best %s = %s after %d evaluations of a %d-point space, %d cache hits, %.2fs",
            stopped_by,
            self.result.params,
            self.result.score,
            n_evaluations,
            space,
            n_cache_hits,
            self.result.elapsed,
        )
        return self.result
