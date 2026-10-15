"""The genetic algorithm."""
from dataclasses import dataclass
from typing import Any, Literal

from .cache import _py


@dataclass(frozen=True, slots=True)
class Result:
    params: dict[str, Any]
    score: float
    generation: int
    n_evaluations: int
    n_cache_hits: int
    elapsed: float
    stopped_by: Literal["generations", "early_stopping", "target_score"]

    def __post_init__(self):
        object.__setattr__(self, "params", {k: _py(v) for k, v in self.params.items()})
        object.__setattr__(self, "score", float(self.score))
