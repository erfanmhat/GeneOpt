"""Problems from COCO bbob-mixint and IOHprofiler PBO (spec 015).

Each problem exposes the integer index space every optimiser searches (``sizes``), the value the
optimiser minimises (``objective``), the scored value of every evaluation in order (``scored``)
and, from ``finish()``, the reference the scored values are measured against.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GRID = 256  # shared grid for bbob-mixint's continuous variables, identical for every optimiser
MIXINT_FUNCTIONS = tuple(range(1, 25))
PBO_FUNCTIONS = (4, 5, 6, 7, 8, 9, 10, 18, 19, 22, 23, 24, 25)
MIXINT_TARGETS = tuple(10.0 ** (2 - 0.2 * k) for k in range(51))  # COCO's f - f_opt levels, 1e2 .. 1e-8
PBO_TARGETS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0)  # relative gap to the reference


@dataclass
class SuiteProblem:
    suite: str
    function: int
    dim: int
    instance: int
    sizes: tuple[int, ...]
    maximize: bool  # direction of the *scored* values; optimisers always minimise ``objective``
    evaluate: Callable[[tuple[int, ...]], tuple[float, float]]  # index -> (minimised, scored)
    close: Callable[[], float | None]
    scored: list[float] = field(default_factory=list)

    def objective(self, index) -> float:
        value, scored = self.evaluate(tuple(int(i) for i in index))
        self.scored.append(scored)
        return value

    def finish(self) -> float | None:
        """Flush the suite's logs; return f_opt (mixint) or the stated optimum (PBO, None if unknown)."""
        return self.close()


def mixint(function: int, dim: int, instance: int, log_dir: Path, label: str = "run") -> SuiteProblem:
    import cocoex

    cocoex.log_level("warning")
    suite = cocoex.Suite(
        "bbob-mixint", f"instances:{instance}", f"function_indices:{function} dimensions:{dim}"
    )
    p = suite.get_problem(0)
    folder = Path(log_dir).resolve()
    run = f"{label}/f{function}_d{dim}_i{instance}"
    # shortcut: COCO splits options on spaces, so log paths must not contain any
    observer = cocoex.Observer(
        "bbob-mixint", f"outer_folder: {folder} result_folder: {run} algorithm_name: {label}"
    )
    p.observe_with(observer)
    n_int = p.number_of_integer_variables
    lo, hi = p.lower_bounds, p.upper_bounds
    values = [
        np.arange(lo[i], hi[i] + 1) if i < n_int else np.linspace(lo[i], hi[i], GRID) for i in range(dim)
    ]

    def evaluate(index):
        f = float(p(np.array([values[i][j] for i, j in enumerate(index)])))
        return f, f

    def close():
        p.free()
        dat = next((folder / run).rglob("*.dat"), None)  # COCO writes no log before the first evaluation
        if dat is None:
            return None
        return float(re.search(r"Fopt \(([^)]+)\)", dat.read_text().splitlines()[0]).group(1))

    return SuiteProblem(
        "mixint", function, dim, instance, tuple(len(v) for v in values), False, evaluate, close
    )


def pbo(function: int, dim: int, instance: int, log_dir: Path, label: str = "run") -> SuiteProblem:
    import ioh

    q = ioh.get_problem(function, instance=instance, dimension=dim, problem_class=ioh.ProblemClass.PBO)
    logger = ioh.logger.Analyzer(
        root=str(Path(log_dir).resolve()),
        folder_name=f"{label}-F{function}-n{dim}-i{instance}",
        algorithm_name=label,
    )
    q.attach_logger(logger)
    # instance 1 is untransformed, so its stated optimum is on the internal scale we score
    stated = ioh.get_problem(
        function, instance=1, dimension=dim, problem_class=ioh.ProblemClass.PBO
    ).optimum.y

    def evaluate(index):
        y = q(list(index))  # PBO maximises the transformed value the optimiser sees
        return -y, q.state.current_internal.y

    def close():
        logger.close()
        return None if np.isinf(stated) else float(stated)

    return SuiteProblem("pbo", function, dim, instance, (2,) * dim, True, evaluate, close)


BUILDERS = {"mixint": mixint, "pbo": pbo}
FUNCTIONS = {"mixint": MIXINT_FUNCTIONS, "pbo": PBO_FUNCTIONS}
TARGETS = {"mixint": MIXINT_TARGETS, "pbo": PBO_TARGETS}
