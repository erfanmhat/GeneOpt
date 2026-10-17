"""Per-generation record of the whole population, cache hits included."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cache import _py


@dataclass
class History:
    environment: dict
    maximize: bool = True
    genomes: list = field(default_factory=list)  # per generation: (pop, n_params) int32 index array
    scores: list = field(default_factory=list)  # per generation: (pop,) float
    hits: list = field(default_factory=list)  # per generation: (pop,) bool, served from cache

    def append(self, genomes, scores, hits) -> None:
        # int32 halves the memory; a parameter would need 2**31 values to overflow it
        self.genomes.append(np.array(genomes, dtype=np.int32))
        self.scores.append(np.asarray(scores, dtype=float).copy())
        self.hits.append(np.asarray(hits, dtype=bool).copy())

    def __len__(self) -> int:
        return len(self.genomes)

    def save(self, path) -> Path:
        """Write the history to a compressed ``.npz`` (no pickle). Returns the file written:
        numpy adds ``.npz`` when ``path`` lacks it. Environment values must be JSON values."""
        environment = {}
        for name, values in self.environment.items():
            values = [_py(v) for v in values]
            bad = next((v for v in values if not isinstance(v, (str, int, float, bool, type(None)))), None)
            if bad is not None:
                raise TypeError(f"cannot save parameter {name!r}: value {bad!r} is not a JSON value")
            environment[name] = values
        n = len(environment)
        path = Path(path)
        np.savez_compressed(
            path,
            environment=json.dumps(environment),
            maximize=self.maximize,
            genomes=np.stack(self.genomes) if self.genomes else np.empty((0, 0, n), dtype=np.int32),
            scores=np.stack(self.scores) if self.scores else np.empty((0, 0)),
            hits=np.stack(self.hits) if self.hits else np.empty((0, 0), dtype=bool),
        )
        return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")

    @classmethod
    def load(cls, path) -> "History":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                json.loads(str(data["environment"])),
                bool(data["maximize"]),
                list(data["genomes"]),
                list(data["scores"]),
                list(data["hits"]),
            )

    def to_frame(self):
        """Long DataFrame: one row per individual per generation (needs pandas)."""
        import pandas as pd

        names = list(self.environment)
        rows = []
        for gen, (genomes, scores, hits) in enumerate(zip(self.genomes, self.scores, self.hits, strict=True)):
            for i, (genome, score, hit) in enumerate(zip(genomes, scores, hits, strict=True)):
                values = [self.environment[n][g] for n, g in zip(names, genome, strict=True)]
                rows.append([gen, i, *values, float(score), bool(hit)])
        return pd.DataFrame(rows, columns=["generation", "individual", *names, "score", "cache_hit"])
