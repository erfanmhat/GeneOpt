"""Per-generation record of the whole population, cache hits included."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class History:
    environment: dict
    maximize: bool = True
    genomes: list = field(default_factory=list)  # per generation: (pop, n_params) int index array
    scores: list = field(default_factory=list)   # per generation: (pop,) float
    hits: list = field(default_factory=list)     # per generation: (pop,) bool, served from cache

    def append(self, genomes, scores, hits) -> None:
        self.genomes.append(np.array(genomes, copy=True))
        self.scores.append(np.asarray(scores, dtype=float).copy())
        self.hits.append(np.asarray(hits, dtype=bool).copy())

    def __len__(self) -> int:
        return len(self.genomes)

    def to_frame(self):
        """Long DataFrame: one row per individual per generation (needs pandas)."""
        import pandas as pd

        names = list(self.environment)
        rows = []
        for gen, (genomes, scores, hits) in enumerate(zip(self.genomes, self.scores, self.hits)):
            for i, (genome, score, hit) in enumerate(zip(genomes, scores, hits)):
                values = [self.environment[n][g] for n, g in zip(names, genome)]
                rows.append([gen, i, *values, float(score), bool(hit)])
        return pd.DataFrame(rows, columns=["generation", "individual", *names, "score", "cache_hit"])
