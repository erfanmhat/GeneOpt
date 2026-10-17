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
