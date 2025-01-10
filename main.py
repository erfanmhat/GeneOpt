import numpy as np

from GeneOpt import GeneticAlgorithm, GeneticAlgorithmComparisonType, GeneticAlgorithmCacheType, Logger


def holder_table(x1, x2):
    X = (x1, x2)
    return -abs(np.sin(X[0]) * np.cos(X[1]) * np.exp(abs(1 - np.sqrt(X[0] ** 2 + X[1] ** 2) / np.pi)))


if __name__ == "__main__":
    environment = {
        "x1": np.linspace(-10, 10, 2 ** 12),
        "x2": np.linspace(-10, 10, 2 ** 12)
    }
    genetic_algorithm = GeneticAlgorithm(
        objective_func=holder_table,
        comparison_type=GeneticAlgorithmComparisonType.minimize,
        optimizer_name="op_holder_table",
        environment=environment,
        seed=42,
        mutation_rate=0.06,
        number_of_generations=100,
        number_of_population=100,
        r_cross=0.63,
        start_number_of_population=100,
        tournament_selection_number=4,
        cache_type=GeneticAlgorithmCacheType.HardDisk,
        verbose=1
    )
    best, score = genetic_algorithm.start()
    Logger.log_m("Done!")
    Logger.log_m(f"{best} = {score}")
    genetic_algorithm.plot()
