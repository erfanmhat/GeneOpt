import numpy as np

from GeneOpt import GeneticAlgorithm, GeneticAlgorithmComparisonType, GeneticAlgorithmCacheType, Logger


def run_objective_func(a, b, c):
    return a + b + c


if __name__ == "__main__":
    environment = {
        "a": np.linspace(0, 100, 1024),
        "b": np.linspace(0, 100, 1024),
        "c": np.linspace(0, 100, 1024)
    }
    genetic_algorithm = GeneticAlgorithm(
        objective_func=run_objective_func,
        comparison_type=GeneticAlgorithmComparisonType.maximize,
        optimizer_name="test_op",
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
