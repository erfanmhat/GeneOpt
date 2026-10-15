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
    ga = GeneticAlgorithm(validation_score, environment, optimizer_name="hparams",
                          population_size=population_size, generations=generations, seed=0)
    result = ga.start()
    print(f"best: {result.params} -> {result.score:.3f} "
          f"({result.n_evaluations} of {3 * 3 * 6 * 5 * 5} configurations evaluated)")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
