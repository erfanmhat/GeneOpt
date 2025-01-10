import asyncio
import sys

import aiohttp
import numpy as np

from GeneOpt import GeneticAlgorithm, GeneticAlgorithmComparisonType, GeneticAlgorithmCacheType, Logger

timeout = 60 * 30


async def objective_func(port, **parameters):
    """Objective function to evaluate parameters."""
    url = f"http://127.0.0.1:{port}/run_objective"
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=parameters, timeout=timeout) as response:
                    response.raise_for_status()
                    res_json = await response.json()
                    if res_json and res_json.get("data"):
                        return float(res_json["data"])
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f"Error occurred: {e}, retrying...")
            await asyncio.sleep(5)


def run_async_objective_func(port, **parameters):
    """Wrapper for running async objective function."""
    return asyncio.run(objective_func(port=port, **parameters))


if __name__ == "__main__":
    environment = {
        "x1": np.linspace(-10, 10, 2 ** 12),
        "x2": np.linspace(-10, 10, 2 ** 12)
    }
    genetic_algorithm = GeneticAlgorithm(
        objective_func=run_async_objective_func,
        comparison_type=GeneticAlgorithmComparisonType.minimize,
        optimizer_name="test_op2",
        environment=environment,
        seed=42,
        mutation_rate=0.06,
        number_of_generations=100,
        number_of_population=200,
        r_cross=0.8,
        start_number_of_population=200,
        tournament_selection_number=4,
        cache_type=GeneticAlgorithmCacheType.Ram,
        verbose=1,
        number_of_workers=int(sys.argv[1])
    )
    best, score = genetic_algorithm.start()
    Logger.log_m("Done!")
    Logger.log_m(f"{best} = {score}")
    genetic_algorithm.plot()
