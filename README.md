# GeneOpt - Genetic Algorithm Python Library

GeneOpt is a Python library designed for creating and optimizing genetic algorithms, including support for distributed evaluation and detailed plotting tools. It provides a robust framework for solving optimization problems by simulating the principles of natural selection.

## Features

- **Custom Objective Function**: Easily integrate your specific optimization goals.
- **Flexible Optimization Goals**: Support for both maximization and minimization.
- **Distributed Computation**: Enables scaling across multiple servers for resource-intensive tasks.
- **Advanced Configurations**: Control mutation rates, crossover rates, and population dynamics.
- **Comprehensive Caching**: Choose between `Ram` or `HardDisk` for efficient handling of large datasets.
- **Detailed Plotting**: Visualize best scores, parameter values, diversity, and more across generations.

## Repository

Find the source code and contribute on GitHub: [GeneOpt Repository](https://github.com/erfanmhat/GeneOpt)

## Installation

Install GeneOpt via pip:

```bash
pip install GeneOpt
```

## Getting Started

Here's a simplified example to get started with GeneOpt:

### Local Optimization

This example demonstrates using GeneOpt to optimize a function locally:

```python
import numpy as np
from GeneOpt import GeneticAlgorithm, GeneticAlgorithmComparisonType, GeneticAlgorithmCacheType, Logger


def objective_function(a, b, c):
    return a + b + c


if __name__ == "__main__":
    # Define the search space
    environment = {
        "a": np.linspace(0, 100, 1024),
        "b": np.linspace(0, 100, 1024),
        "c": np.linspace(0, 100, 1024)
    }

    # Initialize the genetic algorithm
    genetic_algorithm = GeneticAlgorithm(
        objective_func=objective_function,
        comparison_type=GeneticAlgorithmComparisonType.maximize,
        optimizer_name="test_optimizer",
        environment=environment,
        seed=42,
        mutation_rate=0.06,
        number_of_generations=100,
        number_of_population=100,
        r_cross=0.63,
        start_number_of_population=100,
        tournament_selection_number=4,
        cache_type=GeneticAlgorithmCacheType.Ram,
        verbose=1
    )

    # Run the genetic algorithm
    best_solution, best_score = genetic_algorithm.start()

    # Log results
    Logger.log_m("Optimization Complete!")
    Logger.log_m(f"Best Solution: {best_solution}, Best Score: {best_score}")

    # Plot results
    genetic_algorithm.plot_best_score()
    genetic_algorithm.plot_parameter_values_for_the_best_scores()
    genetic_algorithm.plot_population_diversity()
    genetic_algorithm.plot_gene_frequencies()
    genetic_algorithm.feature_plots()
```

### Outputs

1. **Best Solution**: Prints the optimal parameters and corresponding fitness score.
2. **Plots**: Saves detailed graphs to the `./cache/` directory.

## Plotting Functions

GeneOpt provides a variety of detailed plots to understand the algorithm's performance:

### 1. Best Score Over Generations
Plots the best score achieved in each generation.
![Best Score Over Generations Image](./image/best_score_over_generations.png)

### 2. Best Parameter Values Over Generations
Shows the evolution of parameters contributing to the best scores.
![Best Parameter a Value Over Generations Image](./image/best_parameter_a_value_over_generations.png)

### 3. Population Diversity
Illustrates the diversity within the population for each generation.
![Population Diversity Image](./image/population_diversity.png)

### 4. Gene Frequencies
Displays the frequency of genes across generations using a heatmap.
![Gene Frequencies Image](./image/gene_frequencies.png)

### 5. Feature Analysis
Includes:
- Pairwise correlation with score.
![Pairwise correlation with score Image](./image/pairwise_correlation_with_score.png)
- Pairwise correlation of features.
![Pairwise correlation of features Image](./image/pairwise_correlation_of_features.png)
- Predicted vs Actual values using regression.
![Predicted vs Actual values using regression Image](./image/predicted_vs_actual_values_using_regression.png)
- Feature importance analysis.
![Feature importance analysis Image](./image/feature_importance_analysis.png)
- Histograms plot of features.
![Histograms plot of features Image](./image/histograms_plot_of_features.png)
- scatter plot of features.
![scatter plot of features Image](./image/scatter_plot_of_features.png)
- KDE plot of features.
![KDE plot of features](./image/kde_plot_of_features.png)
- Parameter and Parameter and score scatter plots
![Parameter and Parameter and score scatter plots Image](./image/parameter_and_parameter_and_score_scatter_plots.png)

## Distributed Optimization with GeneOpt

GeneOpt can also be used for distributed optimization by leveraging multiple servers. This setup evaluates the objective function across different ports, enabling efficient parallel processing for computationally intensive tasks.

### Components

1. **Server**: A Flask-based API to evaluate the objective function(or any other programing language and framework).
2. **Server Runner**: A script to start multiple instances of the server.
3. **Main**: The main genetic algorithm execution script using asynchronous requests.

### Server Code (`server.py`)

The server evaluates the objective function via HTTP GET requests.

```python
import sys
from flask import Flask, jsonify, request

app = Flask(__name__)

def objective_func(a, b, c):
    return (a - 5) ** 2 + (b - 20) ** 2 + (c - 33.33) ** 4

@app.route('/run_objective', methods=['GET'])
def run_objective():
    input_parameters = request.args.to_dict()
    try:
        parameters = {key: float(value) for key, value in input_parameters.items()}
    except ValueError:
        res = jsonify({"data": "Bad request: All parameters must be numeric"})
        res.status_code = 400
        return res

    return jsonify({"data": objective_func(**parameters)})

if __name__ == '__main__':
    app.run(
        debug=True,
        port=int(sys.argv[1]),
        threaded=False
    )
```

### Server Runner (`RunNServer.py`)

This script launches multiple server instances on different ports.

```python
import subprocess
import sys

processes = []

for index in range(int(sys.argv[1])):
    process = subprocess.Popen(["python", "server.py", str(5000 + index)])
    processes.append(process)

# Wait for all processes to finish
for process in processes:
    process.wait()
```

### Genetic Algorithm Execution (`main.py`)

The main script initializes the genetic algorithm and evaluates the objective function asynchronously.

```python
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
        "a": np.linspace(0, 100, 1024),
        "b": np.linspace(0, 100, 1024),
        "c": np.linspace(0, 100, 1024)
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
```

### Usage

1. **Start Servers**:
   ```bash
   python RunNServer.py <number_of_servers>
   ```

2. **Run Genetic Algorithm**:
   ```bash
   python main.py <number_of_servers>
   ```

### Example Workflow

- Start 4 servers:
  ```bash
  python RunNServer.py 4
  ```

- Run the genetic algorithm using these servers:
  ```bash
  python main.py 4
  ```

### Outputs

1. **Optimized Solution**: Logs the best solution and its fitness score.
2. **Plots**: Visualizations saved to the `./cache/` directory.

This distributed setup allows GeneOpt to scale across multiple servers, making it ideal for resource-intensive optimization tasks.

## Configuration Parameters

| Parameter                     | Description                                                                                 |
|-------------------------------|---------------------------------------------------------------------------------------------|
| `objective_func`              | The fitness function to optimize.                                                           |
| `comparison_type`             | Either `maximize` or `minimize` to specify optimization goal.                               |
| `optimizer_name`              | Name for the optimizer instance.                                                            |
| `environment`                 | Dictionary defining the search space for variables.                                         |
| `seed`                        | Random seed for reproducibility.                                                            |
| `mutation_rate`               | Probability of mutation in the population.                                                  |
| `number_of_generations`       | Number of generations to evolve.                                                            |
| `number_of_population`        | Total size of the population.                                                               |
| `r_cross`                     | Probability of crossover between individuals.                                               |
| `start_number_of_population`  | Initial population size.                                                                    |
| `tournament_selection_number` | Number of candidates in tournament selection.                                               |
| `cache_type`                  | Choose `Ram` or `HardDisk` for caching.                                                     |
| `verbose`                     | Verbosity level for logging (-1 for no logging, 0 for minimal, up to 3 for maximum detail). |
| number_of_workers             | Number of worker servers (default is -1, meaning no workers are used).                      |

## Requirements

- Python 3.6 or higher
- NumPy
- Matplotlib
- Seaborn
- CatBoost (optional for feature analysis)

## License

This project is licensed under the MIT License. Feel free to use, modify, and contribute.

---

Happy optimizing with GeneOpt! 🚀
