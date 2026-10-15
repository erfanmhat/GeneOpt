"""Run the genetic algorithm against N objective servers: python client.py N [BASE_PORT]

Each executor thread borrows a free port, calls the server and returns the port,
so every server evaluates one genome at a time.
"""

import json
import logging
import queue
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from geneopt import GeneticAlgorithm


class ObjectiveError(RuntimeError):
    """The evaluation failed for good (bad request, or retries exhausted)."""


def remote_score(port: int, params: dict, retries: int = 3, timeout: float = 1800) -> float:
    query = urllib.parse.urlencode(
        {k: json.dumps(v.item() if isinstance(v, np.generic) else v) for k, v in params.items()}
    )
    url = f"http://127.0.0.1:{port}/run_objective?{query}"
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return float(json.load(response)["score"])  # 0.0 is a valid score
        except urllib.error.HTTPError as e:
            if e.code < 500:  # our request is wrong: retrying cannot help
                raise ObjectiveError(f"{e.code} for {params}: {e.read()[:200]!r}") from e
            error = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            error = e
        if attempt < retries:
            time.sleep(min(2**attempt, 30))
    raise ObjectiveError(f"gave up after {retries + 1} attempts for {params}") from error


def main(ports, generations=100, population_size=100):
    free_ports = queue.Queue()
    for port in ports:
        free_ports.put(port)

    def objective(x1, x2):
        port = free_ports.get()
        try:
            return remote_score(port, {"x1": x1, "x2": x2})
        finally:
            free_ports.put(port)

    environment = {
        "x1": np.linspace(-10, 10, 2**12),
        "x2": np.linspace(-10, 10, 2**12),
    }
    with ThreadPoolExecutor(len(ports)) as executor:
        ga = GeneticAlgorithm(
            objective,
            environment,
            optimizer_name="holder_table_http",
            maximize=False,
            population_size=population_size,
            generations=generations,
            executor=executor,
            seed=42,
        )
        result = ga.start()
    print(f"{result.params} = {result.score}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n, base = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    main(ports=[base + i for i in range(n)])
