import math
import queue
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
import random
import concurrent.futures
import threading

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from Util import read_json_file, write_json_file, Logger, to_datetime

DPI = 500


class GeneticAlgorithmComparisonType(Enum):
    maximize = "maximize"
    minimize = "minimize"


class GeneticAlgorithmCacheType(Enum):
    Ram = "Ram"
    HardDisk = "HardDisk"


class CachePath(Enum):
    MyOS = "./cache/"
    Colab = "/content/gdrive/MyDrive/"


class GeneticAlgorithmCache:
    def __init__(
            self,
            optimizer_name: str,
            genetic_algorithm_cache_type: GeneticAlgorithmCacheType,
            cache_path: CachePath
    ) -> None:
        self._optimizer_name = optimizer_name
        self._genetic_algorithm_cache_type = genetic_algorithm_cache_type

        if self._genetic_algorithm_cache_type == GeneticAlgorithmCacheType.HardDisk:
            self._cache_file_path_csv = f"{cache_path.value}{optimizer_name}.csv"
            self._cache_file_path_json = f"{cache_path.value}{optimizer_name}.json"
            try:
                self.ram_cache_score = pd.read_csv(self._cache_file_path_csv, dtype=np.float64)
                self.ram_cache_other = read_json_file(self._cache_file_path_json, {})
            except FileNotFoundError:
                self.ram_cache_score = pd.DataFrame()
                self.ram_cache_other = {}
        else:
            self.ram_cache_score = pd.DataFrame()
            self.ram_cache_other = {}

    def get_score(self, search_params: dict, default=None):
        """Search for a row based on search_params and return the last column value."""
        if self.ram_cache_score.empty:
            return default  # If cache is empty, no row exists

        query = np.ones(len(self.ram_cache_score), dtype=bool)
        for col, value in search_params.items():
            query &= np.isclose(self.ram_cache_score[col], value, atol=1e-8)

        match = self.ram_cache_score[query]
        return match.iloc[0, -1] if not match.empty else None

    def get_other(self, key: str, default=None):
        return self.ram_cache_other.get(key, default)

    def set_score(self, data: dict, score: float):
        """Add a new row to the cache and append it to the file if using HardDisk."""
        data["score"] = score
        new_data = pd.DataFrame([data])
        self.ram_cache_score = pd.concat([self.ram_cache_score, new_data], ignore_index=True).drop_duplicates()

        if self._genetic_algorithm_cache_type == GeneticAlgorithmCacheType.HardDisk:
            new_data.to_csv(self._cache_file_path_csv, mode='a',
                            header=not pd.io.common.file_exists(self._cache_file_path_csv),
                            index=False)

    def set_other(self, key: str, value):
        self.ram_cache_other[key] = value
        if self._genetic_algorithm_cache_type == GeneticAlgorithmCacheType.HardDisk:
            write_json_file(self._cache_file_path_json, self.ram_cache_other)


# Example usage:
if __name__ == "__main__":
    cache = GeneticAlgorithmCache(
        optimizer_name="example_optimizer",
        genetic_algorithm_cache_type=GeneticAlgorithmCacheType.HardDisk,
        cache_path=CachePath.MyOS
    )

    # Adding data to cache
    cache.set_score({"col0": 1.1111111111111, "col1": 2.2222222222222, "col2": 3.3333333333333}, score=55)
    cache.set_score({"col0": 4.4444444444444, "col1": 5.5555555555555, "col2": 6.6666666666666}, score=66)
    cache.set_other("test1", 1)
    cache.set_other("test2", 2)

    # Searching in cache
    print("Search result:",
          cache.get_score({"col0": 1.1111111111111, "col1": 2.2222222222222, "col2": 3.3333333333333}))
    print("Search result:",
          cache.get_score({"col0": 4.4444444444444, "col1": 5.5555555555555, "col2": 6.6666666666666}))
    print("Search result:",
          cache.get_score({"col0": 1.1111111111112, "col1": 2.2222222222222, "col2": 3.3333333333333}))
    print("other: ", cache.get_other("test1"))
    print("other: ", cache.get_other("test2"))
    print("other: ", cache.get_other("test3"))


class GeneticAlgorithm:
    def __init__(
            self,
            objective_func,
            start_number_of_population: int,
            optimizer_name: str,
            cache_type: GeneticAlgorithmCacheType = GeneticAlgorithmCacheType.Ram,
            cache_path: CachePath = CachePath.MyOS,
            number_of_generations: int = None,
            number_of_population: int = None,
            target_score: float = None,
            comparison_type: GeneticAlgorithmComparisonType = GeneticAlgorithmComparisonType.maximize,
            r_cross: float = 0.99,  # 0..1.0 => 0.6..1.0
            tournament_selection_number: int = 3,
            mutation_rate: float = 0.05,  # 0..1.0 => 0.001..0.1
            early_stopping_rounds: int = None,
            environment: dict = None,
            seed=0,
            verbose=0,
            number_of_workers=-1,
            is_plot=False,
            server_start_port=5000
    ) -> None:
        self.population = None
        self._objectives_df = None
        self._last_generation_with_new_best = None
        self._setup_random_instances(seed)
        self._cache = GeneticAlgorithmCache(
            optimizer_name=optimizer_name,
            genetic_algorithm_cache_type=cache_type,
            cache_path=cache_path
        )
        self.verbose = verbose
        self._objective_func = objective_func
        if environment is not None:
            e = {}
            for s_key in sorted(environment.keys()):
                e[s_key] = environment[s_key]
            environment = e
        if number_of_generations is None:
            number_of_generations = 100
            if self.verbose >= 0:
                Logger.log_m("auto number of generation :", number_of_generations)
        self._number_of_generations = number_of_generations
        if number_of_population is None:
            number_of_population = 100
            if self.verbose >= 0:
                Logger.log_m("auto number of population :", number_of_population)
        self._number_of_population = number_of_population
        if self._number_of_population % 2 == 1:
            self._number_of_population -= 1
        self._environment_num_prod = 1
        for key in environment:
            self._environment_num_prod *= len(environment[key])
        self._start_number_of_population = start_number_of_population
        if self._start_number_of_population % 2 == 1:
            self._start_number_of_population -= 1
        self._target_score = target_score
        self._r_cross = r_cross
        self._tournament_selection_number = tournament_selection_number
        self._mutation_rate = mutation_rate
        self._early_stopping_rounds = early_stopping_rounds
        self._comparison_type = comparison_type
        if comparison_type == GeneticAlgorithmComparisonType.maximize:
            self._comparison_func = self._maximize
        elif comparison_type == GeneticAlgorithmComparisonType.minimize:
            self._comparison_func = self._minimize
        else:
            raise Exception("unknown GeneticAlgorithmComparisonType", str(comparison_type))
        self._environment = environment
        self._init_environment_decode()
        self.start_datetime = to_datetime(time.time(), unit="s")
        self._number_of_workers = number_of_workers
        self._is_plot = is_plot
        self._objectives_df_datas = []
        self._current_generation = 0
        self._lock = threading.Lock()
        self.plot_index = 0
        self.server_start_port = server_start_port

    def _setup_random_instances(self, seed):
        self._python_random_generator = random.Random(seed)
        self._np_random_generator = np.random.default_rng(seed=seed)

    @staticmethod
    def _maximize(a, b):
        return a > b

    @staticmethod
    def _minimize(a, b):
        return a < b

    def _objective(self, chromosome):
        objective_input = self._decode(chromosome)
        scores = self._cache.get_score(objective_input)
        if scores is not None:
            if self.verbose >= 3:
                Logger.log_m("read cache ->", str(objective_input))
            if self._is_plot:
                data = []
                for key in objective_input:
                    data.append(objective_input[key])
                data.append(scores[str(objective_input)])
                data.append(self._current_generation)
                self._objectives_df_datas.append(data)
            return scores[str(objective_input)]
        if not objective_input:
            if self._comparison_type == GeneticAlgorithmComparisonType.maximize:
                self._cache.set_score(objective_input, -np.inf)
                return -np.inf
            elif self._comparison_type == GeneticAlgorithmComparisonType.minimize:
                self._cache.set_score(objective_input, np.inf)
                return np.inf
            else:
                self._cache.set_score(objective_input, 0)
                return 0
        if self.verbose >= 2:
            Logger.log_m("_" * 60)
            Logger.log_m(str(to_datetime(time.time(), unit="s") - self.start_datetime), "have been spent")
        this_count_of_run_objective = self._cache.get_other("count_of_run_objective", 0) + 1
        if len(objective_input) <= 10:
            if self.verbose >= 2:
                Logger.log_m("run objective", this_count_of_run_objective, objective_input)
        else:
            if self.verbose >= 2:
                Logger.log_m("run objective", this_count_of_run_objective)
        if isinstance(objective_input, dict):
            objective_result = self._objective_func(**objective_input)
        else:
            objective_result = self._objective_func(objective_input)
        self._cache.set_other("count_of_run_objective", this_count_of_run_objective)
        if self.verbose >= 1:
            Logger.log_m("objective result is :", objective_result)
        if self.verbose >= 2:
            Logger.log_m("_" * 60)
        self._cache.set_score(objective_input, objective_result)
        if self._is_plot:
            data = []
            for key in objective_input:
                data.append(objective_input[key])
            data.append(objective_result)
            data.append(self._current_generation)
            self._objectives_df_datas.append(data)
        return objective_result

    def _concurrent_objective(self, worker, task_queue):
        while not task_queue.empty():
            try:
                chromosome = None
                with self._lock:
                    if not task_queue.empty():
                        chromosome = task_queue.get_nowait()
                if chromosome is None:
                    break
                objective_input = self._decode(chromosome)
                with self._lock:
                    cache_result = self._cache.get_score(objective_input)
                    if cache_result is not None:
                        if self.verbose >= 3:
                            Logger.log_m("read cache ->", str(objective_input))
                        if self._is_plot:
                            data = []
                            for key in objective_input:
                                data.append(objective_input[key])
                            data.append(cache_result)
                            data.append(self._current_generation)
                            self._objectives_df_datas.append(data)
                        yield chromosome, cache_result
                        continue
                    else:
                        if not objective_input:
                            if self._comparison_type == GeneticAlgorithmComparisonType.maximize:
                                self._cache.set_score(objective_input, -np.inf)
                                yield chromosome, -np.inf
                            elif self._comparison_type == GeneticAlgorithmComparisonType.minimize:
                                self._cache.set_score(objective_input, np.inf)
                                yield chromosome, np.inf
                            else:
                                self._cache.set_score(objective_input, 0)
                                yield chromosome, 0
                            continue
                        if self.verbose >= 2:
                            Logger.log_m("_" * 60)
                            Logger.log_m(str(to_datetime(time.time(), unit="s") - self.start_datetime),
                                         "have been spent")
                        this_count_of_run_objective = self._cache.get_other("count_of_run_objective", 0) + 1
                        if len(objective_input) <= 10:
                            if self.verbose >= 2:
                                Logger.log_m("run objective", this_count_of_run_objective, objective_input)
                        else:
                            if self.verbose >= 2:
                                Logger.log_m("run objective", this_count_of_run_objective)
                objective_result = self._objective_func(**objective_input, port=worker['port'])
                with self._lock:
                    self._cache.set_other("count_of_run_objective",
                                          self._cache.get_other("count_of_run_objective", 0) + 1)
                    if self.verbose >= 1:
                        Logger.log_m(f"{worker['port']}> objective {this_count_of_run_objective} result is :",
                                     objective_result)
                    if self.verbose >= 2:
                        Logger.log_m("_" * 60)
                    self._cache.set_score(objective_input, objective_result)
                    if self._is_plot:
                        data = []
                        for key in objective_input:
                            data.append(objective_input[key])
                        data.append(objective_result)
                        data.append(self._current_generation)
                        self._objectives_df_datas.append(data)
                    yield chromosome, objective_result
            except KeyboardInterrupt as e:
                raise e
            except queue.Empty:
                break

    def _decode(self, bits):
        result_values = []
        start_index = 0
        for key in self._environment_param_n_bits:
            end_index = start_index + self._environment_param_n_bits[key]
            arr_to_decode = bits[start_index:end_index]
            start_index = end_index
            arr_to_decode = arr_to_decode[::-1]
            value = 0
            for index, bit in enumerate(arr_to_decode):
                value += bit * 2 ** index
            if value >= len(self._environment[key]):
                value = len(self._environment[key]) - 1
            result_values.append(value)
        return {key: self._environment[key][value] for key, value in zip(self._environment.keys(), result_values)}

    def _init_environment_decode(self):
        self._environment_param_n_bits = {}
        self._n_bits = 0
        for key in self._environment:
            param_env_size = len(self._environment[key])
            x = param_env_size
            n_bits = 0
            while x > 0:
                n_bits += 1
                x //= 2
            lower_or_equal_power_of_two = 0
            while 2 ** lower_or_equal_power_of_two < param_env_size:
                lower_or_equal_power_of_two += 1
            if param_env_size % (2 ** lower_or_equal_power_of_two) == 0:
                n_bits -= 1
            self._environment_param_n_bits[key] = n_bits
            self._n_bits += n_bits

    def _selection(
            self,
            population,
            scores
    ):
        try:
            selection_ix = self._np_random_generator.integers(len(population))
            for ix in self._np_random_generator.choice(len(population), self._tournament_selection_number,
                                                       replace=False):
                if self._comparison_func(scores[ix], scores[selection_ix]):
                    selection_ix = ix
            return population[selection_ix]
        except KeyboardInterrupt as e:
            raise e
        except:
            selection_ix = self._np_random_generator.integers(len(population))
            for ix in self._np_random_generator.choice(len(population), self._tournament_selection_number,
                                                       replace=True):
                if self._comparison_func(scores[ix], scores[selection_ix]):
                    selection_ix = ix
            return population[selection_ix]

    def _selection_population_scores(self, population_scores):
        try:
            selection_ix = self._np_random_generator.integers(len(population_scores))
            for ix in self._np_random_generator.choice(len(population_scores), self._tournament_selection_number,
                                                       replace=False):
                if self._comparison_func(population_scores[ix][1], population_scores[selection_ix][1]):
                    selection_ix = ix
            return population_scores[selection_ix][0]
        except KeyboardInterrupt as e:
            raise e
        except:
            selection_ix = self._np_random_generator.integers(len(population_scores))
            for ix in self._np_random_generator.choice(len(population_scores), self._tournament_selection_number,
                                                       replace=True):
                if self._comparison_func(population_scores[ix][1], population_scores[selection_ix][1]):
                    selection_ix = ix
            return population_scores[selection_ix][0]

    def _crossover(
            self,
            p1,
            p2
    ):
        c1, c2 = p1.copy(), p2.copy()
        if self._np_random_generator.random() < self._r_cross:
            start_index = 0
            for key in self._environment_param_n_bits:
                end_index = start_index + self._environment_param_n_bits[key]
                if self._np_random_generator.random() < 1.0 / len(self._environment_param_n_bits):
                    temp = c1[start_index:end_index]
                    c1[start_index:end_index] = c2[start_index:end_index]
                    c2[start_index:end_index] = temp
                start_index = end_index
        return [c1, c2]

    # def _mutation(
    #         self,
    #         bitstring
    # ):
    #     start_index = 0
    #     for key in self._environment_param_n_bits:
    #         end_index = start_index + self._environment_param_n_bits[key]
    #         if self._np_random_generator.random() < 1.0 / len(self._environment_param_n_bits):
    #             for i in range(start_index, end_index):
    #                 if self._np_random_generator.random() < 1.0 / self._environment_param_n_bits[key]:
    #                     bitstring[i] = 1 - bitstring[i]
    #         start_index = end_index
    #     return bitstring

    def _mutation(self, bitstring, adaptive_mutation_rate):
        start_index = 0
        for key in self._environment_param_n_bits:
            end_index = start_index + self._environment_param_n_bits[key]
            if self._np_random_generator.random() < adaptive_mutation_rate:
                for i in range(start_index, end_index):
                    if self._np_random_generator.random() < adaptive_mutation_rate:
                        bitstring[i] = 1 - bitstring[i]
            start_index = end_index
        return bitstring

    def _get_adaptive_mutation_rate(self, diversity):
        return self._mutation_rate / diversity if diversity != 0 else self._mutation_rate

    def _create_population(self, size):
        return self._np_random_generator.integers(0, 2, (size, self._n_bits))

    def start(self):
        if self._number_of_workers == -1:
            return self.normal_start()
        else:
            return self.concurrent_start()

    def normal_start(self):
        self.population = self._create_population(
            size=self._start_number_of_population
        )
        best_chromosome, best_chromosome_score = self.population[0], self._objective(self.population[0])
        self._last_generation_with_new_best = 0
        for self._current_generation in range(self._number_of_generations):
            if self._early_stopping_rounds is not None \
                    and self._early_stopping_rounds < (self._current_generation - self._last_generation_with_new_best):
                Logger.log_m("early stopping with", self._early_stopping_rounds, "rounds")
                return self.on_end(self._decode(best_chromosome), best_chromosome_score)
            if self.verbose >= 0:
                Logger.log_m("+=" * 40)
                Logger.log_m(">>>generation :", self._current_generation, "/", self._number_of_generations)
            chromosome_decode = self._decode(best_chromosome)
            if self.verbose >= 0:
                Logger.log_m("Best => ", chromosome_decode, f"With Score : {best_chromosome_score}")
            scores = []
            for chromosome in self.population:
                score = self._objective(chromosome)
                scores.append(score)
                if self._comparison_func(score, best_chromosome_score):
                    self._last_generation_with_new_best = self._current_generation
                    best_chromosome, best_chromosome_score = chromosome, score
                    if self.verbose >= 0:
                        Logger.log_m(">>> New Best <<<")
                if self._target_score is not None and (
                        score == self._target_score or
                        self._comparison_func(score, self._target_score)
                ):
                    Logger.log_m("early stopping with", best_chromosome_score, "score")
                    return self.on_end(self._decode(best_chromosome), best_chromosome_score)
            selected_parent = [self._selection(self.population, scores) for _ in range(self._number_of_population)]
            children = list()
            adaptive_mutation_rate = self._get_adaptive_mutation_rate(calculate_diversity(self.population))
            for i in range(0, self._number_of_population, 2):
                p1, p2 = selected_parent[i], selected_parent[i + 1]
                for c in self._crossover(p1, p2):
                    c = self._mutation(c, adaptive_mutation_rate)
                    # c = self._mutation(c)
                    children.append(c)
            self.population = np.array(children)
        return self.on_end(self._decode(best_chromosome), best_chromosome_score)

    def concurrent_start(self):
        self.population = self._create_population(size=self._start_number_of_population)
        workers_list = [{"port": self.server_start_port + index} for index in range(self._number_of_workers)]
        task_queue = queue.Queue()
        best_chromosome = None
        if self._comparison_type == GeneticAlgorithmComparisonType.maximize:
            best_chromosome_score = -np.inf
        else:
            best_chromosome_score = np.inf
        self._last_generation_with_new_best = 0
        for self._current_generation in range(self._number_of_generations):
            if self._early_stopping_rounds is not None and \
                    self._early_stopping_rounds < (self._current_generation - self._last_generation_with_new_best) and \
                    best_chromosome is not None:
                Logger.log_m("early stopping with", self._early_stopping_rounds, "rounds")
                return self.on_end(self._decode(best_chromosome), best_chromosome_score)
            if self.verbose >= 0:
                Logger.log_m("+=" * 40)
                Logger.log_m(">>>generation :", self._current_generation, "/", self._number_of_generations)
            if best_chromosome is not None:
                chromosome_decode = self._decode(best_chromosome)
            else:
                chromosome_decode = None
            if self.verbose >= 0:
                Logger.log_m("Best => ", chromosome_decode, f"With Score : {best_chromosome_score}")
            for chromosome in self.population:
                task_queue.put(chromosome)
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._number_of_workers) as executor:
                futures = [executor.submit(collect_results, self._concurrent_objective(worker, task_queue)) for worker
                           in workers_list]
                population_scores = []
                for future in concurrent.futures.as_completed(futures):
                    population_scores.extend(future.result())

            # Sort the list by the score of each tuple (index 1)
            population_scores = sorted(population_scores, key=lambda x: x[1])

            # Shuffle the sorted list to make selection valid
            self._np_random_generator.shuffle(population_scores)

            for chromosome, score in population_scores:
                if self._comparison_func(score, best_chromosome_score):
                    self._last_generation_with_new_best = self._current_generation
                    best_chromosome, best_chromosome_score = chromosome, score
                    if self.verbose >= 0:
                        Logger.log_m(">>> New Best <<<")
                if self._target_score is not None and (
                        score == self._target_score or
                        self._comparison_func(score, self._target_score)
                ):
                    Logger.log_m("early stopping with", best_chromosome_score, "score")
                    return self.on_end(self._decode(best_chromosome), best_chromosome_score)
            selected_parent = [self._selection_population_scores(population_scores) for _ in
                               range(self._number_of_population)]
            children = list()
            adaptive_mutation_rate = self._get_adaptive_mutation_rate(calculate_diversity(self.population))
            for i in range(0, self._number_of_population, 2):
                p1, p2 = selected_parent[i], selected_parent[i + 1]
                for c in self._crossover(p1, p2):
                    c = self._mutation(c, adaptive_mutation_rate)
                    # c = self._mutation(c)
                    children.append(c)
            self.population = np.array(children)
        return self.on_end(str(self._decode(best_chromosome)), best_chromosome_score)

    def on_end(self, *result):
        self._cache.set_other(
            "best",
            {
                "parameters": result[0],
                "score": result[1]
            }
        )
        if self._is_plot:
            columns = list(self._environment.keys())
            columns.append("score")
            columns.append("generation")
            self._objectives_df = pd.DataFrame(self._objectives_df_datas, columns=columns)
            self._objectives_df.to_csv(f"./cache/{self._cache._optimizer_name}.csv", index=False)
        if not self.verbose >= 0:
            return result
        count_of_run_objective = self._cache.get_other("count_of_run_objective", 0)
        Logger.log_m("@" * 80)
        Logger.log_m("Result of Genetic Algorithm :")
        Logger.log_m("last generation with new best is :", self._last_generation_with_new_best)
        Logger.log_m("total objective run is :", count_of_run_objective)
        Logger.log_m("total spent time is :", to_datetime(time.time(), unit="s") - self.start_datetime)
        Logger.log_m(
            "time per objetive is :",
            (to_datetime(time.time(), unit="s") - self.start_datetime) / count_of_run_objective
        )
        log_ratio = math.log10(self._environment_num_prod) - math.log10(count_of_run_objective)
        Logger.log_m(f"10^{log_ratio:.2f} times faster than Blind Algorithm")
        Logger.log_m("@" * 80)
        return result

    def plot(self):
        if not self._is_plot:
            raise Exception("please set is_plot pram to True in order to display plots")
        if self._objectives_df is None:
            self._objectives_df = pd.read_csv(f"./cache/{self._cache._optimizer_name}.csv")
        try:
            self.plot_best_score()
        except:
            pass
        try:
            self.plot_parameter_values_for_the_best_scores()
        except:
            pass
        try:
            self.plot_population_diversity()
        except:
            pass
        try:
            self.plot_gene_frequencies()
        except:
            pass
        try:
            self.feature_plots()
        except:
            pass

    def plot_best_score(self):
        if self._comparison_type == GeneticAlgorithmComparisonType.maximize:
            best_scores = self._objectives_df.groupby('generation')['score'].max()
        elif self._comparison_type == GeneticAlgorithmComparisonType.minimize:
            best_scores = self._objectives_df.groupby('generation')['score'].min()
        plt.plot(best_scores.index, best_scores.values)
        plt.title('Best Score over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Best Score')
        plt.tight_layout()
        plt.savefig(f"./cache/{self.plot_index}.png", dpi=DPI)
        plt.close()
        self.plot_index += 1

    def plot_parameter_values_for_the_best_scores(self):
        parameters = list(self._objectives_df.columns[:-2])
        for param in parameters:
            if self._comparison_type == GeneticAlgorithmComparisonType.maximize:
                best_params = [
                    self._objectives_df.loc[
                        self._objectives_df[
                            self._objectives_df['generation'] == generation
                            ]["score"].idxmax()
                    ][param]
                    for generation in range(self._number_of_generations)
                ]
            elif self._comparison_type == GeneticAlgorithmComparisonType.minimize:
                best_params = [
                    self._objectives_df.loc[
                        self._objectives_df[
                            self._objectives_df['generation'] == generation
                            ]["score"].idxmin()
                    ][param]
                    for generation in range(self._number_of_generations)
                ]
            plt.plot(best_params)
            plt.title(f'Best {param} over Generations')
            plt.xlabel('Generation')
            plt.ylabel(f'Best {param}')
            plt.tight_layout()
            plt.savefig(f"./cache/{self.plot_index}.png", dpi=DPI)
            plt.close()
            self.plot_index += 1

    def plot_population_diversity(self):
        diversities = [
            calculate_diversity(self._objectives_df[self._objectives_df['generation'] == generation].values)
            for generation in range(self._number_of_generations)
        ]
        plt.plot(diversities)
        plt.title('Population Diversity per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Diversity')
        plt.tight_layout()
        plt.savefig(f"./cache/{self.plot_index}.png", dpi=DPI)
        plt.close()
        self.plot_index += 1

    def plot_gene_frequencies(self):
        gene_frequencies = self._objectives_df.groupby('generation')[
            self._objectives_df.columns[:-2]
        ].apply(lambda x: x.mean())
        gene_frequencies = (gene_frequencies - gene_frequencies.min()) / (
                gene_frequencies.max() - gene_frequencies.min())
        sns.heatmap(gene_frequencies, cmap='viridis')
        plt.title('Gene Frequencies per Generation')
        plt.xlabel('Gene')
        plt.ylabel('Generation')
        plt.tight_layout()
        plt.savefig(f"./cache/{self.plot_index}.png", dpi=DPI)
        plt.close()
        self.plot_index += 1

    def feature_plots(self):
        df = self._objectives_df.drop(self._objectives_df.columns[-1:], axis=1)
        features = self._objectives_df.columns[:-2]
        target = self._objectives_df.columns[-2]
        df.replace([-np.inf], np.NaN, inplace=True)
        df.dropna(axis=0, inplace=True)

        df.columns = df.columns.map(lambda s: s.replace("_", " "))
        temp = []
        for f in features:
            temp.append(f.replace("_", " "))
        features = temp

        plt.title(f"pairwise correlation with {target}")
        df.corrwith(df[target], method="spearman").sort_values().plot(kind="barh")
        plt.tight_layout()
        plt.savefig(f"./cache/{self.plot_index}.png", dpi=DPI)
        plt.close()
        self.plot_index += 1

        sns.heatmap(df.corr(method="spearman"), annot=True, cmap="RdBu")
        plt.tight_layout()
        plt.savefig(f"./cache/{self.plot_index}.png", dpi=DPI)
        plt.close()
        self.plot_index += 1

        model = CatBoostRegressor(
            verbose=1000,
            loss_function='RMSE'
        )
        X = df[features]
        y = df[target]
        model.fit(
            X=X,
            y=y
        )
        y_pred = model.predict(X)
        plt.scatter(y, y_pred, label='Predictions')
        min_val = min(min(y), min(y_pred))
        max_val = max(max(y), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')
        plt.title("Predicted vs Actual")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./cache/{self.plot_index}.png", dpi=DPI)
        plt.close()
        self.plot_index += 1

        r_squared = r2_score(y.values.reshape(-1, 1), y_pred)
        print("R squared value is:", r_squared)

        importance = model.feature_importances_
        feat_imp = pd.Series(importance, index=features).sort_values()

        patches, _ = plt.pie(feat_imp)
        labels = [f'{j:0>5.2f}% - {i}' for i, j in zip(feat_imp.index, 100 * feat_imp / feat_imp.sum())]
        plt.legend(patches, labels, loc="center left", bbox_to_anchor=(-0.4, 0.5))
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(f"./cache/{self.plot_index}.png", dpi=DPI)
        plt.close()
        self.plot_index += 1

        for f in features:
            plt.title(f"hist plot {f}")
            sns.histplot(df[f], bins=20)
            plt.tight_layout()
            plt.savefig(f"./cache/{self.plot_index}.png", dpi=DPI)
            plt.close()
            self.plot_index += 1

        for f in features:
            sns.jointplot(x=df[f], y=df[target], kind='kde', color="red", fill=True)
            plt.tight_layout()
            plt.savefig(f"./cache/{self.plot_index}.png", dpi=DPI)
            plt.close()
            self.plot_index += 1

        for f in features:
            plt.scatter(x=df[[f]], y=df[target])
            slope, intercept = np.polyfit(df[f].values, df[target].values, 1)
            x_range = np.linspace(df[f].min(), df[f].max(), 100)
            y_range = slope * x_range + intercept
            plt.plot(x_range, y_range, color='red')
            plt.title(f"{f} regression line plot")
            plt.xlabel(f)
            plt.ylabel(target)
            plt.tight_layout()
            plt.savefig(f"./cache/{self.plot_index}.png", dpi=DPI)
            plt.close()
            self.plot_index += 1

        for ff1 in features:
            for ff2 in features:
                if ff1 == ff2 or ff1.lower() > ff2.lower():
                    continue

                sns.scatterplot(x=df[ff1], y=df[ff2], hue=df[target])
                plt.title(f"{ff1} & {ff2} & {target} scatterplot", size=20)
                plt.tight_layout()
                plt.savefig(f"./cache/{self.plot_index}.png", dpi=DPI)
                plt.close()
                self.plot_index += 1


def collect_results(generator):
    return list(generator)


def calculate_diversity(population: np.ndarray):
    n = len(population)
    if n <= 1:
        return 1

    # Calculate the Hamming distances using broadcasting
    diff_matrix = np.triu(np.sum(population[:, None, :] != population[None, :, :], axis=2), 1)
    total_distance = np.sum(diff_matrix)

    average_distance = total_distance / (n * (n - 1) / 2)
    return average_distance
