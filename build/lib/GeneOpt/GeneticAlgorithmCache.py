import os

import numpy as np
import pandas as pd

from GeneOpt.CachePath import CachePath
from GeneOpt.GeneticAlgorithmCacheType import GeneticAlgorithmCacheType
from GeneOpt.Util import write_json_file, read_json_file, is_folder_exist


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
        if not is_folder_exist("./cache"):
            os.makedirs("./cache")

    def get_score(self, search_params: dict, default=None):
        """Search for a row based on search_params and return the last column value."""
        if self.ram_cache_score.empty:
            return default  # If cache is empty, no row exists

        filtered_df = self.ram_cache_score

        for col, value in search_params.items():
            # Check if the column exists in the DataFrame
            if col not in filtered_df.columns:
                return default  # If column is missing, return default immediately

            # Filter the DataFrame based on the current column
            filtered_df = filtered_df[np.isclose(filtered_df[col].to_numpy(), value, atol=1e-8)]

            # If no rows remain, return default
            if filtered_df.empty:
                return default

            # If only one row remains, perform a row-wise comparison
            if len(filtered_df) == 1:
                row = filtered_df.iloc[0]
                if all(np.isclose(row[k], v, atol=1e-8) for k, v in search_params.items() if k in row):
                    return row["score"]
                else:
                    return default

        # Return the score of the first matching row
        return filtered_df.iloc[0]["score"] if not filtered_df.empty else default

    def get_other(self, key: str, default=None):
        return self.ram_cache_other.get(key, default)

    def set_score(self, data: dict, score: float, generation: int):
        """Add a new row to the cache and append it to the file if using HardDisk."""
        data["score"] = score
        data["generation"] = generation
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
    cache.set_score(
        {"col0": 1.1111111111111, "col1": 2.2222222222222, "col2": 3.3333333333333},
        score=55,
        generation=1)
    cache.set_score(
        {"col0": 4.4444444444444, "col1": 5.5555555555555, "col2": 6.6666666666666},
        score=66,
        generation=2
    )
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
