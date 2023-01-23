import json
import os
from typing import TypedDict

class Statistics(TypedDict):
    recall: float
    precision: float
    scan_rate: float
    scans_per_found_edge: float
    num_found_edges: int
    num_predicted_edges: int
    num_scanned_w_LP: int


class Parameters(TypedDict):
    K: int
    max_steps: int
    alpha: float
    parallel: bool
    verbose: bool
    time: float
    h: int


class BenchmarkResult(TypedDict):
    date: str
    dataset: str
    method: str
    statistics: Statistics
    parameters: Parameters

class BenchmarkReader:

    def __init__(self, file_name: str = "benchmarks.txt") -> None:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        self.__benchmarks_path = os.path.join(root_dir, file_name)

    def read(self) -> list[BenchmarkResult]:
        if not os.path.exists(self.__benchmarks_path):
            raise FileNotFoundError(self.__benchmarks_path)
        
        benchmarks = list()
        for line in open(self.__benchmarks_path, "r"):
            if line.startswith("#"):
                continue

            benchmarks.append(json.loads(line))
        
        return benchmarks
