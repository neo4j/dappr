import json
import os
from src.candidate_selection.benchmarking.benchmark_reader import BenchmarkReader, BenchmarkResult


class BenchmarkMigration:
    def __init__(self, old_file_name, new_file_name, dataset, algorithm_names) -> None:
        if old_file_name == new_file_name:
            raise "Old and new file name are the same"

        self.__old_file_name = old_file_name
        self.__new_file_name = os.path.dirname(os.path.abspath(__file__)) + "/" + new_file_name
        self.__benchmarks: list[BenchmarkResult] = BenchmarkReader(old_file_name).read()

        self.dataset = dataset
        self.algorithm_names = algorithm_names
        self.data = self.filter_data()

        data = None

    def sort(self):
        self.data = list(sorted(
            list(sorted(
                list(sorted( self.data, key=lambda x: x["parameters"]["h"] )),
                key=lambda x: x["method"],
            )),
        key=lambda x: x["dataset"]
    ))

    def add_field(self, field_name, value):
        for benchmark in self.data:
            benchmark[field_name] = value

    def add_sub_field(self, key, field_name, value):
        for benchmark in self.data:
            benchmark[key][field_name] = value

    def run_func_on_row(self, func):
        for benchmark in self.data:
            benchmark = func(benchmark)

    def write_to_file(self):
        self.sort()
        with open(self.__new_file_name, "a") as file:
            for benchmark in self.data:
                json_result = json.dumps(benchmark)
                file.write(json_result + "\n")
            file.write("#\n")

    def filter_data(self):
        data = self.__benchmarks
        if self.algorithm_names:
            data = list(filter(lambda x: x["method"] in self.algorithm_names, data))
        if self.dataset:
            data = list(filter(lambda x: x["dataset"] == self.dataset.name, data))
        data = list(sorted(data, key=lambda x: x["parameters"]["h"], reverse=True))
        
        return data
