import json
import sys
import os
import datetime
from time import time
from candidate_selection.cadidate_selection_algorithm import EdgeList, CandidateSelectionAlgorithm
from datetime import date
import networkx as nx

class Benchmark:
    predicted_links: set[(int, int)]
    positive_test_edges: set[(int, int)]
    G_test: nx.Graph
    run_time: int
    
    results_path = os.path.dirname(os.path.abspath(__file__)) + "/benchmarks.txt"

    def __init__(self, algorithm: CandidateSelectionAlgorithm, G: nx.Graph, G_test: nx.Graph, h: int, positive_test_edges: EdgeList, dataset_name: str) -> None:
        self.algorithm = algorithm
        self.G = G
        self.G_test = G_test
        self.h = h
        self.positive_test_edges = self.__edge_list_to_set(positive_test_edges)
        self.dataset_name = dataset_name

    def run(self) -> EdgeList:
        t0 = time() # start timer
        predicted_links = self.algorithm.run()
        t1 = time() # end timer

        self.run_time = t1 - t0
        self.predicted_links = self.__edge_list_to_set(predicted_links)
        
        return self.predicted_links

    def write_results_to_file(self, resultsPath: str = None):
        if resultsPath:
            self.results_path = os.path.dirname(os.path.abspath(__file__)) + "/" + resultsPath
            
        with open(self.results_path, "a") as f:
            statistics = self.get_metrics()
            parameters = self.algorithm.get_input_params()
            parameters["h"] = self.h
            parameters["nodes"] = self.G.number_of_nodes()
            parameters["edges"] = self.G.number_of_edges()
            parameters["hidden_edges"] = len(self.positive_test_edges)
            method_name = self.algorithm.get_name()
            result = {
                "date": str(datetime.datetime.now()),
                "method": method_name,
                "dataset": self.dataset_name,
                "h": self.h,
                "statistics": statistics,
                "parameters": parameters
            }

            json_result = json.dumps(result)
            f.write(json_result + "\n")
            # print("Wrote results to file", self.results_path)1

    def __edge_list_to_set(self, predicted_links: EdgeList):
        """Set the predicted links as map for efficiency"""
        edge_set = set()

        for v, u in predicted_links:
            if u <= v:
                edge_set.add((v, u))
            else:
                edge_set.add((u, v))
        
        return edge_set

    def get_metrics(self):
        """How many of all hidden edges did we find?"""
        found_edges = len(self.predicted_links.intersection(self.positive_test_edges))
                
        return dict({
            "recall": found_edges / len(self.positive_test_edges) if len(self.positive_test_edges) > 0 else 0,
            "precision": found_edges / len(self.predicted_links) if len(self.predicted_links) > 0 else 0,
            "scan_rate": self.algorithm.get_n_scanned() / self.G_test.number_of_nodes()**2 if self.G_test.number_of_nodes() > 0 else 0,
            "scans_per_found_edge": self.algorithm.get_n_scanned() / found_edges if found_edges > 0 else 0,
            
            "num_found_edges": found_edges,
            "num_predicted_edges": len(self.predicted_links),
            "num_scanned_w_LP": self.algorithm.get_n_scanned(),
            "time": self.run_time
        })