from copy import deepcopy
import sys
from ..cadidate_selection_algorithm import EdgeList, CandidateSelectionAlgorithm
sys.path.insert(1, '../')

from graph_utils import sample_random_nodes
import networkx as nx
from .types import KnnDict, KnnDict, KnnDict_lowest
from typing import Callable
import numpy as np
from time import time

class KnnSimple(CandidateSelectionAlgorithm):
    """Based on the paper https://www.cs.princeton.edu/cass/papers/www11.pdf"""

    __G: nx.Graph
    __K: int
    __delta: float
    __knn: KnnDict
    __knn_lowest_node: KnnDict_lowest
    __n_scanned: int
    
    def __init__(self, G, K, score_links: Callable[[int, int], float], clf_best_accuracy: float, p_random_neighbors=0.9, max_iterations=20, delta=0.01, verbose=True):
        self.input_params = locals()
        self.exclude_params = ["self", "G", "score_links"]
        
        self.__G = G
        self.__K = K
        self.__score_fn = score_links
        self.clf_best_accuracy = clf_best_accuracy
        self.__delta = delta
        self.__verbose = verbose
        self.__knn = dict()
        self.__knn_lowest_node = dict()
        self.__p_random_neighbors = p_random_neighbors
        self.__max_iterations = max_iterations
        self.__n_scanned = 0
        

        self.all_good_pairs = set()

        if K**2 > G.number_of_nodes():
            print('WARNING: k^2 > v. Running KNN wll be slower than running exhaustive search.')
        
    def set_seed(self, seed):
        np.random.seed(seed)

    def get_predicted_links(self) -> EdgeList:
        """Returns a list of tuples (v, u) where v is a node in the graph and u is a predicted link to v."""
        predicted_links = []
        for v in self.__knn:
            for u in self.__knn[v]:
                predicted_links.append((v, u))
                self.all_good_pairs.add((v, u))
        return predicted_links
        return list(self.all_good_pairs) # Return all links we evaluated to have > 0.5 score

    def get_n_scanned(self) -> int:
        return self.__n_scanned

    def run(self) -> EdgeList:
        self.__initialize()
        for i in range(self.__max_iterations):
            if self.__verbose: print(f"Running iteration {i}")
            start = time()
            should_terminate = self.__iteration()
            if self.__verbose: print(f"Iteration took {time() - start} seconds.")
            if should_terminate: break
        
        predicted_links = self.get_predicted_links()
        return predicted_links
        
        
    def __initialize(self):
        self.__n_scanned = 0

        for v in self.__G.nodes():
            neighbors = list(self.__G.neighbors(v))

            random_neighbors = sample_random_nodes(neighbors, int(self.__K * (1 - self.__p_random_neighbors)))
            num_random_nodes = self.__K - len(random_neighbors)

            random_nodes = sample_random_nodes(self.__G.nodes(), num_random_nodes)
            self.__knn[v] = dict()
            
            for neighbor in random_neighbors:
                self.__knn[v][neighbor] = (0, True) # We want neighbors to be replaced asap, so set score to 0.

            for node in random_nodes:
                score = self.__score_fn([(v, node)])[0][1]
                self.__knn[v][node] = (score, True)

            self.__knn_lowest_node[v] = ((random_nodes + random_neighbors)[0], 0) # Init any node as lowest node. Will not matter later.

    def __iteration(self):
        new_knn = deepcopy(self.__knn) # Should we really copy the entire dict, or just per v?
        new_knn_lowest_node = deepcopy(self.__knn_lowest_node)

        c = 0

        # Time complexity: |V|
        for v in self.__knn:
            
            # Time complexity: K
            # Iterate over v's top_list best candidates
            for u1 in self.__knn[v]:
                # Time complexity: K
                # Iterate over the second degree best candidates and add the best ones.
                pairs_v_u2 = map(lambda u2: (v, u2), self.__knn[u1])
                scores_v_u2 = self.__score_fn(pairs_v_u2)

                for idx, u2 in enumerate(self.__knn[u1]):
                    if u2 in new_knn[v] or self.__G.has_edge(v, u2): # Don't add existing KNN_neighbors or real edges.
                        continue

                    self.__n_scanned += 1
                        
                    score_v_u2 = scores_v_u2[idx][1]
                    lowest_node_v, lowest_score_v = new_knn_lowest_node[v]
                    if score_v_u2 > 0.5:
                        self.all_good_pairs.add((v, u2))

                    if score_v_u2 > lowest_score_v:
                        #print(f"lowest score for {v} was {lowest_score_v}, but {score_v_u2} was found for {u2}")
                        del new_knn[v][lowest_node_v]
                        new_knn[v][u2] = (score_v_u2, True)
                        c += self.__update_knn_lowest_node(new_knn, new_knn_lowest_node, v)

            # k_nearest_neighbors[v] = list(filter(lambda x: x["score"] > 0.4, k_nearest_neighbors[v]))
        self.__knn = new_knn
        self.__knn_lowest_node = new_knn_lowest_node

        should_terminate = c < self.__delta * self.__G.number_of_nodes() * self.__K
        if self.__verbose: print(f"Number of bucket alterations: {c}, Termination Threshold: {self.__delta * self.__G.number_of_nodes() * self.__K}")
        return should_terminate
    
    def __update_knn_lowest_node(self, new_knn: KnnDict, knn_lowest_node: KnnDict_lowest, v: int):
        """Helper method for updating the lowest scored node in self.__knn_lowest_node[v]"""
        min_score = 1
        min_node = None

        for u in new_knn[v]:
            score = new_knn[v][u][0]
            if score < min_score:
                min_score = score
                min_node = u

        if min_node != None: 
            knn_lowest_node[v] = (min_node, min_score)
            return True
        else:
            return False