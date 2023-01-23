import bisect
from functools import partial
from os import cpu_count
from queue import Queue
import sys
from itertools import chain

from ..cadidate_selection_algorithm import EdgeList, CandidateSelectionAlgorithm
sys.path.insert(1, '../')

from joblib import Parallel, delayed
import multiprocessing
from collections import defaultdict
from typing import Callable, Tuple
import networkx as nx
from tqdm import tqdm
from queue import PriorityQueue
import numpy as np
from random import random
import math

def rwr_func(inputs: tuple[nx.Graph, dict, int, int, float, float]):
    
    G, visit_map, q, max_steps, alpha, bayesian_factor = inputs
    k = int(G.degree(q) * bayesian_factor)
    visited_counts = defaultdict(lambda: 0)

    current_node = q
    if len(list(G.neighbors(q))) == 0:
        return q, []

    deleted_edges = []
    # Utilize already build RWR's on immediate neighbors
    for neighbor in G.neighbors(q):
        if neighbor in visit_map:
            steps_to_remove = (max_steps / len(list(G.neighbors(q))))
            max_steps -= int(steps_to_remove) # Mby not ceil

            deleted_edges.append((q, neighbor))
            G.remove_edge(q, neighbor)

            for key, rwr_prob in visit_map[neighbor][:k]:
                visited_counts[key] += rwr_prob * steps_to_remove
            
            break

    counted_steps = 0

    for _ in range(max_steps):
        next_node = np.random.choice(list(G.neighbors(current_node)))
        current_node = next_node

        if random() < alpha:
            current_node = q
        else: 
            visited_counts[current_node] += 1 # Only add visits on non-alpha steps
            counted_steps += 1
    
    visited_probabilities = list(map(lambda x: (x[0], x[1]/counted_steps), sorted(visited_counts.items(), key=lambda x: x[1], reverse=True)))
    visit_map[q] = visited_probabilities # Update visit map in parallel!

    G.add_edges_from(deleted_edges)


# Running pagerank on subgraph from v with max size K

class RandomWalkRestartsTrash(CandidateSelectionAlgorithm):

    def __init__(self, G: nx.Graph, k: int, max_steps: int=1000, alpha: float=0.1, parallel=True, verbose=True, bipartite=False):
        self.input_params = locals()
        self.exclude_params = ["self", "G"]

        self.__verbose = verbose
        self.__parallel = parallel
        self.__G = G
        self.__k = k
        self.__bipartite = bipartite
        self.__avg_edges_to_predict_per_node = (k / G.number_of_nodes()) / 2
        self.__avg_degree = G.number_of_edges() / G.number_of_nodes()
        self.__max_steps = max_steps
        self.__alpha = alpha
        self.__visit_map = multiprocessing.Manager().dict() # Make a shared dict
        self.__candidates = defaultdict(list)
        self.__global_pool: list[((int, int), float)] = list()
        self.__bayesian_factor = self.__avg_edges_to_predict_per_node / self.__avg_degree
        

        self.__average_path_length = math.log(0.5,alpha)

    def get_n_scanned(self) -> int:
        return len(self.get_unique_predicted_links())

    def get_predicted_links(self) -> EdgeList:
        """Returns a list of tuples (v, u) where v is a node in the graph and u is a predicted link to v."""
        links = list()

        for q, candidates in self.__visit_map.items():
            k = int(self.get_k(q))
            top_k = []

            for v, prob in candidates:
                if self.__G.has_edge(q, v): continue
                if self.__bipartite:
                    if self.__G.nodes[q]["bipartite"] == self.__G.nodes[v]["bipartite"]: continue
                    
                if len(top_k) >= k: 
                    links.extend(top_k)
                    break
                else:
                    top_k.append((q, v))

        # if len(links) < self.__k:
        #     self.__global_pool.sort(key=lambda x: x[1], reverse=True)
        #     links.extend(map(lambda x: x[0], self.__global_pool[:self.__k - len(links)]))

        return links
        
    def run(self) -> EdgeList:
        if self.__parallel:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            tqdm( # tqdm for progress bar
                pool.imap_unordered(rwr_func, [(self.__G, self.__visit_map, q, self.__max_steps, self.__alpha, self.__bayesian_factor) for q in self.__G.nodes()], chunksize=200),
                total=self.__G.number_of_nodes()
            )

            pool.close()
            pool.join()
        else:
            pass    


        return self.get_predicted_links()

    
    def get_k(self, node):
        return self.__G.degree(node) * self.__bayesian_factor