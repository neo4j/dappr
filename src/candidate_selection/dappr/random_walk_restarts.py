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
from typing import Callable
import networkx as nx
from tqdm import tqdm
from queue import PriorityQueue
import numpy as np
from random import random

# Running pagerank on subgraph from v with max size K
def rwr_func(inputs: tuple[nx.Graph, int, int, int, float, bool]):
    G, K, v, max_steps, alpha, bipartite = inputs

    visited_counts = defaultdict(lambda: 0)
    current_node = v
    if len(list(G.neighbors(v))) == 0:
        return v, []

    for _ in range(max_steps):
        next_node = np.random.choice(list(G.neighbors(current_node)))
        visited_counts[next_node] += 1
        current_node = next_node
        if random() < alpha:
            current_node = v
    
    visited_counts[v] = 0
    visited_counts = sorted(visited_counts.items(), key=lambda x: x[1], reverse=True)

    top_k_candidates = []
    for u, _ in visited_counts:
        if G.has_edge(v, u): continue
        if bipartite:
            if G.nodes[v]["bipartite"] == G.nodes[u]["bipartite"]: continue
        top_k_candidates.append((v, u))
        if len(top_k_candidates) >= K: break

    return v, top_k_candidates

class RandomWalkRestarts(CandidateSelectionAlgorithm):

    def __init__(self, G: nx.Graph, K: int, max_steps: int=1000, alpha: float=0.1, parallel=True, verbose=True, bipartite=False):
        self.input_params = locals()
        self.exclude_params = ["self", "G"]

        self.__verbose = verbose
        self.__parallel = parallel
        self.__G = G
        self.__K = K
        self.__max_steps = max_steps
        self.__alpha = alpha
        self.__bipartite = bipartite
        self.__candidates = defaultdict(list)
        

    def get_n_scanned(self) -> int:
        return len(self.get_unique_predicted_links())

    def get_predicted_links(self) -> EdgeList:
        """Returns a list of tuples (v, u) where v is a node in the graph and u is a predicted link to v."""
        links = list()
        for candidates in self.__candidates.values():
            links.extend(candidates)
        return links
        
    def run(self) -> EdgeList:
        if self.__parallel:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            res = list(tqdm( # tqdm for progress bar
                pool.imap_unordered(rwr_func, [(self.__G, self.__K, v, self.__max_steps, self.__alpha, self.__bipartite) for v in self.__G.nodes()], chunksize=200),
                total=self.__G.number_of_nodes()
            ))

            pool.close()
            pool.join()

        else:
            res = list(tqdm( # tqdm for progress bar
                map(rwr_func, [(self.__G, self.__K, v, self.__max_steps, self.__alpha, self.__bipartite) for v in self.__G.nodes()]),
                total=self.__G.number_of_nodes()
            ))

        for v, top_k_candidates in res:
            self.__candidates[v] = top_k_candidates

        return self.get_predicted_links()

    

    