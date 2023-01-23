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

# Running pagerank on subgraph from v with max size K
def rwr_func(inputs: tuple[nx.Graph, int, int, int, float, bool]):
    G, minimum_per_node, v, max_steps, alpha, bipartite = inputs
    visited_counts = defaultdict(lambda: 0)
    current_node = v
    if len(list(G.neighbors(v))) == 0:
        return v, [], []

    for _ in range(max_steps):
        next_node = np.random.choice(list(G.neighbors(current_node)))
        visited_counts[next_node] += 1
        current_node = next_node
        if random() < alpha:
            current_node = v
    
    visited_counts[v] = 0
    visited_counts = sorted(visited_counts.items(), key=lambda x: x[1], reverse=True)

    top_k_candidates = []
    remaining = []
    for u, score in visited_counts:
        if G.has_edge(v, u): continue
        if bipartite:
            if G.nodes[v]["bipartite"] == G.nodes[u]["bipartite"]: continue
        if len(top_k_candidates) < minimum_per_node:
            top_k_candidates.append((v, u))
        else:
            remaining.append(((v, u), score))

        if len(remaining) >= minimum_per_node * 2: break

    return v, top_k_candidates, remaining

class RandomWalkRestartsGlobalPool(CandidateSelectionAlgorithm):
    """An implementation of RW-Restarts with global pool of candidates. We select 50% of the pairs in a local pool and 50% in a global pool."""

    def __init__(self, G: nx.Graph, k: int, max_steps: int=1000, alpha: float=0.1, parallel=True, verbose=True, bipartite=False):
        self.input_params = locals()
        self.exclude_params = ["self", "G"]

        self.__verbose = verbose
        self.__parallel = parallel
        self.__bipartite = bipartite
        self.__G = G
        self.__k = k
        self.__minimum_per_node = (k / G.number_of_nodes()) / 2 # Add half the neccessary candidates for each node
        self.__max_steps = max_steps
        self.__alpha = alpha
        self.__global_pool: list[ ((int, int), int) ] = list()
        self.candidates = list()
        

    def get_n_scanned(self) -> int:
        return len(self.get_unique_predicted_links())

    def get_predicted_links(self) -> EdgeList:
        """Returns a list of tuples (v, u) where v is a node in the graph and u is a predicted link to v."""
        return self.candidates
        
    def run(self) -> EdgeList:
        if self.__parallel:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            res = list(tqdm( # tqdm for progress bar
                pool.imap_unordered(rwr_func, [(self.__G, self.__minimum_per_node, v, self.__max_steps, self.__alpha, self.__bipartite) for v in self.__G.nodes()], chunksize=200),
                total=self.__G.number_of_nodes()
            ))
            pool.close()
            pool.join()


        else:
            res = list(tqdm( # tqdm for progress bar
                map(rwr_func, [(self.__G, self.__minimum_per_node, v, self.__max_steps, self.__alpha, self.__bipartite) for v in list(self.__G.nodes())[:10]]),
                total=self.__G.number_of_nodes()
            ))

        for v, top_k_candidates, remaining in res:
            self.candidates.extend(top_k_candidates)
            self.__global_pool.extend(remaining)

        self.__global_pool = sorted(self.__global_pool, key=lambda x: x[1], reverse=True)

        self.candidates.extend([x[0] for x in self.__global_pool[:self.__k - len(self.candidates)]])
        

        return self.get_predicted_links()