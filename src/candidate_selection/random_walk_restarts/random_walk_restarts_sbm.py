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
def rwr_func(inputs: tuple[nx.Graph, int, int, float, float, bool]):
    G, v, max_steps, alpha, bayesian_factor, bipartite = inputs
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
    
    # visited_counts[v] = 0
    # pagerank = nx.pagerank(G, personalization={ v: 1 }, max_iter=5)
    # visited_counts = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    visited_counts = sorted(visited_counts.items(), key=lambda x: x[1], reverse=True)

    deg = G.degree(v)
    k = int(deg * bayesian_factor)

    global_pool_candidates = list()
    top_k_candidates = []
    
    for u, count in visited_counts:
        if G.has_edge(v, u): continue
        if bipartite:
            if G.nodes[v]["bipartite"] == G.nodes[u]["bipartite"]: continue
        if len(top_k_candidates) < k: 
            top_k_candidates.append((v, u))
        else:
            if len(global_pool_candidates) < 0.5 * k: # Add 50% of k extra to global pool
                global_pool_candidates.append(((v, u), count * deg))
            
    # if v == 0:
    #     print_stuff(G, v, visited_counts)

    return v, top_k_candidates, global_pool_candidates

def print_stuff(G: nx.Graph, q, foo: list):
    sum_visits = 0
    new_foo = []

    for node, visits in foo:
        sum_visits += visits

    for node, visits in foo:
        new_foo.append((node, visits / sum_visits, len(nx.shortest_path(G, q, node))))

    print(sorted(new_foo, key=lambda x: x[1], reverse=False))



class RandomWalkRestartsSBM(CandidateSelectionAlgorithm):

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
        self.__candidates = defaultdict(list)
        self.__global_pool: list[((int, int), float)] = list()
        

    def get_n_scanned(self) -> int:
        return len(self.get_unique_predicted_links())

    def get_predicted_links(self) -> EdgeList:
        """Returns a list of tuples (v, u) where v is a node in the graph and u is a predicted link to v."""
        links = list()
        for candidates in self.__candidates.values():
            links.extend(candidates)

        if len(links) < self.__k:
            self.__global_pool.sort(key=lambda x: x[1], reverse=True)
            links.extend(map(lambda x: x[0], self.__global_pool[:self.__k - len(links)]))

        return links
        
    def run(self) -> EdgeList:
        # pbar = tqdm(total=self.__G.number_of_nodes()) if self.__verbose else None
        bayesian_factor = self.__avg_edges_to_predict_per_node / self.__avg_degree
        if self.__parallel:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            res = list(tqdm( # tqdm for progress bar
                pool.imap_unordered(rwr_func, [(self.__G, v, self.__max_steps, self.__alpha, bayesian_factor, self.__bipartite) for v in self.__G.nodes()], chunksize=200),
                total=self.__G.number_of_nodes()
            ))

            pool.close()
            pool.join()

        else:
            res = list(tqdm( # tqdm for progress bar
                map(rwr_func, [(self.__G, v, self.__max_steps, self.__alpha, bayesian_factor, self.__bipartite) for v in list(self.__G.nodes())[:10]]),
                total=self.__G.number_of_nodes()
            ))

        for v, top_k_candidates, global_pool_candidates in res:
            self.__candidates[v] = top_k_candidates
            self.__global_pool.extend(global_pool_candidates)

        

        return self.get_predicted_links()
