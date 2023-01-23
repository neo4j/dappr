import sys
from functools import partial
from ..cadidate_selection_algorithm import EdgeList, CandidateSelectionAlgorithm
sys.path.insert(1, '../')

from joblib import Parallel, delayed
import multiprocessing
from collections import defaultdict
from typing import Callable
from graph_utils import sample_random_nodes
import numpy as np
import networkx as nx
import random as rd
from .types import KnnDict, KnnDict_lowest
from time import time
from tqdm import tqdm

def nn_descent(inputs: tuple[nx.Graph, int, dict[int, set[int]], dict[int, set[int]], dict[int, set[int]], dict[int, set[int]], Callable[[int, int], float], float]) -> int:
    G, v, new, new_reverse, old, old_reverse, sigma, rho = inputs
    k = len(old[v]) + len(new[v])

    if len(old_reverse[v]) > 0:
        sample_size = int(min(len(old_reverse[v]), np.round(rho * k)))
        old[v].union(np.random.choice(list(old_reverse[v]), size=sample_size, replace=False))

    if len(new_reverse[v]) > 0:
        sample_size = int(min(len(new_reverse[v]), np.round(rho * k)))
        new[v].union(np.random.choice(list(new_reverse[v]), size=sample_size, replace=False))
    
    candidate_pairs = []
    for u1 in new[v]:
        for u2 in new[v]:
            if u1 < u2:
                candidate_pairs.append((u1, u2))

        for u2 in old[v]:
            candidate_pairs.append((u1, u2))

    if(len(candidate_pairs) == 0): return 0, []

    l_vec = sigma(candidate_pairs)
    n_scanned = len(candidate_pairs)

    callback_args: list[tuple[int, int, float, bool]] = []
    for idx, (u1, u2) in enumerate(candidate_pairs):
        l = l_vec[idx][1]

        if G.has_edge(u1, u2): continue
        callback_args.extend([
            (u1, u2, l, True),
            (u2, u1, l, True),
        ])

    return n_scanned, callback_args

class Knn(CandidateSelectionAlgorithm):
    """Based on the paper https://www.cs.princeton.edu/cass/papers/www11.pdf"""
    G: nx.Graph
    K: int
    B_lowest_node: KnnDict_lowest
    B: KnnDict
    delta: float
    rho: float
    n_scanned: int

    def __init__(self, G: nx.Graph, num_edges_to_find: int, score_links: Callable[[int, int], float], clf_best_accuracy:float, sample_rate=0.5, delta=0.01, p_random_neighbors=1.0, parallel=True, verbose=True, max_iterations=20):
        self.input_params = locals()
        self.exclude_params = ["self", "G", "score_links"]

        self.__max_iterations = max_iterations
        self.verbose = verbose
        self.__parallel = parallel
        self.G = G
        self.num_edges_to_find = num_edges_to_find
        self.K = num_edges_to_find / self.G.number_of_nodes()
        self.sigma = score_links
        self.clf_best_accuracy = clf_best_accuracy
        self.delta = delta
        self.rho = sample_rate
        self.p_random_neighbors = p_random_neighbors
        self.B = dict()
        self.B_lowest_node = dict()
        self.n_scanned = 0
        

        if self.K**2 > G.number_of_nodes():
            print('WARNING: k^2 > v. Running KNN wll be slower than running exhaustive search.')

    def set_seed(self, seed):
        np.random.seed(seed)


    def get_predicted_links(self) -> EdgeList:
        """Returns a list of tuples (v, u) where v is a node in the graph and u is a predicted link to v."""
        predicted_links = []
        for v in self.B:
            for u in self.B[v]:
                predicted_links.append((v, u))
        return predicted_links

    def get_n_scanned(self) -> int:
        return self.n_scanned
        
    def run(self) -> EdgeList:
        self.__initialize()
        for i in range(self.__max_iterations):
            if self.verbose: print(f"Running iteration {i}")
            start = time()
            should_terminate = self.__iteration()
            if self.verbose: print(f"Iteration took {time() - start} seconds.")
            if should_terminate: break
        
        predicted_links = self.get_predicted_links()
        return predicted_links
        
    def __initialize(self):
        self.n_scanned = 0

        for v in self.G.nodes():
            neighbors = list(self.G.neighbors(v))
            # Make local k, that can be floored or ceiled self.K with a probability of (K - floor(k))
            k = int(self.K)
            if rd.random() < self.K - k:
                k += 1

            random_neighbors = sample_random_nodes(neighbors, int(k * (1 - self.p_random_neighbors)))
            num_random_nodes = k - len(random_neighbors)

            random_nodes = sample_random_nodes(self.G.nodes(), num_random_nodes)
            self.B[v] = dict()
            nodes = random_nodes + random_neighbors
            for node in nodes:
                self.B[v][node] = (0, True)

            if len(nodes) > 0:
                self.B_lowest_node[v] = (nodes[0], 0) # Init any node as lowest node. Will not matter later.
            else:
                self.B_lowest_node[v] = (None, 0) # Init any node as lowest node. Will not matter later.

    def __iteration(self):
        old: dict[int, set[int]] = defaultdict(set)
        new: dict[int, set[int]] = defaultdict(set)

        old_reverse: dict[int, set[int]] = defaultdict(set)
        new_reverse: dict[int, set[int]] = defaultdict(set)

        for v in self.G.nodes():
            for u in self.B[v]:
                score, isNew = self.B[v][u]

                if not isNew:
                    old[v].add(u)
                    old_reverse[u].add(v)

                elif rd.random() <= self.rho:
                    new[v].add(u)
                    self.B[v][u] = (score, False) # Set isNew to False
                    new_reverse[u].add(v)

        self.c = 0

        if self.__parallel:
            # pbar = tqdm(total=self.G.number_of_nodes())
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            
            results = list(
                tqdm(pool.imap_unordered([
                    (
                        nn_descent, 
                        [self.G, q, new, new_reverse, old, old_reverse, self.sigma, self.rho]
                    ) 
                    for q in self.G.nodes()],
                chunksize=round(self.G.number_of_nodes()/10)), total=self.G.number_of_nodes())
                )
            pool.close()
            pool.join()

        else: 
            results = list(
                tqdm(map(nn_descent, [
                    (
                        self.G, q, new, new_reverse, old, old_reverse, self.sigma, self.rho
                    ) 
                for q in self.G.nodes()],
                ), total=self.G.number_of_nodes()),
            )
        
        for result in results:
            n_scanned, args = result
            self.__update_NN(n_scanned, args)

        should_terminate = self.c < self.delta * self.G.number_of_nodes() * self.K
        
        if self.verbose: print(f"Number of bucket alterations: {self.c}, Termination Threshold: {self.delta * self.G.number_of_nodes() * self.K}")
        
        return should_terminate

    def __update_NN(self, n_scanned, args: list[tuple[int, int, float, bool]]):
        """Inserts u2 into B[u1] if the score is greater than the lowes score in B[u1].
        Returns True if updated, Otherwise False"""
        edits = 0
        self.n_scanned += n_scanned

        for u1, u2, l, isNew in args:
            if u2 in self.B[u1]: # Don't add duplicates
                continue
            
            lowest_node, lowest_score = self.B_lowest_node[u1]
            if l > lowest_score:
                del self.B[u1][lowest_node]
                self.B[u1][u2] = (l, isNew)
                # Now update lowest_node
                self.__update_B_lowest_node(u1)
                edits += 1
                continue

            else:
                continue
        
        self.c += edits
        

    
    def __update_B_lowest_node(self, v: int):
        """Helper method for updating the lowest scored node in self.__B_lowest_node[v]"""
        min_score = 1
        min_node = None

        for u in self.B[v]:
            score = self.B[v][u][0]
            if score <= min_score:
                min_score = score
                min_node = u

        if min_node != None: self.B_lowest_node[v] = (min_node, min_score)