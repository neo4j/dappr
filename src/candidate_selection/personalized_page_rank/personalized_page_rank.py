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

# Running pagerank on subgraph from v with max size K
def rwr_func(inputs: tuple[nx.Graph, int, int]):
    G, K, v = inputs
    
    # Old implementation running on subgraphs. Turns out it wasn't any faster
    # subgraph_nodes = set()
    # subgraph_nodes.add(v)

    # for (idx, layer) in enumerate(nx.bfs_layers(G, [v])):
    #     subgraph_nodes.update(layer)
    #     if idx >= 5: break
    # subgraph = G.subgraph(subgraph_nodes)

    pagerank = nx.pagerank_scipy(G, personalization={ v: 1 }, alpha=0.8)
    # if v == 0:
    #     print("pagerank", sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10])
    pagerank_sorted = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

    top_k_candidates = []
    for u, _ in pagerank_sorted[1:]:
        if G.has_edge(v, u): continue
        top_k_candidates.append((v, u))
        if len(top_k_candidates) >= K: break
    # top_k_candidates = [ (v, u) for u, _ in pagerank_sorted[:K] ]
    return v, top_k_candidates

class PersonalizedPageRank(CandidateSelectionAlgorithm):

    def __init__(self, G: nx.Graph, K: int, parallel=True, verbose=True):
        self.input_params = locals()
        self.exclude_params = ["self", "G", "score_links"]

        self.__verbose = verbose
        self.__parallel = parallel
        self.__G = G.to_directed()
        self.__K = K
        self.__n_scanned = 0
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
                pool.imap_unordered(rwr_func, [(self.__G, self.__K, v) for v in self.__G.nodes()], chunksize=200),
                total=self.__G.number_of_nodes()
            ))
            pool.close()
            pool.join()


        else:
            res = list(tqdm( # tqdm for progress bar
                map(rwr_func, [(self.__G, self.__K, v) for v in self.__G.nodes()]),
                total=self.__G.number_of_nodes()
            ))

        for v, top_k_candidates in res:
            self.__candidates[v] = top_k_candidates

        return self.get_predicted_links()

    