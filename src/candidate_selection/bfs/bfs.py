from functools import partial
import sys

sys.path.insert(1, '../')
from ..cadidate_selection_algorithm import EdgeList, CandidateSelectionAlgorithm
import multiprocessing
from collections import defaultdict
from typing import Callable
import networkx as nx
from tqdm import tqdm

def iterate(inputs: tuple[nx.Graph, int, int | None, int]) -> None:
        """Performs the bfs originating in node v. Scores links (v, u) where neigbor_degree(u) <= max_depth"""
        G, max_depth, K, v = inputs
        candidates = list()
        layers = nx.bfs_layers(G, [v])
        next(layers) # Discard first layer as they are already linked
        for depth, layer in enumerate(layers):
            if depth == max_depth: break

            for u in layer:
                candidates.append((v, u))
                if len(candidates) >= K: return v, candidates
        return v, candidates

class Bfs(CandidateSelectionAlgorithm):
    """Selects candidates for each node by combining the query node with it's 2nd+ degree neighbors."""
    def __init__(self, G: nx.Graph, K: int | None, parallel=True, verbose=True, max_depth=4):
        self.input_params = locals()
        self.exclude_params = ["self", "G", "score_links"]

        self.__verbose = verbose
        self.__parallel = parallel
        self.__G = G
        self.__K = K
        self.__candidates_by_node: dict[tuple[int, tuple[int, int]]] = dict()
        self.__max_depth = max_depth
        

    def get_input_params(self) -> int:
        return {key:value for (key,value) in self.input_params.items() if key not in self.exclude_params}

    def get_n_scanned(self) -> int:
        return len(self.get_unique_predicted_links())

    def get_predicted_links(self) -> EdgeList:
        """Returns a list of tuples (v, u) where v is a node in the graph and u is a predicted link to v."""
        predicted_links = list()
        for candidates in self.__candidates_by_node.values():
                predicted_links.extend(candidates)
        return predicted_links
        
    def run(self) -> EdgeList:
        if self.__parallel:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            
            res = list(tqdm( # tqdm for progress bar
                pool.imap_unordered(iterate, [(self.__G, self.__max_depth, self.__K, v) for v in self.__G.nodes()], chunksize=200),
                total=self.__G.number_of_nodes()
            ))
            pool.close()
            pool.join()
            
        else:
            res = list(tqdm( # tqdm for progress bar
                map(iterate, [(self.__G, self.__max_depth, self.__K, v) for v in self.__G.nodes()]),
                total=self.__G.number_of_nodes()
            ))

        for v, candidates in res:
            self.__candidates_by_node[v] = candidates
        
        return self.get_predicted_links()