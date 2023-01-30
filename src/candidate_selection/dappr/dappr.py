from collections import defaultdict
import sys
import time
from ..cadidate_selection_algorithm import EdgeList, CandidateSelectionAlgorithm
sys.path.insert(1, '../')
import multiprocessing
import networkx as nx
from tqdm import tqdm
import numpy as np
import math

def iteration(inputs):
    pi_prev, q, alpha, neighbors_q, sbm_factor = inputs
    
    pi_q = defaultdict(lambda: 0.0)
    pi_q[q] = 1-alpha

    if len(neighbors_q) == 0: 
        return [(q, 1-alpha)], q

    k = len(neighbors_q) * sbm_factor
    truncation = math.ceil(k*20)

    frac = (alpha / len(neighbors_q))

    for v in neighbors_q:
        for w, prob in pi_prev[v][:truncation]:
            pi_q[w] += frac * prob

    pi_q_sorted = sorted(pi_q.items(), key=lambda x: x[1], reverse=True)[:truncation]
    return pi_q_sorted, q


class Dappr(CandidateSelectionAlgorithm):

    def __init__(self, G: nx.Graph, k: int, epsilon: float=0.05, alpha: float=0.8, parallel=True, verbose=True, bipartite=False):
        self.input_params = locals()
        self.exclude_params = ["self", "G"]

        self.__verbose = verbose
        self.__parallel = parallel
        self.__G = G
        self.__k = k
        self.__epsilon = epsilon
        self.__alpha = alpha
        self.__avg_edges_to_predict_per_node = (k / G.number_of_nodes())
        self.__avg_degree = (2*G.number_of_edges()) / G.number_of_nodes() # 2*edges due to Undirected
        self.__sbm_factor = self.__avg_edges_to_predict_per_node / self.__avg_degree
        self.__bipartite = bipartite
        self.__alpha = alpha
        

        self.__pi = {node: [(node, 1)] for node in G.nodes()}
        self.links = None

    def get_n_scanned(self) -> int:
        return len(self.get_unique_predicted_links())

    def get_predicted_links(self) -> EdgeList:
        """Returns a list of tuples (v, u) where v is a node in the graph and u is a predicted link to v."""
        if not self.links:
            self.generate_links()
        return self.links

    def generate_links(self) -> EdgeList:
        links = set()
        global_pool = list()
        total_k = 0
        for q, pi_q in self.__pi.items():
            
            # Select number of candidates per node relative to degree
            deg = self.__G.degree(q)
            k = deg * self.__sbm_factor
            total_k += k
            k_added = 0
            local_global_pool = []

            for v, prob in pi_q:
                if q == v: continue
                if self.__G.has_edge(q, v): continue
                if self.__bipartite:
                    if self.__G.nodes[q]["bipartite"] == self.__G.nodes[v]["bipartite"]: 
                        continue
                    
                if k_added < int(k): 
                    k_added += 1 # Even if (v, q) has already been added, count this anyways, we solve this later in the global pool.
                    if (q,v) not in links and (v,q) not in links:
                        links.add((q, v))

                elif len(local_global_pool) < math.ceil(k):
                    local_global_pool.append(((q, v), prob))
                else:
                    break
            
            global_pool.extend(local_global_pool)

        # Add global pool
        print("Total k:", total_k)
        print("Adding global pool. Amount of missing links:", self.__k - len(links), ". Amount of links:", len(links))
        if len(links) < self.__k:
            global_pool.sort(key=lambda x: x[1], reverse=True)

            for (u, v), prob in global_pool:
                if len(links) == self.__k:
                    break
                    
                if (u, v) in links or (v, u) in links:
                    continue
                else:
                    links.add((u, v))
        self.links = list(links)
        return self.links
        
    def run(self) -> EdgeList:

        for iteration in range(1, 10): # Rarely takes more than 10 iterations to converge. Increase if needed.

            if self.__parallel and self.__G.number_of_nodes() > 3000:
                pool = multiprocessing.Pool(multiprocessing.cpu_count())
                results = list(
                    tqdm(pool.imap_unordered(iteration, [
                        (
                            { node: self.__pi[node] for node in list(self.__G.neighbors(q)) + [q] },
                            q,
                            self.__alpha, 
                            list(self.__G.neighbors(q)),
                            self.__sbm_factor,
                        ) 
                    for q in self.__G.nodes()],
                    chunksize=5000), total=self.__G.number_of_nodes())
                    
                )
                pool.close()
                pool.join()

            else:
                results = list(
                    tqdm(map(iteration, [
                        (
                            {neigh: self.__pi[neigh] for neigh in list(self.__G.neighbors(q)) + [q]},
                            q, 
                            self.__alpha, 
                            list(self.__G.neighbors(q)),
                            self.__sbm_factor,
                        ) 
                    for q in self.__G.nodes()],
                    ), total=self.__G.number_of_nodes()),
                )

            total_distance = 0
            delta = 0.0
            start_time = time.time()
            for pi_q, q in results:
                pi_q_prev_dict = {node: prob for node, prob in self.__pi[q]}
                pi_q_dict = {node: prob for node, prob in pi_q}

                for key in list(pi_q_prev_dict.keys()) + list(pi_q_dict.keys()):
                    delta += abs(pi_q_prev_dict.get(key, 0) - pi_q_dict.get(key, 0))
                    total_distance += pi_q_prev_dict.get(key, 0)

                self.__pi[q] = pi_q
            print("Took total time: ", time.time() - start_time, "iteration", iteration)
            print("Total abs SQRTMSE:", delta, "Total distance:", total_distance, "mse/distance:", delta/total_distance)

            if delta / total_distance < self.__epsilon and iteration >= 3:
                print("Breaking at iteration:", iteration)
                break
            


        return self.get_predicted_links()