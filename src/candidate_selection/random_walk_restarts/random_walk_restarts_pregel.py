import sys
from ..cadidate_selection_algorithm import EdgeList, CandidateSelectionAlgorithm
sys.path.insert(1, '../')
import multiprocessing
import networkx as nx
from tqdm import tqdm
import math

# Total complexity is O(n * (avg_deg * avg_nodes_to_output * CONSTANT ) )
def rwr_func(inputs: tuple[list, int, bool, dict[dict[int]]]):
    neighbors, q, bipartite, rw_prev_step, Pi_q, walk_length_prob, __sbm_factor = inputs

    if len(neighbors) == 0:
        return q, dict(), Pi_q, 0

    pi_q = dict()
    
    random_neighbors = neighbors # Mby random sample
    neighbor_fraction = 1 / len(random_neighbors)
    # to_pick = len(random_neighbors)
    # to_keep = len(random_neighbors)*CONSTANT

    k = len(neighbors) * __sbm_factor
    to_pick = math.ceil(k*10)
    to_keep = math.ceil(k*10)
    
    # O(avg_deg)
    for neighbor in neighbors:
        # Now add the random neighbors pi at last time step
        pi_neighbor = rw_prev_step[neighbor]
        # Mby do a random sample?
        neighbors_probs = pi_neighbor[:to_pick]
        
        # O(avg_nodes_to_output * CONSTANT)
        for visited_node, prob in neighbors_probs: # Take (1 / |N(q)|) * neighbors_rw_table (at t-1)
            if visited_node not in pi_q: pi_q[visited_node] = 0

            pi_q[visited_node] += prob * neighbor_fraction 
    
    # Create a new, shorter rw (avg_nodes_to_output * 4)
    sorted_pi_q = sorted(pi_q.items(), key=lambda x: x[1], reverse=True)[:to_keep] # Priority queue?
    
    total_delta = 0

    # Aggregate the sums for faster aggregation after parallel step
    for key, prob in sorted_pi_q:
        change_in_prob = prob * walk_length_prob
        total_delta += change_in_prob

        if key not in Pi_q: Pi_q[key] = 0
        Pi_q[key] += change_in_prob

    return q, sorted_pi_q, Pi_q, total_delta


class RandomWalkRestartsPregel(CandidateSelectionAlgorithm):

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
        

        self.__Pi = {node: {node: self.__get_walk_length_probability(0, self.__alpha)} for node in G.nodes()}
        self.__pi_prev_walk_length = {node: [(node, 1)] for node in G.nodes()}


    def get_n_scanned(self) -> int:
        return len(self.get_unique_predicted_links())

    def get_predicted_links(self) -> EdgeList:
        """Returns a list of tuples (v, u) where v is a node in the graph and u is a predicted link to v."""
        links = set()
        global_pool = list()
        total_k = 0
        for q, Pi_q in self.__Pi.items():
            
            # Select number of candidates per node relative to degree
            deg = self.__G.degree(q)
            k = deg * self.__sbm_factor
            total_k += k
            k_added = 0
            local_global_pool = []

            sorted_Pi_q = sorted(Pi_q.items(), key=lambda x: x[1], reverse=True)

            for v, prob in sorted_Pi_q:
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
        print("Adding global pool, missing links:", self.__k - len(links), "has", len(links))
        if len(links) < self.__k:
            global_pool.sort(key=lambda x: x[1], reverse=True)

            for (u, v), prob in global_pool:
                if len(links) == self.__k:
                    break
                    
                if (u, v) in links or (v, u) in links:
                    continue
                else:
                    links.add((u, v))
        
        return list(links)
        
    def run(self) -> EdgeList:
        cumulative_walk_length_prob = 0
        prev_step_total_probs = 0

        for walk_length in range(1, 15):
            walk_length_prob = self.__get_walk_length_probability(walk_length, self.__alpha)
            cumulative_walk_length_prob += walk_length_prob

            if self.__parallel:
                pool = multiprocessing.Pool(multiprocessing.cpu_count())
                results = list(
                    tqdm(pool.imap_unordered(rwr_func, [
                        (
                            list(self.__G.neighbors(q)),
                            q,
                            self.__bipartite,
                            self.__pi_prev_walk_length,
                            self.__Pi[q],
                            walk_length_prob,
                            self.__sbm_factor
                        ) 
                    for q in self.__G.nodes()],
                    chunksize=round(self.__G.number_of_nodes()/10)), total=self.__G.number_of_nodes())
                    
                )
                pool.close()
                pool.join()

            else:
                results = list(
                    map(rwr_func, [
                        (
                            list(self.__G.neighbors(q)),
                            q,
                            self.__bipartite,
                            self.__pi_prev_walk_length,
                            self.__Pi[q],
                            walk_length_prob,
                            self.__sbm_factor
                        )
                        for q in self.__G.nodes()]
                    ),
                )

            total_delta = 0

            for q, pi, Pi_q, delta in results:
                self.__Pi[q] = Pi_q
                total_delta += delta
                self.__pi_prev_walk_length[q] = pi
            

            if prev_step_total_probs > 0:
                print("walk_length", walk_length, "total_delta", total_delta, "total_probs", prev_step_total_probs, "prev_step_total_probs/n", prev_step_total_probs / self.__G.number_of_nodes(), "err", total_delta / prev_step_total_probs)
                if total_delta / prev_step_total_probs < self.__epsilon:
                    print("Finished at delta: ", total_delta / prev_step_total_probs)
                    self.final_walk_length = walk_length
                    break
            
            prev_step_total_probs += total_delta
        
        return self.get_predicted_links()

    def __get_walk_length_probability(self, walk_length: int, alpha: float) -> float:
        return (alpha ** walk_length) * (1 - alpha)