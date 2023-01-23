import sys
sys.path.insert(1, '../')

import numpy as np
import networkx as nx
from global_types import EdgeList

def sample_random_nodes(nodes: list[int], K: int) -> list[int]:
    if K == 0: return []

    max_num_nodes = min(K, len(nodes))
    random_nodes = list(np.random.choice(nodes, size=max_num_nodes, replace=False))
    return random_nodes

def get_latent_similarities(embeddings, edges: EdgeList):
    """Note: I don't think we're embedding edges, but rather finding similarity between embeddings?"""
    return np.array(
        [
            embeddings[source] * embeddings[target]
            for (source, target) in edges
        ]
    )

def predict_proba(clf, embeddings, u, v):
    return clf.predict_proba(get_latent_similarities(embeddings, [(v, u)]))[0][1]

def print_graph_overview(G: nx.Graph):
    print(f"Number of nodes in graph: {G.number_of_nodes()}")
    print(f"Number of edges in graph: {G.number_of_edges()}")
    print(f"Avg number of edges/node: {G.number_of_edges() / G.number_of_nodes()}")

def generate_sub_graph(graph: nx.Graph, num_edges: int):
    """Returns a connected subgraph of graph"""
    sub_graph = nx.Graph()

    u = np.random.choice(graph.nodes())
    sub_graph.add_node(u)

    for edge_number in range(num_edges):
        neighbors = list(graph.neighbors(u))
        v = np.random.choice(neighbors)
        
        sub_graph.add_edge(u, v)
        sub_graph.add_edge(v, u)
        u = v
    return sub_graph