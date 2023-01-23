import sys
import warnings

sys.path.insert(1, '../')
from graph_utils import get_latent_similarities
import scipy
from global_types import EdgeList
import numpy as np
import networkx as nx
import random as rd

DataSplittingOutput = (nx.Graph, nx.Graph, EdgeList, EdgeList, EdgeList, EdgeList, EdgeList)

def construct_x_y(pos_edges: EdgeList, neg_edges: EdgeList, embeddings):
    X = get_latent_similarities(embeddings, pos_edges + neg_edges)
    Y = np.concatenate(
        (np.ones((len(pos_edges))), np.zeros((len(neg_edges))))
    )
    
    return X, Y

def get_graphs_and_edges_from_pos_edges(positive_train_edges: EdgeList, positive_validation_edges: EdgeList, positive_test_edges: EdgeList) -> DataSplittingOutput:
    # Remove any duplicate edges
    positive_test_edges = list(set(positive_test_edges))
    positive_train_edges = list(set(positive_train_edges))
    positive_validation_edges = list(set(positive_validation_edges))

    G = nx.Graph(positive_train_edges + positive_validation_edges + positive_test_edges)
    
    negative_train_edges, negative_validation_edges = get_negative_edges(positive_train_edges, positive_validation_edges, positive_test_edges, G)

    # Create subgraphs
    G_test = G.copy()
    G_test.remove_edges_from(positive_test_edges)

    # Remove disjoint nodes from test_edges
    for idx, (v, u) in enumerate(positive_test_edges):
        if G_test.degree(u) == 0 or G_test.degree(v) == 0:
            del positive_test_edges[idx]
    
    return G, G_test, positive_train_edges, negative_train_edges, positive_validation_edges, negative_validation_edges, positive_test_edges
        
def get_graphs_and_edges_from_file(path: str, delimiter: str, train_fraction: float, validation_fraction: float, test_fraction: float, is_bipartite=False) -> DataSplittingOutput:
    if path.endswith(".txt") and is_file_temporal(path, delimiter):
        print("Loading temporal graph")
        (
            G,
            positive_train_edges, 
            positive_validation_edges, 
            positive_test_edges, 
        ) = get_positive_temporal_edges(path, delimiter, train_fraction, validation_fraction, test_fraction, is_bipartite)
    else:
        print("Loading static graph")
        (
            G,
            positive_train_edges, 
            positive_validation_edges, 
            positive_test_edges, 
        ) = get_positive_static_edges(path, delimiter, train_fraction, validation_fraction, test_fraction, is_bipartite)

    negative_train_edges, negative_validation_edges = get_negative_edges(positive_train_edges, positive_validation_edges, positive_test_edges, G)

    # Create subgraphs
    G_test = G.copy()
    G_test.remove_edges_from(positive_test_edges)

    # Remove disjoint nodes from test_edges
    # new_positive_test_edges = list()
    # for idx, (v, u) in enumerate(positive_test_edges):
    #     if G_test.degree(u) > 0 and G_test.degree(v) > 0:
    #         new_positive_test_edges.append((v, u))

    # positive_test_edges = new_positive_test_edges
    
    return G, G_test, positive_train_edges, negative_train_edges, positive_validation_edges, negative_validation_edges, positive_test_edges


def is_file_temporal(path, delimiter):
    for line in open(path, 'r'):
        if line.startswith('#'):
            continue
        if len(tuple(line.strip().split(delimiter))) > 2:
            return True
        return False

def get_positive_static_edges(path: str, delimiter: str, train_fraction, validation_fraction, test_fraction, is_bipartite=False):

    if path.endswith(".mat"):
        G_int = nx.from_scipy_sparse_matrix( scipy.io.loadmat(path)["net"] )
        G = nx.Graph()
        [G.add_edge(str(u), str(v)) for (u, v) in G_int.edges()]
    elif is_bipartite:
        G = nx.bipartite.read_edgelist(path, delimiter=delimiter)
    else:
        G = nx.read_edgelist(path, delimiter=delimiter)

    edges = list(G.edges())
    rd.shuffle(edges)
    
    num_train = int(round(len(edges) * train_fraction))
    num_test = int(round(len(edges) * test_fraction))
    num_validation = int(round(len(edges) * validation_fraction))
    num_remaining = len(edges) - num_train - num_test - num_validation

    positive_remaining_edges = edges[: num_remaining]
    positive_train_edges = edges[num_remaining : num_remaining + num_train]
    positive_validation_edges = edges[num_remaining + num_train : num_remaining + num_train + num_validation]
    positive_test_edges = edges[num_remaining + num_train + num_validation :]

    return G, positive_train_edges, positive_validation_edges, positive_test_edges

def get_positive_temporal_edges(path, delimiter, train_fraction, validation_fraction, test_fraction, is_bipartite=False):

    seen = set()
    G = nx.Graph()
    # Create positive samples
    for line in open(path, 'r'):
        if line.startswith('#'):
            continue
        
        u, v, time = tuple(line.strip().split(delimiter))
        if (u, v) in seen or (v, u) in seen:
            continue
        seen.add((u, v))
        seen.add((v, u))
        
        if is_bipartite:
            G.add_node(u, bipartite=0)
            G.add_node(v, bipartite=1)

        G.add_edge(u, v)
        G.add_edge(v, u)

    edges = G.edges()

    # We must run over the file again now that we have the new ids and know the number of edgest and each split.

    num_train = int(round(len(edges) * train_fraction))
    num_test = int(round(len(edges) * test_fraction))
    num_validation = int(round(len(edges) * validation_fraction))
    num_remaining = len(edges) - num_train - num_test - num_validation

    positive_train_edges: list[tuple[int, int]] = []
    positive_validation_edges: list[tuple[int, int]] = []
    positive_test_edges: list[tuple[int, int]] = []
    positive_remaining_edges: list[tuple[int, int]] = []

    row = 0
    seen = set()
    for line in open(path, 'r'):
        if line.startswith('#'):
            continue
        
        row += 1
        u, v = tuple(line.strip().split(delimiter)[:2])

        if (u, v) in seen or (v, u) in seen:
            continue
        seen.add((u, v))
        seen.add((v, u))

        if row <= num_remaining:
            positive_remaining_edges.append((u, v))
        elif row <= num_remaining + num_train:
            positive_train_edges.append((u, v))
        elif row <= num_remaining + num_train + num_validation:
            positive_validation_edges.append((u, v))
        else:
            positive_test_edges.append((u, v))

    return G, positive_train_edges, positive_validation_edges, positive_test_edges


def get_negative_edges(positive_train_edges: set, positive_validation_edges: set, positive_test_edges: set, G):
    nodes = list(G.nodes())
    samples = set()
    
    all_positive_edges = set(positive_train_edges + positive_validation_edges + positive_test_edges)
    if not len(all_positive_edges) == len(positive_train_edges) + len(positive_validation_edges) + len(positive_test_edges):
        warnings.warn("WARNING: There is an overlap in the train-validation-test splits.")

    while len(samples) < len(positive_validation_edges) + len(positive_train_edges):
        source = rd.choice(nodes)
        target = rd.choice(nodes)

        if (source, target) in samples:
            continue

        if G.has_edge(source, target):
            continue

        samples.add((source, target))

    samples_as_list = list(samples)

    negative_train_edges = samples_as_list[:len(positive_train_edges)]
    negative_validation_edges = samples_as_list[len(positive_train_edges):]

    assert len(negative_train_edges) == len(positive_train_edges)
    assert len(negative_validation_edges) == len(positive_validation_edges)

    return negative_train_edges, negative_validation_edges