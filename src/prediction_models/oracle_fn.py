from graph_utils import get_latent_similarities
import sys
sys.path.insert(1, '../')

import numpy as np
from global_types import EdgeList

def select_oracle(oracle_type, clf, embeddings, positive_test_edges):
    if oracle_type == 'puro':
        def oracle_fn(X):
            return clf.predict_proba(get_latent_similarities(embeddings, X)) #TODO: Rename latent

    elif oracle_type == 'puro+perfect':
        positive_test_edges_set = set(positive_test_edges)
        def oracle_fn(X):
            prediction_probabilities = clf.predict_proba(get_latent_similarities(embeddings, X))
            for idx, (v, u) in enumerate(X):
                if (v, u) in positive_test_edges_set or (u, v) in positive_test_edges_set:
                    prediction_probabilities[idx] = [prediction_probabilities[idx][0], 1]
            return prediction_probabilities

    return oracle_fn