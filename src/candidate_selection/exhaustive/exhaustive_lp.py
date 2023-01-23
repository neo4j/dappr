from typing import Callable
from .exhaustive_metrics import ExhaustiveMetrics
import networkx as nx
from tqdm import tqdm

class ExhaustiveLP:
    metrics: ExhaustiveMetrics

    def __init__(self, G: nx.Graph, score_links: Callable[[int, int], float]):
        self.__G = G
        self.__score_links = score_links
        self.metrics = ExhaustiveMetrics()

    def run(self):
        node_pairs = list()
        progress_bar = tqdm(range(self.__G.number_of_nodes() + 1))
        for u in self.__G.nodes():
            for v in self.__G.nodes():
                if u != v:
                    node_pairs.append((u,v))
            progress_bar.update(1)


        predictions = self.__score_links(node_pairs)
        progress_bar.update(1)

        self.metrics.set_result(node_pairs, predictions)
        return node_pairs, predictions