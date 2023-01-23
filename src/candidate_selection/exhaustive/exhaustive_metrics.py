
class ExhaustiveMetrics:
    __edge_to_score_dict: dict[(int, int), float]

    def __init__(self, node_pairs: list[tuple[int, int]], predictions: list[float]) -> None:
        self.__node_pairs = node_pairs
        self.__predictions = predictions
        self.__edge_to_score_dict = dict()
        
    def set_result(self, node_pairs: list[tuple[int, int]], predictions: list[float]):
        for pair, prediction in zip(node_pairs, predictions):
            self.__edge_to_score_dict[(pair[0], pair[1])] = prediction[1]
    
    def get_predicted_edges_with_strong_conf(self, positive_test_edges: list[tuple[int, int]], conf_threshold=0.9):
        n_found = 0
        for edge in positive_test_edges:
            v = edge[0]
            u = edge[1]
            if self.__edge_to_score_dict[(v,u)] > conf_threshold:
                n_found += 1
            
        return n_found
