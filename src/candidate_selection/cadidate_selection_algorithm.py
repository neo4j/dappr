from abc import ABC, abstractmethod
import networkx as nx

EdgeList = list[tuple[int, int]]

class CandidateSelectionAlgorithm(ABC):
    __n_scanned: int
    input_params: dict
    exclude_params: list[str] = []
    nodes: int
    edges: int

    @abstractmethod
    def run(self) -> EdgeList:
        pass

    @abstractmethod
    def get_predicted_links(self) -> EdgeList:
        pass

    @abstractmethod
    def get_n_scanned(self) -> int:
        pass

    def get_input_params(self) -> dict:
        return {key:value for (key,value) in self.input_params.items() if key not in self.exclude_params}

    def get_unique_predicted_links(self) -> EdgeList:
        edge_set = set()

        for v, u in self.get_predicted_links():
            if u <= v:
                edge_set.add((v, u))
            else:
                edge_set.add((u, v))
        
        return edge_set

    def get_name(self):
        return self.__class__.__name__