from ogb.linkproppred import LinkPropPredDataset
import numpy as np
import networkx as nx
import os
import gzip
import shutil
from enum import Enum
import scipy
from dataset.data_splitting import get_graphs_and_edges_from_file, get_graphs_and_edges_from_pos_edges

class Datasets(Enum):
    # Static
    ARXIV_CONDMAT = "arxiv_condmat",
    ARXIV = "arxiv",
    DBLP = "dblp",
    EPINIONS = "epinions",
    FACEBOOK1 = "facebook1",
    HS_PROTEIN = "hs_protein",
    YEAST = "yeast",

    # Static .tsv
    AS_SKITTER = "as-skitter",
    ROADNET_PA = "roadNet-PA",
    ROADNET_CA = "roadNet-CA",
    AMAZON = "com-amazon",

    # Static .mat
    POWER = "power",
    US_AIR = "us_air",
    ROUTER = "router",

    # Temporal
    SV_WIKIQUOTE_EDITS = "sv_wikiquote_edits",
    DIGG = "digg",
    ENRON = "enron",
    FACEBOOK2 = "facebook2",
    MATH_OVERFLOW = "math_overflow",
    MOVIE_LENS = "movielens",
    REDDIT = "reddit",

    # OGB Static
    OGB_CITATION2 = "OGB_CITATION2",
    OGB_DDI = "OGB_DDI",
    OGB_PPA = "OGB_PPA",
    OGB_COLLAB = "OGB_COLLAB",

static_datasets = [Datasets.ARXIV.value[0], Datasets.DBLP.value[0], Datasets.EPINIONS.value[0], Datasets.FACEBOOK1.value[0], Datasets.HS_PROTEIN.value[0], Datasets.YEAST.value[0], Datasets.ARXIV_CONDMAT.value[0]]
temporal_datasets = [Datasets.DIGG.value[0], Datasets.ENRON.value[0], Datasets.FACEBOOK2.value[0], Datasets.MATH_OVERFLOW.value[0], Datasets.MOVIE_LENS.value[0], Datasets.REDDIT.value[0], Datasets.SV_WIKIQUOTE_EDITS.value[0]] 
tsv = [Datasets.AS_SKITTER.value[0], Datasets.ROADNET_PA.value[0], Datasets.ROADNET_CA.value[0], Datasets.AMAZON.value[0]]
mat = [Datasets.POWER.value[0], Datasets.US_AIR.value[0], Datasets.ROUTER.value[0]]
bipartite_datasets = [Datasets.MOVIE_LENS.value[0], Datasets.SV_WIKIQUOTE_EDITS.value[0], Datasets.ARXIV_CONDMAT.value[0]]

class DatasetReader:

    def __init__(self) -> None:
        self.__ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        
    def read_and_split(self, dataset: Datasets, train_fraction: float, validation_fraction: float, test_fraction: float):
        dataset_mame: str = dataset.value[0]

        bipartite = False
        if dataset_mame in bipartite_datasets:
            bipartite = True

        if(dataset_mame.startswith("OGB")):
            if dataset_mame == Datasets.OGB_DDI.value[0]:
                train_edge, valid_edge, test_edge = self.__get_ogb_data("ddi")

            elif dataset_mame == Datasets.OGB_PPA.value[0]:
                train_edge, valid_edge, test_edge = self.__get_ogb_data("ppa")

            elif dataset_mame == Datasets.OGB_COLLAB.value[0]:
                train_edge, valid_edge, test_edge = self.__get_ogb_data("collab")

            elif dataset_mame == Datasets.OGB_CITATION2.value[0]:
                train_edge, valid_edge, test_edge = self.__get_ogb_data("citation2")

            (
                G, 
                G_test, 
                positive_train_edges, 
                negative_train_edges, 
                positive_validation_edges, 
                negative_validation_edges, 
                positive_test_edges
            ) = get_graphs_and_edges_from_pos_edges(train_edge, valid_edge, test_edge)
            
        else:
            if dataset_mame in temporal_datasets:
                dir_name = "temporal"
                path, delimiter = self.__get_txt_data(dir_name, dataset_mame)
            elif dataset_mame in static_datasets:
                dir_name = "static"
                path, delimiter = self.__get_txt_data(dir_name, dataset_mame)
            elif dataset_mame in mat:
                dir_name = "mat"
                path, delimiter = self.__get_mat_data(dir_name, dataset_mame)
            elif dataset_mame in tsv:
                dir_name = "tsv"
                path, delimiter = self.__get_tsv_data(dir_name, dataset_mame)
            
            else:
                raise Exception("Unknown dataset")

            
            (
                G, 
                G_test, 
                positive_train_edges, 
                negative_train_edges, 
                positive_validation_edges, 
                negative_validation_edges, 
                positive_test_edges
            ) =  get_graphs_and_edges_from_file(path, delimiter, train_fraction, validation_fraction, test_fraction, bipartite)

        return G, G_test, positive_train_edges, negative_train_edges, positive_validation_edges, negative_validation_edges, positive_test_edges, bipartite

    def __get_txt_data(self, dir_name: str, file_name: str) -> nx.Graph:
        path = os.path.join(self.__ROOT_DIR, f"{dir_name}", f"{file_name}.txt")
        delimiter = ' '

        return path, delimiter # nx.read_edgelist(path, delimiter=' ', nodetype=int)

    def __get_tsv_data(self, dir_name: str, file_name: str) -> nx.Graph:
        path = os.path.join(self.__ROOT_DIR, f"{dir_name}", f"{file_name}.tsv")
        delimiter = '\t'

        return path, delimiter

    def __get_mat_data(self, dir_name: str, file_name: str) -> nx.Graph:
        path = os.path.join(self.__ROOT_DIR, f"{dir_name}", f"{file_name}.mat")
        return path, None

    def __get_ogb_data(self, dataset_name: str):
        # Ensure the data is downloaded
        dataset = LinkPropPredDataset(name = "ogbl-" + dataset_name, root = 'dataset/')

        split_edge = dataset.get_edge_split()
        if dataset_name == "citation2":
            print("Building citation2")
            train_edge = list(zip(split_edge["train"]["source_node"], split_edge["train"]["target_node"]) )
            train_edge = train_edge + list(zip(split_edge["train"]["target_node"], split_edge["train"]["source_node"]))
            
            valid_edge = list(zip(split_edge["valid"]["source_node"], split_edge["valid"]["target_node"]) )
            valid_edge = valid_edge + list(zip(split_edge["valid"]["target_node"], split_edge["valid"]["source_node"]))
            
            test_edge = list(zip(split_edge["test"]["source_node"], split_edge["test"]["target_node"]) )
            test_edge = test_edge + list(zip(split_edge["test"]["target_node"], split_edge["test"]["source_node"]))
            
            print("Finished building citation2")
        else:
            train_edge, valid_edge, test_edge = split_edge["train"]["edge"], split_edge["valid"]["edge"], split_edge["test"]["edge"]
        train_edge = [(str(v), str(u)) for u, v in train_edge]
        valid_edge = [(str(v), str(u)) for u, v in valid_edge]
        test_edge = [(str(v), str(u)) for u, v in test_edge]
        
        return train_edge, valid_edge, test_edge
        # return self.__read_ogb_graph("ogbl_" + dataset_name)


    def __read_ogb_graph(self, dir_name: str):
        path = os.path.join(self.__ROOT_DIR, dir_name, 'raw', 'edge.csv')

        if not os.path.exists(path):
            path_gz = path + '.gz'

            with gzip.open(path_gz, 'rb') as f_in:
                with open(path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        delimiter = ','

        return path, delimiter # nx.read_edgelist(path, delimiter=',', nodetype=int)
