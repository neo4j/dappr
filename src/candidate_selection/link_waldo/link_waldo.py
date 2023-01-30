import random
import numpy as np
import networkx as nx
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/src')
from candidate_selection.link_waldo.src.emb import Emb
from candidate_selection.cadidate_selection_algorithm import CandidateSelectionAlgorithm, EdgeList
from netmf_emb import NetMF
from aa_emb import AA
from bine_emb import BiNE
from lapm_selector import LaPMSelector
from dg_selector import DGSelector
from mg_selector import MGSelector
from dg_bailout_selector import DGBailoutSelector
from mg_bailout_selector import MGBailoutSelector
from cn_selector import CNSelector
from js_selector import JSSelector
from aa_selector import AASelector
from sg_selector import SGSelector
from cg_selector import CGSelector
from bagging_ensemble import BaggingEnsemble
import networkx as nx
import argparse
import os



class LinkWaldo(CandidateSelectionAlgorithm):
    __G: nx.Graph
    __n_scanned: int
    candidate_links: EdgeList
    embeddings: Emb

    def __init__(
        self, 
        G: nx.Graph, 
        k: int, 
        test_edges: EdgeList,
        verbose=True,

        method: str='LinkWaldo', 
        embedding_method: str='netmf2',  
        force_emb: bool=False,  
        sampling_method: str='static',  
        percent_test: float=20.0,  
        seed: str=0,  
        bipartite: bool=False,  
        exact_search_tolerance: int=25000000,  
        output_override: str=None,  
        num_groups: int=None,  
        num_groups_alt: int=None,  
        DG: bool=True,  
        SG: bool=True,  
        CG: bool=True,  
        bailout_tol: float=0.5,  
        bag_epsilon: float=1.0,  
        skip_output: bool=False,  
    ):
        print("LinkWaldo predicting", k, "links")
        
        self.input_params = locals()
        self.exclude_params = ["self", "G", "test_edges"]

        self.__verbose = verbose
        self.__k = k
        self.__G = G
        self.__n_scanned = 0
        self.test_edges = test_edges

        self.__method = method
        self.__embedding_method = embedding_method
        self.__force_emb = force_emb
        self.__sampling_method = sampling_method
        self.__percent_test = percent_test
        self.__seed = seed
        self.__bipartite = bipartite
        self.__exact_search_tolerance = exact_search_tolerance
        self.__output_override = output_override
        self.__num_groups = num_groups
        self.__num_groups_alt = num_groups_alt
        self.__DG = DG
        self.__SG = SG
        self.__CG = CG
        self.__bailout_tol = bailout_tol
        self.__bag_epsilon = bag_epsilon
        self.__skip_output = skip_output

        # relabel_dict = {}
        # for n in self.__G.nodes():
        #     relabel_dict[n] = str(n)
        # nx.relabel_nodes(self.__G, relabel_dict)
        

        self.embeddings = None
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.ROOT_DIR, 'output/{}/'.format(self.__sampling_method))
        self.emb_path = os.path.join(self.output_dir, '{}_{}_{}_seed_{}_id_{}.emb'.format(self.__embedding_method, self.__G, self.__percent_test, seed, random.randint(0,1000)))


        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if not self.__output_override:
            self.output_path = os.path.join(self.output_dir, '{}_{}_{}_{}_{}_{}_{}_k_{}.txt'.format(self.__method, self.__G, self.__embedding_method, self.__percent_test, self.__exact_search_tolerance, self.__bailout_tol, seed,  self.__k))
        else:
            self.output_path = self.__output_override
    

    def get_predicted_links(self) -> EdgeList:
        """Returns a list of tuples (v, u) where v is a node in the graph and u is a predicted link to v."""
        return self.candidate_links

    def get_n_scanned(self) -> int:
        return len(self.candidate_links) # LW has actually not run any real scans, but all candidates will need to run LP.

    def generate_embeddings(self) -> Emb:
        seed = self.__seed
        if not self.__bipartite:
            G = nx.Graph()
            for a, b in self.__G.edges():
                G.add_edge(str(a), str(b))
            self.__G = G
        if self.__bipartite:

            G = nx.Graph()
            for a, b in self.__G.edges():
                G.add_node(str(a), bipartite=self.__G.nodes[a]["bipartite"])
                G.add_node(str(b), bipartite=self.__G.nodes[b]["bipartite"])
                G.add_edge(str(a), str(b))
            self.__G = G
         
        if self.__embedding_method == 'netmf1':
            embeddings = NetMF(self.__embedding_method, self.test_edges, self.emb_path, self.__G, normalize=True, window_size=1)
        elif self.__embedding_method == 'netmf2':
            embeddings = NetMF(self.__embedding_method, self.test_edges, self.emb_path, self.__G, normalize=True, window_size=2)
        elif self.__embedding_method == 'bine':
            embeddings = BiNE(self.__embedding_method, self.test_edges, self.emb_path, self.__G, normalize=True)
        elif self.__embedding_method == 'aa':
            embeddings = AA(self.__embedding_method, self.test_edges, self.emb_path, self.__G, normalize=True)
        if self.__force_emb or not os.path.exists(self.emb_path):
            if os.path.exists(self.emb_path.replace('.emb', '_nodeX.npy')):
                os.remove(self.emb_path.replace('.emb', '_nodeX.npy'))
            embeddings.run(self.__G)

        embeddings.load_data(load_embeddings=True)
        
        self.embeddings = embeddings
        
        return embeddings


    def run(self):
        if self.embeddings == None:
            self.generate_embeddings()

        """An extension of original main.py from LinkWaldo, adapted for our benchmarks"""
        if self.__method in {'lapm'}:
            sel = LaPMSelector(self.__method, self.__G,  self.__k, self.embeddings, self.output_path, seed=self.__seed, bipartite=self.__bipartite)
            load_embeddings = True
        elif self.__method in {'cn'}:
            sel = CNSelector(self.__method, self.__G,  self.__k, self.embeddings, self.output_path, seed=self.__seed, bipartite=self.__bipartite)
            load_embeddings = False
        elif self.__method in {'js'}:
            sel = JSSelector(self.__method, self.__G,  self.__k, self.embeddings, self.output_path, seed=self.__seed, bipartite=self.__bipartite)
            load_embeddings = False
        elif self.__method in {'aa'}:
            sel = AASelector(self.__method, self.__G,  self.__k, self.embeddings, self.output_path, seed=self.__seed, bipartite=self.__bipartite)
            load_embeddings = False
        elif self.__method in {'nmf+bag'}:
            sel = BaggingEnsemble(self.__method, self.__G,  self.__k, self.embeddings, self.output_path, seed=self.__seed, bipartite=self.__bipartite)
            load_embeddings = False
        elif self.__method == 'LinkWaldo':
            num_groupings = 0
            if self.__DG:
                num_groupings += 1
            if self.__SG:
                num_groupings += 1
            if self.__CG:
                num_groupings += 1

            if num_groupings > 1:
                if self.__bailout_tol > 0.0:
                    sel = MGBailoutSelector(self.__method, self.__G,  self.__k, self.embeddings, self.output_path, DG=self.__DG, SG=self.__SG, CG=self.__CG, exact_search_tolerance=self.__exact_search_tolerance, seed=self.__seed, bipartite=self.__bipartite)
                else:
                    sel = MGSelector(self.__method, self.__G,  self.__k, self.embeddings, self.output_path, DG=self.__DG, SG=self.__SG, CG=self.__CG, exact_search_tolerance=self.__exact_search_tolerance, seed=self.__seed, bipartite=self.__bipartite)
            else:
                if self.__DG and self.__bailout_tol > 0.0:
                    sel = DGBailoutSelector(self.__method, self.__G,  self.__k, self.embeddings, self.output_path, DG=self.__DG, SG=self.__SG, CG=self.__CG, exact_search_tolerance=self.__exact_search_tolerance, seed=self.__seed, bipartite=self.__bipartite)
                elif self.__DG:
                    sel = DGSelector(self.__method, self.__G,  self.__k, self.embeddings, self.output_path, DG=self.__DG, SG=self.__SG, CG=self.__CG, exact_search_tolerance=self.__exact_search_tolerance, seed=self.__seed, bipartite=self.__bipartite)
                elif self.__SG:
                    sel = SGSelector(self.__method, self.__G,  self.__k, self.embeddings, self.output_path, DG=self.__DG, SG=self.__SG, CG=self.__CG, exact_search_tolerance=self.__exact_search_tolerance, seed=self.__seed, bipartite=self.__bipartite)
                elif self.__CG:
                    sel = CGSelector(self.__method, self.__G,  self.__k, self.embeddings, self.output_path, DG=self.__DG, SG=self.__SG, CG=self.__CG, exact_search_tolerance=self.__exact_search_tolerance, seed=self.__seed, bipartite=self.__bipartite)
            load_embeddings = True

        sel.num_groups = self.__num_groups
        sel.num_groups_alt = self.__num_groups_alt
        sel.bailout_tol = self.__bailout_tol
        sel.bag_epsilon = self.__bag_epsilon
        sel.skip_output = self.__skip_output
        
        verbosity = 10000 if self.__verbose else 0
        _time = sel.select(verbosity)
        self.sel = sel
        
        # self.__predicted_links = list(sel.pairs)
        self.candidate_links = list()
        for u, v in sel.pairs:
            u = int(u.name)
            v = int(v.name)
            if u <= v:
                self.candidate_links.append((u, v))
            else:
                self.candidate_links.append((v, u))

        # sel.write_res(_time)
        # if not self.__skip_output:
        #     sel.write()

        return self.candidate_links