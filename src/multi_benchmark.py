import datetime
from itertools import cycle, repeat
import sys
import os


sys.path.append('candidate_selection')
sys.path.append('./')

from candidate_selection.random_walk_restarts.dappr import Dappr
from candidate_selection.link_waldo.link_waldo import LinkWaldo
from candidate_selection.random_walk_restarts.random_walk_restarts_pregel import RandomWalkRestartsPregel
from candidate_selection.random_walk_restarts.random_walk_restarts_pregel_sbm import RandomWalkRestartsPregelSBM

import collections.abc # Manually overwrite import due to error in LinkWaldo.Bine.mode.Graph - incompatible with Python 3.10
collections.Iterable = collections.abc.Iterable
import numpy as np
import networkx as nx
import traceback
from dataset.dataset_reader import DatasetReader, Datasets, temporal_datasets
from prediction_models.logistic_regression_trainer import LogisticRegressionTrainer
from candidate_selection.benchmarking.benchmark import Benchmark
from candidate_selection.knn.knn import Knn
from candidate_selection.knn.knn_simple import KnnSimple
from candidate_selection.bfs.bfs import Bfs
from candidate_selection.personalized_page_rank.personalized_page_rank import PersonalizedPageRank
from candidate_selection.random_walk_restarts.random_walk_restarts import RandomWalkRestarts
from candidate_selection.random_walk_restarts.random_walk_restarts_global_pool import RandomWalkRestartsGlobalPool
from candidate_selection.random_walk_restarts.random_walk_restarts_sbm import RandomWalkRestartsSBM
import multiprocessing
import pickle
from prediction_models.oracle_fn import select_oracle
from candidate_selection.cadidate_selection_algorithm import CandidateSelectionAlgorithm

# %% [markdown]
# ## Settings

# %% [markdown]
# ### Dataset Selection

# %%
def multirun(
    datasets: Datasets, 
    range_to_iterate: range, 
    n_iterations: int,
    out_file_name: str,
    algorithm_options
):
    print("Starting", datetime.datetime.now())
    print("algorithm_options", algorithm_options)

    for DATASET in datasets:


        # %% [markdown]
        # ### Parameter Selection

        # %%
        VALIDATION_FRACTION = 0.1
        TRAIN_FRACTION = 0.2
        TEST_FRACTION = 0.2

        SEED = 42
        FORCE_TRAIN_CLASSIFIER = True
        N_TRIALS = 1
        ORACLE_TYPE = 'puro+perfect' # TODO: Change
        EMBEDDINGS_TYPE="netmf2"

        # %% [markdown]
        # ## Setup

        # %% [markdown]
        # ### Dataset Preparation and Splitting

        # %%
        (
        G,
        G_test,
        positive_train_edges,
        negative_train_edges,
        positive_validation_edges,
        negative_validation_edges,
        positive_test_edges,
        is_bipartite) = DatasetReader().read_and_split(DATASET, TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION)
        G.name = DATASET.name
        print(f'Loaded {DATASET} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges')
        print(f'Number of hidden edges in test set: {len(positive_test_edges)} ({int(100 * len(positive_test_edges) / G.number_of_edges())}% of all edges)')

        if nx.algorithms.is_bipartite(G):
            print('Graph is bipartite')
        # %% [markdown]
        # ## Create embeddings and train classifier

        # %%            
        # Relabel node_labels to ints
        G = nx.convert_node_labels_to_integers(G, label_attribute="old_id")
        G_test = nx.convert_node_labels_to_integers(G_test, label_attribute="old_id")

        node_labels = dict()
        for node_id, node_data in list(G.nodes(data=True)):
            node_labels[node_data["old_id"]] = node_id

        positive_train_edges = [(node_labels[str(u)], node_labels[str(v)]) for u, v in positive_train_edges]
        negative_train_edges = [(node_labels[str(u)], node_labels[str(v)]) for u, v in negative_train_edges]
        positive_validation_edges = [(node_labels[str(u)], node_labels[str(v)]) for u, v in positive_validation_edges]
        negative_validation_edges = [(node_labels[str(u)], node_labels[str(v)]) for u, v in negative_validation_edges]
        positive_test_edges = [(node_labels[str(u)], node_labels[str(v)]) for u, v in positive_test_edges]

        try:
            if algorithm_options["run_knn"]:
                pickle_path = f"pickles/{DATASET.name}_{EMBEDDINGS_TYPE}_logistic_regression_{G}.pkl"
                if FORCE_TRAIN_CLASSIFIER or not os.path.exists(pickle_path):

                    # Generate embeddings
                    G_link_waldo_runner = LinkWaldo(G, None, positive_test_edges, seed=SEED, embedding_method="netmf2", verbose=False)
                    G_link_waldo_runner.generate_embeddings()

                    embeddings = [0] * len(G_link_waldo_runner.embeddings.node_names_to_nodes)
                    for node_id, node_data in list(G.nodes(data=True)):
                        embeddings[node_id] = G_link_waldo_runner.embeddings.node_names_to_nodes[str(node_id)].emb
                    
                    # Train clf model
                    (clf, clf_best_accuracy) = LogisticRegressionTrainer(
                        SEED,
                        embeddings,
                        positive_train_edges,
                        negative_train_edges,
                        positive_validation_edges,
                        negative_validation_edges,
                        N_TRIALS
                    ).train_model()
                    
                    pickle.dump(
                        {"clf": clf, "best_accuracy": clf_best_accuracy, "embeddings": embeddings, "node_labels": node_labels},
                        open(pickle_path, 'wb')
                    )
                    
                    print("Pickled clf with accuracy:", clf_best_accuracy)
                    
                else:
                    pickled_dict = pickle.load(open(pickle_path, 'rb'))
                    clf = pickled_dict["clf"]
                    clf_best_accuracy = pickled_dict["best_accuracy"]
                    embeddings = pickled_dict["embeddings"]
                    node_labels = pickled_dict["node_labels"]
                    nx.relabel.relabel_nodes(G, node_labels, copy=False)
                    nx.relabel.relabel_nodes(G_test, node_labels, copy=False)

                    positive_train_edges = [(node_labels[str(u)], node_labels[str(v)]) for u, v in positive_train_edges]
                    negative_train_edges = [(node_labels[str(u)], node_labels[str(v)]) for u, v in negative_train_edges]
                    positive_validation_edges = [(node_labels[str(u)], node_labels[str(v)]) for u, v in positive_validation_edges]
                    negative_validation_edges = [(node_labels[str(u)], node_labels[str(v)]) for u, v in negative_validation_edges]
                    positive_test_edges = [(node_labels[str(u)], node_labels[str(v)]) for u, v in positive_test_edges]
                    
                    print("Loaded clf with accuracy:", clf_best_accuracy)

                    del pickled_dict # Save some memory

                oracle_fn = select_oracle(ORACLE_TYPE, clf, embeddings, positive_test_edges)
        except Exception as e:
            print("Failed to generate embs", e)
            traceback.print_exc()

        knn_prev_scan_rate = 0.0
        knn_simple_prev_scan_rate = 0.0

        # print("filtered", list(filter(lambda x: x[1]["old_id"][0] == "u", list(G.nodes(data=True))))[:10])
        # print("filtered", list(filter(lambda x: x[1]["bipartite"] == 0, list(G.nodes(data=True))[:10])))
        # print('nodes', list(G.nodes(data=True))[:10])

        for h in range_to_iterate:
            try:
                print("Running h:", h, "/", range_to_iterate.stop-1, " or dataset:", DATASET)

                # %% [markdown]
                # ### Benchmarking Options

                
                edges_to_find = len(positive_test_edges) * h
                print(f'Number of edges to find: {edges_to_find} ({h} times number of hidden edges)')

                # Since KNN risks running on more edges than brute force, we should stop running it at some point
                for _ in range(n_iterations):
                    # %% [markdown]
                    # ### Simple KNN

                    if algorithm_options["run_knn"]:
                        if knn_prev_scan_rate < 1:
                            knn = Knn(G_test, edges_to_find, oracle_fn, clf_best_accuracy, sample_rate=0.75, delta=0.0001, p_random_neighbors=1.0, max_iterations=40, parallel=False, verbose=False) # Parallel not recommended
                            knn_benchmark = Benchmark(knn, G, G_test, h, positive_test_edges, DATASET.name)
                            knn_benchmark.run()
                            knn_benchmark.write_results_to_file(out_file_name)
                            knn_prev_scan_rate = knn_benchmark.get_metrics()["scan_rate"]
                        else:
                            print("Skipping KNN because scan rate is greater than 1")

                    if algorithm_options["run_knn_simple"]:
                        K = int(np.ceil(edges_to_find / G_test.number_of_nodes()))
                        if knn_simple_prev_scan_rate < 1:
                            knn_simple = KnnSimple(G_test, K, oracle_fn, clf_best_accuracy, delta=0.1, max_iterations=20, verbose=False)
                            knn_simple_benchmark = Benchmark(knn_simple, G, G_test, h, positive_test_edges, DATASET.name)
                            knn_simple_benchmark.run()
                            knn_simple_benchmark.write_results_to_file(out_file_name)
                            knn_simple_prev_scan_rate = knn_simple_benchmark.get_metrics()["scan_rate"]
                        else:
                            print("Skipping KNN_simple because scan rate is greater than 1")

                    if algorithm_options["run_link_waldo"]:
                        # %%
                        embedding = EMBEDDINGS_TYPE
                        if is_bipartite:
                            embedding = "bine"
                            
                        if DATASET.value[0] in temporal_datasets:
                            sampling_method = "temporal"
                        else:
                            sampling_method = "static"
                        lw = LinkWaldo(G_test, edges_to_find, positive_test_edges, bipartite=is_bipartite, seed=SEED, embedding_method=embedding, verbose=False, sampling_method=sampling_method)
                        lw.generate_embeddings() # Do not include in benchmark time
                        lw_benchmark = Benchmark(lw, G, G_test, h, positive_test_edges, DATASET.name)
                        lw_benchmark.run()
                        lw_benchmark.write_results_to_file(out_file_name)

                    if algorithm_options["run_bfs"]:
                        # %%
                        K = int(np.ceil(edges_to_find / G_test.number_of_nodes()))
                        bfs = Bfs(G_test, K, parallel=False, max_depth=4) # Parallel not recommended
                        bfs_benchmark = Benchmark(bfs, G, G_test, h, positive_test_edges, DATASET.name)
                        bfs_benchmark.run()
                        bfs_benchmark.write_results_to_file(out_file_name)


                    if algorithm_options["run_personalized_page_rank"]:
                        K = int(np.ceil(edges_to_find / G_test.number_of_nodes()))
                        pers_page_rank = PersonalizedPageRank(G_test, K, parallel=True)
                        pers_page_rank_benchmark = Benchmark(pers_page_rank, G, G_test, h, positive_test_edges, DATASET.name)
                        pers_page_rank_benchmark.run()
                        pers_page_rank_benchmark.write_results_to_file(out_file_name)
                        print("ppr", pers_page_rank_benchmark.get_metrics())

                    
                    max_steps = int(1000 * (G.number_of_nodes() ** 0.15))
                    # max_steps = 9
                    if algorithm_options["run_random_walk_restarts"]:
                        random_walk = RandomWalkRestarts(G_test, K, parallel=True, max_steps=max_steps, alpha=0.8)
                        random_walk_benchmark = Benchmark(random_walk, G, G_test, h, positive_test_edges, DATASET.name)
                        random_walk_benchmark.run()
                        random_walk_benchmark.write_results_to_file(out_file_name)

                    if algorithm_options["run_random_walk_restarts_global_pool"]:
                        random_walk_global_pool = RandomWalkRestartsGlobalPool(G_test, edges_to_find, parallel=True, max_steps=max_steps, alpha=0.8)
                        random_walk_global_pool_benchmark = Benchmark(random_walk_global_pool, G, G_test, h, positive_test_edges, DATASET.name)
                        random_walk_global_pool_benchmark.run()
                        random_walk_global_pool_benchmark.write_results_to_file(out_file_name)

                    if algorithm_options["run_random_walk_restarts_sbm"]:
                        random_walk_sbm = RandomWalkRestartsSBM(G_test, edges_to_find, parallel=True, max_steps=max_steps, alpha=0.8, bipartite=is_bipartite)
                        random_walk_sbm_benchmark = Benchmark(random_walk_sbm, G, G_test, h, positive_test_edges, DATASET.name)
                        random_walk_sbm_benchmark.run()
                        random_walk_sbm_benchmark.write_results_to_file(out_file_name)

                    if algorithm_options["run_dappr"]:
                        dappr = Dappr(G_test, edges_to_find, parallel=True, alpha=0.8, bipartite=is_bipartite)
                        dappr_benchmark = Benchmark(dappr, G, G_test, h, positive_test_edges, DATASET.name)
                        dappr_benchmark.run()
                        dappr_benchmark.write_results_to_file(out_file_name)
                        print("dappr", dappr_benchmark.get_metrics())
                    

                    if algorithm_options["run_random_walk_restarts_pregel"]:
                        random_walk_pregel = RandomWalkRestartsPregel(G_test, edges_to_find, parallel=True, alpha=0.8, bipartite=is_bipartite)
                        random_walk_pregel_benchmark = Benchmark(random_walk_pregel, G, G_test, h, positive_test_edges, DATASET.name)
                        random_walk_pregel_benchmark.run()
                        random_walk_pregel_benchmark.write_results_to_file(out_file_name)
                        print("pregel", random_walk_pregel_benchmark.get_metrics())

                    if algorithm_options["run_random_walk_restarts_pregel_sbm"]:
                        random_walk_pregel_sbm = RandomWalkRestartsPregelSBM(G_test, edges_to_find, parallel=True, alpha=0.98, bipartite=is_bipartite)
                        random_walk_pregel_sbm_benchmark = Benchmark(random_walk_pregel_sbm, G, G_test, h, positive_test_edges, DATASET.name)
                        random_walk_pregel_sbm_benchmark.run()
                        random_walk_pregel_sbm_benchmark.write_results_to_file(out_file_name)

                    # %%
                    if algorithm_options["run_combo"]:

                        class LW_with_RWR(CandidateSelectionAlgorithm):

                            def __init__(self, union: bool, lw: LinkWaldo, rwr: RandomWalkRestarts):
                                self.union = union

                                lw_links = set(lw.get_unique_predicted_links())
                                rwr_links = set(rwr.get_unique_predicted_links())
                                
                                if union:
                                    self.predicted_links = lw_links.union(rwr_links)
                                    
                                else:
                                    self.predicted_links = lw_links.intersection(rwr_links)
                            
                            def get_name(self):
                                return self.__class__.__name__ + ('_union' if self.union else '_intersection')

                            def run(self):
                                return self.predicted_links

                            def get_predicted_links(self):
                                return self.predicted_links

                            def get_n_scanned(self) -> int:
                                return len(self.predicted_links)

                            def get_input_params(self) -> dict:
                                return {**lw_benchmark.algorithm.get_input_params(), **random_walk_pregel_benchmark.algorithm.get_input_params()} # Merge dicts
                            
                            def get_unique_predicted_links(self):
                                return self.predicted_links

                        merged_benchmark_union = Benchmark(LW_with_RWR(G, True, lw, random_walk_pregel), G_test, h, positive_test_edges, DATASET.name)
                        merged_benchmark_union.run()
                        merged_benchmark_union.write_results_to_file(out_file_name)

                        merged_benchmark_intersection = Benchmark(LW_with_RWR(G, False, lw, random_walk_pregel), G_test, h, positive_test_edges, DATASET.name)
                        merged_benchmark_intersection.run()
                        merged_benchmark_intersection.write_results_to_file(out_file_name)

                    # %% [markdown]
                    # ## Delete LinkWaldo embeddings

                    # %%
                    # path = os.path.join(os.getcwd() + "/candidate_selection/link_waldo/output/static")
                    # for file_name in os.listdir(path):
                    #     # construct full file path
                    #     file = os.path.join(path, file_name)
                    #     if os.path.isfile(file):
                    #         os.remove(file)
                    #         print('Deleted file:', file)
            except Exception as e:
                print("Failed iteration", h, e)
                traceback.print_exc()



if __name__ == "__main__":
    datasets = [
            # Datasets.DBLP,
            # Datasets.FACEBOOK1,
            # Datasets.HS_PROTEIN,
            # Datasets.YEAST,
            # Datasets.POWER,
            # Datasets.US_AIR,
            # Datasets.ROUTER,
            # Datasets.FACEBOOK2,
            # Datasets.MATH_OVERFLOW,
            # Datasets.MOVIE_LENS,
            # Datasets.REDDIT,
            # Datasets.DIGG,
            # Datasets.ENRON,
            # Datasets.OGB_DDI,
            # Datasets.OGB_PPA,
            # Datasets.EPINIONS,
            # Datasets.ARXIV,
            # Datasets.EPINIONS,
            # Datasets.OGB_COLLAB,
            # Datasets.OGB_CITATION2,
            # Datasets.AMAZON,
            Datasets.ROADNET_PA,
            # Datasets.AS_SKITTER,
            # Datasets.OGB_CITATION2,
            # Datasets.ARXIV_CONDMAT,
            # Datasets.SV_WIKIQUOTE_EDITS,
        ]
    ranges = range(1,11)
    filename = "benchmarks-parallel.txt"
    run_linear = True # One at a time
    run_parallel = False # Parallel

    # Run RWR+BFS sequentially
    if run_linear:
        multirun(
            datasets, 
            ranges, 
            1, 
            filename,
            {
                "run_personalized_page_rank": False,
                "run_knn_simple": False,
                "run_random_walk_restarts": False,
                "run_random_walk_restarts_global_pool": False,
                "run_random_walk_restarts_sbm": False,
                "run_random_walk_restarts_pregel_sbm": False,

                "run_bfs": False,
                "run_random_walk_restarts_pregel": False,
                "run_dappr": False,
                "run_knn": False,
                "run_link_waldo": True,
                "run_combo": False,
            }
        )

    
    # # Run others in parallel
    if run_parallel:
        for dataset in datasets:
            pool = multiprocessing.Pool(10)

            pool.starmap(multirun,
                list(
                    zip(
                        repeat([dataset]), 
                        [range(i, i+1) for i in ranges], 
                        repeat(1), 
                        repeat(filename),
                        repeat({
                            "run_personalized_page_rank": False,
                            "run_knn_simple": False,
                            "run_random_walk_restarts": False,
                            "run_random_walk_restarts_global_pool": False,
                            "run_random_walk_restarts_sbm": False,
                            "run_random_walk_restarts_pregel_sbm": False,

                            "run_bfs": False,
                            "run_random_walk_restarts_pregel": False,
                            "run_dappr": False,
                            "run_knn": False,
                            "run_link_waldo": True,
                            "run_combo": False,
                        })
                    )
                )
            )
            
            pool.close()
            pool.join()
