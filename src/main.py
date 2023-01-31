from candidate_selection.dappr.dappr import Dappr
import argparse
import os
import networkx as nx

def parse_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser(description='Run DAPPR to extract candidate node pairs.')
    parser.add_argument('--edgelist-path', type=str, required=True, help="Relative path to edgelist representing the graph.")
    parser.add_argument('--c', '-c', type=int, required=True, help="Max number of candidate pairs to output.")
    parser.add_argument('--lambd', '-l', type=int, required=False, default=40, help="Smaller lambda increases approximation error, but yields faster run time. Increase lambda if too few results are given.")
    parser.add_argument('--alpha', '-a', type=float, required=False, default=0.8, help="Non-restart probability.")
    parser.add_argument('--epsilon', '-e', type=float, required=False, default=0.05, help="Error threshold.")
    parser.add_argument('--parallel', '-p', type=str2bool, required=False, default=True, help="Run parallel processes.")
    parser.add_argument('--verbose', '-v', type=str2bool, required=False, default=False, help="Verbose.")
    parser.add_argument('--bipartite', '-b', type=str2bool, required=False, default=False, help="Whether the evaluated graph is bipartite or not.")
    parser.add_argument('--out', '-o', type=str, required=True, help="Relative path to output file.")
    return parser.parse_args()

def main(args):
    cwd = os.getcwd()
    edgelist_path = os.path.join(cwd, args.edgelist_path)
    G = nx.read_edgelist(edgelist_path)
    dappr = Dappr(G, args.c, args.epsilon, args.alpha, args.parallel, args.verbose, args.bipartite, args.lambd)
    candidate_node_pairs = dappr.run()
    output_path = os.path.join(cwd, args.out)
    if '.' not in output_path:
        output_path = output_path + '.csv'
    with open(output_path, 'w') as f:
        for candidate_node_pair in candidate_node_pairs:
            f.write(f"{str(candidate_node_pair[0])},{str(candidate_node_pair[1])}\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)