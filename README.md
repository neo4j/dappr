# Distributed Approximate Personalized PageRank (DAPPR)
Source code for the masters thesis "Finding Candidate Node Pairs for Link Prediction at Scale" by Kalkan, Filip and Hambiralovic, Mahir.

## About
This repository includes a benchmarking framework for testing candidate selection algorithms, along with implementations of some candidate selection algorithms, one of them being DAPPR.

A number of of datasets of varying domains and sizes are available for testing.

## Setup
The project requires Python 3.10.

Using conda, install all dependencies in `environment.yml`.

## Quick Start
### Compare Algorithms
Once dependencies are installed, you can try running the benchmarks in `multi.ipynb`.

Tools for vizualizing the results are available in `src/graphs.ipynb`.
### Run DAPPR
DAPPR can be run in `src/main.py`. Example:

```
python main.py --edgelist-path "dataset/static/yeast.txt" --c 1000 --lambd 40 --parallel y -out candidate_node_pairs.csv
```

This example runs the `YEAST` dataset and outputs the candidate node pairs to a csv file named `candidate_node_pairs.csv`.