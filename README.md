# Distributed Approximate Personalized PageRank (DAPPR)
Source code for the masters thesis "Finding Candidate Node Pairs for Link Prediction at Scale" by Kalkan, Filip and Hambiralovic, Mahir.

## About
This repository includes a benchmarking framework for testing candidate selection algorithms, along with implementations of some candidate selection algorithms, one of them being DAPPR.

A number of of datasets of varying domains and sizes are available for testing. Tools for vizualizing the results are available in `src/graphs.ipynb`.

## Setup
The project requires Python 3.10.

Using conda, install all dependencies in `environment.yml`.

## Quick Start
Once dependencies are installed, you can try running the benchmarks in `multi.ipynb`.