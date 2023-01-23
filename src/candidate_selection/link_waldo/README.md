# LinkWaldo
A copy of [GemsLab/LinkWaldo](https://github.com/GemsLab/LinkWaldo) commit id 1b722620c6edb890d7c43cbfaf030ee022e6ef14.

# Changes
- We have implemented our own `link_waldo.py` to fit our benchmarks.
- `link_waldo_selector.py` has been altered to 
    1. commented out row 116 assigning true positives, as we do not use those.
- `emb.py` has been altered to 
    1. Not use `edgelist` or `test_path` or take them in constructor
    2. `edgelist` is generated using the input graph
- `netmf_emb.py` has been altered to 
    1. Use the input graph G as embeddings graph (as our graphs do not need to convert string labels to int labels)
