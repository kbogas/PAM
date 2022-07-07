import functools
import os
import pickle
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import scipy
import tqdm
from ogb.linkproppred import LinkPropPredDataset
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, lil_matrix
from sympy import nextprime, primefactors

from utils import (
    get_prime_map_from_rel,
    get_primefactors,
    get_sparsity,
    load_data,
    set_all_seeds,
)

set_all_seeds(42)


# Path to projects
project_to_path = {
    "codex-s": "./data/codex-s/",
    "WN18RR": "./data/WN18RR/",
    "FB15k-237": "./data/FB15k-237/",
    "YAGO3-10-DR": "./data/YAGO3-10-DR/",
    "hetionet": "./data/Hetionet/hetionet-v1.0-edges.tsv",
    "ogbl-wikikg2": "path",
}

res = []
max_order = 5

for project_name, path in project_to_path.items():
    print(project_name)

    # speciic loaders for hetionet
    if project_name == "hetionet":
        df_train = pd.read_csv(path, sep="\t")
        df_train.dropna(inplace=True)
        df_train.columns = ["head", "rel", "tail"]
    # and ogbl-wiki-kg2
    elif project_name == "ogbl-wikikg2":
        dataset = LinkPropPredDataset(name="ogbl-wikikg2")
        graph = dataset[0]
        df_train = pd.DataFrame(graph["edge_index"]).T
        df_train.columns = ["head", "tail"]
        df_train["rel"] = graph["edge_reltype"]
        df_train.shape
    else:
        df_train, _, _, _ = load_data(path, project_name, add_inverse_edges="NO")

    # Statistics
    unique_rels = sorted(list(df_train["rel"].unique()))
    unique_nodes = sorted(
        set(df_train["head"].values.tolist() + df_train["tail"].values.tolist())
    )
    print(
        f"# of unique rels: {len(unique_rels)} \t | # of unique nodes: {len(unique_nodes)}"
    )

    node2id = {}
    id2node = {}
    for i, node in enumerate(unique_nodes):
        node2id[node] = i
        id2node[i] = node

    time_s = time.time()

    # Map the relations to primes
    rel2id, id2rel = get_prime_map_from_rel(
        unique_rels, starting_value=2, spacing_strategy="step_1"
    )

    # Create the adjacency matrix
    df_train["rel_mapped"] = df_train["rel"].map(rel2id)
    df_train["head_mapped"] = df_train["head"].map(node2id)
    df_train["tail_mapped"] = df_train["tail"].map(node2id)
    A_big = csr_matrix(
        (df_train["rel_mapped"], (df_train["head_mapped"], df_train["tail_mapped"])),
        shape=(len(unique_nodes), len(unique_nodes)),
    )

    # Calculate sparsity
    sparsity = get_sparsity(A_big)
    print(A_big.shape, f"Sparsity: {sparsity:.2f} %")

    time_prev = time.time()
    time_setup = time_prev - time_s
    print(f"Total setup: {time_setup:.5f} secs ({time_setup/60:.2f} mins)")

    # Generate the PAM^k matrices
    power_A = [A_big]
    for ii in range(1, max_order):
        updated_power = power_A[-1] * A_big
        updated_power.sort_indices()
        updated_power.eliminate_zeros()
        power_A.append(updated_power)
        print(
            f"Sparsity {ii + 1}-hop: {100 * (1 - updated_power.nnz/(updated_power.shape[0]**2)):.2f} %"
        )

    time_stop = time.time()
    time_calc = time_stop - time_prev
    print(f"A^k calc time: {time_calc:.5f} secs ({time_calc/60:.2f} mins)")

    time_all = time_stop - time_s
    print(f"All time: {time_all:.5f} secs ({time_all/60:.2f} mins)")
    res.append(
        {
            "dataset": project_name,
            "nodes": len(unique_nodes),
            "rels": len(unique_rels),
            "edges": df_train.shape[0],
            "sparsity_start": sparsity,
            "sparsity_end": get_sparsity(power_A[-1]),
            "setup_time": time_setup,
            "cacl_time": time_calc,
            "total_time": time_all,
            # 'total_time_plus': time_setup + time_calc
        }
    )
    print(res[-1])
    # break
    print("\n\n")

res = pd.DataFrame(res)
print(res.to_string())
