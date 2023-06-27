import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sympy import nextprime

from utils import get_sparsity


def get_prime_map_from_rel(
    list_of_rels: list,
    starting_value: int = 1,
    spacing_strategy: str = "step_10",
    add_inverse_edges: bool = False,
) -> tuple[dict, dict]:
    """
    Helper function that given a list of relations returns the mappings to and from the
    prime numbers used.

    Different strategies to map the numbers are available.
    "step_X", increases the step between two prime numbers by adding X to the current prime
    "factor_X", increases the step between two prime numbers by multiplying the current prime with X

    Args:
        list_of_rels (list): iterable, contains a list of the relations that need to be mapped.
        starting_value (int, optional): Starting value of the primes. Defaults to 1.
        spacing_strategy (str, optional):  Spacing strategy for the primes. Defaults to "step_1".
        add_inverse_edges (bool, optional):  Whether to create mapping for inverse edges. Defaults to False.

    Returns:
        rel2prime: dict, relation to prime dictionary e.g. {"rel1":2}.
        prime2rel: dict, prime to relation dictionary e.g. {2:"rel1"}.
    """
    # add inverse edges if needed
    if add_inverse_edges:
        list_of_rels = [str(relid) for relid in list_of_rels] + [
            str(relid) + "__INV" for relid in list_of_rels
        ]
    else:
        list_of_rels = [str(relid) for relid in list_of_rels]

    # Initialize dicts
    rel2prime = {}
    prime2rel = {}
    # Starting value for finding the next prime
    current_int = starting_value
    # Map each relation id to the next available prime according to the strategy used
    for relid in list_of_rels:
        cur_prime = int(nextprime(current_int))  # type: ignore
        rel2prime[relid] = cur_prime
        prime2rel[cur_prime] = relid
        if "step" in spacing_strategy:
            step = float(spacing_strategy.split("_")[1])
            current_int = cur_prime + step
        elif "factor" in spacing_strategy:
            factor = float(spacing_strategy.split("_")[1])
            current_int = cur_prime * factor
        else:
            raise NotImplementedError(
                f"Spacing strategy : {spacing_strategy}  not understood!"
            )
    return rel2prime, prime2rel


def create_pam_matrices(
    df_train: pd.DataFrame,
    max_num_hops: int = 5,
    use_log: bool = True,
    spacing_strategy: str = "step_10",
    break_with_sparsity_threshold: int = -1,
) -> tuple[csr_matrix, list[csr_matrix], dict, dict]:
    """Helper function that creates the pam matrices.

    Args:
        df_train (pd.DataFrame): The triples in the form of a pd.DataFrame with columns
        (head, rel, tail).
        max_num_hops (int, optional): The maximum order for the PAMs (i.e. the k-hops).
        Defaults to 5.
        use_log (bool, optional): Whether to use log of primes for numerical stability.
        Defaults to True.
        spacing_strategy (str, optional): he spacing strategy as mentioned in get_prime_map_from_rel.
        Defaults to "step_10".
        break_with_sparsity_threshold (int, optional): The percentage of sparsity that is not accepted.
        If one of the k-hop PAMs has lower sparsity we break the calculations and do not include it
        in the returned matrices list.
        Defaults to "step_10"

    Returns:
        tuple[csr_matrix, list[csr_matrix], dict, dict]: The first argument is the lossless 1-hop PAM with products.
        The second is a list of the lossy PAMs powers, the third argument is the node2id dictionary and
        the fourth argument is the relation to id dictionary.
    """

    unique_rels = sorted(list(df_train["rel"].unique()))
    unique_nodes = sorted(
        set(df_train["head"].values.tolist() + df_train["tail"].values.tolist())  # type: ignore
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
        unique_rels,
        starting_value=2,
        spacing_strategy=spacing_strategy,
    )

    # Create the adjacency matrix
    df_train["rel_mapped"] = df_train["rel"].map(rel2id)
    df_train["head_mapped"] = df_train["head"].map(node2id)
    df_train["tail_mapped"] = df_train["tail"].map(node2id)

    aggregated_df_lossless = (
        df_train.groupby(["head_mapped", "tail_mapped"])["rel_mapped"]
        .aggregate(np.prod)
        .reset_index()
    )

    pam_1hop_lossless = csr_matrix(
        (
            aggregated_df_lossless["rel_mapped"],
            (
                aggregated_df_lossless["head_mapped"],
                aggregated_df_lossless["tail_mapped"],
            ),
        ),
        shape=(len(unique_nodes), len(unique_nodes)),
    )

    if use_log:
        print(f"Will map to logs!")
        id2rel = {}
        for k, v in rel2id.items():
            rel2id[k] = np.log(v)
            id2rel[np.log(v)] = k

    df_train["rel_mapped"] = df_train["rel"].map(rel2id)

    if use_log:
        aggregated_df = (
            df_train.groupby(["head_mapped", "tail_mapped"])["rel_mapped"]
            .aggregate(np.sum)
            .reset_index()
        )
    else:
        aggregated_df = (
            df_train.groupby(["head_mapped", "tail_mapped"])["rel_mapped"]
            .aggregate(np.prod)
            .reset_index()
        )

    pam_1hop_lossy = csr_matrix(
        (
            aggregated_df["rel_mapped"],
            (aggregated_df["head_mapped"], aggregated_df["tail_mapped"]),
        ),
        shape=(len(unique_nodes), len(unique_nodes)),
        dtype=np.float64,
    )

    # # Calculate sparsity
    sparsity = get_sparsity(pam_1hop_lossy)
    print(pam_1hop_lossy.shape, f"Sparsity: {sparsity:.2f} %")

    # Generate the PAM^k matrices
    pam_powers = [pam_1hop_lossy]
    for ii in range(1, max_num_hops):
        print(f"Hop {ii + 1}")
        updated_power = pam_powers[-1] * pam_1hop_lossy
        # updated_power.sort_indices()
        # updated_power.eliminate_zeros()

        sparsity = get_sparsity(updated_power)
        print(f"Sparsity {ii + 1}-hop: {sparsity:.2f} %")
        if sparsity < break_with_sparsity_threshold:
            print(
                f"Stopped at {ii + 1} hops due to non-sparse matrix.. Current sparsity {sparsity:.2f} % < {break_with_sparsity_threshold}"
            )
            break
        pam_powers.append(updated_power)

    return pam_1hop_lossless, pam_powers, node2id, rel2id


if __name__ == "__main__":
    from data_loading import load_csv

    path = "../test/dummy_data/train.txt"

    df_train_orig, df_train = load_csv(path, add_inverse_edges="NO")

    pam_1hop_lossless, pam_powers, node2id, rel2id = create_pam_matrices(
        df_train, use_log=False, max_num_hops=3
    )
    print(rel2id)
    node_names = list(node2id.keys())
    pam_1 = pd.DataFrame(pam_1hop_lossless.todense(), columns=node_names)
    pam_1.index = node_names  # type:ignore
    print(pam_1)
