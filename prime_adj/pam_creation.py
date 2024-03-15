from typing import Literal

import graphblas as gb
import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from sympy import nextprime


def get_prime_map_from_rel(
    list_of_rels: list,
    starting_value: int = 2,
    spacing_strategy: str = "step_10",
) -> tuple[dict, dict]:
    """
    Helper function that given a list of relations returns the mappings to and from the
    prime numbers used.

    Different strategies to map the numbers are available.
    "step_X", increases the step between two prime numbers by adding X to the current prime
    "factor_X", increases the step between two prime numbers by multiplying the current prime with X
    "natural", increases by 1 starting from 2
    "constant", return the same as step_10 but the values are changed to constants in create_pam_matrices

    Args:
        list_of_rels (list): iterable, contains a list of the relations that need to be mapped.
        starting_value (int, optional): Starting value of the primes. Defaults to 2.
        spacing_strategy (str, optional):  Spacing strategy for the primes. Defaults to "step_10".

    Returns:
        rel2prime: dict, relation to prime dictionary e.g. {"rel1":2}.
        prime2rel: dict, prime to relation dictionary e.g. {2:"rel1"}.
    """

    list_of_rels = [str(relid) for relid in list_of_rels]

    # Initialize dicts
    rel2prime = {}
    prime2rel = {}
    # Starting value for finding the next prime
    current_int = starting_value
    # Map each relation id to the next available prime according to the strategy used
    if spacing_strategy == "natural":
        c = 1
        for relid in list_of_rels:
            rel2prime[relid] = c
            prime2rel[c] = relid
            c += 1
    else:
        if spacing_strategy == "constant":
            spacing_strategy = "step_10"
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
    max_order: int = 5,
    use_log: bool = True,
    method: Literal["plus_times", "plus_plus", "constant"] = "plus_times",
    spacing_strategy: str = "step_10",
    eliminate_diagonal: bool = False,
    break_with_sparsity_threshold: float = -1,
    check_error_lossless: bool = True,
    print_: bool = False,
) -> tuple[csr_array, list[csr_array], dict, dict, bool]:

    """Helper function that creates the pam matrices.

    Args:
        df_train (pd.DataFrame): The triples in the form of a pd.DataFrame with columns
        (head, rel, tail).

        max_order (int, optional): The maximum order for the PAMs (i.e. the k-hops).
        Defaults to 5.

        use_log (bool, optional): Whether to use log of primes for numerical stability.
        Defaults to True.

        method (Literal["plus_times", "plus_plus", "constant"], optional):  Method of multiplication.
        - "plus_times": Generic matrix multiplication GrapBLAS multiplication.
        - "plus_plus": Matrix multiplication using a plus_plus semiring using GraphBLAS. Use with use_log=True recommended.
        - "constant": Replace all values in the 1-hop matrix with ones. For benchmarking purposes.
        Defaults to "plus_times".

        spacing_strategy (str, optional): The spacing strategy as mentioned in get_prime_map_from_rel.
        Defaults to "step_10".

        break_with_sparsity_threshold (int, optional): The percentage of sparsity that is not accepted.
        If one of the k-hop PAMs has lower sparsity we break the calculations and do not include it
        in the returned matrices list.
        Defaults to "step_10".

        eliminate_diagonal (bool, optional): Whether to zero-out the diagonal in each k-hop.
        (This essentially removes cyclic paths from being propagated).
        Defaults to False.

        check_error_lossless (bool, optional): Whether to check for int overflow for the loss-less matrix.
        Defaults to True.

    Returns:
        tuple[csr_matrix, list[csr_matrix], dict, dict, bool]: The first argument is the lossless 1-hop PAM with products.
        The second is a list of the lossy PAMs powers, the third argument is the node2id dictionary and
        the fourth argument is the relation to id dictionary. The fifth argument is weather the matrix creation process
        was broken due to sparsity.
    """

    # Number of unique rels and nodes

    unique_rels = sorted(list(df_train["rel"].unique()))

    unique_nodes = sorted(
        set(df_train["head"].values.tolist() + df_train["tail"].values.tolist())  # type: ignore
    )
    if print_:
        print(
            f"# of unique rels: {len(unique_rels)} \t | # of unique nodes: {len(unique_nodes)}"
        )

    node2id = {}
    id2node = {}
    for i, node in enumerate(unique_nodes):
        node2id[node] = i
        id2node[i] = node

    # Map the relations to primes
    rel2id, id2rel = get_prime_map_from_rel(
        unique_rels,
        starting_value=2,
        spacing_strategy=spacing_strategy,
    )

    # Map all node and rel values to the corresponding numerical ones.
    df_train["rel_mapped"] = df_train["rel"].astype(str).map(rel2id)
    df_train["head_mapped"] = df_train["head"].map(node2id)
    df_train["tail_mapped"] = df_train["tail"].map(node2id)

    # Create the lossless representation (with product). This may lead to overflows due to the product.
    aggregated_df_lossless = (
        df_train.groupby(["head_mapped", "tail_mapped"])["rel_mapped"]
        .aggregate("prod")
        .reset_index()
    )

    # If we don't case about the loss-less scenario we can omit this error.
    if check_error_lossless:
        if any(aggregated_df_lossless["rel_mapped"].values > np.iinfo(np.int64).max):
            raise OverflowError(
                f"You have overflowing due to large prime number products in 1-hop PAM. Please lower the spacing strategy (lowest is 'step_1'), current is {spacing_strategy}"
            )

    # Construct the "lossless" 1-hop array (with products of primes probably)
    pam_1hop_lossless = csr_array(
        (
            aggregated_df_lossless["rel_mapped"],
            (
                aggregated_df_lossless["head_mapped"],
                aggregated_df_lossless["tail_mapped"],
            ),
        ),
        shape=(len(unique_nodes), len(unique_nodes)),
        dtype=np.int64,
    )

    # If use log, will need to re-map the values
    if use_log:
        if print_:
            print(f"Will map to logs!")
        id2rel = {}
        for k, v in rel2id.items():
            rel2id[k] = np.log(v)
            id2rel[np.log(v)] = k

    df_train["rel_mapped"] = df_train["rel"].astype(str).map(rel2id)

    # Create the lossy 1-hop with log-sum-sum
    aggregated_df = (
        df_train.groupby(["head_mapped", "tail_mapped"])["rel_mapped"]
        .aggregate("sum")
        .reset_index()
    )
    pam_1hop_lossy = csr_array(
        (
            aggregated_df["rel_mapped"],
            (aggregated_df["head_mapped"], aggregated_df["tail_mapped"]),
        ),
        shape=(len(unique_nodes), len(unique_nodes)),
        dtype=np.float64,
    )

    # Map everything to 1. This is only for testing scenarios.
    if spacing_strategy == "constant":
        pam_1hop_lossy.data = np.ones_like(pam_1hop_lossy.data)

    # Create the GraphBLAS equivalent
    A_gb = gb.io.from_scipy_sparse(pam_1hop_lossy)

    # Generate the PAM^k matrices
    pam_powers = [pam_1hop_lossy]
    pam_power_gb = [A_gb]
    broke_cause_of_sparsity = False
    for ii in range(1, max_order):
        if print_:
            print(f"Hop {ii + 1}")
        cur_previous_power = pam_power_gb[-1].dup()
        if eliminate_diagonal:
            cur_previous_power.setdiag(0)
        updated_power_gb = cur_previous_power.mxm(A_gb, method).new()

        updated_power = gb.io.to_scipy_sparse(updated_power_gb)

        sparsity = get_sparsity(updated_power)
        if print_:
            print(f"Sparsity {ii + 1}-hop: {sparsity:.2f} %")
        if sparsity < 100 * break_with_sparsity_threshold and ii > 1:
            if print_:
                print(
                    f"Stopped at {ii + 1} hops due to non-sparse matrix.. Current sparsity {sparsity:.2f} % < {break_with_sparsity_threshold}"
                )
            broke_cause_of_sparsity = True
            break
        pam_powers.append(updated_power)
        pam_power_gb.append(updated_power_gb)

    return pam_1hop_lossless, pam_powers, node2id, rel2id, broke_cause_of_sparsity


def get_sparsity(A: csr_array) -> float:
    """Calculate sparsity % of scipy sparse matrix.
    Args:
        A (scipy.sparse): Scipy sparse matrix
    Returns:
        (float)): Sparsity as a float
    """

    return 100 * (1 - A.nnz / (A.shape[0] ** 2))


if __name__ == "__main__":
    from data_loading import load_csv

    path = "../data/dummy_data/train.txt"

    df_train_orig, df_train = load_csv(path, add_inverse_edges="YES")
    # print(df_train_orig)
    (
        pam_lossles,
        pam_powers,
        node2id,
        rel2id,
        broke_cause_of_sparsity,
    ) = create_pam_matrices(
        df_train,
        max_order=2,
        method="plus_times",
        use_log=False,
        spacing_strategy="step_1",
    )
    print(rel2id)
    node_names = list(node2id.keys())
    pam_2 = pd.DataFrame(pam_powers[1].todense(), columns=node_names)
    pam_2.index = node_names  # type:ignore
    print("PAM@2")
    print(pam_2)
