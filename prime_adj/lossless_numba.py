from itertools import product
from typing import List, Union

import numba as nb
import numpy as np
import pandas as pd
import tqdm
from numba import types
from numba.typed import Dict
from scipy.sparse import csr_array
from sympy.ntheory import factorint, nextprime

from prime_adj.pam_creation import create_pam_matrices
from prime_adj.utils import get_sparsity


def extend_paths(
    hop_k_paths: list[Union[tuple, int]], hop_1_vals: list[int]
) -> list[tuple]:
    extended_paths = []
    for comb in product(hop_k_paths, hop_1_vals):
        if isinstance(comb[0], tuple):
            res = tuple(list(comb[0]) + [comb[1]])
        else:
            res = tuple(comb)
        extended_paths.append(res)
    return extended_paths


def create_lattice_mappings(
    rel2id: dict[str, int], num_hops: int, use_log: bool, print_: bool = False
) -> dict[int, dict[str, dict]]:
    mappings = {
        0: {
            "path2prime": rel2id,
            "prime2path": dict(zip(rel2id.values(), rel2id.keys())),
        }
    }
    mappings[1] = {
        "path2prime": dict(
            zip([tuple([val]) for val in rel2id.values()], rel2id.values())
        ),
        "prime2path": dict(
            zip(rel2id.values(), [tuple([val]) for val in rel2id.values()])
        ),
    }
    hop_1_values = list(rel2id.values())
    if print_:
        iterator = tqdm.tqdm(range(2, num_hops + 1), total=num_hops - 1)
    else:
        iterator = range(2, num_hops + 1)
    for k in iterator:
        next_mapping = {}
        cur_prime = 3
        current_path_chains = (
            list(mappings[k - 1]["path2prime"].keys()) if k > 2 else hop_1_values
        )
        extended_paths = extend_paths(current_path_chains, hop_1_values)
        for ext_path in extended_paths:
            if use_log:
                value_to_use = np.log(cur_prime)
            else:
                value_to_use = cur_prime
            next_mapping[ext_path] = value_to_use
            cur_prime = nextprime(cur_prime)
        mappings[k] = {
            "path2prime": next_mapping,
            "prime2path": dict(zip(next_mapping.values(), next_mapping.keys())),
        }
    return mappings


@nb.jit(nopython=True)
def inner_direct(
    value_k_hop,
    value_1_hop,
    integer2path_k_hop,
    integer2paths_nb_1hop,
    path2prime_k_plus_1_hop,
):

    path_chains_at_k = integer2path_k_hop[value_k_hop]
    factor_last_hop = [prime for prime in integer2paths_nb_1hop[value_1_hop].flatten()]
    extended_paths_k_plus_1 = np.zeros(
        (len(path_chains_at_k) * len(factor_last_hop), len(path_chains_at_k[0]) + 1),
        dtype=np.int64,
    )
    product = 1
    path_k_i_counter = 0
    for path_chain in path_chains_at_k:
        for last_hop in factor_last_hop:
            res = np.append(path_chain, np.asarray(last_hop, dtype=np.int64))
            rest_str = "".join([str(item) for item in res])
            extended_paths_k_plus_1[path_k_i_counter, :] = res
            product = product * path2prime_k_plus_1_hop[rest_str]
            path_k_i_counter += 1

    return product, extended_paths_k_plus_1


@nb.jit(nopython=True)
def inner_log_direct(
    value_k_hop,
    value_1_hop,
    float2path_k_hop,
    float2paths_nb_1hop,
    path2prime_k_plus_1_hop,
):

    path_chains_at_k = float2path_k_hop[value_k_hop]
    factor_last_hop = [prime for prime in float2paths_nb_1hop[value_1_hop].flatten()]
    extended_paths_k_plus_1 = np.zeros(
        (len(path_chains_at_k) * len(factor_last_hop), len(path_chains_at_k[0]) + 1),
        dtype=np.int64,
    )
    product = 0
    path_k_i_counter = 0
    for path_chain in path_chains_at_k:
        for last_hop in factor_last_hop:
            res = np.append(path_chain, np.asarray(last_hop, dtype=np.int64))
            rest_str = "".join([str(item) for item in res])
            extended_paths_k_plus_1[path_k_i_counter, :] = res
            product += np.log(path2prime_k_plus_1_hop[rest_str])
            path_k_i_counter += 1
    # if (extended_paths_k_plus_1.sum(axis=1) == 0).sum() > 0:
    #     print("WTF?")
    return product, extended_paths_k_plus_1


@nb.jit(nopython=True)
def sparse_dot2(
    matrix_left_data,
    matrix_left_indices,
    matrix_left_indptr,
    left_n_rows,
    matrix_right_data,
    matrix_right_indices,
    matrix_right_indptr,
    right_n_cols,
    number2paths_nb,
    number2paths_nb_1hop,
    path2prime_k_plus_1_hop_nb,
    semiring,
):
    """Sparse matrix multiplication matrix_left x matrix_right

    Both matrices must be in the CSR sparse format.
    Data, indices and indptr are the standard CSR representation where the column indices for row i
    are stored in indices[indptr[i]:indptr[i+1]] and their corresponding values are stored in
    data[indptr[i]:indptr[i+1]].

    Args:
        matrix_left_data (numpy.array): Non-zero value of the sparse matrix.
        matrix_left_indices (numpy.array): Column positions of non-zero values.
        matrix_left_indptr (numpy.array): Array with the count of non-zero values per row.
        matrix_right_data (numpy.array): Non-zero value of the sparse matrix.
        matrix_right_indices (numpy.array): Column positions of non-zero values.
        matrix_right_indptr (numpy.array): Array with the count of non-zero values per row.

    Returns:
        numpy.array: 2D array with the result of the matrix multiplication.
    """

    rows, cols, values = [], [], []
    k_plus_1_value_to_map = Dict.empty(
        key_type=types.float64,
        value_type=types.int64[:, :],
    )
    value = 0
    for row_left in range(left_n_rows):
        for left_i in range(
            matrix_left_indptr[row_left], matrix_left_indptr[row_left + 1]
        ):
            col_left = matrix_left_indices[left_i]
            value_left = matrix_left_data[left_i]
            for right_i in range(
                matrix_right_indptr[col_left], matrix_right_indptr[col_left + 1]
            ):
                col_right = matrix_right_indices[right_i]
                value_right = matrix_right_data[right_i]
                if "lossless" in semiring:
                    if "log" in semiring:
                        value, extended_paths_k_plus_1 = inner_log_direct(
                            value_left,
                            value_right,
                            number2paths_nb,
                            number2paths_nb_1hop,
                            path2prime_k_plus_1_hop_nb,
                        )
                    else:
                        # value = 0
                        value, extended_paths_k_plus_1 = inner_direct(
                            value_left,
                            value_right,
                            number2paths_nb,
                            number2paths_nb_1hop,
                            path2prime_k_plus_1_hop_nb,
                        )
                # else:
                #     value = value_left * value_right
                #     extended_paths_k_plus_1 = [value_left, value_right]
                k_plus_1_value_to_map[value] = extended_paths_k_plus_1

                # print(
                #     f"P^2[{row_left}, {col_right}] = {value}, generated from P^1[{row_left}, {col_left}] = {value_left},  P^1[_, {col_right}] = {value_left} Sum = {value_left+value_right}"
                # )
                # print(k_plus_1_value_to_map[value])

                rows.append(row_left)
                cols.append(col_right)
                values.append(value)
    return rows, cols, values, k_plus_1_value_to_map


def spmm(
    matrix_left: csr_array,
    matrix_right: csr_array,
    number2paths_nb,
    number2paths_nb_1hop,
    path2prime_k_plus_1_hop_nb,
    semiring,
):

    if semiring not in ["lossless", "plus_times", "lossless_log_plus"]:
        raise NotImplementedError(f"{semiring} not implemented...")

    rows, cols, values, k_plus_1_value_to_map = sparse_dot2(
        matrix_left.data,
        matrix_left.indices,
        matrix_left.indptr,
        matrix_left.shape[0],
        matrix_right.data,
        matrix_right.indices,
        matrix_right.indptr,
        matrix_right.shape[1],
        number2paths_nb,
        number2paths_nb_1hop,
        path2prime_k_plus_1_hop_nb,
        semiring=semiring,
    )

    dictionary_of_next_hop = {}
    k_plus_1_value_to_map_aggregated = {}
    for index, (row, col) in enumerate(zip(rows, cols)):
        value = values[index]
        # if "log" in semiring:
        #    value = np.log(value)
        if (row, col) not in dictionary_of_next_hop:
            dictionary_of_next_hop[(row, col)] = value
            k_plus_1_value_to_map_aggregated[value] = np.asarray(
                k_plus_1_value_to_map[value], np.int64
            )
        else:
            cur_value = dictionary_of_next_hop[(row, col)]
            if semiring == "plus_times" or semiring == "lossless_log_plus":
                dictionary_of_next_hop[(row, col)] = cur_value + value
            else:
                dictionary_of_next_hop[(row, col)] = cur_value * value
            k_plus_1_value_to_map_aggregated[
                dictionary_of_next_hop[(row, col)]
            ] = np.vstack(
                (
                    k_plus_1_value_to_map_aggregated[cur_value],
                    np.asarray(k_plus_1_value_to_map[value], np.int64),
                )
            )
    indices = list(zip(*list(dictionary_of_next_hop.keys())))
    if "log" in semiring:
        type_ = np.float64
    else:
        type_ = np.int64

    return (
        csr_array(
            (list(dictionary_of_next_hop.values()), (indices[0], indices[1])),
            shape=(matrix_left.shape[0], matrix_right.shape[1]),
            dtype=type_,
        ),
        k_plus_1_value_to_map_aggregated,
    )


def create_lossless_khops(
    df_train_orig: pd.DataFrame, max_hop: int, print_: bool = False
):
    (A_sparse, _, node2id, rel2id, broke_cause_of_sparsity,) = create_pam_matrices(
        df_train_orig,
        max_order=2,
        method="plus_times",
        use_log=False,  ############################## HARDCODED FALSE  #########
        spacing_strategy="step_1",
    )

    if print_:
        print(f"Creating Mappings..")
    mappings = create_lattice_mappings(rel2id, max_hop, False)

    number2paths = {}
    for value, paths in mappings[1]["prime2path"].items():
        number2paths[value] = paths

    number2paths_nb = Dict.empty(
        key_type=types.float64,
        value_type=types.int64[:, :],
    )
    number2paths_nb_1hop = Dict.empty(
        key_type=types.float64,
        value_type=types.int64[:, :],
    )
    for k, v in number2paths.items():
        number2paths_nb[k] = np.asarray(v, dtype=np.int64).reshape(-1, 1)
        number2paths_nb_1hop[k] = number2paths_nb[k]

    for value in A_sparse.data:
        if value not in number2paths_nb_1hop:
            number2paths_nb[value] = np.asarray(
                factorint(value, multiple=True), dtype=np.int64
            ).reshape(-1, 1)
            number2paths_nb_1hop[value] = number2paths_nb[value]

    power_A = [A_sparse]

    type_ = types.float64
    for cur_hop_index in range(max_hop - 1):
        # Prepare mappings
        if print_:
            print(f"K is {cur_hop_index + 2}")
        path2prime_k_plus_1_hop = mappings[cur_hop_index + 2]["path2prime"]

        # print(cur_hop_index + 1)
        # print(prime2path_k_hop)
        # print(path2prime_k_plus_1_hop)
        # print("\n")

        path2prime_k_plus_1_hop_nb = Dict.empty(
            key_type=types.string,
            value_type=type_,
        )

        for k, v in path2prime_k_plus_1_hop.items():
            path2prime_k_plus_1_hop_nb["".join([str(k_) for k_ in k])] = v

        updated_A, new_paths = spmm(
            power_A[-1],
            power_A[0],
            number2paths_nb,
            number2paths_nb_1hop,
            path2prime_k_plus_1_hop_nb,
            "lossless_log_plus",
        )

        mappings[cur_hop_index + 2] = {"path2prime": {}, "prime2path": {}}
        for float_, paths_arr in new_paths.items():
            number2paths_nb[float_] = np.asarray(paths_arr, dtype=np.int64)
            path_list = paths_arr.tolist()

            tupled_paths = tuple([tuple(sub) for sub in path_list])
            mappings[cur_hop_index + 2]["path2prime"][tupled_paths] = float_
            mappings[cur_hop_index + 2]["prime2path"][float_] = tupled_paths

        num_overflow = (updated_A.data < 0).sum()
        num_inf = np.isinf(updated_A.data).sum()
        updated_A.data[(updated_A.data < 0)] = 0
        updated_A.data[np.isinf(updated_A.data)] = 0
        updated_A.eliminate_zeros()
        if num_overflow > 0 or num_inf > 0:
            # pass
            # raise ArithmeticError(f"Created {num_overflow} overflows and {num_inf} infinities during calculations")

            print(
                f"Hop {cur_hop_index + 2}: Created {num_overflow} overflows and {num_inf} infinities during calculations"
            )

        sparsity = get_sparsity(updated_A)
        if print_:
            print(f"Sparsity {cur_hop_index + 2}-hop: {sparsity:.2f} %")

        power_A.append(updated_A)

    return A_sparse, power_A, node2id, rel2id, broke_cause_of_sparsity, mappings


################################################## HARD-CODED FASLE USE_LOG ########################################3

if __name__ == "__main__":

    import time

    from data_loading import load_data

    project_name = "test"
    add_inverse_edges = "NO"
    path = "./data/dummy_data"
    max_hop = 2
    print_ = True

    df_train_orig, df_train, df_eval, df_test, already_seen_triples = load_data(
        path,
        project_name=project_name,
        add_inverse_edges=add_inverse_edges,
        sep=",",
        skiprows=0,
    )

    total_s = time.time()
    (
        A_sparse,
        power_A,
        node2id,
        rel2id,
        broke_cause_of_sparsity,
        mappings,
    ) = create_lossless_khops(df_train_orig, max_hop, print_)

    time_taken = time.time() - total_s
    print(f"Overall {time_taken:.2f} seconds ({time_taken/60:.2f} mins)")
