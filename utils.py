import functools
import os
import random

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy
from sympy import nextprime, primefactors
from sympy.ntheory import factorint


@functools.lru_cache(maxsize=None)
def get_primefactors(value: float) -> tuple[int]:
    """Wrapper functiom that gets a value and returns the list
       of prime factors of the value. It is used as a wrapper around
       primefactors in ordet ot use memoization with cache for speed.

    Args:
        value (float): The float value to decompose

    Returns:
        Tuple[int]: A list of the unique prime factors
    """
    return primefactors(value)


def get_prime_map_from_rel(
    list_of_rels: list,
    starting_value: int = 1,
    spacing_strategy: str = "step_1",
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
        cur_prime = nextprime(current_int)
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


def get_prime_adjacency(
    edges_dict: dict,
    nodes_dict: dict[int, str] | None,
    starting_value: int = 1,
    rel_spacing_strategy: str = "step_1",
    node_spacing_strategy: str = "",
    add_inverse_edges: bool = False,
) -> np.ndarray:
    """Helper function to create adjacency matrices with primes, using grakel dicts as input.
    Currently utilizing edges + node labels if needed. The edge dict has key, values in the form of:
    (head, tail): relation, while the node dict has (node_id:node_label)

    Args:
        edges_dict (dict): Dictionary of edges in the form (head, tail)=relation
        nodes_dict (dict, optional): Dictionary of node lables in the form (node_id:label). Defaults to "".
        starting_value (int, optional): Starting number for prime mapping. Defaults to 1.
        rel_spacing_strategy (str, optional): Strategy to use to find the next prime Check get_prime_map_from_rel for details.
                                              Defaults to "step_1".
        node_spacing_strategy (str, optional): Same as above. Defaults to the same as rel_spacing_strategy if "".
        add_inverse_edges (bool, optional): Whether to add the inverse edges. Defaults to False.

    Raises:
        NotImplementedError: _description_

    Returns:
        np.array: _description_
    """
    nodes = set()
    rels = set()

    for key, value in edges_dict.items():
        nodes.add(key[0])
        nodes.add(key[1])
        rels.add(str(value))
        if add_inverse_edges:
            rels.add(str(value) + "_INV")

    # In case we add node labels
    if nodes_dict:
        if not (node_spacing_strategy):
            node_spacing_strategy = rel_spacing_strategy

    # Node mapping to increasing index
    nodes = sorted(nodes)
    id2node = {}
    node2id = {}
    for i, n in enumerate(nodes):
        id2node[i] = n
        node2id[n] = i

    rels = sorted(rels)
    # create the mapping of rels to primes
    relid2prime, prime2relid = get_prime_map_from_rel(
        rels, starting_value=starting_value, spacing_strategy=rel_spacing_strategy
    )
    # create the adjacency matrix
    adj = np.zeros((len(nodes), len(nodes)))
    for key, value in edges_dict.items():
        adj[node2id[key[0]], node2id[key[1]]] = relid2prime[str(value)]
        if add_inverse_edges:
            adj[node2id[key[1]], node2id[key[0]]] = relid2prime[str(value) + "_INV"]

    # add diagonal values with node labels if needed.
    if nodes_dict:
        node2prime = {}
        current_int = max(relid2prime.values())
        for node_label in set(sorted(list(nodes_dict.values()))):
            cur_prime = nextprime(current_int)
            node2prime[node_label] = cur_prime
            if "step" in node_spacing_strategy:
                step = float(node_spacing_strategy.split("_")[1])
                current_int = cur_prime + step
            elif "factor" in node_spacing_strategy:
                factor = float(node_spacing_strategy.split("_")[1])
                current_int = cur_prime * factor
            else:
                raise NotImplementedError(
                    f"Node spacing strategy : {node_spacing_strategy}  not understood!"
                )

        for node_id, node_label in nodes_dict.items():
            adj[node2id[node_id], node2id[node_id]] = node2prime[node_label]
    return adj


def ILP_solver(denominations, target_value, max_number_of_coins) -> dict[int, int]:
    """
    Solver using GLPK_MI for boolean integer linear programming
    :param denominations: list, list of avaialble denominations to break the target value into
    :param target_value: int, target value that needs to be broken into a sum of denominations
    :param max_number_of_coins: int, the number of denominations used to create the target value
    :return: dict, {'denomination_1':times_used1, 'denomination2':times_used2, ...}
    """
    w = cp.Constant(
        denominations,
    )
    CASH = cp.Constant(target_value)
    max_number_of_coins = cp.Constant(max_number_of_coins)

    x = cp.Variable((1, w.shape[0]), integer=True)

    # We want to minimize the total number of coins returned
    objective = cp.Minimize(cp.abs(max_number_of_coins - cp.sum(x)))
    # print(CASH, max_number_of_coins, w)
    # The constraints
    constraints = [
        w @ x.T == CASH,
        # cp.sum(x) == max_number_of_coins, #
        x >= 0,  # semi-positive coins
    ]
    # Form and solve problem.
    prob = cp.Problem(objective, constraints)
    # Need the GLPK_MI solver because the ECOS_BB is not working correctly.
    prob.solve(solver="GLPK_MI")  # Returns the optimal value.
    if prob.status == "infeasible":
        # print("Infeasible. Can't create %s with %s denominations from: %s"%(CASH.__str__(), max_number_of_coins.__str__(),  w.__str__()))
        return {}
    else:
        # print("Initial cash %s  is changed into %d coins as follows:"%(CASH.__str__(), prob.value))
        return dict(zip([w_ for w_ in w.value], x.value.flatten()))


def load_data(
    path_to_folder: str, project_name: str, add_inverse_edges: str = "NO"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set]:
    """
    Helper function that loads the data in pd.DataFrames and returns them.
    Args:
        path_to_folder (str): path to folder with train.txt, valid.txt, test.txt
        project_name (str): name of the project
        add_inverse_edges (str, optional):  Whether to add the inverse edges.
        Possible values "YES", "YES__INV", "NO". Defaults to "NO".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set]: [description]
    """
    PROJECT_DETAILS = {
        "lc-neo4j": {"skiprows": 1, "sep": "\t"},
        "codex-s": {"skiprows": 0, "sep": "\t"},
        "WN18RR": {"skiprows": 0, "sep": "\t"},
        "YAGO3-10-DR": {"skiprows": 0, "sep": "\t"},
        "YAGO3-10": {"skiprows": 0, "sep": "\t"},
        "FB15k-237": {"skiprows": 0, "sep": "\t"},
    }

    df_train = pd.read_csv(
        os.path.join(path_to_folder, "train.txt"),
        sep=PROJECT_DETAILS[project_name]["sep"],
        header=None,
        dtype="str",
        skiprows=PROJECT_DETAILS[project_name]["skiprows"],
    )
    df_train.columns = ["head", "rel", "tail"]

    # If we want to add inverse edges as well
    if "YES" in add_inverse_edges:
        print(f"Will add the inverse train edges as well..")
        df_train["rel"] = df_train["rel"].astype(str)
        df_train_inv = df_train.copy()
        df_train_inv["head"] = df_train["tail"]
        df_train_inv["tail"] = df_train["head"]
        # We may opt to denote this edges with a distinct __INV style
        # E.g. "a - related - b" -> "b - related__INV - a"
        if add_inverse_edges == "YES__INV":
            df_train_inv["rel"] = df_train["rel"] + "__INV"
        df_train = df_train.append(df_train_inv)
    if project_name in ["lc-neo4j"]:
        df_eval = None
        df_test = None
        already_seen_triples = set(df_train.to_records(index=False).tolist())
    else:
        try:
            df_eval = pd.read_csv(
                os.path.join(path_to_folder, "valid.txt"),
                sep=PROJECT_DETAILS[project_name]["sep"],
                header=None,
                dtype="str",
                skiprows=PROJECT_DETAILS[project_name]["skiprows"],
            )
            df_eval.columns = ["head", "rel", "tail"]
        except FileNotFoundError:
            df_eval = df_train.copy()
        df_test = pd.read_csv(
            os.path.join(path_to_folder, "test.txt"),
            sep=PROJECT_DETAILS[project_name]["sep"],
            header=None,
            dtype="str",
            skiprows=PROJECT_DETAILS[project_name]["skiprows"],
        )
        df_test.columns = ["head", "rel", "tail"]
        if "YAGO" in project_name:
            for cur_df in [df_train, df_eval, df_test]:
                for col in cur_df.columns:
                    cur_df[col] = cur_df[col] + "_YAGO"

        already_seen_triples = set(
            df_train.to_records(index=False).tolist()
            + df_eval.to_records(index=False).tolist()
        )

    print(f"Total: {len(already_seen_triples)} triples in train + eval!)")
    print(f"In train: {len(df_train)}")
    print(f"In valid: {len(df_eval)}")
    print(f"In test: {len(df_test)}")
    return df_train, df_eval, df_test, already_seen_triples


def set_all_seeds(seed: int = 0):
    """Fix random seeds

    Args:
        seed (int): Random seed
    """

    random.seed(seed)
    np.random.seed(seed)
    return 1


def get_sparsity(A: scipy.sparse.csc_matrix) -> float:
    """Calculate sparsity % of scipy sparse matrix.

    Args:
        A (scipy.sparse): Scipy sparse matrix

    Returns:
        (float)): Sparsity as a float
    """

    return 100 * (1 - A.nnz / (A.shape[0] ** 2))
