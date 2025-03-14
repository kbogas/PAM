import functools
import random

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sympy import factorint, primefactors


def set_all_seeds(seed: int = 0):
    """Fix random seeds

    Args:
        seed (int): Random seed
    """

    random.seed(seed)
    np.random.seed(seed)
    return 1


def get_sparsity(A: csr_matrix) -> float:
    """Calculate sparsity % of scipy sparse matrix.

    Args:
        A (scipy.sparse): Scipy sparse matrix

    Returns:
        (float)): Sparsity as a float
    """

    return 100 * (1 - A.nnz / (A.shape[0] ** 2))


@functools.lru_cache(maxsize=None)
def get_primefactors(value: float) -> list[int]:
    """Wrapper functiom that gets a value and returns the list
       of prime factors of the value. It is used as a wrapper around
       primefactors in ordet ot use memoization with cache for speed.

    Args:
        value (float): The float value to decompose

    Returns:
        Tuple[int]: A list of the unique prime factors
    """
    return primefactors(value)


@functools.lru_cache(maxsize=None)
def get_primefactors_multiplicity(value: float) -> list[int]:
    """Wrapper function that gets a value and returns the list
       of prime factors of the value with multiplicity. It is used as a wrapper around
       factorint in ordet ot use memoization with cache for speed.

    Args:
        value (float): The float value to decompose

    Returns:
        list[int]: A list of the (multiple maybe) prime factors
    """

    total = []
    for factor, multiplicity in factorint(value).items():
        total.extend([factor] * multiplicity)
    return total


def calculate_hits_at_k(
    results: pd.DataFrame, wanted_ranks: list[int] = [1, 3, 5, 10], print_: bool = False
) -> dict[str, float]:
    """Helper function to print results.
    The given dataframe has one column named 'rank_correct' ranging from 0 to inf
    denoting the rank of the correct target entity in the predictions for each query (row).
    If the correct target was not found it is 0.
    Will also print the upper bound, meaning if we had an ORACLE selector to provide us with the
    correct target each time irrelevant of the actual rank of the target, what would our score be.
    This essentially is the percentage of test samples where the correct target was found somewhere in
    the predictions made.

    Args:
        results (pd.DataFrame): Dataframe with the predictions per test query.
        wanted_ranks (list[int], optional): Ranks at which we calculate the values. Defaults to [1, 3, 5, 10].
        print_ (bool, optional): Whether to print them as well. Defaults to False.

    Returns:
        dict[str,float]: The ranks in dict
    """
    hits_at_k = {}
    num_test_samples = results.shape[0]
    for k in wanted_ranks:
        hits_at_k[k] = (
            results[
                (results["rank_correct"] > 0) & (results["rank_correct"] <= k)
            ].shape[0]
            / num_test_samples
        )
    hits_at_k["UpperBound"] = (
        num_test_samples - results[(results["rank_correct"] == 0)].shape[0]
    ) / num_test_samples
    if print_:
        for hit_k, perc_hit in hits_at_k.items():
            print(f"Tail Hits@{hit_k}: {100*perc_hit:.2f} %")
    return hits_at_k
