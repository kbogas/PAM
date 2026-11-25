from typing import Literal
import warnings
import math
from dataclasses import dataclass

import graphblas as gb
import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from sympy import nextprime

from prime_adj.utils import get_sparsity


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


@dataclass(frozen=True)
class PAMMetadata:
    """Configuration needed to interpret a numeric PAM result."""

    encoding: str
    use_log: bool
    semiring: str
    eliminate_diagonal: bool
    minimum_sparsity: float | None


@dataclass(frozen=True)
class PAMResult:
    """Typed PAM result; matrices are ordered from one hop upward."""

    one_hop_exact: csr_array
    powers: tuple[csr_array, ...]
    node_map: dict
    relation_map: dict
    metadata: PAMMetadata
    stopping_reason: Literal["max_order", "sparsity_threshold", "empty_graph"]

    def as_legacy_tuple(self) -> tuple[csr_array, list[csr_array], dict, dict, bool]:
        """Return the five-element tuple used by releases through 0.0.10."""

        return (
            self.one_hop_exact,
            list(self.powers),
            self.node_map,
            self.relation_map,
            self.stopping_reason == "sparsity_threshold",
        )


def _create_pam_matrices(
    df_train: pd.DataFrame,
    max_order: int = 5,
    use_log: bool = True,
    method: Literal["plus_times", "plus_plus", "constant"] = "plus_times",
    spacing_strategy: str = "step_10",
    eliminate_diagonal: bool = False,
    break_with_sparsity_threshold: float = -1,
    check_error_lossless: bool = True,
    print_: bool = False,
    relation_mapping: dict[str, int | float] | None = None,
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
        - "constant": Deprecated legacy alias for ``method="plus_times",
          spacing_strategy="constant"``.
        Defaults to "plus_times".

        spacing_strategy (str, optional): The spacing strategy as mentioned in get_prime_map_from_rel.
        Defaults to "step_10".

        break_with_sparsity_threshold (int, optional): The percentage of sparsity that is not accepted.
        If one of the k-hop PAMs has lower sparsity we break the calculations and do not include it
        in the returned matrices list.
        Defaults to -1.

        eliminate_diagonal (bool, optional): Whether to zero-out the diagonal in each k-hop.
        (This essentially removes cyclic paths from being propagated).
        Defaults to False.

        check_error_lossless (bool, optional): Whether to check for int overflow for the loss-less matrix.
        Defaults to True.

        relation_mapping (dict, optional): A pre-fitted relation-to-number map.
        All relations in ``df_train`` must be present. Extra entries are retained,
        allowing one shared map to be reused by graphs with different subsets.

    Returns:
        tuple[csr_matrix, list[csr_matrix], dict, dict, bool]: The first argument is the lossless 1-hop PAM with products.
        The second is a list of the lossy PAMs powers, the third argument is the node2id dictionary and
        the fourth argument is the relation to id dictionary. The fifth argument is weather the matrix creation process
        was broken due to sparsity.
    """

    if max_order < 1:
        raise ValueError("max_order must be at least 1")
    if method == "constant":
        warnings.warn(
            "method='constant' is deprecated; use method='plus_times' with "
            "spacing_strategy='constant'",
            DeprecationWarning,
            stacklevel=2,
        )
        method = "plus_times"
        spacing_strategy = "constant"
    if method not in {"plus_times", "plus_plus"}:
        raise ValueError(f"Unsupported multiplication method: {method!r}")
    if spacing_strategy == "constant" and use_log:
        raise ValueError(
            "constant-one encoding cannot be combined with use_log=True: "
            "log(1) is zero and does not carry an edge through multiplication"
        )
    if break_with_sparsity_threshold != -1 and not (
        0 <= break_with_sparsity_threshold <= 1
    ):
        raise ValueError(
            "break_with_sparsity_threshold must be -1 (disabled) or a "
            "minimum sparsity fraction in [0, 1]"
        )

    # Mapping columns are implementation details and must never leak into the
    # caller's frame or make repeated calls order-dependent.
    df_train = df_train.copy(deep=True)

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
        node2id[node] = int(i)
        id2node[int(i)] = node

    # Map the relations to primes
    if relation_mapping is None:
        rel2id, id2rel = get_prime_map_from_rel(
            unique_rels,
            starting_value=2,
            spacing_strategy=spacing_strategy,
        )
    else:
        relation_mapping = {str(key): value for key, value in relation_mapping.items()}
        missing = sorted(set(unique_rels) - set(relation_mapping))
        if missing:
            raise ValueError(f"relation_mapping is missing relations: {missing}")
        if any(not np.isfinite(value) or value <= 0 for value in relation_mapping.values()):
            raise ValueError("relation_mapping values must be finite and positive")
        if any(float(value) != int(value) for value in relation_mapping.values()):
            raise ValueError(
                "relation_mapping values must be integers before an optional log transform"
            )
        rel2id = dict(relation_mapping)
        id2rel = {value: relation for relation, value in rel2id.items()}

    if spacing_strategy == "constant":
        rel2id = {relation: 1 for relation in rel2id}
        id2rel = {1: tuple(rel2id)}

    # Map all node and rel values to the corresponding numerical ones.
    df_train["rel_mapped"] = df_train["rel"].astype(str).map(rel2id)
    df_train["head_mapped"] = df_train["head"].map(node2id)
    df_train["tail_mapped"] = df_train["tail"].map(node2id)

    # Create the lossless representation (with product). This may lead to overflows due to the product.
    aggregated_df_lossless = (
        df_train.groupby(["head_mapped", "tail_mapped"])["rel_mapped"]
        .aggregate(lambda values: math.prod(int(value) for value in values))
        .reset_index()
    )

    # If we don't case about the loss-less scenario we can omit this error.
    if check_error_lossless:
        if any(
            int(value) > np.iinfo(np.int64).max
            for value in aggregated_df_lossless["rel_mapped"]
        ):
            raise OverflowError(
                f"You have overflowing due to large prime number products in 1-hop PAM. Please lower the spacing strategy (lowest is 'step_1'), current is {spacing_strategy}"
            )
    aggregated_df_lossless["rel_mapped"] = aggregated_df_lossless[
        "rel_mapped"
    ].astype(np.int64)

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

    if not unique_nodes:
        return pam_1hop_lossless, [pam_1hop_lossy], node2id, rel2id, False

    # Map everything to 1. This is only for testing scenarios.
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
            cur_previous_power = cur_previous_power.select(gb.select.offdiag).new()
        updated_power_gb = cur_previous_power.mxm(A_gb, method).new()

        updated_power = gb.io.to_scipy_sparse(updated_power_gb)

        sparsity = get_sparsity(updated_power)
        if print_:
            print(f"Sparsity {ii + 1}-hop: {sparsity:.2f} %")
        if (
            break_with_sparsity_threshold != -1
            and sparsity / 100 < break_with_sparsity_threshold
        ):
            if print_:
                print(
                    f"Stopped at {ii + 1} hops due to non-sparse matrix.. Current sparsity {sparsity:.2f} % < {break_with_sparsity_threshold}"
                )
            broke_cause_of_sparsity = True
            break
        pam_powers.append(updated_power)
        pam_power_gb.append(updated_power_gb)

    return pam_1hop_lossless, pam_powers, node2id, rel2id, broke_cause_of_sparsity


def create_pam(
    df_train: pd.DataFrame,
    max_order: int = 5,
    use_log: bool = True,
    method: Literal["plus_times", "plus_plus", "constant"] = "plus_times",
    spacing_strategy: str = "step_10",
    eliminate_diagonal: bool = False,
    break_with_sparsity_threshold: float = -1,
    check_error_lossless: bool = True,
    print_: bool = False,
    relation_mapping: dict[str, int | float] | None = None,
) -> PAMResult:
    """Create PAMs and return a typed result with interpretation metadata."""

    legacy_method = method
    values = _create_pam_matrices(
        df_train=df_train,
        max_order=max_order,
        use_log=use_log,
        method=method,
        spacing_strategy=spacing_strategy,
        eliminate_diagonal=eliminate_diagonal,
        break_with_sparsity_threshold=break_with_sparsity_threshold,
        check_error_lossless=check_error_lossless,
        print_=print_,
        relation_mapping=relation_mapping,
    )
    one_hop, powers, node_map, fitted_relation_map, stopped = values
    effective_method = "plus_times" if legacy_method == "constant" else legacy_method
    effective_encoding = "constant" if legacy_method == "constant" else spacing_strategy
    if not node_map:
        stopping_reason = "empty_graph"
    elif stopped:
        stopping_reason = "sparsity_threshold"
    else:
        stopping_reason = "max_order"
    return PAMResult(
        one_hop_exact=one_hop,
        powers=tuple(powers),
        node_map=node_map,
        relation_map=fitted_relation_map,
        metadata=PAMMetadata(
            encoding=effective_encoding,
            use_log=use_log,
            semiring=effective_method,
            eliminate_diagonal=eliminate_diagonal,
            minimum_sparsity=(
                None
                if break_with_sparsity_threshold == -1
                else break_with_sparsity_threshold
            ),
        ),
        stopping_reason=stopping_reason,
    )


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
    relation_mapping: dict[str, int | float] | None = None,
) -> tuple[csr_array, list[csr_array], dict, dict, bool]:
    """Compatibility adapter returning the historical five-element tuple."""

    return create_pam(
        df_train=df_train,
        max_order=max_order,
        use_log=use_log,
        method=method,
        spacing_strategy=spacing_strategy,
        eliminate_diagonal=eliminate_diagonal,
        break_with_sparsity_threshold=break_with_sparsity_threshold,
        check_error_lossless=check_error_lossless,
        print_=print_,
        relation_mapping=relation_mapping,
    ).as_legacy_tuple()
