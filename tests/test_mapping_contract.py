from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prime_adj.pam_creation import create_pam_matrices, get_prime_map_from_rel


def frame(triples):
    return pd.DataFrame(triples, columns=["head", "rel", "tail"])


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("step_10", {"a": 3, "b": 17, "c": 29}),
        ("step_1", {"a": 3, "b": 5, "c": 7}),
        ("factor_2", {"a": 3, "b": 7, "c": 17}),
        ("natural", {"a": 1, "b": 2, "c": 3}),
    ],
)
def test_builtin_mapping_table(strategy, expected):
    relation_to_value, _ = get_prime_map_from_rel(
        ["a", "b", "c"], spacing_strategy=strategy
    )
    assert relation_to_value == expected


def test_parallel_reductions_and_log_transform_are_independent():
    triples = frame([("x", "a", "y"), ("x", "b", "y")])
    exact, powers, nodes, relations, _ = create_pam_matrices(
        triples.copy(), max_order=1, use_log=False
    )
    assert exact[nodes["x"], nodes["y"]] == 3 * 17
    assert powers[0][nodes["x"], nodes["y"]] == 3 + 17

    exact_log, powers_log, nodes_log, relations_log, _ = create_pam_matrices(
        triples.copy(), max_order=1, use_log=True
    )
    assert exact_log[nodes_log["x"], nodes_log["y"]] == 3 * 17
    assert relations_log == {"a": np.log(3), "b": np.log(17)}
    assert powers_log[0][nodes_log["x"], nodes_log["y"]] == pytest.approx(
        np.log(3) + np.log(17)
    )


def test_constant_counts_edges_and_legacy_method_translates():
    triples = frame(
        [("x", "a", "y"), ("x", "b", "y"), ("x", "a", "y")]
    )
    _, powers, nodes, relations, _ = create_pam_matrices(
        triples.copy(), max_order=1, use_log=False, spacing_strategy="constant"
    )
    assert relations == {"a": 1, "b": 1}
    assert powers[0][nodes["x"], nodes["y"]] == 3

    with pytest.deprecated_call(match="method='constant'"):
        _, legacy_powers, legacy_nodes, legacy_relations, _ = create_pam_matrices(
            triples.copy(), max_order=1, use_log=False, method="constant"
        )
    assert legacy_relations == relations
    assert legacy_powers[0][legacy_nodes["x"], legacy_nodes["y"]] == 3


def test_shared_fitted_map_aligns_relation_subsets():
    shared = {"a": 3, "b": 17, "c": 29}
    _, _, _, first, _ = create_pam_matrices(
        frame([("x", "a", "y")]),
        max_order=1,
        use_log=False,
        relation_mapping=shared,
    )
    _, _, _, second, _ = create_pam_matrices(
        frame([("p", "c", "q")]),
        max_order=1,
        use_log=False,
        relation_mapping=shared,
    )
    assert first == second == shared


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spacing_strategy": "constant", "use_log": True},
        {"relation_mapping": {"a": 3}, "use_log": False},
        {"relation_mapping": {"a": 0, "b": 2}, "use_log": False},
        {"method": "not_a_semiring", "use_log": False},
    ],
)
def test_invalid_combinations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        create_pam_matrices(
            frame([("x", "a", "y"), ("y", "b", "z")]),
            max_order=1,
            **kwargs,
        )
