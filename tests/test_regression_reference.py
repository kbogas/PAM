from __future__ import annotations

import pandas as pd

from prime_adj.pam_creation import create_pam_matrices
from prime_adj.rule_generation import create_ruleset
from prime_adj.tail_prediction_with_rules import predict_tail_with_ranks
from tests.reference_walks import encode_plus_times, enumerate_walks


DUMMY_TRAIN = [
    ("Kostas", "knows", "Nick"),
    ("Nick", "knows", "George"),
    ("Kostas", "knows", "George"),
    ("George", "knows", "Anna"),
    ("Kostas", "lives_in", "Athens"),
    ("Kostas", "works_in", "Athens"),
    ("George", "works_in", "Athens"),
    ("Nick", "lives_in", "Athens"),
    ("Anna", "lives_in", "Athens"),
]
DUMMY_TEST = [
    ("Nick", "knows", "Anna"),
    ("George", "lives_in", "Athens"),
]


def frame(triples):
    return pd.DataFrame(triples, columns=["head", "rel", "tail"])


def coordinates(matrix):
    coo = matrix.tocoo()
    return list(zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()))


def test_shipped_dummy_workflow_snapshot():
    one_hop, powers, nodes, relations, stopped = create_pam_matrices(
        frame(DUMMY_TRAIN), use_log=False, max_order=3
    )
    rules = create_ruleset(one_hop, powers, use_log=False)
    results = predict_tail_with_ranks(
        frame(DUMMY_TEST), rules, [one_hop] + powers[1:], nodes, relations
    )

    assert nodes == {"Anna": 0, "Athens": 1, "George": 2, "Kostas": 3, "Nick": 4}
    assert relations == {"knows": 3, "lives_in": 17, "works_in": 29}
    assert stopped is False
    assert coordinates(one_hop) == [
        (0, 1, 17),
        (2, 0, 3),
        (2, 1, 29),
        (3, 1, 493),
        (3, 2, 3),
        (3, 4, 3),
        (4, 1, 17),
        (4, 2, 3),
    ]
    assert coordinates(powers[1]) == [
        (2, 1, 51.0),
        (3, 0, 9.0),
        (3, 1, 138.0),
        (3, 2, 9.0),
        (4, 0, 9.0),
        (4, 1, 87.0),
    ]
    assert coordinates(powers[2]) == [
        (3, 0, 27.0),
        (3, 1, 414.0),
        (4, 1, 153.0),
    ]
    assert rules.groupby("hop").size().to_dict() == {1: 2, 2: 5, 3: 3}
    assert results["predictions"].tolist() == [[nodes["Anna"]], [nodes["Athens"]]]
    assert results["rank_correct"].tolist() == [1, 1]


def test_reference_handles_parallel_duplicate_reverse_loop_isolate_and_disconnect():
    triples = [
        ("a", "r", "b"),
        ("a", "s", "b"),  # parallel relation
        ("a", "r", "b"),  # duplicate triple
        ("b", "t", "a"),  # reverse edge
        ("b", "u", "b"),  # self-loop
        ("x", "v", "y"),  # disconnected pair
    ]
    walks = enumerate_walks(triples, 2, nodes=["isolated"])

    assert walks[0][("a", "b")][("r",)] == 2
    assert walks[0][("a", "b")][("s",)] == 1
    assert walks[1][("a", "a")][("r", "t")] == 2
    assert walks[1][("a", "a")][("s", "t")] == 1
    assert walks[1][("b", "b")][("u", "u")] == 1
    assert all("isolated" not in endpoints for hop in walks for endpoints in hop)
    assert ("x", "y") in walks[0]
    assert ("x", "y") not in walks[1]

    _, powers, nodes, relations, _ = create_pam_matrices(
        frame(triples), use_log=False, max_order=2
    )
    expected = encode_plus_times(walks[1], relations)
    for (head, tail), value in expected.items():
        assert powers[1][nodes[head], nodes[tail]] == value


def test_reference_exposes_reversed_chain_collision():
    triples = [
        ("a", "r", "m1"),
        ("m1", "s", "z"),
        ("b", "s", "m2"),
        ("m2", "r", "z"),
    ]
    walks = enumerate_walks(triples, 2)
    _, powers, nodes, _, _ = create_pam_matrices(
        frame(triples), use_log=False, max_order=2
    )

    assert walks[1][("a", "z")] == {("r", "s"): 1}
    assert walks[1][("b", "z")] == {("s", "r"): 1}
    assert powers[1][nodes["a"], nodes["z"]] == powers[1][nodes["b"], nodes["z"]]
