from __future__ import annotations

import pandas as pd
from scipy.sparse import csr_array

from prime_adj.prediction import generate_candidates, predict_links_with_ranks
from prime_adj.tail_prediction_with_rules import (
    predict_tail_with_explanations,
    predict_tail_with_ranks,
)
from prime_adj.utils import calculate_hits_at_k


def fixture():
    matrix = csr_array(
        ([2.0, 2.0, 2.0], ([0, 0, 1], [1, 2, 2])), shape=(3, 3)
    )
    rules = pd.DataFrame(
        [{"head_rel": 3, "body": 2.0, "hop": 1, "score": 0.5}]
    )
    nodes = {"a": 0, "b": 1, "c": 2}
    relations = {"r": 3}
    return matrix, rules, nodes, relations


def test_deterministic_ties_use_entity_id_and_work_in_both_directions():
    matrix, rules, _, _ = fixture()
    tails = generate_candidates(0, 3, rules, [matrix], direction="tail")
    heads = generate_candidates(2, 3, rules, [matrix], direction="head")
    assert [item["entity"] for item in tails] == [1, 2]
    assert [item["entity"] for item in heads] == [0, 1]


def test_filtered_head_and_tail_ranking_keep_the_correct_answer():
    matrix, rules, nodes, relations = fixture()
    known = pd.DataFrame(
        [("a", "r", "b"), ("a", "r", "c"), ("b", "r", "c")],
        columns=["head", "rel", "tail"],
    )
    tail_query = pd.DataFrame(
        [("a", "r", "c")], columns=["head", "rel", "tail"]
    )
    head_query = pd.DataFrame(
        [("b", "r", "c")], columns=["head", "rel", "tail"]
    )
    tail = predict_links_with_ranks(
        tail_query, rules, [matrix], nodes, relations, known_triples=known
    )
    head = predict_links_with_ranks(
        head_query,
        rules,
        [matrix],
        nodes,
        relations,
        direction="head",
        known_triples=known,
    )
    assert tail.loc[0, "predictions"] == [nodes["c"]]
    assert head.loc[0, "predictions"] == [nodes["b"]]
    assert tail.loc[0, "rank_correct"] == head.loc[0, "rank_correct"] == 1


def test_ranked_and_explained_legacy_apis_share_candidates(capsys):
    matrix, rules, nodes, relations = fixture()
    query = pd.DataFrame(
        [("a", "r", "c")], columns=["head", "rel", "tail"]
    )
    ranked = predict_tail_with_ranks(query, rules, [matrix], nodes, relations)
    explained = predict_tail_with_explanations(
        query, rules, [matrix], nodes, relations
    )
    assert ranked.loc[0, "predictions"] == explained.loc[0, "predictions"]
    assert "Correct Match" in capsys.readouterr().out


def test_unknown_ids_missing_answers_and_empty_metrics_are_explicit():
    matrix, rules, nodes, relations = fixture()
    queries = pd.DataFrame(
        [("unknown", "r", "c"), ("a", "r", "unknown")],
        columns=["head", "rel", "tail"],
    )
    results = predict_links_with_ranks(
        queries, rules, [matrix], nodes, relations
    )
    assert results["status"].tolist() == ["unknown_id", "missing_answer"]
    assert results["rank_correct"].tolist() == [0, 0]
    assert calculate_hits_at_k(pd.DataFrame(columns=["rank_correct"])) == {
        1: 0.0,
        3: 0.0,
        5: 0.0,
        10: 0.0,
        "UpperBound": 0.0,
    }


def test_legacy_public_imports_smoke():
    from prime_adj.data_loading import load_data
    from prime_adj.pam_creation import create_pam_matrices
    from prime_adj.path_feature_extraction import generate_path_features
    from prime_adj.rule_generation import create_ruleset

    assert all(
        callable(item)
        for item in [
            load_data,
            create_pam_matrices,
            generate_path_features,
            create_ruleset,
        ]
    )
