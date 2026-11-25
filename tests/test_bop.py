from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from prime_adj.bop import (
    BagOfPaths,
    BoPDocument,
    generate_edge_embeddings,
    generate_graph_embeddings,
    generate_node_embeddings,
    generate_subgraph_embeddings,
)


def sample_pams():
    return (
        csr_array([[0, 2, 0], [3, 0, 5], [0, 0, 0]], dtype=float),
        csr_array([[6, 0, 10], [0, 6, 0], [0, 0, 0]], dtype=float),
    )


def test_node_count_features_preserve_order_duplicates_and_isolates():
    pams = sample_pams()
    docs = [
        BoPDocument(pams, "node", node, graph_id="g")
        for node in [1, 0, 1, 2]
    ]
    transformer = BagOfPaths(level="node", weighting="count", norm=None)
    features = transformer.fit_transform(docs)
    names = transformer.get_feature_names_out().tolist()

    assert features.shape == (4, len(names))
    np.testing.assert_array_equal(features[0].toarray(), features[2].toarray())
    assert features[3].nnz == 2  # incoming hop-one 5 and hop-two 10
    assert "h1|out|v=3" in names
    assert "h1|in|v=2" in names
    assert "h2|out|v=6" in names


def test_tfidf_is_fit_only_on_training_documents_and_unseen_tokens_are_ignored():
    pams = sample_pams()
    train = [BoPDocument(pams, "node", node) for node in [0, 1]]
    transformer = BagOfPaths(level="node", weighting="tfidf", norm=None).fit(train)
    names_before = transformer.get_feature_names_out().copy()
    transformed = transformer.transform([BoPDocument(pams, "node", 1)])

    assert transformed.shape[1] == len(names_before)
    np.testing.assert_array_equal(names_before, transformer.get_feature_names_out())
    # h1 outgoing value 5 occurs in one of two training documents.
    index = names_before.tolist().index("h1|out|v=5")
    assert transformed[0, index] == pytest.approx(np.log(3 / 2) + 1)


def test_pair_features_namespace_directions_and_endpoint_roles():
    pams = sample_pams()
    transformer = BagOfPaths(level="pair", weighting="count", norm=None)
    matrix = transformer.fit_transform([BoPDocument(pams, "pair", (0, 1))])
    names = set(transformer.get_feature_names_out())

    assert matrix.shape[0] == 1
    assert "h1|pair_out|v=2" in names
    assert "h1|pair_in|v=3" in names
    assert "head|h1|out|v=2" in names
    assert "tail|h1|out|v=3" in names


def test_sklearn_clone_empty_transform_and_mapping_validation():
    signature = (("r", 3.0), ("s", 17.0))
    docs = [BoPDocument(sample_pams(), "node", 0, mapping_signature=signature)]
    original = BagOfPaths(level="node", max_hop=1, weighting="count", norm=None)
    copied = clone(original)
    copied.fit(docs)
    pipeline = Pipeline([("bop", clone(original))])
    assert pipeline.fit_transform(docs).shape[0] == 1
    assert copied.transform([]).shape == (0, len(copied.get_feature_names_out()))
    with pytest.raises(ValueError, match="differs"):
        copied.transform(
            [
                BoPDocument(
                    sample_pams(),
                    "node",
                    0,
                    mapping_signature=(("r", 5.0),),
                )
            ]
        )


def test_one_shot_helpers_return_sparse_rows_per_requested_object():
    pams = sample_pams()
    assert generate_node_embeddings(
        pams, [0, 0, 2], weighting="count", norm=None
    ).shape[0] == 3
    assert generate_edge_embeddings(
        pams, [(0, 1), (1, 0)], weighting="count", norm=None
    ).shape[0] == 2
    assert generate_graph_embeddings(
        [pams, pams], weighting="count", norm=None
    ).shape[0] == 2


def test_subgraphs_recompute_walks_without_leave_and_reenter_paths():
    graph = pd.DataFrame(
        [("a", "r", "b"), ("b", "s", "a"), ("a", "r", "c")],
        columns=["head", "rel", "tail"],
    )
    # {a,c} contains one direct edge; the a-b-a two-hop walk must not survive.
    features = generate_subgraph_embeddings(
        graph,
        [{"a", "c"}, {"a", "b"}],
        max_hop=2,
        weighting="count",
        norm=None,
    )
    assert features.shape[0] == 2
    assert features[0].sum() == 1
    assert features[1].sum() > features[0].sum()
