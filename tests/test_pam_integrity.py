from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prime_adj.pam_creation import PAMResult, create_pam, create_pam_matrices


def frame(triples=()):
    return pd.DataFrame(triples, columns=["head", "rel", "tail"])


def test_input_is_not_mutated_and_repeated_calls_are_stable():
    triples = frame([("a", "r", "b"), ("b", "s", "c")])
    original = triples.copy(deep=True)
    first = create_pam_matrices(triples, max_order=2, use_log=False)
    second = create_pam_matrices(triples, max_order=2, use_log=False)

    pd.testing.assert_frame_equal(triples, original)
    assert first[2:] == second[2:]
    for left, right in zip(first[1], second[1]):
        assert (left != right).nnz == 0


def test_one_hop_overflow_is_detected_before_fixed_width_cast():
    triples = frame([("a", "r", "b"), ("a", "s", "b")])
    with pytest.raises(OverflowError):
        create_pam_matrices(
            triples,
            max_order=1,
            use_log=False,
            relation_mapping={"r": 3_037_000_500, "s": 3_037_000_501},
        )


def test_typed_empty_result_and_legacy_adapter():
    result = create_pam(frame(), max_order=3, use_log=False)
    assert isinstance(result, PAMResult)
    assert result.stopping_reason == "empty_graph"
    assert result.one_hop_exact.shape == (0, 0)
    assert len(result.powers) == 1
    assert result.as_legacy_tuple()[-1] is False


def test_sparsity_fraction_can_stop_before_hop_two_is_appended():
    result = create_pam(
        frame([("a", "r", "b"), ("b", "s", "c")]),
        max_order=3,
        use_log=False,
        break_with_sparsity_threshold=0.9,
    )
    assert result.stopping_reason == "sparsity_threshold"
    assert len(result.powers) == 1
    assert result.metadata.minimum_sparsity == 0.9


def test_diagonal_elimination_is_structural_under_plus_plus():
    triples = frame(
        [("a", "r", "a"), ("a", "s", "b"), ("b", "t", "c")]
    )
    kept = create_pam(
        triples, max_order=2, use_log=False, method="plus_plus"
    )
    removed = create_pam(
        triples,
        max_order=2,
        use_log=False,
        method="plus_plus",
        eliminate_diagonal=True,
    )
    a, b = kept.node_map["a"], kept.node_map["b"]
    assert kept.powers[1][a, b] != 0
    assert removed.powers[1][a, b] == 0
    assert (a, b) not in set(zip(*removed.powers[1].nonzero()))


def test_explicit_log_zero_edge_is_preserved():
    result = create_pam(
        frame([("a", "first", "b"), ("b", "second", "c")]),
        max_order=2,
        use_log=True,
        spacing_strategy="natural",
        method="plus_plus",
    )
    a, b, c = (result.node_map[node] for node in ("a", "b", "c"))
    assert result.powers[0][a, b] == 0
    assert (a, b) in set(zip(*result.powers[0].nonzero())) or result.powers[0].nnz == 2
    assert result.powers[1][a, c] == pytest.approx(np.log(2))


@pytest.mark.parametrize("threshold", [-2, 1.01])
def test_invalid_sparsity_fraction_is_rejected(threshold):
    with pytest.raises(ValueError, match="minimum sparsity fraction"):
        create_pam(
            frame([("a", "r", "b")]),
            use_log=False,
            break_with_sparsity_threshold=threshold,
        )
