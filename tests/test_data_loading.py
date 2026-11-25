from __future__ import annotations

import pandas as pd
import pytest

from prime_adj.data_loading import (
    TRIPLE_COLUMNS,
    load_csv,
    load_data,
    require_held_out_split,
)
from prime_adj.tail_prediction_with_rules import predict_tail_with_ranks


def write_split(folder, name, rows):
    (folder / f"{name}.txt").write_text(
        "".join(",".join(row) + "\n" for row in rows), encoding="utf-8"
    )


def records(frame):
    return set(frame.itertuples(index=False, name=None))


@pytest.mark.parametrize(
    ("mode", "inverse_relation"),
    [("YES", "r"), ("YES__INV", "r__INV")],
)
def test_load_csv_and_load_data_share_inverse_logic(tmp_path, mode, inverse_relation):
    write_split(tmp_path, "train", [("a", "r", "b")])
    original_csv, augmented_csv = load_csv(
        str(tmp_path / "train.txt"), add_inverse_edges=mode
    )
    with pytest.warns(UserWarning, match="valid.txt"):
        with pytest.warns(UserWarning, match="test.txt"):
            original_data, augmented_data, valid, test, seen = load_data(
                str(tmp_path), "fixture", add_inverse_edges=mode, sep=","
            )

    pd.testing.assert_frame_equal(original_csv, original_data)
    pd.testing.assert_frame_equal(augmented_csv, augmented_data)
    assert records(augmented_data) == {
        ("a", "r", "b"),
        ("b", inverse_relation, "a"),
    }
    assert seen == records(augmented_data)
    assert valid.empty and test.empty
    assert list(valid.columns) == list(test.columns) == TRIPLE_COLUMNS


@pytest.mark.parametrize("mode", ["NO", "YES", "YES__INV"])
def test_fully_split_directory_loads_splits_in_every_inverse_mode(tmp_path, mode):
    train_rows = [("a", "r", "b")]
    valid_rows = [("b", "s", "c")]
    test_rows = [("c", "t", "d")]
    write_split(tmp_path, "train", train_rows)
    write_split(tmp_path, "valid", valid_rows)
    write_split(tmp_path, "test", test_rows)

    original, train, valid, test, seen = load_data(
        str(tmp_path), "fixture", add_inverse_edges=mode, sep=","
    )

    assert records(original) == set(train_rows)
    assert records(valid) == set(valid_rows)
    assert records(test) == set(test_rows)
    assert seen == records(train) | set(valid_rows)
    assert list(original.columns) == list(train.columns) == TRIPLE_COLUMNS
    assert list(valid.columns) == list(test.columns) == TRIPLE_COLUMNS


def test_missing_validation_does_not_replace_it_with_training_data(tmp_path):
    write_split(tmp_path, "train", [("a", "r", "b")])
    write_split(tmp_path, "test", [("b", "s", "c")])

    with pytest.warns(UserWarning, match="valid.txt"):
        original, train, valid, test, seen = load_data(
            str(tmp_path), "fixture", sep=","
        )

    assert valid.empty
    assert records(test) == {("b", "s", "c")}
    assert seen == records(train) == records(original)


def test_evaluation_requires_a_real_nonempty_split(tmp_path):
    empty = pd.DataFrame(columns=TRIPLE_COLUMNS)
    with pytest.raises(ValueError, match="held-out test"):
        require_held_out_split(empty)

    held_out = pd.DataFrame([("a", "r", "b")], columns=TRIPLE_COLUMNS)
    assert require_held_out_split(held_out) is held_out

    with pytest.raises(ValueError, match="held-out test"):
        predict_tail_with_ranks(empty, pd.DataFrame(), [], {}, {})


def test_invalid_inverse_mode_is_rejected(tmp_path):
    write_split(tmp_path, "train", [("a", "r", "b")])
    with pytest.raises(ValueError, match="add_inverse_edges"):
        load_csv(str(tmp_path / "train.txt"), add_inverse_edges="MAYBE")
