"""Compatibility API for rule-based tail prediction.

New code may use :mod:`prime_adj.prediction` for direction-agnostic ranking.
"""

from __future__ import annotations

import string

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from prime_adj.data_loading import require_held_out_split
from prime_adj.prediction import predict_links_with_ranks


def noisy_or_aggregator(confidences: list):
    return 1 - np.prod([1 - confidence for confidence in confidences])


def generate_rule_string(
    chain_of_relations: list[str], head_rel: str
) -> tuple[str, str, str]:
    letters = string.ascii_uppercase
    parts = [
        f"{relation}( {letters[index]} , {letters[index + 1]} )"
        for index, relation in enumerate(chain_of_relations)
    ]
    last = letters[len(chain_of_relations)]
    return " ^ ".join(parts) + f" -> {head_rel}( A , {last} )", "A", last


def map_dataframe(
    df: pd.DataFrame,
    node2id: dict[str, int],
    rel2id: dict[str, int],
    dropna: bool = True,
) -> pd.DataFrame:
    """Legacy label-to-number helper retained for existing callers."""

    mapped = df.copy()
    mapped["rel"] = df["rel"].map(rel2id)
    mapped["head"] = df["head"].map(node2id)
    mapped["tail"] = df["tail"].map(node2id)
    if dropna:
        mapped = mapped.dropna()
    if not mapped.empty:
        mapped["head"] = mapped["head"].astype(int)
        mapped["tail"] = mapped["tail"].astype(int)
    return mapped


def predict_tail_with_ranks(
    df_test: pd.DataFrame,
    all_rules_df: pd.DataFrame,
    pam_powers: list[csr_array],
    node2id: dict[str, int],
    rel2id: dict[str, int],
    rank_rules_by_: str = "score",
    aggregate_rules_by_str: str = "max",
) -> pd.DataFrame:
    """Legacy tail-ranking wrapper over the shared link predictor."""

    require_held_out_split(df_test)
    return predict_links_with_ranks(
        df_test,
        all_rules_df,
        pam_powers,
        node2id,
        rel2id,
        direction="tail",
        score_column=rank_rules_by_,
        aggregation=aggregate_rules_by_str,
    )


def predict_tail_with_explanations(
    df_test: pd.DataFrame,
    all_rules_df: pd.DataFrame,
    pam_powers: list[csr_array],
    node2id: dict[str, int],
    rel2id: dict[str, int],
    rank_rules_by_: str = "score",
) -> pd.DataFrame:
    """Print and return the same candidates produced by ranked prediction."""

    results = predict_tail_with_ranks(
        df_test,
        all_rules_df,
        pam_powers,
        node2id,
        rel2id,
        rank_rules_by_=rank_rules_by_,
    )
    id2node = {value: key for key, value in node2id.items()}
    id2rel = {value: key for key, value in rel2id.items()}
    for index, row in results.iterrows():
        head = id2node.get(row["head"], row["head"])
        relation = id2rel.get(row["rel"], row["rel"])
        answer = id2node.get(row["tail"], row["tail"])
        print(f"Query ({index}): {head} - {relation} - {answer}")
        if not row["predictions"]:
            print("No rules triggered for this query...\n")
        for candidate, score, evidence in zip(
            row["predictions"],
            row["predictions_scores"],
            row["candidate_evidence"],
        ):
            label = "Correct Match" if candidate == row["tail"] else "Wrong Match"
            print(
                f"({label}) {score:.4f} : "
                f"{id2node.get(candidate, candidate)} via {evidence}"
            )
    return results
