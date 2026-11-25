"""Shared deterministic head/tail candidate generation and ranking."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

Direction = Literal["head", "tail"]


def noisy_or(scores):
    return float(1 - np.prod([1 - score for score in scores]))


def build_filter_indexes(known_triples, node2id, rel2id):
    """Index known facts for constant-time filtered head and tail lookup."""

    tails = defaultdict(set)
    heads = defaultdict(set)
    if known_triples is None:
        return tails, heads
    rows = (
        known_triples.itertuples(index=False, name=None)
        if isinstance(known_triples, pd.DataFrame)
        else known_triples
    )
    for head, relation, tail in rows:
        if head in node2id and tail in node2id and relation in rel2id:
            h, r, t = node2id[head], rel2id[relation], node2id[tail]
            tails[(h, r)].add(t)
            heads[(r, t)].add(h)
    return tails, heads


def generate_candidates(
    query_entity: int,
    query_relation,
    rules: pd.DataFrame,
    pam_powers: list[csr_array],
    *,
    direction: Direction = "tail",
    score_column: str = "score",
    aggregation: Literal["max", "noisy_or"] = "max",
):
    """Return deterministically ranked candidate dictionaries for one query."""

    if direction not in {"head", "tail"}:
        raise ValueError("direction must be 'head' or 'tail'")
    if aggregation not in {"max", "noisy_or"}:
        raise ValueError("aggregation must be 'max' or 'noisy_or'")
    if score_column not in rules.columns:
        raise ValueError(f"Unknown rule score column: {score_column!r}")

    matching = rules[rules["head_rel"] == query_relation]
    evidence = defaultdict(list)
    for hop, hop_rules in matching.groupby("hop", sort=True):
        if hop < 1 or hop > len(pam_powers):
            continue
        body_scores = (
            hop_rules.groupby("body", sort=False)[score_column].max().to_dict()
        )
        matrix = pam_powers[int(hop) - 1]
        vector = (
            matrix[[query_entity], :]
            if direction == "tail"
            else matrix[:, [query_entity]].tocoo()
        )
        if direction == "tail":
            entities, values = vector.indices, vector.data
        else:
            entities, values = vector.row, vector.data
        for entity, body in zip(entities, values):
            if body in body_scores:
                evidence[int(entity)].append(
                    {"body": body, "hop": int(hop), "score": body_scores[body]}
                )

    candidates = []
    for entity, items in evidence.items():
        scores = [item["score"] for item in items]
        score = max(scores) if aggregation == "max" else noisy_or(scores)
        candidates.append({"entity": entity, "score": score, "evidence": items})
    return sorted(candidates, key=lambda item: (-item["score"], item["entity"]))


def predict_links_with_ranks(
    queries: pd.DataFrame,
    rules: pd.DataFrame,
    pam_powers: list[csr_array],
    node2id: dict[str, int],
    rel2id: dict[str, int | float],
    *,
    direction: Direction = "tail",
    score_column: str = "score",
    aggregation: Literal["max", "noisy_or"] = "max",
    known_triples=None,
) -> pd.DataFrame:
    """Rank heads or tails, retaining invalid queries with explicit status."""

    required = {"head", "rel", "tail"}
    if not required.issubset(queries.columns):
        raise ValueError(f"queries must contain columns {sorted(required)}")
    tail_filter, head_filter = build_filter_indexes(known_triples, node2id, rel2id)
    results = []
    for row in queries.itertuples(index=False):
        raw = row._asdict()
        head = node2id.get(raw["head"])
        relation = rel2id.get(raw["rel"])
        tail = node2id.get(raw["tail"])
        answer = tail if direction == "tail" else head
        query_entity = head if direction == "tail" else tail
        status = "ok"
        if relation is None or query_entity is None:
            status = "unknown_id"
            candidates = []
        else:
            candidates = generate_candidates(
                query_entity,
                relation,
                rules,
                pam_powers,
                direction=direction,
                score_column=score_column,
                aggregation=aggregation,
            )
        if answer is None:
            status = "missing_answer" if status == "ok" else status
        elif relation is not None and query_entity is not None:
            blocked = (
                tail_filter[(head, relation)]
                if direction == "tail"
                else head_filter[(relation, tail)]
            )
            candidates = [
                item
                for item in candidates
                if item["entity"] == answer or item["entity"] not in blocked
            ]
        entities = [item["entity"] for item in candidates]
        scores = [item["score"] for item in candidates]
        result = {
            "head": head,
            "rel": relation,
            "tail": tail,
            "predictions": entities,
            "predictions_scores": scores,
            "candidate_evidence": [item["evidence"] for item in candidates],
            "rank_correct": entities.index(answer) + 1 if answer in entities else 0,
            "status": status,
            "direction": direction,
        }
        results.append(result)
    return pd.DataFrame(results)
