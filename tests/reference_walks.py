"""Small, dependency-free reference implementation for exact directed walks."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable

Node = Hashable
Relation = Hashable
Triple = tuple[Node, Relation, Node]
WalkCounts = dict[tuple[Node, Node], Counter[tuple[Relation, ...]]]


def enumerate_walks(
    triples: Iterable[Triple],
    max_hop: int,
    *,
    nodes: Iterable[Node] = (),
) -> list[WalkCounts]:
    """Count exact ordered relation sequences for every endpoint pair and hop.

    Triples are deliberately not deduplicated: repeated triples and parallel
    relations contribute their full multiplicity. ``nodes`` lets fixtures name
    isolates, which consequently have no entries in the returned sparse maps.
    """

    if max_hop < 1:
        raise ValueError("max_hop must be at least 1")

    edges: dict[Node, list[tuple[Relation, Node]]] = defaultdict(list)
    known_nodes = set(nodes)
    triples = list(triples)
    for head, relation, tail in triples:
        known_nodes.update((head, tail))
        edges[head].append((relation, tail))

    current: dict[tuple[Node, Node, tuple[Relation, ...]], int] = defaultdict(int)
    for head, relation, tail in triples:
        current[(head, tail, (relation,))] += 1

    result: list[WalkCounts] = []
    for hop in range(1, max_hop + 1):
        counts: dict[tuple[Node, Node], Counter[tuple[Relation, ...]]] = defaultdict(
            Counter
        )
        for (head, tail, relations), multiplicity in current.items():
            counts[(head, tail)][relations] += multiplicity
        result.append(dict(counts))

        if hop == max_hop:
            break
        following: dict[tuple[Node, Node, tuple[Relation, ...]], int] = defaultdict(
            int
        )
        for (head, middle, relations), multiplicity in current.items():
            for relation, tail in edges.get(middle, ()):
                following[(head, tail, relations + (relation,))] += multiplicity
        current = following

    return result


def encode_plus_times(
    walks: WalkCounts, relation_values: dict[Relation, int]
) -> dict[tuple[Node, Node], int]:
    """Encode exact signatures using the numeric plus-times PAM recurrence."""

    encoded = {}
    for endpoints, signatures in walks.items():
        value = 0
        for relations, multiplicity in signatures.items():
            product = 1
            for relation in relations:
                product *= relation_values[relation]
            value += multiplicity * product
        encoded[endpoints] = value
    return encoded
