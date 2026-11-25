"""Sparse Bag of Paths feature extraction for PAM matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_array, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils.validation import check_is_fitted

from prime_adj.pam_creation import create_pam


ObjectKind = Literal["node", "pair", "graph"]

BOP_PRESETS = {
    "tiny": {"min_df": 1, "max_df": 1.0, "max_features": None},
    "appendix_c1": {"min_df": 2, "max_df": 0.99, "max_features": 10_000},
}


@dataclass(frozen=True)
class BoPDocument:
    """One requested object and the graph PAMs used to describe it."""

    pams: Sequence[csr_array]
    kind: ObjectKind
    selection: int | tuple[int, int] | None = None
    graph_id: Hashable | None = None
    mapping_signature: tuple[tuple[str, float], ...] | None = None


def _value_token(value) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else value.hex()


class BagOfPaths(TransformerMixin, BaseEstimator):
    """Convert requested graph objects into sparse count or TF–IDF features.

    Cell values are lossy scalar PAM tokens. Exact ordered-path tokens are
    intentionally reserved for a future exact backend.
    """

    def __init__(
        self,
        level: ObjectKind = "node",
        max_hop: int | None = None,
        weighting: Literal["count", "tfidf"] = "tfidf",
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        max_features: int | None = None,
        norm: Literal["l1", "l2"] | None = "l2",
        include_pair_directions: bool = True,
        include_endpoints: bool = True,
        neighbor_aggregation: bool = False,
        self_weight: float = 1.0,
    ):
        self.level = level
        self.max_hop = max_hop
        self.weighting = weighting
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.norm = norm
        self.include_pair_directions = include_pair_directions
        self.include_endpoints = include_endpoints
        self.neighbor_aggregation = neighbor_aggregation
        self.self_weight = self_weight

    def _validate(self, documents):
        if self.level not in {"node", "pair", "graph"}:
            raise ValueError("level must be 'node', 'pair', or 'graph'")
        if self.weighting not in {"count", "tfidf"}:
            raise ValueError("weighting must be 'count' or 'tfidf'")
        if self.max_hop is not None and self.max_hop < 1:
            raise ValueError("max_hop must be positive")
        if self.self_weight < 0:
            raise ValueError("self_weight must be non-negative")
        docs = list(documents)
        for document in docs:
            if not isinstance(document, BoPDocument):
                raise TypeError("X must contain BoPDocument objects")
            if document.kind != self.level:
                raise ValueError(
                    f"Expected {self.level!r} documents, got {document.kind!r}"
                )
            shapes = {pam.shape for pam in document.pams}
            if len(shapes) > 1 or any(shape[0] != shape[1] for shape in shapes):
                raise ValueError("All PAMs in a document must share one square shape")
        return docs

    @staticmethod
    def _cell_tokens(pam, rows, cols, hop, role):
        values = np.asarray(pam[rows, cols]).reshape(-1)
        return [
            f"h{hop}|{role}|v={_value_token(value)}"
            for value in values
        ]

    def _node_tokens(self, pams, node, prefix=""):
        tokens = []
        for hop, pam in enumerate(pams, 1):
            row = pam[[node], :]
            tokens.extend(
                f"{prefix}h{hop}|out|v={_value_token(value)}" for value in row.data
            )
            col = pam[:, [node]]
            tokens.extend(
                f"{prefix}h{hop}|in|v={_value_token(value)}" for value in col.data
            )
        return tokens

    @staticmethod
    def _stored_value(pam, row, col):
        sparse_row = pam[[row], :]
        matches = np.flatnonzero(sparse_row.indices == col)
        if not len(matches):
            return None
        return sparse_row.data[matches[0]]

    def _tokens(self, document):
        pams = tuple(document.pams[: self.max_hop])
        if document.kind == "graph":
            return [
                f"h{hop}|graph|v={_value_token(value)}"
                for hop, pam in enumerate(pams, 1)
                for value in pam.data
            ]
        if document.kind == "node":
            if not isinstance(document.selection, int):
                raise TypeError("node selection must be an integer matrix index")
            node = document.selection
            tokens = self._node_tokens(pams, node)
            if self.neighbor_aggregation and pams:
                neighbors = set(pams[0][[node], :].indices) | set(
                    pams[0][:, [node]].tocoo().row
                )
                for neighbor in sorted(neighbors - {node}):
                    tokens.extend(self._node_tokens(pams, neighbor, prefix="neighbor|"))
            return tokens

        if not (
            isinstance(document.selection, tuple)
            and len(document.selection) == 2
        ):
            raise TypeError("pair selection must be a two-integer tuple")
        head, tail = document.selection
        tokens = []
        for hop, pam in enumerate(pams, 1):
            forward = self._stored_value(pam, head, tail)
            if forward is not None:
                tokens.append(f"h{hop}|pair_out|v={_value_token(forward)}")
            if self.include_pair_directions:
                reverse = self._stored_value(pam, tail, head)
                if reverse is not None:
                    tokens.append(f"h{hop}|pair_in|v={_value_token(reverse)}")
        if self.include_endpoints:
            tokens.extend(self._node_tokens(pams, head, prefix="head|"))
            tokens.extend(self._node_tokens(pams, tail, prefix="tail|"))
        return tokens

    def _check_mapping(self, documents, *, fitting=False):
        signatures = {
            document.mapping_signature
            for document in documents
            if document.mapping_signature is not None
        }
        if len(signatures) > 1:
            raise ValueError("Documents use incompatible relation mappings")
        signature = next(iter(signatures), None)
        if fitting:
            self.mapping_signature_ = signature
        elif (
            self.mapping_signature_ is not None
            and signature is not None
            and signature != self.mapping_signature_
        ):
            raise ValueError("Transform relation mapping differs from fitted mapping")

    def fit(self, X, y=None):
        documents = self._validate(X)
        if not documents:
            raise ValueError("Cannot fit BagOfPaths on an empty selection")
        self._check_mapping(documents, fitting=True)
        token_documents = [self._tokens(document) for document in documents]
        self.vectorizer_ = CountVectorizer(
            analyzer=lambda tokens: tokens,
            lowercase=False,
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
        )
        counts = self.vectorizer_.fit_transform(token_documents)
        if self.weighting == "tfidf":
            self.tfidf_ = TfidfTransformer(norm=self.norm)
            self.tfidf_.fit(counts)
        else:
            self.tfidf_ = None
        return self

    def transform(self, X):
        check_is_fitted(self, "vectorizer_")
        documents = self._validate(X)
        self._check_mapping(documents)
        if not documents:
            return csr_matrix((0, len(self.vectorizer_.vocabulary_)), dtype=float)
        counts = self.vectorizer_.transform(
            [self._tokens(document) for document in documents]
        )
        if self.neighbor_aggregation and self.self_weight != 1:
            scale = np.array(
                [
                    1.0 if name.startswith("neighbor|") else self.self_weight
                    for name in self.get_feature_names_out()
                ]
            )
            counts = counts.multiply(scale)
        if self.weighting == "tfidf":
            return self.tfidf_.transform(counts)
        if self.norm is None:
            return counts
        return TfidfTransformer(use_idf=False, norm=self.norm).fit_transform(counts)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "vectorizer_")
        return self.vectorizer_.get_feature_names_out()


def _signature(relation_map):
    if relation_map is None:
        return None
    return tuple(sorted((str(key), float(value)) for key, value in relation_map.items()))


def generate_node_embeddings(pams, nodes, relation_map=None, **kwargs):
    documents = [
        BoPDocument(pams, "node", node, mapping_signature=_signature(relation_map))
        for node in nodes
    ]
    return BagOfPaths(level="node", **kwargs).fit_transform(documents)


def generate_edge_embeddings(pams, pairs, relation_map=None, **kwargs):
    documents = [
        BoPDocument(pams, "pair", pair, mapping_signature=_signature(relation_map))
        for pair in pairs
    ]
    return BagOfPaths(level="pair", **kwargs).fit_transform(documents)


def generate_graph_embeddings(graph_pams, relation_map=None, **kwargs):
    documents = [
        BoPDocument(pams, "graph", mapping_signature=_signature(relation_map))
        for pams in graph_pams
    ]
    return BagOfPaths(level="graph", **kwargs).fit_transform(documents)


def generate_subgraph_embeddings(
    graph: pd.DataFrame, node_sets, *, max_hop=3, relation_map=None, **kwargs
):
    graph_pams = []
    for nodes in node_sets:
        nodes = set(nodes)
        induced = graph[
            graph["head"].isin(nodes) & graph["tail"].isin(nodes)
        ].copy()
        result = create_pam(
            induced,
            max_order=max_hop,
            use_log=False,
            relation_mapping=relation_map,
        )
        graph_pams.append(result.powers)
    return generate_graph_embeddings(
        graph_pams, relation_map=relation_map, max_hop=max_hop, **kwargs
    )
