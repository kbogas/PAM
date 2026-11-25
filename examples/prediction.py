"""Self-contained triples → PAM → rules → deterministic tail ranks example."""

import pandas as pd

from prime_adj.pam_creation import create_pam
from prime_adj.prediction import predict_links_with_ranks
from prime_adj.rule_generation import create_ruleset


train = pd.DataFrame(
    [
        ("alice", "knows", "bob"),
        ("bob", "knows", "carol"),
        ("alice", "likes", "carol"),
    ],
    columns=["head", "rel", "tail"],
)
query = pd.DataFrame(
    [("alice", "likes", "carol")], columns=["head", "rel", "tail"]
)

result = create_pam(train, max_order=2, use_log=False)
rules = create_ruleset(result.one_hop_exact, list(result.powers), use_log=False)
ranks = predict_links_with_ranks(
    query,
    rules,
    [result.one_hop_exact, *result.powers[1:]],
    result.node_map,
    result.relation_map,
    known_triples=train,
)
print(ranks[["predictions", "rank_correct", "status"]].to_string(index=False))
