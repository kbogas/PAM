import pandas as pd

from prime_adj.bop import generate_edge_embeddings
from prime_adj.pam_creation import create_pam

triples = pd.DataFrame(
    [("a", "r", "b"), ("b", "s", "c"), ("c", "r", "a")],
    columns=["head", "rel", "tail"],
)
result = create_pam(triples, max_order=2, use_log=False)
pairs = [(result.node_map["a"], result.node_map["c"]), (result.node_map["c"], result.node_map["a"])]
X = generate_edge_embeddings(
    result.powers, pairs, result.relation_map, weighting="count", norm=None
)
print(X.shape, X.nnz)
