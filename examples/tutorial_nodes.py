import pandas as pd

from prime_adj.bop import generate_node_embeddings
from prime_adj.pam_creation import create_pam

triples = pd.DataFrame(
    [("a", "knows", "b"), ("b", "likes", "c"), ("c", "knows", "a")],
    columns=["head", "rel", "tail"],
)
result = create_pam(triples, max_order=2, use_log=False)
nodes = [result.node_map[name] for name in ["a", "c", "a"]]
X = generate_node_embeddings(
    result.powers, nodes, result.relation_map, weighting="count", norm=None
)
print(X.shape, X.nnz)
