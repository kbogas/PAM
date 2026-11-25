import pandas as pd

from prime_adj.bop import generate_subgraph_embeddings

graph = pd.DataFrame(
    [("a", "r", "b"), ("b", "s", "a"), ("a", "r", "c")],
    columns=["head", "rel", "tail"],
)
X = generate_subgraph_embeddings(
    graph, [{"a", "c"}, {"a", "b"}], max_hop=2, weighting="count", norm=None
)
print(X.shape, X.nnz)
