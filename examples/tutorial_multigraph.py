import pandas as pd

from prime_adj.bop import BagOfPaths, BoPDocument
from prime_adj.pam_creation import create_pam

shared = {"r": 3, "s": 17}
graphs = [
    pd.DataFrame([("a", "r", "b")], columns=["head", "rel", "tail"]),
    pd.DataFrame([("x", "s", "y")], columns=["head", "rel", "tail"]),
]
signature = tuple((key, float(value)) for key, value in shared.items())
documents = []
for graph_id, graph in enumerate(graphs):
    result = create_pam(graph, max_order=1, use_log=False, relation_mapping=shared)
    documents.append(BoPDocument(result.powers, "graph", graph_id=graph_id, mapping_signature=signature))
transformer = BagOfPaths(level="graph", weighting="count", norm=None)
X = transformer.fit_transform(documents)
print(X.shape, X.nnz)
