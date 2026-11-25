"""README quickstart: triples to PAM to sparse Bag of Paths features."""

import pandas as pd

from prime_adj.bop import BagOfPaths, BoPDocument
from prime_adj.pam_creation import create_pam

triples = pd.DataFrame(
    [("alice", "knows", "bob"), ("bob", "knows", "carol"), ("alice", "likes", "carol")],
    columns=["head", "rel", "tail"],
)
result = create_pam(triples, max_order=2, use_log=False)
signature = tuple(sorted((label, float(weight)) for label, weight in result.relation_map.items()))
documents = [
    BoPDocument(result.powers, "node", node, mapping_signature=signature)
    for node in range(len(result.node_map))
]
transformer = BagOfPaths(level="node", weighting="count", norm=None)
features = transformer.fit_transform(documents)

print("nodes:", result.node_map)
print("relations:", result.relation_map)
print("nonzeros by hop:", [pam.nnz for pam in result.powers])
print("BoP shape:", features.shape)
print("features:", transformer.get_feature_names_out().tolist())
print("counts:", features.toarray().astype(int).tolist())
