# Sparse Bag of Paths

`prime_adj.bop.BagOfPaths` is an sklearn transformer that turns one requested
object into one sparse feature row. A `BoPDocument` binds ordered hop matrices
to a node, pair, or whole graph. `graph_id` qualifies selections in multi-graph
batches but is deliberately excluded from feature tokens, so local node IDs do
not affect column alignment.

Tokens are namespaced by hop and role. Nodes distinguish incoming and outgoing
cells. Pairs distinguish both pair directions and head/tail endpoint features.
Graphs pool all structurally stored cells. These are explicitly lossy scalar
cell tokens; the interface reserves exact ordered-path tokens for the later
lossless backend.

`fit` learns vocabulary, document frequencies and IDF only from its input.
`transform` reuses them, ignores unseen tokens, preserves requested order and
duplicates, and returns a correctly shaped sparse matrix for an empty request.
Provided mapping signatures must agree across fit and transform. Use one shared
fitted PAM relation map for graphs with overlapping relation subsets.

Parameters expose hop cutoff, count or TF–IDF weighting, `min_df`, `max_df`,
feature cap, normalization, optional pair directions/endpoints, and optional
one-hop node-neighbor aggregation with self weighting. `BOP_PRESETS["tiny"]`
works for small collections. `BOP_PRESETS["appendix_c1"]` records the paper's
Appendix C.1 vocabulary settings (`min_df=2`, `max_df=0.99`, 10,000 features);
dataset-specific hop and self weights must still be selected for the task.

The one-shot helpers are `generate_node_embeddings`,
`generate_edge_embeddings`, `generate_graph_embeddings`, and
`generate_subgraph_embeddings`. The subgraph helper filters triples to the
induced node set and recomputes every walk, preventing paths from leaving and
re-entering the selected subgraph.
