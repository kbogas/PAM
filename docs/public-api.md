# Public API and prediction behavior

## PAM construction

`create_pam(triples, ...) -> PAMResult` is the typed API. `PAMResult.powers` is
an ordered tuple of square SciPy CSR arrays from one hop upward; `node_map` maps
labels to zero-based rows/columns; `relation_map` contains numeric weights;
metadata describes encoding and semiring; and `stopping_reason` is explicit.

`create_pam_matrices` remains the legacy adapter returning
`(one_hop_exact, powers_list, node_map, relation_map, stopped_for_sparsity)`.
Existing imports remain supported.

## Link prediction

`generate_candidates` is the shared candidate engine.
`predict_links_with_ranks(..., direction="tail" | "head")` ranks either side.
Candidates sort by descending aggregate score and then ascending numeric entity
ID, making ties deterministic. `max` and `noisy_or` aggregation are supported.

Known facts may be supplied as a frame or iterable of `(head, relation, tail)`
labels. Indexed filtering removes other known answers but always retains the
query's correct answer. Unknown query-side IDs yield `status="unknown_id"`;
unknown answers yield `status="missing_answer"`; both receive rank zero.
Empty metric input returns zero for every requested Hits@K and the upper bound.

The legacy `predict_tail_with_ranks` and `predict_tail_with_explanations` imports
delegate to the same engine, so their candidates and scores agree. Explained
prediction now also returns the result frame after printing it.

## Examples

Run `python examples/prediction.py` from a source checkout or installed example
bundle. It is self-contained and does not depend on repository data paths.

## Optimization benchmark

`benchmarks/prediction_filtering.py` compares indexed filtering with repeated
frame scans and serial with threaded candidate generation. On the development
Mac/Python 3.14 fixture, indexed filtering took 0.0010 s versus 0.1721 s for
scans. Serial candidate generation took 0.2421 s versus 0.3086 s with threads.
The indexed semantic filter is retained; parallelism is not enabled because it
was slower on this workload. Broad benchmark claims are intentionally avoided.
