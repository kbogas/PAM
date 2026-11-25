# PAM relation mapping contract

This document fixes the terminology and behavior used by the legacy
`create_pam_matrices` API. The scalar PAM is a lossy numeric summary; canonical
relation identity and numeric encoding are separate concepts.

## Identities, weights, and reductions

- A canonical relation ID is its string label. Sorting labels is used only to
  fit a deterministic numeric map; a caller may instead supply one fitted map.
- `relation_mapping` is a relation-label → positive numeric-weight mapping. A
  graph may use any subset of a shared map. Missing labels are an error and
  unused labels remain in the returned map so independently processed graphs
  stay aligned.
- The legacy tuple's `rel2id` element contains numeric weights, not canonical
  IDs. With `use_log=True`, it contains transformed weights.
- Parallel one-hop edges are reduced by product in the first returned
  `pam_1hop_lossless` matrix and by sum in `pam_powers[0]`. Higher hops use the
  selected GraphBLAS semiring.
- In constant-one mode, the lossy one-hop sum counts individual edges, including
  duplicate triples and different parallel labels. It does not merely count
  occupied endpoint pairs.

## Built-in encodings

For sorted relations `a`, `b`, `c`, the defaults produce:

| Encoding request | Numeric map | One-hop lossy cell for parallel `a,b` | Exact decoding |
| --- | --- | --- | --- |
| `step_10` | `a=3, b=17, c=29` | `20` | one-hop product matrix only, by prime factors |
| `step_1` | `a=3, b=5, c=7` | `8` | one-hop product matrix only, by prime factors |
| `factor_2` | `a=3, b=7, c=17` | `10` | one-hop product matrix only, by prime factors |
| `natural` | `a=1, b=2, c=3` | `3` | unsupported |
| `constant` | all labels `=1` | `2` | unsupported |

Prime fitting uses the prime **strictly greater** than the current cursor. Thus
the default start cursor `2` yields `3`, not `2`. Natural numbering starts at
`1`. Additive spacing moves the cursor to `prime + X`; multiplicative spacing
moves it to `prime × X`; the next assigned value is again the prime strictly
above that cursor.

`use_log=True` applies the natural logarithm to each relation weight before the
lossy one-hop sum. Thus parallel `a,b` becomes `log(a)+log(b)`. It does not
change `pam_1hop_lossless`, whose integer product remains available for legacy
one-hop factorization. Exact decoding of floating multi-hop cells is not
supported. `log(1)=0`, so constant-one plus log is rejected rather than silently
turning present edges into multiplicative zeros.

## Multiplication and legacy translations

`method="plus_times"` uses ordinary sum-of-products. `method="plus_plus"`
extends a path by addition and reduces competing paths by addition. Encoding,
log transformation, one-hop reduction, and multi-hop semiring are independent
choices except for combinations explicitly rejected above.

The old `method="constant"` spelling is accepted with a `DeprecationWarning`
and translated to `method="plus_times", spacing_strategy="constant"`. Unknown
methods, missing shared-map labels, non-positive/non-finite map values, and
constant-plus-log are rejected before matrix construction.

## Result and stopping behavior

`create_pam` returns a `PAMResult` containing the exact one-hop matrix, ordered
lossy powers, node and relation maps, interpretation metadata, and one of three
stopping reasons: `max_order`, `sparsity_threshold`, or `empty_graph`.
`create_pam_matrices` is the compatibility adapter that converts this object to
the historical five-element tuple.

The sparsity threshold is a minimum sparse-cell fraction in `[0, 1]`; `-1`
disables it. It is checked on every newly computed power starting at hop two,
and a rejected power is not returned. Empty graphs return one empty 0×0
one-hop power. Input frames are copied before mapping. One-hop products are
formed as unbounded Python integers and checked before conversion to `int64`.
Diagonal elimination removes stored diagonal entries structurally, while other
explicit zero weights (for example `log(1)` under natural encoding) remain part
of the sparse graph.
