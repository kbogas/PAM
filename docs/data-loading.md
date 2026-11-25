# Data loading and evaluation splits

`load_data` always returns five values with stable schemas:

1. the original, unaugmented training triples;
2. the training graph, optionally augmented with inverse edges;
3. validation triples;
4. test triples; and
5. the filter set containing the augmented training graph plus validation facts.

All four frames use `head`, `rel`, and `tail` columns. Inverse augmentation is
applied only to the training graph. `YES` reverses endpoints while retaining the
relation label; `YES__INV` also appends `__INV`. `load_csv` and `load_data` use
the same implementation.

Missing `valid.txt` or `test.txt` files now produce a warning and an empty frame.
They are never replaced by training data. This changes the pre-0.0.11 behavior,
which could silently report training performance as evaluation performance.
Train-only directories therefore load successfully with empty held-out frames;
directories missing only validation retain their genuine test split.

Before computing metrics, call `require_held_out_split(test)`. It rejects empty
or malformed splits, making the distinction between graph construction and
evaluation explicit.
