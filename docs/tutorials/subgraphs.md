# Induced subgraph embeddings

Subgraphs are not slices of global PAMs. Triples are filtered to each selected
node set first, then walks are recomputed so a path cannot leave and re-enter.

```bash
python examples/tutorial_subgraphs.py
```

Expected output: `(2, 3) 4`.

See
[`examples/tutorial_subgraphs.py`](https://github.com/kbogas/PAM/blob/main/examples/tutorial_subgraphs.py).
