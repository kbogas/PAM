# Multiple graphs

Fit one relation map on the union of training relation labels and reuse it for
every graph. Local node IDs and graph IDs are not included in tokens.

```bash
python examples/tutorial_multigraph.py
```

Expected output: `(2, 2) 2`.

See
[`examples/tutorial_multigraph.py`](https://github.com/kbogas/PAM/blob/main/examples/tutorial_multigraph.py).
