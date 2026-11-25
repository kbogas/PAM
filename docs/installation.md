# Installation and migration

## Installation

```bash
python -m pip install prime-adj
python -m pip install "prime-adj[bop]"       # sparse sklearn transformer
python -m pip install "prime-adj[lossless]"  # experimental Numba backend
```

For local development and documentation:

```bash
python -m pip install -e ".[bop,lossless,dev,docs]"
mkdocs build --strict
```

## Migration from 0.0.10

- Use `create_pam(...) -> PAMResult` for typed matrices, maps, metadata, and a
  stopping reason. `create_pam_matrices(...)` remains the five-tuple adapter.
- `python-graphblas[default]` replaces the misspelled direct SuiteSparse
  dependency.
- Missing validation/test files produce empty frames instead of copied training
  data. Evaluation requires a genuine non-empty held-out split.
- `method="constant"` is deprecated. Use `method="plus_times"` with
  `spacing_strategy="constant"`.
- Sparsity thresholds are fractions in `[0, 1]`; `-1` disables the check.
- Ranked and explained tail prediction now share deterministic candidates.
  `predict_links_with_ranks` also supports head prediction and filtering.

No distribution or import name changed: install `prime-adj`, import `prime_adj`.
