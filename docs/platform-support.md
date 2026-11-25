# Platform and dependency support

The supported stable interpreter range is Python 3.11 through 3.14. Python 3.14
is the newest verified stable minor as of 25 November 2026. CI tests every minor
in that range on Linux and macOS, runs both numeric semirings through the test
suite, checks SciPy/GraphBLAS sparse conversion, executes the example, builds
both distributions, and imports the wheel from outside the checkout.

The core numeric boundary is python-graphblas with SuiteSparse for PAM
multiplication, SciPy CSR for the public sparse boundary, and NumPy for arrays.
Torch and GPU libraries are not installation dependencies.

Optional features are isolated as extras:

- `prime-adj[lossless]` installs Numba for the experimental lossless accelerator.
- `prime-adj[bop]` installs scikit-learn for Bag of Paths transformers.
- `prime-adj[dev]` installs build and test tooling.

Python 3.15 is not declared supported while it is prerelease. It should be added
as an allowed-failure CI candidate once setup-python and all binary dependencies
publish usable prerelease artifacts, then promoted only after the same core and
optional checks pass.
