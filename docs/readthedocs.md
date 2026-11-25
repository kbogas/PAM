# Read the Docs status

The previously supplied address, `https://pypu.readthedocs.io/`, is not this
project. Its page metadata identifies the Read the Docs slug `pypu`, repository
`izavits/PyPU`, and package `pypu-learn`, a positive-unlabeled learning library.

The project is registered on Read the Docs with display name **PrimeAdj**, slug
`primeadj`, and canonical URL:

```text
https://primeadj.readthedocs.io/
```

Read the Docs builds the site directly from GitHub using `.readthedocs.yaml`.
That configuration installs the `bop` and `docs` extras and builds
`mkdocs.yml`. No API token or GitHub upload workflow is required. The ordinary
GitHub CI workflow independently regenerates the figures, runs the tutorials,
and performs a strict documentation build so pull requests fail before a
broken site reaches Read the Docs.

The hosted `latest` version tracks `main`. Release tags should be activated in
Read the Docs and the matching released version selected as `stable`.
