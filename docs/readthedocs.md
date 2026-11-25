# Read the Docs status

The previously supplied address, `https://pypu.readthedocs.io/`, is not this
project. Its page metadata identifies the Read the Docs slug `pypu`, repository
`izavits/PyPU`, and package `pypu-learn`, a positive-unlabeled learning library.

The project is registered on Read the Docs with display name **PrimeAdj**, slug
`primeadj`, and canonical URL:

```text
https://primeadj.readthedocs.io/
```

The project uses Read the Docs direct upload. `.github/workflows/docs.yml`
builds the MkDocs site, publishes branch, tag and same-repository pull-request
versions, updates `latest` from `main`, and updates `stable` from every `v*`
release tag. The GitHub repository must contain an Actions secret named
`READTHEDOCS_TOKEN` holding a project token with only the **Upload
documentation** permission. Never commit or paste that token into project
files.

The upload action is pinned to an immutable commit. Fork pull requests cannot
access the upload token; the ordinary CI workflow still builds their docs.
`.readthedocs.yaml` remains available if the project later switches from direct
upload to native Read the Docs builds.
