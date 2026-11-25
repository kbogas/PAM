# Prime Adjacency Matrices and Bag of Paths

`prime-adj` is a CPU-oriented Python implementation of Prime Adjacency Matrices
(PAMs) for multi-relational directed graphs. PAM uses numeric relation weights
and sparse multiplication to summarize multi-hop walks in a fast and efficient manner. Bag of Paths (BoP) turns those summaries into sparse features for nodes, pairs, and graphs.

The distribution is `prime-adj`.

## Installation

Python 3.11–3.14 is supported, tested on Linux and macOS.

```bash
python -m pip install prime-adj
python -m pip install "prime-adj[bop]"       # sklearn BoP transformer
python -m pip install "prime-adj[lossless]"  # experimental Numba path
```

The backend is python-graphblas with SuiteSparse, with SciPy CSR arrays at the
public sparse boundary.

## Quickstart: triples → PAM → BoP

```python
import pandas as pd
from prime_adj.bop import BagOfPaths, BoPDocument
from prime_adj.pam_creation import create_pam

triples = pd.DataFrame(
    [("alice", "knows", "bob"),
     ("bob", "knows", "carol"),
     ("alice", "likes", "carol")],
    columns=["head", "rel", "tail"],
)
result = create_pam(triples, max_order=2, use_log=False)
signature = tuple(sorted(
    (label, float(weight)) for label, weight in result.relation_map.items()
))
documents = [
    BoPDocument(result.powers, "node", node, mapping_signature=signature)
    for node in range(len(result.node_map))
]
transformer = BagOfPaths(level="node", weighting="count", norm=None)
features = transformer.fit_transform(documents)

print("nodes:", result.node_map)
print("relations:", result.relation_map)
print("nonzeros by hop:", [pam.nnz for pam in result.powers])
print("BoP shape:", features.shape)
print("features:", transformer.get_feature_names_out().tolist())
print("counts:", features.toarray().astype(int).tolist())
```

Output:

```text
nodes: {'alice': 0, 'bob': 1, 'carol': 2}
relations: {'knows': 3, 'likes': 17}
nonzeros by hop: [3, 1]
BoP shape: (3, 6)
features: ['h1|in|v=17', 'h1|in|v=3', 'h1|out|v=17', 'h1|out|v=3', 'h2|in|v=9', 'h2|out|v=9']
counts: [[0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0]]
```

Run this as `python examples/quickstart.py`. See also
[`examples/prediction.py`](examples/prediction.py), the
[`mapping contract`](docs/mapping-contract.md), and the
[`BoP guide`](docs/bag-of-paths.md).

## Exactness limits

The integer one-hop product can recover parallel prime-encoded relations while
it fits in `int64`. Ordinary numeric multi-hop PAMs are compact but lossy:
multiplication is commutative, so reversed relation sequences can collide, and
different walk collections can share a scalar cell. The default-used log values also introduce
floating-point effects. Thus, better treat scalar BoP tokens, as features of path signatures, not proof of a unique ordered path.

## Publications

### Primary citation

**From primes to paths: Enabling fast multi-relational graph analysis** —
Konstantinos Bougiatiotis and Georgios Paliouras. *Data & Knowledge
Engineering*, article 102554, 2026. North-Holland.
[Journal article](https://doi.org/10.1016/j.datak.2026.102554), [code and experiments](https://github.com/kbogas/PAM_BoP)

### Earlier versions

The main idea was initially explored in:
- **Analysing Biomedical Knowledge Graphs using Prime Adjacency Matrices** —
  Konstantinos Bougiatiotis and Georgios Paliouras. *2023 IEEE 36th
  International Symposium on Computer-Based Medical Systems (CBMS)*,
  pp. 628–633, 2023. IEEE.
  [IEEE/DOI](https://doi.org/10.1109/CBMS58004.2023.00291)

- **Efficient Complex Network Representation Using Prime Numbers** —
  Konstantinos Bougiatiotis and Georgios Paliouras. *International Conference
  on Complex Networks and Their Applications*, pp. 75–86, 2023. Springer
  Nature Switzerland, Cham.
  [Springer/DOI](https://doi.org/10.1007/978-3-031-53472-0_7)

### Other uses

**Identifying drug repurposing candidates for rare neuro-muscular disorders,
using different AI methods on the literature knowledge graph** — Fotios
Papadimas, Stavroula Svolou, Konstantinos Bougiatiotis, Fotis Aisopos,
Anastasia Krithara, and Georgios Paliouras. *JAMIA Open* 9(2), ooag050,
2026. Oxford University Press.
[Oxford Academic/DOI](https://doi.org/10.1093/jamiaopen/ooag050)

This application uses path-based features for drug–disease link prediction on
a literature knowledge graph.

## License

This repository retains its existing dual licensing terms. Individual
non-commercial use and use by legal entities have different conditions; review
[`LICENSE.md`](LICENSE.md) before using, modifying, or distributing the code.
