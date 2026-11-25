"""Microbenchmark indexed filtering and serial/threaded candidate generation."""

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import pandas as pd
from scipy.sparse import csr_array

from prime_adj.prediction import build_filter_indexes, generate_candidates


N = 2_000
known = pd.DataFrame(
    [(f"n{i % 100}", "r", f"n{(i * 7) % 100}") for i in range(N)],
    columns=["head", "rel", "tail"],
)
nodes = {f"n{i}": i for i in range(100)}
relations = {"r": 3}
queries = [(i % 100, 3) for i in range(1_000)]

start = perf_counter()
tail_index, _ = build_filter_indexes(known, nodes, relations)
for head, relation in queries:
    _ = tail_index[(head, relation)]
indexed_seconds = perf_counter() - start

start = perf_counter()
for head, relation in queries:
    head_label = f"n{head}"
    _ = known[(known["head"] == head_label) & (known["rel"] == "r")]
scan_seconds = perf_counter() - start

matrix = csr_array(([2.0] * 99, ([0] * 99, list(range(1, 100)))), shape=(100, 100))
rules = pd.DataFrame([{"head_rel": 3, "body": 2.0, "hop": 1, "score": 0.5}])

start = perf_counter()
for _ in queries:
    generate_candidates(0, 3, rules, [matrix])
serial_seconds = perf_counter() - start

start = perf_counter()
with ThreadPoolExecutor() as pool:
    list(pool.map(lambda _: generate_candidates(0, 3, rules, [matrix]), queries))
threaded_seconds = perf_counter() - start

print(
    {
        "indexed_filter_seconds": indexed_seconds,
        "scan_filter_seconds": scan_seconds,
        "serial_candidates_seconds": serial_seconds,
        "threaded_candidates_seconds": threaded_seconds,
    }
)
