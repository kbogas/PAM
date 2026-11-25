"""Render documentation figures from committed machine-readable results."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "experiments" / "results"
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(exist_ok=True)
mpl.rcParams["svg.hashsalt"] = "prime-adj-docs"

wn = pd.read_csv(RESULTS / "wn18rr_reference.csv")
figure, axes = plt.subplots(1, 2, figsize=(8, 3.2))
axes[0].bar(wn["method"], wn["mrr"], color=["#4c78a8", "#f58518"])
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("MRR")
axes[0].set_title("WN18RR task quality")
axes[1].bar(wn["method"], wn["cpu_seconds"], color=["#4c78a8", "#f58518"])
axes[1].set_ylabel("CPU seconds")
axes[1].set_title("Reference runtime")
figure.tight_layout()
figure.savefig(ASSETS / "wn18rr_quality_runtime.svg", metadata={"Date": None})
plt.close(figure)

collision = pd.read_csv(RESULTS / "collision_fixture.csv")
figure, axis = plt.subplots(figsize=(4.5, 3.2))
axis.plot(
    collision["hop"],
    collision["ambiguous_signature_fraction"],
    marker="o",
    color="#e45756",
)
axis.set_xticks(collision["hop"])
axis.set_ylim(0, 1.05)
axis.set_xlabel("Hop")
axis.set_ylabel("Fraction in ambiguous groups")
axis.set_title("Deterministic reversed-chain fixture")
figure.tight_layout()
figure.savefig(ASSETS / "collision_by_hop.svg", metadata={"Date": None})
plt.close(figure)
