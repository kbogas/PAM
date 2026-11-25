from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def test_readme_quickstart_output_matches_documentation():
    root = Path(__file__).parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "examples" / "quickstart.py")],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "nodes: {'alice': 0, 'bob': 1, 'carol': 2}",
        "relations: {'knows': 3, 'likes': 17}",
        "nonzeros by hop: [3, 1]",
        "BoP shape: (3, 6)",
        "features: ['h1|in|v=17', 'h1|in|v=3', 'h1|out|v=17', 'h1|out|v=3', 'h2|in|v=9', 'h2|out|v=9']",
        "counts: [[0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0]]",
    ]


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        ("tutorial_nodes.py", "(3, 6) 12"),
        ("tutorial_pairs.py", "(2, 16) 20"),
        ("tutorial_subgraphs.py", "(2, 3) 4"),
        ("tutorial_multigraph.py", "(2, 2) 2"),
    ],
)
def test_documentation_tutorial_output(script: str, expected: str):
    root = Path(__file__).parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "examples" / script)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == expected
