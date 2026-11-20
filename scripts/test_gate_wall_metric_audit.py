"""Document the frozen wall metric's one-to-one alignment ceiling."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.check_set_gate import gate_metrics  # noqa: E402
from scripts.edge_reliability import SceneGT  # noqa: E402


def test_wall_count_cannot_register_multiple_nodes_per_instance() -> None:
    gt = SceneGT(
        scene="synthetic",
        object_ids=np.asarray([1, 2]),
        centroids=np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        aabbs=np.asarray([
            [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            [[9.5, -0.5, -0.5], [10.5, 0.5, 0.5]],
        ]),
        corners=np.empty((2, 8, 3)),
        regions=np.empty(0, dtype=int),
        names={"1": "wall", "2": "wall"},
    )
    nodes = [
        {"id": f"fragment-{i}", "centroid": [0.01 * i, 0.0, 0.0]}
        for i in range(8)
    ]

    metrics = gate_metrics("synthetic", {"nodes": nodes}, gt)

    assert metrics["n_aligned"] == 1
    assert metrics["precision"] == 1 / 8
    # The unmatched annotated wall is omitted; the eight fragments around
    # the first wall still produce a wall count of one.
    assert metrics["median_nodes_per_wall"] == 1.0


if __name__ == "__main__":
    test_wall_count_cannot_register_multiple_nodes_per_instance()
    print("PASS  one-to-one alignment caps the wall metric at one")
