"""Synthetic checks for the separate class-conditioned gate prototype."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.class_aware_gate_diagnostic import (
    class_conditioned_alignment,
    class_conditioned_metrics,
)
from scripts.check_set_gate import gate_metrics
from scripts.edge_reliability import SceneGT


def make_gt(ids, labels, centers, half_extent=0.1) -> SceneGT:
    centers = np.asarray(centers, dtype=float)
    extents = np.full_like(centers, half_extent)
    return SceneGT(
        scene="synthetic",
        object_ids=np.asarray(ids, dtype=int),
        centroids=centers,
        aabbs=np.stack([centers - extents, centers + extents], axis=1),
        corners=np.empty((len(ids), 8, 3)),
        regions=np.empty(0, dtype=int),
        names={str(oid): label for oid, label in zip(ids, labels)},
    )


def test_oracle_labels_and_geometry_match() -> None:
    gt = make_gt([1, 2], ["chair", "lamp"], [[0, 0, 0], [2, 0, 0]])
    nodes = [
        {"id": "p1", "label": "chair", "centroid": [0, 0, 0]},
        {"id": "p2", "label": "lamp", "centroid": [2, 0, 0]},
    ]

    metrics = class_conditioned_metrics(nodes, gt)

    assert metrics["alignment"] == {"p1": 1, "p2": 2}
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_label_swap_is_not_counted_as_an_instance_match() -> None:
    gt = make_gt(
        [1, 2, 3, 4],
        ["chair", "chair", "lamp", "lamp"],
        [[0, 0, 0], [1, 0, 0], [3, 0, 0], [4, 0, 0]],
    )
    nodes = [
        {"id": "p1", "label": "lamp", "centroid": [0, 0, 0]},
        {"id": "p2", "label": "lamp", "centroid": [1, 0, 0]},
        {"id": "p3", "label": "chair", "centroid": [3, 0, 0]},
        {"id": "p4", "label": "chair", "centroid": [4, 0, 0]},
    ]

    legacy = gate_metrics(
        "synthetic",
        {"nodes": nodes},
        gt,
    )
    metrics = class_conditioned_metrics(nodes, gt)

    assert legacy["recall"] == 1.0
    assert legacy["precision"] == 1.0
    assert legacy["class_ratios"] == {
        "chair": (2, 2, 1.0),
        "lamp": (2, 2, 1.0),
    }
    assert metrics["alignment"] == {}
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["per_class"]["chair"] == {
        "predicted": 2,
        "ground_truth": 2,
        "tp": 0,
        "fp": 2,
        "fn": 2,
        "precision": 0.0,
        "recall": 0.0,
    }
    assert metrics["per_class"]["lamp"]["fn"] == 2


def test_missing_labels_are_not_matched_to_each_other() -> None:
    gt = make_gt([1], [""], [[0, 0, 0]])
    nodes = [{"id": "p1", "centroid": [0, 0, 0]}]

    assert class_conditioned_alignment(nodes, gt) == {}


def test_assignment_maximizes_valid_match_count_before_distance() -> None:
    gt = make_gt(
        [1, 2],
        ["chair", "chair"],
        [[0, 0, 0], [1, 0, 0]],
        half_extent=0.5,
    )
    nodes = [
        {"id": "shared", "label": "chair", "centroid": [0.3, 0, 0]},
        {"id": "only-first", "label": "chair", "centroid": [-0.7, 0, 0]},
    ]

    alignment = class_conditioned_alignment(nodes, gt)

    assert alignment == {"only-first": 1, "shared": 2}


def test_exact_distance_ties_use_sorted_prediction_and_gt_ids() -> None:
    gt = make_gt(
        [10, 2],
        ["chair", "chair"],
        [[-1, 0, 0], [1, 0, 0]],
        half_extent=2.0,
    )
    nodes = [
        {"id": "z", "label": "chair", "centroid": [0, 0, 0]},
        {"id": "a", "label": "chair", "centroid": [0, 0, 0]},
    ]

    alignment = class_conditioned_alignment(nodes, gt)

    assert alignment == {"a": 2, "z": 10}


def test_near_tie_still_minimizes_distance_before_lexical_order() -> None:
    gt = make_gt(
        [1, 2],
        ["chair", "chair"],
        [[0, 0, 0], [1, 0, 0]],
        half_extent=2.0,
    )
    nodes = [
        {
            "id": "a",
            "label": "chair",
            "centroid": [0.5000000000001, 0, 0],
        },
        {"id": "b", "label": "chair", "centroid": [0.5, 0, 0]},
    ]

    alignment = class_conditioned_alignment(nodes, gt)

    assert alignment == {"a": 2, "b": 1}


def test_wall_proposal_count_is_not_limited_by_one_to_one_matching() -> None:
    gt = make_gt(
        [1, 2],
        ["wall", "wall"],
        [[0, 0, 0], [5, 0, 0]],
    )
    nodes = [
        {
            "id": f"fragment-{i}",
            "label": "wall",
            "centroid": [0.01 * i, 0, 0],
        }
        for i in range(8)
    ]

    metrics = class_conditioned_metrics(nodes, gt)

    assert metrics["n_matched"] == 1
    assert metrics["wall_match_counts"] == {"1": 1, "2": 0}
    assert metrics["median_nodes_per_wall"] == 0.5
    assert metrics["wall_proposal_counts"] == {"1": 8, "2": 0}
    assert metrics["median_wall_proposals"] == 4.0


def test_unmatched_extra_node_counts_as_false_positive() -> None:
    gt = make_gt([1], ["chair"], [[0, 0, 0]])
    nodes = [
        {"id": "matched", "label": "chair", "centroid": [0, 0, 0]},
        {"id": "extra", "label": "chair", "centroid": [5, 0, 0]},
    ]

    metrics = class_conditioned_metrics(nodes, gt)

    assert metrics["n_matched"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["per_class"]["chair"]["fp"] == 1


def test_wall_metric_includes_unmatched_annotated_walls() -> None:
    gt = make_gt(
        [1, 2],
        ["wall", "wall"],
        [[0, 0, 0], [5, 0, 0]],
    )
    nodes = [
        {"id": "wall-node", "label": "wall", "centroid": [0, 0, 0]},
    ]

    metrics = class_conditioned_metrics(nodes, gt)

    assert metrics["wall_match_counts"] == {"1": 1, "2": 0}
    assert metrics["median_nodes_per_wall"] == 0.5


def test_cli_reports_class_blind_and_class_conditioned_metrics() -> None:
    root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        graph_dir = base / "graphs"
        gt_dir = base / "gt"
        graph_dir.mkdir()
        gt_dir.mkdir()

        graph = {
            "nodes": [
                {"id": "p1", "label": "lamp", "centroid": [0, 0, 0]},
                {"id": "p2", "label": "chair", "centroid": [2, 0, 0]},
            ]
        }
        gt = {
            "scene": "synthetic",
            "instances": [
                {
                    "object_id": 1,
                    "raw_name": "chair",
                    "coarse_name": "chair",
                    "bbox": [[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]],
                },
                {
                    "object_id": 2,
                    "raw_name": "lamp",
                    "coarse_name": "lamp",
                    "bbox": [[1.9, -0.1, -0.1], [2.1, 0.1, 0.1]],
                },
            ],
        }
        (graph_dir / "synthetic.json").write_text(
            json.dumps(graph), encoding="utf-8"
        )
        (gt_dir / "synthetic.json").write_text(
            json.dumps(gt), encoding="utf-8"
        )
        result = subprocess.run(
            [
                sys.executable,
                str(root / "scripts/class_aware_gate_diagnostic.py"),
                "--graph-dir", str(graph_dir),
                "--gt-dir", str(gt_dir),
                "--scenes", "synthetic",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    row = json.loads(result.stdout)["scenes"][0]
    assert row["class_blind"]["precision"] == 1.0
    assert row["class_blind"]["recall"] == 1.0
    assert row["class_conditioned"]["precision"] == 0.0
    assert row["class_conditioned"]["recall"] == 0.0


if __name__ == "__main__":
    tests = [
        test_oracle_labels_and_geometry_match,
        test_label_swap_is_not_counted_as_an_instance_match,
        test_missing_labels_are_not_matched_to_each_other,
        test_assignment_maximizes_valid_match_count_before_distance,
        test_exact_distance_ties_use_sorted_prediction_and_gt_ids,
        test_near_tie_still_minimizes_distance_before_lexical_order,
        test_wall_proposal_count_is_not_limited_by_one_to_one_matching,
        test_unmatched_extra_node_counts_as_false_positive,
        test_cli_reports_class_blind_and_class_conditioned_metrics,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
