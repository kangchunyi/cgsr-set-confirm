"""Synthetic checks for the frozen class-conditioned confirmation gate."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.edge_reliability import SceneGT
from scripts.run_class_conditioned_gate import run_source, scene_report


GATE = {
    "one_to_one_precision_min": 0.75,
    "one_to_one_recall_min": 0.75,
    "class_count_ratio": [0.75, 1.25],
    "median_wall_proposals_max": 2.0,
}


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


def test_a_perfect_scene_passes() -> None:
    gt = make_gt([1, 2], ["chair", "chair"], [[0, 0, 0], [2, 0, 0]])
    graph = {"nodes": [
        {"id": "p1", "label": "Chair", "centroid": [0, 0, 0]},
        {"id": "p2", "label": "chair", "centroid": [2, 0, 0]},
    ]}

    row = scene_report("synthetic", graph, gt, GATE, 0.25)

    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["classes_in_band"] == 1
    assert row["verdict"] == "PASS"


def test_a_label_swap_fails_even_when_geometry_matches() -> None:
    gt = make_gt([1, 2], ["chair", "lamp"], [[0, 0, 0], [2, 0, 0]])
    graph = {"nodes": [
        {"id": "p1", "label": "lamp", "centroid": [0, 0, 0]},
        {"id": "p2", "label": "chair", "centroid": [2, 0, 0]},
    ]}

    row = scene_report("synthetic", graph, gt, GATE, 0.25)

    assert row["n_matched"] == 0
    assert row["verdict"] == "FAIL"
    assert row["checks"]["precision"]["passed"] is False
    assert row["checks"]["recall"]["passed"] is False


def test_extra_same_class_nodes_fail_on_precision_and_count_band() -> None:
    gt = make_gt([1, 2], ["chair", "chair"], [[0, 0, 0], [2, 0, 0]])
    graph = {"nodes": [
        {"id": f"p{i}", "label": "chair", "centroid": [2.0 * i, 0, 0]}
        for i in range(3)
    ]}

    row = scene_report("synthetic", graph, gt, GATE, 0.25)

    assert row["precision"] == 2 / 3
    assert row["classes_in_band"] == 0
    assert row["verdict"] == "FAIL"


def test_wall_fragments_fail_on_the_independent_wall_count() -> None:
    gt = make_gt(
        [1, 2], ["wall", "wall"], [[0, 0, 0], [10, 0, 0]]
    )
    graph = {"nodes": [
        {"id": f"w{i}", "label": "wall", "centroid": [0.01 * i, 0, 0]}
        for i in range(8)
    ]}

    row = scene_report("synthetic", graph, gt, GATE, 0.25)

    assert row["median_wall_proposals"] == 4.0
    assert row["checks"]["median_wall_proposals"]["passed"] is False
    assert row["verdict"] == "FAIL"


def _write_scene(base: Path, scene: str, label: str) -> None:
    gt_dir = base / "gt"
    graph_dir = base / "graphs"
    gt_dir.mkdir(exist_ok=True)
    graph_dir.mkdir(exist_ok=True)
    gt = {
        "scene": scene,
        "instances": [{
            "object_id": 1,
            "raw_name": label,
            "coarse_name": label,
            "bbox": [[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]],
        }],
    }
    graph = {"nodes": [
        {"id": "p1", "label": label, "centroid": [0, 0, 0]}
    ]}
    (gt_dir / f"{scene}.json").write_text(json.dumps(gt), encoding="utf-8")
    (graph_dir / f"{scene}.json").write_text(
        json.dumps(graph), encoding="utf-8"
    )


def test_source_with_missing_graphs_gets_no_verdict() -> None:
    scenes = [f"synthetic{i:02d}" for i in range(10)]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for scene in scenes[:9]:
            _write_scene(base, scene, "chair")
        protocol = {
            "scenes": {"confirmation": scenes},
            "sources": {"source": {"status": "available",
                                   "graph_dir": "graphs"}},
        }

        result = run_source(
            "source", protocol["sources"]["source"], protocol,
            str(base / "gt"), str(base), GATE, 0.25,
        )

    assert result["status"] == "incomplete"
    assert result["missing_scenes"] == [scenes[-1]]
    assert result["verdict"] is None


def test_complete_source_is_evaluated_scene_by_scene() -> None:
    scenes = [f"synthetic{i:02d}" for i in range(10)]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for scene in scenes:
            _write_scene(base, scene, "chair")
        protocol = {
            "scenes": {"confirmation": scenes},
            "sources": {"source": {"status": "available",
                                   "graph_dir": "graphs"}},
        }

        result = run_source(
            "source", protocol["sources"]["source"], protocol,
            str(base / "gt"), str(base), GATE, 0.25,
        )

    assert result["status"] == "complete"
    assert result["verdict"] == "PASS"
    assert result["pooled_descriptive_only"]["scenes_passed"] == 10
    assert len(result["scenes"]) == 10


if __name__ == "__main__":
    tests = [
        test_a_perfect_scene_passes,
        test_a_label_swap_fails_even_when_geometry_matches,
        test_extra_same_class_nodes_fail_on_precision_and_count_band,
        test_wall_fragments_fail_on_the_independent_wall_count,
        test_source_with_missing_graphs_gets_no_verdict,
        test_complete_source_is_evaluated_scene_by_scene,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
