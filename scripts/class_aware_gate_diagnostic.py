"""Prototype class-conditioned instance matching for a future gate.

This is deliberately separate from ``check_set_gate.py`` and is not part
of any frozen protocol. It reports a centroid/AABB matching diagnostic;
it does not establish mask-level instance equivalence.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from scipy.optimize import linear_sum_assignment

from scripts.check_set_gate import gate_metrics
from scripts.edge_reliability import SceneGT
from scripts.scannet_gt import load_gt


def canonical_label(value: object) -> str:
    """Normalize formatting only; semantic aliases require a declared map."""
    label = " ".join(str(value or "").lower().replace("_", " ").split())
    return label or "<missing>"


def _solve_matching_summary(
    pred_indices: Sequence[int],
    gt_indices: Sequence[int],
    valid: np.ndarray,
    distances: np.ndarray,
) -> Tuple[int, float]:
    """Return max cardinality and min distance for one bipartite subproblem.

    A square assignment with dummy rows/columns gives invalid and unmatched
    pairs zero cardinality while a valid pair receives a large negative
    reward.  The reward therefore makes cardinality primary and distance
    secondary; the returned distance is recomputed from valid pairs so it is
    independent of dummy assignments.
    """
    n_pred = len(pred_indices)
    n_gt = len(gt_indices)
    if not n_pred or not n_gt:
        return 0, 0.0

    sub_valid = valid[np.ix_(pred_indices, gt_indices)]
    sub_distances = distances[np.ix_(pred_indices, gt_indices)]
    finite = sub_distances[sub_valid]
    if finite.size == 0:
        return 0, 0.0

    scale = max(1.0, float(np.max(finite)) * max(n_pred, n_gt) + 1.0)
    invalid = scale * (max(n_pred, n_gt) + 1.0)
    size = n_pred + n_gt
    costs = np.zeros((size, size), dtype=float)
    costs[:n_pred, :n_gt] = np.where(
        sub_valid, sub_distances - scale, invalid
    )
    rows, cols = linear_sum_assignment(costs)
    matched = [
        (int(row), int(col))
        for row, col in zip(rows, cols)
        if row < n_pred and col < n_gt and sub_valid[row, col]
    ]
    return len(matched), math.fsum(
        float(sub_distances[row, col]) for row, col in matched
    )


def _same_distance(left: float, right: float) -> bool:
    """Treat only sub-precision residuals as equal optimizer totals."""
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-14)


def _deterministic_group_matching(
    pred_group: List[dict],
    gt_group: List[int],
    gt_idx: Dict[int, int],
    gt: SceneGT,
    tol: float,
) -> List[Tuple[str, int]]:
    """Match one label group with a declared deterministic tie policy.

    The objective is maximum cardinality, then minimum total centroid
    distance.  Remaining exact ties are resolved by the lexicographically
    smallest sequence of assigned integer GT IDs for predictions sorted by
    their string node IDs; an unmatched prediction is ordered after every
    available GT ID.  This makes the policy independent of optimizer tie
    behavior.
    """
    pred_group = sorted(pred_group, key=lambda node: str(node["id"]))
    gt_group = sorted(int(oid) for oid in gt_group)
    n_pred = len(pred_group)
    n_gt = len(gt_group)
    valid = np.zeros((n_pred, n_gt), dtype=bool)
    distances = np.zeros((n_pred, n_gt), dtype=float)

    for i, node in enumerate(pred_group):
        centroid = node.get("centroid")
        if centroid is None:
            continue
        point = np.asarray(centroid, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            continue
        for j, oid in enumerate(gt_group):
            index = gt_idx[oid]
            lo = gt.aabbs[index][0] - tol
            hi = gt.aabbs[index][1] + tol
            if np.all(point >= lo) and np.all(point <= hi):
                valid[i, j] = True
                distances[i, j] = float(
                    np.linalg.norm(gt.centroids[index] - point)
                )

    all_pred = list(range(n_pred))
    all_gt = list(range(n_gt))
    target_cardinality, target_distance = _solve_matching_summary(
        all_pred, all_gt, valid, distances
    )
    if target_cardinality == 0:
        return []

    fixed: Dict[int, Optional[int]] = {}
    used_gt = set()
    fixed_cardinality = 0
    fixed_distance = 0.0
    chosen: List[Tuple[str, int]] = []

    for pred_index in all_pred:
        candidates = [
            gt_index for gt_index in all_gt
            if valid[pred_index, gt_index] and gt_index not in used_gt
        ]
        candidates.sort(key=lambda gt_index: gt_group[gt_index])
        candidates.append(None)

        for gt_index in candidates:
            if gt_index is not None and gt_index in used_gt:
                continue
            candidate_cardinality = (
                fixed_cardinality + (1 if gt_index is not None else 0)
            )
            candidate_distance = fixed_distance + (
                distances[pred_index, gt_index]
                if gt_index is not None else 0.0
            )
            remaining_pred = list(range(pred_index + 1, n_pred))
            remaining_gt = [
                index for index in all_gt if index not in used_gt
                and index != gt_index
            ]
            remaining_cardinality, remaining_distance = (
                _solve_matching_summary(
                    remaining_pred, remaining_gt, valid, distances
                )
            )
            if (
                candidate_cardinality + remaining_cardinality
                == target_cardinality
                and _same_distance(
                    candidate_distance + remaining_distance,
                    target_distance,
                )
            ):
                fixed[pred_index] = gt_index
                if gt_index is not None:
                    used_gt.add(gt_index)
                    fixed_cardinality = candidate_cardinality
                    fixed_distance = candidate_distance
                    chosen.append((
                        str(pred_group[pred_index]["id"]),
                        gt_group[gt_index],
                    ))
                else:
                    fixed_cardinality = candidate_cardinality
                    fixed_distance = candidate_distance
                break
        else:  # pragma: no cover - protects the frozen policy contract
            raise RuntimeError(
                "could not preserve the optimal assignment while applying "
                "the deterministic tie policy"
            )

    return chosen


def class_conditioned_alignment(
    nodes: List[dict], gt: SceneGT, tol: float = 0.25
) -> Dict[str, int]:
    """Match same-label centroids to padded GT AABBs, one-to-one.

    The assignment maximizes the number of eligible matches first, minimizes
    total centroid distance second, and then applies the deterministic
    predicted-node-ID/GT-ID tie policy declared above.
    """
    gt_idx = gt.object_index()
    pred_by_label: Dict[str, List[dict]] = {}
    gt_by_label: Dict[str, List[int]] = {}

    for node in nodes:
        label = canonical_label(node.get("label"))
        pred_by_label.setdefault(label, []).append(node)
    for oid in gt.object_ids:
        oid_int = int(oid)
        label = canonical_label(gt.names.get(str(oid_int), ""))
        if label == "<missing>":
            continue
        gt_by_label.setdefault(label, []).append(oid_int)

    matches: Dict[str, int] = {}
    for label in sorted(pred_by_label.keys() & gt_by_label.keys()):
        for pred_id, gt_id in _deterministic_group_matching(
            pred_by_label[label], gt_by_label[label], gt_idx, gt, tol
        ):
            matches[pred_id] = gt_id
    return matches


def wall_proposal_counts(
    nodes: List[dict], gt: SceneGT, tol: float = 0.25
) -> Dict[int, int]:
    """Count wall-labelled node centroids in each padded GT wall AABB.

    This deliberately does not use one-to-one matching: multiple predicted
    nodes may count toward the same wall, and unmatched walls remain zero.
    """
    gt_idx = gt.object_index()
    wall_boxes = {
        int(oid): (
            gt.aabbs[gt_idx[int(oid)]][0] - tol,
            gt.aabbs[gt_idx[int(oid)]][1] + tol,
        )
        for oid in gt.object_ids
        if canonical_label(gt.names.get(str(int(oid)), "")) == "wall"
    }
    counts = {oid: 0 for oid in wall_boxes}
    for node in nodes:
        if canonical_label(node.get("label")) != "wall":
            continue
        centroid = node.get("centroid")
        if centroid is None:
            continue
        point = np.asarray(centroid, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            continue
        for oid, (lo, hi) in wall_boxes.items():
            if np.all(point >= lo) and np.all(point <= hi):
                counts[oid] += 1
    return dict(sorted(counts.items()))


def class_conditioned_metrics(
    nodes: List[dict], gt: SceneGT, tol: float = 0.25
) -> Dict[str, object]:
    """Return micro and per-class TP/FP/FN for the prototype alignment."""
    alignment = class_conditioned_alignment(nodes, gt, tol)
    pred_labels = Counter(canonical_label(node.get("label")) for node in nodes)
    gt_labels = Counter(
        canonical_label(gt.names.get(str(int(oid)), ""))
        for oid in gt.object_ids
    )
    tp = Counter()
    node_labels = {
        str(node["id"]): canonical_label(node.get("label")) for node in nodes
    }
    for node_id in alignment:
        tp[node_labels[node_id]] += 1

    per_class = {}
    for label in sorted(pred_labels.keys() | gt_labels.keys()):
        pred_count = pred_labels[label]
        gt_count = gt_labels[label]
        matched = tp[label]
        per_class[label] = {
            "predicted": pred_count,
            "ground_truth": gt_count,
            "tp": matched,
            "fp": pred_count - matched,
            "fn": gt_count - matched,
            "precision": matched / pred_count if pred_count else None,
            "recall": matched / gt_count if gt_count else None,
        }

    wall_counts = {
        int(oid): 0
        for oid in gt.object_ids
        if canonical_label(gt.names.get(str(int(oid)), "")) == "wall"
    }
    for oid in alignment.values():
        if oid in wall_counts:
            wall_counts[oid] += 1
    proposal_counts = wall_proposal_counts(nodes, gt, tol)

    n_pred = len(nodes)
    n_gt = len(gt.object_ids)
    n_matched = len(alignment)
    return {
        "n_predicted": n_pred,
        "n_ground_truth": n_gt,
        "n_matched": n_matched,
        "precision": n_matched / n_pred if n_pred else None,
        "recall": n_matched / n_gt if n_gt else None,
        "alignment": alignment,
        "matched_pairs": [
            {"predicted_node_id": pred_id, "ground_truth_instance_id": gt_id}
            for pred_id, gt_id in sorted(
                alignment.items(), key=lambda item: str(item[0])
            )
        ],
        "per_class": per_class,
        "wall_match_counts": {
            str(oid): count for oid, count in sorted(wall_counts.items())
        },
        "median_nodes_per_wall": (
            float(np.median(list(wall_counts.values())))
            if wall_counts else 0.0
        ),
        "wall_proposal_counts": {
            str(oid): count for oid, count in proposal_counts.items()
        },
        "median_wall_proposals": (
            float(np.median(list(proposal_counts.values())))
            if proposal_counts else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", default="data/scannet/graphs")
    parser.add_argument("--gt-dir", default="data/scannet/gt")
    parser.add_argument(
        "--scenes", nargs="+",
        default=["scene0006_00", "scene0014_00"],
    )
    parser.add_argument("--tol-m", type=float, default=0.25)
    parser.add_argument(
        "--out",
        default=None,
        help="optional local JSON artifact; no file is written by default",
    )
    args = parser.parse_args()

    rows = []
    for scene in args.scenes:
        graph_path = os.path.join(args.graph_dir, f"{scene}.json")
        if not os.path.isfile(graph_path):
            parser.error(f"missing graph: {graph_path}")
        with io.open(graph_path, encoding="utf-8") as fh:
            graph = json.load(fh)
        gt = load_gt(args.gt_dir, scene, coarse=False)

        class_blind = gate_metrics(scene, graph, gt)
        class_aware = class_conditioned_metrics(
            graph["nodes"], gt, tol=args.tol_m
        )
        rows.append({
            "scene": scene,
            "class_blind": {
                "n_predicted": class_blind["n_nodes"],
                "n_ground_truth": class_blind["n_instances"],
                "n_matched": class_blind["n_aligned"],
                "precision": class_blind["precision"],
                "recall": class_blind["recall"],
                "class_count_ratios": {
                    label: {
                        "predicted": values[0],
                        "ground_truth": values[1],
                        "ratio": values[2],
                    }
                    for label, values in class_blind["class_ratios"].items()
                },
            },
            "class_conditioned": class_aware,
        })

    payload = {
        "diagnostic": (
            "unfrozen same-label centroid/AABB matching; not a gate verdict "
            "or mask-equivalence test"
        ),
        "matching_policy": (
            "maximum cardinality, minimum total centroid distance, then "
            "lexicographically smallest GT-ID sequence for predicted node "
            "IDs sorted by their string IDs; unmatched is last"
        ),
        "tolerance_m": args.tol_m,
        "scenes": rows,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with io.open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    print(text, end="")
    if args.out:
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
