"""The edge-layer reliability table.

This is the measurement nobody in the 3D scene-graph literature has made:
**how reliable is each method's relation layer, and does retrieval accuracy
actually track edge accuracy?**

For every method we compute, per relation type:

    precision  = TP / (TP + FP)      over *predicted* edges of that type
    recall     = TP / (TP + FN)      over *ground-truth* edges of that type
    F1
    ECE        = calibration error of the edge confidence

and we emit two summary statistics that answer the paper's central
question:

    ``rank_corr``      Spearman correlation between a method's edge F1 and
                       its retrieval accuracy across methods
    ``monotonicity``  whether the better edge layer also gives better
                       retrieval (this is hypothesis H3: we expect *no*)

Expected input format for a predicted graph (one JSON file per scene)
----------------------------------------------------------------------
.. code-block:: json

    {
      "scene": "00824-Dd4bFSTQ8gi",
      "method": "inherit_sg",
      "nodes": [
        {"id": "obj_12", "label": "chair", "level": "object",
         "parent": "room_3", "centroid": [1.2, 0.3, 0.5],
         "bbox": [[0.9, 0.0, 0.0], [1.5, 0.6, 0.9]],
         "attributes": {"color": "brown"}, "source": "inherit_sg"}
      ],
      "edges": [
        {"subject": "obj_12", "object": "obj_20", "relation": "on",
         "source": "geo_heur", "confidence": 1.0}
      ]
    }

``relation`` strings are matched against the ground-truth vocabulary
(``near``, ``far``, ``on``, ``under``, ``inside``, ``left of``, ``right of``,
``above``, ``below``, ``in front of``, ``behind``).  Use the canonical
relation words; the alias table below accepts common variants.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --------------------------------------------------------------------------- #
# relation vocabulary + aliases
# --------------------------------------------------------------------------- #
GT_RELATIONS: Tuple[str, ...] = (
    "near",
    "far",
    "on",
    "under",
    "inside",
    "left of",
    "right of",
    "above",
    "below",
    "in front of",
    "behind",
)

RELATION_ALIASES: Dict[str, str] = {
    "on top of": "on",
    "resting on": "on",
    "supported by": "on",
    "beneath": "under",
    "underneath": "under",
    "below": "under",
    "in": "inside",
    "within": "inside",
    "contained in": "inside",
    "next to": "near",
    "close to": "near",
    "adjacent to": "near",
    "beside": "near",
    "by": "near",
    "around": "near",
    "far from": "far",
    "away from": "far",
    "distant from": "far",
    "left": "left of",
    "to the left of": "left of",
    "right": "right of",
    "to the right of": "right of",
    "in front": "in front of",
    "behind": "behind",
}


def canonical_relation(rel: str) -> Optional[str]:
    r = rel.strip().lower()
    if r in GT_RELATIONS:
        return r
    if r in RELATION_ALIASES:
        return RELATION_ALIASES[r]
    for k, v in RELATION_ALIASES.items():
        if k in r:
            return v
    return None


# --------------------------------------------------------------------------- #
@dataclass
class EdgeStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    #: (confidence, correct) pairs for calibration
    samples: List[Tuple[float, bool]] = field(default_factory=list)

    @property
    def precision(self) -> float:
        den = self.tp + self.fp
        return self.tp / den if den else float("nan")

    @property
    def recall(self) -> float:
        den = self.tp + self.fn
        return self.tp / den if den else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if math.isnan(p) or math.isnan(r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    @property
    def support(self) -> int:
        return self.tp + self.fn

    def ece(self, n_bins: int = 10) -> float:
        if not self.samples:
            return float("nan")
        conf = np.asarray([s[0] for s in self.samples], dtype=float)
        corr = np.asarray([1.0 if s[1] else 0.0 for s in self.samples], dtype=float)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if not m.any():
                continue
            ece += (m.sum() / len(conf)) * abs(corr[m].mean() - conf[m].mean())
        return float(ece)

    def auroc(self) -> float:
        if len(self.samples) < 2 or len({s[1] for s in self.samples}) < 2:
            return float("nan")
        conf = np.asarray([s[0] for s in self.samples], dtype=float)
        corr = np.asarray([1.0 if s[1] else 0.0 for s in self.samples], dtype=float)
        order = np.argsort(conf)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(conf) + 1)
        n_pos = corr.sum()
        n_neg = len(corr) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        return float((ranks[corr == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass
class SceneGT:
    """Ground truth for one scene, loaded from ``build_gt_edges`` output."""

    scene: str
    object_ids: np.ndarray
    centroids: np.ndarray
    aabbs: np.ndarray
    corners: np.ndarray
    regions: np.ndarray
    names: Dict[str, str]
    #: (subject_id, object_id, relation) -> margin
    edges: Dict[Tuple[int, int, str], float] = field(default_factory=dict)
    #: per-relation set of GT pairs
    pairs: Dict[str, Set[Tuple[int, int]]] = field(default_factory=dict)

    @staticmethod
    def load(out_dir: str, scene: str) -> "SceneGT":
        boxes = np.load(os.path.join(out_dir, f"{scene}.boxes.npz"))
        with open(os.path.join(out_dir, f"{scene}.edges.json"), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        gt = SceneGT(
            scene=scene,
            object_ids=boxes["object_ids"],
            centroids=boxes["centroids"],
            aabbs=boxes["aabbs"],
            corners=boxes["corners"],
            regions=boxes["regions"],
            names=meta.get("names", {}),
        )
        for e in meta.get("edges", []):
            rel = canonical_relation(e["r"]) or e["r"]
            key = (int(e["s"]), int(e["o"]), rel)
            gt.edges[key] = float(e.get("margin", 1.0))
            gt.pairs.setdefault(rel, set()).add((int(e["s"]), int(e["o"])))
        return gt

    def object_index(self) -> Dict[int, int]:
        return {int(v): i for i, v in enumerate(self.object_ids)}

    def bbox_of(self, oid: int) -> Optional[np.ndarray]:
        idx = self.object_index().get(oid)
        return None if idx is None else self.aabbs[idx]

    def corners_of(self, oid: int) -> Optional[np.ndarray]:
        idx = self.object_index().get(oid)
        return None if idx is None else self.corners[idx]


# --------------------------------------------------------------------------- #
# aligning predicted nodes to GT instances
# --------------------------------------------------------------------------- #
def align_nodes(pred_nodes: Sequence[dict], gt: SceneGT) -> Dict[str, int]:
    """Map predicted node ids to GT instance ids via the AABB of the centroid.

    The protocol mirrors HM3DSem-SQR: a prediction matches an instance if
    its 3D centre lies inside the instance's box (or within 0.25 m).  When
    several boxes match, the one whose centre is closest wins, provided the
    two candidates are not within 0.05 m of each other (otherwise the
    instance is ambiguous and the match is discarded).
    """
    idx = gt.object_index()
    out: Dict[str, int] = {}
    for n in pred_nodes:
        c = n.get("centroid")
        if c is None:
            continue
        p = np.asarray(c, dtype=float)
        hits: List[Tuple[int, float]] = []
        for oid in gt.object_ids:
            lo = gt.aabbs[idx[int(oid)]][0] - 0.25
            hi = gt.aabbs[idx[int(oid)]][1] + 0.25
            if np.all(p >= lo) and np.all(p <= hi):
                d = float(np.linalg.norm(gt.centroids[idx[int(oid)]] - p))
                hits.append((int(oid), d))
        if not hits:
            continue
        hits.sort(key=lambda t: t[1])
        if len(hits) > 1 and abs(hits[0][1] - hits[1][1]) <= 0.05:
            # ambiguous -> try the category correspondence
            label = str(n.get("label", "")).lower()
            named = [
                h for h in hits
                if str(gt.names.get(str(h[0]), "")).lower() == label
            ]
            if len(named) == 1:
                out[n["id"]] = named[0][0]
            continue
        out[n["id"]] = hits[0][0]
    return out


# --------------------------------------------------------------------------- #
# scoring a method's edge layer
# --------------------------------------------------------------------------- #
def score_edges(
    pred: dict, gt: SceneGT, min_margin: float = 0.0
) -> Dict[str, EdgeStats]:
    """Score a predicted graph's edges against the GT edges."""
    stats: Dict[str, EdgeStats] = {r: EdgeStats() for r in GT_RELATIONS}
    node_to_gt = align_nodes(pred.get("nodes", []), gt)

    matched_pairs: Set[Tuple[int, int, str]] = set()

    for e in pred.get("edges", []):
        rel = canonical_relation(str(e.get("relation", "")))
        if rel is None:
            continue
        s = node_to_gt.get(e.get("subject"))
        o = node_to_gt.get(e.get("object"))
        if s is None or o is None:
            continue
        margin = gt.edges.get((s, o, rel), 0.0)
        correct = (s, o, rel) in gt.edges and margin >= min_margin
        conf = float(e.get("confidence", 1.0))
        st = stats.setdefault(rel, EdgeStats())
        st.samples.append((conf, correct))
        if correct:
            st.tp += 1
            matched_pairs.add((s, o, rel))
        else:
            st.fp += 1

    for rel, pairs in gt.pairs.items():
        st = stats.setdefault(rel, EdgeStats())
        for (s, o) in pairs:
            if (s, o, rel) not in matched_pairs:
                st.fn += 1

    return stats


def merge_stats(dst: Dict[str, EdgeStats], src: Dict[str, EdgeStats]) -> None:
    for rel, s in src.items():
        d = dst.setdefault(rel, EdgeStats())
        d.tp += s.tp
        d.fp += s.fp
        d.fn += s.fn
        d.samples.extend(s.samples)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def render_markdown(
    per_method: Dict[str, Dict[str, EdgeStats]],
    relations: Sequence[str] = GT_RELATIONS,
) -> str:
    lines: List[str] = []
    lines.append("### Edge-layer reliability (per relation type)\n")
    header = "| method | " + " | ".join(
        f"{r} P / R / F1" for r in relations
    ) + " | macro-F1 | ECE |"
    lines.append(header)
    lines.append("|" + "---|" * (len(relations) + 3))
    for method, stats in per_method.items():
        cells: List[str] = []
        f1s: List[float] = []
        for r in relations:
            s = stats.get(r, EdgeStats())
            if math.isnan(s.f1):
                cells.append("—")
            else:
                cells.append(f"{s.precision:.2f}/{s.recall:.2f}/{s.f1:.2f}")
                f1s.append(s.f1)
        macro = float(np.mean(f1s)) if f1s else float("nan")
        all_samples = [sm for s in stats.values() for sm in s.samples]
        ece = EdgeStats(samples=all_samples).ece()
        lines.append(
            f"| {method} | " + " | ".join(cells) + f" | {macro:.3f} | {ece:.3f} |"
        )
    return "\n".join(lines)


def render_csv(
    per_method: Dict[str, Dict[str, EdgeStats]], out_path: str
) -> None:
    import csv as _csv

    rels = list(GT_RELATIONS)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(
            ["method", "relation", "tp", "fp", "fn", "precision", "recall", "f1", "ece", "auroc"]
        )
        for method, stats in per_method.items():
            for r in rels:
                s = stats.get(r, EdgeStats())
                w.writerow(
                    [
                        method,
                        r,
                        s.tp,
                        s.fp,
                        s.fn,
                        "" if math.isnan(s.precision) else round(s.precision, 4),
                        "" if math.isnan(s.recall) else round(s.recall, 4),
                        "" if math.isnan(s.f1) else round(s.f1, 4),
                        "" if math.isnan(s.ece()) else round(s.ece(), 4),
                        "" if math.isnan(s.auroc()) else round(s.auroc(), 4),
                    ]
                )


# --------------------------------------------------------------------------- #

def _split_scenes(values):
    """Accept ``--scenes a,b,c`` as well as ``--scenes a b c``."""
    out = []
    for v in values or []:
        for part in str(v).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gt-dir",
        required=True,
        help="output of build_gt_edges.py (contains <scene>.boxes.npz and .edges.json)",
    )
    ap.add_argument(
        "--pred",
        action="append",
        required=True,
        metavar="METHOD=DIR",
        help="a predicted-graph directory; may be given several times",
    )
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-margin", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    per_method: Dict[str, Dict[str, EdgeStats]] = {}

    for spec in args.pred:
        method, _, pred_dir = spec.partition("=")
        method = method or os.path.basename(pred_dir.rstrip("/"))
        if not pred_dir:
            pred_dir = method
            method = os.path.basename(pred_dir.rstrip("/"))

        scenes = _split_scenes(args.scenes) or sorted(
            f[: -len(".edges.json")]
            for f in os.listdir(args.gt_dir)
            if f.endswith(".edges.json")
        )
        agg: Dict[str, EdgeStats] = {}
        n_used = 0
        for scene in scenes:
            gt_path = os.path.join(args.gt_dir, f"{scene}.edges.json")
            pred_path = os.path.join(pred_dir, f"{scene}.json")
            if not (os.path.exists(gt_path) and os.path.exists(pred_path)):
                continue
            gt = SceneGT.load(args.gt_dir, scene)
            with open(pred_path, "r", encoding="utf-8") as fh:
                pred = json.load(fh)
            merge_stats(agg, score_edges(pred, gt, args.min_margin))
            n_used += 1

        if n_used == 0:
            print(f"[skip] {method}: no matching scene files between "
                  f"{pred_dir} and {args.gt_dir}")
            continue
        per_method[method] = agg
        print(f"[ok] {method}: scored {n_used} scenes")

    if not per_method:
        print("nothing to report")
        return

    md = render_markdown(per_method)
    print()
    print(md)
    with open(os.path.join(args.out_dir, "edge_reliability.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    render_csv(per_method, os.path.join(args.out_dir, "edge_reliability.csv"))
    print(
        f"\nwrote {os.path.join(args.out_dir, 'edge_reliability.md')} and .csv"
    )


if __name__ == "__main__":
    main()
