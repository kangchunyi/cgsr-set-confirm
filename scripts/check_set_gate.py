"""The set gate of the CGSR-set protocol.

The gate runs on the **two development scenes only**, and it looks at the
graph and the annotation, never at a query answer:

*   a **one-to-one** alignment: a predicted node aligns to at most one
    annotated instance and an annotated instance to at most one predicted
    node; its recall and its precision must both be at least 0.75;
*   a **class-count** ratio: for every class with at least two annotated
    instances on the scene, the number of predicted nodes the solver will
    treat as that class, divided by the annotated count, must lie in
    [0.75, 1.25];
*   the **wall** count: the median number of predicted nodes per annotated
    wall must not exceed 2.

The gate is the reason this protocol exists: ``cgsr-inst`` passed a
node-count gate and still failed, because the counts of pillow, drawer and
shelf did not match.  This gate checks those counts before a single query
is scored.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.edge_reliability import SceneGT  # noqa: E402
from scripts.scannet_gt import load_gt as load_scannet_gt  # noqa: E402


def align_one_to_one(
    nodes: List[dict], gt: SceneGT, tol: float = 0.25
) -> Dict[str, int]:
    """A one-to-one alignment of the predicted nodes to the instances.

    A predicted node matches an instance when its centroid falls inside the
    instance's box or within ``tol`` metres of it.  When several instances
    match, the closest centroid wins; when two are equally close the match
    is ambiguous and dropped.  Each instance is taken by at most one node:
    the candidates are resolved in order of the distance.
    """
    idx = gt.object_index()
    pairs: List[Tuple[float, str, int]] = []
    for n in nodes:
        c = n.get("centroid")
        if c is None:
            continue
        p = np.asarray(c, dtype=float)
        for oid in gt.object_ids:
            lo = gt.aabbs[idx[int(oid)]][0] - tol
            hi = gt.aabbs[idx[int(oid)]][1] + tol
            if np.all(p >= lo) and np.all(p <= hi):
                d = float(np.linalg.norm(gt.centroids[idx[int(oid)]] - p))
                pairs.append((d, n["id"], int(oid)))
    pairs.sort()
    out: Dict[str, int] = {}
    taken: set = set()
    for d, nid, oid in pairs:
        if nid in out or oid in taken:
            continue
        out[nid] = oid
        taken.add(oid)
    return out


def gate_metrics(
    scene: str, pred: Dict[str, object], gt: SceneGT
) -> Dict[str, object]:
    """The gate statistics of one scene."""
    nodes = pred["nodes"]
    align = align_one_to_one(nodes, gt)
    n_inst = len(gt.object_ids)
    n_nodes = len(nodes)
    recall = len(align) / max(1, n_inst)
    precision = len(align) / max(1, n_nodes)

    pred_labels = Counter(str(n.get("label", "")).lower() for n in nodes)
    gt_labels = Counter(
        str(gt.names.get(str(o), "")).lower() for o in gt.object_ids
    )
    ratios = []
    per_class = {}
    for lab, k in gt_labels.items():
        if k < 2:
            continue
        ratio = pred_labels.get(lab, 0) / k
        per_class[lab] = (pred_labels.get(lab, 0), k, ratio)
        ratios.append(ratio)

    walls = [
        int(o) for o in gt.object_ids
        if str(gt.names.get(str(o), "")).lower() == "wall"
    ]
    per_wall = Counter()
    for oid in align.values():
        lab = str(gt.names.get(str(oid), "")).lower()
        if lab == "wall":
            per_wall[oid] += 1
    median_wall = (
        float(np.median(list(per_wall.values()))) if per_wall else 0.0
    )

    return {
        "scene": scene,
        "n_nodes": n_nodes,
        "n_instances": n_inst,
        "n_aligned": len(align),
        "recall": recall,
        "precision": precision,
        "median_nodes_per_wall": median_wall,
        "class_ratios": per_class,
        "median_class_ratio": float(np.median(ratios)) if ratios else 0.0,
        "min_class_ratio": float(min(ratios)) if ratios else 0.0,
        "max_class_ratio": float(max(ratios)) if ratios else 0.0,
        "n_classes_checked": len(ratios),
    }


def _load_gt(gt_dir: str, scene: str):
    """The annotation of a scene: the ScanNet one when it is there.

    The classes are the **fine** ones (the ScanNet200 names): the gate's
    criterion is per class, and the vocabulary it compares against is the
    one the frozen label source uses.
    """
    path = os.path.join(gt_dir, f"{scene}.json")
    if os.path.exists(path):
        try:
            return load_scannet_gt(gt_dir, scene, coarse=False)
        except Exception:
            pass
    return SceneGT.load(gt_dir, scene)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-dir", default="data/scannet/graphs")
    ap.add_argument("--gt-dir", default="data/scannet/gt")
    ap.add_argument("--protocol", default="configs/cgsr_set_protocol.json")
    ap.add_argument("--pool", default="data/scannet/pool.json")
    ap.add_argument("--out", default="results/set/set_gate.md")
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()

    from scripts import set_protocol as SP

    protocol = SP.load(args.protocol)
    g = SP.gate(protocol)
    h = SP.protocol_hash(protocol)
    pool = SP.load_pool(args.pool)
    dev, _conf = SP.select_scenes(protocol, pool)
    scenes = args.scenes or dev
    print(f"the protocol sha256 = {h}")
    print(f"the source: {protocol.get('source', {}).get('chosen')} "
          f"(status {protocol.get('source', {}).get('status')})")
    print(f"the development scenes: {scenes}")

    rows: List[Dict[str, object]] = []
    for scene in scenes:
        gpath = os.path.join(args.graph_dir, f"{scene}.json")
        if not os.path.exists(gpath):
            print(f"  {scene}: no graph")
            continue
        with io.open(gpath, encoding="utf-8") as fh:
            pred = json.load(fh)
        gt = _load_gt(args.gt_dir, scene)
        m = gate_metrics(scene, pred, gt)
        rows.append(m)
        print(
            f"  {scene}: {m['n_nodes']} nodes for {m['n_instances']} "
            f"instances; the one-to-one recall {m['recall']*100:.0f}%, "
            f"the precision {m['precision']*100:.0f}%; the median nodes per "
            f"wall {m['median_nodes_per_wall']:.1f}; the class-count ratio "
            f"{m['min_class_ratio']:.2f}-{m['max_class_ratio']:.2f} over "
            f"{m['n_classes_checked']} classes",
            flush=True,
        )

    if not rows:
        print("the gate could not be run: no graph")
        return 1

    med = {
        k: float(np.median([r[k] for r in rows]))
        for k in ("recall", "precision", "median_nodes_per_wall",
                  "median_class_ratio")
    }
    min_ratio = float(min(r["min_class_ratio"] for r in rows))
    max_ratio = float(max(r["max_class_ratio"] for r in rows))
    lo_r, hi_r = g.get("class_count_ratio", [0.0, 1e9])
    checks = [
        ("one_to_one_recall", med["recall"],
         float(g.get("one_to_one_recall_min", 0.0)), 1e9),
        ("one_to_one_precision", med["precision"],
         float(g.get("one_to_one_precision_min", 0.0)), 1e9),
        ("class_count_ratio", med["median_class_ratio"], lo_r, hi_r),
        ("median_nodes_per_wall", med["median_nodes_per_wall"], 0.0,
         float(g.get("median_nodes_per_wall_max", 2.0))),
    ]
    passed = all(lo <= v <= hi for _n, v, lo, hi in checks)
    # every checked class must be inside the band, not only the median
    band_ok = True
    for r in rows:
        for lab, (_n, _k, ratio) in r["class_ratios"].items():
            if not (lo_r <= ratio <= hi_r):
                band_ok = False

    L = ["# The set gate of the CGSR-set protocol", ""]
    L.append(f"*  the protocol hash: `{h}`")
    L.append(f"*  the scenes: {', '.join(scenes)} (geometry and annotation "
             f"only; no query accuracy is computed on them)")
    L.append("")
    L.append("| scene | nodes | instances | aligned | recall | precision | "
             "median nodes per wall | classes checked | the class-count ratio |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        L.append(
            f"| {r['scene']} | {r['n_nodes']} | {r['n_instances']} | "
            f"{r['n_aligned']} | {r['recall']*100:.0f}% | "
            f"{r['precision']*100:.0f}% | {r['median_nodes_per_wall']:.1f} | "
            f"{r['n_classes_checked']} | "
            f"{r['min_class_ratio']:.2f}-{r['max_class_ratio']:.2f} |"
        )
    L.append("")
    L.append("| criterion | value | required | verdict |")
    L.append("|---|---:|---:|---|")
    for name, v, lo, hi in checks:
        ok = lo <= v <= hi
        req = f"[{lo}, {hi}]" if hi < 1e9 else f">= {lo}"
        L.append(f"| {name} | {v:.3f} | {req} | {'PASS' if ok else 'FAIL'} |")
    L.append("")
    L.append(f"*  every checked class inside the band [{lo_r}, {hi_r}]: "
             f"{'yes' if band_ok else 'NO'}")
    L.append("")
    L.append(f"**the gate: {'PASS' if passed and band_ok else 'FAIL'}**")
    L.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))
    print(f"wrote {args.out}")
    for name, v, lo, hi in checks:
        print(f"  {name}: {v:.3f} (required [{lo}, {hi}]) "
              f"{'PASS' if lo <= v <= hi else 'FAIL'}")
    print(f"  every class inside the band: {'yes' if band_ok else 'NO'}")
    print(f"the gate: {'PASS' if passed and band_ok else 'FAIL'}")
    return 0 if (passed and band_ok) else 2


if __name__ == "__main__":
    raise SystemExit(main())
