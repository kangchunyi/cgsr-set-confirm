"""The class-conditioned confirmation gate.

This runs the gate that ``configs/cgsr_set_confirm_protocol.json`` freezes:
a predicted node and an annotated instance must match in canonical class as
well as geometry, and a source passes only if every one of the ten
confirmation scenes passes every criterion.  It is a screening study; it
computes no query accuracy.

The matcher is ``scripts/class_aware_gate_diagnostic.py``; this script adds
the frozen protocol, the complete-source rule, the per-scene verdicts, the
wall-proposal count, and the reports.  It refuses to write a verdict for a
source whose ten graphs are not all present.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from scripts.class_aware_gate_diagnostic import (  # noqa: E402
    canonical_label,
    class_conditioned_metrics,
)
from scripts.scannet_gt import load_gt  # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with io.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scene_report(
    scene: str,
    graph: Dict[str, object],
    gt,
    gate: Dict[str, object],
    tol: float,
) -> Dict[str, object]:
    """One scene's class-conditioned gate statistics and verdict."""
    nodes = list(graph.get("nodes", []))
    m = class_conditioned_metrics(nodes, gt, tol=tol)

    lo, hi = (float(v) for v in gate["class_count_ratio"])
    min_inst = 2
    per_class = {}
    for label, row in m["per_class"].items():
        if label == "<missing>" or int(row["ground_truth"]) < min_inst:
            continue
        pred = int(row["predicted"])
        truth = int(row["ground_truth"])
        ratio = pred / truth
        per_class[label] = {
            "predicted": pred,
            "ground_truth": truth,
            "ratio": ratio,
            "in_band": lo <= ratio <= hi,
        }
    ratios = [row["ratio"] for row in per_class.values()]
    wall_counts = {str(k): int(v) for k, v in m["wall_proposal_counts"].items()}
    median_wall = (
        float(np.median(list(wall_counts.values()))) if wall_counts else 0.0
    )

    precision = m["precision"]
    recall = m["recall"]
    checks = {
        "precision": {
            "value": precision,
            "required": f">= {float(gate['one_to_one_precision_min'])}",
            "passed": precision is not None
            and float(precision) >= float(gate["one_to_one_precision_min"]),
        },
        "recall": {
            "value": recall,
            "required": f">= {float(gate['one_to_one_recall_min'])}",
            "passed": recall is not None
            and float(recall) >= float(gate["one_to_one_recall_min"]),
        },
        "class_counts": {
            "value": (
                f"{sum(1 for r in ratios if lo <= r <= hi)}/{len(ratios)}"
                if ratios else "no checked class"
            ),
            "required": f"every ratio in [{lo}, {hi}]",
            "passed": all(lo <= r <= hi for r in ratios) if ratios else True,
        },
        "median_wall_proposals": {
            "value": median_wall,
            "required": (
                f"<= {float(gate['median_wall_proposals_max'])}"
            ),
            "passed": median_wall <= float(gate["median_wall_proposals_max"]),
        },
    }
    verdict = all(c["passed"] for c in checks.values())
    return {
        "scene": scene,
        "n_nodes": int(m["n_predicted"]),
        "n_instances": int(m["n_ground_truth"]),
        "n_matched": int(m["n_matched"]),
        "precision": precision,
        "recall": recall,
        "classes_checked": len(ratios),
        "classes_in_band": sum(1 for r in ratios if lo <= r <= hi),
        "min_class_ratio": min(ratios) if ratios else None,
        "max_class_ratio": max(ratios) if ratios else None,
        "per_class": per_class,
        "n_walls": len(wall_counts),
        "median_wall_proposals": median_wall,
        "max_wall_proposals": max(wall_counts.values()) if wall_counts else 0,
        "wall_proposal_counts": wall_counts,
        "matched_pairs": m["matched_pairs"],
        "checks": checks,
        "verdict": "PASS" if verdict else "FAIL",
    }


def _pooled(rows: List[Dict[str, object]]) -> Dict[str, object]:
    nodes = sum(int(r["n_nodes"]) for r in rows)
    inst = sum(int(r["n_instances"]) for r in rows)
    matched = sum(int(r["n_matched"]) for r in rows)
    return {
        "scenes": len(rows),
        "nodes": nodes,
        "instances": inst,
        "matched": matched,
        "precision": matched / nodes if nodes else None,
        "recall": matched / inst if inst else None,
        "scenes_passed": sum(1 for r in rows if r["verdict"] == "PASS"),
    }


def run_source(
    name: str,
    spec: Dict[str, object],
    protocol: Dict[str, object],
    gt_dir: str,
    graph_root: str,
    gate: Dict[str, object],
    tol: float,
) -> Dict[str, object]:
    """Evaluate one complete source, or report it incomplete."""
    scenes = list(protocol["scenes"]["confirmation"])
    graph_dir = os.path.join(graph_root, str(spec["graph_dir"]))
    missing = [
        scene for scene in scenes
        if not os.path.isfile(os.path.join(graph_dir, f"{scene}.json"))
    ]
    if missing:
        return {
            "status": "incomplete",
            "graph_dir": graph_dir,
            "missing_scenes": missing,
            "verdict": None,
            "scenes": [],
        }

    rows = []
    for scene in scenes:
        with io.open(
            os.path.join(graph_dir, f"{scene}.json"), encoding="utf-8"
        ) as fh:
            graph = json.load(fh)
        gt = load_gt(gt_dir, scene, coarse=False)
        rows.append(scene_report(scene, graph, gt, gate, tol))

    verdict = "PASS" if all(r["verdict"] == "PASS" for r in rows) else "FAIL"
    return {
        "status": "complete",
        "graph_dir": graph_dir,
        "missing_scenes": [],
        "verdict": verdict,
        "scenes": rows,
        "pooled_descriptive_only": _pooled(rows),
    }


def sensitivity(
    sources: Dict[str, Dict[str, object]],
    protocol: Dict[str, object],
) -> Dict[str, object]:
    """Descriptive replay without the poorly registered scenes."""
    excluded = set(protocol.get("sensitivity", {}).get("scenes", []))
    out = {}
    for name, source in sources.items():
        if source.get("status") != "complete":
            continue
        rows = [r for r in source["scenes"] if r["scene"] not in excluded]
        out[name] = {
            "excluded": sorted(excluded),
            "role": "descriptive only; not a gate verdict",
            "scenes_evaluated": len(rows),
            "scenes_passed": sum(1 for r in rows if r["verdict"] == "PASS"),
            "pooled": _pooled(rows),
        }
    return out


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def render_md(
    protocol: Dict[str, object],
    protocol_path: str,
    protocol_hash: str,
    sources: Dict[str, Dict[str, object]],
    diagnostics: Dict[str, Dict[str, object]],
    sens: Dict[str, object],
) -> str:
    scenes = list(protocol["scenes"]["confirmation"])
    L = [
        "# The class-conditioned gate on the confirmation scenes",
        "",
        f"*  the protocol: `{protocol_path}`",
        f"*  the protocol sha256: `{protocol_hash}`",
        f"*  the scenes: {len(scenes)} "
        f"({', '.join(scenes)})",
        "*  the matching: canonical class and centroid inside the padded "
        "annotated box, one-to-one per class",
        "*  no query accuracy is computed",
        "",
        "## Summary",
        "",
        "| source | scenes | nodes | instances | matched | pooled precision | "
        "pooled recall | scenes passed | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, source in sources.items():
        if source.get("status") != "complete":
            L.append(
                f"| {name} | {len(scenes)} | - | - | - | - | - | - | "
                f"INCOMPLETE |"
            )
            continue
        p = source["pooled_descriptive_only"]
        L.append(
            f"| {name} | {p['scenes']} | {p['nodes']} | {p['instances']} | "
            f"{p['matched']} | {_pct(p['precision'])} | {_pct(p['recall'])} | "
            f"{p['scenes_passed']}/{p['scenes']} | {source['verdict']} |"
        )
    L += ["", "Pooled values are descriptive only; a source's verdict is "
          "decided scene by scene.", ""]

    for name, source in sources.items():
        L += [f"## {name}", ""]
        if source.get("status") != "complete":
            L += [
                f"The source is incomplete: {len(source['missing_scenes'])} "
                f"of {len(scenes)} confirmation graphs are missing "
                f"({', '.join(source['missing_scenes'])}). No verdict is "
                "reported.",
                "",
            ]
            continue
        L += [
            "| scene | nodes | instances | matched | precision | recall | "
            "classes in band | median wall proposals | verdict |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in source["scenes"]:
            band = (
                f"{r['classes_in_band']}/{r['classes_checked']}"
                if r["classes_checked"] else "none"
            )
            L.append(
                f"| {r['scene']} | {r['n_nodes']} | {r['n_instances']} | "
                f"{r['n_matched']} | {_pct(r['precision'])} | "
                f"{_pct(r['recall'])} | {band} | "
                f"{r['median_wall_proposals']:.2f} | {r['verdict']} |"
            )
        L += ["", f"**the source: {source['verdict']}**", ""]
        for r in source["scenes"]:
            failed = [k for k, c in r["checks"].items() if not c["passed"]]
            if failed:
                L.append(f"*  `{r['scene']}` fails on {', '.join(failed)}")
        L.append("")

    if diagnostics:
        L += ["## Diagnostics (descriptive only)", ""]
        for name, row in diagnostics.items():
            if row.get("status") != "complete":
                L.append(
                    f"*  {name}: incomplete "
                    f"({len(row.get('missing_scenes', []))} graphs missing); "
                    "no descriptive run."
                )
                continue
            p = row["pooled_descriptive_only"]
            L.append(
                f"*  {name}: {p['scenes_passed']}/{p['scenes']} scenes pass "
                f"the same criteria; pooled precision {_pct(p['precision'])}, "
                f"recall {_pct(p['recall'])}. The role is descriptive; this "
                "is not a source verdict."
            )
        L.append("")

    L += ["## Sensitivity (descriptive only)", ""]
    for name, row in sens.items():
        p = row["pooled"]
        L.append(
            f"*  {name}: {row['scenes_passed']}/{row['scenes_evaluated']} "
            f"scenes pass after excluding {', '.join(row['excluded'])}; "
            f"pooled precision {_pct(p['precision'])}, recall "
            f"{_pct(p['recall'])}. This is not a verdict."
        )
    L += [
        "",
        "## OpenMask3D",
        "",
        f"Not evaluated: {protocol['sources']['openmask3d']['reason']}",
        "",
    ]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--protocol", default="configs/cgsr_set_confirm_protocol.json"
    )
    ap.add_argument("--gt-dir", default="data/scannet/gt")
    ap.add_argument(
        "--graph-root", default=".",
        help="the directory the protocol's graph_dir paths are relative to",
    )
    ap.add_argument("--expect-hash", default=None)
    ap.add_argument(
        "--out-json",
        default="results/set/class_conditioned_gate_confirmation.json",
    )
    ap.add_argument(
        "--out-md",
        default="results/set/class_conditioned_gate_confirmation.md",
    )
    args = ap.parse_args()

    protocol_path = args.protocol
    if not os.path.isabs(protocol_path):
        protocol_path = os.path.join(ROOT, protocol_path)
    with io.open(protocol_path, encoding="utf-8") as fh:
        protocol = json.load(fh)
    protocol_hash = sha256_file(protocol_path)

    if args.expect_hash and args.expect_hash.lower() != protocol_hash:
        print(
            f"the protocol hash is {protocol_hash}, not the frozen "
            f"{args.expect_hash}; the run is void",
            file=sys.stderr,
        )
        return 4

    gate = protocol["gate"]
    tol = float(protocol["matching"]["tolerance_m"])
    scenes = protocol["scenes"]["confirmation"]
    if len(scenes) != 10:
        print(f"the confirmation list has {len(scenes)} scenes, not 10",
              file=sys.stderr)
        return 5

    sources: Dict[str, Dict[str, object]] = {}
    for name, spec in protocol["sources"].items():
        if spec.get("status") != "available":
            continue
        print(f"=== {name} ===", flush=True)
        sources[name] = run_source(
            name, spec, protocol, args.gt_dir, args.graph_root, gate, tol
        )
        if sources[name]["status"] == "complete":
            s = sources[name]["pooled_descriptive_only"]
            print(
                f"  {s['scenes_passed']}/{s['scenes']} scenes pass; "
                f"pooled precision {_pct(s['precision'])}, recall "
                f"{_pct(s['recall'])}; the source: "
                f"{sources[name]['verdict']}",
                flush=True,
            )
        else:
            print(
                f"  incomplete: {len(sources[name]['missing_scenes'])} "
                f"graphs missing",
                flush=True,
            )

    sens = sensitivity(sources, protocol)
    diagnostics: Dict[str, Dict[str, object]] = {}
    for name, spec in protocol.get("diagnostics", {}).items():
        print(f"=== diagnostic: {name} ===", flush=True)
        diagnostics[name] = run_source(
            name, spec, protocol, args.gt_dir, args.graph_root, gate, tol
        )
        diagnostics[name]["role"] = spec.get("role")
        if diagnostics[name]["status"] == "complete":
            p = diagnostics[name]["pooled_descriptive_only"]
            print(
                f"  {p['scenes_passed']}/{p['scenes']} scenes pass; "
                f"pooled precision {_pct(p['precision'])}, recall "
                f"{_pct(p['recall'])} (descriptive)",
                flush=True,
            )
        else:
            print(
                f"  incomplete: {len(diagnostics[name]['missing_scenes'])} "
                f"graphs missing",
                flush=True,
            )
    payload = {
        "protocol": {
            "path": os.path.relpath(protocol_path, ROOT),
            "sha256": protocol_hash,
            "name": protocol.get("name"),
        },
        "matching": protocol["matching"],
        "gate": gate,
        "sources": sources,
        "diagnostics": diagnostics,
        "sensitivity": sens,
        "openmask3d": protocol["sources"]["openmask3d"],
    }

    for path in (args.out_json, args.out_md):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with io.open(args.out_json, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    with io.open(args.out_md, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_md(
            protocol, os.path.relpath(protocol_path, ROOT), protocol_hash,
            sources, diagnostics, sens,
        ) + "\n")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")

    if any(s["status"] != "complete" for s in sources.values()):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
