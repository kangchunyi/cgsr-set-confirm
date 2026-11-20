# CGSR-set class-conditioned confirmation gate

Code-only release for the confirmation screen in

> **77/129 = 0.597 < 0.75: A Failed Set-Quality Gate for ScanNet Ordinal-Query Sources**
> Chunyi Kang, 2026

This repository contains the frozen protocol, the gate runner, the
deterministic same-label centroid/AABB matcher, and the tests. It contains
no ScanNet data, ground truth, predicted graphs, model weights, or derived
result files. See `DATA.md`.

## What the gate measures

A predicted node matches an annotated instance only if the two share a
canonical class and the predicted centroid lies inside the annotated
axis-aligned box padded by 0.25 m. Matching is one-to-one within a class,
maximizes cardinality, then minimizes total centroid distance, then breaks
exact ties deterministically.

A scene passes only if:

1. one-to-one precision is at least 0.75;
2. one-to-one recall is at least 0.75;
3. every class with at least two annotated instances has a predicted count
   in [0.75, 1.25];
4. the median number of wall-labelled nodes per annotated wall is at most 2.

A source passes only if all ten confirmation scenes pass. The screen
computes no query accuracy; it is a screening/construct diagnostic, not a
mask-identity test.

## Install

Python 3.10+:

```bash
python -m pip install -r requirements.txt
```

## Tests (synthetic data only)

```bash
python -m pytest \
  scripts/test_run_class_conditioned_gate.py \
  scripts/test_class_aware_gate_diagnostic.py \
  scripts/test_gate_wall_metric_audit.py -q
```

The tests can also be run directly:

```bash
python scripts/test_run_class_conditioned_gate.py
python scripts/test_class_aware_gate_diagnostic.py
python scripts/test_gate_wall_metric_audit.py
```

## Run the frozen gate

Expected data layout (not included):

```text
data/scannet/gt/<scene>.json
data/scannet/graphs/<scene>.json          # cgsr-set
data/scannet/graphs_new2/<scene>.json     # cgsr-set2
data/scannet/graphs_cloud/<scene>.json    # cloud, negative control
```

Command:

```bash
python scripts/run_class_conditioned_gate.py \
  --protocol configs/cgsr_set_confirm_protocol.json \
  --gt-dir data/scannet/gt \
  --graph-root . \
  --expect-hash 58114ef4c2737b31a2b66834479c2ebf7296b0a16e696a734c0658e2bb24f1ba \
  --out-json results/class_conditioned_gate_confirmation.json \
  --out-md results/class_conditioned_gate_confirmation.md
```

The `--expect-hash` guard makes the run void if the frozen protocol changes.
The protocol fixes the ten confirmation scenes, the matching rule, the four
criteria, and the complete-source verdict.

## Reproducibility boundary

The runner and matcher are the frozen versions whose SHA-256 hashes are
recorded in the paper's remote replay; the supporting modules are included
as dependencies. The source graphs and ground-truth files are not
redistributed, so a rerun on different data need not reproduce the paper's
numbers. The paper's saved report and its remote replay are described in
the manuscript.

## Data and license

Code: MIT (see `LICENSE`). No data rights are granted. ScanNet and any
third-party assets remain subject to their own terms; see `DATA.md`.

## Citation

If you use this code, cite the paper above. A BibTeX entry will be added
when the venue is fixed.
