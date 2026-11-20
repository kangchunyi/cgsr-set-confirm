"""The ScanNet annotation, in the shape the gate and the scorer expect.

The ScanNet instance annotation is a pair of files: the aggregation file
(the instances, their raw class names and their axis-aligned boxes) and
the segments file (the twenty-class label of every segment).  This module
loads them into the same duck-typed shape as ``scripts.edge_reliability``'s
``SceneGT``, so the gate and the scorer can be written once.
"""

from __future__ import annotations

import io
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np


class ScanNetGT:
    """The annotated instances of one ScanNet scene."""

    def __init__(self, path: str, coarse: bool = True):
        with io.open(path, encoding="utf-8") as fh:
            j = json.load(fh)
        inst = j["instances"]
        self.scene = j["scene"]
        self.object_ids = np.asarray(
            [int(i["object_id"]) for i in inst], dtype=np.int64
        )
        self.aabbs = np.asarray(
            [i["bbox"] for i in inst], dtype=float
        ).reshape(-1, 2, 3) if inst else np.zeros((0, 2, 3))
        self.centroids = (
            self.aabbs.mean(axis=1) if len(self.aabbs) else
            np.zeros((0, 3))
        )
        key = "coarse_name" if coarse else "raw_name"
        self.names = {
            str(i["object_id"]): str(i[key]) for i in inst
        }
        self.raw_names = {
            str(i["object_id"]): str(i["raw_name"]) for i in inst
        }
        self.coarse_names = {
            str(i["object_id"]): str(i["coarse_name"]) for i in inst
        }

    def __len__(self) -> int:
        return len(self.object_ids)

    def object_index(self) -> Dict[int, int]:
        return {int(v): i for i, v in enumerate(self.object_ids)}


def load_gt(gt_dir: str, scene: str, coarse: bool = True) -> ScanNetGT:
    """The annotation of one scene, from the prepared directory."""
    return ScanNetGT(os.path.join(gt_dir, f"{scene}.json"), coarse=coarse)
