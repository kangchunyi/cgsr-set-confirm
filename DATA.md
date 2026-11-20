# Data availability and expected schemas

This is a code-only release. It does not contain or redistribute:

- ScanNet RGB-D frames, meshes, or annotations;
- ScanNet200 ground-truth instances;
- predicted scene graphs from HOV-SG, SAM 2.1 + CLIPSeg, or the cloud
  proposal;
- model checkpoints or weights;
- derived result JSON or Markdown files.

The manuscript's data-use section asserts no permission to redistribute
ScanNet data or derived files. Obtain ScanNet from its official release and
follow its terms of use. The code expects the local layout below; the user
supplies the data.

## Ground truth

`data/scannet/gt/<scene>.json`:

```json
{
  "scene": "scene0015_00",
  "n_instances": 53,
  "instances": [
    {
      "object_id": 0,
      "raw_name": "chair",
      "coarse_name": "",
      "bbox": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
      "centroid": [0.5, 0.5, 0.5],
      "n_points": 1234
    }
  ]
}
```

`raw_name` is the label source used by the matcher; `coarse_name` may be
empty. The coordinates are metres in the ScanNet world frame.

## Predicted graph

`data/scannet/graphs/<scene>.json` and the sibling source directories:

```json
{
  "scene": "scene0015_00",
  "variant": "cgsr-set",
  "nodes": [
    {
      "id": "p0",
      "label": "chair",
      "centroid": [0.5, 0.5, 0.5],
      "bbox": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    }
  ],
  "edges": [],
  "meta": {}
}
```

An empty `label` never matches a labelled instance. The matcher reads `id`,
`label`, `centroid`, and `bbox`; other node fields are ignored.

## Frozen protocol

`configs/cgsr_set_confirm_protocol.json`, SHA-256:

```text
58114ef4c2737b31a2b66834479c2ebf7296b0a16e696a734c0658e2bb24f1ba
```

The protocol lists the ten confirmation scenes and the exact matching and
verdict rules. It is the file that must be passed to `--protocol`.
