# FireViewer map production v2 — current status

Status date: **2026-08-20**.

This document records the current map-production boundary so the stable path, the comparison path and the RunPod candidate are not confused during later work.

## Status summary

| Path | Role | Status |
| --- | --- | --- |
| `deploy/Dockerfile.lightning-map-production` | Stable Lightning fallback/current production path | retained unchanged |
| `deploy/Dockerfile.lightning-map-production-compare` | 9-tile Lightning reference image | built/pushed; live comparison pending |
| `deploy/Dockerfile.runpod-map-production` | Legacy RunPod path | retained for rollback/reference |
| `deploy/Dockerfile.runpod-map-production-v2` | factual-v2 RunPod candidate | built/pushed; live comparison pending |

The factual-v2 path is **not promoted as the canonical production provider yet**. Promotion requires the controlled 9-tile comparison and review of its published evidence.

## factual-v2 placement changes

The v2 profile is intentionally additive and isolated from the stable Lightning path.

### Buildings

- BD TOPO footprint is the XY geometry authority for an instantiated building;
- MNT provides ground elevation;
- MNS−MNT provides measured height within the footprint;
- morphology-only building candidates are not instantiated;
- discrete building representation must resolve to a reviewed real catalog USD asset;
- primitive/procedural placeholder buildings are forbidden.

### Trees

The first comparison keeps the legacy 1 m candidate quantity/status so the A/B remains controlled, but can refine final XY/ground/height from the native 0.5 m MNT/MNS pair inside the same original 1 m peak cell.

This count remains an estimate of individual crowns, not a certified count of tree stems. No quota or viewer thinning is introduced.

### Context assets

Road, rail or hydro geometry no longer creates a generic equipment object merely because the source feature exists. Discrete context objects require an explicit validated placement/source and a reviewed catalog asset.

Continuous roads, rail and hydro may still use source-driven procedural geometry because they are geographic surfaces/lines rather than discrete object assets.

## Validation publication

The current comparison pipeline is **folder-native**.

Validation ZIPs are forbidden. The retired ZIP-based validation scripts were removed and replaced by:

- `blender/map_validation_folder.py`;
- `blender/compare_validation_folders.py`.

Published evidence lives under:

```text
validation/<zone_id>/<build_id>/<provider>/
```

It contains compact comparison evidence such as the zone plan/receipt, per-tile placement inventories, source receipts/hashes and the viewer receipt when available.

A validation folder is **evidence only**. It is not a second or simplified map.

## Viewer contract

The browser viewer is the complete browser representation of the corresponding map build.

Canonical runtime publication:

```text
maps/<zone_id>/<build_id>/runtime/
  viewer.glb
  viewer-scene.v1.json
```

The validation job fails closed unless the viewer reports:

- `policy = fail_closed_exact_visual_scene`;
- `mesh_coverage = complete`;
- exact building/tree/context-asset logical instance counts;
- zero placeholder instances;
- matching GLB SHA-256 and byte count.

Runtime optimisation may share meshes, textures and instances, but must not remove canonical logical objects or alter their factual placement.

## Reference validation campaign

The current reference request is the Croix de Justin area near Die.

Canonical request used for both runs:

```json
{
  "fixed_asset_placements": null,
  "latitude": 44.7439034409,
  "longitude": 5.3531898409,
  "side_km": 1.5
}
```

Planned zone:

- `zone_id`: `GPS-0E12F428C04E6EEE`;
- 9 tiles;
- 500 m tile size;
- production bounds L93: `[885500, 6407000, 887000, 6408500]`.

Exactly two paid comparison runs are planned:

1. Lightning compare r2;
2. RunPod factual-v2 r2.

The exact same request must be used for both. No automatic third paid run is part of this validation plan.

## Current images

```text
charlibillabert/fireviewer-simple-production-ui:pilot-v2-20260820-r2-lightning-compare
charlibillabert/fireviewer-simple-production-ui:pilot-v2-20260820-r2-runpod-validation
```

The images have been built and pushed. This does not count as live map-production validation.

## Promotion gate

Before factual-v2 can replace the stable path, record at least:

- successful publication of both validation folders;
- successful publication of both complete viewers;
- same 9-tile identities/origins;
- source/hash comparability;
- building/tree/context counts;
- XY and height deltas;
- zero placeholders;
- viewer completeness;
- runtime and artifact-size measurements;
- explicit promotion/rejection decision.

## Known work after the comparison

Even if the first v2 comparison passes, the following remain separate work items:

- complete dimension-aware real-asset matching review;
- future tree/canopy segmentation improvements beyond the controlled 1 m candidate count;
- final continuous road/rail/hydro integration checks;
- provider-neutral backend callbacks for the final RunPod production wrapper;
- full scientific/source-folder publication from the promoted provider;
- recovery/cancellation and archived cost/runtime measurements.

Do not infer GPU acceleration from the RunPod hardware choice: the current map pipeline is still predominantly CPU-bound and must be benchmarked before making performance claims.
