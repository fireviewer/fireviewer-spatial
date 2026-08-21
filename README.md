# FireViewer Spatial

**Headless map production, portable spatial packages and temporal fire layers for reproducible FireViewer incidents.**

This repository owns the spatial-production side of FireViewer: building measured geographic references, validating and sealing spatial artifacts, producing complete browser viewers and generating time-indexed perimeter layers that can later be replayed without rebuilding the terrain.

The cross-project architecture and project positioning are maintained in [`fireviewer/Fireviewer_doc`](https://github.com/fireviewer/Fireviewer_doc).

> FireViewer Spatial is a research/engineering component. It is not an emergency-response system, a certified wildfire forecast or an official geographic authority.

## Role in FireViewer

```text
map request
   ↓
headless spatial production
   ↓
sealed map build
   ├── complete browser viewer
   ├── validation / provenance evidence
   ├── scientific/source artifacts
   ├── temporal layers
   └── replay / datasets / benchmarks
```

The browser is a consumer of derived artifacts. It is not responsible for reconstructing the terrain.

## Canonical map builder

The active engine [`simple_production_api.py`](./blender/simple_production_api.py) exposes a headless production path from a geographic centre and requested square side length.

The build is planned on a **Lambert-93 / EPSG:2154 grid of 500 m tiles**. Per-tile processing can temporarily acquire MNT, MNS, orthophotography and geographic context required by the active placement profile.

The builder produces, among other artifacts:

- deterministic terrain geometry with multiple LODs;
- a baked local ground texture;
- measured/contextual object placement;
- references to the exact reviewed assets actually used;
- compact provenance receipts;
- integrity hashes;
- portable OpenUSD/Blender scene artifacts.

Raw geographic rasters are temporary processing inputs and do not need to remain in the sealed publication folder once their derived artifacts have passed validation.

## Current production transition

The stable Lightning production path remains available as a fallback.

A parallel **factual-v2** path has been implemented for controlled provider comparison. It tightens several placement rules without modifying the stable fallback. It is not yet promoted as the canonical production provider; representative live validation remains a gate.

The current deployment/validation boundary is documented in [`docs/SIMPLE_PRODUCTION_POD.md`](./docs/SIMPLE_PRODUCTION_POD.md).

### factual-v2 in brief

- buildings require the expected BD TOPO footprint authority, with MNT/MNS providing measured ground/height;
- morphology-only building candidates are not instantiated;
- tree quantity/status remains compatible with the historical 1 m detector for the first controlled comparison, while final position/height can be refined from native 0.5 m elevation inside the same source cell;
- road/rail/hydro features no longer imply generic discrete equipment objects;
- discrete buildings, trees and context equipment must use reviewed real catalog assets rather than placeholder primitives.

## Zone package

A completed zone contains files such as:

```text
zone.usda
zone.blend
zone.done.json
zone-plan.json
zone-context.json
packages/<tile>/
payloads/
shared/prototype-bundles/
provenance/<tile>/
```

`zone.usda` and `zone.blend` are portable scene representations of the accepted build. The production artifact is the validated spatial package itself with its provenance and integrity metadata, not a screenshot gallery.

Active map package contracts include:

- `fireviewer.simple-measured-map-package.v2`;
- `fireviewer.simple-measured-map-upload-contract.v2`.

## Folder-native validation

New validation evidence is **folder-native**. Validation ZIPs are not part of the current comparison contract.

The bounded comparison runner publishes compact evidence under:

```text
validation/<zone_id>/<build_id>/<provider>/
```

This evidence can include zone plans/receipts, per-tile placement inventories, source receipts/hashes and a viewer receipt.

A validation folder is **not another map**. It exists only to compare two builds without duplicating large runtime/scientific artifacts.

The comparison helpers are:

- [`map_validation_job.py`](./blender/map_validation_job.py);
- [`map_validation_folder.py`](./blender/map_validation_folder.py);
- [`compare_validation_folders.py`](./blender/compare_validation_folders.py);
- [`plan_9_tile_validation.py`](./blender/plan_9_tile_validation.py).

The first provider comparison is deliberately limited to exactly 9 tiles and requires the same request on both workers.

## Complete browser viewer

The public browser viewer is a **complete representation of the corresponding map build**, not a simplified second map.

Canonical runtime layout:

```text
maps/<zone_id>/<build_id>/runtime/
  viewer-tiled/
    far.glb
    catalog.json
    viewer-tiled-scene.v1.json
    prototypes/*.glb
    tiles/<tile_id>/terrain.glb
    tiles/<tile_id>/instances.fvi
```

The validation path fails closed unless every catalogued payload matches its SHA-256 and byte count, every GLB is self-contained, and tile/prototype counts exactly match the sealed zone. The tiled viewer is built directly from sealed tile packages and `zone.blend`; a monolithic `viewer.glb` is optional test evidence for small maps only.

Runtime optimisation may share meshes, textures and instances, but must not remove canonical logical objects or alter factual placement.

## Public artifact repository

Measured spatial builds and their validation evidence are published in the public Hugging Face dataset:

[`fireviewer/simple-measured-scenes-v1`](https://huggingface.co/datasets/fireviewer/simple-measured-scenes-v1)

The backend records immutable viewer identity using repository, revision, path, hash, size and completeness metadata.

Publishing a tiled package to Hugging Face does not automatically make it the active map of an incident. Incident attachment/replacement remains an explicit, versioned backend action.

## Endpoint-driven execution

Map production runs behind a provider-independent job boundary.

The FireViewer administration calls the backend, which dispatches a bounded asynchronous spatial job to the currently selected compute provider. Provider credentials remain server-side.

The heavy map-production environment is intended to be ephemeral rather than permanently online.

Provider choice is an implementation detail below the map-job contract and should not change incident identity, provenance or publication semantics.

## Observed perimeter layers

[`geographic_perimeter_layer.py`](./blender/geographic_perimeter_layer.py) normalises FireViewer JSON/GeoJSON geometry to `EPSG:2154` and can produce:

```text
geographic-perimeters.usda
fire-progression-timeline.json
perimeters.normalized.json
perimeter-layer.manifest.json
preview/perimeter-viewer.manifest.json
preview/frame-*.glb
```

Each state belongs to an explicit observation time or interval. Between known states, the canonical value may remain `undefined`.

No interpolation, propagation speed or future perimeter is silently invented.

The OpenUSD layer, normalised geometry and timeline are reference artifacts. Browser-oriented GLB files are derived views.

## Retrospective reconstruction

Historical reconstruction packs are a separate artifact family.

A geometry marked `reconstructed` may be derived from historical maps, reports, sectors, remote-sensing information and reviewed evidence, but it must not be published under the observed-perimeter contract.

```text
observed ≠ reconstructed ≠ interpolated ≠ simulated ≠ predicted
```

## Replay and downstream consumers

[`scene-consumer-input.schema.json`](./contracts/spatial/v1/scene-consumer-input.schema.json) allows replay, dataset and optional simulation workflows to reference accepted map builds and temporal packages without silently reconstructing them.

A consumer can create a new derived study or simulation artifact, but it must preserve the identity of the exact inputs it consumed.

## No Unity / Omniverse dependency in the core path

The canonical FireViewer map builder does **not** depend on Unity or NVIDIA Omniverse.

Older experiments may remain in repository history or specialised directories. Omniverse may remain useful inside optional synthetic-data research in `fireviewer-sdg`, but it is not a runtime dependency of this repository's canonical production contract.

## Coordinate systems

- request coordinates: `EPSG:4326`;
- production CRS: `EPSG:2154` / Lambert-93;
- current vertical reference: `NGF-IGN69` where required by the active contract;
- OpenUSD units: metres, Z-up.

## Package integrity

[`portable_scene_package.py`](./blender/portable_scene_package.py) inventories and hashes accepted package content before publication.

A reproducible package should preserve enough information to identify the exact build/contract revision, source receipts, CRS/transforms, used asset bundle, integrity hashes and exceptional repairs/source gaps.

The canonical doctrine is documented in [FireViewer Provenance and Reproducibility](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/PROVENANCE_AND_REPRODUCIBILITY.md).

## Validation status

A successful image build or local unit test does not prove live provider reliability or spatial accuracy.

The factual-v2/provider comparison remains a live-validation gate. Until it is completed, the stable fallback remains retained and no README should present the candidate path as fully promoted production.

Project-wide evidence levels are maintained in the canonical [Status Matrix](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/STATUS_MATRIX.md).

## Local validation

```powershell
python -m pytest -q
python -m ruff check .
```

A green local suite does not prove live geographic-source availability, deployed provider reliability or field accuracy.

## Data and licences

This Git repository does not contain production maps, raw orthophotos, model weights, provider credentials or runtime secrets.

The code is licensed under **AGPL-3.0-or-later** through [`LICENSE`](LICENSE). Repository documentation is licensed under **CC BY 4.0** through [`LICENSE-DOCS.md`](LICENSE-DOCS.md). External geographic sources and assets retain their own licences and attribution requirements.

## Project documentation

- [FireViewer documentation](https://github.com/fireviewer/Fireviewer_doc)
- [Map Builder](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/MAP_BUILDER.md)
- [Status Matrix](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/STATUS_MATRIX.md)
- [Provenance and Reproducibility](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/PROVENANCE_AND_REPRODUCIBILITY.md)

## Contact

FireViewer is maintained by **Unicorn Who Dev**.

Research collaboration, provenance, security and data-removal requests: **unicornwhodev@gmail.com**.
