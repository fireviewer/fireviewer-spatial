# FireViewer Spatial

**Headless map production, portable spatial packages and temporal fire layers for reproducible FireViewer incidents.**

This repository owns the spatial-production side of FireViewer: building a measured geographic reference, validating and sealing its artifacts, and producing time-indexed perimeter layers that can later be replayed without rebuilding the terrain.

The cross-project architecture and project positioning are maintained in [`fireviewer/Fireviewer_doc`](https://github.com/fireviewer/Fireviewer_doc).

> FireViewer Spatial is a research/engineering component. It is not an emergency-response system, a certified wildfire forecast or an official geographic authority.

## Role in FireViewer

```text
map request
   ↓
headless spatial production
   ↓
immutable map package
   ├── observed/reviewed temporal layers
   ├── replay / post-event studies
   ├── datasets / benchmarks
   └── browser-friendly derived views
```

The browser is a consumer of these artifacts. It is not responsible for reconstructing the terrain.

## Canonical map builder

The active engine [`simple_production_api.py`](./blender/simple_production_api.py) exposes a headless production path from a geographic centre and requested square side length.

The build is planned on a **Lambert-93 / EPSG:2154 grid of 500 m tiles**. For each tile, the pipeline can temporarily acquire:

- MNT / terrain elevation;
- MNS / surface elevation;
- orthophotography;
- geographic context required by the active placement rules.

It then produces, among other artifacts:

- deterministic terrain geometry with multiple LODs;
- a baked local ground texture;
- context/object placement derived from measured `MNS − MNT` information and validated rules;
- references to the exact assets actually used;
- compact provenance receipts;
- integrity hashes.

Raw geographic rasters are temporary processing inputs and are not required in the delivered map ZIP once the tile has passed validation.

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
shared/prototype-bundles/v1-<sha256>/
provenance/<tile>/
```

<<<<<<< HEAD
`zone.usda` définit chaque prototype utilisé une seule fois et charge des
payloads regroupant jusqu'à 4 × 4 tuiles. `zone.blend` conserve les arbres et
bâtiments comme PointInstancers et est sauvegardé avec la compression interne
de Blender. La production active ne calcule plus de galerie PNG ; elle refuse
un pack hors budget avant tout upload, puis privilégie son contrôle dans
l'administration et son ouverture indépendante.
=======
`zone.usda` and `zone.blend` are autonomous scene representations of the accepted build.
>>>>>>> 6c8ce484031a1e766f0d2ad22ffe23523d05ab54

The active production path prioritises the **validated package itself**. A PNG capture gallery is not part of the canonical measured-map contract.

Active contracts include:

- `fireviewer.simple-measured-map-package.v2`;
- `fireviewer.simple-measured-map-upload-contract.v2`.

## Endpoint-driven execution

Map production is designed to run behind a provider-independent job boundary.

The FireViewer administration calls the backend, which can submit an asynchronous map job to the current compute provider. Provider credentials remain server-side.

The current Lightning-oriented deployment path is documented in [`docs/SIMPLE_PRODUCTION_POD.md`](./docs/SIMPLE_PRODUCTION_POD.md) and uses an ephemeral batch workload rather than keeping heavy compute online permanently.

The HTTP/job contract is described by [`simple_production_api_contract.v1.json`](./blender/simple_production_api_contract.v1.json).

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

Active contracts include:

- `fireviewer.observed-perimeter-package.v1`;
- `fireviewer.observed-perimeter-upload-contract.v1`.

## Retrospective reconstruction

Historical FireViewer reconstruction packs are a separate artifact family.

A geometry marked `reconstructed` can be derived from historical maps, area reports, sectors, remote-sensing information and reviewed evidence, but it must not be published under the observed-perimeter contract.

```text
observed ≠ reconstructed ≠ simulated ≠ predicted
```

## Replay and downstream consumers

[`scene-consumer-input.schema.json`](./contracts/spatial/v1/scene-consumer-input.schema.json) allows a replay, dataset or optional simulation workflow to reference an accepted map build and temporal package without silently reconstructing them.

A consumer can create a **new derived study or simulation artifact**, but it must preserve the identity of the exact inputs it consumed.

The original map ZIP remains the canonical spatial artifact when published or archived.

## No Unity / Omniverse dependency in the core path

The canonical FireViewer map builder does **not** depend on Unity or NVIDIA Omniverse.

Older experiments may remain in repository history or specialised directories. Omniverse can also remain useful inside optional synthetic-data research in `fireviewer-sdg`, but it is not a dependency of this repository's canonical production contract.

## Coordinate systems

- request coordinates: `EPSG:4326`, longitude/latitude order in JSON;
- production CRS: `EPSG:2154` / Lambert-93;
- current vertical reference: `NGF-IGN69` where required by the active contract;
- OpenUSD units: metres, Z-up.

## Package integrity

[`portable_scene_package.py`](./blender/portable_scene_package.py) inventories and hashes accepted package content before archive publication.

A reproducible package should preserve enough information to identify:

- exact build/contract revision;
- source receipts;
- CRS and transformations;
- used prototype/asset bundle;
- integrity hashes;
- exceptional repairs or source gaps.

The canonical doctrine is documented in [FireViewer Provenance and Reproducibility](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/PROVENANCE_AND_REPRODUCIBILITY.md).

## Local validation

```powershell
$env:TEMP = 'C:\FireViewerWorkspace\temp'
$env:TMP = $env:TEMP
$env:PYTHONPYCACHEPREFIX = 'C:\FireViewerWorkspace\cache\pycache'

python -m pytest -q
python -m ruff check .
```

A green local suite does not prove live geographic-source availability, deployed provider reliability or field accuracy. Those gates are tracked separately in the canonical [Status Matrix](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/STATUS_MATRIX.md).

## Data and licences

This Git repository does not contain production maps, orthophotos, generated scene archives, model weights, tokens or production datasets.

The code is licensed under AGPL-3.0-or-later and the repository documentation under CC BY 4.0. External geographic sources and assets retain their own licences and attribution requirements.

## Support and collaboration

The spatial workstream benefits directly from CPU compute credits, object storage, bandwidth, GIS/geodesy expertise and independent package/replay validation.

See the FireViewer [Funding Brief](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/FUNDING_BRIEF.md) and [Support & Partnerships](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/SUPPORT_AND_PARTNERSHIPS.md).

## Contact

FireViewer is maintained by **Unicorn Who Dev**.

Research collaboration, infrastructure support, provenance, security and data-removal requests: **unicornwhodev@gmail.com**.
