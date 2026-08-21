# Map Builder reference v1

This directory records the immutable semantic reference for the AWS migration.
The engine commit is tagged `map-builder-reference-v1`; the observed golden build
is `dad37cf44fc83faa6469d9483abec897d9648daece402d0e6d22f1d8cfd8887b`.

`request.json`, `semantic-baseline.json`, `metrics.json` and `hashes.json` are
versioned.  Heavy artifacts are preserved locally below `artifacts/` and are
ignored by Git; after the AWS storage gate they are copied to the versioned build
bucket.  Datasets, embedded asset sources, models, secrets and captures never enter
this directory or the Git history.

The original run did not capture CPU, RAM and block-I/O counters.  Those fields are
explicitly null in `metrics.json` and must be populated by the first replay using
the provider-neutral container monitor.  They are not inferred.

The direct AWS replay passed G2/G3 on 2026-08-21. Its compact, versioned receipt is
`aws-g2-validation.json`; full build artifacts remain in the versioned S3 build
prefix. The receipt records semantic parity, capacity measurements, atomic
publication and the post-run zero-compute check without committing the GLB, USD,
Blend, source datasets or local caches.

Semantic parity requires the same CRS, bounds, tile count, object-family counts,
zero placeholders, complete viewer representation and valid provenance.  Binary
identity is informative but not mandatory for Blender/glTF files containing
variable metadata.
