"""Deterministic Blender 4.5 visual QA for a complete adaptive-terrain zone.

The module deliberately separates three states:

* a hash-locked render job;
* a technical receipt whose only successful status is
  ``rendered_pending_zone_visual_review``;
* a human acceptance receipt, emitted only from an explicit exhaustive review.

Every primary-camera terrain surface is imported from the package LOD0 USD
variant.  LOD1/LOD2 are never imported for a capture.  The Blender reference
shader is the same four-profile compositor used by
``validate_adaptive_terrain_usd.py``: basecolor, normal, height/bump and ORM are
derived from the shared runtime atlas plus the per-tile 1 m IDs/weights,
confidence and orientation maps.  Confidence is evidence only; orientation
rotates the metric EPSG:2154 sampling axes.  No orthophoto or procedural
material enters the runtime package.

Run the render phase inside Blender 4.5 LTS, for example::

    blender.exe --background --factory-startup --disable-autoexec --offline-mode \
      --python-exit-code 1 --python validate_adaptive_terrain_zone.py -- \
      render --job D:\\...\\zone-visual-job.v1.json

The caller must redirect TEMP, Blender and Python caches to D:.  The module
also rejects every production or output path outside D: on Windows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from typing import Any, Iterable, Mapping, Sequence
import zlib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BLENDER_MODULE_ROOT = Path(__file__).resolve().parent
for _module_root in (REPOSITORY_ROOT, BLENDER_MODULE_ROOT):
    if str(_module_root) not in sys.path:
        sys.path.insert(0, str(_module_root))

try:
    from frustum_streaming import TerrainTileCatalog
    from ground_material_contract import RUNTIME_TEXTURE_ROLES, resolve_zone_asset
    from validate_adaptive_terrain_usd import (
        COVERAGE_AOV,
        TERRAIN_AOV,
        _create_float_image,
        _load_image_pixels,
        _operator_arguments,
        compose_reference_pbr_maps,
        inspect_package,
        validate_primary_terrain_aovs,
    )
except ModuleNotFoundError:  # pragma: no cover - package-style import
    from blender.frustum_streaming import TerrainTileCatalog
    from blender.ground_material_contract import (
        RUNTIME_TEXTURE_ROLES,
        resolve_zone_asset,
    )
    from blender.validate_adaptive_terrain_usd import (
        COVERAGE_AOV,
        TERRAIN_AOV,
        _create_float_image,
        _load_image_pixels,
        _operator_arguments,
        compose_reference_pbr_maps,
        inspect_package,
        validate_primary_terrain_aovs,
    )


JOB_SCHEMA = "fireviewer.zone-visual-job.v1"
CAPTURE_PLAN_SCHEMA = "fireviewer.zone-visual-capture-plan.v1"
CAPTURE_RECEIPT_SCHEMA = "fireviewer.zone-visual-capture-receipt.v1"
TECHNICAL_RECEIPT_SCHEMA = "fireviewer.zone-visual-technical-receipt.v1"
HUMAN_REVIEW_SCHEMA = "fireviewer.zone-visual-human-review.v1"
ACCEPTANCE_SCHEMA = "fireviewer.zone-blender-visual-acceptance.v2"
CATALOG_SCHEMA = "fireviewer.terrain-tile-catalog.v1"
TECHNICAL_STATUS = "rendered_pending_zone_visual_review"
ACCEPTED_STATUS = "accepted_blender_visual"
TILE_SIZE_M = 500.0
MINIMUM_RESERVE_RATIO = 0.25
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
NETWORK_ROLES = frozenset({"path", "road", "rail", "hydro"})
CAPTURE_ARTIFACT_NAMES = {
    "beauty": "beauty.png",
    "terrain_lod": "terrain-lod.exr",
    "terrain_coverage": "terrain-coverage.exr",
}
LEGACY_TILE_PACKAGE_SCHEMA = "fireviewer.tile-package.v2"
LEGACY_TILE_DONE_SCHEMA = "fireviewer.tile.done.v2"
TILE_PACKAGE_SCHEMA = "fireviewer.tile-package.v3"
TILE_DONE_SCHEMA = "fireviewer.tile.done.v3"
LEGACY_TILE_PACKAGE_FILE_NAME = "tile-package.v2.json"
LEGACY_TILE_DONE_FILE_NAME = "tile.done.v2.json"
TILE_PACKAGE_FILE_NAME = "tile-package.v3.json"
TILE_DONE_FILE_NAME = "tile.done.v3.json"
LEGACY_CANONICAL_TILE_OUTPUTS = {
    "terrain_lod0": "terrain-lod0.fvtq",
    "terrain_lod1": "terrain-lod1.fvtq",
    "terrain_lod2": "terrain-lod2.fvtq",
    "hag_max_2m": "hag-max-2m.tif",
    "ground_profile_ids": "ground-profile-ids.png",
    "ground_profile_weights": "ground-profile-weights.png",
    "surface_overlays": "surface-overlays.json.gz",
    "tile_composition": "tile-composition.json.gz",
}
CANONICAL_TILE_OUTPUTS = {
    "terrain_lod0": "terrain-lod0.fvtq",
    "terrain_lod1": "terrain-lod1.fvtq",
    "terrain_lod2": "terrain-lod2.fvtq",
    "hag_max_2m": "hag-max-2m.tif",
    "ground_profile_ids": "ground-profile-ids.png",
    "ground_profile_weights": "ground-profile-weights.png",
    "ground_confidence": "ground-confidence.png",
    "ground_orientation": "ground-orientation.png",
}
V3_SURFACE_MAP_CONTRACT = {
    "profile_ids": {
        "file": "ground-profile-ids.png",
        "mode": "RGBA8",
        "encoding": "four_zero_based_stable_profile_indices",
    },
    "profile_weights": {
        "file": "ground-profile-weights.png",
        "mode": "RGBA8",
        "encoding": "four_profile_weights_sum_exactly_255_per_pixel",
    },
    "confidence": {
        "file": "ground-confidence.png",
        "mode": "L8",
        "encoding": "best_vs_next_semantic_class_margin_0_to_255",
    },
    "orientation": {
        "file": "ground-orientation.png",
        "mode": "L8",
        "encoding": "undirected_angle_0_to_pi_mapped_to_uint8",
    },
}


class ZoneVisualQaError(RuntimeError):
    """A zone visual proof is incomplete, mutable, over budget or incoherent."""


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json_bytes(payload)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ZoneVisualQaError(f"Invalid {label}: {error}") from error
    if not isinstance(payload, dict):
        raise ZoneVisualQaError(f"{label} must be a JSON object")
    return payload


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise ZoneVisualQaError(f"{label} must be a lowercase SHA-256")
    return value


def _require_d_path(path: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if os.name == "nt" and resolved.drive.upper() != "D:":
        raise ZoneVisualQaError(f"{label} must remain on D:, got {resolved}")
    return resolved


def _relative_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ZoneVisualQaError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ZoneVisualQaError(f"{label} escapes the zone package")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:  # pragma: no cover - defensive on odd filesystems
        raise ZoneVisualQaError(f"{label} escapes the zone package") from error
    return resolved


def _artifact(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ZoneVisualQaError(f"Missing proof artifact: {path}")
    rendered_path = (
        path.resolve().relative_to(relative_to.resolve()).as_posix()
        if relative_to is not None
        else str(path.resolve())
    )
    return {
        "path": rendered_path,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_artifact(
    root: Path, record: Any, label: str, *, exact_path: str | None = None
) -> Path:
    if not isinstance(record, Mapping):
        raise ZoneVisualQaError(f"{label} artifact record is missing")
    value = record.get("path")
    if exact_path is not None and value != exact_path:
        raise ZoneVisualQaError(f"{label} path differs from the capture contract")
    path = _relative_path(root, value, label)
    if not path.is_file():
        raise ZoneVisualQaError(f"{label} artifact is absent: {path}")
    if record.get("bytes") != path.stat().st_size or record.get("sha256") != _sha256(
        path
    ):
        raise ZoneVisualQaError(f"{label} artifact hash mismatch")
    return path


def _validate_tile_record(
    root: Path, record: Any, label: str, *, expected_path: str | None = None
) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "byte_count",
        "sha256",
    }:
        raise ZoneVisualQaError(f"{label} must contain path, byte_count and sha256")
    if expected_path is not None and record.get("path") != expected_path:
        raise ZoneVisualQaError(f"{label} path differs from the canonical contract")
    path = _relative_path(root, record.get("path"), label)
    byte_count = record.get("byte_count")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
    ):
        raise ZoneVisualQaError(f"{label}.byte_count must be non-negative")
    _require_hash(record.get("sha256"), f"{label}.sha256")
    return path


def _validate_tile_done_without_optional_dependencies(
    tile_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rehash tile.done/package in Blender without importing Pillow or NumPy."""

    root = Path(tile_root).resolve()
    package_path = root / TILE_PACKAGE_FILE_NAME
    done_path = root / TILE_DONE_FILE_NAME
    legacy_v2 = False
    if not package_path.is_file():
        package_path = root / LEGACY_TILE_PACKAGE_FILE_NAME
        done_path = root / LEGACY_TILE_DONE_FILE_NAME
        legacy_v2 = True
    package = _read_json(package_path, "tile package manifest")
    done = _read_json(done_path, "tile completion receipt")
    expected_package_schema = (
        LEGACY_TILE_PACKAGE_SCHEMA if legacy_v2 else TILE_PACKAGE_SCHEMA
    )
    expected_done_schema = LEGACY_TILE_DONE_SCHEMA if legacy_v2 else TILE_DONE_SCHEMA
    canonical_outputs = (
        LEGACY_CANONICAL_TILE_OUTPUTS if legacy_v2 else CANONICAL_TILE_OUTPUTS
    )
    if package.get("schema") != expected_package_schema:
        raise ZoneVisualQaError("Unsupported tile package manifest")
    if done.get("schema") != expected_done_schema:
        raise ZoneVisualQaError("Unsupported tile completion receipt")
    for identity in ("tile_id", "recipe_id", "recipe_build_id"):
        if identity != "tile_id":
            _require_hash(package.get(identity), f"tile package {identity}")
        if done.get(identity) != package.get(identity):
            raise ZoneVisualQaError(
                f"Tile completion {identity} differs from its package"
            )
    identity_fields = [
        "normal_halo_sha256",
        "stitch_variants",
        "inputs",
        "ground_material",
    ]
    if not legacy_v2:
        identity_fields.append("surface_mapping")
        for forbidden in ("surface-overlays.json.gz", "tile-composition.json.gz"):
            if (root / forbidden).exists():
                raise ZoneVisualQaError(
                    f"Mapping v3 contains a forbidden runtime artifact: {forbidden}"
                )
        mapping = package.get("surface_mapping")
        if (
            not isinstance(mapping, Mapping)
            or mapping.get("schema") != "fireviewer.ground-surface-mapping.v3"
            or mapping.get("grid_size_px") != [500, 500]
            or mapping.get("cell_size_m") != 1
            or mapping.get("runtime_procedural_material") != "forbidden"
            or mapping.get("runtime_orthophoto") != "forbidden"
            or any(
                mapping.get(name) != expected
                for name, expected in V3_SURFACE_MAP_CONTRACT.items()
            )
        ):
            raise ZoneVisualQaError("Tile surface mapping v3 is invalid")
    for identity in identity_fields:
        if done.get(identity) != package.get(identity):
            raise ZoneVisualQaError(
                f"Tile completion {identity} differs from its package"
            )
    for collection in ("inputs", "outputs"):
        records = package.get(collection)
        if not isinstance(records, Mapping) or not records:
            raise ZoneVisualQaError(f"Tile package {collection} are missing")
        if collection == "outputs" and set(records) != set(canonical_outputs):
            raise ZoneVisualQaError("Canonical tile output set is incomplete")
        for name, record in records.items():
            expected_path = canonical_outputs[name] if collection == "outputs" else None
            artifact_path = _validate_tile_record(
                root, record, f"{collection}.{name}", expected_path=expected_path
            )
            if collection == "outputs":
                if not artifact_path.is_file():
                    raise ZoneVisualQaError(f"Canonical tile output is absent: {name}")
                if record["byte_count"] != artifact_path.stat().st_size or record[
                    "sha256"
                ] != _sha256(artifact_path):
                    raise ZoneVisualQaError(
                        f"Canonical tile output hash mismatch: {name}"
                    )
    package_record = {
        "path": package_path.name,
        "byte_count": package_path.stat().st_size,
        "sha256": _sha256(package_path),
    }
    expected_done_outputs = {**package["outputs"], "tile_package": package_record}
    if done.get("outputs") != expected_done_outputs:
        raise ZoneVisualQaError("Tile completion output signatures differ")
    return done, package


@dataclass(frozen=True)
class ZoneTile:
    tile_id: str
    grid_x: int
    grid_y: int
    bounds: tuple[float, float, float, float, float, float]
    tile_root: Path
    package: Mapping[str, Any]
    done: Mapping[str, Any]
    lod0_cpu_bytes: int
    lod0_gpu_bytes: int
    lod0_triangles: int
    lod0_sha256: str

    @property
    def horizontal_bounds(self) -> tuple[float, float, float, float]:
        return (self.bounds[0], self.bounds[1], self.bounds[3], self.bounds[4])

    @property
    def height_range_m(self) -> float:
        return self.bounds[5] - self.bounds[2]


@dataclass(frozen=True)
class ZoneInspection:
    job_path: Path
    job: Mapping[str, Any]
    zone_root: Path
    output_root: Path
    catalog_path: Path
    catalog_sha256: str
    qa_metrics_path: Path
    qa_metrics_sha256: str
    qa_result: Mapping[str, Any]
    tiles: tuple[ZoneTile, ...]
    seam_metrics: tuple[Mapping[str, Any], ...]
    material_contract_sha256: str

    @property
    def tiles_by_id(self) -> dict[str, ZoneTile]:
        return {tile.tile_id: tile for tile in self.tiles}

    @property
    def zone_bounds(self) -> tuple[float, float, float, float, float, float]:
        return (
            min(tile.bounds[0] for tile in self.tiles),
            min(tile.bounds[1] for tile in self.tiles),
            min(tile.bounds[2] for tile in self.tiles),
            max(tile.bounds[3] for tile in self.tiles),
            max(tile.bounds[4] for tile in self.tiles),
            max(tile.bounds[5] for tile in self.tiles),
        )


def build_zone_visual_job(
    *,
    zone_id: str,
    revision: str,
    recipe_id: str,
    recipe_build_id: str,
    build_id: str,
    zone_root: Path,
    catalog_path: Path,
    qa_metrics_path: Path,
    output_root: Path,
    resolution: int,
    maximum_seams: int,
    cpu_budget_bytes: int,
    gpu_budget_bytes: int,
    triangle_budget: int,
    reserve_ratio: float = MINIMUM_RESERVE_RATIO,
) -> dict[str, Any]:
    """Build the exact backend-facing job contract without rendering it."""

    resolved_zone = Path(zone_root).resolve()
    resolved_catalog = Path(catalog_path).resolve()
    try:
        catalog_relative = resolved_catalog.relative_to(resolved_zone).as_posix()
    except ValueError as error:
        raise ZoneVisualQaError("catalog_path must remain inside zone_root") from error
    for value, label in ((zone_id, "zone_id"), (revision, "revision")):
        if not isinstance(value, str) or not value.strip():
            raise ZoneVisualQaError(f"{label} must be a non-empty string")
    _require_hash(recipe_id, "recipe_id")
    _require_hash(recipe_build_id, "recipe_build_id")
    _require_hash(build_id, "build_id")
    if not resolved_catalog.is_file() or not Path(qa_metrics_path).resolve().is_file():
        raise ZoneVisualQaError("catalog and qa_metrics must exist before job creation")
    if isinstance(resolution, bool) or not 512 <= resolution <= 2048:
        raise ZoneVisualQaError("resolution must be between 512 and 2048")
    if isinstance(maximum_seams, bool) or not 0 <= maximum_seams <= 20:
        raise ZoneVisualQaError("maximum_seams must be between 0 and 20")
    for value, label in (
        (cpu_budget_bytes, "cpu_budget_bytes"),
        (gpu_budget_bytes, "gpu_budget_bytes"),
        (triangle_budget, "triangle_budget"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ZoneVisualQaError(f"{label} must be a positive integer")
    if (
        not math.isfinite(float(reserve_ratio))
        or not MINIMUM_RESERVE_RATIO <= float(reserve_ratio) < 1.0
    ):
        raise ZoneVisualQaError("reserve_ratio must be at least 0.25 and below 1")
    return {
        "schema": JOB_SCHEMA,
        "zone_id": zone_id,
        "revision": revision,
        "recipe_id": recipe_id,
        "recipe_build_id": recipe_build_id,
        "build_id": build_id,
        "zone_root": str(resolved_zone),
        "catalog": {
            "path": catalog_relative,
            "bytes": resolved_catalog.stat().st_size,
            "sha256": _sha256(resolved_catalog),
        },
        "qa_metrics": _artifact(Path(qa_metrics_path).resolve()),
        "output_root": str(Path(output_root).resolve()),
        "render": {
            "resolution": resolution,
            "maximum_worst_seams": maximum_seams,
            "engine": "BLENDER_EEVEE_NEXT",
            "required_aovs": [TERRAIN_AOV, COVERAGE_AOV],
            "primary_camera_allowed_lods": [0],
        },
        "budget": {
            "cpu_bytes": cpu_budget_bytes,
            "gpu_bytes": gpu_budget_bytes,
            "triangles": triangle_budget,
            "reserve_ratio": float(reserve_ratio),
        },
        "prohibited_dependencies": {
            "network": True,
            "orthophoto": True,
            "omniverse": True,
            "lod1_or_lod2_primary_camera": True,
        },
    }


def write_zone_visual_job(path: Path, **arguments: Any) -> dict[str, Any]:
    payload = build_zone_visual_job(**arguments)
    _atomic_json(Path(path), payload)
    return payload


def _extract_qa_result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = payload.get("result", payload)
    if not isinstance(candidate, Mapping) or candidate.get("status") != "passed":
        raise ZoneVisualQaError("Zone QA metrics are not in passed state")
    return candidate


def _tile_resource_cost(raw_tile: Mapping[str, Any]) -> tuple[int, int, int, str]:
    resource_cost = raw_tile.get("resource_costs", {}).get("lod0")
    if not isinstance(resource_cost, Mapping):
        raise ZoneVisualQaError(f"LOD0 resource cost is missing: {raw_tile.get('id')}")
    values = []
    for key in ("cpu_bytes", "gpu_bytes", "triangles"):
        value = resource_cost.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ZoneVisualQaError(f"Invalid LOD0 {key}: {raw_tile.get('id')}")
        values.append(value)
    return (*values, _require_hash(resource_cost.get("sha256"), "LOD0 payload hash"))


def _catalog_seams(tiles: Sequence[ZoneTile]) -> dict[str, tuple[str, str, str]]:
    by_grid = {(tile.grid_x, tile.grid_y): tile for tile in tiles}
    seams: dict[str, tuple[str, str, str]] = {}
    for tile in sorted(tiles, key=lambda item: (item.grid_y, item.grid_x)):
        for direction, offset in (("vertical", (1, 0)), ("horizontal", (0, 1))):
            neighbour = by_grid.get((tile.grid_x + offset[0], tile.grid_y + offset[1]))
            if neighbour is None:
                continue
            seam_id = f"{tile.tile_id}--{neighbour.tile_id}"
            seams[seam_id] = (tile.tile_id, neighbour.tile_id, direction)
    return seams


def _validate_seam_metrics(
    tiles: Sequence[ZoneTile], qa_result: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    expected = _catalog_seams(tiles)
    raw_metrics = qa_result.get("seam_metrics")
    if not isinstance(raw_metrics, list):
        raise ZoneVisualQaError("Zone QA seam_metrics are missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_metrics:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str):
            raise ZoneVisualQaError("Every seam metric requires a string id")
        seam_id = raw["id"]
        if seam_id in by_id:
            raise ZoneVisualQaError(f"Duplicate seam metric: {seam_id}")
        if seam_id not in expected:
            raise ZoneVisualQaError(f"Unknown or mono-tile seam metric: {seam_id}")
        first, second, _direction = expected[seam_id]
        if first == second:
            raise ZoneVisualQaError(f"Seam must bind two distinct tiles: {seam_id}")
        for key in (
            "maximum_height_gap_mm",
            "normal_mismatch_count",
            "stitch_signature_mismatch_count",
            "composition_failure_count",
        ):
            value = raw.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ZoneVisualQaError(f"Invalid {key} for seam {seam_id}")
        by_id[seam_id] = raw
    if set(by_id) != set(expected):
        missing = sorted(set(expected) - set(by_id))
        raise ZoneVisualQaError(f"Zone QA did not measure every seam: {missing[:3]}")
    metrics = qa_result.get("metrics")
    if not isinstance(metrics, Mapping) or metrics.get("seam_count") != len(expected):
        raise ZoneVisualQaError("Zone QA seam_count differs from the catalog grid")
    return tuple(by_id[seam_id] for seam_id in sorted(by_id))


def inspect_zone_job(
    job_path: Path, *, require_d: bool = True, validate_packages: bool = True
) -> ZoneInspection:
    """Rehash the complete job and validate every LOD0 package before Blender."""

    resolved_job = Path(job_path).resolve()
    if require_d:
        resolved_job = _require_d_path(resolved_job, "job")
    job = _read_json(resolved_job, "zone visual job")
    if job.get("schema") != JOB_SCHEMA:
        raise ZoneVisualQaError("Unsupported zone visual job")
    for key in ("zone_id", "revision"):
        if not isinstance(job.get(key), str) or not job[key]:
            raise ZoneVisualQaError(f"Job {key} is missing")
    recipe_id = _require_hash(job.get("recipe_id"), "job.recipe_id")
    recipe_build_id = _require_hash(job.get("recipe_build_id"), "job.recipe_build_id")
    build_id = _require_hash(job.get("build_id"), "job.build_id")
    zone_root = Path(str(job.get("zone_root", ""))).resolve()
    output_root = Path(str(job.get("output_root", ""))).resolve()
    if require_d:
        zone_root = _require_d_path(zone_root, "zone_root")
        output_root = _require_d_path(output_root, "output_root")
    if not zone_root.is_dir():
        raise ZoneVisualQaError(f"Zone package is absent: {zone_root}")
    if output_root == zone_root or zone_root in output_root.parents:
        raise ZoneVisualQaError("Visual output must not overwrite the canonical zone")

    catalog_record = job.get("catalog")
    if not isinstance(catalog_record, Mapping):
        raise ZoneVisualQaError("Job catalog artifact is missing")
    catalog_path = _relative_path(zone_root, catalog_record.get("path"), "catalog")
    if not catalog_path.is_file():
        raise ZoneVisualQaError(f"Terrain catalog is absent: {catalog_path}")
    if catalog_record.get("bytes") != catalog_path.stat().st_size or catalog_record.get(
        "sha256"
    ) != _sha256(catalog_path):
        raise ZoneVisualQaError("Terrain catalog was modified after job creation")
    catalog = _read_json(catalog_path, "terrain tile catalog")
    if catalog.get("schema") != CATALOG_SCHEMA:
        raise ZoneVisualQaError("Unsupported terrain tile catalog")
    try:
        TerrainTileCatalog.from_manifest(catalog)
    except (KeyError, TypeError, ValueError) as error:
        raise ZoneVisualQaError(f"Invalid terrain tile catalog: {error}") from error

    qa_record = job.get("qa_metrics")
    if not isinstance(qa_record, Mapping) or not isinstance(qa_record.get("path"), str):
        raise ZoneVisualQaError("Job QA metrics artifact is missing")
    qa_metrics_path = Path(qa_record["path"]).resolve()
    if require_d:
        qa_metrics_path = _require_d_path(qa_metrics_path, "qa_metrics")
    if not qa_metrics_path.is_file():
        raise ZoneVisualQaError(f"QA metrics are absent: {qa_metrics_path}")
    if qa_record.get("bytes") != qa_metrics_path.stat().st_size or qa_record.get(
        "sha256"
    ) != _sha256(qa_metrics_path):
        raise ZoneVisualQaError("QA metrics were modified after job creation")
    qa_payload = _read_json(qa_metrics_path, "zone QA metrics")
    qa_result = _extract_qa_result(qa_payload)
    for identity, expected in (
        ("recipe_id", recipe_id),
        ("recipe_build_id", recipe_build_id),
        ("build_id", build_id),
    ):
        declared = qa_result.get(identity, qa_payload.get(identity))
        if declared is not None and declared != expected:
            raise ZoneVisualQaError(f"QA {identity} differs from the visual job")

    raw_tiles = catalog.get("tiles")
    if not isinstance(raw_tiles, list) or not raw_tiles:
        raise ZoneVisualQaError("Terrain catalog has no tiles")
    tiles: list[ZoneTile] = []
    material_hashes: set[str] = set()
    for raw in sorted(raw_tiles, key=lambda item: str(item.get("id"))):
        if not isinstance(raw, Mapping):
            raise ZoneVisualQaError("Terrain catalog tile is not an object")
        tile_id = raw.get("id")
        if (
            not isinstance(tile_id, str)
            or not tile_id
            or any(character in tile_id for character in ("/", "\\", "\0"))
        ):
            raise ZoneVisualQaError("Terrain catalog tile id is not portable")
        if raw.get("build_id") != build_id:
            raise ZoneVisualQaError(f"Catalog build_id mismatch: {tile_id}")
        bounds = raw.get("bounds_l93_ngf_m")
        if (
            not isinstance(bounds, list)
            or len(bounds) != 6
            or not all(isinstance(value, (int, float)) for value in bounds)
            or not all(math.isfinite(float(value)) for value in bounds)
            or float(bounds[3]) - float(bounds[0]) != TILE_SIZE_M
            or float(bounds[4]) - float(bounds[1]) != TILE_SIZE_M
            or float(bounds[5]) < float(bounds[2])
        ):
            raise ZoneVisualQaError(f"Invalid tile bounds: {tile_id}")
        grid_x = raw.get("grid_x")
        grid_y = raw.get("grid_y")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (grid_x, grid_y)
        ):
            raise ZoneVisualQaError(f"Invalid tile grid coordinates: {tile_id}")
        cpu_bytes, gpu_bytes, triangles, payload_hash = _tile_resource_cost(raw)
        tile_root = zone_root / "tiles" / tile_id
        if not tile_root.is_dir():
            raise ZoneVisualQaError(f"Tile package is absent: {tile_id}")
        package: Mapping[str, Any] = {}
        done: Mapping[str, Any] = {}
        if validate_packages:
            try:
                done, tile_package = _validate_tile_done_without_optional_dependencies(
                    tile_root
                )
                inspected = inspect_package(tile_root)
            except (OSError, RuntimeError, ValueError) as error:
                raise ZoneVisualQaError(
                    f"Invalid tile package {tile_id}: {error}"
                ) from error
            package = inspected["manifest"]
            if (
                done.get("recipe_id") != recipe_id
                or done.get("recipe_build_id") != recipe_build_id
                or tile_package.get("recipe_id") != recipe_id
                or tile_package.get("recipe_build_id") != recipe_build_id
                or package.get("recipe_id") != recipe_id
                or package.get("recipe_build_id") != recipe_build_id
                or package.get("tile_id") != tile_id
                or inspected.get("selected_lod") != 0
                or package.get("primary_camera_allowed_lods") != [0]
            ):
                raise ZoneVisualQaError(
                    f"Tile identity/LOD contract mismatch: {tile_id}"
                )
            expected_origin = [float(bounds[0]), float(bounds[1])]
            if [
                float(value) for value in package.get("tile_origin_l93_m", [])
            ] != expected_origin:
                raise ZoneVisualQaError(f"Tile origin differs from catalog: {tile_id}")
            lod0_path = tile_root / "terrain-lod0.fvtq"
            if _sha256(lod0_path) != payload_hash:
                raise ZoneVisualQaError(f"Catalog LOD0 hash mismatch: {tile_id}")
            metrics = package.get("lod_metrics", {}).get("lod0", {})
            if metrics.get("triangle_count") != triangles:
                raise ZoneVisualQaError(
                    f"Catalog LOD0 triangle count mismatch: {tile_id}"
                )
            material_hash = (
                package.get("ground_material", {}).get("contract", {}).get("sha256")
            )
            material_hashes.add(_require_hash(material_hash, "ground material hash"))
        tiles.append(
            ZoneTile(
                tile_id=tile_id,
                grid_x=grid_x,
                grid_y=grid_y,
                bounds=tuple(float(value) for value in bounds),  # type: ignore[arg-type]
                tile_root=tile_root,
                package=package,
                done=done,
                lod0_cpu_bytes=cpu_bytes,
                lod0_gpu_bytes=gpu_bytes,
                lod0_triangles=triangles,
                lod0_sha256=payload_hash,
            )
        )
    if validate_packages and len(material_hashes) != 1:
        raise ZoneVisualQaError("Zone tiles do not share one ground material contract")
    validated_ids = qa_result.get("validated_tile_ids")
    expected_ids = sorted(tile.tile_id for tile in tiles)
    if not isinstance(validated_ids, list) or sorted(validated_ids) != expected_ids:
        raise ZoneVisualQaError(
            "Zone QA did not validate every catalog tile exactly once"
        )
    seam_metrics = _validate_seam_metrics(tiles, qa_result)
    inspection = ZoneInspection(
        job_path=resolved_job,
        job=job,
        zone_root=zone_root,
        output_root=output_root,
        catalog_path=catalog_path,
        catalog_sha256=_sha256(catalog_path),
        qa_metrics_path=qa_metrics_path,
        qa_metrics_sha256=_sha256(qa_metrics_path),
        qa_result=qa_result,
        tiles=tuple(tiles),
        seam_metrics=seam_metrics,
        material_contract_sha256=(
            next(iter(material_hashes)) if material_hashes else "0" * 64
        ),
    )
    validate_zone_budget(inspection)
    return inspection


def validate_zone_budget(inspection: ZoneInspection) -> dict[str, Any]:
    """Fail before Blender import if full-zone LOD0 plus reserve cannot fit."""

    budget = inspection.job.get("budget")
    if not isinstance(budget, Mapping):
        raise ZoneVisualQaError("Visual job has no explicit runtime budget")
    reserve = budget.get("reserve_ratio")
    if (
        not isinstance(reserve, (int, float))
        or isinstance(reserve, bool)
        or not (MINIMUM_RESERVE_RATIO <= float(reserve) < 1.0)
    ):
        raise ZoneVisualQaError("Runtime budget reserve must be at least 25 percent")
    measured = {
        "cpu_bytes": sum(tile.lod0_cpu_bytes for tile in inspection.tiles),
        "gpu_bytes": sum(tile.lod0_gpu_bytes for tile in inspection.tiles),
        "triangles": sum(tile.lod0_triangles for tile in inspection.tiles),
    }
    limits: dict[str, int] = {}
    for key in measured:
        raw = budget.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ZoneVisualQaError(f"Runtime budget {key} must be positive")
        limits[key] = math.floor(raw * (1.0 - float(reserve)))
        if measured[key] > limits[key]:
            raise ZoneVisualQaError(
                f"Full-zone LOD0 {key} exceeds reserved budget: "
                f"{measured[key]} > {limits[key]}"
            )
    return {
        "measured_lod0": measured,
        "declared": {key: int(budget[key]) for key in measured},
        "usable_after_reserve": limits,
        "reserve_ratio": float(reserve),
        "status": "within_budget",
    }


def _paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    diagonal_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= diagonal_distance:
        return left
    if above_distance <= diagonal_distance:
        return above
    return upper_left


def _read_rgba8_png(path: Path) -> tuple[int, int, bytes]:
    """Read the deterministic non-interlaced RGBA8 maps without Pillow."""

    payload = path.read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ZoneVisualQaError(f"Composition map is not PNG: {path}")
    offset = 8
    width = height = 0
    compressed = bytearray()
    seen_end = False
    while offset + 12 <= len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(payload):
            raise ZoneVisualQaError(f"Truncated PNG chunk: {path}")
        content = payload[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", payload[offset + 8 + length : end])[0]
        actual_crc = zlib.crc32(chunk_type + content) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ZoneVisualQaError(f"PNG chunk CRC mismatch: {path}")
        if chunk_type == b"IHDR":
            if len(content) != 13:
                raise ZoneVisualQaError(f"Invalid PNG header: {path}")
            width, height, bit_depth, colour_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", content)
            )
            if (
                width <= 0
                or height <= 0
                or bit_depth != 8
                or colour_type != 6
                or compression != 0
                or filtering != 0
                or interlace != 0
            ):
                raise ZoneVisualQaError(
                    f"Composition PNG must be non-interlaced RGBA8: {path}"
                )
        elif chunk_type == b"IDAT":
            compressed.extend(content)
        elif chunk_type == b"IEND":
            seen_end = True
            break
        offset = end
    if not seen_end or not width or not compressed:
        raise ZoneVisualQaError(f"Incomplete composition PNG: {path}")
    try:
        filtered = zlib.decompress(bytes(compressed))
    except zlib.error as error:
        raise ZoneVisualQaError(f"Invalid compressed PNG data: {path}") from error
    stride = width * 4
    expected_length = height * (stride + 1)
    if len(filtered) != expected_length:
        raise ZoneVisualQaError(f"Unexpected composition PNG byte count: {path}")
    output = bytearray(height * stride)
    for row in range(height):
        source_offset = row * (stride + 1)
        filter_type = filtered[source_offset]
        if filter_type not in {0, 1, 2, 3, 4}:
            raise ZoneVisualQaError(f"Unsupported PNG row filter: {path}")
        row_bytes = filtered[source_offset + 1 : source_offset + stride + 1]
        target_offset = row * stride
        for column, raw_value in enumerate(row_bytes):
            left = output[target_offset + column - 4] if column >= 4 else 0
            above = output[target_offset - stride + column] if row else 0
            upper_left = (
                output[target_offset - stride + column - 4]
                if row and column >= 4
                else 0
            )
            if filter_type == 0:
                value = raw_value
            elif filter_type == 1:
                value = raw_value + left
            elif filter_type == 2:
                value = raw_value + above
            elif filter_type == 3:
                value = raw_value + ((left + above) // 2)
            else:
                value = raw_value + _paeth(left, above, upper_left)
            output[target_offset + column] = value & 0xFF
    return width, height, bytes(output)


def _network_score(tile: ZoneTile) -> tuple[float, int]:
    path = tile.tile_root / "surface-overlays.json.gz"
    if not path.is_file():
        # Mapping v3 bakes narrow-surface choices into its profile maps.  The
        # visual plan remains exhaustive but never recreates a runtime vector
        # dependency merely to rank captures.
        return 0.0, 0
    try:
        payload = json.loads(gzip.decompress(path.read_bytes()))
    except (OSError, json.JSONDecodeError) as error:
        raise ZoneVisualQaError(
            f"Invalid surface overlays for {tile.tile_id}"
        ) from error
    features = payload.get("features")
    if not isinstance(features, list):
        raise ZoneVisualQaError(f"Surface overlay list is missing: {tile.tile_id}")

    def line_length(coordinates: Any) -> float:
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            return 0.0
        total = 0.0
        for first, second in zip(coordinates, coordinates[1:]):
            if not (
                isinstance(first, list)
                and isinstance(second, list)
                and len(first) >= 2
                and len(second) >= 2
            ):
                return 0.0
            total += math.hypot(
                float(second[0]) - float(first[0]), float(second[1]) - float(first[1])
            )
        return total

    total_length = 0.0
    count = 0
    for feature in features:
        if not isinstance(feature, Mapping) or feature.get("role") not in NETWORK_ROLES:
            continue
        geometry = feature.get("geometry_l93_m")
        if not isinstance(geometry, Mapping):
            continue
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates")
        if geometry_type == "LineString":
            length = line_length(coordinates)
        elif geometry_type == "MultiLineString" and isinstance(coordinates, list):
            length = sum(line_length(line) for line in coordinates)
        else:
            length = 0.0
        total_length += length
        count += 1
    return round(total_length, 3), count


def _surface_score(tile: ZoneTile) -> tuple[int, int, float]:
    ids_width, ids_height, identifiers = _read_rgba8_png(
        tile.tile_root / "ground-profile-ids.png"
    )
    weights_width, weights_height, weights = _read_rgba8_png(
        tile.tile_root / "ground-profile-weights.png"
    )
    expected_size = (
        500
        if tile.package.get("tile_package_schema") == "fireviewer.tile-package.v3"
        else 100
    )
    if (ids_width, ids_height) != (expected_size, expected_size) or (
        weights_width,
        weights_height,
    ) != (expected_size, expected_size):
        raise ZoneVisualQaError(
            f"Ground composition grid is not {expected_size}x{expected_size}: {tile.tile_id}"
        )
    totals: dict[int, int] = {}
    for index in range(0, len(identifiers), 4):
        cell_weights = weights[index : index + 4]
        if sum(cell_weights) != 255:
            raise ZoneVisualQaError(
                f"Ground profile weights do not sum to 255: {tile.tile_id}"
            )
        for profile, weight in zip(
            identifiers[index : index + 4], cell_weights, strict=True
        ):
            totals[profile] = totals.get(profile, 0) + weight
    if not totals:
        raise ZoneVisualQaError(
            f"Ground composition has no weighted profile: {tile.tile_id}"
        )
    profile, total = min(totals.items(), key=lambda item: (-item[1], item[0]))
    return profile, total, total / (ids_width * ids_height * 255.0)


def _union_bounds(tiles: Iterable[ZoneTile]) -> list[float]:
    selected = tuple(tiles)
    if not selected:
        raise ZoneVisualQaError("A capture cannot have an empty tile set")
    return [
        min(tile.bounds[0] for tile in selected),
        min(tile.bounds[1] for tile in selected),
        min(tile.bounds[2] for tile in selected),
        max(tile.bounds[3] for tile in selected),
        max(tile.bounds[4] for tile in selected),
        max(tile.bounds[5] for tile in selected),
    ]


def _capture(
    capture_id: str,
    category: str,
    projection: str,
    tiles: Iterable[ZoneTile],
    *,
    frame_bounds: Sequence[float] | None = None,
    focus_tile_id: str | None = None,
    selection_basis: Mapping[str, Any] | None = None,
    azimuth_deg: float | None = None,
) -> dict[str, Any]:
    selected = tuple(sorted(tiles, key=lambda item: item.tile_id))
    bounds = list(frame_bounds) if frame_bounds is not None else _union_bounds(selected)
    if len(bounds) != 6:
        raise ZoneVisualQaError(f"Capture {capture_id} bounds must contain six values")
    payload: dict[str, Any] = {
        "capture_id": capture_id,
        "category": category,
        "projection": projection,
        "tile_ids": [tile.tile_id for tile in selected],
        "frame_bounds_l93_ngf_m": [float(value) for value in bounds],
        "focus_tile_id": focus_tile_id,
        "selection_basis": dict(selection_basis or {}),
    }
    if projection == "perspective_oblique":
        payload["camera"] = {
            "lens_mm": 46.0,
            "azimuth_deg": float(azimuth_deg if azimuth_deg is not None else 35.0),
            "elevation_deg": 38.0,
            "clip_start_m": 1.0,
            "clip_end_m": 50_000.0,
        }
    else:
        payload["camera"] = {
            "type": "orthographic",
            "margin_ratio": 0.04,
            "clip_start_m": 0.1,
            "clip_end_m": 50_000.0,
        }
    return payload


def build_capture_plan(inspection: ZoneInspection) -> dict[str, Any]:
    """Select the mandatory full, 3x3, worst-seam and oblique evidence set."""

    tiles = inspection.tiles
    by_id = inspection.tiles_by_id
    by_grid = {(tile.grid_x, tile.grid_y): tile for tile in tiles}
    zone_bounds = inspection.zone_bounds
    west, south, minimum_z, east, north, maximum_z = zone_bounds
    captures: list[dict[str, Any]] = [
        _capture(
            "overview-full-square",
            "orthographic_full_square",
            "orthographic",
            tiles,
            frame_bounds=zone_bounds,
            selection_basis={"selection": "all_catalog_lod0_tiles"},
        )
    ]

    # Nine deterministic cells are emitted even for a small synthetic grid.
    # A tile is assigned through its centre; an empty cell receives its nearest
    # tile so a requested QA capture can never silently disappear.
    for row in range(3):
        for column in range(3):
            cell_west = west + (east - west) * column / 3.0
            cell_east = west + (east - west) * (column + 1) / 3.0
            cell_south = south + (north - south) * row / 3.0
            cell_north = south + (north - south) * (row + 1) / 3.0
            selected = [
                tile
                for tile in tiles
                if cell_west
                <= (tile.bounds[0] + tile.bounds[3]) * 0.5
                < (cell_east if column < 2 else cell_east + 1.0e-9)
                and cell_south
                <= (tile.bounds[1] + tile.bounds[4]) * 0.5
                < (cell_north if row < 2 else cell_north + 1.0e-9)
            ]
            if not selected:
                centre = (
                    (cell_west + cell_east) * 0.5,
                    (cell_south + cell_north) * 0.5,
                )
                selected = [
                    min(
                        tiles,
                        key=lambda tile: (
                            math.hypot(
                                (tile.bounds[0] + tile.bounds[3]) * 0.5 - centre[0],
                                (tile.bounds[1] + tile.bounds[4]) * 0.5 - centre[1],
                            ),
                            tile.tile_id,
                        ),
                    )
                ]
            capture_bounds = [
                cell_west,
                cell_south,
                min(tile.bounds[2] for tile in selected),
                cell_east,
                cell_north,
                max(tile.bounds[5] for tile in selected),
            ]
            captures.append(
                _capture(
                    f"grid-r{row + 1}-c{column + 1}",
                    "orthographic_grid_3x3",
                    "orthographic",
                    selected,
                    frame_bounds=capture_bounds,
                    selection_basis={"row": row + 1, "column": column + 1},
                )
            )

    seams = _catalog_seams(tiles)
    maximum_seams = int(inspection.job["render"]["maximum_worst_seams"])
    worst = sorted(
        inspection.seam_metrics,
        key=lambda item: (
            -int(item["maximum_height_gap_mm"]),
            -int(item["normal_mismatch_count"]),
            -int(item["stitch_signature_mismatch_count"]),
            -int(item["composition_failure_count"]),
            str(item["id"]),
        ),
    )[:maximum_seams]
    for rank, metric in enumerate(worst, start=1):
        first_id, second_id, direction = seams[str(metric["id"])]
        if first_id == second_id or first_id not in by_id or second_id not in by_id:
            raise ZoneVisualQaError(f"Invalid worst seam pair: {metric['id']}")
        pair = (by_id[first_id], by_id[second_id])
        frame = _union_bounds(pair)
        captures.append(
            _capture(
                f"seam-{rank:02d}-{hashlib.sha256(str(metric['id']).encode()).hexdigest()[:10]}",
                "orthographic_worst_seam",
                "orthographic",
                pair,
                frame_bounds=frame,
                selection_basis={
                    "rank": rank,
                    "seam_id": metric["id"],
                    "orientation": direction,
                    "metrics": dict(metric),
                    "render_contract": "exactly_two_adjacent_lod0_tiles",
                },
            )
        )

    def ring(tile: ZoneTile) -> tuple[ZoneTile, ...]:
        return tuple(
            sorted(
                (
                    candidate
                    for (grid_x, grid_y), candidate in by_grid.items()
                    if abs(grid_x - tile.grid_x) <= 1 and abs(grid_y - tile.grid_y) <= 1
                ),
                key=lambda item: item.tile_id,
            )
        )

    relief_selections = (
        (
            "oblique-relief-highest",
            max(tiles, key=lambda tile: (tile.bounds[5], tile.tile_id)),
            "maximum_elevation_m",
            35.0,
        ),
        (
            "oblique-relief-lowest",
            min(tiles, key=lambda tile: (tile.bounds[2], tile.tile_id)),
            "minimum_elevation_m",
            125.0,
        ),
        (
            "oblique-relief-range",
            max(tiles, key=lambda tile: (tile.height_range_m, tile.tile_id)),
            "maximum_local_relief_m",
            215.0,
        ),
    )
    for capture_id, tile, basis, azimuth in relief_selections:
        captures.append(
            _capture(
                capture_id,
                "oblique_relief_extreme",
                "perspective_oblique",
                ring(tile),
                focus_tile_id=tile.tile_id,
                selection_basis={
                    basis: tile.bounds[5]
                    if basis == "maximum_elevation_m"
                    else tile.bounds[2]
                    if basis == "minimum_elevation_m"
                    else tile.height_range_m
                },
                azimuth_deg=azimuth,
            )
        )

    network_scores = {tile.tile_id: _network_score(tile) for tile in tiles}
    network_tile = max(
        tiles,
        key=lambda tile: (
            network_scores[tile.tile_id][0],
            network_scores[tile.tile_id][1],
            tile.tile_id,
        ),
    )
    captures.append(
        _capture(
            "oblique-networks",
            "oblique_network_richness",
            "perspective_oblique",
            ring(network_tile),
            focus_tile_id=network_tile.tile_id,
            selection_basis={
                "network_length_m": network_scores[network_tile.tile_id][0],
                "network_feature_count": network_scores[network_tile.tile_id][1],
                "roles": sorted(NETWORK_ROLES),
            },
            azimuth_deg=305.0,
        )
    )

    surface_scores = {tile.tile_id: _surface_score(tile) for tile in tiles}
    surface_candidates = sorted(
        tiles,
        key=lambda tile: (
            -surface_scores[tile.tile_id][2],
            surface_scores[tile.tile_id][0],
            tile.tile_id,
        ),
    )
    used_profiles: set[int] = set()
    selected_surfaces: list[ZoneTile] = []
    for tile in surface_candidates:
        profile_index = surface_scores[tile.tile_id][0]
        if profile_index in used_profiles:
            continue
        used_profiles.add(profile_index)
        selected_surfaces.append(tile)
        if len(selected_surfaces) == 3:
            break
    for rank, tile in enumerate(selected_surfaces, start=1):
        profile_index, total_weight, ratio = surface_scores[tile.tile_id]
        captures.append(
            _capture(
                f"oblique-dominant-surface-{rank:02d}",
                "oblique_dominant_surface",
                "perspective_oblique",
                ring(tile),
                focus_tile_id=tile.tile_id,
                selection_basis={
                    "profile_index": profile_index,
                    "total_weight": total_weight,
                    "weighted_share": ratio,
                },
                azimuth_deg=(55.0 + 90.0 * (rank - 1)) % 360.0,
            )
        )

    identifiers = [capture["capture_id"] for capture in captures]
    if len(identifiers) != len(set(identifiers)):
        raise ZoneVisualQaError("Capture plan identifiers are not unique")
    plan_basis = {
        "schema": CAPTURE_PLAN_SCHEMA,
        "zone_id": inspection.job["zone_id"],
        "revision": inspection.job["revision"],
        "recipe_id": inspection.job["recipe_id"],
        "recipe_build_id": inspection.job["recipe_build_id"],
        "build_id": inspection.job["build_id"],
        "catalog_sha256": inspection.catalog_sha256,
        "qa_metrics_sha256": inspection.qa_metrics_sha256,
        "zone_bounds_l93_ngf_m": list(zone_bounds),
        "selection_contract": {
            "orthographic_full_square": 1,
            "orthographic_grid_3x3": 9,
            "maximum_worst_seams": maximum_seams,
            "available_seams": len(inspection.seam_metrics),
            "selected_worst_seams": len(worst),
            "worst_seam_sort": [
                "maximum_height_gap_mm desc",
                "normal_mismatch_count desc",
                "stitch_signature_mismatch_count desc",
                "composition_failure_count desc",
                "id asc",
            ],
            "oblique": [
                "highest",
                "lowest",
                "maximum_local_relief",
                "network_richest",
                "up_to_three_distinct_dominant_profiles",
            ],
        },
        "captures": captures,
    }
    return {**plan_basis, "plan_sha256": _canonical_sha256(plan_basis)}


def _validate_capture_plan_artifact(
    receipt_path: Path, receipt: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    plan_path = _validate_artifact(
        receipt_path.parent,
        receipt.get("capture_plan"),
        "capture plan",
        exact_path="zone-visual-capture-plan.v1.json",
    )
    plan = _read_json(plan_path, "zone visual capture plan")
    if plan.get("schema") != CAPTURE_PLAN_SCHEMA:
        raise ZoneVisualQaError("Unsupported zone visual capture plan")
    for identity in (
        "zone_id",
        "revision",
        "recipe_id",
        "recipe_build_id",
        "build_id",
        "catalog_sha256",
        "qa_metrics_sha256",
    ):
        if plan.get(identity) != receipt.get(identity):
            raise ZoneVisualQaError(
                f"Capture plan {identity} differs from the technical receipt"
            )
    declared_plan_hash = _require_hash(plan.get("plan_sha256"), "plan.plan_sha256")
    receipt_plan_hash = _require_hash(
        receipt.get("capture_plan_sha256"), "capture_plan_sha256"
    )
    calculated_plan_hash = _canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if (
        declared_plan_hash != calculated_plan_hash
        or receipt_plan_hash != declared_plan_hash
    ):
        raise ZoneVisualQaError("Capture plan canonical hash mismatch")

    selection = plan.get("selection_contract")
    captures = plan.get("captures")
    if not isinstance(selection, Mapping) or not isinstance(captures, list):
        raise ZoneVisualQaError("Capture plan selection contract is incomplete")
    maximum_seams = selection.get("maximum_worst_seams")
    available_seams = selection.get("available_seams")
    selected_seams = selection.get("selected_worst_seams")
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (maximum_seams, available_seams, selected_seams)
        )
        or not 0 <= maximum_seams <= 20
        or available_seams < 0
        or selected_seams != min(maximum_seams, available_seams)
    ):
        raise ZoneVisualQaError("Capture plan seam selection is inconsistent")

    allowed_categories = {
        "orthographic_full_square",
        "orthographic_grid_3x3",
        "orthographic_worst_seam",
        "oblique_relief_extreme",
        "oblique_network_richness",
        "oblique_dominant_surface",
    }
    expected_projection = {
        "orthographic_full_square": "orthographic",
        "orthographic_grid_3x3": "orthographic",
        "orthographic_worst_seam": "orthographic",
        "oblique_relief_extreme": "perspective_oblique",
        "oblique_network_richness": "perspective_oblique",
        "oblique_dominant_surface": "perspective_oblique",
    }
    by_id: dict[str, Mapping[str, Any]] = {}
    counts = {category: 0 for category in allowed_categories}
    for capture in captures:
        if not isinstance(capture, Mapping):
            raise ZoneVisualQaError("Capture plan entry is not an object")
        capture_id = capture.get("capture_id")
        category = capture.get("category")
        tile_ids = capture.get("tile_ids")
        if (
            not isinstance(capture_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", capture_id) is None
            or capture_id in by_id
        ):
            raise ZoneVisualQaError(
                "Capture plan identifiers are invalid or duplicated"
            )
        if category not in allowed_categories:
            raise ZoneVisualQaError(f"Unsupported capture category: {category}")
        if capture.get("projection") != expected_projection[category]:
            raise ZoneVisualQaError(
                f"Capture projection differs from category: {capture_id}"
            )
        if (
            not isinstance(tile_ids, list)
            or not tile_ids
            or any(not isinstance(tile_id, str) or not tile_id for tile_id in tile_ids)
            or len(tile_ids) != len(set(tile_ids))
        ):
            raise ZoneVisualQaError(f"Capture tile set is invalid: {capture_id}")
        if category == "orthographic_worst_seam" and len(tile_ids) != 2:
            raise ZoneVisualQaError(
                "A worst-seam capture must contain exactly two tiles"
            )
        counts[category] += 1
        by_id[capture_id] = capture

    if (
        counts["orthographic_full_square"] != 1
        or counts["orthographic_grid_3x3"] != 9
        or counts["orthographic_worst_seam"] != selected_seams
        or counts["oblique_relief_extreme"] != 3
        or counts["oblique_network_richness"] != 1
        or not 1 <= counts["oblique_dominant_surface"] <= 3
    ):
        raise ZoneVisualQaError(
            "Capture plan does not contain the exhaustive required set"
        )
    return plan, by_id


def _reference_material_sources(bpy: Any, inspection: ZoneInspection) -> dict[str, Any]:
    """Load the shared, hash-validated atlas once into numerical arrays."""

    inspected = inspect_package(inspection.tiles[0].tile_root)
    contract = inspected["ground_material_contract"]
    manifest = inspected["manifest"]
    atlas_maps: dict[str, Any] = {}
    atlas_hashes: dict[str, str] = {}
    for role in RUNTIME_TEXTURE_ROLES:
        record = manifest["ground_material"]["runtime_atlas"][role]
        atlas_path = resolve_zone_asset(
            inspected["zone_root"], record["path"], f"runtime_atlas.{role}"
        )
        atlas_maps[role] = _load_image_pixels(
            bpy, atlas_path, non_color=(role != "basecolor")
        )
        atlas_hashes[role] = record["sha256"]
    return {
        "atlas_maps": atlas_maps,
        "atlas_hashes": atlas_hashes,
        "profile_table": contract["profile_table"],
        "ground_material_contract_sha256": inspection.material_contract_sha256,
        "material_model": contract["material_model"] + " Blender reference",
        "connected_channels": ["basecolor", "normal", "height_bump", "orm"],
    }


def _capture_tile_pixels(capture: Mapping[str, Any], tiles: Sequence[ZoneTile]) -> int:
    columns = (
        max(tile.grid_x for tile in tiles) - min(tile.grid_x for tile in tiles) + 1
    )
    rows = max(tile.grid_y for tile in tiles) - min(tile.grid_y for tile in tiles) + 1
    ideal = {
        "orthographic_full_square": 32,
        "orthographic_grid_3x3": 64,
        "orthographic_worst_seam": 256,
        "oblique_relief_extreme": 192,
        "oblique_network_richness": 192,
        "oblique_dominant_surface": 192,
    }.get(str(capture["category"]), 64)
    # Float RGBA mosaics for all four PBR channels stay below 2048 px per axis.
    return max(8, min(ideal, 2048 // max(columns, rows)))


def _build_reference_mosaic(
    bpy: Any,
    capture: Mapping[str, Any],
    tiles: Sequence[ZoneTile],
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    minimum_grid_x = min(tile.grid_x for tile in tiles)
    minimum_grid_y = min(tile.grid_y for tile in tiles)
    columns = max(tile.grid_x for tile in tiles) - minimum_grid_x + 1
    rows = max(tile.grid_y for tile in tiles) - minimum_grid_y + 1
    tile_pixels = _capture_tile_pixels(capture, tiles)
    width = columns * tile_pixels
    height = rows * tile_pixels
    maps = {
        role: np.zeros((height, width, 4), dtype=np.float32)
        for role in RUNTIME_TEXTURE_ROLES
    }
    for role in RUNTIME_TEXTURE_ROLES:
        maps[role][:, :, 3] = 1.0
    derived_tile_hashes: dict[str, Mapping[str, str]] = {}
    for tile in tiles:
        ids_pixels = _load_image_pixels(
            bpy, tile.tile_root / "ground-profile-ids.png", non_color=True
        )
        weight_pixels = _load_image_pixels(
            bpy, tile.tile_root / "ground-profile-weights.png", non_color=True
        )
        identifiers = np.floor(ids_pixels * 255.0 + 0.5).astype(np.uint8)
        weights = np.floor(weight_pixels * 255.0 + 0.5).astype(np.uint8)
        if (tile.tile_root / "ground-confidence.png").is_file():
            confidence_pixels = _load_image_pixels(
                bpy, tile.tile_root / "ground-confidence.png", non_color=True
            )
            orientation_pixels = _load_image_pixels(
                bpy, tile.tile_root / "ground-orientation.png", non_color=True
            )
            ground_confidence = np.floor(
                confidence_pixels[:, :, 0] * 255.0 + 0.5
            ).astype(np.uint8)
            ground_orientation = np.floor(
                orientation_pixels[:, :, 0] * 255.0 + 0.5
            ).astype(np.uint8)
        else:
            ground_confidence = None
            ground_orientation = None
        reference = compose_reference_pbr_maps(
            identifiers,
            weights,
            sources["atlas_maps"],
            sources["profile_table"],
            tile_origin_l93_m=(tile.bounds[0], tile.bounds[1]),
            output_size=tile_pixels,
            ground_confidence=ground_confidence,
            ground_orientation=ground_orientation,
        )
        column = tile.grid_x - minimum_grid_x
        row = tile.grid_y - minimum_grid_y
        y_slice = slice(row * tile_pixels, (row + 1) * tile_pixels)
        x_slice = slice(column * tile_pixels, (column + 1) * tile_pixels)
        for role in RUNTIME_TEXTURE_ROLES:
            maps[role][y_slice, x_slice] = reference["maps"][role]
        derived_tile_hashes[tile.tile_id] = reference["derived_sha256"]
    mosaic_hashes = {
        role: hashlib.sha256(
            np.asarray(maps[role], dtype="<f4").tobytes(order="C")
        ).hexdigest()
        for role in RUNTIME_TEXTURE_ROLES
    }
    return {
        "maps": maps,
        "minimum_grid": [minimum_grid_x, minimum_grid_y],
        "grid_size": [columns, rows],
        "tile_pixels": tile_pixels,
        "size_px": [width, height],
        "mosaic_sha256": mosaic_hashes,
        "derived_tile_sha256": derived_tile_hashes,
        "working_bytes": width * height * 4 * 4 * len(RUNTIME_TEXTURE_ROLES),
    }


def _install_mosaic_material(
    bpy: Any,
    terrain_by_id: Mapping[str, Any],
    tiles_by_id: Mapping[str, ZoneTile],
    mosaic: Mapping[str, Any],
) -> dict[str, Any]:
    material = bpy.data.materials.new("FireViewerZoneGroundSurfaceReference")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    for node in tuple(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    texture_nodes: dict[str, Any] = {}
    for role in RUNTIME_TEXTURE_ROLES:
        image = _create_float_image(
            bpy, f"FireViewerZoneReference_{role}", mosaic["maps"][role]
        )
        texture = nodes.new("ShaderNodeTexImage")
        texture.name = f"FireViewer_{role}"
        texture.image = image
        texture.interpolation = "Linear"
        texture.extension = "EXTEND"
        texture_nodes[role] = texture
    orm = nodes.new("ShaderNodeSeparateColor")
    normal = nodes.new("ShaderNodeNormalMap")
    normal.space = "TANGENT"
    height = nodes.new("ShaderNodeSeparateColor")
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = 0.15
    material.node_tree.links.new(
        texture_nodes["basecolor"].outputs["Color"], shader.inputs["Base Color"]
    )
    material.node_tree.links.new(
        texture_nodes["orm"].outputs["Color"], orm.inputs["Color"]
    )
    material.node_tree.links.new(orm.outputs["Green"], shader.inputs["Roughness"])
    material.node_tree.links.new(orm.outputs["Blue"], shader.inputs["Metallic"])
    material.node_tree.links.new(
        texture_nodes["normal"].outputs["Color"], normal.inputs["Color"]
    )
    material.node_tree.links.new(
        texture_nodes["height"].outputs["Color"], height.inputs["Color"]
    )
    material.node_tree.links.new(height.outputs["Red"], bump.inputs["Height"])
    material.node_tree.links.new(normal.outputs["Normal"], bump.inputs["Normal"])
    material.node_tree.links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    lod_aov = nodes.new("ShaderNodeOutputAOV")
    lod_aov.aov_name = TERRAIN_AOV
    lod_aov.inputs["Value"].default_value = 0.0
    coverage_aov = nodes.new("ShaderNodeOutputAOV")
    coverage_aov.aov_name = COVERAGE_AOV
    coverage_aov.inputs["Value"].default_value = 1.0
    material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    minimum_grid_x, minimum_grid_y = mosaic["minimum_grid"]
    columns, rows = mosaic["grid_size"]
    for tile_id, terrain in terrain_by_id.items():
        tile = tiles_by_id[tile_id]
        uv_layer = terrain.data.uv_layers.get(
            "FireViewerZoneUv"
        ) or terrain.data.uv_layers.new(name="FireViewerZoneUv")
        xs = [vertex.co.x for vertex in terrain.data.vertices]
        ys = [vertex.co.y for vertex in terrain.data.vertices]
        minimum_x, maximum_x = min(xs), max(xs)
        minimum_y, maximum_y = min(ys), max(ys)
        if maximum_x <= minimum_x or maximum_y <= minimum_y:
            raise ZoneVisualQaError(f"Imported terrain bounds are invalid: {tile_id}")
        tile_column = tile.grid_x - minimum_grid_x
        tile_row = tile.grid_y - minimum_grid_y
        for polygon in terrain.data.polygons:
            for loop_index in polygon.loop_indices:
                vertex = terrain.data.vertices[
                    terrain.data.loops[loop_index].vertex_index
                ]
                local_u = (vertex.co.x - minimum_x) / (maximum_x - minimum_x)
                local_v = (vertex.co.y - minimum_y) / (maximum_y - minimum_y)
                uv_layer.data[loop_index].uv = (
                    (tile_column + local_u) / columns,
                    (tile_row + local_v) / rows,
                )
        terrain.data.materials.clear()
        terrain.data.materials.append(material)
    return {
        "material_name": material.name,
        "model": "FireViewerGroundSurface_v1 Blender reference",
        "connected_channels": ["basecolor", "normal", "height_bump", "orm"],
        "aovs": [TERRAIN_AOV, COVERAGE_AOV],
        "mosaic_size_px": mosaic["size_px"],
        "tile_composition_size_px": mosaic["tile_pixels"],
        "mosaic_sha256": mosaic["mosaic_sha256"],
        "working_bytes": mosaic["working_bytes"],
    }


def _validated_tile_root_stage(tile: ZoneTile) -> Path:
    relative_stage = tile.package.get("root_stage")
    root_stage = _relative_path(
        tile.tile_root.resolve(), relative_stage, f"tile {tile.tile_id} root stage"
    )
    outputs = tile.package.get("outputs")
    record = outputs.get(relative_stage) if isinstance(outputs, Mapping) else None
    if not isinstance(record, Mapping) or set(record) != {"bytes", "sha256"}:
        raise ZoneVisualQaError(
            f"Tile root stage output record is missing: {tile.tile_id}"
        )
    _require_hash(record.get("sha256"), f"tile {tile.tile_id} root stage sha256")
    if not root_stage.is_file():
        raise ZoneVisualQaError(f"Tile root stage is absent: {tile.tile_id}")
    if record.get("bytes") != root_stage.stat().st_size or record.get(
        "sha256"
    ) != _sha256(root_stage):
        raise ZoneVisualQaError(f"Tile root stage hash mismatch: {tile.tile_id}")
    return root_stage


def _import_capture_lod0(
    bpy: Any, capture: Mapping[str, Any], inspection: ZoneInspection
) -> dict[str, Any]:
    by_id = inspection.tiles_by_id
    expected_ids = tuple(capture["tile_ids"])
    if capture["category"] == "orthographic_worst_seam" and len(expected_ids) != 2:
        raise ZoneVisualQaError(
            "A seam capture must import exactly two LOD0 neighbours"
        )
    terrain_by_id: dict[str, Any] = {}
    for tile_id in expected_ids:
        tile = by_id[tile_id]
        root_stage = _validated_tile_root_stage(tile)
        before = {item.as_pointer() for item in bpy.context.scene.objects}
        result = bpy.ops.wm.usd_import(
            **_operator_arguments(
                bpy.ops.wm.usd_import,
                {
                    "filepath": str(root_stage),
                    "import_cameras": False,
                    "import_lights": False,
                    "import_materials": False,
                    "import_textures_mode": "IMPORT_NONE",
                    "read_mesh_attributes": True,
                    "attr_import_mode": "ALL",
                    "validate_meshes": True,
                    "relative_path": True,
                },
            )
        )
        if "FINISHED" not in result:
            raise ZoneVisualQaError(
                f"Blender USD import failed for {tile_id}: {result}"
            )
        new_meshes = [
            item
            for item in bpy.context.scene.objects
            if item.as_pointer() not in before and item.type == "MESH"
        ]
        if len(new_meshes) != 1:
            raise ZoneVisualQaError(
                f"LOD0 package {tile_id} imported {len(new_meshes)} meshes"
            )
        terrain = new_meshes[0]
        expected_metrics = tile.package["lod_metrics"]["lod0"]
        if (
            len(terrain.data.vertices) != expected_metrics["vertex_count"]
            or len(terrain.data.polygons) != expected_metrics["triangle_count"]
        ):
            raise ZoneVisualQaError(f"Imported LOD0 counts differ for {tile_id}")
        terrain.name = f"FireViewerTerrain_{tile_id}"
        terrain["fireviewer:tile_id"] = tile_id
        terrain["fireviewer:terrain_lod"] = 0
        terrain_by_id[tile_id] = terrain
    imported_meshes = [
        item for item in bpy.context.scene.objects if item.type == "MESH"
    ]
    if len(imported_meshes) != len(expected_ids) or set(terrain_by_id) != set(
        expected_ids
    ):
        raise ZoneVisualQaError("Capture contains undeclared or missing terrain meshes")
    return terrain_by_id


def _configure_scene(bpy: Any, resolution: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.look = "AgX - Medium High Contrast"
    view_layer = scene.view_layers[0]
    existing_aovs = {aov.name for aov in view_layer.aovs}
    for name in (TERRAIN_AOV, COVERAGE_AOV):
        if name not in existing_aovs:
            aov = view_layer.aovs.add()
            aov.name = name
            aov.type = "VALUE"
    world = scene.world or bpy.data.worlds.new("FireViewerZoneQaWorld")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is None:
        raise ZoneVisualQaError("Blender world has no Background node")
    background.inputs["Color"].default_value = (0.025, 0.035, 0.05, 1.0)
    background.inputs["Strength"].default_value = 0.35
    sun_data = bpy.data.lights.new("FireViewerZoneQaSun", type="SUN")
    sun_data.energy = 2.5
    sun_data.angle = math.radians(4.0)
    sun = bpy.data.objects.new("FireViewerZoneQaSun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(28.0), math.radians(-18.0), math.radians(32.0))


def _capture_camera(
    bpy: Any, capture: Mapping[str, Any], zone_origin_l93_m: Sequence[float]
) -> tuple[Any, dict[str, Any]]:
    from mathutils import Vector

    west, south, minimum_z, east, north, maximum_z = (
        float(value) for value in capture["frame_bounds_l93_ngf_m"]
    )
    centre = Vector(
        (
            (west + east) * 0.5 - float(zone_origin_l93_m[0]),
            (south + north) * 0.5 - float(zone_origin_l93_m[1]),
            (minimum_z + maximum_z) * 0.5,
        )
    )
    span = max(east - west, north - south, TILE_SIZE_M)
    camera_data = bpy.data.cameras.new(f"Camera_{capture['capture_id']}")
    camera_data.clip_start = float(capture["camera"]["clip_start_m"])
    camera_data.clip_end = float(capture["camera"]["clip_end_m"])
    camera = bpy.data.objects.new(f"Camera_{capture['capture_id']}", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    if capture["projection"] == "orthographic":
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = span * (
            1.0 + float(capture["camera"]["margin_ratio"])
        )
        camera.location = (centre.x, centre.y, maximum_z + max(1_000.0, span))
        camera.rotation_euler = (0.0, 0.0, 0.0)
        record = {
            "type": "orthographic",
            "ortho_scale_m": camera_data.ortho_scale,
            "location_zone_m": [float(value) for value in camera.location],
            "target_zone_m": [float(value) for value in centre],
        }
    else:
        camera_data.type = "PERSP"
        camera_data.lens = float(capture["camera"]["lens_mm"])
        azimuth = math.radians(float(capture["camera"]["azimuth_deg"]))
        elevation = math.radians(float(capture["camera"]["elevation_deg"]))
        distance = span * 1.65
        horizontal = distance * math.cos(elevation)
        camera.location = centre + Vector(
            (
                -math.cos(azimuth) * horizontal,
                -math.sin(azimuth) * horizontal,
                distance * math.sin(elevation),
            )
        )
        camera.rotation_euler = (
            (centre - camera.location).to_track_quat("-Z", "Y").to_euler()
        )
        record = {
            "type": "perspective",
            "lens_mm": camera_data.lens,
            "azimuth_deg": float(capture["camera"]["azimuth_deg"]),
            "elevation_deg": float(capture["camera"]["elevation_deg"]),
            "location_zone_m": [float(value) for value in camera.location],
            "target_zone_m": [float(value) for value in centre],
        }
    bpy.context.scene.camera = camera
    return camera, record


def _saved_render_pixels(bpy: Any, path: Path, *, file_format: str) -> Any:
    import numpy as np

    scene = bpy.context.scene
    render_result = bpy.data.images.get("Render Result")
    if render_result is None:
        raise ZoneVisualQaError("Blender did not produce a render result")
    scene.render.image_settings.file_format = file_format
    if file_format == "PNG":
        scene.render.image_settings.color_depth = "8"
    else:
        scene.render.image_settings.color_depth = "32"
    render_result.save_render(filepath=str(path), scene=scene)
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = (int(value) for value in image.size)
        pixels = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        return pixels.reshape((height, width, 4))
    finally:
        bpy.data.images.remove(image)


def _render_capture_artifacts(
    bpy: Any,
    camera: Any,
    output_root: Path,
    resolution: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    scene = bpy.context.scene
    paths = {
        name: output_root / file_name
        for name, file_name in CAPTURE_ARTIFACT_NAMES.items()
    }
    output_root.mkdir(parents=True, exist_ok=True)
    scene.use_nodes = False
    scene.camera = camera
    bpy.ops.render.render(write_still=False)
    beauty = _saved_render_pixels(bpy, paths["beauty"], file_format="PNG")
    if beauty.shape[:2] != (resolution, resolution):
        raise ZoneVisualQaError("Beauty render has unexpected dimensions")

    scene.use_nodes = True
    node_tree = scene.node_tree
    if node_tree is None:
        raise ZoneVisualQaError("Blender scene has no compositor node tree")

    def render_aov(name: str, path: Path) -> Any:
        node_tree.nodes.clear()
        render_layers = node_tree.nodes.new("CompositorNodeRLayers")
        composite = node_tree.nodes.new("CompositorNodeComposite")
        aov_output = render_layers.outputs.get(name)
        if aov_output is None:
            raise ZoneVisualQaError(f"Blender compositor did not expose {name}")
        node_tree.links.new(aov_output, composite.inputs["Image"])
        bpy.ops.render.render(write_still=False)
        pixels = _saved_render_pixels(bpy, path, file_format="OPEN_EXR")
        if pixels.shape[:2] != (resolution, resolution):
            raise ZoneVisualQaError(f"{name} render has unexpected dimensions")
        return pixels

    coverage = render_aov(COVERAGE_AOV, paths["terrain_coverage"])
    terrain_mask = coverage[:, :, 0] > 0.5
    terrain_indices = np.flatnonzero(terrain_mask.ravel())
    if terrain_indices.size == 0:
        raise ZoneVisualQaError("Primary camera rendered no terrain pixels")
    lod = render_aov(TERRAIN_AOV, paths["terrain_lod"])
    aov_metrics = validate_primary_terrain_aovs(
        lod[:, :, 0].ravel()[terrain_indices],
        coverage[:, :, 0].ravel()[terrain_indices],
        expected_lod=0,
    )
    terrain_rgb = np.asarray(beauty[:, :, :3][terrain_mask], dtype=np.float64)
    if terrain_rgb.size == 0 or not np.isfinite(terrain_rgb).all():
        raise ZoneVisualQaError("Beauty proof has no finite terrain pixels")
    metrics = {
        **aov_metrics,
        "frame_coverage_ratio": float(terrain_indices.size / terrain_mask.size),
        "rgb_variance": float(np.var(terrain_rgb)),
        "rgb_mean": [float(value) for value in np.mean(terrain_rgb, axis=0)],
    }
    return (
        {
            name: _artifact(path, relative_to=output_root)
            for name, path in paths.items()
        },
        metrics,
    )


def validate_capture_receipt(
    receipt_path: Path,
    *,
    expected_capture: Mapping[str, Any] | None = None,
    expected_plan_sha256: str | None = None,
    expected_job_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(receipt_path).resolve()
    receipt = _read_json(path, "zone visual capture receipt")
    if (
        receipt.get("schema") != CAPTURE_RECEIPT_SCHEMA
        or receipt.get("status") != "rendered_technical"
    ):
        raise ZoneVisualQaError("Unsupported zone visual capture receipt")
    _require_hash(receipt.get("capture_spec_sha256"), "capture_spec_sha256")
    _require_hash(receipt.get("capture_plan_sha256"), "capture_plan_sha256")
    _require_hash(receipt.get("job_sha256"), "job_sha256")
    if expected_capture is not None:
        if receipt.get("capture_id") != expected_capture.get(
            "capture_id"
        ) or receipt.get("capture_spec_sha256") != _canonical_sha256(expected_capture):
            raise ZoneVisualQaError("Capture receipt differs from its planned capture")
        if receipt.get("tile_ids") != expected_capture.get("tile_ids"):
            raise ZoneVisualQaError("Capture tile identities differ from the plan")
        if (
            expected_capture.get("category") == "orthographic_worst_seam"
            and len(receipt["tile_ids"]) != 2
        ):
            raise ZoneVisualQaError("Seam proof did not render exactly two tiles")
    if (
        expected_plan_sha256 is not None
        and receipt.get("capture_plan_sha256") != expected_plan_sha256
    ):
        raise ZoneVisualQaError("Capture receipt is bound to another capture plan")
    if (
        expected_job_sha256 is not None
        and receipt.get("job_sha256") != expected_job_sha256
    ):
        raise ZoneVisualQaError("Capture receipt is bound to another visual job")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        CAPTURE_ARTIFACT_NAMES
    ):
        raise ZoneVisualQaError("Capture proof artifact set is incomplete")
    for name, file_name in CAPTURE_ARTIFACT_NAMES.items():
        _validate_artifact(path.parent, artifacts[name], name, exact_path=file_name)
    aov = receipt.get("aov")
    if (
        not isinstance(aov, Mapping)
        or aov.get("terrain_lod_name") != TERRAIN_AOV
        or aov.get("terrain_coverage_name") != COVERAGE_AOV
        or aov.get("expected_lod") != 0
        or aov.get("invalid_lod_pixel_count") != 0
        or aov.get("invalid_coverage_pixel_count") != 0
        or not isinstance(aov.get("terrain_pixel_count"), int)
        or aov["terrain_pixel_count"] <= 0
    ):
        raise ZoneVisualQaError("Capture AOV proof is absent or contains forbidden LOD")
    imported = receipt.get("imported_lod0")
    if (
        not isinstance(imported, Mapping)
        or imported.get("lod") != 0
        or imported.get("tile_ids") != receipt.get("tile_ids")
        or imported.get("mesh_count") != len(receipt.get("tile_ids", []))
        or imported.get("forbidden_lod_mesh_count") != 0
    ):
        raise ZoneVisualQaError("Capture imported mesh proof is incomplete")
    material = receipt.get("reference_material")
    if (
        not isinstance(material, Mapping)
        or material.get("model") != "FireViewerGroundSurface_v1 Blender reference"
        or material.get("connected_channels")
        != ["basecolor", "normal", "height_bump", "orm"]
    ):
        raise ZoneVisualQaError("Capture did not use the textured reference material")
    return receipt


def _capture_receipt_path(output_root: Path, capture_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", capture_id):
        raise ZoneVisualQaError(f"Capture id is not portable: {capture_id}")
    return output_root / "captures" / capture_id / "capture.done.v1.json"


def _render_one_capture(
    bpy: Any,
    inspection: ZoneInspection,
    capture: Mapping[str, Any],
    plan_sha256: str,
    job_sha256: str,
    reference_sources: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = _capture_receipt_path(
        inspection.output_root, str(capture["capture_id"])
    )
    if receipt_path.is_file():
        return validate_capture_receipt(
            receipt_path,
            expected_capture=capture,
            expected_plan_sha256=plan_sha256,
            expected_job_sha256=job_sha256,
        )
    bpy.ops.wm.read_factory_settings(use_empty=True)
    resolution = int(inspection.job["render"]["resolution"])
    _configure_scene(bpy, resolution)
    terrain_by_id = _import_capture_lod0(bpy, capture, inspection)
    selected_tiles = tuple(
        inspection.tiles_by_id[tile_id] for tile_id in capture["tile_ids"]
    )
    mosaic = _build_reference_mosaic(bpy, capture, selected_tiles, reference_sources)
    material = _install_mosaic_material(
        bpy, terrain_by_id, inspection.tiles_by_id, mosaic
    )
    zone_origins = {
        tuple(float(value) for value in tile.package["zone_origin_l93_m"])
        for tile in selected_tiles
    }
    if len(zone_origins) != 1:
        raise ZoneVisualQaError("Capture tiles disagree on their zone origin")
    zone_origin = next(iter(zone_origins))
    camera, camera_record = _capture_camera(bpy, capture, zone_origin)
    capture_root = receipt_path.parent
    artifacts, metrics = _render_capture_artifacts(
        bpy, camera, capture_root, resolution
    )
    tile_packages = {
        tile.tile_id: {
            "terrain_usd_manifest_sha256": _sha256(
                tile.tile_root / "terrain-usd-package.v1.json"
            ),
            "terrain_lod0_fvtq_sha256": tile.lod0_sha256,
            "triangle_count": tile.lod0_triangles,
        }
        for tile in selected_tiles
    }
    receipt: dict[str, Any] = {
        "schema": CAPTURE_RECEIPT_SCHEMA,
        "status": "rendered_technical",
        "capture_id": capture["capture_id"],
        "category": capture["category"],
        "capture_spec_sha256": _canonical_sha256(capture),
        "capture_plan_sha256": plan_sha256,
        "job_sha256": job_sha256,
        "tile_ids": list(capture["tile_ids"]),
        "frame_bounds_l93_ngf_m": list(capture["frame_bounds_l93_ngf_m"]),
        "camera": camera_record,
        "render_resolution": [resolution, resolution],
        "imported_lod0": {
            "lod": 0,
            "mesh_count": len(terrain_by_id),
            "tile_ids": list(capture["tile_ids"]),
            "triangle_count": sum(tile.lod0_triangles for tile in selected_tiles),
            "forbidden_lod_mesh_count": 0,
            "tile_packages": tile_packages,
        },
        "reference_material": {
            **material,
            "ground_material_contract_sha256": inspection.material_contract_sha256,
            "runtime_atlas_sha256": reference_sources["atlas_hashes"],
            "derived_tile_sha256": mosaic["derived_tile_sha256"],
        },
        "artifacts": artifacts,
        "aov": {
            "terrain_lod_name": TERRAIN_AOV,
            "terrain_coverage_name": COVERAGE_AOV,
            "expected_lod": 0,
            "terrain_pixel_count": metrics["terrain_pixel_count"],
            "invalid_lod_pixel_count": metrics["invalid_lod_pixel_count"],
            "invalid_coverage_pixel_count": metrics["invalid_coverage_pixel_count"],
            "maximum_lod_absolute_error": metrics["maximum_lod_absolute_error"],
            "frame_coverage_ratio": metrics["frame_coverage_ratio"],
        },
        "beauty_metrics": {
            "rgb_variance": metrics["rgb_variance"],
            "rgb_mean": metrics["rgb_mean"],
        },
        "prohibited_dependencies": {
            "network_requests": 0,
            "orthophoto": False,
            "omniverse": False,
            "lod1_or_lod2_meshes": 0,
        },
    }
    _atomic_json(receipt_path, receipt)
    return validate_capture_receipt(
        receipt_path,
        expected_capture=capture,
        expected_plan_sha256=plan_sha256,
        expected_job_sha256=job_sha256,
    )


def _technical_capture_set_sha256(
    capture_records: Sequence[Mapping[str, Any]],
) -> str:
    return _canonical_sha256(
        [
            {
                "capture_id": record["capture_id"],
                "capture_spec_sha256": record["capture_spec_sha256"],
                "artifacts": record["artifacts"],
                "aov": record["aov"],
            }
            for record in capture_records
        ]
    )


def render_zone_visual_qa(job_path: Path) -> dict[str, Any]:
    """Render every planned capture in Blender; never accept it visually."""

    try:
        import bpy
    except ModuleNotFoundError as error:  # pragma: no cover - Blender only
        raise ZoneVisualQaError("The render phase must run inside Blender") from error
    if tuple(bpy.app.version[:2]) != (4, 5):
        raise ZoneVisualQaError(
            f"Blender 4.5 LTS is required, got {bpy.app.version_string}"
        )
    inspection = inspect_zone_job(job_path, require_d=True, validate_packages=True)
    plan = build_capture_plan(inspection)
    plan_path = inspection.output_root / "zone-visual-capture-plan.v1.json"
    if plan_path.is_file():
        existing = _read_json(plan_path, "existing zone visual capture plan")
        if existing != plan:
            raise ZoneVisualQaError("Refusing to replace a different capture plan")
    else:
        _atomic_json(plan_path, plan)
    job_sha256 = _sha256(inspection.job_path)
    # Read validated atlas bytes before per-capture factory resets.  Numerical
    # arrays survive the reset, unlike Blender image datablocks.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    reference_sources = _reference_material_sources(bpy, inspection)
    capture_records = [
        _render_one_capture(
            bpy,
            inspection,
            capture,
            plan["plan_sha256"],
            job_sha256,
            reference_sources,
        )
        for capture in plan["captures"]
    ]
    if any(
        record["aov"]["invalid_lod_pixel_count"]
        or record["aov"]["invalid_coverage_pixel_count"]
        for record in capture_records
    ):
        raise ZoneVisualQaError("At least one zone capture contains invalid AOV pixels")
    binary_path = Path(bpy.app.binary_path).resolve()
    budget = validate_zone_budget(inspection)
    receipt: dict[str, Any] = {
        "schema": TECHNICAL_RECEIPT_SCHEMA,
        "status": TECHNICAL_STATUS,
        "production_visual_gate_passed": False,
        "human_visual_acceptance": "pending_exhaustive_review",
        "zone_id": inspection.job["zone_id"],
        "revision": inspection.job["revision"],
        "recipe_id": inspection.job["recipe_id"],
        "recipe_build_id": inspection.job["recipe_build_id"],
        "build_id": inspection.job["build_id"],
        "job_sha256": job_sha256,
        "catalog_sha256": inspection.catalog_sha256,
        "qa_metrics_sha256": inspection.qa_metrics_sha256,
        "capture_plan": _artifact(plan_path, relative_to=inspection.output_root),
        "capture_plan_sha256": plan["plan_sha256"],
        "capture_count": len(capture_records),
        "capture_set_sha256": _technical_capture_set_sha256(capture_records),
        "captures": [
            {
                "capture_id": record["capture_id"],
                "category": record["category"],
                "tile_ids": record["tile_ids"],
                "capture_spec_sha256": record["capture_spec_sha256"],
                "receipt": _artifact(
                    _capture_receipt_path(inspection.output_root, record["capture_id"]),
                    relative_to=inspection.output_root,
                ),
                "artifacts": record["artifacts"],
            }
            for record in capture_records
        ],
        "aov": {
            "terrain_lod": TERRAIN_AOV,
            "terrain_coverage": COVERAGE_AOV,
            "expected_lod": 0,
            "invalid_lod_pixel_count": 0,
            "invalid_coverage_pixel_count": 0,
            "terrain_pixel_count": sum(
                int(record["aov"]["terrain_pixel_count"]) for record in capture_records
            ),
        },
        "budget": budget,
        "reference_material": {
            key: value
            for key, value in reference_sources.items()
            if key not in {"atlas_maps", "profile_table"}
        },
        "blender": {
            "version": ".".join(str(value) for value in bpy.app.version),
            "binary_sha256": _sha256(binary_path),
            "headless_contract": [
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--offline-mode",
                "--python-exit-code 1",
            ],
        },
        "acceptance_contract": {
            "automatic_acceptance": False,
            "required_review_schema": HUMAN_REVIEW_SCHEMA,
            "review_scope": "every_capture_and_every_beauty_lod_coverage_hash",
        },
        "prohibited_dependencies": {
            "network_requests": 0,
            "orthophoto": False,
            "omniverse": False,
            "lod1_or_lod2_primary_camera_pixels": 0,
        },
    }
    receipt_path = inspection.output_root / "zone-visual-technical-receipt.v1.json"
    if receipt_path.is_file():
        existing = _read_json(receipt_path, "existing zone technical receipt")
        if existing != receipt:
            raise ZoneVisualQaError("Refusing to replace a different technical receipt")
    else:
        _atomic_json(receipt_path, receipt)
    return validate_technical_receipt(receipt_path)


def validate_technical_receipt(receipt_path: Path) -> dict[str, Any]:
    path = Path(receipt_path).resolve()
    receipt = _read_json(path, "zone visual technical receipt")
    if (
        receipt.get("schema") != TECHNICAL_RECEIPT_SCHEMA
        or receipt.get("status") != TECHNICAL_STATUS
        or receipt.get("production_visual_gate_passed") is not False
        or receipt.get("human_visual_acceptance") != "pending_exhaustive_review"
    ):
        raise ZoneVisualQaError("Technical receipt must remain pending human review")
    _require_hash(receipt.get("recipe_id"), "technical receipt recipe_id")
    _require_hash(receipt.get("recipe_build_id"), "technical receipt recipe_build_id")
    _require_hash(receipt.get("build_id"), "technical receipt build_id")
    _require_hash(receipt.get("job_sha256"), "technical receipt job_sha256")
    _, planned_captures = _validate_capture_plan_artifact(path, receipt)
    captures = receipt.get("captures")
    if (
        not isinstance(captures, list)
        or not captures
        or receipt.get("capture_count") != len(captures)
        or len(captures) != len(planned_captures)
    ):
        raise ZoneVisualQaError("Technical receipt capture set is incomplete")
    capture_records: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for entry in captures:
        if not isinstance(entry, Mapping) or not isinstance(
            entry.get("capture_id"), str
        ):
            raise ZoneVisualQaError("Technical receipt capture entry is invalid")
        capture_id = entry["capture_id"]
        if capture_id in seen or capture_id not in planned_captures:
            raise ZoneVisualQaError(f"Duplicate technical capture: {capture_id}")
        seen.add(capture_id)
        receipt_record = entry.get("receipt")
        capture_path = _validate_artifact(
            path.parent, receipt_record, f"capture {capture_id}"
        )
        record = validate_capture_receipt(
            capture_path,
            expected_capture=planned_captures[capture_id],
            expected_plan_sha256=receipt.get("capture_plan_sha256"),
            expected_job_sha256=receipt.get("job_sha256"),
        )
        if (
            record.get("capture_id") != capture_id
            or record.get("category") != entry.get("category")
            or record.get("tile_ids") != entry.get("tile_ids")
            or record.get("capture_spec_sha256") != entry.get("capture_spec_sha256")
            or record.get("artifacts") != entry.get("artifacts")
        ):
            raise ZoneVisualQaError(f"Technical capture summary mismatch: {capture_id}")
        capture_records.append(record)
    if seen != set(planned_captures):
        raise ZoneVisualQaError(
            "Technical receipt omitted one or more planned captures"
        )
    if receipt.get("capture_set_sha256") != _technical_capture_set_sha256(
        capture_records
    ):
        raise ZoneVisualQaError("Technical capture set hash mismatch")
    aov = receipt.get("aov")
    if (
        not isinstance(aov, Mapping)
        or aov.get("expected_lod") != 0
        or aov.get("invalid_lod_pixel_count") != 0
        or aov.get("invalid_coverage_pixel_count") != 0
    ):
        raise ZoneVisualQaError("Technical aggregate AOV proof is invalid")
    return receipt


def validate_zone_capture_aovs(lod_values: Any, coverage_values: Any) -> dict[str, Any]:
    """Public pure gate: a primary zone capture may contain only LOD0 terrain."""

    try:
        return validate_primary_terrain_aovs(
            lod_values, coverage_values, expected_lod=0
        )
    except RuntimeError as error:
        raise ZoneVisualQaError(str(error)) from error


def create_human_review_template(
    technical_receipt_path: Path, output_path: Path
) -> dict[str, Any]:
    """Create a non-accepting checklist bound to every rendered byte."""

    technical_path = Path(technical_receipt_path).resolve()
    technical = validate_technical_receipt(technical_path)
    template = {
        "schema": HUMAN_REVIEW_SCHEMA,
        "decision": "pending",
        "reviewer": {"kind": "human", "id": ""},
        "decision_recorded_at_utc": "",
        "technical_receipt_sha256": _sha256(technical_path),
        "capture_set_sha256": technical["capture_set_sha256"],
        "capture_reviews": [
            {
                "capture_id": capture["capture_id"],
                "decision": "pending",
                "beauty_sha256": capture["artifacts"]["beauty"]["sha256"],
                "terrain_lod_sha256": capture["artifacts"]["terrain_lod"]["sha256"],
                "terrain_coverage_sha256": capture["artifacts"]["terrain_coverage"][
                    "sha256"
                ],
                "notes": "",
            }
            for capture in technical["captures"]
        ],
        "review_scope": [
            "orthographic_full_square",
            "all_nine_grid_captures",
            "all_selected_worst_seams",
            "all_oblique_relief_network_surface_captures",
            "beauty_and_both_aov_artifacts_for_every_capture",
        ],
        "notes": "",
    }
    _atomic_json(Path(output_path), template)
    return template


def _validate_human_review(
    review_path: Path,
    technical_path: Path,
    technical: Mapping[str, Any],
) -> dict[str, Any]:
    review = _read_json(review_path, "zone human visual review")
    if review.get("schema") != HUMAN_REVIEW_SCHEMA:
        raise ZoneVisualQaError("Unsupported human visual review")
    reviewer = review.get("reviewer")
    if (
        not isinstance(reviewer, Mapping)
        or reviewer.get("kind") != "human"
        or not isinstance(reviewer.get("id"), str)
        or not reviewer["id"].strip()
    ):
        raise ZoneVisualQaError("An identified human reviewer is required")
    recorded_at = review.get("decision_recorded_at_utc")
    if (
        not isinstance(recorded_at, str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", recorded_at) is None
    ):
        raise ZoneVisualQaError("Human decision timestamp must be explicit UTC")
    if review.get("decision") != "accepted":
        raise ZoneVisualQaError("Human zone review has not explicitly accepted the set")
    if review.get("technical_receipt_sha256") != _sha256(technical_path):
        raise ZoneVisualQaError("Human review is bound to another technical receipt")
    if review.get("capture_set_sha256") != technical.get("capture_set_sha256"):
        raise ZoneVisualQaError("Human review is bound to another capture set")
    expected = {capture["capture_id"]: capture for capture in technical["captures"]}
    raw_reviews = review.get("capture_reviews")
    if not isinstance(raw_reviews, list) or len(raw_reviews) != len(expected):
        raise ZoneVisualQaError("Human review must cover every capture exactly once")
    seen: set[str] = set()
    for item in raw_reviews:
        if not isinstance(item, Mapping) or item.get("capture_id") not in expected:
            raise ZoneVisualQaError("Human review contains an unknown capture")
        capture_id = str(item["capture_id"])
        if capture_id in seen:
            raise ZoneVisualQaError(f"Human review duplicates capture {capture_id}")
        seen.add(capture_id)
        expected_capture = expected[capture_id]
        if item.get("decision") != "accepted":
            raise ZoneVisualQaError(f"Human review did not accept capture {capture_id}")
        for field, artifact in (
            ("beauty_sha256", "beauty"),
            ("terrain_lod_sha256", "terrain_lod"),
            ("terrain_coverage_sha256", "terrain_coverage"),
        ):
            if item.get(field) != expected_capture["artifacts"][artifact]["sha256"]:
                raise ZoneVisualQaError(
                    f"Human review artifact hash mismatch: {capture_id}/{artifact}"
                )
    if seen != set(expected):
        raise ZoneVisualQaError("Human review omitted one or more captures")
    return review


def accept_zone_visual_review(
    technical_receipt_path: Path,
    human_review_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Emit final visual acceptance only from a complete explicit human review."""

    technical_path = _require_d_path(technical_receipt_path, "technical_receipt")
    review_path = _require_d_path(human_review_path, "human_review")
    destination = _require_d_path(output_path, "acceptance_output")
    technical = validate_technical_receipt(technical_path)
    review = _validate_human_review(review_path, technical_path, technical)
    acceptance: dict[str, Any] = {
        "schema": ACCEPTANCE_SCHEMA,
        "status": ACCEPTED_STATUS,
        "zone_visual_gate_passed": True,
        "automatic_acceptance": False,
        "review_kind": "explicit_exhaustive_human",
        "zone_id": technical["zone_id"],
        "revision": technical["revision"],
        "recipe_id": technical["recipe_id"],
        "recipe_build_id": technical["recipe_build_id"],
        "build_id": technical["build_id"],
        "technical_receipt": _artifact(technical_path),
        "technical_receipt_sha256": _sha256(technical_path),
        "human_review": _artifact(review_path),
        "human_review_sha256": _sha256(review_path),
        "reviewer": dict(review["reviewer"]),
        "decision_recorded_at_utc": review["decision_recorded_at_utc"],
        "capture_count": technical["capture_count"],
        "capture_set_sha256": technical["capture_set_sha256"],
        "aov": dict(technical["aov"]),
        "scope_boundary": {
            "accepted": "zone terrain visual evidence only",
            "not_accepted": [
                "atlas_library_as_an_independent_dependency",
                "buildings",
                "roads_3d",
                "vegetation",
                "fire_or_simulation",
                "omniverse_runtime",
            ],
        },
    }
    if destination.is_file():
        existing = _read_json(destination, "existing zone visual acceptance")
        if existing != acceptance:
            raise ZoneVisualQaError("Refusing to replace another human acceptance")
    else:
        _atomic_json(destination, acceptance)
    return acceptance


def _parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render or explicitly accept complete-zone adaptive terrain QA."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render")
    render.add_argument("--job", required=True, type=Path)
    template = subparsers.add_parser("review-template")
    template.add_argument("--technical-receipt", required=True, type=Path)
    template.add_argument("--output", required=True, type=Path)
    accept = subparsers.add_parser("accept")
    accept.add_argument("--technical-receipt", required=True, type=Path)
    accept.add_argument("--human-review", required=True, type=Path)
    accept.add_argument("--output", required=True, type=Path)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    raw = (
        list(sys.argv[sys.argv.index("--") + 1 :])
        if arguments is None and "--" in sys.argv
        else list(arguments or sys.argv[1:])
    )
    parsed = _parse_arguments(raw)
    if parsed.command == "render":
        receipt = render_zone_visual_qa(parsed.job)
    elif parsed.command == "review-template":
        receipt = create_human_review_template(parsed.technical_receipt, parsed.output)
    else:
        receipt = accept_zone_visual_review(
            parsed.technical_receipt, parsed.human_review, parsed.output
        )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
