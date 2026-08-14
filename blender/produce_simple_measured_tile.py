"""Produce one measured FireViewer tile from already-local measured sources.

This is deliberately a small, source-free runtime pipeline: one co-registered
0.5 m MNT/MNS pair and one 1 m orthophoto are compiled into FVTG terrain, one
RGB ground texture, an MNT/MNS placement inventory, fixed-terrain USD and a
measured USD scene. It never downloads data and never invokes the retired PBR
pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from PIL import Image

BLENDER_ROOT = Path(__file__).resolve().parent
OMNIVERSE_ROOT = BLENDER_ROOT.parent / "omniverse"
for _module_root in (BLENDER_ROOT, OMNIVERSE_ROOT):
    if str(_module_root) not in sys.path:
        sys.path.insert(0, str(_module_root))

from build_measured_scene_usd import (
    TerrainReference,
    build_measured_scene_usd,
    validate_measured_scene_package,
)
from fixed_asset_placement import (
    FixedAssetPlacementError,
    validate_projected_placements,
)
from fixed_terrain_grid import (
    FixedTerrainTile,
    compile_fixed_terrain_from_canonical_mm,
    read_fixed_terrain,
    write_fixed_terrain,
)
from fixed_terrain_usd import (
    export_fixed_terrain_usd,
    validate_fixed_terrain_usd_package,
)
from mns_mnt_placement_inventory import (
    PlacementInventoryError,
    build_placement_inventory,
    read_hag_1m,
    validate_inventory,
    write_placement_outputs,
)
from orthophoto_ground_texture import (
    compile_aligned_window,
    slice_tile,
    write_tile_outputs,
)

CONTRACT_SCHEMA = "fireviewer.simple-measured-tile-contract.v1"
RECEIPT_SCHEMA = "fireviewer.simple-measured-tile-production.v1"
REQUEST_SCHEMA = "fireviewer.simple-measured-tile-request.v1"
ALGORITHM = "fireviewer.simple-measured-tile-algorithm.v1"
CRS = "EPSG:2154"
TILE_SIZE_M = 500
HALO_M = 10
SOURCE_SIZE = TILE_SIZE_M + 2 * HALO_M
ELEVATION_RESOLUTION_M = 0.5
ELEVATION_SOURCE_SIZE = int(SOURCE_SIZE / ELEVATION_RESOLUTION_M)
PORTABLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
RECEIPT_NAME = "simple-measured-tile-receipt.v1.json"
EXPECTED_OUTPUTS = frozenset(
    {
        "terrain.fvtg",
        "ground/ground-color.png",
        "ground/ground-color.json",
        "placement/placement-hag-1m.tif",
        "placement/placement-inventory.json",
        "terrain-lod0.usda",
        "terrain-lod1.usda",
        "terrain-lod2.usda",
        "terrain-tile.usda",
        "fixed-terrain-usd.v1.json",
        "scene/scene.usda",
        "scene/scene.done.json",
    }
)


class SimpleMeasuredTileError(ValueError):
    """The mono-tile request or its produced package is invalid."""


@dataclass(frozen=True, slots=True)
class SourceBundle:
    mnt_m: np.ndarray
    mns_m: np.ndarray
    orthophoto_rgb_u8: np.ndarray
    orthophoto_revision: str
    request_sources: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PlacementContext:
    building_footprints: tuple[Mapping[str, Any], ...]
    context_geometries: Mapping[str, tuple[Any, ...]]
    context_features: Mapping[str, tuple[Mapping[str, Any], ...]]
    fixed_asset_placements: tuple[Mapping[str, Any], ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class SimpleMeasuredTilePackage:
    output_root: Path
    terrain: Path
    ground_color: Path
    placement_inventory: Path
    terrain_usd: Path
    scene_usd: Path
    receipt: Path
    reused: bool


ProgressCallback = Callable[[str, Mapping[str, Any]], None]


def _emit_progress(
    callback: ProgressCallback | None,
    phase: str,
    **details: Any,
) -> None:
    if callback is not None:
        callback(phase, details)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SimpleMeasuredTileError(f"Invalid {label}: {error}") from error
    if not isinstance(payload, dict):
        raise SimpleMeasuredTileError(f"{label} must contain one JSON object")
    return payload


def _contract_path() -> Path:
    return Path(__file__).with_name("simple_measured_tile_contract.v1.json")


def _load_contract() -> dict[str, Any]:
    payload = _load_json(_contract_path(), "simple measured tile contract")
    if payload.get("schema") != CONTRACT_SCHEMA or payload.get("status") != "locked":
        raise SimpleMeasuredTileError("Unsupported or unlocked simple tile contract")
    expected = {
        "scope": {
            "tile_count": 1,
            "download": "forbidden",
            "pbr": "forbidden",
            "frontend_backend": "forbidden",
            "usage": "technical_pilot_non_final",
        },
        "source": {
            "crs": CRS,
            "tile_size_m": TILE_SIZE_M,
            "halo_m": HALO_M,
            "elevation_resolution_m": ELEVATION_RESOLUTION_M,
            "elevation_shape": [ELEVATION_SOURCE_SIZE, ELEVATION_SOURCE_SIZE],
            "orthophoto_resolution_m": 1,
            "orthophoto_shape": [SOURCE_SIZE, SOURCE_SIZE],
            "row_order": "north_to_south",
            "mnt_mns": "coregistered_single_band_geotiff",
            "orthophoto": "RGB8_PNG",
            "placement_context": (
                "required_EPSG2154_vector_JSON_with_optional_hash_locked_"
                "fixed_asset_placements"
            ),
            "source_receipts": "required_and_rehashed",
        },
        "ground": {
            "output": "ground-color.png",
            "shape": [500, 500, 3],
            "resolution_m": 1,
            "source_payload_retained": False,
        },
        "placement": {
            "source": "MNT_MNS_0.5m_max_HAG_reduced_to_1m",
            "gpkg": "off",
            "quota": "forbidden",
            "thinning": "forbidden",
            "fixed_assets": (
                "exact catalog asset ID at projected XY with MNT-authored Z and "
                "catalog-native scale"
            ),
        },
    }
    for section, wanted in expected.items():
        if payload.get(section) != wanted:
            raise SimpleMeasuredTileError(
                f"Simple measured tile contract section changed: {section}"
            )
    if payload.get("storage", {}).get("windows_drive") != "D:":
        raise SimpleMeasuredTileError(
            "Simple tile storage contract no longer requires D:"
        )
    if payload.get("scene", {}).get("asset_library_count") != 53:
        raise SimpleMeasuredTileError("Simple tile asset library contract changed")
    if (
        payload.get("scene", {}).get("prototype_bundle")
        != "output_local_or_explicit_shared_immutable"
    ):
        raise SimpleMeasuredTileError("Simple tile prototype bundle contract changed")
    return payload


def _require_d_path(
    value: Path | str,
    label: str,
    *,
    kind: str | None = None,
) -> Path:
    lexical = PureWindowsPath(str(value))
    if lexical.drive and lexical.drive.upper() != "D:":
        raise SimpleMeasuredTileError(f"{label} must stay on D:, got {value}")
    path = Path(value).resolve(strict=kind is not None)
    if os.name == "nt" and path.drive.upper() != "D:":
        raise SimpleMeasuredTileError(f"{label} must stay on D:, got {path}")
    if kind == "file" and not path.is_file():
        raise SimpleMeasuredTileError(f"{label} is not a file: {path}")
    if kind == "directory" and not path.is_dir():
        raise SimpleMeasuredTileError(f"{label} is not a directory: {path}")
    return path


def _inside(root: Path, child: Path, label: str) -> Path:
    try:
        child.relative_to(root)
    except ValueError as error:
        raise SimpleMeasuredTileError(
            f"{label} must stay below portable root"
        ) from error
    if child == root:
        raise SimpleMeasuredTileError(f"{label} cannot be the portable root itself")
    return child


def _integer_origin(value: Sequence[float]) -> tuple[int, int]:
    if len(value) != 2:
        raise SimpleMeasuredTileError("tile origin must contain easting and northing")
    floats = tuple(float(component) for component in value)
    if any(
        not math.isfinite(component) or not component.is_integer()
        for component in floats
    ):
        raise SimpleMeasuredTileError("tile origin must use integer metre coordinates")
    origin = int(floats[0]), int(floats[1])
    if any(component % TILE_SIZE_M for component in origin):
        raise SimpleMeasuredTileError("tile origin must align to the global 500 m grid")
    return origin


def _expected_bounds(origin: tuple[int, int]) -> list[int]:
    west, south = origin
    return [
        west - HALO_M,
        south - HALO_M,
        west + TILE_SIZE_M + HALO_M,
        south + TILE_SIZE_M + HALO_M,
    ]


def _validate_source_receipts(
    *,
    elevation_manifest_path: Path,
    orthophoto_manifest_path: Path,
    mnt_path: Path,
    mns_path: Path,
    orthophoto_path: Path,
    zone_id: str,
    tile_id: str,
    origin: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    elevation = _load_json(elevation_manifest_path, "elevation source receipt")
    orthophoto = _load_json(orthophoto_manifest_path, "orthophoto source receipt")
    bounds = _expected_bounds(origin)
    common = {
        "zone_id": zone_id,
        "tile_id": tile_id,
        "crs": CRS,
        "bounds_l93_m": bounds,
    }
    for label, payload in (("elevation", elevation), ("orthophoto", orthophoto)):
        for key, expected in common.items():
            if payload.get(key) != expected:
                raise SimpleMeasuredTileError(f"{label} source receipt {key} mismatch")
    elevation_grid = elevation.get("grid")
    if not isinstance(elevation_grid, Mapping) or any(
        elevation_grid.get(key) != expected
        for key, expected in {
            "resolution_m": ELEVATION_RESOLUTION_M,
            "width": ELEVATION_SOURCE_SIZE,
            "height": ELEVATION_SOURCE_SIZE,
            "halo_m": HALO_M,
            "row_order": "north_to_south",
        }.items()
    ):
        raise SimpleMeasuredTileError("elevation source receipt grid mismatch")
    orthophoto_grid = orthophoto.get("grid")
    if not isinstance(orthophoto_grid, Mapping) or any(
        orthophoto_grid.get(key) != expected
        for key, expected in {
            "resolution_m": 1,
            "width": SOURCE_SIZE,
            "height": SOURCE_SIZE,
            "halo_m": HALO_M,
            "row_order": "north_to_south",
        }.items()
    ):
        raise SimpleMeasuredTileError("orthophoto source receipt grid mismatch")
    if elevation.get("schema") != "fireviewer.mnt-mns-source-pair.v1":
        raise SimpleMeasuredTileError("Unsupported elevation source receipt")
    if orthophoto.get("schema") != "fireviewer.orthophoto-source.v1":
        raise SimpleMeasuredTileError("Unsupported orthophoto source receipt")
    if (
        elevation_manifest_path.parent != mnt_path.parent
        or mnt_path.parent != mns_path.parent
    ):
        raise SimpleMeasuredTileError(
            "Elevation receipt and rasters must share one directory"
        )
    if orthophoto_manifest_path.parent != orthophoto_path.parent:
        raise SimpleMeasuredTileError(
            "Orthophoto receipt and image must share one directory"
        )
    for role, path in (("mnt", mnt_path), ("mns", mns_path)):
        record = elevation.get(role)
        if not isinstance(record, Mapping) or record.get("file") != path.name:
            raise SimpleMeasuredTileError(f"Elevation receipt lacks exact {role} file")
        if record.get("byte_count") != path.stat().st_size or record.get(
            "sha256"
        ) != _sha256_file(path):
            raise SimpleMeasuredTileError(f"Elevation receipt {role} hash mismatch")
    source = orthophoto.get("source")
    if not isinstance(source, Mapping) or source.get("file") != orthophoto_path.name:
        raise SimpleMeasuredTileError("Orthophoto receipt lacks exact source file")
    if source.get("byte_count") != orthophoto_path.stat().st_size or source.get(
        "sha256"
    ) != _sha256_file(orthophoto_path):
        raise SimpleMeasuredTileError("Orthophoto source hash mismatch")
    provider = orthophoto.get("provider")
    revision = provider.get("revision") if isinstance(provider, Mapping) else None
    if not isinstance(revision, str) or PORTABLE_ID.fullmatch(revision) is None:
        raise SimpleMeasuredTileError("Orthophoto source revision is not stable")
    return elevation, orthophoto


def _read_elevation_pair(
    mnt_path: Path,
    mns_path: Path,
    *,
    origin: tuple[int, int],
    elevation_manifest: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float | str]]:
    expected_transform = Affine(
        ELEVATION_RESOLUTION_M,
        0,
        origin[0] - HALO_M,
        0,
        -ELEVATION_RESOLUTION_M,
        origin[1] + TILE_SIZE_M + HALO_M,
    )
    declared_nodata = elevation_manifest.get("grid", {}).get("nodata")

    def read_raster(label: str, path: Path) -> tuple[np.ndarray, tuple[Any, ...]]:
        with rasterio.open(path) as dataset:
            signature = (
                dataset.width,
                dataset.height,
                dataset.count,
                dataset.crs.to_string() if dataset.crs else None,
                tuple(dataset.transform)[:6],
                dataset.nodata,
            )
            if (
                dataset.width != ELEVATION_SOURCE_SIZE
                or dataset.height != ELEVATION_SOURCE_SIZE
                or dataset.count != 1
                or not _is_lambert93_crs(dataset.crs)
                or not dataset.transform.almost_equals(expected_transform)
                or dataset.nodata != declared_nodata
            ):
                raise SimpleMeasuredTileError(
                    f"{label} raster grid differs from its receipt"
                )
            values = dataset.read(1)
            if (
                not np.all(dataset.read_masks(1) == 255)
                or not np.isfinite(values).all()
            ):
                raise SimpleMeasuredTileError(
                    f"{label} contains nodata or non-finite samples"
                )
            if dataset.nodata is not None and np.any(values == dataset.nodata):
                raise SimpleMeasuredTileError(
                    f"{label} contains declared nodata samples"
                )
            return np.asarray(values, dtype="float64"), signature

    mnt, mnt_signature = read_raster("MNT", mnt_path)
    mns_fallback_reason: str | None = None
    try:
        mns, mns_signature = read_raster("MNS", mns_path)
        if mnt_signature != mns_signature:
            raise SimpleMeasuredTileError("MNT and MNS are not exactly co-registered")
    except (
        OSError,
        RuntimeError,
        SimpleMeasuredTileError,
        rasterio.errors.RasterioError,
    ) as error:
        mns = mnt.copy()
        mns_fallback_reason = str(error)

    canonical_mnt, canonical_mns, diagnostics = canonical_elevation_pair_1m_from_05m(
        mnt, mns
    )
    diagnostics["mns_fallback_applied"] = mns_fallback_reason is not None
    diagnostics["mns_fallback_policy"] = "ground_only_on_mns_source_validation_failure"
    if mns_fallback_reason is not None:
        diagnostics["mns_fallback_reason"] = mns_fallback_reason
    return canonical_mnt, canonical_mns, diagnostics


def _round_divide_signed(values: np.ndarray, divisor: int) -> np.ndarray:
    absolute = np.abs(values)
    rounded = (absolute + divisor // 2) // divisor
    return np.where(values < 0, -rounded, rounded)


def canonical_elevation_pair_1m_from_05m(
    mnt_05m: np.ndarray, mns_05m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float | str]]:
    """Reduce native 0.5 m elevation while retaining every measured crown peak."""

    mnt = np.asarray(mnt_05m, dtype="float64")
    mns = np.asarray(mns_05m, dtype="float64")
    expected_shape = (ELEVATION_SOURCE_SIZE, ELEVATION_SOURCE_SIZE)
    if (
        mnt.shape != expected_shape
        or mns.shape != expected_shape
        or not np.isfinite(mnt).all()
        or not np.isfinite(mns).all()
    ):
        raise SimpleMeasuredTileError(
            "Native elevation pair must be two finite 1040 x 1040 arrays"
        )
    mnt_scaled = mnt * 1000.0
    mns_scaled = mns * 1000.0
    mnt_mm = np.where(
        mnt_scaled >= 0.0, np.floor(mnt_scaled + 0.5), np.ceil(mnt_scaled - 0.5)
    ).astype("int64")
    mns_mm = np.where(
        mns_scaled >= 0.0, np.floor(mns_scaled + 0.5), np.ceil(mns_scaled - 0.5)
    ).astype("int64")
    native_delta_mm = mns_mm - mnt_mm
    block_mnt_sum = mnt_mm.reshape(SOURCE_SIZE, 2, SOURCE_SIZE, 2).sum(
        axis=(1, 3), dtype="int64"
    )
    mnt_1m_mm = _round_divide_signed(block_mnt_sum, 4)
    hag_1m_mm = native_delta_mm.reshape(SOURCE_SIZE, 2, SOURCE_SIZE, 2).max(axis=(1, 3))
    mns_1m_mm = mnt_1m_mm + hag_1m_mm
    diagnostics: dict[str, int | float | str] = {
        "native_resolution_m": ELEVATION_RESOLUTION_M,
        "placement_resolution_m": 1,
        "hag_reducer": "maximum_of_four_canonical_0.5m_deltas",
        "minimum_native_delta_mm": int(native_delta_mm.min()),
        "negative_native_sample_count": int(np.count_nonzero(native_delta_mm < 0)),
        "negative_native_below_500mm_count": int(
            np.count_nonzero(native_delta_mm < -500)
        ),
    }
    return mnt_1m_mm / 1000.0, mns_1m_mm / 1000.0, diagnostics


def _is_lambert93_crs(value: Any) -> bool:
    """Accept authoritative EPSG:2154 or the exact IGN WMS fallback definition."""

    if value is None:
        return False
    try:
        if value.to_epsg() == 2154:
            return True
        parameters = value.to_dict()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False
    expected_keys = {
        "proj",
        "lat_0",
        "lon_0",
        "lat_1",
        "lat_2",
        "x_0",
        "y_0",
        "ellps",
        "units",
        "no_defs",
    }
    if set(parameters) != expected_keys:
        return False
    exact = {
        "proj": "lcc",
        "lat_0": 46.5,
        "lon_0": 3,
        "lat_1": 49,
        "lat_2": 44,
        "x_0": 700000,
        "y_0": 6600000,
        "units": "m",
        "no_defs": True,
    }
    if parameters.get("ellps") not in {"WGS84", "GRS80"}:
        return False
    return all(parameters.get(key) == expected for key, expected in exact.items())


def _read_orthophoto(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.load()
            if (
                image.format != "PNG"
                or image.mode != "RGB"
                or image.size != (SOURCE_SIZE, SOURCE_SIZE)
            ):
                raise SimpleMeasuredTileError(
                    "Orthophoto must be one exact 520 x 520 RGB8 PNG"
                )
            return np.asarray(image, dtype="uint8").copy()
    except OSError as error:
        raise SimpleMeasuredTileError(f"Invalid orthophoto PNG: {error}") from error


def _load_sources(
    *,
    mnt_path: Path,
    mns_path: Path,
    orthophoto_path: Path,
    elevation_manifest_path: Path,
    orthophoto_manifest_path: Path,
    zone_id: str,
    tile_id: str,
    origin: tuple[int, int],
) -> SourceBundle:
    elevation, orthophoto = _validate_source_receipts(
        elevation_manifest_path=elevation_manifest_path,
        orthophoto_manifest_path=orthophoto_manifest_path,
        mnt_path=mnt_path,
        mns_path=mns_path,
        orthophoto_path=orthophoto_path,
        zone_id=zone_id,
        tile_id=tile_id,
        origin=origin,
    )
    mnt, mns, elevation_diagnostics = _read_elevation_pair(
        mnt_path, mns_path, origin=origin, elevation_manifest=elevation
    )
    rgb = _read_orthophoto(orthophoto_path)
    return SourceBundle(
        mnt_m=mnt,
        mns_m=mns,
        orthophoto_rgb_u8=rgb,
        orthophoto_revision=str(orthophoto["provider"]["revision"]),
        request_sources={
            "elevation_receipt_sha256": _sha256_file(elevation_manifest_path),
            "mnt_sha256": _sha256_file(mnt_path),
            "mns_sha256": _sha256_file(mns_path),
            "orthophoto_receipt_sha256": _sha256_file(orthophoto_manifest_path),
            "orthophoto_sha256": _sha256_file(orthophoto_path),
            "orthophoto_revision": str(orthophoto["provider"]["revision"]),
            "elevation_reduction": elevation_diagnostics,
        },
    )


def _load_placement_context(
    path: Path,
    *,
    origin: tuple[int, int],
) -> PlacementContext:
    payload = _load_json(path, "placement context")
    if payload.get("schema") != "fireviewer.placement-context-input.v1":
        raise SimpleMeasuredTileError("Unsupported placement context schema")
    if payload.get("crs") != CRS:
        raise SimpleMeasuredTileError("Placement context must use EPSG:2154")
    supplied_origin = payload.get("tile_origin_l93_m")
    if supplied_origin is not None and supplied_origin != list(origin):
        raise SimpleMeasuredTileError("Placement context tile origin mismatch")
    expected_processing_bounds = _expected_bounds(origin)
    supplied_processing_bounds = payload.get("processing_bounds_l93_m")
    if (
        supplied_processing_bounds is not None
        and supplied_processing_bounds != expected_processing_bounds
    ):
        raise SimpleMeasuredTileError("Placement context processing bounds mismatch")
    supplied_core_bounds = payload.get("core_bounds_l93_m")
    expected_core_bounds = [
        origin[0],
        origin[1],
        origin[0] + TILE_SIZE_M,
        origin[1] + TILE_SIZE_M,
    ]
    if (
        supplied_core_bounds is not None
        and supplied_core_bounds != expected_core_bounds
    ):
        raise SimpleMeasuredTileError("Placement context core bounds mismatch")
    footprints = payload.get("building_footprints")
    geometries = payload.get("context_geometries")
    features = payload.get("context_features", {})
    if not isinstance(footprints, list):
        raise SimpleMeasuredTileError(
            "Placement context building_footprints must be a list"
        )
    if not isinstance(geometries, Mapping):
        raise SimpleMeasuredTileError(
            "Placement context context_geometries must be an object"
        )
    if not isinstance(features, Mapping):
        raise SimpleMeasuredTileError(
            "Placement context context_features must be an object"
        )
    allowed = {"vegetation", "roads", "rail", "water"}
    unknown = sorted(set(geometries) - allowed)
    if unknown:
        raise SimpleMeasuredTileError(
            f"Placement context contains unknown geometry keys: {unknown}"
        )
    normalized: dict[str, tuple[Any, ...]] = {}
    for key in sorted(allowed):
        values = geometries.get(key, [])
        if not isinstance(values, list):
            raise SimpleMeasuredTileError(
                f"Placement context geometry collection must be a list: {key}"
            )
        normalized[key] = tuple(values)
    allowed_feature_roles = {
        "vegetation",
        "roads",
        "rail",
        "hydro_lines",
        "hydro_surfaces",
    }
    unknown_features = sorted(set(features) - allowed_feature_roles)
    if unknown_features:
        raise SimpleMeasuredTileError(
            f"Placement context contains unknown feature roles: {unknown_features}"
        )
    normalized_features: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for role in sorted(allowed_feature_roles):
        values = features.get(role, [])
        if not isinstance(values, list) or any(
            not isinstance(value, Mapping) for value in values
        ):
            raise SimpleMeasuredTileError(
                f"Placement context feature collection must be a list: {role}"
            )
        normalized_features[role] = tuple(values)
    fixed_assets = payload.get("fixed_asset_placements", [])
    if not isinstance(fixed_assets, list):
        raise SimpleMeasuredTileError(
            "Placement context fixed_asset_placements must be a list"
        )
    try:
        normalized_fixed_assets = validate_projected_placements(
            fixed_assets,
            tile_origin_l93_m=origin,
        )
    except FixedAssetPlacementError as error:
        raise SimpleMeasuredTileError(
            f"Placement context fixed assets are invalid: {error}"
        ) from error
    return PlacementContext(
        building_footprints=tuple(footprints),
        context_geometries=normalized,
        context_features=normalized_features,
        fixed_asset_placements=normalized_fixed_assets,
        sha256=_sha256_file(path),
    )


def canonical_mnt_normal_halo_mm(mnt_north_to_south_m: np.ndarray) -> np.ndarray:
    """Reproduce the proven 1 m pixel-centre to 2 m vertex sampler exactly."""

    values = np.asarray(mnt_north_to_south_m, dtype="float64")
    if values.shape != (SOURCE_SIZE, SOURCE_SIZE) or not np.isfinite(values).all():
        raise SimpleMeasuredTileError("MNT source must be one finite 520 x 520 grid")
    south_to_north = np.flipud(values)
    scaled = south_to_north * 1000.0
    pixels_mm = np.where(
        scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5)
    ).astype("int64")
    summed = (
        pixels_mm[7:512:2, 7:512:2]
        + pixels_mm[8:513:2, 7:512:2]
        + pixels_mm[7:512:2, 8:513:2]
        + pixels_mm[8:513:2, 8:513:2]
    )
    averaged = np.where(summed >= 0, (summed + 2) // 4, -((-summed + 2) // 4))
    if averaged.shape != (253, 253):
        raise AssertionError("Canonical MNT normal halo shape changed")
    if (
        int(averaged.min()) < np.iinfo("int32").min
        or int(averaged.max()) > np.iinfo("int32").max
    ):
        raise SimpleMeasuredTileError("Canonical MNT exceeds signed int32 millimetres")
    return averaged.astype("<i4")


def _pipeline_file_hashes() -> dict[str, str]:
    paths = {
        "orchestrator": Path(__file__),
        "orchestrator_contract": _contract_path(),
        "fixed_terrain_grid": BLENDER_ROOT / "fixed_terrain_grid.py",
        "fixed_terrain_grid_contract": BLENDER_ROOT
        / "fixed_terrain_grid_contract.v1.json",
        "orthophoto_ground_texture": BLENDER_ROOT / "orthophoto_ground_texture.py",
        "orthophoto_ground_texture_contract": BLENDER_ROOT
        / "orthophoto_ground_texture_contract.v1.json",
        "placement_inventory": BLENDER_ROOT / "mns_mnt_placement_inventory.py",
        "placement_contract": BLENDER_ROOT / "mns_mnt_placement_contract.v1.json",
        "fixed_terrain_usd": OMNIVERSE_ROOT / "fixed_terrain_usd.py",
        "fixed_terrain_usd_contract": OMNIVERSE_ROOT
        / "fixed_terrain_usd_contract.v1.json",
        "measured_scene_usd": OMNIVERSE_ROOT / "build_measured_scene_usd.py",
        "measured_scene_usd_contract": OMNIVERSE_ROOT
        / "measured_scene_usd_contract.v1.json",
    }
    return {
        name: _sha256_file(path.resolve(strict=True))
        for name, path in sorted(paths.items())
    }


def _request_identity(
    *,
    zone_id: str,
    tile_id: str,
    origin: tuple[int, int],
    sources: SourceBundle,
    placement_context: PlacementContext,
    asset_library: Path,
    asset_roots: Mapping[str, Path],
    asset_bundle_root: Path | None,
    portable_root: Path,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "algorithm": ALGORITHM,
        "zone_id": zone_id,
        "tile_id": tile_id,
        "crs": CRS,
        "tile_origin_l93_m": list(origin),
        "core_bounds_l93_m": [
            origin[0],
            origin[1],
            origin[0] + TILE_SIZE_M,
            origin[1] + TILE_SIZE_M,
        ],
        "sources": {
            **dict(sorted(sources.request_sources.items())),
            "placement_context_sha256": placement_context.sha256,
        },
        "asset_library_sha256": _sha256_file(asset_library),
        "asset_root_names": sorted(asset_roots),
        "usage": "technical_pilot_non_final",
        "mns_fallback_policy": "ground_only_on_hag_validation_failure",
        "pipeline_files": _pipeline_file_hashes(),
    }
    if asset_bundle_root is not None:
        request["prototype_bundle"] = {
            "scope": "explicit_shared",
            "portable_path": asset_bundle_root.relative_to(portable_root).as_posix(),
        }
    return request


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    return {
        "path": relative,
        "byte_count": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _is_local_prototype_artifact(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    if len(parts) < 4 or parts[:2] != ("scene", "prototypes"):
        return False
    asset_id = parts[2]
    if PORTABLE_ID.fullmatch(asset_id) is None:
        return False
    if len(parts) == 4:
        return parts[3] in {"source.usd", "source.usda", "prototype.usda"}
    return (
        len(parts) == 5
        and parts[3] == "textures"
        and parts[4].casefold().endswith(".png")
        and PORTABLE_ID.fullmatch(parts[4][:-4]) is not None
    )


def _output_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != RECEIPT_NAME and ".part" not in path.name
    )
    relative_names = {path.relative_to(root).as_posix() for path in files}
    missing = sorted(EXPECTED_OUTPUTS - relative_names)
    unknown = sorted(
        relative
        for relative in relative_names - EXPECTED_OUTPUTS
        if not _is_local_prototype_artifact(relative)
    )
    if missing or unknown:
        raise SimpleMeasuredTileError(
            f"Simple tile output set mismatch; missing={missing}, unknown={unknown}"
        )
    return {path.relative_to(root).as_posix(): _artifact(path, root) for path in files}


def _write_receipt(
    root: Path,
    *,
    request: Mapping[str, Any],
    terrain: FixedTerrainTile,
    placement_inventory: Mapping[str, Any],
    scene_receipt: Mapping[str, Any],
    placement_source: Mapping[str, Any],
) -> Path:
    outputs = _output_artifacts(root)
    without_hash: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "algorithm": ALGORITHM,
        "status": "technical_pilot_non_final",
        "accepted_final": False,
        "build_id": _sha256_bytes(canonical_json_bytes(request)),
        "request": request,
        "terrain": {
            "z_origin_mm": terrain.z_origin_mm,
            "source_grid_sha256": terrain.source_grid_sha256.hex(),
            "normal_halo_sha256": terrain.normal_halo_sha256.hex(),
        },
        "ground": {
            "output": "ground-color.png",
            "shape": [500, 500, 3],
            "resolution_m": 1,
            "source_payload_retained": False,
        },
        "placement": {
            "build_id": placement_inventory["build_id"],
            "inventory_sha256": placement_inventory["inventory_sha256"],
            "building_valid_count": placement_inventory["buildings"]["valid_count"],
            "tree_valid_count": placement_inventory["trees"]["valid_count"],
            "context_asset_valid_count": placement_inventory["context_assets"][
                "valid_count"
            ],
            "quota_applied": False,
            "thinning_applied": False,
            "source": dict(placement_source),
        },
        "scene": {
            "build_id": scene_receipt["build_id"],
            "status": scene_receipt["status"],
            "accepted_final": False,
        },
        "outputs": outputs,
    }
    receipt = dict(without_hash)
    receipt["receipt_sha256"] = _sha256_bytes(canonical_json_bytes(without_hash))
    destination = root / RECEIPT_NAME
    destination.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return destination


def _validate_selected_assets(
    scene_receipt: Mapping[str, Any],
    asset_library: Path,
    asset_roots: Mapping[str, Path],
) -> None:
    library = _load_json(asset_library, "asset library")
    assets = library.get("assets")
    if not isinstance(assets, list):
        raise SimpleMeasuredTileError("Asset library has no asset array")
    indexed = {
        str(asset.get("asset_id")): asset
        for asset in assets
        if isinstance(asset, Mapping) and isinstance(asset.get("asset_id"), str)
    }
    prototypes = scene_receipt.get("prototypes")
    if not isinstance(prototypes, list):
        raise SimpleMeasuredTileError("Measured scene receipt has no prototype list")

    def validate_artifact(
        asset_id: str,
        asset: Mapping[str, Any],
        prototype: Mapping[str, Any],
        *,
        role: str,
        receipt_key: str,
    ) -> None:
        artifact = asset.get(role)
        bundled = prototype.get(receipt_key)
        if not isinstance(artifact, Mapping) or not isinstance(bundled, Mapping):
            raise SimpleMeasuredTileError(
                f"Selected asset has no sealed {role} artifact: {asset_id}"
            )
        logical_root = artifact.get("root")
        relative = artifact.get("path")
        if logical_root not in asset_roots or not isinstance(relative, str):
            raise SimpleMeasuredTileError(
                f"Selected asset {role} root/path is invalid: {asset_id}"
            )
        portable = PurePosixPath(relative)
        if portable.is_absolute() or ".." in portable.parts or "\\" in relative:
            raise SimpleMeasuredTileError(
                f"Selected asset {role} path escapes its root: {asset_id}"
            )
        target = (asset_roots[str(logical_root)] / Path(*portable.parts)).resolve(
            strict=True
        )
        try:
            target.relative_to(asset_roots[str(logical_root)])
        except ValueError as error:
            raise SimpleMeasuredTileError(
                f"Selected asset {role} escapes its root: {asset_id}"
            ) from error
        expected_hash = artifact.get("sha256")
        expected_bytes = artifact.get("byte_count")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes <= 0
            or target.stat().st_size != expected_bytes
            or _sha256_file(target) != expected_hash
            or bundled.get("sha256") != expected_hash
            or bundled.get("byte_count") != expected_bytes
        ):
            raise SimpleMeasuredTileError(
                f"Selected asset {role} hash mismatch: {asset_id}"
            )

    for prototype in prototypes:
        asset_id = prototype.get("asset_id") if isinstance(prototype, Mapping) else None
        asset = indexed.get(str(asset_id))
        if not isinstance(prototype, Mapping) or not isinstance(asset, Mapping):
            raise SimpleMeasuredTileError(
                f"Selected asset is absent from catalogue: {asset_id}"
            )
        validate_artifact(
            str(asset_id), asset, prototype, role="usd", receipt_key="source_usd"
        )
        validate_artifact(
            str(asset_id), asset, prototype, role="texture", receipt_key="texture"
        )


def validate_simple_measured_tile_package(
    output_root: Path | str,
    *,
    expected_request: Mapping[str, Any],
    asset_library: Path,
    asset_roots: Mapping[str, Path],
) -> dict[str, Any]:
    root = _require_d_path(output_root, "tile output root", kind="directory")
    receipt_path = root / RECEIPT_NAME
    receipt = _load_json(receipt_path, "simple tile receipt")
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("algorithm") != ALGORITHM:
        raise SimpleMeasuredTileError(
            "Simple tile receipt schema or algorithm mismatch"
        )
    if (
        receipt.get("status") != "technical_pilot_non_final"
        or receipt.get("accepted_final") is not False
    ):
        raise SimpleMeasuredTileError(
            "Simple tile receipt grants an unsupported status"
        )
    supplied_hash = receipt.get("receipt_sha256")
    without_hash = dict(receipt)
    without_hash.pop("receipt_sha256", None)
    if supplied_hash != _sha256_bytes(canonical_json_bytes(without_hash)):
        raise SimpleMeasuredTileError("Simple tile receipt hash mismatch")
    if receipt.get("request") != expected_request:
        raise SimpleMeasuredTileError(
            "Existing tile was produced from a different request"
        )
    if receipt.get("build_id") != _sha256_bytes(canonical_json_bytes(expected_request)):
        raise SimpleMeasuredTileError("Simple tile build identity mismatch")
    observed_outputs = _output_artifacts(root)
    if receipt.get("outputs") != observed_outputs:
        raise SimpleMeasuredTileError("Simple tile output hashes differ from receipt")

    terrain = read_fixed_terrain(root / "terrain.fvtg")
    origin = tuple(int(value) for value in expected_request["tile_origin_l93_m"])
    if terrain.tile_origin_mm != (origin[0] * 1000, origin[1] * 1000):
        raise SimpleMeasuredTileError("FVTG origin differs from request")
    validate_fixed_terrain_usd_package(root)
    inventory = _load_json(
        root / "placement" / "placement-inventory.json", "placement inventory"
    )
    validate_inventory(inventory)
    _hag, hag_metadata = read_hag_1m(root / "placement" / "placement-hag-1m.tif")
    if hag_metadata["raw_sha256"] != inventory["hag"]["raw_sha256"]:
        raise SimpleMeasuredTileError("HAG raster differs from placement inventory")
    scene_receipt = validate_measured_scene_package(root / "scene")
    _validate_selected_assets(scene_receipt, asset_library, asset_roots)
    if receipt.get("placement", {}).get("build_id") != inventory.get("build_id"):
        raise SimpleMeasuredTileError("Placement build identity differs")
    if receipt.get("scene", {}).get("build_id") != scene_receipt.get("build_id"):
        raise SimpleMeasuredTileError("Scene build identity differs")
    return receipt


def _package(root: Path, *, reused: bool) -> SimpleMeasuredTilePackage:
    return SimpleMeasuredTilePackage(
        output_root=root,
        terrain=root / "terrain.fvtg",
        ground_color=root / "ground" / "ground-color.png",
        placement_inventory=root / "placement" / "placement-inventory.json",
        terrain_usd=root / "terrain-tile.usda",
        scene_usd=root / "scene" / "scene.usda",
        receipt=root / RECEIPT_NAME,
        reused=reused,
    )


def produce_simple_measured_tile(
    *,
    mnt_05m: Path | str,
    mns_05m: Path | str,
    orthophoto_1m: Path | str,
    elevation_source_receipt: Path | str,
    orthophoto_source_receipt: Path | str,
    placement_context: Path | str,
    asset_library: Path | str,
    asset_roots: Mapping[str, Path | str],
    portable_root: Path | str,
    output_root: Path | str,
    zone_id: str,
    tile_id: str,
    tile_origin_l93_m: Sequence[float],
    asset_bundle_root: Path | str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SimpleMeasuredTilePackage:
    """Build or verify exactly one technical pilot tile, with no network access."""

    _load_contract()
    if not isinstance(zone_id, str) or not zone_id.strip():
        raise SimpleMeasuredTileError("zone_id must be non-empty")
    zone_id = zone_id.strip()
    if not isinstance(tile_id, str) or PORTABLE_ID.fullmatch(tile_id) is None:
        raise SimpleMeasuredTileError("tile_id must be one portable identifier")
    origin = _integer_origin(tile_origin_l93_m)
    portable = _require_d_path(portable_root, "portable root", kind="directory")
    destination = _inside(
        portable, _require_d_path(output_root, "tile output root"), "tile output root"
    )
    shared_bundle: Path | None = None
    if asset_bundle_root is not None:
        shared_bundle = _inside(
            portable,
            _require_d_path(asset_bundle_root, "shared asset bundle root"),
            "shared asset bundle root",
        )
        if shared_bundle == destination or shared_bundle.is_relative_to(destination):
            raise SimpleMeasuredTileError(
                "Shared asset bundle root must be outside the tile output"
            )
    inputs = {
        "mnt": _require_d_path(mnt_05m, "MNT input", kind="file"),
        "mns": _require_d_path(mns_05m, "MNS input", kind="file"),
        "orthophoto": _require_d_path(orthophoto_1m, "orthophoto input", kind="file"),
        "elevation_receipt": _require_d_path(
            elevation_source_receipt, "elevation source receipt", kind="file"
        ),
        "orthophoto_receipt": _require_d_path(
            orthophoto_source_receipt, "orthophoto source receipt", kind="file"
        ),
        "placement_context": _require_d_path(
            placement_context, "placement context", kind="file"
        ),
        "asset_library": _require_d_path(asset_library, "asset library", kind="file"),
    }
    roots: dict[str, Path] = {}
    for name, raw_path in sorted(asset_roots.items()):
        if (
            not isinstance(name, str)
            or PORTABLE_ID.fullmatch(name) is None
            or name in roots
        ):
            raise SimpleMeasuredTileError(
                "Asset root names must be unique portable identifiers"
            )
        roots[name] = _inside(
            portable,
            _require_d_path(raw_path, f"asset root {name}", kind="directory"),
            f"asset root {name}",
        )
    if not roots:
        raise SimpleMeasuredTileError("At least one explicit asset root is required")
    _inside(portable, inputs["asset_library"], "asset library")
    sources = _load_sources(
        mnt_path=inputs["mnt"],
        mns_path=inputs["mns"],
        orthophoto_path=inputs["orthophoto"],
        elevation_manifest_path=inputs["elevation_receipt"],
        orthophoto_manifest_path=inputs["orthophoto_receipt"],
        zone_id=zone_id,
        tile_id=tile_id,
        origin=origin,
    )
    context = _load_placement_context(inputs["placement_context"], origin=origin)
    request = _request_identity(
        zone_id=zone_id,
        tile_id=tile_id,
        origin=origin,
        sources=sources,
        placement_context=context,
        asset_library=inputs["asset_library"],
        asset_roots=roots,
        asset_bundle_root=shared_bundle,
        portable_root=portable,
    )
    if destination.exists():
        if not destination.is_dir():
            raise SimpleMeasuredTileError("Existing tile output is not a directory")
        validate_simple_measured_tile_package(
            destination,
            expected_request=request,
            asset_library=inputs["asset_library"],
            asset_roots=roots,
        )
        _emit_progress(
            progress_callback,
            "tile_reused",
            tile_id=tile_id,
            build_id=_load_json(destination / RECEIPT_NAME, "simple tile receipt")[
                "build_id"
            ],
        )
        return _package(destination, reused=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.simple-measured-tile.part")
    if staging.exists():
        if (
            staging.parent != destination.parent
            or staging.name != f".{destination.name}.simple-measured-tile.part"
        ):
            raise SimpleMeasuredTileError("Unsafe simple tile staging path")
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        normal_halo_mm = canonical_mnt_normal_halo_mm(sources.mnt_m)
        terrain = compile_fixed_terrain_from_canonical_mm(
            normal_halo_mm, tile_origin_l93_m=origin
        )
        write_fixed_terrain(terrain, staging / "terrain.fvtg")
        _emit_progress(
            progress_callback,
            "terrain_compiled",
            tile_id=tile_id,
            byte_count=(staging / "terrain.fvtg").stat().st_size,
            lod0_vertices=129 * 129,
            lod1_vertices=33 * 33,
            lod2_vertices=9 * 9,
        )

        core_bounds = [
            origin[0],
            origin[1],
            origin[0] + TILE_SIZE_M,
            origin[1] + TILE_SIZE_M,
        ]
        ground_window = compile_aligned_window(
            sources.orthophoto_rgb_u8,
            transform=Affine(
                1, 0, origin[0] - HALO_M, 0, -1, origin[1] + TILE_SIZE_M + HALO_M
            ),
            crs=CRS,
            core_bounds_l93_m=core_bounds,
            orthophoto_source_manifest_sha256=sources.request_sources[
                "orthophoto_receipt_sha256"
            ],
            orthophoto_revision=sources.orthophoto_revision,
        )
        write_tile_outputs(slice_tile(ground_window, core_bounds), staging / "ground")
        _emit_progress(
            progress_callback,
            "ground_texture_baked",
            tile_id=tile_id,
            byte_count=(staging / "ground" / "ground-color.png").stat().st_size,
            width=500,
            height=500,
        )

        elevation_reduction = sources.request_sources.get("elevation_reduction", {})
        source_mns_fallback = (
            isinstance(elevation_reduction, Mapping)
            and elevation_reduction.get("mns_fallback_applied") is True
        )
        placement_source: dict[str, Any]
        if source_mns_fallback:
            reason = str(elevation_reduction.get("mns_fallback_reason", "invalid MNS"))
            placement_source = {
                "mode": "degraded_mns_fallback",
                "degraded": True,
                "reason": reason,
                "behavior": "MNT used as MNS; no measured height objects inferred",
            }
            _emit_progress(
                progress_callback,
                "placement_mns_fallback",
                tile_id=tile_id,
                reason=reason,
            )
            placement = build_placement_inventory(
                sources.mnt_m,
                sources.mns_m,
                tile_origin_l93_m=origin,
                zone_id=zone_id,
                building_footprints=(),
                context_geometries=context.context_geometries,
                context_features=context.context_features,
                fixed_asset_placements=context.fixed_asset_placements,
            )
        else:
            placement_source = {
                "mode": "measured_mns_minus_mnt",
                "degraded": False,
            }
            try:
                placement = build_placement_inventory(
                    sources.mnt_m,
                    sources.mns_m,
                    tile_origin_l93_m=origin,
                    zone_id=zone_id,
                    building_footprints=context.building_footprints,
                    context_geometries=context.context_geometries,
                    context_features=context.context_features,
                    fixed_asset_placements=context.fixed_asset_placements,
                )
            except PlacementInventoryError as error:
                message = str(error)
                if "corrupt or misaligned" not in message:
                    raise
                placement_source = {
                    "mode": "degraded_mns_fallback",
                    "degraded": True,
                    "reason": message,
                    "behavior": "MNT used as MNS; no measured height objects inferred",
                }
                _emit_progress(
                    progress_callback,
                    "placement_mns_fallback",
                    tile_id=tile_id,
                    reason=message,
                )
                placement = build_placement_inventory(
                    sources.mnt_m,
                    sources.mnt_m,
                    tile_origin_l93_m=origin,
                    zone_id=zone_id,
                    building_footprints=(),
                    context_geometries=context.context_geometries,
                    context_features=context.context_features,
                    fixed_asset_placements=context.fixed_asset_placements,
                )
        write_placement_outputs(
            staging / "placement", placement, tile_origin_l93_m=origin, gpkg="off"
        )
        _emit_progress(
            progress_callback,
            "placement_measured",
            tile_id=tile_id,
            building_count=placement.inventory["buildings"]["valid_count"],
            tree_count=placement.inventory["trees"]["valid_count"],
            context_asset_count=placement.inventory["context_assets"]["valid_count"],
            hag_maximum_cm=placement.inventory["hag"]["maximum_cm"],
        )

        export_fixed_terrain_usd(
            staging / "terrain.fvtg",
            staging / "ground" / "ground-color.png",
            staging / "ground" / "ground-color.json",
            staging,
            tile_id=tile_id,
            zone_origin_l93_m=origin,
        )
        _emit_progress(
            progress_callback,
            "terrain_usd_exported",
            tile_id=tile_id,
            root_stage="terrain-tile.usda",
        )
        _emit_progress(
            progress_callback,
            "prototype_bundle_started",
            tile_id=tile_id,
        )
        scene = build_measured_scene_usd(
            TerrainReference(
                staging / "terrain-tile.usda", origin, terrain.z_origin_mm
            ),
            staging / "placement" / "placement-inventory.json",
            inputs["asset_library"],
            staging / "scene",
            portable_root=portable,
            asset_roots=roots,
            asset_bundle_root=shared_bundle,
            usage="technical_pilot_non_final",
            prototype_progress_callback=lambda completed, total, asset_id: (
                _emit_progress(
                    progress_callback,
                    "prototype_bundle_progress",
                    tile_id=tile_id,
                    prototype_completed=completed,
                    prototype_total=total,
                    asset_id=asset_id,
                )
            ),
        )
        scene_receipt = validate_measured_scene_package(scene.output_root)
        _emit_progress(
            progress_callback,
            "scene_usd_built",
            tile_id=tile_id,
            building_count=scene_receipt["reconciliation"]["buildings"][
                "instance_count"
            ],
            tree_count=scene_receipt["reconciliation"]["trees"]["instance_count"],
        )
        _write_receipt(
            staging,
            request=request,
            terrain=terrain,
            placement_inventory=placement.inventory,
            scene_receipt=scene_receipt,
            placement_source=placement_source,
        )
        validate_simple_measured_tile_package(
            staging,
            expected_request=request,
            asset_library=inputs["asset_library"],
            asset_roots=roots,
        )
        _emit_progress(
            progress_callback,
            "tile_staging_validated",
            tile_id=tile_id,
        )
        os.replace(staging, destination)
    except Exception:
        if staging.exists() and staging.parent == destination.parent:
            shutil.rmtree(staging)
        raise
    validate_simple_measured_tile_package(
        destination,
        expected_request=request,
        asset_library=inputs["asset_library"],
        asset_roots=roots,
    )
    _emit_progress(
        progress_callback,
        "tile_published",
        tile_id=tile_id,
        build_id=_load_json(destination / RECEIPT_NAME, "simple tile receipt")[
            "build_id"
        ],
    )
    return _package(destination, reused=False)


def _parse_asset_roots(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path or name in result:
            raise SimpleMeasuredTileError(
                "--asset-root must be unique NAME=PATH values"
            )
        result[name] = Path(raw_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Produce exactly one measured tile from local MNT/MNS/orthophoto sources"
    )
    parser.add_argument("--mnt-05m", type=Path, required=True)
    parser.add_argument("--mns-05m", type=Path, required=True)
    parser.add_argument("--orthophoto-1m", type=Path, required=True)
    parser.add_argument("--elevation-source-receipt", type=Path, required=True)
    parser.add_argument("--orthophoto-source-receipt", type=Path, required=True)
    parser.add_argument("--placement-context", type=Path, required=True)
    parser.add_argument("--asset-library", type=Path, required=True)
    parser.add_argument(
        "--asset-root",
        action="append",
        required=True,
        help="Logical root mapping NAME=PATH",
    )
    parser.add_argument("--portable-root", type=Path, required=True)
    parser.add_argument(
        "--asset-bundle-root",
        type=Path,
        help="Optional immutable shared prototype bundle inside portable-root",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--zone-id", required=True)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument(
        "--tile-origin",
        type=float,
        nargs=2,
        required=True,
        metavar=("EASTING", "NORTHING"),
    )
    parser.add_argument("--execute", action="store_true", required=True)
    options = parser.parse_args(argv)
    package = produce_simple_measured_tile(
        mnt_05m=options.mnt_05m,
        mns_05m=options.mns_05m,
        orthophoto_1m=options.orthophoto_1m,
        elevation_source_receipt=options.elevation_source_receipt,
        orthophoto_source_receipt=options.orthophoto_source_receipt,
        placement_context=options.placement_context,
        asset_library=options.asset_library,
        asset_roots=_parse_asset_roots(options.asset_root),
        portable_root=options.portable_root,
        asset_bundle_root=options.asset_bundle_root,
        output_root=options.output_root,
        zone_id=options.zone_id,
        tile_id=options.tile_id,
        tile_origin_l93_m=options.tile_origin,
    )
    receipt = _load_json(package.receipt, "simple tile receipt")
    print(
        json.dumps(
            {
                "schema": RECEIPT_SCHEMA,
                "status": receipt["status"],
                "accepted_final": False,
                "build_id": receipt["build_id"],
                "output_root": str(package.output_root),
                "reused": package.reused,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ALGORITHM",
    "CONTRACT_SCHEMA",
    "RECEIPT_SCHEMA",
    "SimpleMeasuredTileError",
    "SimpleMeasuredTilePackage",
    "canonical_mnt_normal_halo_mm",
    "main",
    "produce_simple_measured_tile",
    "validate_simple_measured_tile_package",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
