"""Prepare one atomic measured-source bundle for the simple terrain pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pyogrio.raw
import rasterio
import requests
from affine import Affine
from fixed_asset_placement import (
    FixedAssetPlacementError,
    validate_projected_placements,
)
from fixed_asset_placement import (
    canonical_json_bytes as canonical_fixed_asset_bytes,
)
from fixed_asset_placement import (
    schema_sha256 as fixed_asset_schema_sha256,
)
from mid_distance_roads import resolve_road_width_m
from PIL import Image
from prepare_simple_measured_zone_context import load_zone_context
from rasterio.io import MemoryFile
from shapely import from_wkb
from shapely.geometry import box, mapping, shape

SCHEMA = "fireviewer.simple-measured-tile-source-bundle.v1"
CRS = "EPSG:2154"
TILE_SIZE_M = 500
HALO_M = 10
ELEVATION_SIZE = 1040
ORTHOPHOTO_SIZE = 520
NODATA = -9999.0
WMS_ENDPOINT = "https://data.geopf.fr/wms-r"
MNT_LAYER = "IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93"
MNS_LAYER = "IGNF_LIDAR-HD_MNS_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93"
ORTHOPHOTO_LAYER = "ORTHOIMAGERY.ORTHOPHOTOS"
HASH_LENGTH = 64
_HTTP_LOCAL = threading.local()


class SimpleMeasuredTileSourceError(RuntimeError):
    """A remote response or local context cannot form a locked source bundle."""


@dataclass(frozen=True, slots=True)
class PreparedSources:
    root: Path
    mnt: Path
    mns: Path
    orthophoto: Path
    elevation_receipt: Path
    orthophoto_receipt: Path
    placement_context: Path
    reused: bool


HttpGet = Callable[[str], bytes]
ProgressCallback = Callable[[str, Mapping[str, Any]], None]


def _emit_progress(
    callback: ProgressCallback | None,
    phase: str,
    **details: Any,
) -> None:
    if callback is not None:
        callback(phase, details)


def _canonical_bytes(value: Any) -> bytes:
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


def _require_d_path(value: Path | str, label: str, *, exists: bool) -> Path:
    lexical_drive = PureWindowsPath(str(value)).drive.upper()
    if lexical_drive and lexical_drive != "D:":
        raise SimpleMeasuredTileSourceError(f"{label} must remain on D:; got {value}")
    try:
        path = Path(value).resolve(strict=exists)
    except OSError as error:
        raise SimpleMeasuredTileSourceError(f"Missing {label}: {value}") from error
    if os.name == "nt" and path.drive.upper() != "D:":
        raise SimpleMeasuredTileSourceError(f"{label} must remain on D:; got {path}")
    return path


def _origin(value: Sequence[float]) -> tuple[int, int]:
    if len(value) != 2:
        raise SimpleMeasuredTileSourceError("Tile origin needs easting and northing")
    numbers = tuple(float(component) for component in value)
    if any(
        not math.isfinite(component) or not component.is_integer()
        for component in numbers
    ):
        raise SimpleMeasuredTileSourceError("Tile origin must use integer metres")
    result = int(numbers[0]), int(numbers[1])
    if any(component % TILE_SIZE_M for component in result):
        raise SimpleMeasuredTileSourceError("Tile origin must align to the 500 m grid")
    return result


def _bounds(origin: tuple[int, int]) -> tuple[int, int, int, int]:
    return (
        origin[0] - HALO_M,
        origin[1] - HALO_M,
        origin[0] + TILE_SIZE_M + HALO_M,
        origin[1] + TILE_SIZE_M + HALO_M,
    )


def wms_url(layer: str, bounds: Sequence[int], size: int, image_format: str) -> str:
    """Return the canonical IGN WMS-R request used in receipts."""

    parameters = [
        ("SERVICE", "WMS"),
        ("VERSION", "1.3.0"),
        ("REQUEST", "GetMap"),
        ("LAYERS", layer),
        ("STYLES", ""),
        ("CRS", CRS),
        ("BBOX", ",".join(str(int(value)) for value in bounds)),
        ("WIDTH", str(size)),
        ("HEIGHT", str(size)),
        ("FORMAT", image_format),
        ("TRANSPARENT", "FALSE"),
    ]
    return f"{WMS_ENDPOINT}?{urlencode(parameters)}"


def _default_http_get(url: str) -> bytes:
    session = getattr(_HTTP_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(
            {"User-Agent": "FireViewer/simple-measured-tile-sources-v1"}
        )
        _HTTP_LOCAL.session = session
    response = session.get(
        url,
        timeout=(20, 180),
    )
    response.raise_for_status()
    if not response.content:
        raise SimpleMeasuredTileSourceError(f"Empty WMS response: {url}")
    return bytes(response.content)


def _elevation_array(payload: bytes, label: str) -> np.ndarray:
    try:
        # IGN WMS responses may omit an internal affine.  The requested BBOX,
        # dimensions and CRS are canonical and are written into the sealed
        # output immediately afterwards, so this one warning is expected.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=rasterio.errors.NotGeoreferencedWarning
            )
            with MemoryFile(payload) as memory, memory.open() as dataset:
                if (
                    dataset.width != ELEVATION_SIZE
                    or dataset.height != ELEVATION_SIZE
                    or dataset.count < 1
                ):
                    raise SimpleMeasuredTileSourceError(
                        f"{label} response grid is not 1040 x 1040"
                    )
                values = np.asarray(dataset.read(1), dtype="float32")
                mask = dataset.read_masks(1)
    except (OSError, rasterio.errors.RasterioError) as error:
        raise SimpleMeasuredTileSourceError(
            f"Invalid {label} GeoTIFF response"
        ) from error
    if not np.isfinite(values).all() or not np.all(mask == 255):
        raise SimpleMeasuredTileSourceError(f"{label} response contains nodata")
    return values


def _orthophoto_image(payload: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(payload)) as source:
            source.load()
            if source.size != (ORTHOPHOTO_SIZE, ORTHOPHOTO_SIZE):
                raise SimpleMeasuredTileSourceError(
                    "Orthophoto response grid is not 520 x 520"
                )
            return source.convert("RGB")
    except OSError as error:
        raise SimpleMeasuredTileSourceError(
            "Invalid orthophoto image response"
        ) from error


def _write_elevation(path: Path, values: np.ndarray, origin: tuple[int, int]) -> None:
    transform = Affine(
        0.5,
        0,
        origin[0] - HALO_M,
        0,
        -0.5,
        origin[1] + TILE_SIZE_M + HALO_M,
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=ELEVATION_SIZE,
        height=ELEVATION_SIZE,
        count=1,
        dtype="float32",
        crs=CRS,
        transform=transform,
        nodata=NODATA,
        compress="DEFLATE",
        predictor=3,
    ) as dataset:
        dataset.write(np.asarray(values, dtype="float32"), 1)


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(_canonical_bytes(payload) + b"\n")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SimpleMeasuredTileSourceError(f"Invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise SimpleMeasuredTileSourceError(f"{label} must be a JSON object")
    return value


def _read_layer(
    gpkg: Path,
    layer: str,
    bounds: tuple[int, int, int, int],
    columns: Sequence[str],
) -> list[tuple[Any, dict[str, Any]]]:
    metadata, _fids, geometry_wkb, fields = pyogrio.raw.read(
        gpkg,
        layer=layer,
        bbox=bounds,
        columns=list(columns),
    )
    names = [str(name) for name in metadata["fields"]]
    result: list[tuple[Any, dict[str, Any]]] = []
    for index, encoded in enumerate(geometry_wkb):
        geometry = from_wkb(encoded)
        properties: dict[str, Any] = {}
        for name, values in zip(names, fields, strict=True):
            value = values[index]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and math.isnan(value):
                value = None
            properties[name] = value
        result.append((geometry, properties))
    return result


def _building_footprints(
    snapshot: Mapping[str, Any], bounds: tuple[int, int, int, int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    features = snapshot.get("features")
    if snapshot.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise SimpleMeasuredTileSourceError(
            "Building snapshot is not a FeatureCollection"
        )
    window = box(*bounds)
    filtered_features: list[dict[str, Any]] = []
    footprints: list[dict[str, Any]] = []
    seen: set[str] = set()
    for feature in features:
        if not isinstance(feature, Mapping) or not isinstance(
            feature.get("geometry"), Mapping
        ):
            raise SimpleMeasuredTileSourceError(
                "Building snapshot feature is malformed"
            )
        geometry = shape(feature["geometry"])
        if not geometry.intersects(window):
            continue
        properties = feature.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        source_id = properties.get("cleabs") or feature.get("id")
        if not isinstance(source_id, str) or not source_id or source_id in seen:
            raise SimpleMeasuredTileSourceError(
                "Building snapshot has invalid source IDs"
            )
        seen.add(source_id)
        normalized_geometry = mapping(geometry)
        filtered_features.append(
            {
                "type": "Feature",
                "id": feature.get("id", source_id),
                "properties": dict(properties),
                "geometry": normalized_geometry,
            }
        )
        footprints.append(
            {
                "source_id": source_id,
                "geometry": normalized_geometry,
                "properties": dict(properties),
            }
        )
    filtered_features.sort(key=lambda item: str(item["id"]))
    footprints.sort(key=lambda item: item["source_id"])
    return filtered_features, footprints


def _placement_context(
    *,
    origin: tuple[int, int],
    gpkg: Path,
    gpkg_manifest: Path,
    footprints: list[dict[str, Any]],
    building_snapshot: Path,
    building_snapshot_receipt: Path,
) -> dict[str, Any]:
    bounds = _bounds(origin)
    vegetation: list[Any] = []
    vegetation_features: list[dict[str, Any]] = []
    vegetation_ids: list[str] = []
    for geometry, properties in _read_layer(
        gpkg, "landcover", bounds, ("id", "code_cs")
    ):
        code = properties.get("code_cs")
        if isinstance(code, str) and code.startswith("CS2.1.1"):
            source_id = str(properties.get("id"))
            normalized_geometry = mapping(geometry)
            vegetation.append(normalized_geometry)
            vegetation_ids.append(source_id)
            vegetation_features.append(
                {
                    "source_id": source_id,
                    "geometry": normalized_geometry,
                    "properties": dict(properties),
                }
            )

    roads: list[Any] = []
    road_features: list[dict[str, Any]] = []
    road_ids: list[str] = []
    road_widths: list[dict[str, Any]] = []
    for geometry, properties in _read_layer(
        gpkg,
        "roads",
        bounds,
        ("cleabs", "nature", "importance", "largeur_de_chaussee"),
    ):
        width, method = resolve_road_width_m(properties)
        source_id = str(properties.get("cleabs"))
        roads.append(
            mapping(geometry.buffer(width / 2, cap_style="flat", join_style="mitre"))
        )
        road_ids.append(source_id)
        road_widths.append({"source_id": source_id, "width_m": width, "method": method})
        road_features.append(
            {
                "source_id": source_id,
                "geometry": mapping(geometry),
                "properties": {**dict(properties), "resolved_width_m": width},
            }
        )

    rail_records = _read_layer(gpkg, "railways", bounds, ("cleabs",))
    rail = [mapping(geometry) for geometry, _properties in rail_records]
    rail_ids = [str(properties.get("cleabs")) for _geometry, properties in rail_records]
    hydro_line_records = _read_layer(gpkg, "hydro_lines", bounds, ("cleabs",))
    hydro_surface_records = _read_layer(gpkg, "hydro_surfaces", bounds, ("cleabs",))
    water = [
        mapping(geometry)
        for geometry, _properties in (*hydro_line_records, *hydro_surface_records)
    ]
    water_ids = [
        str(properties.get("cleabs"))
        for _geometry, properties in (*hydro_line_records, *hydro_surface_records)
    ]
    rail_features = [
        {
            "source_id": str(properties.get("cleabs")),
            "geometry": mapping(geometry),
            "properties": dict(properties),
        }
        for geometry, properties in rail_records
    ]
    hydro_line_features = [
        {
            "source_id": str(properties.get("cleabs")),
            "geometry": mapping(geometry),
            "properties": dict(properties),
        }
        for geometry, properties in hydro_line_records
    ]
    hydro_surface_features = [
        {
            "source_id": str(properties.get("cleabs")),
            "geometry": mapping(geometry),
            "properties": dict(properties),
        }
        for geometry, properties in hydro_surface_records
    ]
    return {
        "schema": "fireviewer.placement-context-input.v1",
        "crs": CRS,
        "tile_origin_l93_m": list(origin),
        "processing_bounds_l93_m": list(bounds),
        "building_footprints": footprints,
        "context_geometries": {
            "vegetation": vegetation,
            "roads": roads,
            "rail": rail,
            "water": water,
        },
        "context_features": {
            "vegetation": sorted(
                vegetation_features, key=lambda item: item["source_id"]
            ),
            "roads": sorted(road_features, key=lambda item: item["source_id"]),
            "rail": sorted(rail_features, key=lambda item: item["source_id"]),
            "hydro_lines": sorted(
                hydro_line_features, key=lambda item: item["source_id"]
            ),
            "hydro_surfaces": sorted(
                hydro_surface_features, key=lambda item: item["source_id"]
            ),
        },
        "provenance": {
            "ground_context_package_sha256": _sha256_file(gpkg),
            "ground_context_manifest_sha256": _sha256_file(gpkg_manifest),
            "building_snapshot_sha256": _sha256_file(building_snapshot),
            "building_snapshot_receipt_sha256": _sha256_file(building_snapshot_receipt),
            "landcover_filter": "code_cs startswith CS2.1.1",
            "landcover_feature_ids": sorted(vegetation_ids),
            "road_width_policy": "mid_distance_roads.resolve_road_width_m.v1",
            "road_feature_ids": sorted(road_ids),
            "road_widths": sorted(road_widths, key=lambda item: item["source_id"]),
            "rail_feature_ids": sorted(rail_ids),
            "water_feature_ids": sorted(water_ids),
            "feature_counts": {
                "buildings": len(footprints),
                "vegetation": len(vegetation),
                "roads": len(roads),
                "rail": len(rail),
                "water": len(water),
            },
        },
    }


def _placement_context_from_zone(
    *,
    origin: tuple[int, int],
    zone_context: Mapping[str, Any],
    zone_context_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    bounds = _bounds(origin)
    declared_bounds = zone_context.get("bounds_l93_m")
    if not isinstance(declared_bounds, list) or len(declared_bounds) != 4:
        raise SimpleMeasuredTileSourceError("Zone context bounds are invalid")
    if (
        declared_bounds[0] > bounds[0]
        or declared_bounds[1] > bounds[1]
        or declared_bounds[2] < bounds[2]
        or declared_bounds[3] < bounds[3]
    ):
        raise SimpleMeasuredTileSourceError(
            "Zone context does not cover the tile processing halo"
        )
    layers = zone_context.get("layers")
    if not isinstance(layers, Mapping):
        raise SimpleMeasuredTileSourceError("Zone context layers are invalid")
    window = box(*bounds)

    def selected(role: str) -> list[tuple[Any, Mapping[str, Any], str]]:
        record = layers.get(role)
        features = record.get("features") if isinstance(record, Mapping) else None
        if not isinstance(features, list):
            raise SimpleMeasuredTileSourceError(
                f"Zone context layer is invalid: {role}"
            )
        result: list[tuple[Any, Mapping[str, Any], str]] = []
        for feature in features:
            if not isinstance(feature, Mapping) or not isinstance(
                feature.get("geometry"), Mapping
            ):
                raise SimpleMeasuredTileSourceError(
                    f"Zone context feature is invalid: {role}"
                )
            geometry = shape(feature["geometry"])
            if not geometry.intersects(window):
                continue
            source_id = feature.get("id")
            if not isinstance(source_id, str) or not source_id:
                raise SimpleMeasuredTileSourceError(
                    f"Zone context feature ID is invalid: {role}"
                )
            properties = feature.get("properties")
            properties = properties if isinstance(properties, Mapping) else {}
            result.append((geometry, properties, source_id))
        return sorted(result, key=lambda item: item[2])

    building_features = [
        {
            "type": "Feature",
            "id": source_id,
            "properties": dict(properties),
            "geometry": mapping(geometry),
        }
        for geometry, properties, source_id in selected("buildings")
    ]
    filtered_features, footprints = _building_footprints(
        {"type": "FeatureCollection", "features": building_features}, bounds
    )

    vegetation_records = selected("vegetation")
    vegetation = [
        mapping(geometry) for geometry, _properties, _id in vegetation_records
    ]

    road_records = selected("roads")
    roads: list[Any] = []
    road_widths: list[dict[str, Any]] = []
    for geometry, properties, source_id in road_records:
        width, method = resolve_road_width_m(properties)
        roads.append(
            mapping(geometry.buffer(width / 2, cap_style="flat", join_style="mitre"))
        )
        road_widths.append({"source_id": source_id, "width_m": width, "method": method})

    rail_records = selected("rail")
    hydro_line_records = selected("hydro_lines")
    hydro_surface_records = selected("hydro_surfaces")
    rail = [mapping(geometry) for geometry, _properties, _id in rail_records]
    water = [
        mapping(geometry)
        for geometry, _properties, _id in (
            *hydro_line_records,
            *hydro_surface_records,
        )
    ]
    source_revision = zone_context.get("source", {}).get("revision")
    context = {
        "schema": "fireviewer.placement-context-input.v1",
        "crs": CRS,
        "tile_origin_l93_m": list(origin),
        "processing_bounds_l93_m": list(bounds),
        "building_footprints": footprints,
        "context_geometries": {
            "vegetation": vegetation,
            "roads": roads,
            "rail": rail,
            "water": water,
        },
        "context_features": {
            "vegetation": [
                {
                    "source_id": source_id,
                    "geometry": mapping(geometry),
                    "properties": dict(properties),
                }
                for geometry, properties, source_id in vegetation_records
            ],
            "roads": [
                {
                    "source_id": source_id,
                    "geometry": mapping(geometry),
                    "properties": dict(properties),
                }
                for geometry, properties, source_id in road_records
            ],
            "rail": [
                {
                    "source_id": source_id,
                    "geometry": mapping(geometry),
                    "properties": dict(properties),
                }
                for geometry, properties, source_id in rail_records
            ],
            "hydro_lines": [
                {
                    "source_id": source_id,
                    "geometry": mapping(geometry),
                    "properties": dict(properties),
                }
                for geometry, properties, source_id in hydro_line_records
            ],
            "hydro_surfaces": [
                {
                    "source_id": source_id,
                    "geometry": mapping(geometry),
                    "properties": dict(properties),
                }
                for geometry, properties, source_id in hydro_surface_records
            ],
        },
        "provenance": {
            "zone_context_file_sha256": _sha256_file(zone_context_path),
            "zone_context_content_sha256": zone_context["content_sha256"],
            "zone_context_source_revision": source_revision,
            "vegetation_policy": "BDTOPO_V3:zone_de_vegetation semantic confirmation",
            "vegetation_feature_ids": [item[2] for item in vegetation_records],
            "road_width_policy": "mid_distance_roads.resolve_road_width_m.v1",
            "road_feature_ids": [item[2] for item in road_records],
            "road_widths": road_widths,
            "rail_feature_ids": [item[2] for item in rail_records],
            "water_feature_ids": [
                item[2] for item in (*hydro_line_records, *hydro_surface_records)
            ],
            "feature_counts": {
                "buildings": len(footprints),
                "vegetation": len(vegetation),
                "roads": len(roads),
                "rail": len(rail),
                "water": len(water),
            },
        },
    }
    return filtered_features, footprints, context


def _validate_existing(
    root: Path,
    *,
    zone_id: str,
    tile_id: str,
    origin: tuple[int, int],
    fixed_asset_placements: Sequence[Mapping[str, Any]] = (),
) -> None:
    elevation = _read_json(root / "elevation-source-05m.json", "elevation receipt")
    orthophoto = _read_json(root / "orthophoto-source.json", "orthophoto receipt")
    context = _read_json(root / "placement-context.json", "placement context")
    expected = {
        "zone_id": zone_id,
        "tile_id": tile_id,
        "bounds_l93_m": list(_bounds(origin)),
    }
    for receipt in (elevation, orthophoto):
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise SimpleMeasuredTileSourceError(
                "Existing source receipt identity differs"
            )
    if context.get("tile_origin_l93_m") != list(origin):
        raise SimpleMeasuredTileSourceError(
            "Existing placement context identity differs"
        )
    try:
        existing_fixed = validate_projected_placements(
            context.get("fixed_asset_placements", []),
            tile_origin_l93_m=origin,
        )
    except FixedAssetPlacementError as error:
        raise SimpleMeasuredTileSourceError(
            f"Existing fixed asset placements are invalid: {error}"
        ) from error
    if canonical_fixed_asset_bytes(existing_fixed) != canonical_fixed_asset_bytes(
        fixed_asset_placements
    ):
        raise SimpleMeasuredTileSourceError(
            "Existing fixed asset placements differ from the request"
        )
    for record, file_name in (
        (elevation.get("mnt"), "mnt-05m.tif"),
        (elevation.get("mns"), "mns-05m.tif"),
        (orthophoto.get("source"), "orthophoto-1m.png"),
    ):
        path = root / file_name
        if (
            not isinstance(record, Mapping)
            or record.get("file") != file_name
            or not path.is_file()
            or record.get("byte_count") != path.stat().st_size
            or record.get("sha256") != _sha256_file(path)
        ):
            raise SimpleMeasuredTileSourceError(f"Existing source changed: {file_name}")


def prepare_sources(
    *,
    output_root: Path | str,
    zone_id: str,
    tile_id: str,
    tile_origin_l93_m: Sequence[float],
    elevation_revision: str,
    orthophoto_revision: str,
    ground_context_gpkg: Path | str | None = None,
    ground_context_manifest: Path | str | None = None,
    building_snapshot: Path | str | None = None,
    building_snapshot_receipt: Path | str | None = None,
    zone_context: Path | str | None = None,
    fixed_asset_placements: Sequence[Mapping[str, Any]] = (),
    http_get: HttpGet | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PreparedSources:
    """Download, validate and atomically publish one complete source bundle."""

    origin = _origin(tile_origin_l93_m)
    try:
        normalized_fixed_assets = validate_projected_placements(
            fixed_asset_placements,
            tile_origin_l93_m=origin,
        )
    except FixedAssetPlacementError as error:
        raise SimpleMeasuredTileSourceError(
            f"Fixed asset placements are invalid: {error}"
        ) from error
    if not zone_id.strip() or not tile_id.strip():
        raise SimpleMeasuredTileSourceError("Zone and tile IDs are required")
    for label, revision in (
        ("elevation", elevation_revision),
        ("orthophoto", orthophoto_revision),
    ):
        if not revision or revision != revision.strip():
            raise SimpleMeasuredTileSourceError(f"Stable {label} revision is required")
    destination = _require_d_path(output_root, "source output root", exists=False)
    legacy_values = (
        ground_context_gpkg,
        ground_context_manifest,
        building_snapshot,
        building_snapshot_receipt,
    )
    if zone_context is not None and any(value is not None for value in legacy_values):
        raise SimpleMeasuredTileSourceError(
            "Use either one zone context or the complete legacy context inputs"
        )
    if zone_context is None and any(value is None for value in legacy_values):
        raise SimpleMeasuredTileSourceError(
            "One zone context or all four legacy context inputs are required"
        )
    zone_context_path: Path | None = None
    zone_payload: dict[str, Any] | None = None
    gpkg: Path | None = None
    gpkg_manifest: Path | None = None
    snapshot_path: Path | None = None
    snapshot_receipt_path: Path | None = None
    if zone_context is not None:
        zone_context_path = _require_d_path(zone_context, "zone context", exists=True)
        zone_payload = load_zone_context(zone_context_path)
        if zone_payload.get("zone_id") != zone_id:
            raise SimpleMeasuredTileSourceError("Zone context identity differs")
    else:
        gpkg = _require_d_path(ground_context_gpkg, "ground context GPKG", exists=True)
        gpkg_manifest = _require_d_path(
            ground_context_manifest, "ground context manifest", exists=True
        )
        snapshot_path = _require_d_path(
            building_snapshot, "building snapshot", exists=True
        )
        snapshot_receipt_path = _require_d_path(
            building_snapshot_receipt, "building snapshot receipt", exists=True
        )
    if destination.exists():
        _validate_existing(
            destination,
            zone_id=zone_id,
            tile_id=tile_id,
            origin=origin,
            fixed_asset_placements=normalized_fixed_assets,
        )
        _emit_progress(
            progress_callback,
            "sources_reused",
            tile_id=tile_id,
            byte_count=sum(
                path.stat().st_size for path in destination.iterdir() if path.is_file()
            ),
        )
        return _prepared(destination, reused=True)

    snapshot: dict[str, Any] | None = None
    if zone_payload is None:
        assert snapshot_path is not None and snapshot_receipt_path is not None
        snapshot = _read_json(snapshot_path, "building snapshot")
        snapshot_receipt = _read_json(
            snapshot_receipt_path, "building snapshot receipt"
        )
        response = snapshot_receipt.get("response")
        if (
            snapshot_receipt.get("schema")
            != "fireviewer.building-confirmation-source.v1"
            or snapshot_receipt.get("role") != "semantic_confirmation_only"
            or snapshot_receipt.get("placement_measurement") != "MNS-MNT"
            or not isinstance(response, Mapping)
            or response.get("byte_count") != snapshot_path.stat().st_size
            or response.get("sha256") != _sha256_file(snapshot_path)
        ):
            raise SimpleMeasuredTileSourceError("Building snapshot receipt is invalid")

    getter = http_get or _default_http_get
    bounds = _bounds(origin)
    urls = {
        "mnt": wms_url(MNT_LAYER, bounds, ELEVATION_SIZE, "image/tiff"),
        "mns": wms_url(MNS_LAYER, bounds, ELEVATION_SIZE, "image/tiff"),
        "orthophoto": wms_url(ORTHOPHOTO_LAYER, bounds, ORTHOPHOTO_SIZE, "image/png"),
    }
    responses: dict[str, bytes] = {}
    for name in ("mnt", "mns", "orthophoto"):
        _emit_progress(
            progress_callback,
            f"download_{name}_started",
            tile_id=tile_id,
            request_url=urls[name],
        )
        responses[name] = getter(urls[name])
        _emit_progress(
            progress_callback,
            f"download_{name}_completed",
            tile_id=tile_id,
            byte_count=len(responses[name]),
            response_sha256=_sha256_bytes(responses[name]),
        )
    mnt = _elevation_array(responses["mnt"], "MNT")
    mns = _elevation_array(responses["mns"], "MNS")
    ortho = _orthophoto_image(responses["orthophoto"])
    _emit_progress(
        progress_callback,
        "sources_decoded",
        tile_id=tile_id,
        elevation_shape=list(mnt.shape),
        orthophoto_size=list(ortho.size),
        mnt_minimum_m=round(float(mnt.min()), 3),
        mnt_maximum_m=round(float(mnt.max()), 3),
        mns_minimum_m=round(float(mns.min()), 3),
        mns_maximum_m=round(float(mns.max()), 3),
    )
    if zone_payload is not None:
        assert zone_context_path is not None
        filtered_features, footprints, context = _placement_context_from_zone(
            origin=origin,
            zone_context=zone_payload,
            zone_context_path=zone_context_path,
        )
    else:
        assert snapshot is not None
        filtered_features, footprints = _building_footprints(snapshot, bounds)
        context = None
    _emit_progress(
        progress_callback,
        "tile_context_prepared",
        tile_id=tile_id,
        building_feature_count=len(filtered_features),
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.simple-sources.part")
    if staging.exists():
        if staging.parent != destination.parent:
            raise SimpleMeasuredTileSourceError("Unsafe source staging path")
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        mnt_path = staging / "mnt-05m.tif"
        mns_path = staging / "mns-05m.tif"
        ortho_path = staging / "orthophoto-1m.png"
        _write_elevation(mnt_path, mnt, origin)
        _write_elevation(mns_path, mns, origin)
        ortho.save(ortho_path, format="PNG", optimize=False)
        grid = {
            "resolution_m": 0.5,
            "width": ELEVATION_SIZE,
            "height": ELEVATION_SIZE,
            "halo_m": HALO_M,
            "row_order": "north_to_south",
            "affine": [0.5, 0, bounds[0], 0, -0.5, bounds[3]],
            "nodata": NODATA,
        }
        elevation_receipt = {
            "schema": "fireviewer.mnt-mns-source-pair.v1",
            "status": "downloaded_coregistered_verified",
            "zone_id": zone_id,
            "tile_id": tile_id,
            "crs": CRS,
            "bounds_l93_m": list(bounds),
            "grid": grid,
            "provider": {
                "service": "IGN Geoplateforme WMS-R",
                "revision": elevation_revision,
            },
            "coregistration": {
                "same_shape": True,
                "same_crs": True,
                "same_affine": True,
                "finite_samples": True,
            },
            "mnt": _elevation_record(
                mnt_path, MNT_LAYER, urls["mnt"], responses["mnt"], mnt
            ),
            "mns": _elevation_record(
                mns_path, MNS_LAYER, urls["mns"], responses["mns"], mns
            ),
        }
        _json_write(staging / "elevation-source-05m.json", elevation_receipt)
        _json_write(
            staging / "orthophoto-source.json",
            {
                "schema": "fireviewer.orthophoto-source.v1",
                "status": "downloaded_verified",
                "zone_id": zone_id,
                "tile_id": tile_id,
                "crs": CRS,
                "bounds_l93_m": list(bounds),
                "grid": {
                    "resolution_m": 1,
                    "width": ORTHOPHOTO_SIZE,
                    "height": ORTHOPHOTO_SIZE,
                    "halo_m": HALO_M,
                    "row_order": "north_to_south",
                },
                "provider": {
                    "service": "IGN Geoplateforme WMS-R",
                    "revision": orthophoto_revision,
                    "layer": ORTHOPHOTO_LAYER,
                    "request_url": urls["orthophoto"],
                    "response_byte_count": len(responses["orthophoto"]),
                    "response_sha256": _sha256_bytes(responses["orthophoto"]),
                },
                "source": _file_record(ortho_path),
            },
        )
        subset_path = staging / "bdtopo-buildings.geojson"
        _json_write(
            subset_path,
            {"type": "FeatureCollection", "features": filtered_features},
        )
        _json_write(
            staging / "building-source.json",
            {
                "schema": "fireviewer.building-confirmation-source.v1",
                "provider": "IGN",
                "product": "BD TOPO V3",
                "layer": "BDTOPO_V3:batiment",
                "crs": CRS,
                "bounds_l93_m": list(bounds),
                **(
                    {
                        "source_zone_context_sha256": _sha256_file(zone_context_path),
                        "source_zone_context_content_sha256": zone_payload[
                            "content_sha256"
                        ],
                    }
                    if zone_payload is not None and zone_context_path is not None
                    else {
                        "source_snapshot_sha256": _sha256_file(snapshot_path),
                        "source_snapshot_receipt_sha256": _sha256_file(
                            snapshot_receipt_path
                        ),
                    }
                ),
                "response": {
                    **_file_record(subset_path),
                    "feature_count": len(filtered_features),
                },
                "role": "semantic_confirmation_only",
                "placement_measurement": "MNS-MNT",
                "status": "validated" if filtered_features else "validated_empty",
            },
        )
        if context is None:
            assert (
                gpkg is not None
                and gpkg_manifest is not None
                and snapshot_path is not None
                and snapshot_receipt_path is not None
            )
            context = _placement_context(
                origin=origin,
                gpkg=gpkg,
                gpkg_manifest=gpkg_manifest,
                footprints=footprints,
                building_snapshot=snapshot_path,
                building_snapshot_receipt=snapshot_receipt_path,
            )
        context = dict(context)
        context["fixed_asset_placements"] = [
            dict(placement) for placement in normalized_fixed_assets
        ]
        provenance = context.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise SimpleMeasuredTileSourceError(
                "Placement context provenance is invalid"
            )
        context["provenance"] = {
            **dict(provenance),
            "fixed_asset_placement_schema_sha256": fixed_asset_schema_sha256(),
            "fixed_asset_placement_count": len(normalized_fixed_assets),
            "fixed_asset_placements_sha256": _sha256_bytes(
                canonical_fixed_asset_bytes(normalized_fixed_assets)
            ),
        }
        _json_write(staging / "placement-context.json", context)
        bundle = {
            "schema": SCHEMA,
            "zone_id": zone_id,
            "tile_id": tile_id,
            "tile_origin_l93_m": list(origin),
            "files": {
                name: _file_record(staging / name)
                for name in (
                    "mnt-05m.tif",
                    "mns-05m.tif",
                    "orthophoto-1m.png",
                    "elevation-source-05m.json",
                    "orthophoto-source.json",
                    "bdtopo-buildings.geojson",
                    "building-source.json",
                    "placement-context.json",
                )
            },
        }
        bundle["bundle_sha256"] = _sha256_bytes(_canonical_bytes(bundle))
        _json_write(staging / "simple-measured-tile-sources.v1.json", bundle)
        _validate_existing(
            staging,
            zone_id=zone_id,
            tile_id=tile_id,
            origin=origin,
            fixed_asset_placements=normalized_fixed_assets,
        )
        os.replace(staging, destination)
    except Exception:
        if staging.exists() and staging.parent == destination.parent:
            shutil.rmtree(staging)
        raise
    _validate_existing(
        destination,
        zone_id=zone_id,
        tile_id=tile_id,
        origin=origin,
        fixed_asset_placements=normalized_fixed_assets,
    )
    _emit_progress(
        progress_callback,
        "sources_published",
        tile_id=tile_id,
        byte_count=sum(
            path.stat().st_size for path in destination.iterdir() if path.is_file()
        ),
        bundle_sha256=bundle["bundle_sha256"],
    )
    return _prepared(destination, reused=False)


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "byte_count": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _elevation_record(
    path: Path,
    layer: str,
    request_url: str,
    response: bytes,
    values: np.ndarray,
) -> dict[str, Any]:
    return {
        **_file_record(path),
        "layer": layer,
        "request_url": request_url,
        "response_byte_count": len(response),
        "response_sha256": _sha256_bytes(response),
        "minimum_m": float(values.min()),
        "maximum_m": float(values.max()),
    }


def _prepared(root: Path, *, reused: bool) -> PreparedSources:
    return PreparedSources(
        root=root,
        mnt=root / "mnt-05m.tif",
        mns=root / "mns-05m.tif",
        orthophoto=root / "orthophoto-1m.png",
        elevation_receipt=root / "elevation-source-05m.json",
        orthophoto_receipt=root / "orthophoto-source.json",
        placement_context=root / "placement-context.json",
        reused=reused,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--zone-id", required=True)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--tile-origin", nargs=2, type=float, required=True)
    parser.add_argument("--elevation-revision", required=True)
    parser.add_argument("--orthophoto-revision", required=True)
    parser.add_argument("--zone-context", type=Path)
    parser.add_argument("--ground-context-gpkg", type=Path)
    parser.add_argument("--ground-context-manifest", type=Path)
    parser.add_argument("--building-snapshot", type=Path)
    parser.add_argument("--building-snapshot-receipt", type=Path)
    parser.add_argument("--execute", action="store_true", required=True)
    options = parser.parse_args(argv)
    prepared = prepare_sources(
        output_root=options.output_root,
        zone_id=options.zone_id,
        tile_id=options.tile_id,
        tile_origin_l93_m=options.tile_origin,
        elevation_revision=options.elevation_revision,
        orthophoto_revision=options.orthophoto_revision,
        ground_context_gpkg=options.ground_context_gpkg,
        ground_context_manifest=options.ground_context_manifest,
        building_snapshot=options.building_snapshot,
        building_snapshot_receipt=options.building_snapshot_receipt,
        zone_context=options.zone_context,
    )
    bundle = _read_json(
        prepared.root / "simple-measured-tile-sources.v1.json", "source bundle"
    )
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "bundle_sha256": bundle["bundle_sha256"],
                "output_root": str(prepared.root),
                "reused": prepared.reused,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
