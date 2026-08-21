"""Deterministic 1 m MNS/MNT placement inventory for one 500 m terrain tile.

The module produces measurements, not scene instances.  Buildings are bound to
stable source-footprint identities and vegetation is segmented from measured
MNS-MNT maxima without quotas or thinning.  The 10 m processing halo is used
both for complete measurements and for deterministic cross-tile seam evidence.

Input arrays use conventional north-to-south raster row order.  Their exact
shape is 520 x 520: a 500 m core plus 10 m on every side, at one metre per
sample in EPSG:2154.  The tile origin is the south-west corner of the core.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import rasterio
from fixed_asset_placement import (
    FixedAssetPlacementError,
    validate_projected_placements,
)
from fixed_asset_placement import (
    canonical_json_bytes as canonical_fixed_asset_bytes,
)
from rasterio.features import rasterize
from rasterio.features import shapes as raster_shapes
from rasterio.transform import from_origin
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    find_objects,
    label,
    maximum_filter,
    minimum_filter,
    watershed_ift,
)
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

SCHEMA = "fireviewer.mns-mnt-placement-inventory.v1"
CONTRACT_SCHEMA = "fireviewer.mns-mnt-placement-contract.v1"
ALGORITHM = "fireviewer.mns-mnt-placement-algorithm.v3"
HAG_SCHEMA = "fireviewer.placement-hag-1m.v1"
CRS = "EPSG:2154"
RESOLUTION_M = 1
TILE_SIZE_M = 500
HALO_M = 10
PROCESSING_SIZE = TILE_SIZE_M + HALO_M * 2
CORE_START = HALO_M
CORE_STOP = HALO_M + TILE_SIZE_M
CORE_SHAPE = (TILE_SIZE_M, TILE_SIZE_M)
NODATA_UINT16 = 65_535
MAX_HAG_CM = NODATA_UINT16 - 1
NEGATIVE_HAG_TOLERANCE_CM = 50
NEGATIVE_HAG_HARD_LIMIT_CM = 100
NEGATIVE_HAG_MAX_OUTLIER_COUNT = 32
NEGATIVE_HAG_MAX_OUTLIER_FRACTION = 0.000125
POSITIVE_HAG_MAX_OUTLIER_COUNT = 32
POSITIVE_HAG_MAX_OUTLIER_FRACTION = 0.000125
MIN_HEIGHT_CM = 200
WOODY_CONTEXT_MIN_HEIGHT_CM = 100
MIN_CROWN_AREA_M2 = 4
WOODY_CONTEXT_TALL_MIN_CROWN_AREA_M2 = 1
WOODY_CONTEXT_LOW_MIN_CROWN_AREA_M2 = 2
FLAT_WOODY_PLATEAU_MIN_AREA_M2 = 32
FLAT_WOODY_PLATEAU_MARKER_SPACING_M = 4
_CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)
_CONNECTIVITY_4 = np.array(
    [[False, True, False], [True, True, True], [False, True, False]], dtype=bool
)
_CONTEXT_KEYS = ("buildings", "vegetation", "roads", "rail", "water")
_CONTEXT_FEATURE_ROLES = (
    "vegetation",
    "forest_composition",
    "roads",
    "rail",
    "hydro_lines",
    "hydro_surfaces",
)
_CONTEXT_ASSET_CATEGORY = {
    "roads": "road_equipment",
    "rail": "rail_equipment",
    "hydro_lines": "hydro_equipment",
    "hydro_surfaces": "hydro_equipment",
}


class PlacementInventoryError(ValueError):
    """Raised when inputs or an inventory violate the locked contract."""


class GeoPackageUnavailableError(RuntimeError):
    """Raised when the optional Fiona GeoPackage writer is not installed."""


@dataclass(frozen=True)
class PlacementResult:
    """In-memory canonical outputs for one tile."""

    hag_core_cm: np.ndarray
    inventory: dict[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    """Encode canonical JSON and reject non-finite values."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _contract_sha256() -> str:
    path = Path(__file__).with_name("mns_mnt_placement_contract.v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA:
        raise RuntimeError("MNS/MNT placement contract schema is invalid")
    return _sha256(canonical_json_bytes(payload))


def _stable_id(kind: str, *parts: object) -> str:
    digest = _sha256(canonical_json_bytes([ALGORITHM, kind, *parts]))[:24]
    return f"{kind}-{digest}"


def assert_d_storage_path(path: Path | str) -> Path:
    """Reject C: everywhere and require D: for writes on Windows."""

    destination = Path(path)
    lexical_drive = PureWindowsPath(str(path)).drive.upper()
    if lexical_drive == "C:":
        raise PlacementInventoryError(
            "FireViewer placement outputs are forbidden on C:"
        )
    if os.name == "nt":
        resolved = destination.resolve()
        if resolved.drive.upper() != "D:":
            raise PlacementInventoryError(
                "FireViewer placement outputs must be stored on D:"
            )
    return destination


def _origin(value: Sequence[float]) -> tuple[int, int]:
    if len(value) != 2:
        raise PlacementInventoryError(
            "tile_origin_l93_m must contain easting and northing"
        )
    coordinates = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in coordinates):
        raise PlacementInventoryError("tile origin must contain finite values")
    rounded = tuple(round(component) for component in coordinates)
    if any(
        not math.isclose(component, integer, abs_tol=1e-9)
        for component, integer in zip(coordinates, rounded)
    ):
        raise PlacementInventoryError("tile origin must align to the global 1 m grid")
    return rounded


def _quantize_mm(values: np.ndarray) -> np.ndarray:
    scaled = np.asarray(values, dtype="float64") * 1000.0
    rounded = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    if float(np.max(np.abs(rounded))) > np.iinfo("int32").max:
        raise PlacementInventoryError("MNT/MNS elevation exceeds signed millimetres")
    return rounded.astype("int32")


def _source_grid(
    mnt_m: Any, mns_m: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    mnt = np.asarray(mnt_m, dtype="float64")
    mns = np.asarray(mns_m, dtype="float64")
    expected = (PROCESSING_SIZE, PROCESSING_SIZE)
    if mnt.shape != expected or mns.shape != expected:
        raise PlacementInventoryError(
            f"MNT and MNS must be co-registered arrays with shape {expected}"
        )
    if not np.isfinite(mnt).all() or not np.isfinite(mns).all():
        raise PlacementInventoryError("MNT and MNS must not contain nodata or NaN")
    mnt_mm = _quantize_mm(mnt)
    mns_mm = _quantize_mm(mns)
    delta_mm = mns_mm.astype("int64") - mnt_mm.astype("int64")
    minimum_delta_mm = int(delta_mm.min())
    if minimum_delta_mm < -(NEGATIVE_HAG_HARD_LIMIT_CM * 10):
        raise PlacementInventoryError(
            "MNS lies more than 100 cm below MNT; grids are corrupt or misaligned"
        )
    negative_outlier_count = int(
        np.count_nonzero(delta_mm < -(NEGATIVE_HAG_TOLERANCE_CM * 10))
    )
    negative_outlier_fraction = negative_outlier_count / int(delta_mm.size)
    if (
        negative_outlier_count > NEGATIVE_HAG_MAX_OUTLIER_COUNT
        or negative_outlier_fraction > NEGATIVE_HAG_MAX_OUTLIER_FRACTION
    ):
        raise PlacementInventoryError(
            "MNS has too many samples more than 50 cm below MNT; "
            "grids are corrupt or misaligned"
        )
    negative_sample_count = int(np.count_nonzero(delta_mm < 0))
    delta_cm = (np.maximum(delta_mm, 0) + 5) // 10
    maximum_delta_mm = int(delta_mm.max())
    positive_outliers = delta_cm > MAX_HAG_CM
    positive_outlier_count = int(np.count_nonzero(positive_outliers))
    positive_outlier_fraction = positive_outlier_count / int(delta_mm.size)
    if (
        positive_outlier_count > POSITIVE_HAG_MAX_OUTLIER_COUNT
        or positive_outlier_fraction > POSITIVE_HAG_MAX_OUTLIER_FRACTION
    ):
        raise PlacementInventoryError(
            "MNS-MNT has too many samples above the uint16 centimetre contract; "
            "grids are corrupt or misaligned"
        )
    if positive_outlier_count:
        mns_mm = mns_mm.copy()
        delta_mm = delta_mm.copy()
        delta_cm = delta_cm.copy()
        mns_mm[positive_outliers] = mnt_mm[positive_outliers]
        delta_mm[positive_outliers] = 0
        delta_cm[positive_outliers] = 0
    diagnostics = {
        "minimum_source_delta_mm": minimum_delta_mm,
        "maximum_source_delta_mm_before_repair": maximum_delta_mm,
        "negative_source_sample_count_clamped": negative_sample_count,
        "negative_outlier_below_tolerance_count": negative_outlier_count,
        "negative_outlier_fraction": round(negative_outlier_fraction, 12),
        "positive_uint16_outlier_count_repaired_to_ground": positive_outlier_count,
        "positive_uint16_outlier_fraction": round(positive_outlier_fraction, 12),
    }
    return mnt_mm, mns_mm, delta_cm.astype("uint16"), diagnostics


def _mask(value: Any, *, name: str, default: bool) -> np.ndarray:
    if value is None:
        return np.full((PROCESSING_SIZE, PROCESSING_SIZE), default, dtype=bool)
    result = np.asarray(value)
    if result.shape != (PROCESSING_SIZE, PROCESSING_SIZE):
        raise PlacementInventoryError(
            f"context mask {name!r} must have shape {(PROCESSING_SIZE, PROCESSING_SIZE)}"
        )
    if result.dtype.kind != "b":
        raise PlacementInventoryError(f"context mask {name!r} must be boolean")
    return result.astype(bool, copy=True)


def _mask_sha256(value: np.ndarray) -> str:
    packed = np.packbits(np.asarray(value, dtype="uint8"), bitorder="little")
    return _sha256(packed.tobytes(order="C"))


def _normalise_geometry(value: Any, *, name: str) -> BaseGeometry:
    if isinstance(value, BaseGeometry):
        geometry = value
    else:
        payload = (
            value.get("geometry")
            if isinstance(value, Mapping) and value.get("type") == "Feature"
            else value
        )
        try:
            geometry = shape(payload)
        except Exception as error:
            raise PlacementInventoryError(f"{name} geometry is malformed") from error
    if geometry.is_empty or not geometry.is_valid:
        raise PlacementInventoryError(f"{name} geometry must be non-empty and valid")
    normalise = getattr(geometry, "normalize", None)
    return normalise() if callable(normalise) else geometry


def _geometry_payload(geometry: BaseGeometry) -> dict[str, Any]:
    return json.loads(canonical_json_bytes(mapping(geometry)).decode("utf-8"))


def _geometry_context(
    geometries: Iterable[Any], *, name: str
) -> tuple[list[BaseGeometry], str]:
    normalised = [
        _normalise_geometry(value, name=f"{name}[{index}]")
        for index, value in enumerate(geometries)
    ]
    encoded = sorted(
        canonical_json_bytes(_geometry_payload(item)) for item in normalised
    )
    return normalised, _sha256(b"\n".join(encoded))


def _rasterize_geometries(
    geometries: Iterable[BaseGeometry],
    *,
    transform: Any,
) -> np.ndarray:
    shapes = [(mapping(geometry), 1) for geometry in geometries]
    if not shapes:
        return np.zeros((PROCESSING_SIZE, PROCESSING_SIZE), dtype=bool)
    return rasterize(
        shapes,
        out_shape=(PROCESSING_SIZE, PROCESSING_SIZE),
        transform=transform,
        fill=0,
        default_value=1,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)


def _normalise_properties(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PlacementInventoryError(f"{name} properties must be an object")
    result: dict[str, Any] = {}
    for raw_key, raw_value in sorted(value.items(), key=lambda item: str(item[0])):
        key = str(raw_key)
        if not key:
            raise PlacementInventoryError(f"{name} property key is empty")
        if raw_value is not None and not isinstance(raw_value, (str, int, float, bool)):
            raise PlacementInventoryError(
                f"{name} property {key} is not a canonical scalar"
            )
        if isinstance(raw_value, float) and not math.isfinite(raw_value):
            raise PlacementInventoryError(f"{name} property {key} is not finite")
        result[key] = raw_value
    return result


def _normalise_footprints(
    footprints: Iterable[Mapping[str, Any]],
) -> list[tuple[str, BaseGeometry, str, dict[str, Any]]]:
    result: list[tuple[str, BaseGeometry, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, footprint in enumerate(footprints):
        if not isinstance(footprint, Mapping):
            raise PlacementInventoryError(
                f"building footprint {index} must be an object"
            )
        source_id = footprint.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise PlacementInventoryError(
                f"building footprint {index} requires a non-empty source_id"
            )
        source_id = source_id.strip()
        if source_id in seen:
            raise PlacementInventoryError(f"duplicate building source_id: {source_id}")
        seen.add(source_id)
        geometry = _normalise_geometry(
            footprint.get("geometry"), name=f"building footprint {source_id}"
        )
        if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise PlacementInventoryError(
                f"building footprint {source_id} must be Polygon or MultiPolygon"
            )
        geometry_hash = _sha256(canonical_json_bytes(_geometry_payload(geometry)))
        properties = _normalise_properties(
            footprint.get("properties"), name=f"building footprint {source_id}"
        )
        result.append((source_id, geometry, geometry_hash, properties))
    result.sort(key=lambda item: item[0])
    return result


def _footprint_confirmations(
    footprints: list[tuple[str, BaseGeometry, str, dict[str, Any]]],
    *,
    transform: Any,
) -> tuple[list[dict[str, Any]], str]:
    confirmations: list[dict[str, Any]] = []
    hashes: list[dict[str, str]] = []
    for source_id, geometry, geometry_hash, properties in footprints:
        # Cell-centre ownership avoids double-confirming boundary pixels shared
        # by adjacent cadastral polygons.  The MNS-MNT pixels still author the
        # measured candidate geometry; the footprint only confirms its class.
        confirmation_mask = rasterize(
            [(mapping(geometry), 1)],
            out_shape=(PROCESSING_SIZE, PROCESSING_SIZE),
            transform=transform,
            fill=0,
            default_value=1,
            dtype="uint8",
            all_touched=False,
        ).astype(bool)
        confirmations.append(
            {
                "source_id": source_id,
                "geometry_sha256": geometry_hash,
                "source_properties": properties,
                "mask": confirmation_mask,
            }
        )
        hashes.append(
            {
                "source_id": source_id,
                "geometry_sha256": geometry_hash,
                "source_properties": properties,
            }
        )
    return confirmations, _sha256(canonical_json_bytes(hashes))


def _is_owned(x_m: float, y_m: float, west: int, south: int) -> bool:
    return west <= x_m < west + TILE_SIZE_M and south <= y_m < south + TILE_SIZE_M


def _normalise_context_features(
    features: Mapping[str, Iterable[Mapping[str, Any]]] | None,
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    supplied = dict(features or {})
    unknown = sorted(set(supplied) - set(_CONTEXT_FEATURE_ROLES))
    if unknown:
        raise PlacementInventoryError(f"unknown context feature roles: {unknown}")
    result: dict[str, list[dict[str, Any]]] = {
        role: [] for role in _CONTEXT_FEATURE_ROLES
    }
    hash_records: list[dict[str, Any]] = []
    for role in _CONTEXT_FEATURE_ROLES:
        seen: set[str] = set()
        for index, feature in enumerate(supplied.get(role, ())):
            if not isinstance(feature, Mapping):
                raise PlacementInventoryError(
                    f"context feature {role}[{index}] must be an object"
                )
            source_id = feature.get("source_id")
            if not isinstance(source_id, str) or not source_id.strip():
                raise PlacementInventoryError(
                    f"context feature {role}[{index}] requires source_id"
                )
            source_id = source_id.strip()
            if source_id in seen:
                raise PlacementInventoryError(
                    f"duplicate context feature source_id: {role}/{source_id}"
                )
            seen.add(source_id)
            geometry = _normalise_geometry(
                feature.get("geometry"), name=f"context feature {role}/{source_id}"
            )
            properties = _normalise_properties(
                feature.get("properties"), name=f"context feature {role}/{source_id}"
            )
            geometry_payload = _geometry_payload(geometry)
            record = {
                "source_id": source_id,
                "geometry": geometry,
                "geometry_sha256": _sha256(canonical_json_bytes(geometry_payload)),
                "source_properties": properties,
            }
            result[role].append(record)
            hash_records.append(
                {
                    "role": role,
                    "source_id": source_id,
                    "geometry_sha256": record["geometry_sha256"],
                    "source_properties": properties,
                }
            )
        result[role].sort(key=lambda item: item["source_id"])
    return result, _sha256(canonical_json_bytes(hash_records))


def _geometry_segments(
    geometry: BaseGeometry,
) -> list[tuple[float, float, float, float]]:
    geometry_type = geometry.geom_type
    sequences: list[Any] = []
    if geometry_type in {"LineString", "LinearRing"}:
        sequences = [geometry.coords]
    elif geometry_type == "Polygon":
        sequences = [geometry.exterior.coords]
    elif hasattr(geometry, "geoms"):
        segments: list[tuple[float, float, float, float]] = []
        for child in geometry.geoms:
            segments.extend(_geometry_segments(child))
        return segments
    result: list[tuple[float, float, float, float]] = []
    for coordinates in sequences:
        points = list(coordinates)
        for first, second in pairwise(points):
            result.append(
                (float(first[0]), float(first[1]), float(second[0]), float(second[1]))
            )
    return result


def _geometry_yaw(geometry: BaseGeometry) -> float:
    segments = _geometry_segments(geometry)
    if not segments:
        return 0.0
    x0, y0, x1, y1 = max(
        segments,
        key=lambda segment: (
            (segment[2] - segment[0]) ** 2 + (segment[3] - segment[1]) ** 2,
            segment,
        ),
    )
    yaw = math.atan2(y1 - y0, x1 - x0)
    return (yaw + math.pi / 2.0) % math.pi - math.pi / 2.0


def _geometry_anchor(geometry: BaseGeometry) -> tuple[float, float]:
    if geometry.geom_type in {"LineString", "MultiLineString"}:
        point = geometry.interpolate(0.5, normalized=True)
    elif geometry.geom_type == "Point":
        point = geometry
    else:
        point = geometry.representative_point()
    return float(point.x), float(point.y)


def _ground_at(
    mnt_mm: np.ndarray, *, x_m: float, y_m: float, west: int, south: int
) -> int:
    column = math.floor(x_m - (west - HALO_M))
    row = math.floor((south + TILE_SIZE_M + HALO_M) - y_m)
    if not (0 <= row < PROCESSING_SIZE and 0 <= column < PROCESSING_SIZE):
        raise PlacementInventoryError("context candidate lies outside MNT/MNS window")
    return int(mnt_mm[row, column])


def _context_asset_inventory(
    features: Mapping[str, list[dict[str, Any]]],
    *,
    mnt_mm: np.ndarray,
    west: int,
    south: int,
    fixed_asset_placements: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    halo_only_count = 0
    for role, category in _CONTEXT_ASSET_CATEGORY.items():
        selection_context = {
            "roads": "road",
            "rail": "rail",
            "hydro_lines": "hydro",
            "hydro_surfaces": "hydro",
        }[role]
        for feature in features[role]:
            x_m, y_m = _geometry_anchor(feature["geometry"])
            if not _is_owned(x_m, y_m, west, south):
                halo_only_count += 1
                continue
            candidates.append(
                {
                    "candidate_id": _stable_id(
                        "context-asset", role, feature["source_id"]
                    ),
                    "status": "valid",
                    "reason_codes": [],
                    "asset_category": category,
                    "selection_context": selection_context,
                    "context_role": role,
                    "source_ids": [feature["source_id"]],
                    "source_geometry_sha256": feature["geometry_sha256"],
                    "source_properties": feature["source_properties"],
                    "position_l93_m": [round(x_m, 3), round(y_m, 3)],
                    "ground_elevation_mm": _ground_at(
                        mnt_mm, x_m=x_m, y_m=y_m, west=west, south=south
                    ),
                    "yaw_rad": round(_geometry_yaw(feature["geometry"]), 9),
                    "scale_policy": "catalog_native_uniform",
                    "evidence": "stable_SIG_feature_with_MNT_ground",
                }
            )

    water_features = [*features["hydro_lines"], *features["hydro_surfaces"]]
    for road in features["roads"]:
        for water in water_features:
            crossing = road["geometry"].intersection(water["geometry"])
            if crossing.is_empty:
                continue
            x_m, y_m = _geometry_anchor(crossing)
            if not _is_owned(x_m, y_m, west, south):
                halo_only_count += 1
                continue
            source_ids = [road["source_id"], water["source_id"]]
            candidates.append(
                {
                    "candidate_id": _stable_id(
                        "context-asset", "road_hydro_crossing", *source_ids
                    ),
                    "status": "valid",
                    "reason_codes": [],
                    "asset_category": "drainage_equipment",
                    "selection_context": "road_hydro_crossing",
                    "context_role": "road_hydro_crossing",
                    "source_ids": source_ids,
                    "source_geometry_sha256": _sha256(
                        canonical_json_bytes(_geometry_payload(crossing))
                    ),
                    "source_properties": {
                        "road_nature": road["source_properties"].get("nature"),
                        "water_nature": water["source_properties"].get("nature"),
                    },
                    "position_l93_m": [round(x_m, 3), round(y_m, 3)],
                    "ground_elevation_mm": _ground_at(
                        mnt_mm, x_m=x_m, y_m=y_m, west=west, south=south
                    ),
                    "yaw_rad": round(_geometry_yaw(road["geometry"]), 9),
                    "scale_policy": "catalog_native_uniform",
                    "evidence": "stable_road_hydro_intersection_with_MNT_ground",
                }
            )
    for placement in fixed_asset_placements:
        x_m, y_m = (
            float(placement["position_l93_m"][0]),
            float(placement["position_l93_m"][1]),
        )
        if not _is_owned(x_m, y_m, west, south):
            raise PlacementInventoryError(
                f"fixed placement is not owned by this tile: {placement['placement_id']}"
            )
        candidates.append(
            {
                "candidate_id": _stable_id(
                    "fixed-context-asset", placement["placement_id"]
                ),
                "status": "valid",
                "reason_codes": [],
                "fixed_placement_id": placement["placement_id"],
                "fixed_asset_id": placement["asset_id"],
                "asset_category": placement["asset_category"],
                "selection_context": "fixed_user_coordinate",
                "context_role": "fixed_user_coordinate",
                "source_ids": [placement["placement_id"]],
                "source_geometry_sha256": _sha256(
                    canonical_json_bytes(
                        {
                            "type": "Point",
                            "coordinates": placement["position_l93_m"],
                        }
                    )
                ),
                "source_properties": {
                    "latitude": placement["source_wgs84"][0],
                    "longitude": placement["source_wgs84"][1],
                    "source_crs": "EPSG:4326",
                },
                "position_l93_m": [round(x_m, 3), round(y_m, 3)],
                "ground_elevation_mm": _ground_at(
                    mnt_mm, x_m=x_m, y_m=y_m, west=west, south=south
                ),
                "yaw_rad": placement["yaw_rad"],
                "scale_policy": "catalog_native_uniform",
                "evidence": "fixed_WGS84_coordinate_with_MNT_ground",
            }
        )
    candidates.sort(key=lambda item: item["candidate_id"])
    counts = {
        status: sum(candidate["status"] == status for candidate in candidates)
        for status in ("valid", "ambiguous", "rejected")
    }
    return {
        "source_count": len(candidates),
        "valid_count": counts["valid"],
        "ambiguous_count": counts["ambiguous"],
        "rejected_count": counts["rejected"],
        "placement_ready_count": counts["valid"],
        "placement_blocked_count": counts["ambiguous"] + counts["rejected"],
        "instantiated_asset_count": 0,
        "halo_only_feature_count": halo_only_count,
        "quantity_policy": (
            "one_candidate_per_stable_feature_crossing_or_fixed_coordinate_no_quota"
        ),
        "candidates": candidates,
    }


def _nearest_rank(values: np.ndarray, numerator: int, denominator: int) -> int:
    flat = np.sort(np.asarray(values).reshape(-1))
    if flat.size == 0:
        raise PlacementInventoryError("cannot calculate a percentile without samples")
    rank = (numerator * int(flat.size) + denominator - 1) // denominator
    return int(flat[max(0, rank - 1)])


def _lower_median(values: np.ndarray) -> int:
    flat = np.sort(np.asarray(values).reshape(-1))
    if flat.size == 0:
        raise PlacementInventoryError("cannot calculate a median without samples")
    return int(flat[(int(flat.size) - 1) // 2])


def _building_inventory(
    footprints: list[tuple[str, BaseGeometry, str, dict[str, Any]]],
    *,
    mnt_mm: np.ndarray,
    hag_cm: np.ndarray,
    transform: Any,
    west: int,
    south: int,
) -> tuple[dict[str, Any], np.ndarray, str]:
    processing_bounds = box(
        west - HALO_M,
        south - HALO_M,
        west + TILE_SIZE_M + HALO_M,
        south + TILE_SIZE_M + HALO_M,
    )
    footprint_mask = np.zeros((PROCESSING_SIZE, PROCESSING_SIZE), dtype=bool)
    candidates: list[dict[str, Any]] = []
    halo_only_count = 0
    context_hash_records: list[dict[str, str]] = []
    for source_id, geometry, geometry_hash, source_properties in footprints:
        context_hash_records.append(
            {
                "source_id": source_id,
                "geometry_sha256": geometry_hash,
                "source_properties": source_properties,
            }
        )
        if not geometry.intersects(processing_bounds):
            continue
        sampled = _rasterize_geometries([geometry], transform=transform)
        footprint_mask |= sampled
        anchor = geometry.representative_point()
        anchor_x = float(anchor.x)
        anchor_y = float(anchor.y)
        if not _is_owned(anchor_x, anchor_y, west, south):
            halo_only_count += 1
            continue
        candidate_id = _stable_id("building", source_id)
        sample_count = int(np.count_nonzero(sampled))
        reasons: list[str] = []
        if sample_count == 0:
            status = "rejected"
            reasons.append("footprint_has_no_1m_sample")
            ground_mm = None
            height_cm = None
            top_mm = None
        else:
            ground_mm = _lower_median(mnt_mm[sampled])
            height_cm = _nearest_rank(hag_cm[sampled], 95, 100)
            top_mm = ground_mm + height_cm * 10
            if not processing_bounds.covers(geometry):
                status = "ambiguous"
                reasons.append("footprint_exceeds_processing_halo")
            elif height_cm < MIN_HEIGHT_CM:
                status = "ambiguous"
                reasons.append("mns_mnt_height_below_2m")
            else:
                status = "valid"
        candidates.append(
            {
                "candidate_id": candidate_id,
                "source_id": source_id,
                "geometry_sha256": geometry_hash,
                "source_properties": source_properties,
                "status": status,
                "reason_codes": reasons,
                "anchor_l93_m": [round(anchor_x, 3), round(anchor_y, 3)],
                "footprint_geojson": _geometry_payload(geometry),
                "footprint_area_m2": round(float(geometry.area), 3),
                "sample_count": sample_count,
                "ground_elevation_mm": ground_mm,
                "height_cm": height_cm,
                "top_elevation_mm": top_mm,
            }
        )
    candidates.sort(key=lambda item: item["candidate_id"])
    counts = {
        status: sum(candidate["status"] == status for candidate in candidates)
        for status in ("valid", "ambiguous", "rejected")
    }
    source_count = len(candidates)
    if source_count != sum(counts.values()):
        raise RuntimeError("building reconciliation failed internally")
    payload = {
        "detection_mode": "sig_footprints_constrained_by_mns_mnt",
        "source_count": source_count,
        "valid_count": counts["valid"],
        "ambiguous_count": counts["ambiguous"],
        "rejected_count": counts["rejected"],
        "placement_ready_count": counts["valid"],
        "placement_blocked_count": counts["ambiguous"] + counts["rejected"],
        "instantiated_asset_count": 0,
        "halo_only_footprint_count": halo_only_count,
        "candidates": candidates,
    }
    context_hash = _sha256(canonical_json_bytes(context_hash_records))
    return payload, footprint_mask, context_hash


def _component_source_id(
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    west: int,
    south: int,
) -> str:
    cells = np.empty((len(rows), 2), dtype="<i4")
    for index, (row, column) in enumerate(zip(rows, columns)):
        cells[index] = _global_cell(west, south, int(row), int(column))
    order = np.lexsort((cells[:, 1], cells[:, 0]))
    digest = _sha256(cells[order].tobytes(order="C"))[:24]
    return f"hag-component-{digest}"


def _component_geometry(mask: np.ndarray, transform: Any) -> BaseGeometry:
    polygons = [
        shape(geometry)
        for geometry, value in raster_shapes(
            mask.astype("uint8"), mask=mask, transform=transform
        )
        if int(value) == 1
    ]
    if not polygons:
        raise RuntimeError("HAG building component has no raster footprint")
    geometry = unary_union(polygons)
    normalise = getattr(geometry, "normalize", None)
    return normalise() if callable(normalise) else geometry


def _confirmed_building_candidate(
    *,
    confirmation: Mapping[str, Any],
    component_mask: np.ndarray,
    mnt_mm: np.ndarray,
    hag_cm: np.ndarray,
    local_range_cm: np.ndarray,
    transform: Any,
    west: int,
    south: int,
    footprint_pixel_count: int,
    overlaps_another_confirmation: bool,
) -> dict[str, Any]:
    """Measure one semantically confirmed building only from its HAG pixels."""

    rows, columns = np.nonzero(component_mask)
    area_m2 = len(rows)
    centroid_row = float(np.mean(rows))
    centroid_column = float(np.mean(columns))
    anchor_choice = min(
        range(area_m2),
        key=lambda index: (
            (float(rows[index]) - centroid_row) ** 2
            + (float(columns[index]) - centroid_column) ** 2,
            _global_cell(west, south, int(rows[index]), int(columns[index])),
        ),
    )
    anchor_row = int(rows[anchor_choice])
    anchor_column = int(columns[anchor_choice])
    anchor_x, anchor_y = _global_cell(west, south, anchor_row, anchor_column)
    values_cm = hag_cm[rows, columns]
    height_p10_cm = _nearest_rank(values_cm, 10, 100)
    height_p50_cm = _nearest_rank(values_cm, 50, 100)
    height_p90_cm = _nearest_rank(values_cm, 90, 100)
    height_p95_cm = _nearest_rank(values_cm, 95, 100)
    dispersion_cm = height_p90_cm - height_p10_cm
    planar_pixel_count = int(np.count_nonzero(local_range_cm[rows, columns] <= 75))
    bbox_width = int(columns.max() - columns.min() + 1)
    bbox_height = int(rows.max() - rows.min() + 1)
    rectangularity = area_m2 / (bbox_width * bbox_height)
    perimeter_edges = int(
        4 * area_m2
        - 2
        * (
            np.count_nonzero(component_mask[:, 1:] & component_mask[:, :-1])
            + np.count_nonzero(component_mask[1:, :] & component_mask[:-1, :])
        )
    )
    compactness = (
        4.0 * math.pi * area_m2 / (perimeter_edges**2) if perimeter_edges else 0.0
    )
    touches_processing_edge = bool(
        rows.min() == 0
        or rows.max() == PROCESSING_SIZE - 1
        or columns.min() == 0
        or columns.max() == PROCESSING_SIZE - 1
    )
    if overlaps_another_confirmation:
        status = "ambiguous"
        reasons = ["building_confirmation_overlap_not_univocal"]
    elif touches_processing_edge:
        status = "ambiguous"
        reasons = ["component_exceeds_processing_halo"]
    else:
        status = "valid"
        reasons = []
    source_id = _component_source_id(rows, columns, west=west, south=south)
    geometry = _component_geometry(component_mask, transform)
    ground_mm = _lower_median(mnt_mm[rows, columns])
    confirmation_match = {
        "source_id": str(confirmation["source_id"]),
        "geometry_sha256": str(confirmation["geometry_sha256"]),
        "intersection_pixel_count": area_m2,
        "component_overlap_ratio": 1.0,
        "footprint_overlap_ratio": round(area_m2 / footprint_pixel_count, 6),
    }
    source_properties = dict(confirmation.get("source_properties", {}))
    return {
        "candidate_id": _stable_id("building", source_id),
        "source_id": source_id,
        "geometry_sha256": _sha256(canonical_json_bytes(_geometry_payload(geometry))),
        "status": status,
        "reason_codes": reasons,
        "anchor_l93_m": [anchor_x + 0.5, anchor_y + 0.5],
        "footprint_geojson": _geometry_payload(geometry),
        "footprint_area_m2": area_m2,
        "sample_count": area_m2,
        "ground_elevation_mm": ground_mm,
        "height_cm": height_p95_cm,
        "top_elevation_mm": ground_mm + height_p95_cm * 10,
        "confirmation_matches": [confirmation_match],
        "confirmed_source_id": (
            None if overlaps_another_confirmation else str(confirmation["source_id"])
        ),
        "source_properties": source_properties,
        "metrics": {
            "height_p10_cm": height_p10_cm,
            "height_p50_cm": height_p50_cm,
            "height_p90_cm": height_p90_cm,
            "height_dispersion_p90_p10_cm": dispersion_cm,
            "planar_pixel_count": planar_pixel_count,
            "planar_ratio": round(planar_pixel_count / area_m2, 6),
            "rectangularity": round(rectangularity, 6),
            "compactness": round(compactness, 6),
            "vegetation_overlap_ratio": 0.0,
            "building_prior_ratio": 1.0,
            "interior_pixel_count": int(
                np.count_nonzero(
                    binary_erosion(
                        component_mask, structure=_CONNECTIVITY_4, border_value=0
                    )
                )
            ),
            "touches_processing_edge": touches_processing_edge,
        },
    }


def _autodetect_building_inventory(
    *,
    mnt_mm: np.ndarray,
    hag_cm: np.ndarray,
    transform: Any,
    west: int,
    south: int,
    vegetation_prior: np.ndarray,
    building_prior: np.ndarray,
    footprint_confirmations: Sequence[Mapping[str, Any]],
    infrastructure_exclusion: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, str]:
    """Classify every >=2 m object component as roof-like or non-building."""

    elevated = (hag_cm >= MIN_HEIGHT_CM) & ~infrastructure_exclusion
    local_range_cm = maximum_filter(hag_cm, size=3, mode="nearest").astype(
        "int32"
    ) - minimum_filter(hag_cm, size=3, mode="nearest").astype("int32")
    confirmed_pixel_claims = np.zeros_like(hag_cm, dtype="uint16")
    confirmed_inputs: list[tuple[Mapping[str, Any], np.ndarray, int]] = []
    confirmation_component_ids: dict[str, set[str]] = defaultdict(set)
    directly_segmented_confirmation_ids: set[str] = set()
    for confirmation in footprint_confirmations:
        confirmation_mask = np.asarray(confirmation["mask"], dtype=bool)
        footprint_pixel_count = int(np.count_nonzero(confirmation_mask))
        if footprint_pixel_count == 0:
            continue
        confirmed_pixels = confirmation_mask & elevated
        intersection_count = int(np.count_nonzero(confirmed_pixels))
        if intersection_count < 4 or intersection_count / footprint_pixel_count < 0.25:
            continue
        component_labels, component_count = label(
            confirmed_pixels, structure=_CONNECTIVITY_4
        )
        source_id = str(confirmation["source_id"])
        for component_id, component_slice in enumerate(
            find_objects(component_labels, max_label=component_count), start=1
        ):
            if component_slice is None:
                continue
            local_component = component_labels[component_slice] == component_id
            if int(np.count_nonzero(local_component)) < 4:
                continue
            component_mask = np.zeros_like(confirmed_pixels)
            component_mask[component_slice] = local_component
            rows, columns = np.nonzero(component_mask)
            component_source_id = _component_source_id(
                rows, columns, west=west, south=south
            )
            confirmed_inputs.append(
                (confirmation, component_mask, footprint_pixel_count)
            )
            confirmed_pixel_claims[component_mask] += 1
            confirmation_component_ids[source_id].add(component_source_id)
            directly_segmented_confirmation_ids.add(source_id)
    elevated_for_morphology = elevated & (confirmed_pixel_claims == 0)
    components, component_count = label(elevated, structure=_CONNECTIVITY_4)
    if confirmed_inputs:
        components, component_count = label(
            elevated_for_morphology, structure=_CONNECTIVITY_4
        )
    component_slices = find_objects(components, max_label=component_count)
    component_sizes = np.bincount(
        components[components > 0], minlength=component_count + 1
    )
    confirmation_matches: dict[int, list[dict[str, Any]]] = defaultdict(list)
    non_univocal_confirmation_ids: set[str] = set()
    for confirmation, component_mask, _footprint_pixel_count in confirmed_inputs:
        if np.any(confirmed_pixel_claims[component_mask] > 1):
            non_univocal_confirmation_ids.add(str(confirmation["source_id"]))
    for confirmation in footprint_confirmations:
        source_id = str(confirmation["source_id"])
        if source_id in confirmation_component_ids:
            continue
        confirmation_mask = np.asarray(confirmation["mask"], dtype=bool)
        footprint_pixel_count = int(np.count_nonzero(confirmation_mask))
        if footprint_pixel_count == 0:
            continue
        labels_in_footprint, intersections = np.unique(
            components[confirmation_mask & elevated], return_counts=True
        )
        for component_value, intersection_value in zip(
            labels_in_footprint, intersections
        ):
            component_value = int(component_value)
            if component_value <= 0:
                continue
            intersection_count = int(intersection_value)
            component_ratio = intersection_count / int(component_sizes[component_value])
            footprint_ratio = intersection_count / footprint_pixel_count
            if (
                intersection_count < 4
                or component_ratio < 0.25
                or footprint_ratio < 0.25
            ):
                continue
            confirmation_matches[component_value].append(
                {
                    "source_id": source_id,
                    "geometry_sha256": str(confirmation["geometry_sha256"]),
                    "intersection_pixel_count": intersection_count,
                    "component_overlap_ratio": round(component_ratio, 6),
                    "footprint_overlap_ratio": round(footprint_ratio, 6),
                    "source_properties": dict(
                        confirmation.get("source_properties", {})
                    ),
                }
            )
            confirmation_component_ids[source_id].add(component_value)
    building_exclusion = np.zeros_like(elevated)
    candidates: list[dict[str, Any]] = []
    halo_only_count = 0
    for confirmation, confirmed_pixels, footprint_pixel_count in confirmed_inputs:
        candidate = _confirmed_building_candidate(
            confirmation=confirmation,
            component_mask=confirmed_pixels,
            mnt_mm=mnt_mm,
            hag_cm=hag_cm,
            local_range_cm=local_range_cm,
            transform=transform,
            west=west,
            south=south,
            footprint_pixel_count=footprint_pixel_count,
            overlaps_another_confirmation=bool(
                np.any(confirmed_pixel_claims[confirmed_pixels] > 1)
            ),
        )
        if not _is_owned(
            float(candidate["anchor_l93_m"][0]),
            float(candidate["anchor_l93_m"][1]),
            west,
            south,
        ):
            halo_only_count += 1
            continue
        candidates.append(candidate)
        if candidate["status"] == "valid":
            building_exclusion |= confirmed_pixels
    for component_id, component_slice in enumerate(component_slices, start=1):
        if component_slice is None:
            continue
        local_component = components[component_slice] == component_id
        row_offset = int(component_slice[0].start or 0)
        column_offset = int(component_slice[1].start or 0)
        local_rows, local_columns = np.nonzero(local_component)
        rows = local_rows + row_offset
        columns = local_columns + column_offset
        area_m2 = len(rows)
        if area_m2 == 0:
            continue
        centroid_row = float(np.mean(rows))
        centroid_column = float(np.mean(columns))
        anchor_choice = min(
            range(area_m2),
            key=lambda index: (
                (float(rows[index]) - centroid_row) ** 2
                + (float(columns[index]) - centroid_column) ** 2,
                _global_cell(west, south, int(rows[index]), int(columns[index])),
            ),
        )
        anchor_row = int(rows[anchor_choice])
        anchor_column = int(columns[anchor_choice])
        anchor_x, anchor_y = _global_cell(west, south, anchor_row, anchor_column)
        if not _is_owned(anchor_x + 0.5, anchor_y + 0.5, west, south):
            halo_only_count += 1
            continue
        source_id = _component_source_id(rows, columns, west=west, south=south)
        candidate_id = _stable_id("building", source_id)
        values_cm = hag_cm[rows, columns]
        height_p10_cm = _nearest_rank(values_cm, 10, 100)
        height_p50_cm = _nearest_rank(values_cm, 50, 100)
        height_p90_cm = _nearest_rank(values_cm, 90, 100)
        height_p95_cm = _nearest_rank(values_cm, 95, 100)
        dispersion_cm = height_p90_cm - height_p10_cm
        planar_pixel_count = int(np.count_nonzero(local_range_cm[rows, columns] <= 75))
        planar_ratio = planar_pixel_count / area_m2
        bbox_width = int(columns.max() - columns.min() + 1)
        bbox_height = int(rows.max() - rows.min() + 1)
        rectangularity = area_m2 / (bbox_width * bbox_height)
        component_mask = np.zeros_like(elevated)
        component_mask[rows, columns] = True
        eroded = binary_erosion(
            component_mask, structure=_CONNECTIVITY_4, border_value=0
        )
        perimeter_edges = int(
            4 * area_m2
            - 2
            * (
                np.count_nonzero(component_mask[:, 1:] & component_mask[:, :-1])
                + np.count_nonzero(component_mask[1:, :] & component_mask[:-1, :])
            )
        )
        compactness = (
            4.0 * math.pi * area_m2 / (perimeter_edges**2) if perimeter_edges else 0.0
        )
        vegetation_overlap_ratio = float(
            np.count_nonzero(vegetation_prior[rows, columns]) / area_m2
        )
        building_prior_ratio = float(
            np.count_nonzero(building_prior[rows, columns]) / area_m2
        )
        matches = sorted(
            confirmation_matches.get(component_id, []),
            key=lambda match: match["source_id"],
        )
        univocal_footprint_match = bool(
            len(matches) == 1
            and len(confirmation_component_ids[matches[0]["source_id"]]) == 1
        )
        touches_processing_edge = bool(
            rows.min() == 0
            or rows.max() == PROCESSING_SIZE - 1
            or columns.min() == 0
            or columns.max() == PROCESSING_SIZE - 1
        )
        roof_like = (
            area_m2 >= 16
            and dispersion_cm <= 100
            and planar_ratio >= 0.50
            and rectangularity >= 0.55
            and compactness >= 0.20
            and vegetation_overlap_ratio <= 0.25
        )
        semantic_confirmation = univocal_footprint_match or (
            building_prior_ratio >= 0.50 and vegetation_overlap_ratio <= 0.25
        )
        prior_supported = (
            area_m2 >= 9 and semantic_confirmation and dispersion_cm <= 200
        )
        weak_roof_like = (
            area_m2 >= 9
            and dispersion_cm <= 200
            and planar_ratio >= 0.25
            and rectangularity >= 0.35
            and vegetation_overlap_ratio <= 0.50
        )
        reasons: list[str] = []
        if touches_processing_edge and (roof_like or prior_supported):
            status = "ambiguous"
            reasons.append("component_exceeds_processing_halo")
        elif (roof_like or prior_supported) and semantic_confirmation:
            status = "valid"
        elif roof_like:
            status = "ambiguous"
            reasons.append("morphology_requires_semantic_confirmation")
        elif prior_supported or weak_roof_like:
            status = "ambiguous" if prior_supported else "rejected"
            reasons.append(
                "weak_roof_morphology"
                if prior_supported
                else "morphology_not_conclusive"
            )
        else:
            status = "rejected"
            if area_m2 < 9:
                reasons.append("component_area_below_9m2")
            if dispersion_cm > 200:
                reasons.append("height_dispersion_tree_like")
            if planar_ratio < 0.25:
                reasons.append("insufficient_roof_planarity")
            if rectangularity < 0.35:
                reasons.append("insufficient_rectangularity")
            if vegetation_overlap_ratio > 0.50:
                reasons.append("vegetation_prior_dominant")
            if not reasons:
                reasons.append("morphology_not_building_like")
        if matches and not univocal_footprint_match:
            status = "ambiguous"
            reasons = ["building_confirmation_match_not_univocal"]
            non_univocal_confirmation_ids.update(
                str(match["source_id"]) for match in matches
            )
        # A morphology-only ambiguity is deliberately not removed from the
        # tree detector: doing so would silently reduce vegetation because a
        # flat crown can look rectangular in a 1 m HAG.  Only semantically
        # confirmed buildings are exclusive.
        if status == "valid":
            building_exclusion |= component_mask
        geometry = _component_geometry(component_mask, transform)
        ground_mm = _lower_median(mnt_mm[rows, columns])
        candidates.append(
            {
                "candidate_id": candidate_id,
                "source_id": source_id,
                "geometry_sha256": _sha256(
                    canonical_json_bytes(_geometry_payload(geometry))
                ),
                "status": status,
                "reason_codes": reasons,
                "anchor_l93_m": [anchor_x + 0.5, anchor_y + 0.5],
                "footprint_geojson": (
                    _geometry_payload(geometry)
                    if status in {"valid", "ambiguous"}
                    else None
                ),
                "footprint_area_m2": area_m2,
                "sample_count": area_m2,
                "ground_elevation_mm": ground_mm,
                "height_cm": height_p95_cm,
                "top_elevation_mm": ground_mm + height_p95_cm * 10,
                "confirmation_matches": matches,
                "confirmed_source_id": (
                    matches[0]["source_id"] if univocal_footprint_match else None
                ),
                "source_properties": (
                    dict(matches[0].get("source_properties", {}))
                    if univocal_footprint_match
                    else {}
                ),
                "metrics": {
                    "height_p10_cm": height_p10_cm,
                    "height_p50_cm": height_p50_cm,
                    "height_p90_cm": height_p90_cm,
                    "height_dispersion_p90_p10_cm": dispersion_cm,
                    "planar_pixel_count": planar_pixel_count,
                    "planar_ratio": round(planar_ratio, 6),
                    "rectangularity": round(rectangularity, 6),
                    "compactness": round(compactness, 6),
                    "vegetation_overlap_ratio": round(vegetation_overlap_ratio, 6),
                    "building_prior_ratio": round(building_prior_ratio, 6),
                    "interior_pixel_count": int(np.count_nonzero(eroded)),
                    "touches_processing_edge": touches_processing_edge,
                },
            }
        )
    candidates.sort(key=lambda item: item["candidate_id"])
    counts = {
        status: sum(candidate["status"] == status for candidate in candidates)
        for status in ("valid", "ambiguous", "rejected")
    }
    source_count = len(candidates)
    payload = {
        "detection_mode": "mns_mnt_hag_with_semantic_confirmation",
        "source_count": source_count,
        "valid_count": counts["valid"],
        "ambiguous_count": counts["ambiguous"],
        "rejected_count": counts["rejected"],
        "placement_ready_count": counts["valid"],
        "placement_blocked_count": counts["ambiguous"] + counts["rejected"],
        "instantiated_asset_count": 0,
        "halo_only_footprint_count": halo_only_count,
        "confirmation_source_count": len(footprint_confirmations),
        "confirmed_hag_component_count": len(confirmed_inputs),
        "multi_component_confirmation_count": sum(
            len(confirmation_component_ids[source_id]) > 1
            for source_id in directly_segmented_confirmation_ids
        ),
        "unmatched_confirmation_count": sum(
            not confirmation_component_ids.get(str(item["source_id"]))
            for item in footprint_confirmations
        ),
        "non_univocal_confirmation_count": len(non_univocal_confirmation_ids),
        "thresholds": {
            "minimum_height_cm": MIN_HEIGHT_CM,
            "valid_minimum_area_m2": 16,
            "valid_maximum_dispersion_cm": 100,
            "valid_minimum_planar_ratio": 0.50,
            "valid_minimum_rectangularity": 0.55,
            "semantic_confirmation_minimum_building_prior_ratio": 0.50,
            "morphology_without_semantic_confirmation": "ambiguous_never_valid",
        },
        "candidates": candidates,
    }
    context_hash = _sha256(
        canonical_json_bytes(
            {
                "mode": payload["detection_mode"],
                "building_prior_sha256": _mask_sha256(building_prior),
                "footprint_confirmations": [
                    {
                        "source_id": str(item["source_id"]),
                        "geometry_sha256": str(item["geometry_sha256"]),
                    }
                    for item in footprint_confirmations
                ],
                "vegetation_prior_sha256": _mask_sha256(vegetation_prior),
                "infrastructure_sha256": _mask_sha256(infrastructure_exclusion),
            }
        )
    )
    return payload, building_exclusion, context_hash


def _global_cell(west: int, south: int, row: int, column: int) -> tuple[int, int]:
    grid_west = west - HALO_M
    grid_north = south + TILE_SIZE_M + HALO_M
    return grid_west + column, grid_north - row - 1


def _peak_coordinates(
    candidate_pixels: np.ndarray,
    hag_cm: np.ndarray,
    *,
    west: int,
    south: int,
    flat_plateau_split_mask: np.ndarray | None = None,
) -> tuple[
    list[tuple[int, int, int, int]],
    int,
    int,
    tuple[tuple[tuple[int, int, int, int], ...], ...],
]:
    components, component_count = label(candidate_pixels, structure=_CONNECTIVITY_8)
    peaks: list[tuple[int, int, int, int]] = []
    extra_marker_count = 0
    split_plateau_count = 0
    flat_plateau_groups: list[tuple[tuple[int, int, int, int], ...]] = []
    for component_id, component_slice in enumerate(
        find_objects(components, max_label=component_count), start=1
    ):
        if component_slice is None:
            continue
        local = components[component_slice]
        rows_local, columns_local = np.nonzero(local == component_id)
        row_offset = int(component_slice[0].start or 0)
        column_offset = int(component_slice[1].start or 0)
        rows = rows_local + row_offset
        columns = columns_local + column_offset
        maximum = int(np.max(hag_cm[rows, columns]))
        finalists = np.flatnonzero(hag_cm[rows, columns] == maximum)
        choices: list[tuple[int, int, int, int]] = []
        for finalist in finalists:
            row = int(rows[finalist])
            column = int(columns[finalist])
            global_x, global_y = _global_cell(west, south, row, column)
            choices.append((global_x, global_y, row, column))
        primary = min(choices)
        selected = {primary}
        if flat_plateau_split_mask is not None:
            woody_choices = [
                choice
                for choice in choices
                if bool(flat_plateau_split_mask[choice[2], choice[3]])
            ]
            if len(woody_choices) >= FLAT_WOODY_PLATEAU_MIN_AREA_M2:
                lattice_choices = {
                    choice
                    for choice in woody_choices
                    if choice[0] % FLAT_WOODY_PLATEAU_MARKER_SPACING_M == 0
                    and choice[1] % FLAT_WOODY_PLATEAU_MARKER_SPACING_M == 0
                }
                if lattice_choices:
                    # Use only the global lattice for a split plateau.  Keeping
                    # the per-window lexicographic primary as an extra marker
                    # would make a canopy crossing two tile halos acquire two
                    # different seam-dependent trees.
                    selected = lattice_choices
                    split_plateau_count += 1
                    flat_plateau_groups.append(tuple(sorted(lattice_choices)))
        peaks.extend(selected)
        extra_marker_count += len(selected) - 1
    peaks.sort(key=lambda item: (item[0], item[1]))
    return (
        peaks,
        extra_marker_count,
        split_plateau_count,
        tuple(flat_plateau_groups),
    )


def _seam_keys_by_label(
    labels: np.ndarray,
    *,
    west: int,
    south: int,
) -> dict[int, set[str]]:
    keys: dict[int, set[str]] = defaultdict(set)
    for row in range(CORE_START, CORE_STOP):
        _, global_y = _global_cell(west, south, row, CORE_START)
        core_label = int(labels[row, CORE_START])
        if core_label > 0 and core_label == int(labels[row, CORE_START - 1]):
            keys[core_label].add(f"x:{west}:y:{global_y}")
        core_label = int(labels[row, CORE_STOP - 1])
        if core_label > 0 and core_label == int(labels[row, CORE_STOP]):
            keys[core_label].add(f"x:{west + TILE_SIZE_M}:y:{global_y}")
    for column in range(CORE_START, CORE_STOP):
        global_x, _ = _global_cell(west, south, CORE_START, column)
        core_label = int(labels[CORE_START, column])
        if core_label > 0 and core_label == int(labels[CORE_START - 1, column]):
            keys[core_label].add(f"y:{south + TILE_SIZE_M}:x:{global_x}")
        core_label = int(labels[CORE_STOP - 1, column])
        if core_label > 0 and core_label == int(labels[CORE_STOP, column]):
            keys[core_label].add(f"y:{south}:x:{global_x}")
    return keys


def _tree_inventory(
    *,
    mnt_mm: np.ndarray,
    hag_cm: np.ndarray,
    vegetation_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    west: int,
    south: int,
) -> dict[str, Any]:
    # MNS-MNT remains the sole positive evidence for crowns.  Accepted woody
    # context only lowers the measured-height floor so shrubs and young trees
    # between 1 m and 2 m are not silently lost inside wooded masses.  It
    # cannot create an instance where HAG has no measured height.
    height_floor_cm = np.where(
        vegetation_mask, WOODY_CONTEXT_MIN_HEIGHT_CM, MIN_HEIGHT_CM
    )
    measured_canopy = hag_cm >= height_floor_cm
    eligible = ~exclusion_mask & measured_canopy
    neighbourhood = maximum_filter(
        hag_cm,
        size=3,
        mode="constant",
        cval=0,
    )
    local_maxima = eligible & (hag_cm == neighbourhood)
    (
        peaks,
        flat_plateau_extra_marker_count,
        split_flat_plateau_count,
        flat_plateau_groups,
    ) = _peak_coordinates(
        local_maxima,
        hag_cm,
        west=west,
        south=south,
        flat_plateau_split_mask=eligible & vegetation_mask,
    )
    # ``watershed_ift`` treats every non-zero marker, including negative
    # values, as a competing basin. Initialising the excluded background to
    # -1 therefore let that basin capture measured crown pixels before the
    # post-mask was applied. Keep the image unlabelled until the measured
    # peaks are inserted, then mask the flooded result afterwards.
    markers = np.zeros(hag_cm.shape, dtype="int32")
    candidate_id_by_label: dict[int, str] = {}
    peak_by_label: dict[int, tuple[int, int, int, int]] = {}
    marker_label_by_peak: dict[tuple[int, int, int, int], int] = {}
    for marker_id, peak in enumerate(peaks, start=1):
        global_x, global_y, row, column = peak
        markers[row, column] = marker_id
        candidate_id_by_label[marker_id] = _stable_id("tree", global_x, global_y)
        peak_by_label[marker_id] = peak
        marker_label_by_peak[peak] = marker_id
    flat_plateau_marker_labels = {
        marker_label_by_peak[peak] for group in flat_plateau_groups for peak in group
    }
    if peaks:
        maximum = int(hag_cm[eligible].max())
        cost = np.where(eligible, maximum - hag_cm.astype("int64"), maximum).astype(
            "uint16"
        )
        labels = watershed_ift(cost, markers, structure=_CONNECTIVITY_8).astype(
            "int32", copy=False
        )
        for group in flat_plateau_groups:
            group_marker_labels = np.asarray(
                [marker_label_by_peak[peak] for peak in group], dtype="int32"
            )
            region = eligible & np.isin(labels, group_marker_labels)
            region_rows, region_columns = np.nonzero(region)
            if region_rows.size == 0:
                continue
            row_start = int(region_rows.min())
            row_stop = int(region_rows.max()) + 1
            column_start = int(region_columns.min())
            column_stop = int(region_columns.max()) + 1
            component_slice = (
                slice(row_start, row_stop),
                slice(column_start, column_stop),
            )
            local_seed_background = np.ones(
                (row_stop - row_start, column_stop - column_start), dtype=bool
            )
            local_seed_labels = np.zeros(local_seed_background.shape, dtype="int32")
            for peak in group:
                marker_id = marker_label_by_peak[peak]
                row = peak[2] - row_start
                column = peak[3] - column_start
                local_seed_background[row, column] = False
                local_seed_labels[row, column] = marker_id
            _distances, nearest_indices = distance_transform_edt(
                local_seed_background, return_indices=True
            )
            nearest_labels = local_seed_labels[nearest_indices[0], nearest_indices[1]]
            local_region = region[component_slice]
            local_labels = labels[component_slice]
            local_labels[local_region] = nearest_labels[local_region]
        labels[~eligible] = -1
    else:
        labels = markers

    core_labels = labels[CORE_START:CORE_STOP, CORE_START:CORE_STOP]
    core_counts = np.bincount(core_labels[core_labels > 0], minlength=len(peaks) + 1)
    observed_counts = np.bincount(labels[labels > 0], minlength=len(peaks) + 1)
    object_slices = find_objects(labels, max_label=len(peaks)) if peaks else []
    seam_keys = _seam_keys_by_label(labels, west=west, south=south)
    fragments: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for marker_id in range(1, len(peaks) + 1):
        core_count = int(core_counts[marker_id])
        keys = sorted(seam_keys.get(marker_id, set()))
        if core_count == 0 and not keys:
            continue
        global_x, global_y, row, column = peak_by_label[marker_id]
        candidate_id = candidate_id_by_label[marker_id]
        component_slice = object_slices[marker_id - 1]
        touches_processing_edge = False
        if component_slice is not None:
            row_start = int(component_slice[0].start or 0)
            row_stop = int(component_slice[0].stop or 0)
            column_start = int(component_slice[1].start or 0)
            column_stop = int(component_slice[1].stop or 0)
            local_labels = labels[component_slice]
            local_mask = local_labels == marker_id
            touches_processing_edge = bool(
                (row_start == 0 and np.any(local_mask[0, :]))
                or (row_stop == PROCESSING_SIZE and np.any(local_mask[-1, :]))
                or (column_start == 0 and np.any(local_mask[:, 0]))
                or (column_stop == PROCESSING_SIZE and np.any(local_mask[:, -1]))
            )
            component_rows, component_columns = np.nonzero(local_mask)
            minimum_row = row_start + int(component_rows.min())
            maximum_row = row_start + int(component_rows.max())
            minimum_column = column_start + int(component_columns.min())
            maximum_column = column_start + int(component_columns.max())
            minimum_x, maximum_y_cell = _global_cell(
                west, south, minimum_row, minimum_column
            )
            maximum_x, minimum_y_cell = _global_cell(
                west, south, maximum_row, maximum_column
            )
            crown_bounds_l93_m = [
                minimum_x,
                minimum_y_cell,
                maximum_x + 1,
                maximum_y_cell + 1,
            ]
        else:
            crown_bounds_l93_m = [global_x, global_y, global_x + 1, global_y + 1]
        observed_count = int(observed_counts[marker_id])
        owned = CORE_START <= row < CORE_STOP and CORE_START <= column < CORE_STOP
        fragment = {
            "fragment_id": _stable_id("tree-fragment", candidate_id, west, south),
            "candidate_id": candidate_id,
            "peak_cell_l93": [global_x, global_y],
            "owned": owned,
            "core_pixel_count": core_count,
            "observed_pixel_count": observed_count,
            "touches_processing_edge": touches_processing_edge,
            "seam_keys": keys,
            "marker_policy": (
                "flat_woody_plateau_global_lattice"
                if marker_id in flat_plateau_marker_labels
                else "measured_local_maximum"
            ),
        }
        fragments.append(fragment)
        if not owned:
            continue
        peak_height_cm = int(hag_cm[row, column])
        vegetation_prior_peak = bool(vegetation_mask[row, column])
        if vegetation_prior_peak:
            minimum_crown_area_m2 = (
                WOODY_CONTEXT_TALL_MIN_CROWN_AREA_M2
                if peak_height_cm >= MIN_HEIGHT_CM
                else WOODY_CONTEXT_LOW_MIN_CROWN_AREA_M2
            )
        else:
            minimum_crown_area_m2 = MIN_CROWN_AREA_M2
        if observed_count < minimum_crown_area_m2:
            status = "rejected"
            reasons = ["crown_area_below_contextual_minimum"]
        elif touches_processing_edge:
            status = "ambiguous"
            reasons = ["crown_exceeds_processing_halo"]
        else:
            status = "valid"
            reasons = []
        height_cm = int(hag_cm[row, column])
        ground_mm = int(mnt_mm[row, column])
        vegetation_prior_ratio = float(
            np.count_nonzero(vegetation_mask[labels == marker_id]) / observed_count
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "status": status,
                "reason_codes": reasons,
                "peak_cell_l93": [global_x, global_y],
                "position_l93_m": [global_x + 0.5, global_y + 0.5],
                "ground_elevation_mm": ground_mm,
                "height_cm": height_cm,
                "top_elevation_mm": ground_mm + height_cm * 10,
                "observed_crown_area_m2": observed_count,
                "minimum_crown_area_m2_applied": minimum_crown_area_m2,
                "owned_core_area_m2": core_count,
                "equivalent_crown_radius_m": round(
                    math.sqrt(observed_count / math.pi), 3
                ),
                "crown_bounds_l93_m": crown_bounds_l93_m,
                "touches_processing_edge": touches_processing_edge,
                "vegetation_prior_peak": vegetation_prior_peak,
                "vegetation_prior_overlap_ratio": round(vegetation_prior_ratio, 6),
                "marker_policy": (
                    "flat_woody_plateau_global_lattice"
                    if marker_id in flat_plateau_marker_labels
                    else "measured_local_maximum"
                ),
                "context_classification": (
                    "low_1_to_3m_vegetation_prior"
                    if height_cm < 300 and vegetation_prior_ratio > 0.0
                    else "low_1_to_3m_hag_only"
                    if height_cm < 300
                    else "vegetation_prior"
                    if vegetation_prior_ratio > 0.0
                    else "hag_only"
                ),
            }
        )
    candidates.sort(key=lambda item: item["candidate_id"])
    fragments.sort(key=lambda item: item["fragment_id"])
    counts = {
        status: sum(candidate["status"] == status for candidate in candidates)
        for status in ("valid", "ambiguous", "rejected")
    }
    source_count = len(candidates)
    if source_count != sum(counts.values()):
        raise RuntimeError("tree reconciliation failed internally")
    return {
        "source_count": source_count,
        "valid_count": counts["valid"],
        "ambiguous_count": counts["ambiguous"],
        "rejected_count": counts["rejected"],
        "placement_ready_count": counts["valid"],
        "placement_blocked_count": counts["ambiguous"] + counts["rejected"],
        "instantiated_asset_count": 0,
        "candidate_count_without_quota_or_thinning": source_count,
        "eligible_pixel_count": int(np.count_nonzero(eligible)),
        "excluded_pixel_count": int(np.count_nonzero(measured_canopy & exclusion_mask)),
        "default_minimum_height_cm": MIN_HEIGHT_CM,
        "woody_context_minimum_height_cm": WOODY_CONTEXT_MIN_HEIGHT_CM,
        "default_minimum_crown_area_m2": MIN_CROWN_AREA_M2,
        "woody_context_tall_minimum_crown_area_m2": (
            WOODY_CONTEXT_TALL_MIN_CROWN_AREA_M2
        ),
        "woody_context_low_minimum_crown_area_m2": (
            WOODY_CONTEXT_LOW_MIN_CROWN_AREA_M2
        ),
        "vegetation_prior_pixel_count": int(np.count_nonzero(vegetation_mask)),
        "vegetation_prior_policy": (
            "measured_HAG_floor_100cm_in_woody_context_200cm_elsewhere;"
            "crown_area_minimum_1m2_at_or_above_200cm_and_2m2_from_100_to_199cm_"
            "in_woody_context_4m2_elsewhere;"
            "never_quota_or_unmeasured_authorization"
        ),
        "local_maximum_count_processing_window": len(peaks),
        "flat_woody_plateau_minimum_area_m2": FLAT_WOODY_PLATEAU_MIN_AREA_M2,
        "flat_woody_plateau_marker_spacing_m": (FLAT_WOODY_PLATEAU_MARKER_SPACING_M),
        "split_flat_woody_plateau_count_processing_window": (split_flat_plateau_count),
        "flat_woody_plateau_extra_marker_count_processing_window": (
            flat_plateau_extra_marker_count
        ),
        "candidates": candidates,
        "fragments": fragments,
    }


def build_placement_inventory(
    mnt_m: Any,
    mns_m: Any,
    *,
    tile_origin_l93_m: Sequence[float],
    zone_id: str,
    building_footprints: Iterable[Mapping[str, Any]] = (),
    context_masks: Mapping[str, Any] | None = None,
    context_geometries: Mapping[str, Iterable[Any]] | None = None,
    context_features: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    fixed_asset_placements: Sequence[Mapping[str, Any]] = (),
) -> PlacementResult:
    """Build the canonical HAG and placement inventory for exactly one tile."""

    if not isinstance(zone_id, str) or not zone_id.strip():
        raise PlacementInventoryError("zone_id must be a non-empty string")
    zone_id = zone_id.strip()
    west, south = _origin(tile_origin_l93_m)
    try:
        normalized_fixed_assets = validate_projected_placements(
            fixed_asset_placements,
            tile_origin_l93_m=(west, south),
        )
    except FixedAssetPlacementError as error:
        raise PlacementInventoryError(
            f"fixed asset placements are invalid: {error}"
        ) from error
    mnt_mm, mns_mm, hag_cm, source_diagnostics = _source_grid(mnt_m, mns_m)
    masks_input = dict(context_masks or {})
    geometries_input = dict(context_geometries or {})
    unknown_masks = sorted(set(masks_input) - set(_CONTEXT_KEYS))
    unknown_geometries = sorted(set(geometries_input) - set(_CONTEXT_KEYS[1:]))
    if unknown_masks or unknown_geometries:
        raise PlacementInventoryError(
            f"unknown context keys: {unknown_masks + unknown_geometries}"
        )
    transform = from_origin(
        west - HALO_M,
        south + TILE_SIZE_M + HALO_M,
        RESOLUTION_M,
        RESOLUTION_M,
    )
    masks: dict[str, np.ndarray] = {}
    geometry_hashes: dict[str, str] = {}
    geometry_masks: dict[str, np.ndarray] = {}
    for key in _CONTEXT_KEYS:
        masks[key] = _mask(
            masks_input.get(key), name=key, default=(key == "vegetation")
        )
    for key in _CONTEXT_KEYS[1:]:
        geometries, geometry_hash = _geometry_context(
            geometries_input.get(key, ()), name=key
        )
        geometry_hashes[key] = geometry_hash
        geometry_masks[key] = _rasterize_geometries(geometries, transform=transform)
    vegetation_context_supplied = "vegetation" in masks_input or bool(
        geometries_input.get("vegetation")
    )
    if geometry_masks["vegetation"].any():
        if "vegetation" in masks_input:
            masks["vegetation"] |= geometry_masks["vegetation"]
        else:
            masks["vegetation"] = geometry_masks["vegetation"]
    for key in ("roads", "rail", "water"):
        masks[key] |= geometry_masks[key]
    footprints = _normalise_footprints(building_footprints)
    normalised_features, context_features_hash = _normalise_context_features(
        context_features
    )
    footprint_confirmations, footprint_context_hash = _footprint_confirmations(
        footprints, transform=transform
    )
    vegetation_prior = (
        masks["vegetation"]
        if vegetation_context_supplied
        else np.zeros_like(masks["vegetation"])
    )
    buildings, detected_building_mask, _ = _autodetect_building_inventory(
        mnt_mm=mnt_mm,
        hag_cm=hag_cm,
        transform=transform,
        west=west,
        south=south,
        vegetation_prior=vegetation_prior,
        building_prior=masks["buildings"],
        footprint_confirmations=footprint_confirmations,
        infrastructure_exclusion=(masks["roads"] | masks["rail"] | masks["water"]),
    )
    # Raw BD TOPO or orthophoto masks are semantic confirmations only.  They
    # never author geometry and never exclude HAG pixels by themselves.
    exclusion = detected_building_mask | masks["roads"] | masks["rail"] | masks["water"]
    trees = _tree_inventory(
        mnt_mm=mnt_mm,
        hag_cm=hag_cm,
        vegetation_mask=vegetation_prior,
        exclusion_mask=exclusion,
        west=west,
        south=south,
    )
    context_assets = _context_asset_inventory(
        normalised_features,
        mnt_mm=mnt_mm,
        west=west,
        south=south,
        fixed_asset_placements=normalized_fixed_assets,
    )

    context_record = {
        "building_footprints_sha256": footprint_context_hash,
        "masks_sha256": {key: _mask_sha256(masks[key]) for key in _CONTEXT_KEYS},
        "geometries_sha256": geometry_hashes,
        "features_sha256": context_features_hash,
        "fixed_asset_placement_count": len(normalized_fixed_assets),
        "fixed_asset_placements_sha256": _sha256(
            canonical_fixed_asset_bytes(normalized_fixed_assets)
        ),
        "vegetation_context_supplied": vegetation_context_supplied,
    }
    contract_sha256 = _contract_sha256()
    sources = {
        "mnt_mm_sha256": _sha256(np.asarray(mnt_mm, dtype="<i4").tobytes(order="C")),
        "mns_mm_sha256": _sha256(np.asarray(mns_mm, dtype="<i4").tobytes(order="C")),
        "context_sha256": _sha256(canonical_json_bytes(context_record)),
        "contract_sha256": contract_sha256,
        "algorithm_sha256": _sha256(ALGORITHM.encode("ascii")),
    }
    build_id = _sha256(
        canonical_json_bytes(
            {
                "zone_id": zone_id,
                "tile_origin_l93_m": [west, south],
                "sources": sources,
            }
        )
    )
    hag_core = np.asarray(
        hag_cm[CORE_START:CORE_STOP, CORE_START:CORE_STOP], dtype="uint16"
    ).copy()
    inventory: dict[str, Any] = {
        "schema": SCHEMA,
        "contract_schema": CONTRACT_SCHEMA,
        "algorithm": ALGORITHM,
        "build_id": build_id,
        "zone_id": zone_id,
        "tile_id": f"E{west}-N{south}",
        "crs": CRS,
        "grid": {
            "resolution_m": RESOLUTION_M,
            "processing_halo_m": HALO_M,
            "processing_shape": [PROCESSING_SIZE, PROCESSING_SIZE],
            "core_shape": [TILE_SIZE_M, TILE_SIZE_M],
            "core_bounds_l93_m": [
                west,
                south,
                west + TILE_SIZE_M,
                south + TILE_SIZE_M,
            ],
            "row_order": "north_to_south",
            "ownership": "half_open",
        },
        "sources": sources,
        "context": context_record,
        "hag": {
            "schema": HAG_SCHEMA,
            "dtype": "uint16",
            "unit": "centimetre",
            "nodata": NODATA_UINT16,
            "minimum_cm": int(hag_core.min()),
            "maximum_cm": int(hag_core.max()),
            "raw_sha256": _sha256(np.asarray(hag_core, dtype="<u2").tobytes(order="C")),
            **source_diagnostics,
        },
        "buildings": buildings,
        "trees": trees,
        "context_assets": context_assets,
    }
    inventory["inventory_sha256"] = _sha256(canonical_json_bytes(inventory))
    validate_inventory(inventory)
    return PlacementResult(hag_core_cm=hag_core, inventory=inventory)


def validate_inventory(inventory: Mapping[str, Any]) -> None:
    """Fail closed on corruption or silent candidate loss."""

    if inventory.get("schema") != SCHEMA or inventory.get("crs") != CRS:
        raise PlacementInventoryError("placement inventory schema or CRS is invalid")
    hag = inventory.get("hag")
    if not isinstance(hag, Mapping):
        raise PlacementInventoryError("placement inventory lacks HAG diagnostics")
    minimum_delta = hag.get("minimum_source_delta_mm")
    negative_count = hag.get("negative_source_sample_count_clamped")
    outlier_count = hag.get("negative_outlier_below_tolerance_count")
    outlier_fraction = hag.get("negative_outlier_fraction")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (minimum_delta, negative_count, outlier_count)
    ):
        raise PlacementInventoryError("HAG source diagnostics are invalid")
    factual_v2_negative_policy = (
        inventory.get("placement_profile")
        == "fireviewer.factual-placement-profile.v2"
        and hag.get("negative_outlier_policy")
        == "clamp_all_when_below_minus_100cm_fraction_is_at_most_1pct"
    )
    severe_negative_count = hag.get("severe_negative_below_100cm_count")
    severe_negative_fraction = hag.get("severe_negative_below_100cm_fraction")
    factual_v2_severe_diagnostics_valid = (
        factual_v2_negative_policy
        and isinstance(severe_negative_count, int)
        and not isinstance(severe_negative_count, bool)
        and severe_negative_count >= 0
        and severe_negative_count <= negative_count
        and isinstance(severe_negative_fraction, (int, float))
        and not isinstance(severe_negative_fraction, bool)
        and math.isfinite(float(severe_negative_fraction))
        and 0 <= severe_negative_fraction <= 0.01
        and math.isclose(
            float(severe_negative_fraction),
            severe_negative_count / (PROCESSING_SIZE * PROCESSING_SIZE),
            abs_tol=5e-13,
        )
    )
    if (
        (
            minimum_delta < -(NEGATIVE_HAG_HARD_LIMIT_CM * 10)
            and not factual_v2_severe_diagnostics_valid
        )
        or negative_count < 0
        or outlier_count < 0
        or (factual_v2_negative_policy and not factual_v2_severe_diagnostics_valid)
        or (
            not factual_v2_negative_policy
            and outlier_count > NEGATIVE_HAG_MAX_OUTLIER_COUNT
        )
        or not isinstance(outlier_fraction, (int, float))
        or isinstance(outlier_fraction, bool)
        or not math.isfinite(float(outlier_fraction))
        or outlier_fraction < 0
        or (
            not factual_v2_negative_policy
            and outlier_fraction > NEGATIVE_HAG_MAX_OUTLIER_FRACTION
        )
        or outlier_count > negative_count
    ):
        raise PlacementInventoryError("HAG source diagnostics violate the contract")
    for family in ("buildings", "trees", "context_assets"):
        payload = inventory.get(family)
        if not isinstance(payload, Mapping):
            raise PlacementInventoryError(f"placement inventory lacks {family}")
        source_count = payload.get("source_count")
        status_total = sum(
            int(payload.get(f"{status}_count", -1))
            for status in ("valid", "ambiguous", "rejected")
        )
        candidates = payload.get("candidates")
        if not isinstance(source_count, int) or source_count != status_total:
            raise PlacementInventoryError(f"{family} source reconciliation is invalid")
        if not isinstance(candidates, list) or len(candidates) != source_count:
            raise PlacementInventoryError(f"{family} candidate count is invalid")
        if payload.get("placement_ready_count") != payload.get("valid_count"):
            raise PlacementInventoryError(f"{family} ready count is corrupt")
        if payload.get("placement_blocked_count") != (
            payload.get("ambiguous_count") + payload.get("rejected_count")
        ):
            raise PlacementInventoryError(f"{family} blocked count is corrupt")
        if source_count != (
            payload.get("placement_ready_count")
            + payload.get("placement_blocked_count")
        ):
            raise PlacementInventoryError(
                f"{family} source-to-placement reconciliation is invalid"
            )
        if payload.get("instantiated_asset_count") != 0:
            raise PlacementInventoryError(
                f"{family} inventory must not instantiate assets"
            )
        identifiers = [candidate.get("candidate_id") for candidate in candidates]
        if len(identifiers) != len(set(identifiers)):
            raise PlacementInventoryError(f"{family} candidate IDs are not unique")
        observed = {
            status: sum(candidate.get("status") == status for candidate in candidates)
            for status in ("valid", "ambiguous", "rejected")
        }
        if any(
            observed[status] != payload.get(f"{status}_count") for status in observed
        ):
            raise PlacementInventoryError(f"{family} status counts are corrupt")
    supplied_hash = inventory.get("inventory_sha256")
    if not isinstance(supplied_hash, str):
        raise PlacementInventoryError("placement inventory hash is missing")
    without_hash = dict(inventory)
    without_hash.pop("inventory_sha256", None)
    if _sha256(canonical_json_bytes(without_hash)) != supplied_hash:
        raise PlacementInventoryError("placement inventory hash mismatch")


def write_hag_1m(
    path: Path | str,
    values_cm: Any,
    *,
    tile_origin_l93_m: Sequence[float],
) -> str:
    """Write an atomic, compact, self-validating HAG GeoTIFF on D:."""

    destination = assert_d_storage_path(path)
    values = np.asarray(values_cm)
    if values.shape != CORE_SHAPE or values.dtype.kind not in {"i", "u"}:
        raise PlacementInventoryError("HAG must be a 500x500 integer grid")
    if int(values.min()) < 0 or int(values.max()) > MAX_HAG_CM:
        raise PlacementInventoryError("HAG values collide with uint16 nodata")
    canonical = np.asarray(values, dtype="uint16")
    west, south = _origin(tile_origin_l93_m)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.tif")
    temporary.unlink(missing_ok=True)
    raw_hash = _sha256(np.asarray(canonical, dtype="<u2").tobytes(order="C"))
    try:
        with rasterio.open(
            temporary,
            "w",
            driver="GTiff",
            width=TILE_SIZE_M,
            height=TILE_SIZE_M,
            count=1,
            dtype="uint16",
            crs=CRS,
            transform=from_origin(
                west, south + TILE_SIZE_M, RESOLUTION_M, RESOLUTION_M
            ),
            nodata=NODATA_UINT16,
            compress="DEFLATE",
            predictor=2,
            zlevel=9,
            tiled=True,
            blockxsize=128,
            blockysize=128,
            BIGTIFF="NO",
            NUM_THREADS="1",
        ) as dataset:
            dataset.write(canonical, 1)
            dataset.update_tags(
                FIREVIEWER_SCHEMA=HAG_SCHEMA,
                FIREVIEWER_UNIT="centimetre",
                FIREVIEWER_ROW_ORDER="north_to_south",
                FIREVIEWER_RAW_SHA256=raw_hash,
                FIREVIEWER_SOURCE="MNS_minus_MNT_1m",
            )
        content = temporary.read_bytes()
        if destination.exists():
            if destination.read_bytes() != content:
                raise FileExistsError(
                    f"refusing to replace different HAG: {destination}"
                )
            temporary.unlink()
        else:
            temporary.replace(destination)
        return _sha256(content)
    finally:
        temporary.unlink(missing_ok=True)


def read_hag_1m(path: Path | str) -> tuple[np.ndarray, dict[str, Any]]:
    """Read and verify a retained HAG GeoTIFF."""

    with rasterio.open(path) as dataset:
        if (
            dataset.crs is None
            or dataset.crs.to_string() != CRS
            or dataset.width != TILE_SIZE_M
            or dataset.height != TILE_SIZE_M
            or dataset.count != 1
            or dataset.dtypes[0] != "uint16"
            or dataset.nodata != NODATA_UINT16
        ):
            raise PlacementInventoryError("HAG raster contract mismatch")
        tags = dataset.tags()
        if tags.get("FIREVIEWER_SCHEMA") != HAG_SCHEMA:
            raise PlacementInventoryError("HAG schema tag mismatch")
        values = dataset.read(1)
        raw_hash = _sha256(np.asarray(values, dtype="<u2").tobytes(order="C"))
        if tags.get("FIREVIEWER_RAW_SHA256") != raw_hash:
            raise PlacementInventoryError("HAG raw payload hash mismatch")
        metadata = {
            "schema": HAG_SCHEMA,
            "crs": CRS,
            "bounds_l93_m": list(dataset.bounds),
            "resolution_m": RESOLUTION_M,
            "raw_sha256": raw_hash,
        }
    return values, metadata


def write_inventory_json(path: Path | str, inventory: Mapping[str, Any]) -> str:
    """Write the authoritative canonical inventory atomically on D:."""

    validate_inventory(inventory)
    destination = assert_d_storage_path(path)
    content = canonical_json_bytes(inventory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_bytes(content)
        if destination.exists():
            if destination.read_bytes() != content:
                raise FileExistsError(
                    f"refusing to replace different inventory: {destination}"
                )
            temporary.unlink()
        else:
            temporary.replace(destination)
        return _sha256(content)
    finally:
        temporary.unlink(missing_ok=True)


def gpkg_supported() -> bool:
    """Return whether the optional Fiona derivative writer is available."""

    try:
        import fiona  # noqa: F401
    except ImportError:
        return False
    return True


def write_inventory_gpkg(path: Path | str, inventory: Mapping[str, Any]) -> None:
    """Write point layers when Fiona exists; canonical JSON remains authoritative."""

    validate_inventory(inventory)
    destination = assert_d_storage_path(path)
    try:
        import fiona
    except ImportError as error:
        raise GeoPackageUnavailableError(
            "Fiona is not installed; retain canonical placement-inventory.json"
        ) from error
    if destination.exists():
        raise FileExistsError(f"refusing to replace GeoPackage: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    crs = "EPSG:2154"
    schemas = {
        "building_candidates": {
            "geometry": "Unknown",
            "properties": {
                "candidate": "str:40",
                "source_id": "str:120",
                "status": "str:12",
                "height_cm": "int",
            },
        },
        "tree_candidates": {
            "geometry": "Point",
            "properties": {
                "candidate": "str:40",
                "status": "str:12",
                "height_cm": "int",
                "area_m2": "int",
            },
        },
    }
    with fiona.open(
        destination,
        "w",
        driver="GPKG",
        layer="building_candidates",
        schema=schemas["building_candidates"],
        crs=crs,
    ) as layer:
        for candidate in inventory["buildings"]["candidates"]:
            layer.write(
                {
                    "geometry": {
                        **candidate["footprint_geojson"],
                    },
                    "properties": {
                        "candidate": candidate["candidate_id"],
                        "source_id": candidate["source_id"],
                        "status": candidate["status"],
                        "height_cm": candidate["height_cm"] or 0,
                    },
                }
            )
    with fiona.open(
        destination,
        "w",
        driver="GPKG",
        layer="tree_candidates",
        schema=schemas["tree_candidates"],
        crs=crs,
    ) as layer:
        for candidate in inventory["trees"]["candidates"]:
            layer.write(
                {
                    "geometry": {
                        "type": "Point",
                        "coordinates": candidate["position_l93_m"],
                    },
                    "properties": {
                        "candidate": candidate["candidate_id"],
                        "status": candidate["status"],
                        "height_cm": candidate["height_cm"],
                        "area_m2": candidate["observed_crown_area_m2"],
                    },
                }
            )


def merge_tree_inventories(
    inventories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge cross-tile fragments using stable peaks and seam evidence.

    Core ownership remains authoritative.  A seam may join provisional IDs
    when both tiles prove that one watershed label crosses the same boundary
    cell.  The lexicographically smallest member ID is the canonical merge ID.
    """

    parent: dict[str, str] = {}
    owned: dict[str, list[str]] = defaultdict(list)
    core_area_by_candidate: dict[str, int] = defaultdict(int)
    seam_occurrences: dict[str, list[tuple[str, str]]] = defaultdict(list)
    fragment_tiles_by_candidate: dict[str, set[str]] = defaultdict(set)
    marker_policy_by_candidate: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    tile_ids = [str(inventory.get("tile_id")) for inventory in inventories]
    if len(tile_ids) != len(set(tile_ids)):
        raise PlacementInventoryError("tree seam merge received duplicate tile IDs")
    for inventory in inventories:
        validate_inventory(inventory)
        tile_id = str(inventory["tile_id"])
        for candidate in inventory["trees"]["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            find(candidate_id)
            owned[candidate_id].append(tile_id)
        for fragment in inventory["trees"]["fragments"]:
            candidate_id = str(fragment["candidate_id"])
            find(candidate_id)
            marker_policy = str(fragment.get("marker_policy", "measured_local_maximum"))
            previous_policy = marker_policy_by_candidate.setdefault(
                candidate_id, marker_policy
            )
            if previous_policy != marker_policy:
                raise PlacementInventoryError(
                    f"tree marker policy differs across tiles: {candidate_id}"
                )
            fragment_tiles_by_candidate[candidate_id].add(tile_id)
            core_area_by_candidate[candidate_id] += int(fragment["core_pixel_count"])
            for seam_key in fragment["seam_keys"]:
                seam_occurrences[str(seam_key)].append((tile_id, candidate_id))
    for occurrences in seam_occurrences.values():
        by_tile = {tile_id for tile_id, _ in occurrences}
        if len(by_tile) < 2:
            continue
        identifiers = sorted({candidate_id for _, candidate_id in occurrences})
        if len(identifiers) > 1 and (
            all(
                marker_policy_by_candidate.get(candidate_id)
                == "flat_woody_plateau_global_lattice"
                for candidate_id in identifiers
            )
            or all(
                len(fragment_tiles_by_candidate[candidate_id]) > 1
                for candidate_id in identifiers
            )
        ):
            # Both labels are already proven global peaks observed by each
            # neighbouring processing halo.  A watershed tie on one boundary
            # cell must not collapse those two measured trees into one group.
            continue
        for candidate_id in identifiers[1:]:
            union(identifiers[0], candidate_id)
    groups: dict[str, set[str]] = defaultdict(set)
    for candidate_id in sorted(parent):
        groups[find(candidate_id)].add(candidate_id)
    result_groups: list[dict[str, Any]] = []
    duplicate_ownership: list[dict[str, Any]] = []
    for canonical_id, members in sorted(groups.items()):
        owner_tiles = sorted(
            {tile for member in members for tile in owned.get(member, [])}
        )
        if len(owner_tiles) > 1:
            duplicate_ownership.append(
                {
                    "canonical_candidate_id": canonical_id,
                    "owner_tiles": owner_tiles,
                }
            )
        result_groups.append(
            {
                "canonical_candidate_id": canonical_id,
                "member_candidate_ids": sorted(members),
                "owner_tiles": owner_tiles,
                "total_owned_core_area_m2": sum(
                    core_area_by_candidate[member] for member in members
                ),
            }
        )
    if duplicate_ownership:
        raise PlacementInventoryError(
            f"tree ownership conflict after seam merge: {duplicate_ownership}"
        )
    return {
        "schema": "fireviewer.mns-mnt-tree-seam-merge.v1",
        "tile_count": len(inventories),
        "candidate_group_count": sum(
            bool(group["owner_tiles"]) for group in result_groups
        ),
        "groups": result_groups,
        "shared_seam_key_count": sum(
            len({tile_id for tile_id, _ in occurrences}) >= 2
            for occurrences in seam_occurrences.values()
        ),
    }


def write_placement_outputs(
    output_directory: Path | str,
    result: PlacementResult,
    *,
    tile_origin_l93_m: Sequence[float],
    gpkg: str = "auto",
) -> dict[str, Any]:
    """Persist the canonical outputs and, optionally, the GeoPackage derivative."""

    if gpkg not in {"auto", "off", "require"}:
        raise PlacementInventoryError("gpkg must be auto, off or require")
    directory = assert_d_storage_path(output_directory)
    hag_path = directory / "placement-hag-1m.tif"
    inventory_path = directory / "placement-inventory.json"
    outputs: dict[str, Any] = {
        "hag": {
            "path": hag_path.name,
            "sha256": write_hag_1m(
                hag_path,
                result.hag_core_cm,
                tile_origin_l93_m=tile_origin_l93_m,
            ),
        },
        "inventory": {
            "path": inventory_path.name,
            "sha256": write_inventory_json(inventory_path, result.inventory),
        },
        "gpkg": {"status": "disabled"},
    }
    if gpkg != "off":
        if gpkg_supported():
            gpkg_path = directory / "placement-candidates.gpkg"
            write_inventory_gpkg(gpkg_path, result.inventory)
            outputs["gpkg"] = {
                "status": "written",
                "path": gpkg_path.name,
                "sha256": _sha256(gpkg_path.read_bytes()),
            }
        elif gpkg == "require":
            raise GeoPackageUnavailableError(
                "Fiona is required by --gpkg=require but is not installed"
            )
        else:
            outputs["gpkg"] = {
                "status": "unavailable",
                "reason": "fiona_not_installed_canonical_json_retained",
            }
    return outputs


def _load_cli_context(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "building_footprints": [],
            "context_geometries": {},
            "context_features": {},
        }
    assert_d_storage_path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PlacementInventoryError("context JSON must contain one object")
    if payload.get("crs", CRS) != CRS:
        raise PlacementInventoryError("context JSON must use EPSG:2154")
    footprints = payload.get("building_footprints", [])
    geometries = payload.get("context_geometries", {})
    features = payload.get("context_features", {})
    if (
        not isinstance(footprints, list)
        or not isinstance(geometries, Mapping)
        or not isinstance(features, Mapping)
    ):
        raise PlacementInventoryError("context JSON collections are malformed")
    return {
        "building_footprints": footprints,
        "context_geometries": geometries,
        "context_features": features,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for one already co-registered tile; no download or asset instancing."""

    parser = argparse.ArgumentParser(
        description="Derive one deterministic MNS/MNT placement inventory"
    )
    parser.add_argument(
        "--source-npz",
        type=Path,
        required=True,
        help="NPZ on D: with mnt, mns and optional mask_<context> arrays",
    )
    parser.add_argument("--context-json", type=Path)
    parser.add_argument("--zone", required=True)
    parser.add_argument(
        "--tile-origin",
        nargs=2,
        type=float,
        required=True,
        metavar=("EASTING", "NORTHING"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpkg", choices=("auto", "off", "require"), default="auto")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    source_path = assert_d_storage_path(arguments.source_npz)
    if not source_path.is_file():
        parser.error(f"source NPZ does not exist: {source_path}")
    context = _load_cli_context(arguments.context_json)
    with np.load(source_path, allow_pickle=False) as source:
        if "mnt" not in source or "mns" not in source:
            raise PlacementInventoryError("source NPZ requires mnt and mns arrays")
        masks = {
            key: np.asarray(source[f"mask_{key}"])
            for key in _CONTEXT_KEYS
            if f"mask_{key}" in source
        }
        result = build_placement_inventory(
            source["mnt"],
            source["mns"],
            tile_origin_l93_m=arguments.tile_origin,
            zone_id=arguments.zone,
            building_footprints=context["building_footprints"],
            context_masks=masks,
            context_geometries=context["context_geometries"],
            context_features=context["context_features"],
        )
    summary: dict[str, Any] = {
        "schema": "fireviewer.mns-mnt-placement-cli-result.v1",
        "mode": "dry-run" if arguments.dry_run else "execute",
        "build_id": result.inventory["build_id"],
        "tile_id": result.inventory["tile_id"],
        "buildings": {
            key: result.inventory["buildings"][key]
            for key in (
                "source_count",
                "placement_ready_count",
                "placement_blocked_count",
            )
        },
        "trees": {
            key: result.inventory["trees"][key]
            for key in (
                "source_count",
                "placement_ready_count",
                "placement_blocked_count",
            )
        },
        "context_assets": {
            key: result.inventory["context_assets"][key]
            for key in (
                "source_count",
                "placement_ready_count",
                "placement_blocked_count",
            )
        },
    }
    if arguments.execute:
        if arguments.output_dir is None:
            parser.error("--output-dir is required with --execute")
        summary["outputs"] = write_placement_outputs(
            arguments.output_dir,
            result,
            tile_origin_l93_m=arguments.tile_origin,
            gpkg=arguments.gpkg,
        )
    print(canonical_json_bytes(summary).decode("utf-8"))
    return 0


__all__ = [
    "ALGORITHM",
    "CONTRACT_SCHEMA",
    "CRS",
    "HAG_SCHEMA",
    "MAX_HAG_CM",
    "NODATA_UINT16",
    "SCHEMA",
    "GeoPackageUnavailableError",
    "PlacementInventoryError",
    "PlacementResult",
    "assert_d_storage_path",
    "build_placement_inventory",
    "canonical_json_bytes",
    "gpkg_supported",
    "main",
    "merge_tree_inventories",
    "read_hag_1m",
    "validate_inventory",
    "write_hag_1m",
    "write_inventory_gpkg",
    "write_inventory_json",
    "write_placement_outputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
