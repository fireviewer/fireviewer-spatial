"""Compile deterministic 5 m ground composition for one 500 m terrain tile.

The compiler consumes globally identified EPSG:2154 context features.  It keeps
wide surfaces in two compact RGBA maps and preserves narrow or directional
features as clipped vector overlays.  All choices are made from global source
identifiers and contract hashes; tile identifiers, local paths, clocks and
worker order are deliberately absent from every seed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from io import BytesIO
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    Point,
    Polygon,
    box,
    mapping,
    shape,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, substring
from shapely.strtree import STRtree

from ground_context_binding import classify_feature, select_profile


SCHEMA = "fireviewer.tile-composition.v1"
OVERLAY_SCHEMA = "fireviewer.surface-overlays.v1"
CRS = "EPSG:2154"
TILE_SIZE_M = 500.0
GRID_CELL_SIZE_M = 5.0
GRID_SIZE = 100
SAMPLES_PER_CELL = 4
LINEAR_SEGMENT_LENGTH_M = 250.0
COORDINATE_PRECISION_M = 0.001

ROLE_PRIORITY = {
    "natural": 0,
    "agriculture": 1,
    "cliff": 2,
    "path": 3,
    "road": 4,
    "rail": 5,
    "hydro": 6,
    "crossing_override": 7,
}
LINEAR_ROLES = {"path", "road", "rail", "hydro"}
TRANSPORT_ROLES = {"path", "road", "rail"}
VALID_CROSSING_KINDS = {"bridge", "tunnel", "ford", "culvert"}
GLOBAL_GEOMETRY_SCOPE = "global_feature"
CLIPPED_GEOMETRY_SCOPE = "clipped_feature"
VALID_GEOMETRY_SCOPES = {GLOBAL_GEOMETRY_SCOPE, CLIPPED_GEOMETRY_SCOPE}


@dataclass(frozen=True)
class _Feature:
    feature_id: str
    layer_id: str
    role: str
    geometry: BaseGeometry
    properties: Mapping[str, Any]
    semantic_tags: frozenset[str]


@dataclass(frozen=True)
class TileComposition:
    """Pure compilation result, ready for deterministic serialization."""

    manifest: dict[str, Any]
    overlays: dict[str, Any]
    profile_ids: np.ndarray
    profile_weights: np.ndarray


class _SpatialIndex:
    def __init__(self, features: Sequence[_Feature]) -> None:
        self.features = tuple(features)
        self.tree = (
            STRtree([feature.geometry for feature in self.features])
            if features
            else None
        )

    def covering(self, point: Point) -> list[_Feature]:
        if self.tree is None:
            return []
        return [
            self.features[int(index)]
            for index in self.tree.query(point)
            if self.features[int(index)].geometry.covers(point)
        ]


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_property_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Context properties must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_property_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_property_value(item) for item in value]
    raise ValueError(f"Unsupported context property value: {type(value).__name__}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _seed(*parts: str) -> str:
    raw = "\0".join((SCHEMA, *parts)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _finite_number(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _normalize_angle(value: float) -> float:
    result = value % 180.0
    if math.isclose(result, 180.0, abs_tol=1e-12):
        result = 0.0
    return round(result, 6)


def _geometry(value: Any) -> BaseGeometry:
    geometry = value if isinstance(value, BaseGeometry) else shape(value)
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("Context feature geometry must be valid and non-empty")
    return geometry


def _global_feature_id(value: Any) -> str:
    feature_id = str(value or "").strip()
    if not feature_id:
        raise ValueError("Every context feature requires a global feature_id")
    if any(character in feature_id for character in ("/", "\\", "\0")):
        raise ValueError("global feature_id must not contain a path or NUL separator")
    return feature_id


def _feature_role(layer_id: str, properties: Mapping[str, Any], declared: Any) -> str:
    if declared is not None:
        role = str(declared)
        if role not in ROLE_PRIORITY:
            raise ValueError(f"Unsupported surface role: {role}")
        return role
    if layer_id in {"landcover", "geology", "land_parcels"}:
        return "natural"
    if layer_id == "agricultural_parcels":
        return "agriculture"
    if layer_id == "roads":
        return (
            "path"
            if "transport:path" in classify_feature(layer_id, properties)
            else "road"
        )
    if layer_id == "railways":
        return "rail"
    if layer_id in {"hydro_lines", "hydro_surfaces"}:
        return "hydro"
    if layer_id == "cliffs":
        return "cliff"
    raise ValueError(f"No surface role is defined for context layer: {layer_id}")


def _semantic_tags(
    layer_id: str, properties: Mapping[str, Any], declared: Any, role: str
) -> frozenset[str]:
    tags: set[str]
    if layer_id == "cliffs":
        geology = str(properties.get("geology", "")).strip()
        if not geology:
            raise ValueError("A cliff feature requires explicit geology")
        tags = classify_feature("geology", {"formation": geology}) | {"terrain:cliff"}
    else:
        tags = classify_feature(layer_id, properties)
    if declared is not None:
        if not isinstance(declared, (list, tuple, set)) or any(
            not isinstance(tag, str) or ":" not in tag for tag in declared
        ):
            raise ValueError("semantic_tags must be a collection of namespaced strings")
        tags.update(declared)
    if role == "cliff":
        tags.add("terrain:cliff")
    return frozenset(tags)


def _prepare_features(features: Iterable[Mapping[str, Any]]) -> list[_Feature]:
    prepared: list[_Feature] = []
    identifiers: set[str] = set()
    for raw in features:
        feature_id = _global_feature_id(raw.get("feature_id"))
        if feature_id in identifiers:
            raise ValueError(f"Duplicate global feature_id: {feature_id}")
        identifiers.add(feature_id)
        layer_id = str(raw.get("layer_id", "")).strip()
        properties = raw.get("properties", {})
        if not layer_id or not isinstance(properties, Mapping):
            raise ValueError(f"Invalid context feature metadata: {feature_id}")
        role = _feature_role(layer_id, properties, raw.get("role"))
        prepared.append(
            _Feature(
                feature_id=feature_id,
                layer_id=layer_id,
                role=role,
                geometry=_geometry(raw.get("geometry")),
                properties=properties,
                semantic_tags=_semantic_tags(
                    layer_id, properties, raw.get("semantic_tags"), role
                ),
            )
        )
    return sorted(prepared, key=lambda feature: feature.feature_id)


def _normalize_crossing_overrides(
    overrides: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    identifiers: set[tuple[str, str]] = set()
    for override in overrides:
        if not isinstance(override, Mapping):
            raise ValueError("Every crossing override must be an object")
        transport_id = _global_feature_id(override.get("transport_feature_id"))
        hydro_id = _global_feature_id(override.get("hydro_feature_id"))
        pair = transport_id, hydro_id
        if pair in identifiers:
            raise ValueError(
                f"Duplicate crossing override: {transport_id} / {hydro_id}"
            )
        identifiers.add(pair)
        kind = str(override.get("kind", "")).strip()
        if kind not in VALID_CROSSING_KINDS:
            raise ValueError(f"Unsupported crossing override kind: {kind}")
        canonical = _canonical_property_value(override)
        if not isinstance(canonical, dict):
            raise AssertionError("Canonical crossing override must remain an object")
        canonical.update(
            {
                "transport_feature_id": transport_id,
                "hydro_feature_id": hydro_id,
                "kind": kind,
            }
        )
        normalized.append(canonical)
    return sorted(
        normalized,
        key=lambda override: (
            str(override["transport_feature_id"]),
            str(override["hydro_feature_id"]),
            _canonical_bytes(override),
        ),
    )


def _context_input_payload(
    features: Sequence[_Feature], context_contract: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "context_contract": _canonical_property_value(context_contract),
        "features": [
            {
                "feature_id": feature.feature_id,
                "layer_id": feature.layer_id,
                "role": feature.role,
                "semantic_tags": sorted(feature.semantic_tags),
                "properties": _canonical_property_value(feature.properties),
                "geometry_l93_m": _geometry_payload(feature.geometry),
            }
            for feature in features
        ],
    }


def _validate_bounds(bounds: Sequence[float]) -> tuple[float, float, float, float]:
    if len(bounds) != 4:
        raise ValueError("Tile bounds must contain four EPSG:2154 coordinates")
    minimum_x, minimum_y, maximum_x, maximum_y = (
        _finite_number(value, label="tile bound") for value in bounds
    )
    if not math.isclose(
        maximum_x - minimum_x, TILE_SIZE_M, abs_tol=1e-6
    ) or not math.isclose(maximum_y - minimum_y, TILE_SIZE_M, abs_tol=1e-6):
        raise ValueError("A composition tile must be exactly 500 m square")
    for value in (minimum_x, minimum_y, maximum_x, maximum_y):
        if not math.isclose(
            value / TILE_SIZE_M, round(value / TILE_SIZE_M), abs_tol=1e-9
        ):
            raise ValueError(
                "Tile bounds must align to the global 500 m Lambert-93 grid"
            )
    return minimum_x, minimum_y, maximum_x, maximum_y


def _single_cover(index: _SpatialIndex, point: Point, *, label: str) -> _Feature:
    matches = index.covering(point)
    if len(matches) != 1:
        coordinate = f"({point.x:.3f},{point.y:.3f})"
        raise ValueError(
            f"{label} coverage is {'missing' if not matches else 'ambiguous'} at {coordinate}"
        )
    return matches[0]


def _optional_single_cover(
    index: _SpatialIndex, point: Point, *, label: str
) -> _Feature | None:
    matches = index.covering(point)
    if len(matches) > 1:
        raise ValueError(
            f"{label} coverage is ambiguous at ({point.x:.3f},{point.y:.3f})"
        )
    return matches[0] if matches else None


def _profile_table(catalog: Mapping[str, Any]) -> tuple[list[str], dict[str, int]]:
    identifiers = sorted(
        str(profile.get("id", "")) for profile in catalog.get("profiles", [])
    )
    if not identifiers or any(not identifier for identifier in identifiers):
        raise ValueError("The atlas catalog has no usable profiles")
    if len(identifiers) != len(set(identifiers)) or len(identifiers) > 255:
        raise ValueError("Profile identifiers must be unique and fit in RGBA8 indices")
    return identifiers, {
        identifier: index for index, identifier in enumerate(identifiers)
    }


def _choose_profile(
    context_contract: Mapping[str, Any],
    atlas_catalog: Mapping[str, Any],
    *,
    family: str,
    tags: set[str],
    seed: str,
    excluded: Sequence[str] = (),
) -> dict[str, Any]:
    if excluded:
        reduced_catalog = dict(atlas_catalog)
        reduced_catalog["profiles"] = [
            profile
            for profile in atlas_catalog.get("profiles", [])
            if profile.get("id") not in set(excluded)
        ]
    else:
        reduced_catalog = atlas_catalog
    return select_profile(
        context_contract,
        reduced_catalog,
        family=family,
        semantic_tags=tags,
        seed=seed,
    )


def _area_profile(
    feature: _Feature,
    *,
    landcover: _Feature | None,
    geology: _Feature | None,
    context_contract: Mapping[str, Any],
    atlas_catalog: Mapping[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    if feature.role == "agriculture":
        family = "agriculture_field"
        # A parcel is one globally identified directional surface.  Its crop
        # semantics, profile and seed must not change where it crosses a
        # geology or landcover polygon boundary.
        tags = set(feature.semantic_tags)
        identifiers = feature.feature_id
    elif feature.role == "cliff":
        family = "cliff_surface"
        tags = set(feature.semantic_tags)
        identifiers = feature.feature_id
    else:
        if landcover is None or geology is None:
            raise AssertionError(
                "Natural ground selection requires landcover and geology"
            )
        if not any(tag.startswith("landcover:") for tag in landcover.semantic_tags):
            raise ValueError(
                f"Landcover classification has no approved semantic match: {landcover.feature_id}"
            )
        if not any(tag.startswith("geology:") for tag in geology.semantic_tags):
            raise ValueError(
                f"Geology classification has no approved semantic match: {geology.feature_id}"
            )
        family = "natural_ground"
        tags = set(landcover.semantic_tags | geology.semantic_tags)
        identifiers = ":".join(
            sorted({feature.feature_id, landcover.feature_id, geology.feature_id})
        )
    return _choose_profile(
        context_contract,
        atlas_catalog,
        family=family,
        tags=tags,
        seed=_seed(contract_sha256, identifiers, family),
    )


def _quantized_weights(counts: Counter[int]) -> list[tuple[int, int]]:
    ordered = sorted(counts)
    raw_numerators = {profile: counts[profile] * 255 for profile in ordered}
    weights = {
        profile: raw_numerators[profile] // SAMPLES_PER_CELL for profile in ordered
    }
    remaining = 255 - sum(weights.values())
    remainder_order = sorted(
        ordered,
        key=lambda profile: (-(raw_numerators[profile] % SAMPLES_PER_CELL), profile),
    )
    for profile in remainder_order[:remaining]:
        weights[profile] += 1
    result = [(profile, weights[profile]) for profile in ordered]
    if sum(weight for _profile, weight in result) != 255:
        raise AssertionError("Internal profile weight quantization error")
    return result


def _compile_maps(
    bounds: tuple[float, float, float, float],
    features: Sequence[_Feature],
    *,
    context_contract: Mapping[str, Any],
    atlas_catalog: Mapping[str, Any],
    contract_sha256: str,
    profile_indices: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    landcover = _SpatialIndex(
        [feature for feature in features if feature.layer_id == "landcover"]
    )
    geology = _SpatialIndex(
        [feature for feature in features if feature.layer_id == "geology"]
    )
    agriculture = _SpatialIndex(
        [feature for feature in features if feature.role == "agriculture"]
    )
    cliffs = _SpatialIndex([feature for feature in features if feature.role == "cliff"])
    profile_ids = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.uint8)
    profile_weights = np.zeros_like(profile_ids)
    cache: dict[tuple[str, str, str], int] = {}
    minimum_x, minimum_y, _maximum_x, maximum_y = bounds
    offsets = (0.25, 0.75)
    for row in range(GRID_SIZE):
        cell_top = maximum_y - row * GRID_CELL_SIZE_M
        for column in range(GRID_SIZE):
            cell_left = minimum_x + column * GRID_CELL_SIZE_M
            counts: Counter[int] = Counter()
            for offset_y in offsets:
                for offset_x in offsets:
                    point = Point(
                        cell_left + offset_x * GRID_CELL_SIZE_M,
                        cell_top - offset_y * GRID_CELL_SIZE_M,
                    )
                    landcover_feature = _single_cover(
                        landcover, point, label="landcover"
                    )
                    geology_feature = _single_cover(geology, point, label="geology")
                    agriculture_feature = _optional_single_cover(
                        agriculture, point, label="agriculture"
                    )
                    cliff_feature = _optional_single_cover(cliffs, point, label="cliff")
                    selected_feature = (
                        cliff_feature or agriculture_feature or landcover_feature
                    )
                    family = {
                        "cliff": "cliff_surface",
                        "agriculture": "agriculture_field",
                    }.get(selected_feature.role, "natural_ground")
                    context_key = ":".join(
                        sorted(
                            {
                                selected_feature.feature_id,
                                landcover_feature.feature_id,
                                geology_feature.feature_id,
                            }
                        )
                    )
                    cache_key = family, context_key, selected_feature.feature_id
                    if cache_key not in cache:
                        profile = _area_profile(
                            selected_feature,
                            landcover=landcover_feature,
                            geology=geology_feature,
                            context_contract=context_contract,
                            atlas_catalog=atlas_catalog,
                            contract_sha256=contract_sha256,
                        )
                        cache[cache_key] = profile_indices[profile["id"]]
                    counts[cache[cache_key]] += 1
            quantized = _quantized_weights(counts)
            for channel, (profile_index, weight) in enumerate(quantized):
                profile_ids[row, column, channel] = profile_index
                profile_weights[row, column, channel] = weight
    if not np.all(profile_weights.sum(axis=2, dtype=np.uint16) == 255):
        raise AssertionError("Every ground profile pixel must sum to 255")
    return profile_ids, profile_weights


def _as_line(geometry: BaseGeometry, *, feature_id: str) -> LineString:
    if isinstance(geometry, LineString):
        line = geometry
    elif isinstance(geometry, MultiLineString):
        merged = linemerge(geometry)
        if not isinstance(merged, LineString):
            raise ValueError(
                f"Linear feature is not one continuous chain: {feature_id}"
            )
        line = merged
    else:
        raise ValueError(f"Linear feature requires a line geometry: {feature_id}")
    if line.length <= 0:
        raise ValueError(f"Linear feature has no measurable length: {feature_id}")
    return line


def _canonical_transport_line(feature: _Feature) -> LineString:
    line = _as_line(feature.geometry, feature_id=feature.feature_id)
    scope = _linear_geometry_scope(feature)
    if feature.role == "hydro":
        direction = str(feature.properties.get("flow_direction", "")).casefold()
        if direction not in {"forward", "reverse"}:
            raise ValueError(
                f"Hydrographic flow direction is missing or ambiguous: {feature.feature_id}"
            )
        return LineString(reversed(line.coords)) if direction == "reverse" else line
    if scope == CLIPPED_GEOMETRY_SCOPE:
        direction = str(feature.properties.get("chain_direction", "")).casefold()
        if direction not in {"forward", "reverse"}:
            raise ValueError(
                "A clipped transport feature requires chain_direction forward or "
                f"reverse: {feature.feature_id}"
            )
        return LineString(reversed(line.coords)) if direction == "reverse" else line
    start = tuple(line.coords[0][:2])
    end = tuple(line.coords[-1][:2])
    if start == end:
        raise ValueError(
            f"Closed linear feature requires an explicit network origin: {feature.feature_id}"
        )
    return LineString(reversed(line.coords)) if end < start else line


def _width(feature: _Feature) -> float:
    if "width_m" not in feature.properties:
        raise ValueError(f"Linear feature width is missing: {feature.feature_id}")
    value = _finite_number(feature.properties["width_m"], label="feature width")
    if value <= 0:
        raise ValueError(f"Linear feature width must be positive: {feature.feature_id}")
    return round(value, 3)


def _linear_geometry_scope(feature: _Feature) -> str:
    scope = str(feature.properties.get("geometry_scope", GLOBAL_GEOMETRY_SCOPE)).strip()
    if scope not in VALID_GEOMETRY_SCOPES:
        raise ValueError(
            f"Linear feature geometry_scope must be one of "
            f"{sorted(VALID_GEOMETRY_SCOPES)}: {feature.feature_id}"
        )
    return scope


def _network_chain_id(feature: _Feature) -> str:
    scope = _linear_geometry_scope(feature)
    value = feature.properties.get("network_chain_id")
    if value is None and scope == GLOBAL_GEOMETRY_SCOPE:
        return feature.feature_id
    return _global_feature_id(value)


def _global_abscissa_start(feature: _Feature) -> float:
    scope = _linear_geometry_scope(feature)
    value = feature.properties.get("global_abscissa_start_m")
    if value is None:
        if scope == CLIPPED_GEOMETRY_SCOPE:
            raise ValueError(
                "A clipped linear feature requires global_abscissa_start_m: "
                f"{feature.feature_id}"
            )
        return 0.0
    result = _finite_number(value, label="global linear abscissa origin")
    if result < 0.0:
        raise ValueError(
            f"global_abscissa_start_m must be non-negative: {feature.feature_id}"
        )
    return result


def _global_uv_origin(feature: _Feature, line: LineString) -> tuple[float, float]:
    scope = _linear_geometry_scope(feature)
    value = feature.properties.get("global_uv_origin_l93_m")
    if value is None:
        if scope == CLIPPED_GEOMETRY_SCOPE:
            raise ValueError(
                "A clipped linear feature requires global_uv_origin_l93_m: "
                f"{feature.feature_id}"
            )
        return float(line.coords[0][0]), float(line.coords[0][1])
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"global_uv_origin_l93_m must contain two coordinates: {feature.feature_id}"
        )
    return (
        _finite_number(value[0], label="global UV origin easting"),
        _finite_number(value[1], label="global UV origin northing"),
    )


def _global_abscissa(feature: _Feature, local_abscissa_m: float) -> float:
    return _global_abscissa_start(feature) + local_abscissa_m


def _line_family(role: str) -> str:
    return {
        "path": "path_surface",
        "road": "road_surface",
        "rail": "railway_bed",
        "hydro": "watercourse",
    }[role]


def _line_schedule(
    feature: _Feature,
    line: LineString,
    *,
    context_contract: Mapping[str, Any],
    atlas_catalog: Mapping[str, Any],
    contract_sha256: str,
) -> list[dict[str, Any]]:
    family = _line_family(feature.role)
    chain_id = _network_chain_id(feature)
    global_start_m = _global_abscissa_start(feature)
    global_end_m = global_start_m + line.length
    schedule = []
    # A globally seeded, non-zero millimetre phase keeps profile transitions
    # away from the 500 m tile lattice.  The same feature therefore has the
    # same profile on both sides of every tile edge while still changing on a
    # deterministic 250 m schedule.
    phase_digest = hashlib.sha256(
        f"{contract_sha256}\0{chain_id}\0{family}\0phase".encode("utf-8")
    ).digest()
    phase_m = (1 + int.from_bytes(phase_digest[:4], "little") % 249_999) / 1_000.0
    if global_start_m < phase_m:
        segment_index = 0
        next_boundary_m = phase_m
    else:
        segment_index = 1 + math.floor(
            (global_start_m - phase_m + 1.0e-9) / LINEAR_SEGMENT_LENGTH_M
        )
        next_boundary_m = phase_m + segment_index * LINEAR_SEGMENT_LENGTH_M
    cursor_m = global_start_m
    intervals: list[tuple[int, float, float]] = []
    while cursor_m < global_end_m - 1.0e-9:
        end_m = min(global_end_m, next_boundary_m)
        if end_m <= cursor_m + 1.0e-9:
            segment_index += 1
            next_boundary_m += LINEAR_SEGMENT_LENGTH_M
            continue
        intervals.append((segment_index, cursor_m, end_m))
        cursor_m = end_m
        if cursor_m >= next_boundary_m - 1.0e-9:
            segment_index += 1
            next_boundary_m += LINEAR_SEGMENT_LENGTH_M
    profiles_by_index: dict[int, str] = {}
    previous: list[str] = []
    maximum_index = max(index for index, _start, _end in intervals)
    for index in range(maximum_index + 1):
        profile = _choose_profile(
            context_contract,
            atlas_catalog,
            family=family,
            tags=set(feature.semantic_tags),
            seed=_seed(contract_sha256, chain_id, family, str(index)),
            excluded=previous[-2:],
        )
        profiles_by_index[index] = profile["id"]
        previous.append(profile["id"])
    for index, start_m, end_m in intervals:
        schedule.append(
            {
                "index": index,
                "start_m": start_m,
                "end_m": end_m,
                "local_start_m": start_m - global_start_m,
                "local_end_m": end_m - global_start_m,
                "profile_id": profiles_by_index[index],
            }
        )
    return schedule


def _line_parts(geometry: BaseGeometry) -> list[LineString]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry] if geometry.length > 0 else []
    if isinstance(geometry, MultiLineString):
        return [part for part in geometry.geoms if part.length > 0]
    if isinstance(geometry, GeometryCollection):
        return [part for child in geometry.geoms for part in _line_parts(child)]
    return []


def _round_coordinate(value: float) -> float:
    result = round(float(value), 3)
    return 0.0 if result == 0 else result


def _coordinate_payload(value: Any) -> Any:
    if isinstance(value, (float, int)):
        return _round_coordinate(float(value))
    if isinstance(value, (tuple, list)):
        return [_coordinate_payload(item) for item in value]
    return value


def _geometry_payload(
    geometry: BaseGeometry, *, preserve_line_order: bool = False
) -> dict[str, Any]:
    canonical = geometry if preserve_line_order else geometry.normalize()
    payload = mapping(canonical)
    return {
        "type": payload["type"],
        "coordinates": _coordinate_payload(payload["coordinates"]),
    }


def _tangent_angle(line: LineString, abscissa_m: float) -> float:
    epsilon = min(0.25, line.length / 4.0)
    before = line.interpolate(max(0.0, abscissa_m - epsilon))
    after = line.interpolate(min(line.length, abscissa_m + epsilon))
    if before.equals(after):
        raise ValueError("Cannot derive a stable linear feature orientation")
    return _normalize_angle(
        math.degrees(math.atan2(after.y - before.y, after.x - before.x))
    )


def _linear_overlays(
    feature: _Feature,
    tile_geometry: Polygon,
    *,
    context_contract: Mapping[str, Any],
    atlas_catalog: Mapping[str, Any],
    contract_sha256: str,
) -> tuple[list[dict[str, Any]], LineString, list[dict[str, Any]]]:
    line = _canonical_transport_line(feature)
    width_m = _width(feature)
    schedule = _line_schedule(
        feature,
        line,
        context_contract=context_contract,
        atlas_catalog=atlas_catalog,
        contract_sha256=contract_sha256,
    )
    chain_id = _network_chain_id(feature)
    global_uv_origin = _global_uv_origin(feature, line)
    global_abscissa_start_m = _global_abscissa_start(feature)
    output = []
    for segment in schedule:
        complete_segment = substring(
            line, segment["local_start_m"], segment["local_end_m"]
        )
        for part in _line_parts(complete_segment.intersection(tile_geometry)):
            local_start_m = line.project(Point(part.coords[0]))
            local_end_m = line.project(Point(part.coords[-1]))
            if local_end_m < local_start_m:
                part = LineString(reversed(part.coords))
                local_start_m, local_end_m = local_end_m, local_start_m
            start_m = global_abscissa_start_m + local_start_m
            end_m = global_abscissa_start_m + local_end_m
            local_midpoint = (local_start_m + local_end_m) / 2.0
            output.append(
                {
                    "feature_id": feature.feature_id,
                    "source_layer": feature.layer_id,
                    "role": feature.role,
                    "priority": ROLE_PRIORITY[feature.role],
                    "profile_id": segment["profile_id"],
                    "width_m": width_m,
                    "orientation_deg": _tangent_angle(line, local_midpoint),
                    "abscissa_m": [round(start_m, 3), round(end_m, 3)],
                    "uv_origin_l93_m": [
                        _round_coordinate(global_uv_origin[0]),
                        _round_coordinate(global_uv_origin[1]),
                    ],
                    "network_chain_id": chain_id,
                    "global_segment_index": segment["index"],
                    "uv_seed": _seed(
                        contract_sha256,
                        chain_id,
                        segment["profile_id"],
                        str(segment["index"]),
                    ),
                    "geometry_l93_m": _geometry_payload(part, preserve_line_order=True),
                }
            )
    return output, line, schedule


def _polygon_orientation(feature: _Feature) -> float:
    declared = feature.properties.get(
        "aspect_deg" if feature.role == "cliff" else "orientation_deg"
    )
    if declared is not None:
        return _normalize_angle(_finite_number(declared, label="surface orientation"))
    if feature.role == "cliff":
        raise ValueError(f"Cliff aspect is missing: {feature.feature_id}")
    rectangle = feature.geometry.minimum_rotated_rectangle
    if not isinstance(rectangle, Polygon):
        raise ValueError(f"Cannot derive parcel orientation: {feature.feature_id}")
    coordinates = list(rectangle.exterior.coords)
    edges = []
    for start, end in zip(coordinates, coordinates[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        edges.append(
            (math.hypot(dx, dy), _normalize_angle(math.degrees(math.atan2(dy, dx))))
        )
    lengths = sorted({round(length, 6) for length, _angle in edges}, reverse=True)
    if len(lengths) < 2 or math.isclose(lengths[0], lengths[1], abs_tol=1e-6):
        raise ValueError(f"Parcel orientation is ambiguous: {feature.feature_id}")
    return min(
        angle
        for length, angle in edges
        if math.isclose(length, lengths[0], abs_tol=1e-6)
    )


def _polygon_width(geometry: BaseGeometry) -> float:
    rectangle = geometry.minimum_rotated_rectangle
    if not isinstance(rectangle, Polygon):
        return 0.0
    coordinates = list(rectangle.exterior.coords)
    lengths = [
        math.hypot(end[0] - start[0], end[1] - start[1])
        for start, end in zip(coordinates, coordinates[1:])
    ]
    return round(min(lengths), 3)


def _area_overlays(
    feature: _Feature,
    tile_geometry: Polygon,
    *,
    land_parcels: Sequence[_Feature] = (),
    context_contract: Mapping[str, Any],
    atlas_catalog: Mapping[str, Any],
    contract_sha256: str,
) -> list[dict[str, Any]]:
    clipped = feature.geometry.intersection(tile_geometry)
    if clipped.is_empty or clipped.area <= 0:
        return []
    if feature.role == "hydro":
        profile = _choose_profile(
            context_contract,
            atlas_catalog,
            family="watercourse",
            tags=set(feature.semantic_tags),
            seed=_seed(contract_sha256, feature.feature_id, "watercourse", "area"),
        )
        declared_orientation = feature.properties.get("orientation_deg")
        orientation = (
            None
            if declared_orientation is None
            else _normalize_angle(
                _finite_number(declared_orientation, label="surface orientation")
            )
        )
    else:
        profile = _area_profile(
            feature,
            landcover=None,
            geology=None,
            context_contract=context_contract,
            atlas_catalog=atlas_catalog,
            contract_sha256=contract_sha256,
        )
        orientation = _polygon_orientation(feature)
    linked_land_parcel_ids: list[str] | None = None
    if feature.role == "agriculture":
        linked_land_parcel_ids = sorted(
            parcel.feature_id
            for parcel in land_parcels
            if feature.geometry.intersection(parcel.geometry).area > 0.0
        )
        if not linked_land_parcel_ids:
            raise ValueError(
                "Agricultural surface has no approved land parcel link: "
                f"{feature.feature_id}"
            )
    anchor = feature.geometry.representative_point()
    overlay = {
        "feature_id": feature.feature_id,
        "source_layer": feature.layer_id,
        "role": feature.role,
        "priority": ROLE_PRIORITY[feature.role],
        "profile_id": profile["id"],
        "width_m": _polygon_width(feature.geometry),
        "orientation_deg": orientation,
        "abscissa_m": None,
        "uv_origin_l93_m": [
            _round_coordinate(anchor.x),
            _round_coordinate(anchor.y),
        ],
        "uv_seed": _seed(contract_sha256, feature.feature_id, profile["id"], "area"),
        "geometry_l93_m": _geometry_payload(clipped),
    }
    if linked_land_parcel_ids is not None:
        overlay["land_parcel_feature_ids"] = linked_land_parcel_ids
        overlay["land_parcel_link_policy"] = "positive_area_intersection"
    return [overlay]


def _schedule_profile_at(
    schedule: Sequence[Mapping[str, Any]], abscissa_m: float
) -> str:
    for segment in schedule:
        if (
            float(segment["start_m"]) - 1e-9
            <= abscissa_m
            <= float(segment["end_m"]) + 1e-9
        ):
            return str(segment["profile_id"])
    raise AssertionError("No linear profile schedule segment covers an intersection")


def _point_parts(geometry: BaseGeometry) -> list[Point]:
    if isinstance(geometry, Point):
        return [geometry]
    if geometry.geom_type == "MultiPoint":
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return [point for child in geometry.geoms for point in _point_parts(child)]
    return []


def _surface_crossing_points(geometry: BaseGeometry) -> list[Point]:
    points = _point_parts(geometry)
    if points:
        return points
    if isinstance(geometry, LineString):
        return (
            [geometry.interpolate(0.5, normalized=True)] if geometry.length > 0 else []
        )
    if isinstance(geometry, MultiLineString):
        return [
            part.interpolate(0.5, normalized=True)
            for part in geometry.geoms
            if part.length > 0
        ]
    if isinstance(geometry, GeometryCollection):
        return [
            point
            for child in geometry.geoms
            for point in _surface_crossing_points(child)
        ]
    return []


def _crossing_overlays(
    linear: Mapping[str, tuple[_Feature, LineString, list[dict[str, Any]]]],
    hydro_surfaces: Mapping[str, _Feature],
    tile_geometry: Polygon,
    overrides: Sequence[Mapping[str, Any]],
    *,
    contract_sha256: str,
) -> list[dict[str, Any]]:
    override_by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    for override in overrides:
        pair = (
            _global_feature_id(override.get("transport_feature_id")),
            _global_feature_id(override.get("hydro_feature_id")),
        )
        if pair in override_by_pair:
            raise ValueError(f"Duplicate crossing override: {pair[0]} / {pair[1]}")
        kind = str(override.get("kind", ""))
        if kind not in VALID_CROSSING_KINDS:
            raise ValueError(f"Unsupported crossing override kind: {kind}")
        override_by_pair[pair] = override

    output = []
    transports = [
        value for value in linear.values() if value[0].role in TRANSPORT_ROLES
    ]
    line_waters = [value for value in linear.values() if value[0].role == "hydro"]
    for transport, transport_line, transport_schedule in transports:
        for water, water_line, _water_schedule in line_waters:
            intersection = transport_line.intersection(water_line).intersection(
                tile_geometry
            )
            if intersection.is_empty:
                continue
            points = _point_parts(intersection)
            if not points:
                raise ValueError(
                    f"Ambiguous non-point transport/hydro crossing: {transport.feature_id} / {water.feature_id}"
                )
            pair = (transport.feature_id, water.feature_id)
            override = override_by_pair.get(pair)
            if override is None:
                raise ValueError(
                    f"Missing crossing override: {transport.feature_id} / {water.feature_id}"
                )
            for point_index, point in enumerate(
                sorted(points, key=lambda value: (value.x, value.y))
            ):
                transport_local_abscissa = transport_line.project(point)
                water_local_abscissa = water_line.project(point)
                transport_abscissa = _global_abscissa(
                    transport, transport_local_abscissa
                )
                water_abscissa = _global_abscissa(water, water_local_abscissa)
                profile_id = _schedule_profile_at(
                    transport_schedule, transport_abscissa
                )
                transport_uv_origin = _global_uv_origin(transport, transport_line)
                crossing_id = _seed(
                    contract_sha256,
                    transport.feature_id,
                    water.feature_id,
                    str(override["kind"]),
                    f"{point.x:.3f}",
                    f"{point.y:.3f}",
                )
                output.append(
                    {
                        "feature_id": f"crossing:{crossing_id}",
                        "source_layer": "zone-overrides.v1",
                        "role": "crossing_override",
                        "priority": ROLE_PRIORITY["crossing_override"],
                        "profile_id": profile_id,
                        "width_m": _width(transport),
                        "orientation_deg": _tangent_angle(
                            transport_line, transport_local_abscissa
                        ),
                        "abscissa_m": [
                            round(transport_abscissa, 3),
                            round(water_abscissa, 3),
                        ],
                        "uv_origin_l93_m": [
                            _round_coordinate(transport_uv_origin[0]),
                            _round_coordinate(transport_uv_origin[1]),
                        ],
                        "network_chain_id": _network_chain_id(transport),
                        "uv_seed": _seed(
                            contract_sha256,
                            _network_chain_id(transport),
                            _network_chain_id(water),
                            profile_id,
                        ),
                        "crossing_kind": override["kind"],
                        "transport_feature_id": transport.feature_id,
                        "hydro_feature_id": water.feature_id,
                        "point_index": point_index,
                        "geometry_l93_m": _geometry_payload(point),
                    }
                )
        for water in hydro_surfaces.values():
            intersection = transport_line.intersection(water.geometry).intersection(
                tile_geometry
            )
            if intersection.is_empty:
                continue
            points = _surface_crossing_points(intersection)
            if not points:
                raise ValueError(
                    f"Ambiguous transport/hydro-surface crossing: {transport.feature_id} / {water.feature_id}"
                )
            pair = (transport.feature_id, water.feature_id)
            override = override_by_pair.get(pair)
            if override is None:
                raise ValueError(
                    f"Missing crossing override: {transport.feature_id} / {water.feature_id}"
                )
            for point_index, point in enumerate(
                sorted(points, key=lambda value: (value.x, value.y))
            ):
                transport_local_abscissa = transport_line.project(point)
                transport_abscissa = _global_abscissa(
                    transport, transport_local_abscissa
                )
                profile_id = _schedule_profile_at(
                    transport_schedule, transport_abscissa
                )
                transport_uv_origin = _global_uv_origin(transport, transport_line)
                crossing_id = _seed(
                    contract_sha256,
                    transport.feature_id,
                    water.feature_id,
                    str(override["kind"]),
                    f"{point.x:.3f}",
                    f"{point.y:.3f}",
                )
                output.append(
                    {
                        "feature_id": f"crossing:{crossing_id}",
                        "source_layer": "zone-overrides.v1",
                        "role": "crossing_override",
                        "priority": ROLE_PRIORITY["crossing_override"],
                        "profile_id": profile_id,
                        "width_m": _width(transport),
                        "orientation_deg": _tangent_angle(
                            transport_line, transport_local_abscissa
                        ),
                        "abscissa_m": [round(transport_abscissa, 3), None],
                        "uv_origin_l93_m": [
                            _round_coordinate(transport_uv_origin[0]),
                            _round_coordinate(transport_uv_origin[1]),
                        ],
                        "network_chain_id": _network_chain_id(transport),
                        "uv_seed": _seed(
                            contract_sha256,
                            _network_chain_id(transport),
                            water.feature_id,
                            profile_id,
                        ),
                        "crossing_kind": override["kind"],
                        "transport_feature_id": transport.feature_id,
                        "hydro_feature_id": water.feature_id,
                        "point_index": point_index,
                        "geometry_l93_m": _geometry_payload(point),
                    }
                )
    # The caller may pass the complete immutable zone override set to every
    # tile.  Overrides whose intersection is outside this tile are valid and
    # intentionally produce no local overlay.
    return output


def _compile_overlays(
    bounds: tuple[float, float, float, float],
    features: Sequence[_Feature],
    crossing_overrides: Sequence[Mapping[str, Any]],
    *,
    context_contract: Mapping[str, Any],
    atlas_catalog: Mapping[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    tile_geometry = box(*bounds)
    overlays: list[dict[str, Any]] = []
    linear: dict[str, tuple[_Feature, LineString, list[dict[str, Any]]]] = {}
    hydro_surfaces: dict[str, _Feature] = {}
    land_parcels = tuple(
        feature for feature in features if feature.layer_id == "land_parcels"
    )
    for feature in features:
        if feature.role in {"agriculture", "cliff"}:
            overlays.extend(
                _area_overlays(
                    feature,
                    tile_geometry,
                    land_parcels=land_parcels,
                    context_contract=context_contract,
                    atlas_catalog=atlas_catalog,
                    contract_sha256=contract_sha256,
                )
            )
        elif feature.role == "hydro" and feature.geometry.geom_type in {
            "Polygon",
            "MultiPolygon",
        }:
            overlays.extend(
                _area_overlays(
                    feature,
                    tile_geometry,
                    context_contract=context_contract,
                    atlas_catalog=atlas_catalog,
                    contract_sha256=contract_sha256,
                )
            )
            hydro_surfaces[feature.feature_id] = feature
        elif feature.role in LINEAR_ROLES:
            segments, line, schedule = _linear_overlays(
                feature,
                tile_geometry,
                context_contract=context_contract,
                atlas_catalog=atlas_catalog,
                contract_sha256=contract_sha256,
            )
            overlays.extend(segments)
            linear[feature.feature_id] = feature, line, schedule
    overlays.extend(
        _crossing_overlays(
            linear,
            hydro_surfaces,
            tile_geometry,
            crossing_overrides,
            contract_sha256=contract_sha256,
        )
    )
    overlays.sort(
        key=lambda overlay: (
            int(overlay["priority"]),
            str(overlay["feature_id"]),
            -1.0 if overlay["abscissa_m"] is None else float(overlay["abscissa_m"][0]),
            str(overlay["profile_id"]),
        )
    )
    return {
        "schema": OVERLAY_SCHEMA,
        "crs": CRS,
        "coordinate_precision_m": COORDINATE_PRECISION_M,
        "priority_low_to_high": list(ROLE_PRIORITY),
        "feature_count": len(overlays),
        "features": overlays,
    }


def compile_tile_composition(
    *,
    bounds_l93_m: Sequence[float],
    features: Iterable[Mapping[str, Any]],
    context_contract: Mapping[str, Any],
    atlas_catalog: Mapping[str, Any],
    contract_sha256: str,
    context_sha256: str,
    crossing_overrides: Sequence[Mapping[str, Any]] = (),
) -> TileComposition:
    """Compile one tile without accessing paths, clocks or process state."""

    bounds = _validate_bounds(bounds_l93_m)
    if not contract_sha256:
        raise ValueError("A ground composition contract SHA-256 is required")
    if not context_sha256:
        raise ValueError("A global ground context snapshot SHA-256 is required")
    if atlas_catalog.get("schema") != "fireviewer.ground-surface-atlas-library.v3":
        raise ValueError("Unsupported ground surface atlas catalog")
    catalog_sha256 = str(atlas_catalog.get("catalog_sha256", "")).strip()
    if not catalog_sha256:
        raise ValueError("The atlas catalog must be hash-locked")
    prepared = _prepare_features(features)
    normalized_crossing_overrides = _normalize_crossing_overrides(crossing_overrides)
    crossing_overrides_sha256 = _sha256_bytes(
        _canonical_bytes(
            {
                "schema": "fireviewer.zone-crossing-overrides.v1",
                "overrides": normalized_crossing_overrides,
            }
        )
    )
    context_feature_set_sha256 = _sha256_bytes(
        _canonical_bytes(_context_input_payload(prepared, context_contract))
    )
    # SCHEMA is the algorithm version and is automatically prepended by
    # _seed().  The namespace explicitly binds the other three immutable
    # dependencies used by every profile choice and UV seed.
    seed_namespace = _seed(
        contract_sha256,
        catalog_sha256,
        context_sha256,
    )
    profile_table, profile_indices = _profile_table(atlas_catalog)
    profile_ids, profile_weights = _compile_maps(
        bounds,
        prepared,
        context_contract=context_contract,
        atlas_catalog=atlas_catalog,
        contract_sha256=seed_namespace,
        profile_indices=profile_indices,
    )
    overlays = _compile_overlays(
        bounds,
        prepared,
        normalized_crossing_overrides,
        context_contract=context_contract,
        atlas_catalog=atlas_catalog,
        contract_sha256=seed_namespace,
    )
    dependency_ids = sorted(feature.feature_id for feature in prepared)
    manifest = {
        "schema": SCHEMA,
        "crs": CRS,
        "bounds_l93_m": list(bounds),
        "grid": {
            "cell_size_m": GRID_CELL_SIZE_M,
            "size_px": [GRID_SIZE, GRID_SIZE],
            "sample_pattern": "four_quarter_cell_centres",
            "row_order": "north_to_south",
            "column_order": "west_to_east",
            "maximum_profiles_per_cell": 4,
        },
        "profile_table": profile_table,
        "profile_id_encoding": "zero_based_profile_table_index_rgba8",
        "unused_profile_id_policy": "ignored_when_corresponding_weight_is_zero",
        "profile_weight_encoding": "rgba8_sum_exactly_255",
        "overlay_schema": OVERLAY_SCHEMA,
        "priority_low_to_high": list(ROLE_PRIORITY),
        "orthophoto_dependency": "forbidden",
        "seed_inputs": [
            "composition_schema",
            "ground_contract_sha256",
            "atlas_catalog_sha256",
            "context_sha256",
            "global_feature_id",
            "global_curvilinear_segment_index",
        ],
        "forbidden_seed_inputs": ["tile_id", "path", "clock", "worker_order"],
        "ground_contract_sha256": contract_sha256,
        "atlas_catalog_sha256": catalog_sha256,
        "context_sha256": context_sha256,
        "seed_namespace_sha256": seed_namespace,
        "context_feature_ids": dependency_ids,
        "context_feature_set_sha256": context_feature_set_sha256,
        "crossing_override_count": len(normalized_crossing_overrides),
        "crossing_overrides_sha256": crossing_overrides_sha256,
        "land_parcel_link_policy": "positive_area_intersection_required_for_agriculture",
    }
    return TileComposition(
        manifest=manifest,
        overlays=overlays,
        profile_ids=profile_ids,
        profile_weights=profile_weights,
    )


def _png_bytes(values: np.ndarray) -> bytes:
    if values.shape != (GRID_SIZE, GRID_SIZE, 4) or values.dtype != np.uint8:
        raise ValueError("Ground composition maps must be 100x100 RGBA8 arrays")
    output = BytesIO()
    Image.fromarray(values, mode="RGBA").save(
        output, format="PNG", optimize=False, compress_level=9
    )
    return output.getvalue()


def _gzip_bytes(payload: Mapping[str, Any]) -> bytes:
    return gzip.compress(_canonical_bytes(payload), compresslevel=9, mtime=0)


def serialized_outputs(composition: TileComposition) -> dict[str, bytes]:
    """Return all stable bytes without writing to disk."""

    profile_ids = _png_bytes(composition.profile_ids)
    profile_weights = _png_bytes(composition.profile_weights)
    overlays = _gzip_bytes(composition.overlays)
    manifest = dict(composition.manifest)
    manifest["outputs"] = {
        "ground-profile-ids.png": {
            "sha256": _sha256_bytes(profile_ids),
            "bytes": len(profile_ids),
        },
        "ground-profile-weights.png": {
            "sha256": _sha256_bytes(profile_weights),
            "bytes": len(profile_weights),
        },
        "surface-overlays.json.gz": {
            "sha256": _sha256_bytes(overlays),
            "bytes": len(overlays),
        },
    }
    return {
        "ground-profile-ids.png": profile_ids,
        "ground-profile-weights.png": profile_weights,
        "surface-overlays.json.gz": overlays,
        "tile-composition.json.gz": _gzip_bytes(manifest),
    }


def write_tile_composition(
    composition: TileComposition, output_root: Path
) -> dict[str, Any]:
    """Atomically write one immutable composition package."""

    output_root.mkdir(parents=True, exist_ok=True)
    outputs = serialized_outputs(composition)
    for name, content in outputs.items():
        destination = output_root / name
        if destination.exists() and (
            not destination.is_file() or destination.read_bytes() != content
        ):
            raise FileExistsError(
                f"Refusing to overwrite a different tile composition output: {destination}"
            )
    for name, content in outputs.items():
        destination = output_root / name
        if destination.is_file():
            continue
        temporary = output_root / f".{name}.tmp"
        temporary.write_bytes(content)
        temporary.replace(destination)
    return {
        "schema": SCHEMA,
        "outputs": {
            name: {"sha256": _sha256_bytes(content), "bytes": len(content)}
            for name, content in sorted(outputs.items())
        },
    }
