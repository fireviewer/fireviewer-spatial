"""Pure geospatial helpers for the Blender control-scene builder.

This module deliberately has no dependency on ``bpy`` so that coordinate,
GeoJSON and metadata behaviour can be tested with the regular project Python.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

TARGET_CRS = "EPSG:2154"


def lambert93_validation(crs: Any) -> str | None:
    """Return how a raster CRS was proven to be Lambert-93.

    Some official IGN WMS GeoTIFF files expose the correct Lambert-93
    projection but an unnamed datum, making ``to_epsg()`` return ``None``.
    Such files are accepted only when every defining projection parameter
    matches Lambert-93.
    """

    if crs is None:
        return None
    if crs.to_epsg() == 2154:
        return "epsg_authority"
    values = crs.to_dict()
    expected_text = {"proj": "lcc", "units": "m"}
    expected_numbers = {
        "lat_0": 46.5,
        "lon_0": 3.0,
        "lat_1": 49.0,
        "lat_2": 44.0,
        "x_0": 700000.0,
        "y_0": 6600000.0,
    }
    if any(
        str(values.get(key, "")).casefold() != value
        for key, value in expected_text.items()
    ):
        return None
    for key, expected in expected_numbers.items():
        try:
            actual = float(values[key])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-8):
            return None
    return "lambert93_projection_parameters"


@dataclass(frozen=True)
class PolygonFeature:
    """A polygonal feature reprojected to the target CRS."""

    feature_id: str
    geometry: Any
    properties: Mapping[str, Any]


@dataclass(frozen=True)
class LineFeature:
    """A linear feature reprojected to the target CRS."""

    feature_id: str
    geometry: Any
    properties: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_absolute_local_paths(value: Any, location: str = "root") -> list[str]:
    """Return JSON locations containing absolute Windows or POSIX paths."""

    matches: list[str] = []
    if isinstance(value, Mapping):
        for key, member in value.items():
            matches.extend(find_absolute_local_paths(member, f"{location}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, member in enumerate(value):
            matches.extend(find_absolute_local_paths(member, f"{location}[{index}]"))
    elif isinstance(value, str) and (
        PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()
    ):
        matches.append(location)
    return matches


def _iter_coordinate_pairs(value: Any) -> Iterator[tuple[float, float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return
    if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
        yield float(value[0]), float(value[1])
        return
    for item in value:
        yield from _iter_coordinate_pairs(item)


def _declared_geojson_crs(payload: Mapping[str, Any]) -> str | None:
    crs = payload.get("crs")
    if not isinstance(crs, Mapping):
        return None
    properties = crs.get("properties")
    if not isinstance(properties, Mapping):
        return None
    name = str(properties.get("name", ""))
    match = re.search(r"EPSG(?::|::|/)(\d+)", name, re.IGNORECASE)
    if not match:
        return None
    return f"EPSG:{match.group(1)}"


def infer_geojson_crs(payload: Mapping[str, Any]) -> str:
    """Infer only the two coordinate systems accepted by this pipeline.

    RFC 7946 GeoJSON is WGS84 and normally omits the deprecated ``crs``
    member. Lambert-93 extracts often still include that member. Ambiguous or
    unexpected coordinate ranges are rejected instead of silently misplaced.
    """

    declared = _declared_geojson_crs(payload)
    if declared:
        return declared

    geometries: list[Mapping[str, Any]] = []
    if payload.get("type") == "FeatureCollection":
        for feature in payload.get("features", []):
            if isinstance(feature, Mapping) and isinstance(
                feature.get("geometry"), Mapping
            ):
                geometries.append(feature["geometry"])
    elif payload.get("type") == "Feature" and isinstance(
        payload.get("geometry"), Mapping
    ):
        geometries.append(payload["geometry"])
    elif isinstance(payload.get("coordinates"), Sequence):
        geometries.append(payload)

    for geometry in geometries:
        pair = next(_iter_coordinate_pairs(geometry.get("coordinates")), None)
        if pair is None:
            continue
        x, y = pair
        if -180.0 <= x <= 180.0 and -90.0 <= y <= 90.0:
            return "EPSG:4326"
        if 0.0 <= x <= 2_000_000.0 and 5_000_000.0 <= y <= 8_000_000.0:
            return TARGET_CRS
        raise ValueError(
            f"Cannot infer GeoJSON CRS from coordinate ({x}, {y}); pass an explicit --*-crs value"
        )
    raise ValueError("Cannot infer CRS from an empty GeoJSON payload")


def iter_polygon_parts(geometry: Any) -> Iterator[Any]:
    """Yield Polygon members from Polygon, MultiPolygon or collections."""

    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
        return
    if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for member in geometry.geoms:
            yield from iter_polygon_parts(member)


def iter_line_parts(geometry: Any) -> Iterator[Any]:
    """Yield LineString members from line, multi-line or collections."""

    if geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        yield geometry
        return
    if geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
        for member in geometry.geoms:
            yield from iter_line_parts(member)


def read_polygon_features(
    path: Path,
    source_crs: str = "auto",
    target_crs: str = TARGET_CRS,
    swap_xy: bool = False,
) -> tuple[list[PolygonFeature], str]:
    """Load, validate and reproject polygonal GeoJSON features."""

    from pyproj import CRS, Transformer
    from shapely import make_valid
    from shapely.geometry import shape
    from shapely.ops import transform

    payload = json.loads(path.read_text(encoding="utf-8"))
    detected_crs = (
        infer_geojson_crs(payload) if source_crs.lower() == "auto" else source_crs
    )
    source = CRS.from_user_input(detected_crs)
    target = CRS.from_user_input(target_crs)
    transformer = (
        None
        if source == target
        else Transformer.from_crs(source, target, always_xy=True)
    )

    if payload.get("type") == "FeatureCollection":
        raw_features = payload.get("features", [])
    elif payload.get("type") == "Feature":
        raw_features = [payload]
    else:
        raw_features = [{"type": "Feature", "geometry": payload, "properties": {}}]

    features: list[PolygonFeature] = []
    for index, feature in enumerate(raw_features):
        if not isinstance(feature, Mapping) or not feature.get("geometry"):
            continue
        geometry = shape(feature["geometry"])
        if swap_xy:
            geometry = transform(
                lambda x, y, z=None: (y, x) if z is None else (y, x, z),
                geometry,
            )
        if transformer is not None:
            geometry = transform(transformer.transform, geometry)
        if not geometry.is_valid:
            geometry = make_valid(geometry)
        properties = feature.get("properties") or {}
        feature_id = str(feature.get("id", properties.get("id", index)))
        for part_index, polygon in enumerate(iter_polygon_parts(geometry)):
            if polygon.is_empty or polygon.area <= 0:
                continue
            features.append(
                PolygonFeature(
                    feature_id=f"{feature_id}:{part_index}",
                    geometry=polygon,
                    properties=properties,
                )
            )

    features.sort(
        key=lambda item: (item.feature_id, item.geometry.bounds, item.geometry.wkb_hex)
    )
    return features, source.to_string()


def read_line_features(
    path: Path,
    source_crs: str = "auto",
    target_crs: str = TARGET_CRS,
) -> tuple[list[LineFeature], str]:
    """Load and reproject LineString/MultiLineString GeoJSON features."""

    from pyproj import CRS, Transformer
    from shapely.geometry import shape
    from shapely.ops import transform

    payload = json.loads(path.read_text(encoding="utf-8"))
    detected_crs = (
        infer_geojson_crs(payload) if source_crs.lower() == "auto" else source_crs
    )
    source = CRS.from_user_input(detected_crs)
    target = CRS.from_user_input(target_crs)
    transformer = (
        None
        if source == target
        else Transformer.from_crs(source, target, always_xy=True)
    )

    if payload.get("type") == "FeatureCollection":
        raw_features = payload.get("features", [])
    elif payload.get("type") == "Feature":
        raw_features = [payload]
    else:
        raw_features = [{"type": "Feature", "geometry": payload, "properties": {}}]

    features: list[LineFeature] = []
    for index, feature in enumerate(raw_features):
        if not isinstance(feature, Mapping) or not feature.get("geometry"):
            continue
        geometry = shape(feature["geometry"])
        if transformer is not None:
            geometry = transform(transformer.transform, geometry)
        properties = feature.get("properties") or {}
        feature_id = str(feature.get("id", properties.get("id", index)))
        for part_index, line in enumerate(iter_line_parts(geometry)):
            if line.is_empty or line.length <= 0 or len(line.coords) < 2:
                continue
            features.append(
                LineFeature(
                    feature_id=f"{feature_id}:{part_index}",
                    geometry=line,
                    properties=properties,
                )
            )
    features.sort(
        key=lambda item: (item.feature_id, item.geometry.bounds, item.geometry.wkb_hex)
    )
    return features, source.to_string()


def numeric_property(
    properties: Mapping[str, Any], keys: Iterable[str]
) -> float | None:
    """Read the first finite numeric property, accepting French decimal text."""

    folded = {str(key).casefold(): value for key, value in properties.items()}
    for key in keys:
        value = folded.get(key.casefold())
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            number = float(value)
        else:
            match = re.search(r"[-+]?\d+(?:[.,]\d+)?", str(value))
            if not match:
                continue
            number = float(match.group(0).replace(",", "."))
        if math.isfinite(number):
            return number
    return None


def positive_numeric_property(
    properties: Mapping[str, Any], keys: Iterable[str]
) -> float | None:
    """Return the first strictly positive numeric value in priority order."""

    for key in keys:
        value = numeric_property(properties, (key,))
        if value is not None and value > 0:
            return value
    return None


def choose_local_origin(
    bounds: tuple[float, float, float, float],
    minimum_elevation: float,
) -> tuple[float, float, float]:
    """Choose a stable metre-aligned origin near the geometry centre."""

    min_x, min_y, max_x, max_y = bounds

    def nearest_metre(value: float) -> int:
        return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)

    return (
        nearest_metre((min_x + max_x) / 2.0),
        nearest_metre((min_y + max_y) / 2.0),
        math.floor(minimum_elevation),
    )
