"""Compile observed geographic perimeters into fixed USD layers and a timeline.

The source remains geographic JSON/GeoJSON in WGS84.  Every observation is
projected to Lambert-93 and authored as an immutable USD layer.  A separate
canonical JSON timeline records the elapsed real time between observations;
it never interpolates or predicts fire propagation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from pyproj import Transformer
from shapely import constrained_delaunay_triangles
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import transform, unary_union
from shapely.validation import explain_validity

CONTRACT_SCHEMA = "fireviewer.geographic-perimeter-layer-contract.v1"
NORMALIZED_SCHEMA = "fireviewer.geographic-perimeter-observations.v1"
TIMELINE_SCHEMA = "fireviewer.fire-progression-timeline.v1"
MANIFEST_SCHEMA = "fireviewer.geographic-perimeter-layer-package.v1"
STAGE_NAME = "geographic-perimeters.usda"
TIMELINE_NAME = "fire-progression-timeline.json"
SOURCE_NAME = "perimeters.normalized.json"
MANIFEST_NAME = "perimeter-layer.manifest.json"
ARCHIVE_NAME = "fireviewer-perimeter-layer.zip"
ORIGIN_ALIGNMENT_M = 500
MAX_INPUT_BYTES = 50 * 1024 * 1024
MAX_FRAMES = 366
MAX_COORDINATES = 500_000
_L93_LIMITS = (-100_000.0, 5_900_000.0, 1_500_000.0, 7_300_000.0)
_TO_L93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)


class GeographicPerimeterError(RuntimeError):
    """The supplied observations cannot produce a deterministic layer."""


@dataclass(frozen=True, slots=True)
class CompiledPerimeterLayer:
    dataset_id: str
    build_id: str
    normalized_source: dict[str, Any]
    timeline: dict[str, Any]
    stage_text: str
    contract_sha256: str
    compiler_sha256: str
    affected_area_ha: float
    active_area_ha: float


@dataclass(frozen=True, slots=True)
class PerimeterLayerProduct:
    package_root: Path
    archive: Path
    manifest: dict[str, Any]


def _contract_path() -> Path:
    return Path(__file__).with_name("geographic_perimeter_layer_contract.v1.json")


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


def _load_contract() -> dict[str, Any]:
    path = _contract_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GeographicPerimeterError(
            f"contrat de calque invalide: {error}"
        ) from error
    expected = {
        "schema": CONTRACT_SCHEMA,
        "status": "locked",
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise GeographicPerimeterError("contrat de calque absent ou non verrouillé")
    if (
        payload.get("input", {}).get("crs") != "EPSG:4326"
        or payload.get("projection", {}).get("crs") != "EPSG:2154"
        or payload.get("timeline", {}).get("source")
        != "observed timestamps or explicit time_window start/end only"
        or payload.get("timeline", {}).get("explicit_range_representation")
        != "source_start_and_end_preserved_without_inference"
        or payload.get("timeline", {}).get("between_observations") != "undefined"
        or payload.get("timeline", {}).get("prediction") != "forbidden"
        or payload.get("semantics", {}).get("fixed_layers") is not True
        or payload.get("semantics", {}).get("simulation_timeline_data") is not True
    ):
        raise GeographicPerimeterError("contrat de calque incohérent")
    return payload


def _safe_id(value: Any, fallback: str) -> str:
    candidate = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "").strip()).strip("-_")
    return (candidate[:80] or fallback).lower()


def _parse_timestamp(value: Any, frame_index: int) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise GeographicPerimeterError(
            f"observation {frame_index}: timestamp ISO-8601 obligatoire"
        )
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise GeographicPerimeterError(
            f"observation {frame_index}: timestamp ISO-8601 invalide"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise GeographicPerimeterError(
            f"observation {frame_index}: fuseau horaire obligatoire"
        )
    return parsed.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _frame_time_fields(value: Mapping[str, Any], frame_index: int) -> dict[str, str]:
    direct = _first(
        value,
        ("observed_at", "timestamp", "datetime", "date", "time"),
    )
    raw_window = value.get("time_window")
    normalized_window = False
    if raw_window is None and "time_range" in value:
        raw_window = value.get("time_range")
        normalized_window = True
    if raw_window is not None and not isinstance(raw_window, Mapping):
        raise GeographicPerimeterError(
            f"observation {frame_index}: time_window doit être un objet"
        )
    window = raw_window if isinstance(raw_window, Mapping) else {}
    raw_start = _first(window, ("start", "from", "begin", "valid_from"))
    raw_end = _first(window, ("end", "to", "until", "valid_to"))
    if raw_window is not None and (raw_start is None or raw_end is None):
        raise GeographicPerimeterError(
            f"observation {frame_index}: time_window.start et time_window.end obligatoires"
        )
    if direct is None:
        direct = raw_end if raw_end is not None else raw_start
    observed = _parse_timestamp(direct, frame_index)
    start = (
        _parse_timestamp(raw_start, frame_index) if raw_start is not None else observed
    )
    end = _parse_timestamp(raw_end, frame_index) if raw_end is not None else observed
    if end < start:
        raise GeographicPerimeterError(
            f"observation {frame_index}: fin de plage antérieure au début"
        )
    if observed < start or observed > end:
        raise GeographicPerimeterError(
            f"observation {frame_index}: timestamp hors de la plage temporelle"
        )
    temporal_kind = "explicit_interval" if raw_window is not None else "instant"
    if normalized_window:
        recorded_kind = window.get("kind")
        if recorded_kind not in {"instant", "explicit_interval"}:
            raise GeographicPerimeterError(
                f"observation {frame_index}: kind de plage temporelle invalide"
            )
        temporal_kind = str(recorded_kind)
    return {
        "timestamp": _timestamp_text(observed),
        "valid_from": _timestamp_text(start),
        "valid_to": _timestamp_text(end),
        "temporal_kind": temporal_kind,
    }


def _nesting_depth(value: Any) -> int:
    if not isinstance(value, (list, tuple)) or not value:
        return 0
    return 1 + _nesting_depth(value[0])


def _geometry_from_coordinates(value: Sequence[Any]) -> BaseGeometry:
    depth = _nesting_depth(value)
    try:
        if depth == 2:
            return Polygon(value)
        if depth == 3:
            return Polygon(value[0], value[1:])
        if depth == 4:
            return MultiPolygon([Polygon(poly[0], poly[1:]) for poly in value])
    except (TypeError, ValueError, IndexError) as error:
        raise GeographicPerimeterError("coordonnées de polygone invalides") from error
    raise GeographicPerimeterError(
        "coordonnées attendues: anneau, Polygon ou MultiPolygon"
    )


def _coordinate_count(geometry: BaseGeometry) -> int:
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    return sum(
        len(polygon.exterior.coords)
        + sum(len(interior.coords) for interior in polygon.interiors)
        for polygon in polygons
    )


def _validate_wgs84_geometry(geometry: BaseGeometry, label: str) -> BaseGeometry:
    if geometry.is_empty:
        raise GeographicPerimeterError(f"{label}: géométrie vide")
    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise GeographicPerimeterError(
            f"{label}: seuls Polygon/MultiPolygon sont acceptés"
        )
    if not geometry.is_valid:
        raise GeographicPerimeterError(
            f"{label}: polygone invalide ({explain_validity(geometry)})"
        )
    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    if not all(math.isfinite(value) for value in geometry.bounds):
        raise GeographicPerimeterError(f"{label}: coordonnées non finies")
    if min_lon < -180 or max_lon > 180 or min_lat < -90 or max_lat > 90:
        raise GeographicPerimeterError(f"{label}: coordonnées hors WGS84")
    if geometry.area <= 0:
        raise GeographicPerimeterError(f"{label}: surface nulle")
    normalized = geometry.normalize()
    if normalized.geom_type == "Polygon":
        normalized = MultiPolygon([normalized])
    return normalized


def _geometry_from_value(value: Any, label: str) -> BaseGeometry | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if (
            value.get("type") == "Feature"
            or "geometry" in value
            and not value.get("type")
        ):
            value = value.get("geometry")
        elif "polygons" in value:
            value = value.get("polygons")
        elif "coordinates" in value and value.get("type") not in {
            "Polygon",
            "MultiPolygon",
        }:
            value = value.get("coordinates")
    try:
        geometry = (
            shape(value)
            if isinstance(value, Mapping) and value.get("type")
            else _geometry_from_coordinates(value)
        )
    except (TypeError, ValueError, KeyError) as error:
        raise GeographicPerimeterError(f"{label}: géométrie illisible") from error
    return _validate_wgs84_geometry(geometry, label)


def _first(item: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in item and item[name] is not None:
            return item[name]
    return None


def _merge_geometries(
    values: Sequence[BaseGeometry], label: str
) -> BaseGeometry | None:
    if not values:
        return None
    merged = unary_union(values)
    return _validate_wgs84_geometry(merged, label)


def _feature_category(properties: Mapping[str, Any], index: int) -> str:
    raw = _first(properties, ("category", "perimeter_type", "status", "layer", "kind"))
    if raw is None:
        return "affected"
    token = str(raw).strip().lower()
    if any(word in token for word in ("active", "front", "progression", "category2")):
        return "active"
    if any(
        word in token
        for word in ("affected", "touched", "cumulative", "burn", "category1")
    ):
        return "affected"
    raise GeographicPerimeterError(
        f"feature {index}: catégorie inconnue {raw!r}; affected ou active attendu"
    )


def _frames_from_feature_collection(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise GeographicPerimeterError("FeatureCollection sans feature")
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, feature in enumerate(features):
        if not isinstance(feature, Mapping) or feature.get("type") != "Feature":
            raise GeographicPerimeterError(f"feature {index}: objet GeoJSON invalide")
        properties = feature.get("properties")
        if not isinstance(properties, Mapping):
            properties = {}
        time_fields = _frame_time_fields(properties, index)
        key = (
            time_fields["timestamp"],
            time_fields["valid_from"],
            time_fields["valid_to"],
        )
        category = _feature_category(properties, index)
        geometry = _geometry_from_value(feature.get("geometry"), f"feature {index}")
        if geometry is None:
            raise GeographicPerimeterError(f"feature {index}: géométrie absente")
        record = grouped.setdefault(
            key,
            {
                **time_fields,
                "affected_geometries": [],
                "active_geometries": [],
            },
        )
        record[f"{category}_geometries"].append(geometry)
    frames: list[dict[str, Any]] = []
    for record in grouped.values():
        timestamp = record["timestamp"]
        frames.append(
            {
                **{
                    field: record[field]
                    for field in (
                        "timestamp",
                        "valid_from",
                        "valid_to",
                        "temporal_kind",
                    )
                },
                "affected": _merge_geometries(
                    record["affected_geometries"], f"affected {timestamp}"
                ),
                "active": _merge_geometries(
                    record["active_geometries"], f"active {timestamp}"
                ),
            }
        )
    return frames


def _frames_from_timeline(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_frames = payload.get("timeline", payload.get("frames"))
    if not isinstance(raw_frames, list) or not raw_frames:
        raise GeographicPerimeterError("timeline d’observations absente")
    frames: list[dict[str, Any]] = []
    for index, item in enumerate(raw_frames):
        if not isinstance(item, Mapping):
            raise GeographicPerimeterError(f"observation {index}: objet JSON attendu")
        time_fields = _frame_time_fields(item, index)
        affected_value = _first(
            item,
            (
                "affected",
                "touched",
                "cumulative_affected_area",
                "cumulative_perimeter",
                "category1",
                "category1_polygons",
            ),
        )
        active_value = _first(
            item,
            (
                "active",
                "active_zone",
                "active_front",
                "new_affected_area",
                "category2",
                "category2_polygons",
            ),
        )
        affected = _geometry_from_value(affected_value, f"observation {index} affected")
        active = _geometry_from_value(active_value, f"observation {index} active")
        if affected is None and active is None:
            raise GeographicPerimeterError(
                f"observation {index}: aucun périmètre affected/active"
            )
        frames.append(
            {
                **time_fields,
                "affected": affected,
                "active": active,
                "source_frame_id": _first(item, ("frame_id", "id")),
            }
        )
    return frames


def _rounded_mapping(geometry: BaseGeometry | None) -> dict[str, Any] | None:
    if geometry is None:
        return None

    def rounded(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [rounded(item) for item in value]
        if isinstance(value, float):
            return round(value, 10)
        return value

    value = mapping(geometry.normalize())
    return {"type": value["type"], "coordinates": rounded(value["coordinates"])}


def _normalize_dataset(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise GeographicPerimeterError("la source doit être un objet JSON")
    if payload.get("type") == "FeatureCollection":
        raw_frames = _frames_from_feature_collection(payload)
    else:
        raw_frames = _frames_from_timeline(payload)
    if len(raw_frames) > MAX_FRAMES:
        raise GeographicPerimeterError(
            f"{len(raw_frames)} observations; limite: {MAX_FRAMES}"
        )
    ordered = sorted(
        raw_frames,
        key=lambda item: (item["valid_from"], item["timestamp"], item["valid_to"]),
    )
    timestamps = [item["timestamp"] for item in ordered]
    if len(timestamps) != len(set(timestamps)):
        raise GeographicPerimeterError("deux observations ont le même timestamp")
    coordinate_count = sum(
        _coordinate_count(geometry)
        for item in ordered
        for geometry in (item.get("affected"), item.get("active"))
        if geometry is not None
    )
    if coordinate_count > MAX_COORDINATES:
        raise GeographicPerimeterError(
            f"{coordinate_count} coordonnées; limite: {MAX_COORDINATES}"
        )
    raw_id = payload.get("dataset_id", payload.get("id"))
    name = str(payload.get("name", payload.get("title", "Périmètres observés"))).strip()
    source_frames: list[dict[str, Any]] = []
    for index, item in enumerate(ordered):
        source_frames.append(
            {
                "frame_id": _safe_id(item.get("source_frame_id"), f"frame-{index:04d}"),
                "observed_at": item["timestamp"],
                "time_range": {
                    "start": item["valid_from"],
                    "end": item["valid_to"],
                    "kind": item["temporal_kind"],
                },
                "affected": _rounded_mapping(item.get("affected")),
                "active": _rounded_mapping(item.get("active")),
            }
        )
    identity_seed = {
        "name": name,
        "frames": source_frames,
    }
    fallback_id = "perimeters-" + _sha256_bytes(_canonical_bytes(identity_seed))[:12]
    return {
        "schema": NORMALIZED_SCHEMA,
        "dataset_id": _safe_id(raw_id, fallback_id),
        "name": name or "Périmètres observés",
        "description": str(payload.get("description", "")).strip(),
        "source_crs": "EPSG:4326",
        "coordinate_order": "longitude_latitude",
        "frames": source_frames,
        "coordinate_count": coordinate_count,
    }


def _geometry_from_normalized(value: Any, label: str) -> BaseGeometry | None:
    return _geometry_from_value(value, label) if value is not None else None


def _project_geometry(geometry: BaseGeometry | None, label: str) -> BaseGeometry | None:
    if geometry is None:
        return None
    projected = transform(_TO_L93.transform, geometry)
    if projected.is_empty or not projected.is_valid:
        raise GeographicPerimeterError(f"{label}: projection Lambert-93 invalide")
    west, south, east, north = projected.bounds
    min_x, min_y, max_x, max_y = _L93_LIMITS
    if west < min_x or east > max_x or south < min_y or north > max_y:
        raise GeographicPerimeterError(f"{label}: hors couverture Lambert-93")
    return projected.normalize()


def _number(value: float) -> str:
    rounded = round(float(value), 3)
    if abs(rounded) < 0.0005:
        rounded = 0.0
    return f"{rounded:.3f}".rstrip("0").rstrip(".") or "0"


def _usd_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _rgb(hex_color: str) -> tuple[float, float, float]:
    clean = hex_color.removeprefix("#")
    return tuple(int(clean[index : index + 2], 16) / 255 for index in (0, 2, 4))


def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    if geometry.geom_type == "Polygon":
        return [geometry]
    return list(geometry.geoms)


def _mesh_block(
    geometry: BaseGeometry,
    *,
    origin: tuple[float, float],
    z_offset: float,
    material_path: str,
    indent: str,
) -> list[str]:
    lines: list[str] = []
    for polygon_index, polygon in enumerate(_polygons(geometry)):
        triangulation = constrained_delaunay_triangles(polygon)
        triangles = sorted(
            (orient(part, sign=1.0) for part in triangulation.geoms),
            key=lambda part: (
                round(part.bounds[0], 6),
                round(part.bounds[1], 6),
                round(part.centroid.x, 6),
                round(part.centroid.y, 6),
            ),
        )
        points: list[str] = []
        indices: list[str] = []
        for triangle in triangles:
            for x, y in list(triangle.exterior.coords)[:3]:
                points.append(
                    f"({_number(x - origin[0])}, {_number(y - origin[1])}, {_number(z_offset)})"
                )
                indices.append(str(len(indices)))
        if not points:
            raise GeographicPerimeterError("triangulation vide")
        min_x, min_y, max_x, max_y = polygon.bounds
        lines.extend(
            [
                f'{indent}def Mesh "Part_{polygon_index:04d}"',
                f"{indent}{{",
                f"{indent}    rel material:binding = <{material_path}>",
                f"{indent}    point3f[] points = [{', '.join(points)}]",
                f"{indent}    int[] faceVertexCounts = [{', '.join('3' for _ in triangles)}]",
                f"{indent}    int[] faceVertexIndices = [{', '.join(indices)}]",
                f'{indent}    uniform token orientation = "rightHanded"',
                f'{indent}    uniform token subdivisionScheme = "none"',
                f"{indent}    float3[] extent = [({_number(min_x - origin[0])}, {_number(min_y - origin[1])}, {_number(z_offset)}), ({_number(max_x - origin[0])}, {_number(max_y - origin[1])}, {_number(z_offset)})]",
                f"{indent}}}",
            ]
        )
        curve_points: list[str] = []
        curve_counts: list[str] = []
        for ring in (polygon.exterior, *polygon.interiors):
            coordinates = list(ring.coords)[:-1]
            if len(coordinates) < 3:
                continue
            curve_counts.append(str(len(coordinates)))
            curve_points.extend(
                f"({_number(x - origin[0])}, {_number(y - origin[1])}, {_number(z_offset + 0.05)})"
                for x, y in coordinates
            )
        if curve_points:
            lines.extend(
                [
                    f'{indent}def BasisCurves "Boundary_{polygon_index:04d}"',
                    f"{indent}{{",
                    f'{indent}    uniform token type = "linear"',
                    f'{indent}    uniform token wrap = "periodic"',
                    f"{indent}    int[] curveVertexCounts = [{', '.join(curve_counts)}]",
                    f"{indent}    point3f[] points = [{', '.join(curve_points)}]",
                    f"{indent}    float[] widths = [1.5]",
                    f"{indent}}}",
                ]
            )
    return lines


def _stage_text(
    *,
    normalized: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
    origin: tuple[float, float],
    build_id: str,
    contract: Mapping[str, Any],
) -> str:
    affected = contract["categories"]["affected"]
    active = contract["categories"]["active"]
    affected_rgb = _rgb(affected["color"])
    active_rgb = _rgb(active["color"])
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "GeographicPerimeters"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "GeographicPerimeters"',
        "{",
        f"    custom string fireviewer:dataset_id = {_usd_string(normalized['dataset_id'])}",
        f"    custom string fireviewer:build_id = {_usd_string(build_id)}",
        '    custom string fireviewer:crs = "EPSG:2154"',
        f"    custom double2 fireviewer:origin_l93_m = ({_number(origin[0])}, {_number(origin[1])})",
        f"    custom int fireviewer:frame_count = {len(frames)}",
        f"    custom asset fireviewer:timeline = @{TIMELINE_NAME}@",
        '    custom string fireviewer:between_observations = "undefined"',
        '    custom string fireviewer:terrain_drape = "consumer_responsibility"',
        "",
        '    def Scope "Looks"',
        "    {",
    ]
    for label, color, opacity in (
        ("Affected", affected_rgb, affected["opacity"]),
        ("Active", active_rgb, active["opacity"]),
    ):
        lines.extend(
            [
                f'        def Material "{label}Material"',
                "        {",
                f"            token outputs:surface.connect = </GeographicPerimeters/Looks/{label}Material/PBR.outputs:surface>",
                '            def Shader "PBR"',
                "            {",
                '                uniform token info:id = "UsdPreviewSurface"',
                f"                color3f inputs:diffuseColor = ({_number(color[0])}, {_number(color[1])}, {_number(color[2])})",
                f"                float inputs:opacity = {_number(opacity)}",
                "                float inputs:roughness = 0.45",
                "                token outputs:surface",
                "            }",
                "        }",
            ]
        )
    lines.extend(["    }", ""])
    for category, label, z_offset in (
        ("affected", "AffectedFixedLayers", affected["z_offset_m"]),
        ("active", "ActiveFixedLayers", active["z_offset_m"]),
    ):
        lines.extend([f'    def Scope "{label}"', "    {"])
        for frame in frames:
            geometry = frame.get(category)
            if geometry is None:
                continue
            frame_name = f"Frame_{frame['index']:04d}"
            material = (
                "/GeographicPerimeters/Looks/AffectedMaterial"
                if category == "affected"
                else "/GeographicPerimeters/Looks/ActiveMaterial"
            )
            lines.extend(
                [
                    f'        def Xform "{frame_name}"',
                    "        {",
                    f"            custom string fireviewer:frame_id = {_usd_string(frame['frame_id'])}",
                    f"            custom string fireviewer:observed_at = {_usd_string(frame['observed_at'])}",
                    f"            custom string fireviewer:valid_from = {_usd_string(frame['time_range']['start'])}",
                    f"            custom string fireviewer:valid_to = {_usd_string(frame['time_range']['end'])}",
                    f"            custom double fireviewer:elapsed_seconds = {_number(frame['elapsed_seconds'])}",
                    f"            custom double fireviewer:elapsed_start_seconds = {_number(frame['elapsed_start_seconds'])}",
                    f"            custom double fireviewer:elapsed_end_seconds = {_number(frame['elapsed_end_seconds'])}",
                    f"            custom double fireviewer:area_ha = {_number(frame[f'{category}_area_ha'])}",
                ]
            )
            lines.extend(
                _mesh_block(
                    geometry,
                    origin=origin,
                    z_offset=float(z_offset),
                    material_path=material,
                    indent="            ",
                )
            )
            lines.append("        }")
        lines.extend(["    }", ""])
    lines.extend(["}", ""])
    return "\n".join(lines)


def compile_perimeter_layer(payload: Mapping[str, Any]) -> CompiledPerimeterLayer:
    """Normalize observations and compile deterministic fixed layers/timeline."""

    contract = _load_contract()
    normalized = _normalize_dataset(payload)
    projected_frames: list[dict[str, Any]] = []
    all_geometries: list[BaseGeometry] = []
    parsed_times = [
        _parse_timestamp(frame["observed_at"], index)
        for index, frame in enumerate(normalized["frames"])
    ]
    parsed_starts = [
        _parse_timestamp(frame["time_range"]["start"], index)
        for index, frame in enumerate(normalized["frames"])
    ]
    parsed_ends = [
        _parse_timestamp(frame["time_range"]["end"], index)
        for index, frame in enumerate(normalized["frames"])
    ]
    time_origin = min(parsed_starts)
    for index, (source_frame, observed_at) in enumerate(
        zip(normalized["frames"], parsed_times, strict=True)
    ):
        affected = _project_geometry(
            _geometry_from_normalized(
                source_frame.get("affected"), f"observation {index} affected"
            ),
            f"observation {index} affected",
        )
        active = _project_geometry(
            _geometry_from_normalized(
                source_frame.get("active"), f"observation {index} active"
            ),
            f"observation {index} active",
        )
        all_geometries.extend(
            geometry for geometry in (affected, active) if geometry is not None
        )
        projected_frames.append(
            {
                "index": index,
                "frame_id": source_frame["frame_id"],
                "observed_at": source_frame["observed_at"],
                "time_range": source_frame["time_range"],
                "elapsed_seconds": int((observed_at - time_origin).total_seconds()),
                "elapsed_start_seconds": int(
                    (parsed_starts[index] - time_origin).total_seconds()
                ),
                "elapsed_end_seconds": int(
                    (parsed_ends[index] - time_origin).total_seconds()
                ),
                "affected": affected,
                "active": active,
                "affected_area_ha": round(affected.area / 10_000, 4)
                if affected is not None
                else 0.0,
                "active_area_ha": round(active.area / 10_000, 4)
                if active is not None
                else 0.0,
            }
        )
    if not all_geometries:
        raise GeographicPerimeterError("aucune géométrie projetée")
    total_bounds = unary_union(all_geometries).bounds
    origin = (
        math.floor(total_bounds[0] / ORIGIN_ALIGNMENT_M) * ORIGIN_ALIGNMENT_M,
        math.floor(total_bounds[1] / ORIGIN_ALIGNMENT_M) * ORIGIN_ALIGNMENT_M,
    )
    contract_sha256 = _sha256_file(_contract_path())
    compiler_sha256 = _sha256_file(Path(__file__))
    source_sha256 = _sha256_bytes(_canonical_bytes(normalized))
    build_id = _sha256_bytes(
        _canonical_bytes(
            {
                "schema": MANIFEST_SCHEMA,
                "source_sha256": source_sha256,
                "contract_sha256": contract_sha256,
                "compiler_sha256": compiler_sha256,
            }
        )
    )
    timeline_frames: list[dict[str, Any]] = []
    for frame in projected_frames:
        timeline_frames.append(
            {
                "index": frame["index"],
                "frame_id": frame["frame_id"],
                "observed_at": frame["observed_at"],
                "elapsed_seconds": frame["elapsed_seconds"],
                "time_range": {
                    **frame["time_range"],
                    "elapsed_start_seconds": frame["elapsed_start_seconds"],
                    "elapsed_end_seconds": frame["elapsed_end_seconds"],
                },
                "affected": {
                    "present": frame["affected"] is not None,
                    "area_ha": frame["affected_area_ha"],
                    "prim_path": f"/GeographicPerimeters/AffectedFixedLayers/Frame_{frame['index']:04d}"
                    if frame["affected"] is not None
                    else None,
                    "geometry_sha256": _sha256_bytes(
                        _canonical_bytes(
                            normalized["frames"][frame["index"]]["affected"]
                        )
                    )
                    if frame["affected"] is not None
                    else None,
                },
                "active": {
                    "present": frame["active"] is not None,
                    "area_ha": frame["active_area_ha"],
                    "prim_path": f"/GeographicPerimeters/ActiveFixedLayers/Frame_{frame['index']:04d}"
                    if frame["active"] is not None
                    else None,
                    "geometry_sha256": _sha256_bytes(
                        _canonical_bytes(normalized["frames"][frame["index"]]["active"])
                    )
                    if frame["active"] is not None
                    else None,
                },
            }
        )
    timeline = {
        "schema": TIMELINE_SCHEMA,
        "status": "observed_fixed_progression",
        "dataset_id": normalized["dataset_id"],
        "build_id": build_id,
        "crs": "EPSG:2154",
        "origin_l93_m": [origin[0], origin[1]],
        "fixed_layer_stage": STAGE_NAME,
        "time_origin": _timestamp_text(time_origin),
        "time_unit": "seconds",
        "between_observations": "undefined",
        "prediction": "none",
        "simulation_driver": "explicit_observed_ranges_and_timestamps",
        "frames": timeline_frames,
    }
    stage = _stage_text(
        normalized=normalized,
        frames=projected_frames,
        origin=origin,
        build_id=build_id,
        contract=contract,
    )
    return CompiledPerimeterLayer(
        dataset_id=normalized["dataset_id"],
        build_id=build_id,
        normalized_source=normalized,
        timeline=timeline,
        stage_text=stage,
        contract_sha256=contract_sha256,
        compiler_sha256=compiler_sha256,
        affected_area_ha=max(frame["affected_area_ha"] for frame in projected_frames),
        active_area_ha=max(frame["active_area_ha"] for frame in projected_frames),
    )


def _output_bytes(compiled: CompiledPerimeterLayer) -> dict[str, bytes]:
    return {
        STAGE_NAME: compiled.stage_text.encode("utf-8"),
        TIMELINE_NAME: _canonical_bytes(compiled.timeline) + b"\n",
        SOURCE_NAME: _canonical_bytes(compiled.normalized_source) + b"\n",
    }


def _manifest(
    compiled: CompiledPerimeterLayer, outputs: Mapping[str, bytes]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "compiled_observed_layers",
        "dataset_id": compiled.dataset_id,
        "build_id": compiled.build_id,
        "contract_sha256": compiled.contract_sha256,
        "compiler_sha256": compiled.compiler_sha256,
        "frame_count": len(compiled.timeline["frames"]),
        "fixed_layer_count": sum(
            int(frame[category]["present"])
            for frame in compiled.timeline["frames"]
            for category in ("affected", "active")
        ),
        "maximum_affected_area_ha": compiled.affected_area_ha,
        "maximum_active_area_ha": compiled.active_area_ha,
        "timeline_semantics": {
            "source": "observed_real_perimeters",
            "between_observations": "undefined",
            "prediction": "none",
        },
        "outputs": {
            name: {"sha256": _sha256_bytes(content), "byte_count": len(content)}
            for name, content in sorted(outputs.items())
        },
    }
    payload["manifest_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    return payload


def _require_d_output(path: Path) -> Path:
    lexical = PureWindowsPath(str(path))
    if lexical.drive and lexical.drive.upper() != "D:":
        raise GeographicPerimeterError("les sorties FireViewer doivent rester sur D:")
    resolved = path.resolve(strict=False)
    if os.name == "nt" and resolved.drive.upper() != "D:":
        raise GeographicPerimeterError("les sorties FireViewer doivent rester sur D:")
    return resolved


def write_perimeter_layer_package(
    compiled: CompiledPerimeterLayer, output_root: Path | str
) -> dict[str, Any]:
    """Publish an immutable package directory atomically."""

    destination = _require_d_output(Path(output_root))
    if destination.exists():
        existing = validate_perimeter_layer_package(destination)
        if existing.get("build_id") != compiled.build_id:
            raise GeographicPerimeterError(
                "un autre calque existe déjà à cette destination"
            )
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    outputs = _output_bytes(compiled)
    manifest = _manifest(compiled, outputs)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}-staging-", dir=str(destination.parent)
        )
    )
    try:
        for name, content in outputs.items():
            (staging / name).write_bytes(content)
        (staging / MANIFEST_NAME).write_bytes(_canonical_bytes(manifest) + b"\n")
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return validate_perimeter_layer_package(destination)


def validate_perimeter_layer_package(package_root: Path | str) -> dict[str, Any]:
    """Recompile the normalized source and rehash every package output."""

    root = Path(package_root).resolve(strict=True)
    expected_names = {STAGE_NAME, TIMELINE_NAME, SOURCE_NAME, MANIFEST_NAME}
    actual_names = {path.name for path in root.iterdir() if path.is_file()}
    if actual_names != expected_names or any(path.is_dir() for path in root.iterdir()):
        raise GeographicPerimeterError("contenu du package de calque inattendu")
    try:
        manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
        normalized = json.loads((root / SOURCE_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GeographicPerimeterError(
            f"package de calque JSON invalide: {error}"
        ) from error
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise GeographicPerimeterError("manifest de calque invalide")
    sealed = dict(manifest)
    recorded_manifest_hash = sealed.pop("manifest_sha256", None)
    if recorded_manifest_hash != _sha256_bytes(_canonical_bytes(sealed)):
        raise GeographicPerimeterError("hash du manifest de calque invalide")
    rebuilt = compile_perimeter_layer(normalized)
    if rebuilt.build_id != manifest.get("build_id"):
        raise GeographicPerimeterError("identité du calque incohérente")
    expected = _output_bytes(rebuilt)
    for name, content in expected.items():
        path = root / name
        if path.read_bytes() != content:
            raise GeographicPerimeterError(f"sortie de calque altérée: {name}")
        record = manifest.get("outputs", {}).get(name)
        if record != {"sha256": _sha256_bytes(content), "byte_count": len(content)}:
            raise GeographicPerimeterError(f"reçu de sortie invalide: {name}")
    return manifest


def _deterministic_archive(package_root: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.part")
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for source in sorted(package_root.iterdir(), key=lambda path: path.name):
            info = zipfile.ZipInfo(source.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())
    if destination.exists():
        if destination.read_bytes() != temporary.read_bytes():
            temporary.unlink()
            raise GeographicPerimeterError("archive existante différente")
        temporary.unlink()
    else:
        os.replace(temporary, destination)


def produce_perimeter_layer(
    source_file: Path | str, work_root: Path | str
) -> PerimeterLayerProduct:
    """Compile an uploaded JSON/GeoJSON into a reusable package and ZIP."""

    source_path = Path(source_file).resolve(strict=True)
    if source_path.suffix.lower() not in {".json", ".geojson"}:
        raise GeographicPerimeterError("un fichier .json ou .geojson est attendu")
    byte_count = source_path.stat().st_size
    if byte_count <= 0 or byte_count > MAX_INPUT_BYTES:
        raise GeographicPerimeterError(
            f"taille JSON invalide ({byte_count} octets; limite {MAX_INPUT_BYTES})"
        )
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GeographicPerimeterError(f"fichier JSON invalide: {error}") from error
    compiled = compile_perimeter_layer(payload)
    root = _require_d_output(Path(work_root)) / "perimeter-layers"
    root.mkdir(parents=True, exist_ok=True)
    package_root = root / compiled.build_id
    manifest = write_perimeter_layer_package(compiled, package_root)
    archive = root / f"{compiled.dataset_id}-{compiled.build_id[:12]}.zip"
    _deterministic_archive(package_root, archive)
    return PerimeterLayerProduct(
        package_root=package_root, archive=archive, manifest=manifest
    )


__all__ = [
    "ARCHIVE_NAME",
    "MANIFEST_SCHEMA",
    "TIMELINE_SCHEMA",
    "CompiledPerimeterLayer",
    "GeographicPerimeterError",
    "PerimeterLayerProduct",
    "compile_perimeter_layer",
    "produce_perimeter_layer",
    "validate_perimeter_layer_package",
    "write_perimeter_layer_package",
]
