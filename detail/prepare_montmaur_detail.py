"""Produce deterministic Montmaur detail vectors from local classified COPC.

No network access is performed. The tree catalogue represents detected crown
apices only: it is not, and cannot be, a proof that every physical tree is
present in the source point cloud.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Iterable, Sequence

import laspy
import numpy as np
import pyproj
import rasterio
from rasterio.features import geometry_mask
from rasterio.merge import merge as merge_rasters
from rasterio.windows import Window, from_bounds
import scipy
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Polygon,
    box,
    mapping,
    shape,
)
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union
from shapely.validation import make_valid


TARGET_CRS = "EPSG:2154"
VEGETATION_CLASSES = (3, 4, 5)
BUILDING_CLASS = 6
POINT_DEDUPLICATION_METRES = 0.001


@dataclass(frozen=True)
class DetailParameters:
    canopy_cell_m: float = 0.5
    local_peak_radius_m: float = 1.5
    crown_radius_height_ratio: float = 0.28
    min_crown_radius_m: float = 1.5
    max_crown_radius_m: float = 8.0
    min_tree_height_m: float = 2.0
    min_crown_point_height_m: float = 1.0
    min_tree_points: int = 12
    hedge_search_radius_m: float = 3.0
    min_hedge_height_m: float = 0.5
    min_hedge_points: int = 8
    min_building_roof_points: int = 5
    min_raster_pixels: int = 3

    def validate(self) -> None:
        positive = {
            "canopy_cell_m": self.canopy_cell_m,
            "local_peak_radius_m": self.local_peak_radius_m,
            "crown_radius_height_ratio": self.crown_radius_height_ratio,
            "min_crown_radius_m": self.min_crown_radius_m,
            "max_crown_radius_m": self.max_crown_radius_m,
            "min_tree_height_m": self.min_tree_height_m,
            "min_crown_point_height_m": self.min_crown_point_height_m,
            "hedge_search_radius_m": self.hedge_search_radius_m,
            "min_hedge_height_m": self.min_hedge_height_m,
        }
        invalid = [
            name
            for name, value in positive.items()
            if not math.isfinite(value) or value <= 0
        ]
        if invalid:
            raise ValueError(f"Parameters must be positive finite numbers: {invalid}")
        if self.max_crown_radius_m < self.min_crown_radius_m:
            raise ValueError("max_crown_radius_m must be >= min_crown_radius_m")
        for name in (
            "min_tree_points",
            "min_hedge_points",
            "min_building_roof_points",
            "min_raster_pixels",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")


@dataclass(frozen=True)
class ClassifiedPoints:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    classification: np.ndarray

    def __post_init__(self) -> None:
        sizes = {len(self.x), len(self.y), len(self.z), len(self.classification)}
        if len(sizes) != 1:
            raise ValueError("Point arrays have different lengths")
        if not (
            np.isfinite(self.x).all()
            and np.isfinite(self.y).all()
            and np.isfinite(self.z).all()
        ):
            raise ValueError("Point coordinates must be finite")

    def subset(self, selector: np.ndarray) -> "ClassifiedPoints":
        return ClassifiedPoints(
            self.x[selector],
            self.y[selector],
            self.z[selector],
            self.classification[selector],
        )

    def __len__(self) -> int:
        return len(self.x)


@dataclass(frozen=True)
class VectorFeature:
    source_id: str
    geometry: Polygon | MultiPolygon | LineString | MultiLineString
    properties: dict[str, Any]


@dataclass(frozen=True)
class ZonalValues:
    total_count: int
    mnt: np.ndarray
    mns: np.ndarray
    height: np.ndarray


@dataclass(frozen=True)
class RasterPair:
    mnt: np.ndarray
    mns: np.ndarray
    transform: rasterio.Affine
    source_bounds: tuple[float, float, float, float]
    source_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.mnt.shape != self.mns.shape or self.mnt.ndim != 2:
            raise ValueError("MNT and MNS arrays must be aligned two-dimensional grids")

    def ground_at(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = np.floor((x - self.transform.c) / self.transform.a).astype(np.int64)
        row = np.floor((y - self.transform.f) / self.transform.e).astype(np.int64)
        inside = (
            (row >= 0)
            & (col >= 0)
            & (row < self.mnt.shape[0])
            & (col < self.mnt.shape[1])
        )
        values = np.full(len(x), np.nan, dtype=np.float64)
        values[inside] = self.mnt[row[inside], col[inside]]
        valid = inside & np.isfinite(values)
        return values, valid

    def zonal(self, geometry: Polygon | MultiPolygon) -> ZonalValues:
        min_x, min_y, max_x, max_y = geometry.bounds
        fractional = from_bounds(min_x, min_y, max_x, max_y, transform=self.transform)
        col_start = max(0, math.floor(fractional.col_off))
        row_start = max(0, math.floor(fractional.row_off))
        col_stop = min(
            self.mnt.shape[1], math.ceil(fractional.col_off + fractional.width)
        )
        row_stop = min(
            self.mnt.shape[0], math.ceil(fractional.row_off + fractional.height)
        )
        if col_stop <= col_start or row_stop <= row_start:
            return ZonalValues(0, np.empty(0), np.empty(0), np.empty(0))
        window = Window(
            col_start, row_start, col_stop - col_start, row_stop - row_start
        )
        ground = self.mnt[row_start:row_stop, col_start:col_stop]
        surface = self.mns[row_start:row_stop, col_start:col_stop]
        inside = geometry_mask(
            [mapping(geometry)],
            out_shape=ground.shape,
            transform=rasterio.windows.transform(window, self.transform),
            invert=True,
            all_touched=False,
        )
        total = int(np.count_nonzero(inside))
        valid_ground = inside & np.isfinite(ground)
        valid_surface = valid_ground & np.isfinite(surface)
        return ZonalValues(
            total,
            ground[valid_ground].astype(np.float64, copy=False),
            surface[valid_surface].astype(np.float64, copy=False),
            np.maximum(0.0, surface[valid_surface] - ground[valid_surface]),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _source_asset(path: Path) -> dict[str, Any]:
    return {
        "file_name": path.name,
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _output_asset(path: Path, feature_count: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.name,
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if feature_count is not None:
        result["feature_count"] = feature_count
    return result


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def _geojson_features(document: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    if document.get("type") == "FeatureCollection" and isinstance(
        document.get("features"), list
    ):
        return [
            feature for feature in document["features"] if isinstance(feature, dict)
        ]
    if document.get("type") == "Feature":
        return [document]
    raise ValueError(f"Expected a GeoJSON FeatureCollection or Feature: {path}")


def _document_crs(document: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    crs = document.get("crs")
    name = crs.get("properties", {}).get("name", "") if isinstance(crs, dict) else ""
    return TARGET_CRS if "2154" in str(name) else "EPSG:4326"


def _polygonal_only(geometry: Any) -> Polygon | MultiPolygon | None:
    geometry = make_valid(geometry)
    if geometry.is_empty:
        return None
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        parts: list[Polygon] = []
        for item in geometry.geoms:
            part = _polygonal_only(item)
            if isinstance(part, Polygon):
                parts.append(part)
            elif isinstance(part, MultiPolygon):
                parts.extend(part.geoms)
        return _polygonal_only(unary_union(parts)) if parts else None
    return None


def _linear_only(geometry: Any) -> LineString | MultiLineString | None:
    geometry = make_valid(geometry)
    if geometry.is_empty:
        return None
    if isinstance(geometry, (LineString, MultiLineString)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        parts: list[LineString] = []
        for item in geometry.geoms:
            part = _linear_only(item)
            if isinstance(part, LineString):
                parts.append(part)
            elif isinstance(part, MultiLineString):
                parts.extend(part.geoms)
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else MultiLineString(parts)
    return None


def load_aoi(path: Path, crs: str = "auto") -> Polygon | MultiPolygon:
    document = json.loads(path.read_text(encoding="utf-8"))
    source_crs = _document_crs(document, crs)
    transformer = pyproj.Transformer.from_crs(source_crs, TARGET_CRS, always_xy=True)
    geometries: list[Polygon | MultiPolygon] = []
    for feature in _geojson_features(document, path):
        if feature.get("geometry"):
            geometry = _polygonal_only(
                transform_geometry(transformer.transform, shape(feature["geometry"]))
            )
            if geometry is not None:
                geometries.append(geometry)
    aoi = _polygonal_only(unary_union(geometries)) if geometries else None
    if aoi is None or aoi.area <= 0:
        raise ValueError(f"AOI has no polygonal area: {path}")
    return aoi


def load_vector_features(
    path: Path,
    aoi: Polygon | MultiPolygon,
    *,
    crs: str = "auto",
    kind: str,
) -> tuple[list[VectorFeature], dict[str, int]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    transformer = pyproj.Transformer.from_crs(
        _document_crs(document, crs), TARGET_CRS, always_xy=True
    )
    result: list[VectorFeature] = []
    seen: set[str] = set()
    stats = Counter(input=0, duplicate=0, invalid=0, outside=0)
    for index, feature in enumerate(_geojson_features(document, path)):
        stats["input"] += 1
        properties = feature.get("properties") or {}
        if not isinstance(properties, dict) or not feature.get("geometry"):
            stats["invalid"] += 1
            continue
        projected = transform_geometry(
            transformer.transform, shape(feature["geometry"])
        )
        clipped_raw = projected.intersection(aoi)
        geometry = (
            _polygonal_only(clipped_raw)
            if kind == "building"
            else _linear_only(clipped_raw)
        )
        if geometry is None:
            stats["outside"] += 1
            continue
        source_id_value = properties.get("cleabs")
        source_id = str(source_id_value).strip() if source_id_value else ""
        if not source_id:
            source_id = f"SOURCELESS-{hashlib.sha256(geometry.wkb).hexdigest()[:16]}"
        if source_id in seen:
            stats["duplicate"] += 1
            continue
        seen.add(source_id)
        result.append(VectorFeature(source_id, geometry, dict(properties)))
    result.sort(key=lambda item: item.source_id)
    stats["output"] = len(result)
    return result, dict(stats)


def _validated_raster_metadata(
    dataset: rasterio.io.DatasetReader, aoi: Polygon | MultiPolygon
) -> dict[str, Any]:
    transform = dataset.transform
    bounds = dataset.bounds
    if (
        transform.a <= 0
        or transform.e >= 0
        or abs(transform.b) > 1e-9
        or abs(transform.d) > 1e-9
    ):
        raise ValueError(f"Raster must be north-up: {dataset.name}")
    if not (
        -200_000 <= bounds.left <= 1_500_000
        and -200_000 <= bounds.right <= 1_500_000
        and 5_500_000 <= bounds.bottom <= 7_300_000
        and 5_500_000 <= bounds.top <= 7_300_000
    ):
        raise ValueError(
            f"Raster bounds are not plausible EPSG:2154 metres: {dataset.name}"
        )
    if not box(*bounds).intersects(aoi):
        raise ValueError(f"Raster does not intersect the Montmaur AOI: {dataset.name}")
    epsg = dataset.crs.to_epsg() if dataset.crs is not None else None
    if epsg is not None and epsg != 2154:
        raise ValueError(
            f"Raster declares EPSG:{epsg}, expected EPSG:2154: {dataset.name}"
        )
    return {
        "observed_crs": dataset.crs.to_string() if dataset.crs is not None else None,
        "assigned_crs": TARGET_CRS,
        "assignment": "explicit_after_transform_and_lambert93_bounds_validation",
        "source_shape": [dataset.height, dataset.width],
        "source_bounds_l93": [bounds.left, bounds.bottom, bounds.right, bounds.top],
        "source_pixel_size_m": [transform.a, abs(transform.e)],
        "source_nodata": dataset.nodata,
    }


def _as_paths(value: Path | Sequence[Path]) -> list[Path]:
    return [value] if isinstance(value, Path) else list(value)


def load_raster_pair(
    mnt_path: Path | Sequence[Path],
    mns_path: Path | Sequence[Path],
    aoi: Polygon | MultiPolygon,
    *,
    required_resolution_m: float | None = 0.5,
) -> RasterPair:
    mnt_paths = _as_paths(mnt_path)
    mns_paths = _as_paths(mns_path)
    if not mnt_paths or not mns_paths:
        raise ValueError("At least one MNT and one MNS tile are required")
    with ExitStack() as stack:
        mnt_sources = [stack.enter_context(rasterio.open(path)) for path in mnt_paths]
        mns_sources = [stack.enter_context(rasterio.open(path)) for path in mns_paths]
        mnt_meta = [_validated_raster_metadata(source, aoi) for source in mnt_sources]
        mns_meta = [_validated_raster_metadata(source, aoi) for source in mns_sources]
        all_sources = [*mnt_sources, *mns_sources]
        reference = mnt_sources[0]
        resolution = (reference.transform.a, abs(reference.transform.e))
        for source in all_sources:
            if not np.allclose(
                (source.transform.a, abs(source.transform.e)),
                resolution,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    "All MNT and MNS tiles must have the same pixel resolution"
                )
            x_phase = (source.transform.c - reference.transform.c) / resolution[0]
            y_phase = (source.transform.f - reference.transform.f) / resolution[1]
            if not (
                math.isclose(x_phase, round(x_phase), abs_tol=1e-9)
                and math.isclose(y_phase, round(y_phase), abs_tol=1e-9)
            ):
                raise ValueError(
                    "All MNT and MNS tiles must share the same native pixel grid"
                )
        if required_resolution_m is not None and not np.allclose(
            resolution,
            (required_resolution_m, required_resolution_m),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                f"Detail rasters must have {required_resolution_m} m pixels, got "
                f"{resolution[0]} x {resolution[1]} m"
            )
        if not unary_union([box(*source.bounds) for source in mnt_sources]).covers(aoi):
            raise ValueError("The MNT tile union does not cover the Montmaur AOI")
        if not unary_union([box(*source.bounds) for source in mns_sources]).covers(aoi):
            raise ValueError("The MNS tile union does not cover the Montmaur AOI")

        fractional = from_bounds(*aoi.bounds, transform=reference.transform)
        # Negative offsets are valid here because the reference is one native
        # tile while the AOI mosaic extends into adjacent tiles.
        col_start = math.floor(fractional.col_off)
        row_start = math.floor(fractional.row_off)
        col_stop = math.ceil(fractional.col_off + fractional.width)
        row_stop = math.ceil(fractional.row_off + fractional.height)
        window = Window(
            col_start, row_start, col_stop - col_start, row_stop - row_start
        )
        native_bounds = rasterio.windows.bounds(window, reference.transform)
        ground_stack, ground_transform = merge_rasters(
            mnt_sources,
            bounds=native_bounds,
            res=resolution,
            nodata=np.nan,
            dtype="float64",
            masked=True,
            method="first",
        )
        surface_stack, surface_transform = merge_rasters(
            mns_sources,
            bounds=native_bounds,
            res=resolution,
            nodata=np.nan,
            dtype="float64",
            masked=True,
            method="first",
        )
        if ground_stack.shape != surface_stack.shape or not np.allclose(
            tuple(ground_transform), tuple(surface_transform), rtol=0.0, atol=1e-9
        ):
            raise ValueError("MNT and MNS mosaics are not aligned")
        ground = (
            np.ma.asarray(ground_stack[0]).filled(np.nan).astype(np.float64, copy=False)
        )
        surface = (
            np.ma.asarray(surface_stack[0])
            .filled(np.nan)
            .astype(np.float64, copy=False)
        )
        return RasterPair(
            ground,
            surface,
            ground_transform,
            tuple(native_bounds),
            {
                "mnt": mnt_meta,
                "mns": mns_meta,
                "mosaic": {
                    "bounds_l93": list(native_bounds),
                    "shape": [ground.shape[0], ground.shape[1]],
                    "pixel_size_m": list(resolution),
                },
            },
        )


def _is_copc(path: Path) -> bool:
    with laspy.open(path) as reader:
        return any(
            isinstance(vlr, laspy.copc.CopcInfoVlr) for vlr in reader.header.vlrs
        )


def _validate_point_header(
    header: laspy.LasHeader, path: Path, aoi: Polygon | MultiPolygon
) -> dict[str, Any]:
    bounds = (
        float(header.mins[0]),
        float(header.mins[1]),
        float(header.maxs[0]),
        float(header.maxs[1]),
    )
    if not box(*bounds).intersects(aoi):
        raise ValueError(f"Point cloud does not intersect the Montmaur AOI: {path}")
    if not (
        -200_000 <= bounds[0] <= 1_500_000
        and -200_000 <= bounds[2] <= 1_500_000
        and 5_500_000 <= bounds[1] <= 7_300_000
        and 5_500_000 <= bounds[3] <= 7_300_000
    ):
        raise ValueError(
            f"Point cloud bounds are not plausible EPSG:2154 metres: {path}"
        )
    crs = header.parse_crs()
    epsg = crs.to_epsg() if crs is not None else None
    if epsg is not None and epsg != 2154:
        raise ValueError(
            f"Point cloud declares EPSG:{epsg}, expected EPSG:2154: {path}"
        )
    dimensions = set(header.point_format.dimension_names)
    if "classification" not in dimensions:
        raise ValueError(f"Point cloud has no LAS classification dimension: {path}")
    return {
        "bounds_l93": list(bounds),
        "observed_crs": crs.to_string() if crs is not None else None,
        "assigned_crs": TARGET_CRS,
        "point_format": header.point_format.id,
        "scales": [float(value) for value in header.scales],
        "offsets": [float(value) for value in header.offsets],
    }


def deduplicate_points(points: ClassifiedPoints) -> tuple[ClassifiedPoints, int]:
    if not len(points):
        return points, 0
    quantization = POINT_DEDUPLICATION_METRES
    x_key = np.rint(points.x / quantization).astype(np.int64)
    y_key = np.rint(points.y / quantization).astype(np.int64)
    z_key = np.rint(points.z / quantization).astype(np.int64)
    classification = points.classification.astype(np.int16, copy=False)
    order = np.lexsort((classification, z_key, y_key, x_key))
    duplicate = (
        (x_key[order][1:] == x_key[order][:-1])
        & (y_key[order][1:] == y_key[order][:-1])
        & (z_key[order][1:] == z_key[order][:-1])
        & (classification[order][1:] == classification[order][:-1])
    )
    keep = np.r_[True, ~duplicate]
    indices = order[keep]
    unique = ClassifiedPoints(
        points.x[indices],
        points.y[indices],
        points.z[indices],
        points.classification[indices],
    )
    return unique, len(points) - len(unique)


def load_classified_points(
    paths: Sequence[Path],
    aoi: Polygon | MultiPolygon,
    *,
    require_copc: bool = True,
) -> tuple[ClassifiedPoints, list[dict[str, Any]], dict[str, int]]:
    if not paths:
        raise ValueError("At least one local COPC is required")
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    class_parts: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    min_x, min_y, max_x, max_y = aoi.bounds
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        copc = _is_copc(path)
        if require_copc and not copc:
            raise ValueError(f"Production input is not COPC: {path}")
        if copc:
            with laspy.CopcReader.open(path) as reader:
                metadata = _validate_point_header(reader.header, path, aoi)
                points = reader.query(
                    bounds=laspy.copc.Bounds(
                        mins=np.asarray((min_x, min_y), dtype=np.float64),
                        maxs=np.asarray((max_x, max_y), dtype=np.float64),
                    )
                )
                chunks: Iterable[Any] = (points,)
                for chunk in chunks:
                    classification = np.asarray(chunk.classification, dtype=np.uint8)
                    selected = np.isin(
                        classification, (*VEGETATION_CLASSES, BUILDING_CLASS)
                    )
                    if np.any(selected):
                        x_parts.append(np.asarray(chunk.x, dtype=np.float64)[selected])
                        y_parts.append(np.asarray(chunk.y, dtype=np.float64)[selected])
                        z_parts.append(np.asarray(chunk.z, dtype=np.float64)[selected])
                        class_parts.append(classification[selected])
        else:
            with laspy.open(path) as reader:
                metadata = _validate_point_header(reader.header, path, aoi)
                for chunk in reader.chunk_iterator(1_000_000):
                    classification = np.asarray(chunk.classification, dtype=np.uint8)
                    x = np.asarray(chunk.x, dtype=np.float64)
                    y = np.asarray(chunk.y, dtype=np.float64)
                    selected = (
                        (x >= min_x)
                        & (x <= max_x)
                        & (y >= min_y)
                        & (y <= max_y)
                        & np.isin(classification, (*VEGETATION_CLASSES, BUILDING_CLASS))
                    )
                    if np.any(selected):
                        x_parts.append(x[selected])
                        y_parts.append(y[selected])
                        z_parts.append(np.asarray(chunk.z, dtype=np.float64)[selected])
                        class_parts.append(classification[selected])
        sources.append(
            {
                **_source_asset(path),
                "source_kind": "COPC" if copc else "LAS/LAZ",
                **metadata,
            }
        )

    points = ClassifiedPoints(
        np.concatenate(x_parts) if x_parts else np.empty(0, dtype=np.float64),
        np.concatenate(y_parts) if y_parts else np.empty(0, dtype=np.float64),
        np.concatenate(z_parts) if z_parts else np.empty(0, dtype=np.float64),
        np.concatenate(class_parts) if class_parts else np.empty(0, dtype=np.uint8),
    )
    inside = (
        shapely.intersects_xy(aoi, points.x, points.y)
        if len(points)
        else np.empty(0, dtype=bool)
    )
    points = points.subset(inside)
    before_deduplication = len(points)
    points, duplicate_count = deduplicate_points(points)
    class_counts = Counter(map(int, points.classification))
    stats = {
        "inside_aoi_before_deduplication": before_deduplication,
        "duplicate_point_count": duplicate_count,
        "retained_point_count": len(points),
        **{f"class_{key}_count": value for key, value in sorted(class_counts.items())},
    }
    return points, sources, stats


def _feature_collection(name: str, features: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "name": name,
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::2154"}},
        "features": features,
    }


def measure_hedges(
    hedges: Sequence[VectorFeature],
    vegetation: ClassifiedPoints,
    normalized_height: np.ndarray,
    ground: np.ndarray,
    parameters: DetailParameters,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, int]]:
    eligible = np.isfinite(normalized_height) & (
        normalized_height >= parameters.min_hedge_height_m
    )
    best_distance = np.full(len(vegetation), np.inf, dtype=np.float64)
    assignment = np.full(len(vegetation), -1, dtype=np.int64)
    for hedge_index, hedge in enumerate(hedges):
        min_x, min_y, max_x, max_y = hedge.geometry.bounds
        candidate = np.flatnonzero(
            eligible
            & (vegetation.x >= min_x - parameters.hedge_search_radius_m)
            & (vegetation.x <= max_x + parameters.hedge_search_radius_m)
            & (vegetation.y >= min_y - parameters.hedge_search_radius_m)
            & (vegetation.y <= max_y + parameters.hedge_search_radius_m)
        )
        if not len(candidate):
            continue
        distances = np.asarray(
            shapely.distance(
                hedge.geometry,
                shapely.points(vegetation.x[candidate], vegetation.y[candidate]),
            ),
            dtype=np.float64,
        )
        improve = (distances <= parameters.hedge_search_radius_m) & (
            distances < best_distance[candidate]
        )
        selected = candidate[improve]
        best_distance[selected] = distances[improve]
        assignment[selected] = hedge_index

    output: list[dict[str, Any]] = []
    insufficient = 0
    for hedge_index, hedge in enumerate(hedges):
        indices = np.flatnonzero(assignment == hedge_index)
        sufficient = len(indices) >= parameters.min_hedge_points
        height = (
            float(np.percentile(normalized_height[indices], 75)) if sufficient else None
        )
        base = float(np.median(ground[indices])) if sufficient else None
        measured_width = (
            float(2.0 * np.percentile(best_distance[indices], 90))
            if sufficient
            else None
        )
        if measured_width is not None and measured_width <= 0:
            measured_width = None
        if not sufficient:
            insufficient += 1
        properties = dict(hedge.properties)
        properties.update(
            {
                "detail_id": f"MONTMAUR-HEDGE-{hedge.source_id}",
                "base_elevation_m": _round(base),
                "height_m": _round(height),
                "width_m": _round(measured_width),
                "point_count": int(len(indices)),
                "height_method": "lidar_vegetation_p75_above_mnt"
                if height is not None
                else None,
                "width_method": "twice_p90_lateral_distance_to_bdtopo_axis"
                if measured_width is not None
                else None,
                "quality": "measured" if sufficient else "insufficient_points",
                "association_radius_m": parameters.hedge_search_radius_m,
            }
        )
        output.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": mapping(hedge.geometry),
            }
        )
    return (
        output,
        assignment >= 0,
        {
            "source_hedge_count": len(hedges),
            "measured_hedge_count": len(hedges) - insufficient,
            "insufficient_hedge_count": insufficient,
            "vegetation_points_assigned_to_hedges": int(
                np.count_nonzero(assignment >= 0)
            ),
        },
    )


def _apex_candidates(
    points: ClassifiedPoints,
    heights: np.ndarray,
    aoi_bounds: tuple[float, float, float, float],
    parameters: DetailParameters,
) -> np.ndarray:
    min_x, min_y, max_x, max_y = aoi_bounds
    columns = max(1, math.ceil((max_x - min_x) / parameters.canopy_cell_m))
    rows = max(1, math.ceil((max_y - min_y) / parameters.canopy_cell_m))
    cell_x = np.clip(
        ((points.x - min_x) / parameters.canopy_cell_m).astype(np.int64), 0, columns - 1
    )
    cell_y = np.clip(
        ((points.y - min_y) / parameters.canopy_cell_m).astype(np.int64), 0, rows - 1
    )
    flat = cell_y * columns + cell_x
    order = np.lexsort((points.z, points.y, points.x, -heights, flat))
    sorted_flat = flat[order]
    first = np.r_[True, sorted_flat[1:] != sorted_flat[:-1]]
    highest_indices = order[first]
    grid = np.full(rows * columns, -np.inf, dtype=np.float32)
    grid[flat[highest_indices]] = heights[highest_indices].astype(np.float32)
    grid = grid.reshape(rows, columns)
    radius_cells = max(
        1, math.ceil(parameters.local_peak_radius_m / parameters.canopy_cell_m)
    )
    local_maximum = maximum_filter(
        grid,
        size=2 * radius_cells + 1,
        mode="constant",
        cval=-np.inf,
    )
    candidate_cells = np.flatnonzero(
        np.isfinite(grid.ravel())
        & (grid.ravel() >= parameters.min_tree_height_m)
        & (grid.ravel() == local_maximum.ravel())
    )
    if not len(candidate_cells):
        return np.empty(0, dtype=np.int64)
    position = np.searchsorted(sorted_flat[first], candidate_cells)
    candidate_indices = highest_indices[position]
    candidate_order = np.lexsort(
        (
            points.z[candidate_indices],
            points.y[candidate_indices],
            points.x[candidate_indices],
            -heights[candidate_indices],
        )
    )
    candidates = candidate_indices[candidate_order]
    accepted: list[int] = []
    accepted_radius: list[float] = []
    for candidate in candidates:
        if accepted:
            dx = points.x[np.asarray(accepted)] - points.x[candidate]
            dy = points.y[np.asarray(accepted)] - points.y[candidate]
            distance = np.hypot(dx, dy)
            if np.any(distance < np.asarray(accepted_radius)):
                continue
        accepted.append(int(candidate))
        accepted_radius.append(
            float(
                np.clip(
                    heights[candidate] * parameters.crown_radius_height_ratio,
                    parameters.min_crown_radius_m,
                    parameters.max_crown_radius_m,
                )
            )
        )
    return np.asarray(accepted, dtype=np.int64)


def detect_trees(
    vegetation: ClassifiedPoints,
    normalized_height: np.ndarray,
    ground: np.ndarray,
    excluded_by_hedge: np.ndarray,
    aoi_bounds: tuple[float, float, float, float],
    parameters: DetailParameters,
    *,
    zone_id: str = "montmaur",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    candidate_mask = (
        np.isfinite(normalized_height)
        & ~excluded_by_hedge
        & (normalized_height >= parameters.min_crown_point_height_m)
    )
    candidate_points = vegetation.subset(candidate_mask)
    candidate_heights = normalized_height[candidate_mask]
    candidate_ground = ground[candidate_mask]
    apex_local_indices = _apex_candidates(
        candidate_points, candidate_heights, aoi_bounds, parameters
    )
    if not len(apex_local_indices):
        return (
            [],
            [],
            {
                "eligible_vegetation_point_count": len(candidate_points),
                "apex_candidate_count": 0,
                "accepted_tree_count": 0,
                "rejected_sparse_crown_count": 0,
            },
        )

    apex_xy = np.column_stack(
        (candidate_points.x[apex_local_indices], candidate_points.y[apex_local_indices])
    )
    distances, nearest = cKDTree(apex_xy).query(
        np.column_stack((candidate_points.x, candidate_points.y)), k=1
    )
    radii = np.clip(
        candidate_heights[apex_local_indices] * parameters.crown_radius_height_ratio,
        parameters.min_crown_radius_m,
        parameters.max_crown_radius_m,
    )
    relative_minimum = np.maximum(
        parameters.min_crown_point_height_m,
        candidate_heights[apex_local_indices[nearest]] * 0.2,
    )
    assigned = (distances <= radii[nearest]) & (candidate_heights >= relative_minimum)

    tree_records: list[dict[str, Any]] = []
    crown_records: list[dict[str, Any]] = []
    rejected_sparse = 0
    zone_prefix = "".join(
        character if character.isalnum() else "-" for character in zone_id.upper()
    )
    used_ids: Counter[str] = Counter()
    for apex_number, apex_index in enumerate(apex_local_indices):
        members = np.flatnonzero(assigned & (nearest == apex_number))
        if len(members) < parameters.min_tree_points:
            rejected_sparse += 1
            continue
        apex_x = float(candidate_points.x[apex_index])
        apex_y = float(candidate_points.y[apex_index])
        apex_z = float(candidate_points.z[apex_index])
        apex_height = float(candidate_heights[apex_index])
        radial = np.hypot(
            candidate_points.x[members] - apex_x, candidate_points.y[members] - apex_y
        )
        diameter = (
            float(2.0 * np.percentile(radial, 95)) if np.any(radial > 0) else None
        )
        horizontal_points = np.column_stack(
            (candidate_points.x[members], candidate_points.y[members])
        )
        crown = MultiPoint(horizontal_points).convex_hull
        crown_geometry = (
            crown if isinstance(crown, Polygon) and crown.area > 0 else None
        )
        base_id = f"{zone_prefix}-TREE-{int(round(apex_x * 100)):09d}-{int(round(apex_y * 100)):010d}"
        used_ids[base_id] += 1
        tree_id = (
            base_id if used_ids[base_id] == 1 else f"{base_id}-{used_ids[base_id]}"
        )
        classes = Counter(map(int, candidate_points.classification[members]))
        properties = {
            "tree_id": tree_id,
            "position_method": "highest_observed_lidar_vegetation_return_in_apex_cell",
            "apex_elevation_m": _round(apex_z),
            "ground_elevation_m": _round(float(candidate_ground[apex_index])),
            "height_m": _round(apex_height),
            "height_method": "observed_apex_z_minus_mnt_nearest_pixel",
            "crown_diameter_m": _round(diameter),
            "diameter_method": "twice_p95_radial_distance_of_assigned_returns"
            if diameter is not None
            else None,
            "assigned_point_count": int(len(members)),
            "class_3_count": classes.get(3, 0),
            "class_4_count": classes.get(4, 0),
            "class_5_count": classes.get(5, 0),
            "crown_geometry_quality": "convex_hull_of_assigned_returns"
            if crown_geometry is not None
            else "degenerate",
            "completeness_claim": "detected_crown_only_not_every_physical_tree",
        }
        tree_records.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "Point", "coordinates": [apex_x, apex_y, apex_z]},
            }
        )
        if crown_geometry is not None:
            crown_records.append(
                {
                    "type": "Feature",
                    "properties": {
                        "tree_id": tree_id,
                        "area_m2": _round(float(crown_geometry.area)),
                        "method": "convex_hull_of_apex_assigned_lidar_returns",
                    },
                    "geometry": mapping(crown_geometry),
                }
            )
    tree_records.sort(key=lambda feature: feature["properties"]["tree_id"])
    crown_records.sort(key=lambda feature: feature["properties"]["tree_id"])
    return (
        tree_records,
        crown_records,
        {
            "eligible_vegetation_point_count": len(candidate_points),
            "apex_candidate_count": len(apex_local_indices),
            "accepted_tree_count": len(tree_records),
            "rejected_sparse_crown_count": rejected_sparse,
        },
    )


def measure_buildings(
    buildings: Sequence[VectorFeature],
    building_points: ClassifiedPoints,
    rasters: RasterPair,
    parameters: DetailParameters,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    point_tree = (
        cKDTree(np.column_stack((building_points.x, building_points.y)))
        if len(building_points)
        else None
    )
    output: list[dict[str, Any]] = []
    methods: Counter[str] = Counter()
    null_height_count = 0
    for building in buildings:
        zonal = rasters.zonal(building.geometry)
        ground_sufficient = len(zonal.mnt) >= parameters.min_raster_pixels
        base = float(np.median(zonal.mnt)) if ground_sufficient else None
        roof_indices = np.empty(0, dtype=np.int64)
        if point_tree is not None:
            min_x, min_y, max_x, max_y = building.geometry.bounds
            centre = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
            radius = math.hypot(max_x - min_x, max_y - min_y) / 2.0
            candidates = np.asarray(
                point_tree.query_ball_point(centre, radius + 1e-9), dtype=np.int64
            )
            if len(candidates):
                inside = shapely.intersects_xy(
                    building.geometry,
                    building_points.x[candidates],
                    building_points.y[candidates],
                )
                roof_indices = candidates[inside]

        roof = None
        height = None
        method = None
        quality = (
            "insufficient_ground" if base is None else "insufficient_roof_observations"
        )
        if (
            base is not None
            and len(roof_indices) >= parameters.min_building_roof_points
        ):
            roof = float(np.percentile(building_points.z[roof_indices], 95))
            measured = roof - base
            if measured > 0:
                height = measured
                method = "copc_class_6_roof_p95_minus_mnt_median"
                quality = "measured_classified_points"
        if (
            base is not None
            and height is None
            and len(zonal.mns) >= parameters.min_raster_pixels
        ):
            roof = float(np.percentile(zonal.mns, 75))
            measured = roof - base
            if measured > 0:
                height = measured
                method = "mns_p75_minus_mnt_median"
                quality = "measured_raster_fallback"
        if height is None:
            null_height_count += 1
            method_key = "null"
        else:
            method_key = str(method)
        methods[method_key] += 1
        properties = dict(building.properties)
        properties.update(
            {
                "detail_id": f"MONTMAUR-BUILDING-{building.source_id}",
                "base_elevation_m": _round(base),
                "roof_elevation_m": _round(roof),
                "height_m": _round(height),
                "height_method": method,
                "quality": quality,
                "roof_class_6_point_count": int(len(roof_indices)),
                "mnt_pixel_count": int(len(zonal.mnt)),
                "mns_pixel_count": int(len(zonal.mns)),
                "footprint_method": "bdtopo_clipped_to_detail_aoi",
                "invented_default_height": None,
            }
        )
        output.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": mapping(building.geometry),
            }
        )
    return output, {
        "source_building_count": len(buildings),
        "output_building_count": len(output),
        "null_height_count": null_height_count,
        **{
            f"height_method_{key}_count": value
            for key, value in sorted(methods.items())
        },
    }


def produce_detail_from_data(
    *,
    aoi: Polygon | MultiPolygon,
    rasters: RasterPair,
    points: ClassifiedPoints,
    buildings: Sequence[VectorFeature],
    hedges: Sequence[VectorFeature],
    parameters: DetailParameters,
    zone_id: str = "montmaur",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    parameters.validate()
    vegetation_mask = np.isin(points.classification, VEGETATION_CLASSES)
    building_mask = points.classification == BUILDING_CLASS
    vegetation = points.subset(vegetation_mask)
    building_points = points.subset(building_mask)
    ground, valid_ground = rasters.ground_at(vegetation.x, vegetation.y)
    normalized = np.full(len(vegetation), np.nan, dtype=np.float64)
    normalized[valid_ground] = vegetation.z[valid_ground] - ground[valid_ground]
    physically_valid = valid_ground & (normalized >= 0)
    invalid_ground_count = int(np.count_nonzero(~valid_ground))
    negative_height_count = int(np.count_nonzero(valid_ground & (normalized < 0)))
    normalized[~physically_valid] = np.nan

    hedge_features, hedge_exclusion, hedge_stats = measure_hedges(
        hedges, vegetation, normalized, ground, parameters
    )
    tree_features, crown_features, tree_stats = detect_trees(
        vegetation,
        normalized,
        ground,
        hedge_exclusion,
        aoi.bounds,
        parameters,
        zone_id=zone_id,
    )
    building_features, building_stats = measure_buildings(
        buildings, building_points, rasters, parameters
    )
    collections = {
        "buildings": _feature_collection("montmaur_buildings_l93", building_features),
        "trees": _feature_collection("montmaur_detected_trees_l93", tree_features),
        "tree_crowns": _feature_collection(
            "montmaur_detected_tree_crowns_l93", crown_features
        ),
        "hedges": _feature_collection("montmaur_hedges_l93", hedge_features),
    }
    stats = {
        "points": {
            "input_selected_class_point_count": len(points),
            "vegetation_point_count": len(vegetation),
            "building_class_6_point_count": len(building_points),
            "vegetation_missing_mnt_count": invalid_ground_count,
            "vegetation_negative_normalized_height_count": negative_height_count,
        },
        "buildings": building_stats,
        "hedges": hedge_stats,
        "trees": tree_stats,
    }
    return collections, stats


def produce_detail(
    *,
    aoi_path: Path,
    mnt_path: Path | Sequence[Path],
    mns_path: Path | Sequence[Path],
    copc_paths: Sequence[Path],
    buildings_path: Path,
    hedges_path: Path,
    output_dir: Path,
    parameters: DetailParameters,
    aoi_crs: str = "auto",
    buildings_crs: str = "auto",
    hedges_crs: str = "auto",
    zone_id: str = "montmaur",
    require_copc: bool = True,
    required_raster_resolution_m: float | None = 0.5,
) -> Path:
    mnt_paths = _as_paths(mnt_path)
    mns_paths = _as_paths(mns_path)
    source_paths = [
        aoi_path,
        buildings_path,
        hedges_path,
        *mnt_paths,
        *mns_paths,
        *copc_paths,
    ]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing local inputs: {missing}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite detail package: {output_dir}")

    aoi = load_aoi(aoi_path, aoi_crs)
    rasters = load_raster_pair(
        mnt_paths,
        mns_paths,
        aoi,
        required_resolution_m=required_raster_resolution_m,
    )
    buildings, building_source_stats = load_vector_features(
        buildings_path, aoi, crs=buildings_crs, kind="building"
    )
    hedges, hedge_source_stats = load_vector_features(
        hedges_path, aoi, crs=hedges_crs, kind="hedge"
    )
    points, point_sources, point_source_stats = load_classified_points(
        copc_paths, aoi, require_copc=require_copc
    )
    collections, processing_stats = produce_detail_from_data(
        aoi=aoi,
        rasters=rasters,
        points=points,
        buildings=buildings,
        hedges=hedges,
        parameters=parameters,
        zone_id=zone_id,
    )

    output_dir.mkdir(parents=True)
    output_paths = {
        "buildings": output_dir / "buildings.l93.geojson",
        "trees": output_dir / "trees-detected.l93.geojson",
        "tree_crowns": output_dir / "tree-crowns-detected.l93.geojson",
        "hedges": output_dir / "hedges.l93.geojson",
    }
    for key, path in output_paths.items():
        _write_json(path, collections[key])
    min_x, min_y, max_x, max_y = aoi.bounds
    valid_ground = rasters.mnt[np.isfinite(rasters.mnt)]
    origin_z = (
        float(math.floor(float(valid_ground.min()))) if len(valid_ground) else 0.0
    )
    manifest = {
        "schema_version": "1.0",
        "package_id": f"{zone_id}-detail-lidar-v1",
        "zone_id": zone_id,
        "crs": TARGET_CRS,
        "origin_l93": {
            "x": round((min_x + max_x) / 2.0, 3),
            "y": round((min_y + max_y) / 2.0, 3),
            "z": origin_z,
            "method": "aoi_bounds_center_and_floor_minimum_observed_mnt",
        },
        "aoi": {
            **_source_asset(aoi_path),
            "bounds_l93": [round(value, 3) for value in aoi.bounds],
            "area_m2": round(aoi.area, 3),
        },
        "inputs": {
            "mnt": [
                {**_source_asset(path), **metadata}
                for path, metadata in zip(mnt_paths, rasters.source_metadata["mnt"])
            ],
            "mns": [
                {**_source_asset(path), **metadata}
                for path, metadata in zip(mns_paths, rasters.source_metadata["mns"])
            ],
            "raster_mosaic": rasters.source_metadata["mosaic"],
            "copc": point_sources,
            "buildings": {
                **_source_asset(buildings_path),
                "selection": building_source_stats,
            },
            "hedges": {**_source_asset(hedges_path), "selection": hedge_source_stats},
        },
        "parameters": parameters.__dict__,
        "required_raster_resolution_m": required_raster_resolution_m,
        "statistics": {"source_points": point_source_stats, **processing_stats},
        "outputs": {
            key: _output_asset(path, len(collections[key]["features"]))
            for key, path in output_paths.items()
        },
        "tree_detection_contract": {
            "claim": "detected_crown_apices_only_not_every_physical_tree",
            "apex": "highest_observed_class_3_4_5_return in deterministic local maximum cell",
            "segmentation": "variable-radius apex suppression then nearest-apex crown assignment",
            "diameter": "twice p95 radial distance of assigned classified returns",
            "known_false_negatives": [
                "occluded or unobserved stems",
                "merged overlapping crowns",
                "trees below thresholds or without enough classified returns",
                "returns misclassified by the source dataset",
                "trees removed from candidate points because they overlap a BD TOPO hedge corridor",
            ],
            "invented_tree_geometry": False,
            "exhaustive_tree_inventory": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "laspy": laspy.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "rasterio": rasterio.__version__,
            "shapely": shapely.__version__,
            "pyproj": pyproj.__version__,
            "pdal_cli_used": False,
            "gdal_cli_used": False,
            "network_access": "none",
        },
    }
    manifest_path = output_dir / "detail-manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", type=Path, required=True)
    parser.add_argument("--aoi-crs", default="auto")
    parser.add_argument("--mnt", type=Path, action="append", required=True)
    parser.add_argument("--mns", type=Path, action="append", required=True)
    parser.add_argument("--copc", type=Path, action="append", required=True)
    parser.add_argument("--buildings", type=Path, required=True)
    parser.add_argument("--buildings-crs", default="auto")
    parser.add_argument("--hedges", type=Path, required=True)
    parser.add_argument("--hedges-crs", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zone-id", default="montmaur")
    parser.add_argument("--canopy-cell-m", type=float, default=0.5)
    parser.add_argument("--local-peak-radius-m", type=float, default=1.5)
    parser.add_argument("--min-tree-height-m", type=float, default=2.0)
    parser.add_argument("--min-tree-points", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parameters = DetailParameters(
        canopy_cell_m=args.canopy_cell_m,
        local_peak_radius_m=args.local_peak_radius_m,
        min_tree_height_m=args.min_tree_height_m,
        min_tree_points=args.min_tree_points,
    )
    manifest_path = produce_detail(
        aoi_path=args.aoi,
        mnt_path=args.mnt,
        mns_path=args.mns,
        copc_paths=args.copc,
        buildings_path=args.buildings,
        hedges_path=args.hedges,
        output_dir=args.output_dir,
        parameters=parameters,
        aoi_crs=args.aoi_crs,
        buildings_crs=args.buildings_crs,
        hedges_crs=args.hedges_crs,
        zone_id=args.zone_id,
        require_copc=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "ok",
                "manifest": str(manifest_path.resolve()),
                "outputs": manifest["outputs"],
                "statistics": manifest["statistics"],
                "tree_detection_claim": manifest["tree_detection_contract"]["claim"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
