"""Prepare clipped Lambert-93 building and vegetation blocks for Blender.

This command is deliberately offline: every source is supplied as a local file.
GeoJSON inputs use RFC 7946 longitude/latitude coordinates (WGS84). Raster
coordinates are validated as a north-up metropolitan Lambert-93 grid, then
explicitly interpreted as EPSG:2154 because IGN GeoTIFF WKT metadata can be
incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.geometry import (
    GeometryCollection,
    MultiPolygon,
    Polygon,
    box,
    mapping,
    shape,
)
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union
from shapely.validation import make_valid


SOURCE_CRS = "EPSG:4326"
TARGET_CRS = "EPSG:2154"
MIN_VEGETATION_SAMPLES = 3
MIN_VALID_RATIO = 0.25
GOOD_VALID_RATIO = 0.75
MAX_CREDIBLE_BDTOPO_Z_PRECISION_METRES = 5.0


@dataclass(frozen=True)
class ZonalSamples:
    total_count: int
    mnt: np.ndarray
    height: np.ndarray

    @property
    def mnt_count(self) -> int:
        return int(self.mnt.size)

    @property
    def height_count(self) -> int:
        return int(self.height.size)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _geojson_geometries(document: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(document, dict):
        raise ValueError("GeoJSON root must be an object")
    kind = document.get("type")
    if kind == "FeatureCollection":
        for feature in document.get("features", []):
            geometry = feature.get("geometry") if isinstance(feature, dict) else None
            if geometry:
                yield geometry
        return
    if kind == "Feature":
        geometry = document.get("geometry")
        if geometry:
            yield geometry
        return
    if kind:
        yield document
        return
    raise ValueError("Unsupported GeoJSON root")


def _polygonal_only(geometry: Any) -> Polygon | MultiPolygon | None:
    if geometry.is_empty:
        return None
    geometry = make_valid(geometry)
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygons: list[Polygon] = []
        for part in geometry.geoms:
            polygonal = _polygonal_only(part)
            if isinstance(polygonal, Polygon):
                polygons.append(polygonal)
            elif isinstance(polygonal, MultiPolygon):
                polygons.extend(polygonal.geoms)
        if not polygons:
            return None
        merged = unary_union(polygons)
        return _polygonal_only(merged)
    return None


def load_aoi_l93(path: Path) -> Polygon | MultiPolygon:
    transformer = Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True)
    projected = []
    for raw_geometry in _geojson_geometries(_load_json(path)):
        geometry = _polygonal_only(
            transform_geometry(transformer.transform, shape(raw_geometry))
        )
        if geometry is not None:
            projected.append(geometry)
    if not projected:
        raise ValueError(f"AOI contains no polygon: {path}")
    union = _polygonal_only(unary_union(projected))
    if union is None or union.area <= 0:
        raise ValueError(f"AOI has no positive polygon area: {path}")
    return union


def _observed_crs(dataset: rasterio.io.DatasetReader) -> str | None:
    return dataset.crs.to_string() if dataset.crs is not None else None


def validate_l93_raster(
    dataset: rasterio.io.DatasetReader, aoi_l93: Polygon | MultiPolygon
) -> dict[str, Any]:
    transform = dataset.transform
    coefficients = np.asarray(tuple(transform), dtype=np.float64)
    if not np.isfinite(coefficients).all():
        raise ValueError(f"Raster transform is not finite: {dataset.name}")
    if (
        transform.a <= 0
        or transform.e >= 0
        or abs(transform.b) > 1e-9
        or abs(transform.d) > 1e-9
    ):
        raise ValueError(
            f"Raster must be a north-up grid: {dataset.name} ({transform})"
        )
    if dataset.count < 1 or dataset.width <= 0 or dataset.height <= 0:
        raise ValueError(f"Raster has no usable first band: {dataset.name}")

    bounds = dataset.bounds
    values = np.asarray(
        (bounds.left, bounds.bottom, bounds.right, bounds.top), dtype=np.float64
    )
    if (
        not np.isfinite(values).all()
        or bounds.left >= bounds.right
        or bounds.bottom >= bounds.top
    ):
        raise ValueError(f"Raster bounds are invalid: {dataset.name} ({bounds})")
    # Broad metropolitan-France Lambert-93 envelope. This catches degrees and
    # Web Mercator while accepting IGN grids near the national boundary.
    if not (
        -200_000 <= bounds.left <= 1_500_000 and -200_000 <= bounds.right <= 1_500_000
    ):
        raise ValueError(
            f"Raster X bounds are not plausible Lambert-93 metres: {dataset.name} ({bounds})"
        )
    if not (
        5_500_000 <= bounds.bottom <= 7_300_000 and 5_500_000 <= bounds.top <= 7_300_000
    ):
        raise ValueError(
            f"Raster Y bounds are not plausible Lambert-93 metres: {dataset.name} ({bounds})"
        )
    if not box(*bounds).intersects(aoi_l93):
        raise ValueError(f"Raster does not intersect the AOI: {dataset.name}")

    observed_epsg = dataset.crs.to_epsg() if dataset.crs is not None else None
    if observed_epsg is not None and observed_epsg != 2154:
        raise ValueError(
            f"Raster declares EPSG:{observed_epsg}, expected Lambert-93: {dataset.name}"
        )
    if dataset.crs is not None and dataset.crs.is_geographic:
        raise ValueError(
            f"Raster declares a geographic CRS but has metre-like coordinates: {dataset.name}"
        )

    return {
        "observed_crs": _observed_crs(dataset),
        "assigned_crs": TARGET_CRS,
        "assignment": "explicit_after_north_up_transform_and_lambert93_bounds_validation",
        "bounds_l93": [
            float(bounds.left),
            float(bounds.bottom),
            float(bounds.right),
            float(bounds.top),
        ],
        "shape": [dataset.height, dataset.width],
        "pixel_size_m": [float(transform.a), float(abs(transform.e))],
        "nodata": None if dataset.nodata is None else float(dataset.nodata),
        "dtype": dataset.dtypes[0],
    }


def require_aligned_grids(
    mnt: rasterio.io.DatasetReader,
    mns: rasterio.io.DatasetReader,
) -> None:
    if (
        mnt.width != mns.width
        or mnt.height != mns.height
        or not np.allclose(
            tuple(mnt.transform), tuple(mns.transform), rtol=0.0, atol=1e-9
        )
    ):
        raise ValueError("MNT and MNS grids must have identical shape and transform")


def _window_for_geometry(
    dataset: rasterio.io.DatasetReader, geometry: Polygon | MultiPolygon
) -> Window | None:
    clipped_bounds = box(*dataset.bounds).intersection(box(*geometry.bounds))
    if clipped_bounds.is_empty:
        return None
    fractional = from_bounds(*clipped_bounds.bounds, transform=dataset.transform)
    col_start = max(0, math.floor(fractional.col_off))
    row_start = max(0, math.floor(fractional.row_off))
    col_stop = min(dataset.width, math.ceil(fractional.col_off + fractional.width))
    row_stop = min(dataset.height, math.ceil(fractional.row_off + fractional.height))
    if col_stop <= col_start or row_stop <= row_start:
        return None
    return Window(col_start, row_start, col_stop - col_start, row_stop - row_start)


def zonal_samples(
    mnt: rasterio.io.DatasetReader,
    mns: rasterio.io.DatasetReader,
    geometry: Polygon | MultiPolygon,
) -> ZonalSamples:
    window = _window_for_geometry(mnt, geometry)
    if window is None:
        return ZonalSamples(total_count=0, mnt=np.empty(0), height=np.empty(0))
    mnt_values = mnt.read(1, window=window, masked=True)
    mns_values = mns.read(1, window=window, masked=True)
    inside = geometry_mask(
        [mapping(geometry)],
        out_shape=mnt_values.shape,
        transform=mnt.window_transform(window),
        invert=True,
        all_touched=False,
    )
    total_count = int(np.count_nonzero(inside))
    mnt_data = np.asarray(mnt_values.data, dtype=np.float64)
    mns_data = np.asarray(mns_values.data, dtype=np.float64)
    valid_mnt = inside & ~np.ma.getmaskarray(mnt_values) & np.isfinite(mnt_data)
    valid_both = valid_mnt & ~np.ma.getmaskarray(mns_values) & np.isfinite(mns_data)
    ground = mnt_data[valid_mnt]
    height = np.maximum(0.0, mns_data[valid_both] - mnt_data[valid_both])
    return ZonalSamples(total_count=total_count, mnt=ground, height=height)


def sample_mnt(
    dataset: rasterio.io.DatasetReader, geometry: Polygon | MultiPolygon
) -> float | None:
    point = geometry.representative_point()
    if not box(*dataset.bounds).covers(point):
        return None
    sampled = next(dataset.sample([(point.x, point.y)], indexes=1, masked=True))
    value = sampled[0]
    if np.ma.is_masked(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_number(value: Any) -> float | None:
    number = _finite_number(value)
    return number if number is not None and number > 0 else None


def _bdtopo_z_quality(properties: dict[str, Any]) -> tuple[bool, str]:
    precision = _finite_number(properties.get("precision_altimetrique"))
    method_value = properties.get("methode_d_acquisition_altimetrique")
    method = str(method_value).strip().casefold() if method_value is not None else ""
    if any(marker in method for marker in ("pas de z", "sans z", "aucun z")):
        return False, "rejected_method_without_z"
    if precision is None:
        return False, "rejected_missing_altimetric_precision"
    if precision < 0 or precision > MAX_CREDIBLE_BDTOPO_Z_PRECISION_METRES:
        return False, "rejected_altimetric_precision"
    return True, "credible_bdtopo_z"


def _quality(
    valid_count: int, total_count: int, minimum_count: int
) -> tuple[str, float, bool]:
    ratio = valid_count / total_count if total_count else 0.0
    sufficient = valid_count >= minimum_count and ratio >= MIN_VALID_RATIO
    if not sufficient:
        return "insufficient", ratio, False
    return ("good" if ratio >= GOOD_VALID_RATIO else "partial"), ratio, True


def _round_optional(value: float | None) -> float | None:
    return None if value is None else round(float(value), 3)


def _building_properties(
    original: dict[str, Any],
    geometry: Polygon | MultiPolygon,
    mnt: rasterio.io.DatasetReader,
    mns: rasterio.io.DatasetReader,
) -> dict[str, Any]:
    properties = dict(original)
    credible_z, z_quality = _bdtopo_z_quality(original)
    source_base = (
        _finite_number(original.get("altitude_minimale_sol")) if credible_z else None
    )
    mnt_base = sample_mnt(mnt, geometry)
    if mnt_base is not None:
        base = mnt_base
        base_method: str | None = "mnt_representative_point_aligned"
    else:
        base = source_base
        base_method = (
            "bdtopo_altitude_minimale_sol_fallback" if base is not None else None
        )
    source_base_delta_to_mnt = (
        source_base - mnt_base
        if source_base is not None and mnt_base is not None
        else None
    )

    height = _positive_number(original.get("hauteur"))
    height_method: str | None = "bdtopo_hauteur" if height is not None else None
    height_count = 0
    height_total = 0
    height_ratio = 0.0
    height_quality = "source" if height is not None else "not_sampled"

    if height is None and base is not None:
        roof = (
            _finite_number(original.get("altitude_maximale_toit"))
            if credible_z
            else None
        )
        roof_reference_base = source_base if source_base is not None else base
        roof_height = None if roof is None else roof - roof_reference_base
        if roof_height is not None and roof_height > 0:
            height = roof_height
            height_method = "bdtopo_altitude_maximale_toit_minus_base"
            height_quality = "source"

    if height is None:
        samples = zonal_samples(mnt, mns, geometry)
        height_count = samples.height_count
        height_total = samples.total_count
        height_quality, height_ratio, sufficient = _quality(
            height_count, height_total, 1
        )
        if sufficient:
            raster_height = float(np.percentile(samples.height, 75))
            if raster_height > 0:
                height = raster_height
                height_method = "raster_p75_clamped_mns_minus_mnt"
            else:
                height_quality = "non_positive"

    properties.update(
        {
            "base_elevation_m": _round_optional(base),
            "base_method": base_method,
            "source_base_elevation_m": _round_optional(source_base),
            "source_base_delta_to_mnt_m": _round_optional(source_base_delta_to_mnt),
            "bdtopo_z_quality": z_quality,
            "block_height_m": _round_optional(height),
            "height_method": height_method,
            "height_sample_count": height_count,
            "height_total_pixel_count": height_total,
            "height_valid_ratio": round(height_ratio, 6),
            "height_quality": height_quality,
        }
    )
    return properties


def _vegetation_properties(
    original: dict[str, Any],
    geometry: Polygon | MultiPolygon,
    mnt: rasterio.io.DatasetReader,
    mns: rasterio.io.DatasetReader,
) -> dict[str, Any]:
    properties = dict(original)
    samples = zonal_samples(mnt, mns, geometry)
    base_quality, base_ratio, base_sufficient = _quality(
        samples.mnt_count, samples.total_count, MIN_VEGETATION_SAMPLES
    )
    height_quality, height_ratio, height_sufficient = _quality(
        samples.height_count, samples.total_count, MIN_VEGETATION_SAMPLES
    )
    base = float(np.median(samples.mnt)) if base_sufficient else None
    height = float(np.percentile(samples.height, 75)) if height_sufficient else None
    properties.update(
        {
            "base_elevation_m": _round_optional(base),
            "base_method": "raster_mnt_median" if base is not None else None,
            "base_sample_count": samples.mnt_count,
            "base_valid_ratio": round(base_ratio, 6),
            "base_quality": base_quality,
            "block_height_m": _round_optional(height),
            "height_method": "raster_p75_clamped_mns_minus_mnt"
            if height is not None
            else None,
            "height_sample_count": samples.height_count,
            "height_total_pixel_count": samples.total_count,
            "height_valid_ratio": round(height_ratio, 6),
            "height_quality": height_quality,
        }
    )
    return properties


def _features(document: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(document, dict) or document.get("type") != "FeatureCollection":
        raise ValueError(f"Expected a GeoJSON FeatureCollection: {path}")
    features = document.get("features")
    if not isinstance(features, list):
        raise ValueError(f"FeatureCollection has no feature list: {path}")
    return [feature for feature in features if isinstance(feature, dict)]


def merge_deduplicate(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    input_count = 0
    duplicate_count = 0
    for path in paths:
        for feature in _features(_load_json(path), path):
            input_count += 1
            properties = feature.get("properties") or {}
            cleabs = properties.get("cleabs") if isinstance(properties, dict) else None
            identity = str(cleabs).strip() if cleabs is not None else ""
            if identity and identity in seen:
                duplicate_count += 1
                continue
            if identity:
                seen.add(identity)
            merged.append(feature)
    return merged, {
        "input_feature_count": input_count,
        "duplicate_cleabs_count": duplicate_count,
        "deduplicated_feature_count": len(merged),
    }


def _prepare_layer(
    features: Sequence[dict[str, Any]],
    aoi_l93: Polygon | MultiPolygon,
    mnt: rasterio.io.DatasetReader,
    mns: rasterio.io.DatasetReader,
    kind: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    transformer = Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True)
    output: list[dict[str, Any]] = []
    rejected_geometry = 0
    outside_aoi = 0
    for feature in features:
        raw_geometry = feature.get("geometry")
        if not raw_geometry:
            rejected_geometry += 1
            continue
        try:
            projected = _polygonal_only(
                transform_geometry(transformer.transform, shape(raw_geometry))
            )
        except Exception:
            projected = None
        if projected is None:
            rejected_geometry += 1
            continue
        if not projected.intersects(aoi_l93):
            outside_aoi += 1
            continue
        clipped = _polygonal_only(projected.intersection(aoi_l93))
        if clipped is None or clipped.area <= 0:
            outside_aoi += 1
            continue
        original_properties = feature.get("properties") or {}
        if not isinstance(original_properties, dict):
            original_properties = {}
        properties = (
            _building_properties(original_properties, clipped, mnt, mns)
            if kind == "buildings"
            else _vegetation_properties(original_properties, clipped, mnt, mns)
        )
        result: dict[str, Any] = {
            "type": "Feature",
            "properties": properties,
            "geometry": mapping(clipped),
        }
        if "id" in feature:
            result["id"] = feature["id"]
        output.append(result)
    return (
        {
            "type": "FeatureCollection",
            "name": f"{kind}_l93",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:EPSG::2154"},
            },
            "features": output,
        },
        {
            "rejected_geometry_count": rejected_geometry,
            "outside_aoi_count": outside_aoi,
            "output_feature_count": len(output),
        },
    )


def _source_asset(path: Path) -> dict[str, Any]:
    """Describe an external snapshot without leaking a workstation path."""
    return {
        "file_name": path.name,
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _catalog_asset(path: Path, package_root: Path, **metadata: Any) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(package_root.resolve()).as_posix(),
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
        **metadata,
    }


def attach_to_package(package_root: Path, vector_manifest_path: Path) -> None:
    """Atomically replace package indexes after vector outputs are complete.

    Individual vector files are already installed with atomic replacements. The
    catalog is replaced only after all three hashes are known, and the package
    manifest is replaced last with the new catalog hash.
    """
    root = package_root.resolve()
    catalog_path = root / "catalog.json"
    package_manifest_path = root / "package-manifest.json"
    if not catalog_path.is_file() or not package_manifest_path.is_file():
        raise FileNotFoundError(
            f"Package root lacks catalog.json or package-manifest.json: {root}"
        )
    vector_root = vector_manifest_path.resolve().parent
    if vector_root != root / "vectors":
        raise ValueError(
            f"Vector outputs must be written to {root / 'vectors'} when --package-root is used"
        )

    vector_manifest = _load_json(vector_manifest_path)
    building_path = vector_root / vector_manifest["outputs"]["buildings"]["path"]
    vegetation_path = vector_root / vector_manifest["outputs"]["vegetation"]["path"]
    catalog = _load_json(catalog_path)
    layers = catalog.get("layers")
    deferred = catalog.get("deferred_layers")
    if not isinstance(layers, dict) or not isinstance(deferred, dict):
        raise ValueError("Package catalog lacks layers or deferred_layers dictionaries")

    layers["buildings_l93"] = _catalog_asset(
        building_path,
        root,
        format="GeoJSON",
        crs=TARGET_CRS,
        geometry_type="Polygon/MultiPolygon",
        feature_count=vector_manifest["outputs"]["buildings"]["feature_count"],
        role="simple_global_building_blocks",
    )
    layers["vegetation_blocks_l93"] = _catalog_asset(
        vegetation_path,
        root,
        format="GeoJSON",
        crs=TARGET_CRS,
        geometry_type="Polygon/MultiPolygon",
        feature_count=vector_manifest["outputs"]["vegetation"]["feature_count"],
        role="simple_global_vegetation_blocks",
    )
    layers["vector_model_manifest"] = _catalog_asset(
        vector_manifest_path,
        root,
        format="JSON",
        role="building_and_vegetation_derivation_contract",
    )
    deferred["buildings"] = {
        "status": "produced",
        "layer": "buildings_l93",
        "feature_count": vector_manifest["outputs"]["buildings"]["feature_count"],
    }
    deferred["vegetation_blocks"] = {
        "status": "produced",
        "layer": "vegetation_blocks_l93",
        "feature_count": vector_manifest["outputs"]["vegetation"]["feature_count"],
    }
    limitations = catalog.get("limitations")
    if isinstance(limitations, list):
        catalog["limitations"] = [
            limitation
            for limitation in limitations
            if limitation
            != "No BD TOPO building, vegetation or hedge data is included in this revision."
        ]
        note = (
            "Building and vegetation blocks are simplified global-view volumes; "
            "their measured derivation is recorded in vectors/vector-manifest.json."
        )
        if note not in catalog["limitations"]:
            catalog["limitations"].append(note)

    _write_json(catalog_path, catalog)
    package_manifest = _load_json(package_manifest_path)
    package_manifest["catalog"] = {
        "path": "catalog.json",
        "format": "JSON",
        "byte_count": catalog_path.stat().st_size,
        "sha256": sha256_file(catalog_path),
    }
    processing = package_manifest.setdefault("processing", {})
    processing["vector_blocks_publication"] = "produced"
    _write_json(package_manifest_path, package_manifest)


def prepare_vector_blocks(
    *,
    aoi_path: Path,
    mnt_path: Path,
    mns_path: Path,
    building_pages: Sequence[Path],
    vegetation_path: Path,
    output_dir: Path,
    package_root: Path | None = None,
) -> Path:
    paths = [aoi_path, mnt_path, mns_path, vegetation_path, *building_pages]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")
    if not building_pages:
        raise ValueError("At least one --building-pages file is required")

    aoi_l93 = load_aoi_l93(aoi_path)
    buildings, building_merge_stats = merge_deduplicate(building_pages)
    vegetation, vegetation_merge_stats = merge_deduplicate([vegetation_path])
    output_dir.mkdir(parents=True, exist_ok=True)
    building_output = output_dir / "buildings.l93.geojson"
    vegetation_output = output_dir / "vegetation.l93.geojson"

    with rasterio.open(mnt_path) as mnt, rasterio.open(mns_path) as mns:
        mnt_metadata = validate_l93_raster(mnt, aoi_l93)
        mns_metadata = validate_l93_raster(mns, aoi_l93)
        require_aligned_grids(mnt, mns)
        building_collection, building_layer_stats = _prepare_layer(
            buildings, aoi_l93, mnt, mns, "buildings"
        )
        vegetation_collection, vegetation_layer_stats = _prepare_layer(
            vegetation, aoi_l93, mnt, mns, "vegetation"
        )

    _write_json(building_output, building_collection)
    _write_json(vegetation_output, vegetation_collection)
    min_x, min_y, max_x, max_y = aoi_l93.bounds
    manifest = {
        "schema_version": "1.0",
        "crs": TARGET_CRS,
        "origin_l93": {
            "x": round((min_x + max_x) / 2.0, 3),
            "y": round((min_y + max_y) / 2.0, 3),
            "z": 0.0,
            "method": "aoi_bounds_center",
        },
        "aoi": {
            **_source_asset(aoi_path),
            "bounds_l93": [round(value, 3) for value in aoi_l93.bounds],
            "area_m2": round(aoi_l93.area, 3),
        },
        "inputs": {
            "mnt": {**_source_asset(mnt_path), **mnt_metadata},
            "mns": {**_source_asset(mns_path), **mns_metadata},
            "building_pages": [_source_asset(path) for path in building_pages],
            "vegetation": _source_asset(vegetation_path),
        },
        "outputs": {
            "buildings": {
                "path": building_output.name,
                "bytes": building_output.stat().st_size,
                "sha256": sha256_file(building_output),
                "feature_count": len(building_collection["features"]),
            },
            "vegetation": {
                "path": vegetation_output.name,
                "bytes": vegetation_output.stat().st_size,
                "sha256": sha256_file(vegetation_output),
                "feature_count": len(vegetation_collection["features"]),
            },
        },
        "statistics": {
            "buildings": {**building_merge_stats, **building_layer_stats},
            "vegetation": {**vegetation_merge_stats, **vegetation_layer_stats},
        },
        "height_contract": {
            "bdtopo_altitude_acceptance": {
                "required": True,
                "maximum_precision_altimetrique_m": MAX_CREDIBLE_BDTOPO_Z_PRECISION_METRES,
                "rejected_method_markers": ["pas de z", "sans z", "aucun z"],
                "missing_precision_is_credible": False,
                "use": "quality_control_and_fallback_when_mnt_is_unavailable",
            },
            "building_base_order": [
                "mnt_representative_point_aligned",
                "credible_bdtopo_altitude_minimale_sol_fallback",
                "null",
            ],
            "building_order": [
                "positive_bdtopo_hauteur",
                "positive_bdtopo_altitude_maximale_toit_minus_base",
                "positive_raster_p75_clamped_mns_minus_mnt",
                "null",
            ],
            "vegetation": "raster_p75_of_max_0_mns_minus_mnt",
            "vegetation_minimum_samples": MIN_VEGETATION_SAMPLES,
            "minimum_valid_ratio": MIN_VALID_RATIO,
            "invented_default_height": None,
        },
    }
    manifest_path = output_dir / "vector-manifest.json"
    _write_json(manifest_path, manifest)
    if package_root is not None:
        attach_to_package(package_root, manifest_path)
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aoi", type=Path, required=True, help="Buffered AOI GeoJSON in WGS84"
    )
    parser.add_argument("--mnt", type=Path, required=True, help="Local IGN MNT GeoTIFF")
    parser.add_argument("--mns", type=Path, required=True, help="Local IGN MNS GeoTIFF")
    parser.add_argument(
        "--building-pages",
        type=Path,
        action="append",
        required=True,
        help="Local BD TOPO building GeoJSON page; repeat for every page",
    )
    parser.add_argument(
        "--vegetation",
        type=Path,
        required=True,
        help="Local BD TOPO vegetation GeoJSON",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory (defaults to PACKAGE/vectors)"
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        help="Attach outputs to an existing hybrid package catalog and package manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is None and args.package_root is None:
        raise SystemExit("Either --output-dir or --package-root is required")
    output_dir = args.output_dir or (args.package_root / "vectors")
    if args.package_root is not None and output_dir.resolve() != (
        args.package_root.resolve() / "vectors"
    ):
        raise SystemExit("With --package-root, --output-dir must be PACKAGE/vectors")
    manifest_path = prepare_vector_blocks(
        aoi_path=args.aoi,
        mnt_path=args.mnt,
        mns_path=args.mns,
        building_pages=args.building_pages,
        vegetation_path=args.vegetation,
        output_dir=output_dir,
        package_root=args.package_root,
    )
    manifest = _load_json(manifest_path)
    print(
        json.dumps(
            {
                "status": "ok",
                "manifest": str(manifest_path.resolve()),
                "outputs": manifest["outputs"],
                "statistics": manifest["statistics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
