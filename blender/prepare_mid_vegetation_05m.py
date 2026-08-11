"""Segment 0.5 m MNS/MNT vegetation into measured mid-distance instances.

The output is intentionally a compact point catalogue, not baked tree meshes.
Blender can instance a small library of recognisable tree prototypes at these
measured apex positions without duplicating geometry hundreds of thousands of
times. Processing is sector-oriented so the full incident is never loaded as
one 0.5 m raster.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rasterio.features import geometry_mask
from rasterio.merge import merge as merge_rasters
from scipy.ndimage import find_objects, gaussian_filter, label, maximum_filter
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from skimage.segmentation import watershed

from tree_instances import TreeInstanceConfig, build_tree_instance_set


SCHEMA = "fireviewer.vegetation-mid-distance-0m50.v1"


@dataclass(frozen=True)
class SegmentationConfig:
    min_tree_height_m: float = 2.0
    local_peak_radius_m: float = 1.0
    smoothing_sigma_m: float = 0.5
    min_crown_area_m2: float = 0.75
    min_crown_radius_m: float = 1.25
    max_crown_radius_m: float = 8.0
    crown_radius_height_ratio: float = 0.35
    crown_support_height_ratio: float = 0.18
    coordinate_precision: int = 3

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name == "coordinate_precision":
                if not isinstance(value, int) or not 0 <= value <= 9:
                    raise ValueError("coordinate_precision must be between 0 and 9")
            elif not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and strictly positive")
        if self.max_crown_radius_m < self.min_crown_radius_m:
            raise ValueError("max_crown_radius_m must be >= min_crown_radius_m")
        if not 0 < self.crown_support_height_ratio <= 1:
            raise ValueError("crown_support_height_ratio must be in ]0, 1]")


def _pixel_metrics(transform: Any) -> tuple[float, float, float]:
    determinant = float(transform.a) * float(transform.e) - float(transform.b) * float(
        transform.d
    )
    pixel_area = abs(determinant)
    if not math.isfinite(pixel_area) or pixel_area <= 0:
        raise ValueError("Raster transform must have a finite non-zero pixel area")
    x_size = math.hypot(float(transform.a), float(transform.d))
    y_size = math.hypot(float(transform.b), float(transform.e))
    nominal = math.sqrt(pixel_area)
    return x_size, y_size, nominal


def _normalised_smoothing(
    canopy_height: np.ndarray,
    eligible: np.ndarray,
    sigma_pixels: float,
) -> np.ndarray:
    values = np.where(eligible, canopy_height, 0.0)
    weights = gaussian_filter(
        eligible.astype("float32"), sigma=sigma_pixels, mode="constant"
    )
    smoothed_values = gaussian_filter(values, sigma=sigma_pixels, mode="constant")
    result = np.full(canopy_height.shape, -np.inf, dtype="float64")
    valid = weights > 1e-6
    result[valid] = smoothed_values[valid] / weights[valid]
    return result


def _plateau_peaks(
    canopy_height: np.ndarray,
    smoothed: np.ndarray,
    eligible: np.ndarray,
    radius_pixels: int,
) -> list[tuple[int, int]]:
    neighbourhood = maximum_filter(
        smoothed,
        size=radius_pixels * 2 + 1,
        mode="constant",
        cval=-np.inf,
    )
    candidates = eligible & np.isclose(smoothed, neighbourhood, rtol=0.0, atol=1e-9)
    components, component_count = label(
        candidates, structure=np.ones((3, 3), dtype=bool)
    )
    peaks: list[tuple[int, int]] = []
    # ``components == component_id`` over the complete 0.5 m raster made the
    # former loop O(pixel_count * peak_count), which is prohibitive in dense
    # forest.  ``find_objects`` supplies the tight bounding slice for every
    # component while preserving the exact deterministic selection below.
    for component_id, component_slice in enumerate(
        find_objects(components, max_label=component_count), start=1
    ):
        if component_slice is None:
            continue
        local = components[component_slice]
        local_rows, local_columns = np.nonzero(local == component_id)
        rows = local_rows + int(component_slice[0].start or 0)
        columns = local_columns + int(component_slice[1].start or 0)
        if not len(rows):
            continue
        heights = canopy_height[rows, columns]
        maximum = float(np.max(heights))
        finalists = np.flatnonzero(heights == maximum)
        chosen = min(
            finalists, key=lambda index: (int(rows[index]), int(columns[index]))
        )
        peaks.append((int(rows[chosen]), int(columns[chosen])))
    peaks.sort(
        key=lambda point: (
            -float(canopy_height[point]),
            point[0],
            point[1],
        )
    )
    return peaks


def _variant(row: int, column: int) -> tuple[int, float]:
    value = (row * 73_856_093) ^ (column * 19_349_663)
    variant = abs(value) % 6
    rotation_degrees = float(abs(value // 7) % 360)
    return variant, rotation_degrees


def segment_vegetation_instances(
    mnt: Any,
    mns: Any,
    transform: Any,
    valid_mask: Any,
    exclusion_mask: Any,
    origin: Sequence[float],
    config: SegmentationConfig | None = None,
) -> tuple[list[list[float | int]], dict[str, Any]]:
    """Return every accepted 0.5 m crown apex without a second spacing filter."""

    active = config or SegmentationConfig()
    active.validate()
    ground = np.asarray(mnt, dtype="float64")
    surface = np.asarray(mns, dtype="float64")
    valid_area = np.asarray(valid_mask, dtype=bool)
    exclusions = np.asarray(exclusion_mask, dtype=bool)
    if ground.ndim != 2 or ground.shape != surface.shape:
        raise ValueError("MNT and MNS must be aligned two-dimensional arrays")
    for name, mask in (("valid_mask", valid_area), ("exclusion_mask", exclusions)):
        if mask.shape != ground.shape:
            raise ValueError(
                f"{name} shape mismatch: expected {ground.shape}, got {mask.shape}"
            )
    if len(origin) != 3 or not all(math.isfinite(float(value)) for value in origin):
        raise ValueError("origin must contain three finite coordinates")

    x_size, y_size, nominal_pixel_size = _pixel_metrics(transform)
    finite = np.isfinite(ground) & np.isfinite(surface)
    canopy_height = np.maximum(surface - ground, 0.0)
    eligible = (
        valid_area & ~exclusions & finite & (canopy_height >= active.min_tree_height_m)
    )
    sigma_pixels = active.smoothing_sigma_m / nominal_pixel_size
    smoothed = _normalised_smoothing(canopy_height, eligible, sigma_pixels)
    radius_pixels = max(
        1, int(math.ceil(active.local_peak_radius_m / nominal_pixel_size))
    )
    peaks = _plateau_peaks(canopy_height, smoothed, eligible, radius_pixels)
    markers = np.zeros(ground.shape, dtype="int32")
    for marker_id, (row, column) in enumerate(peaks, start=1):
        markers[row, column] = marker_id
    labels = (
        watershed(-smoothed, markers=markers, mask=eligible, connectivity=2)
        if peaks
        else markers
    )

    origin_x, origin_y, origin_z = (float(value) for value in origin)
    precision = active.coordinate_precision
    instances: list[list[float | int]] = []
    rejected_small_crown_count = 0
    crown_areas: list[float] = []
    heights: list[float] = []
    for marker_id, (row, column) in enumerate(peaks, start=1):
        tree_height = float(canopy_height[row, column])
        maximum_radius = float(
            np.clip(
                tree_height * active.crown_radius_height_ratio,
                active.min_crown_radius_m,
                active.max_crown_radius_m,
            )
        )
        row_radius = max(1, int(math.ceil(maximum_radius / max(y_size, 1e-9))))
        column_radius = max(1, int(math.ceil(maximum_radius / max(x_size, 1e-9))))
        row_start = max(0, row - row_radius)
        row_stop = min(ground.shape[0], row + row_radius + 1)
        column_start = max(0, column - column_radius)
        column_stop = min(ground.shape[1], column + column_radius + 1)
        local_rows, local_columns = np.ogrid[
            row_start - row : row_stop - row,
            column_start - column : column_stop - column,
        ]
        metric_distance_squared = (local_rows * y_size) ** 2 + (
            local_columns * x_size
        ) ** 2
        support_threshold = max(
            active.min_tree_height_m,
            tree_height * active.crown_support_height_ratio,
        )
        support = (
            (labels[row_start:row_stop, column_start:column_stop] == marker_id)
            & (metric_distance_squared <= maximum_radius**2)
            & (
                canopy_height[row_start:row_stop, column_start:column_stop]
                >= support_threshold
            )
        )
        crown_area = float(np.count_nonzero(support)) * x_size * y_size
        if crown_area < active.min_crown_area_m2:
            rejected_small_crown_count += 1
            continue
        crown_diameter = 2.0 * math.sqrt(crown_area / math.pi)
        x, y = transform * (column + 0.5, row + 0.5)
        variant, rotation_degrees = _variant(row, column)
        instances.append(
            [
                round(float(x) - origin_x, precision),
                round(float(y) - origin_y, precision),
                round(float(ground[row, column]) - origin_z, precision),
                round(tree_height, precision),
                round(crown_diameter, precision),
                variant,
                rotation_degrees,
            ]
        )
        crown_areas.append(crown_area)
        heights.append(tree_height)

    instances.sort(
        key=lambda value: (float(value[1]), float(value[0]), -float(value[3]))
    )
    statistics = {
        "semantics": "all_accepted_0m50_crown_apices_without_post_detection_thinning",
        "completeness_claim": "detected_crowns_not_field_inventory",
        "grid_shape": [int(ground.shape[0]), int(ground.shape[1])],
        "pixel_size_m": [x_size, y_size],
        "eligible_canopy_pixel_count": int(np.count_nonzero(eligible)),
        "excluded_pixel_count": int(np.count_nonzero(exclusions & valid_area)),
        "local_peak_candidate_count": len(peaks),
        "rejected_small_crown_count": rejected_small_crown_count,
        "accepted_instance_count": len(instances),
        "post_detection_spacing_rejected_count": 0,
        "minimum_height_m": min(heights) if heights else None,
        "maximum_height_m": max(heights) if heights else None,
        "minimum_crown_area_m2": min(crown_areas) if crown_areas else None,
        "maximum_crown_area_m2": max(crown_areas) if crown_areas else None,
        "attributes": [
            "local_x_m",
            "local_y_m",
            "local_ground_z_m",
            "height_m",
            "crown_diameter_m",
            "visual_variant",
            "rotation_degrees",
        ],
    }
    return instances, statistics


def build_local_terrain_mesh(
    terrain: Any,
    transform: Any,
    origin: Sequence[float],
    *,
    valid_mask: Any | None = None,
    step_pixels: int = 2,
    coordinate_precision: int = 3,
    bounds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build a light local MNT mesh while preserving measured elevations.

    When ``bounds`` is supplied, the mesh is sampled from the processing
    raster (which must include a halo) on a grid that reaches the four exact
    core boundaries.  Boundary elevations are bilinearly sampled at the same
    Lambert-93 coordinates for every adjacent tile.  This prevents the former
    quarter-pixel inset and gives neighbouring 500 m tiles coincident edges.

    The legacy pixel-centre path remains available when ``bounds`` is omitted.
    """

    values = np.asarray(terrain, dtype="float64")
    if values.ndim != 2:
        raise ValueError("terrain must be a two-dimensional array")
    if (
        isinstance(step_pixels, bool)
        or not isinstance(step_pixels, int)
        or step_pixels < 1
    ):
        raise ValueError("step_pixels must be a strictly positive integer")
    if len(origin) != 3 or not all(math.isfinite(float(value)) for value in origin):
        raise ValueError("origin must contain three finite coordinates")
    active_mask = (
        np.ones(values.shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if active_mask.shape != values.shape:
        raise ValueError(
            f"valid_mask shape mismatch: expected {values.shape}, got {active_mask.shape}"
        )
    if bounds is not None:
        return _build_bounded_local_terrain_mesh(
            values,
            transform,
            origin,
            active_mask,
            bounds,
            step_pixels=step_pixels,
            coordinate_precision=coordinate_precision,
        )
    rows = list(range(0, values.shape[0], step_pixels))
    columns = list(range(0, values.shape[1], step_pixels))
    if rows[-1] != values.shape[0] - 1:
        rows.append(values.shape[0] - 1)
    if columns[-1] != values.shape[1] - 1:
        columns.append(values.shape[1] - 1)
    origin_x, origin_y, origin_z = (float(value) for value in origin)
    vertices: list[list[float]] = []
    indices: dict[tuple[int, int], int] = {}
    for row_position, row in enumerate(rows):
        for column_position, column in enumerate(columns):
            elevation = float(values[row, column])
            if not active_mask[row, column] or not math.isfinite(elevation):
                continue
            x, y = transform * (column + 0.5, row + 0.5)
            indices[(row_position, column_position)] = len(vertices)
            vertices.append(
                [
                    round(float(x) - origin_x, coordinate_precision),
                    round(float(y) - origin_y, coordinate_precision),
                    round(elevation - origin_z, coordinate_precision),
                ]
            )
    faces: list[list[int]] = []
    for row_position in range(len(rows) - 1):
        for column_position in range(len(columns) - 1):
            keys = (
                (row_position, column_position),
                (row_position, column_position + 1),
                (row_position + 1, column_position + 1),
                (row_position + 1, column_position),
            )
            if all(key in indices for key in keys):
                faces.append([indices[key] for key in keys])
    return {
        "vertices": vertices,
        "faces": faces,
        "step_pixels": step_pixels,
        "source_pixel_size_m": list(_pixel_metrics(transform)[:2]),
        "sample_spacing_m": [
            _pixel_metrics(transform)[0] * step_pixels,
            _pixel_metrics(transform)[1] * step_pixels,
        ],
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "elevation_semantics": "unaltered_mnt_samples_no_vertical_smoothing",
    }


def _exact_bounded_axis(
    minimum: float,
    maximum: float,
    spacing: float,
    *,
    descending: bool = False,
) -> np.ndarray:
    """Return a regular axis whose first and last values are exact bounds."""

    span = maximum - minimum
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("terrain bounds must have a strictly positive span")
    count = int(math.floor(span / spacing + 1e-9))
    axis = minimum + np.arange(count + 1, dtype="float64") * spacing
    if axis[-1] < maximum - 1e-9:
        axis = np.append(axis, maximum)
    else:
        axis[-1] = maximum
    if descending:
        axis = axis[::-1].copy()
    return axis


def _build_bounded_local_terrain_mesh(
    values: np.ndarray,
    transform: Any,
    origin: Sequence[float],
    active_mask: np.ndarray,
    bounds: Sequence[float],
    *,
    step_pixels: int,
    coordinate_precision: int,
) -> dict[str, Any]:
    """Sample one exact-boundary terrain grid from a haloed MNT raster."""

    if len(bounds) != 4 or not all(math.isfinite(float(value)) for value in bounds):
        raise ValueError("bounds must contain four finite Lambert-93 coordinates")
    min_x, min_y, max_x, max_y = (float(value) for value in bounds)
    pixel_x_m, pixel_y_m, _ = _pixel_metrics(transform)
    if abs(float(transform.b)) > 1e-12 or abs(float(transform.d)) > 1e-12:
        raise ValueError("exact-boundary terrain requires a north-up raster")
    if float(transform.a) <= 0.0 or float(transform.e) >= 0.0:
        raise ValueError("exact-boundary terrain requires east-right, north-up pixels")

    spacing_x = pixel_x_m * step_pixels
    spacing_y = pixel_y_m * step_pixels
    world_x = _exact_bounded_axis(min_x, max_x, spacing_x)
    world_y = _exact_bounded_axis(min_y, max_y, spacing_y, descending=True)

    # Raster coordinates describe pixel corners. Subtracting one half moves
    # them into the pixel-centre index space required for bilinear sampling.
    column_coordinates = (world_x - float(transform.c)) / float(transform.a) - 0.5
    row_coordinates = (world_y - float(transform.f)) / float(transform.e) - 0.5
    column_zero = np.floor(column_coordinates).astype("int64")
    row_zero = np.floor(row_coordinates).astype("int64")
    column_fraction = column_coordinates - column_zero
    row_fraction = row_coordinates - row_zero
    column_one = column_zero + 1
    row_one = row_zero + 1
    # A point exactly on a source pixel centre needs no sample on its right or
    # lower side. Reusing the same cell there permits exact-boundary sampling
    # at the finite edge of a source while retaining true bilinear sampling
    # for every fractional coordinate.
    column_one[np.isclose(column_fraction, 0.0, atol=1e-12)] = column_zero[
        np.isclose(column_fraction, 0.0, atol=1e-12)
    ]
    row_one[np.isclose(row_fraction, 0.0, atol=1e-12)] = row_zero[
        np.isclose(row_fraction, 0.0, atol=1e-12)
    ]
    if (
        column_zero.min() < 0
        or row_zero.min() < 0
        or column_one.max() >= values.shape[1]
        or row_one.max() >= values.shape[0]
    ):
        raise ValueError(
            "exact-boundary terrain sampling requires a processing-raster halo"
        )

    top_left = values[np.ix_(row_zero, column_zero)]
    top_right = values[np.ix_(row_zero, column_one)]
    bottom_left = values[np.ix_(row_one, column_zero)]
    bottom_right = values[np.ix_(row_one, column_one)]
    horizontal_top = (
        top_left * (1.0 - column_fraction)[None, :]
        + top_right * column_fraction[None, :]
    )
    horizontal_bottom = (
        bottom_left * (1.0 - column_fraction)[None, :]
        + bottom_right * column_fraction[None, :]
    )
    sampled = (
        horizontal_top * (1.0 - row_fraction)[:, None]
        + horizontal_bottom * row_fraction[:, None]
    )

    nearest_columns = np.floor(column_coordinates + 0.5).astype("int64")
    nearest_rows = np.floor(row_coordinates + 0.5).astype("int64")
    sampled_active = active_mask[np.ix_(nearest_rows, nearest_columns)]
    sampled_active &= np.isfinite(sampled)

    grid_x, grid_y = np.meshgrid(world_x, world_y)
    flattened_active = sampled_active.ravel()
    flattened_indices = np.full(flattened_active.shape, -1, dtype="int64")
    flattened_indices[flattened_active] = np.arange(
        int(np.count_nonzero(flattened_active)), dtype="int64"
    )
    indices = flattened_indices.reshape(sampled_active.shape)
    origin_x, origin_y, origin_z = (float(value) for value in origin)
    vertices = [
        [
            round(float(x) - origin_x, coordinate_precision),
            round(float(y) - origin_y, coordinate_precision),
            round(float(elevation) - origin_z, coordinate_precision),
        ]
        for x, y, elevation in zip(
            grid_x.ravel()[flattened_active],
            grid_y.ravel()[flattened_active],
            sampled.ravel()[flattened_active],
            strict=True,
        )
    ]
    faces: list[list[int]] = []
    for row in range(indices.shape[0] - 1):
        for column in range(indices.shape[1] - 1):
            face = (
                int(indices[row, column]),
                int(indices[row, column + 1]),
                int(indices[row + 1, column + 1]),
                int(indices[row + 1, column]),
            )
            if min(face) >= 0:
                faces.append(list(face))
    return {
        "vertices": vertices,
        "faces": faces,
        "step_pixels": step_pixels,
        "source_pixel_size_m": [pixel_x_m, pixel_y_m],
        "sample_spacing_m": [spacing_x, spacing_y],
        "geometric_bounds_l93_m": [min_x, min_y, max_x, max_y],
        "boundary_sampling": (
            "bilinear_processing_halo_at_exact_lambert93_core_coordinates"
        ),
        "adjacent_edge_contract": "coincident_xy_and_identical_sample_coordinates",
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "elevation_semantics": "bilinear_mnt_samples_no_vertical_smoothing",
    }


def _tree_records(
    instances: Sequence[Sequence[float | int]], origin: Sequence[float]
) -> list[dict[str, Any]]:
    origin_x, origin_y, origin_z = (float(value) for value in origin)
    return [
        {
            "source_id": f"crown-{index:08d}",
            "x_m": origin_x + float(instance[0]),
            "y_m": origin_y + float(instance[1]),
            "ground_elevation_m": origin_z + float(instance[2]),
            "height_m": float(instance[3]),
            "crown_diameter_m": float(instance[4]),
        }
        for index, instance in enumerate(instances)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_geometry(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = (
        payload.get("features", [])
        if payload.get("type") == "FeatureCollection"
        else [payload]
    )
    geometries: list[Any] = []
    for feature in features:
        geometry = (
            feature.get("geometry") if feature.get("type") == "Feature" else feature
        )
        if geometry:
            parsed = shape(geometry)
            if not parsed.is_empty:
                geometries.append(parsed)
    return geometries


def _is_lambert93(crs: Any | None) -> bool:
    """Accept canonical EPSG:2154 or IGN WMS-R's equivalent incomplete WKT."""

    if crs is None:
        return False
    if crs.to_epsg() == 2154:
        return True
    parameters = crs.to_dict()
    expected = {
        "proj": "lcc",
        "lat_0": 46.5,
        "lon_0": 3.0,
        "lat_1": 49.0,
        "lat_2": 44.0,
        "x_0": 700000.0,
        "y_0": 6600000.0,
        "units": "m",
    }
    return all(
        parameters.get(key) == value
        if isinstance(value, str)
        else parameters.get(key) is not None
        and math.isclose(float(parameters[key]), value, abs_tol=1e-9)
        for key, value in expected.items()
    )


def _mosaic(
    paths: Sequence[Path], bounds: Sequence[float]
) -> tuple[np.ndarray, Any, list[dict[str, Any]]]:
    import rasterio

    datasets = [rasterio.open(path) for path in paths]
    try:
        for path, dataset in zip(paths, datasets, strict=True):
            if dataset.count != 1:
                raise ValueError(f"Terrain source must have one band: {path}")
            if not _is_lambert93(dataset.crs):
                raise ValueError(f"Terrain source must use EPSG:2154: {path}")
            transform = dataset.transform
            if (
                abs(float(transform.b)) > 1e-12
                or abs(float(transform.d)) > 1e-12
                or not math.isclose(float(transform.a), 0.5, abs_tol=1e-9)
                or not math.isclose(float(transform.e), -0.5, abs_tol=1e-9)
            ):
                raise ValueError(
                    f"Terrain source must use the north-up 0.5 m grid: {path}"
                )
        aligned_bounds = _bounds_aligned_to_native_grid(bounds, datasets[0].transform)
        reference_transform = datasets[0].transform
        for dataset in datasets[1:]:
            transform = dataset.transform
            if (
                not math.isclose(
                    float(transform.a), float(reference_transform.a), abs_tol=1e-9
                )
                or not math.isclose(
                    float(transform.e), float(reference_transform.e), abs_tol=1e-9
                )
                or not math.isclose(
                    (float(transform.c) - float(reference_transform.c))
                    / float(reference_transform.a),
                    round(
                        (float(transform.c) - float(reference_transform.c))
                        / float(reference_transform.a)
                    ),
                    abs_tol=1e-6,
                )
                or not math.isclose(
                    (float(transform.f) - float(reference_transform.f))
                    / float(reference_transform.e),
                    round(
                        (float(transform.f) - float(reference_transform.f))
                        / float(reference_transform.e)
                    ),
                    abs_tol=1e-6,
                )
            ):
                raise ValueError("MNT/MNS source rasters do not share one pixel grid")
        values, transform = merge_rasters(
            datasets,
            # IGN HD rasters have native pixel edges on a quarter-metre phase
            # (for example *.75), while the 500 m tile cores use integer
            # coordinates. Aligning every processing mosaic to the native
            # phase prevents rasterio window rounding from shifting adjacent
            # mosaics by one 0.5 m sample.
            bounds=aligned_bounds,
            res=(0.5, 0.5),
            nodata=np.nan,
            dtype="float32",
        )
    finally:
        for dataset in datasets:
            dataset.close()
    sources = [
        {
            "file_name": path.name,
            "byte_count": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    return values[0].astype("float64", copy=False), transform, sources


def _bounds_aligned_to_native_grid(
    bounds: Sequence[float], transform: Any
) -> tuple[float, float, float, float]:
    """Expand bounds to pixel edges on the native north-up raster phase."""

    if len(bounds) != 4 or not all(math.isfinite(float(value)) for value in bounds):
        raise ValueError("bounds must contain four finite coordinates")
    if abs(float(transform.b)) > 1e-12 or abs(float(transform.d)) > 1e-12:
        raise ValueError("native-grid alignment requires a north-up raster")
    pixel_x = float(transform.a)
    pixel_y = -float(transform.e)
    if pixel_x <= 0.0 or pixel_y <= 0.0:
        raise ValueError("native-grid alignment requires positive pixel sizes")
    min_x, min_y, max_x, max_y = (float(value) for value in bounds)
    if max_x <= min_x or max_y <= min_y:
        raise ValueError("bounds must have strictly positive spans")

    column_min = math.floor((min_x - float(transform.c)) / pixel_x)
    column_max = math.ceil((max_x - float(transform.c)) / pixel_x)
    row_min = math.floor((float(transform.f) - max_y) / pixel_y)
    row_max = math.ceil((float(transform.f) - min_y) / pixel_y)
    return (
        float(transform.c) + column_min * pixel_x,
        float(transform.f) - row_max * pixel_y,
        float(transform.c) + column_max * pixel_x,
        float(transform.f) - row_min * pixel_y,
    )


def _crop_grid_to_bounds(
    values: np.ndarray, transform: Any, bounds: Sequence[float]
) -> tuple[np.ndarray, Any]:
    from rasterio.windows import from_bounds, transform as window_transform

    window = from_bounds(*bounds, transform=transform).round_offsets().round_lengths()
    row_start = int(window.row_off)
    column_start = int(window.col_off)
    row_stop = row_start + int(window.height)
    column_stop = column_start + int(window.width)
    if (
        row_start < 0
        or column_start < 0
        or row_stop > values.shape[0]
        or column_stop > values.shape[1]
    ):
        raise ValueError("Core bounds fall outside the processing raster")
    cropped = values[row_start:row_stop, column_start:column_stop]
    return cropped, window_transform(window, transform)


def _owned_instances(
    instances: Sequence[Sequence[float | int]],
    origin: Sequence[float],
    core_bounds: Sequence[float],
) -> list[list[float | int]]:
    min_x, min_y, max_x, max_y = (float(value) for value in core_bounds)
    origin_x, origin_y = float(origin[0]), float(origin[1])
    return [
        list(instance)
        for instance in instances
        if min_x <= origin_x + float(instance[0]) < max_x
        and min_y <= origin_y + float(instance[1]) < max_y
    ]


def _owned_statistics(
    statistics: dict[str, Any],
    processing_instances: Sequence[Sequence[float | int]],
    instances: Sequence[Sequence[float | int]],
) -> dict[str, Any]:
    result = dict(statistics)
    heights = [float(instance[3]) for instance in instances]
    crown_areas = [math.pi * (float(instance[4]) / 2.0) ** 2 for instance in instances]
    result.update(
        {
            "processing_accepted_instance_count": len(processing_instances),
            "discarded_halo_instance_count": len(processing_instances) - len(instances),
            "accepted_instance_count": len(instances),
            "minimum_height_m": min(heights) if heights else None,
            "maximum_height_m": max(heights) if heights else None,
            "minimum_crown_area_m2": min(crown_areas) if crown_areas else None,
            "maximum_crown_area_m2": max(crown_areas) if crown_areas else None,
            "ownership_rule": "apex_in_half_open_core_min_inclusive_max_exclusive",
        }
    )
    return result


def build_sector_package(args: argparse.Namespace) -> dict[str, Any]:
    core_bounds = tuple(float(value) for value in args.bounds)
    processing_bounds = tuple(
        float(value)
        for value in (getattr(args, "processing_bounds", None) or core_bounds)
    )
    core = box(*core_bounds)
    sector = box(*processing_bounds)
    if not sector.covers(core):
        raise ValueError("processing_bounds must cover the complete core bounds")
    mnt, transform, mnt_sources = _mosaic(args.mnt, processing_bounds)
    mns, mns_transform, mns_sources = _mosaic(args.mns, processing_bounds)
    if mnt.shape != mns.shape or tuple(transform) != tuple(mns_transform):
        raise ValueError("MNT and MNS mosaics are not pixel-aligned")
    include_paths = list(getattr(args, "include_polygons", []))
    include_geometries = [
        geometry for path in include_paths for geometry in _load_geometry(path)
    ]
    valid_geometry = (
        sector.intersection(unary_union(include_geometries))
        if include_geometries
        else sector
    )
    if valid_geometry.is_empty:
        raise ValueError(
            "The processing sector does not intersect the include geometry"
        )
    valid_mask = geometry_mask(
        [mapping(valid_geometry)],
        out_shape=mnt.shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )
    polygon_exclusions = [
        geometry for path in args.exclude_polygons for geometry in _load_geometry(path)
    ]
    line_exclusions = [
        geometry for path in args.exclude_lines for geometry in _load_geometry(path)
    ]
    buffered = [
        *(
            geometry.buffer(args.polygon_clearance_m, join_style=2)
            for geometry in polygon_exclusions
        ),
        *(
            geometry.buffer(args.line_half_width_m, cap_style=2, join_style=2)
            for geometry in line_exclusions
        ),
    ]
    exclusion_geometry = unary_union(buffered) if buffered else None
    exclusion_mask = (
        geometry_mask(
            [mapping(exclusion_geometry)],
            out_shape=mnt.shape,
            transform=transform,
            invert=True,
            all_touched=True,
        )
        if exclusion_geometry is not None and not exclusion_geometry.is_empty
        else np.zeros(mnt.shape, dtype=bool)
    )
    finite_ground = mnt[valid_mask & np.isfinite(mnt)]
    if not finite_ground.size:
        raise ValueError("No finite MNT sample exists in the requested sector")
    origin = (
        args.origin_x
        if args.origin_x is not None
        else round((core_bounds[0] + core_bounds[2]) / 2),
        args.origin_y
        if args.origin_y is not None
        else round((core_bounds[1] + core_bounds[3]) / 2),
        args.origin_z
        if args.origin_z is not None
        else math.floor(float(finite_ground.min())),
    )
    config = SegmentationConfig(
        min_tree_height_m=args.min_tree_height_m,
        local_peak_radius_m=args.local_peak_radius_m,
        smoothing_sigma_m=args.smoothing_sigma_m,
    )
    processing_instances, statistics = segment_vegetation_instances(
        mnt,
        mns,
        transform,
        valid_mask,
        exclusion_mask,
        origin,
        config,
    )
    instances = _owned_instances(processing_instances, origin, core_bounds)
    statistics = _owned_statistics(statistics, processing_instances, instances)
    tree_instances = build_tree_instance_set(
        _tree_records(instances, origin),
        origin,
        TreeInstanceConfig(profile="global_mid"),
    )
    core_mnt, core_transform = _crop_grid_to_bounds(mnt, transform, core_bounds)
    core_valid_mask, valid_transform = _crop_grid_to_bounds(
        valid_mask, transform, core_bounds
    )
    if tuple(core_transform) != tuple(valid_transform):
        raise ValueError("Core terrain and AOI mask are not aligned")
    local_terrain = build_local_terrain_mesh(
        mnt,
        transform,
        origin,
        valid_mask=valid_mask,
        step_pixels=args.terrain_step_pixels,
        coordinate_precision=config.coordinate_precision,
        bounds=core_bounds,
    )
    return {
        "schema": SCHEMA,
        "metadata": {
            "crs": "EPSG:2154",
            "axis_convention": "X=east, Y=north, Z=up",
            "linear_unit": "metre",
            "bounds_l93_m": list(core_bounds),
            "processing_bounds_l93_m": list(processing_bounds),
            "origin_l93_m": list(origin),
            "raster_transform": list(core_transform),
            "segmentation_raster_transform": list(transform),
            "config": asdict(config),
            "sources": {"mnt": mnt_sources, "mns": mns_sources},
            "exclusion_sources": [
                *(
                    {"role": "polygon", "file_name": path.name, "sha256": _sha256(path)}
                    for path in args.exclude_polygons
                ),
                *(
                    {"role": "line", "file_name": path.name, "sha256": _sha256(path)}
                    for path in args.exclude_lines
                ),
            ],
            "include_sources": [
                {"role": "include", "file_name": path.name, "sha256": _sha256(path)}
                for path in include_paths
            ],
        },
        "instances": {
            "attributes": statistics["attributes"],
            "values": instances,
        },
        "tree_instances": tree_instances,
        "terrain": local_terrain,
        "statistics": statistics,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnt", type=Path, action="append", required=True)
    parser.add_argument("--mns", type=Path, action="append", required=True)
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        required=True,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
    )
    parser.add_argument(
        "--processing-bounds",
        type=float,
        nargs=4,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="Optional core-plus-halo bounds used only for crown segmentation",
    )
    parser.add_argument("--include-polygons", type=Path, action="append", default=[])
    parser.add_argument("--exclude-polygons", type=Path, action="append", default=[])
    parser.add_argument("--exclude-lines", type=Path, action="append", default=[])
    parser.add_argument("--polygon-clearance-m", type=float, default=2.0)
    parser.add_argument("--line-half-width-m", type=float, default=4.0)
    parser.add_argument("--min-tree-height-m", type=float, default=2.0)
    parser.add_argument("--local-peak-radius-m", type=float, default=1.0)
    parser.add_argument("--smoothing-sigma-m", type=float, default=0.5)
    parser.add_argument("--terrain-step-pixels", type=int, default=2)
    parser.add_argument("--origin-x", type=float)
    parser.add_argument("--origin-y", type=float)
    parser.add_argument("--origin-z", type=float)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def write_sector_package(package: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        package, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    temporary.replace(output)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    package = build_sector_package(args)
    write_sector_package(package, args.output)
    print(json.dumps(package["statistics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
