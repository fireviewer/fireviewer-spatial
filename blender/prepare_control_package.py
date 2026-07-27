"""Prepare a deterministic, local-coordinate Blender preview package.

This script is executed with the project/system Python, where rasterio,
Shapely and pyproj are available. The resulting gzip JSON is deliberately
readable by Blender without any third-party Python package.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from spatial_data import (  # noqa: E402
    TARGET_CRS,
    LineFeature,
    PolygonFeature,
    choose_local_origin,
    find_absolute_local_paths,
    iter_polygon_parts,
    lambert93_validation,
    numeric_property,
    positive_numeric_property,
    read_line_features,
    read_polygon_features,
    sha256_file,
)
from mid_distance_roads import (  # noqa: E402
    MidDistanceRoadConfig,
    build_mid_distance_road_geometry,
)
from detail_vector_lod import (  # noqa: E402
    TerrainMeshSampler,
    drape_building_prisms_to_tile,
    drape_ribbon_mesh_to_tile,
    drape_triangle_mesh_to_tile,
)
from vegetation_lod import (  # noqa: E402
    VegetationLodConfig,
    generate_mid_distance_vegetation_lod,
)


PACKAGE_SCHEMA = "fireviewer.blender-preview-package.v2"
NON_INCIDENT_PERIMETER_ROLE = "cems-activation-aoi-not-burn-perimeter"


def should_render_fire_perimeter(
    features: Iterable[PolygonFeature], *, explicitly_omitted: bool
) -> bool:
    if explicitly_omitted:
        return False
    roles = {str(feature.properties.get("role", "")) for feature in features}
    return roles != {NON_INCIDENT_PERIMETER_ROLE}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mnt", type=Path, required=True, help="MNT GeoTIFF in EPSG:2154"
    )
    parser.add_argument(
        "--mns",
        type=Path,
        help="MNS GeoTIFF aligned pixel-for-pixel with the MNT; required with --vegetation",
    )
    parser.add_argument(
        "--perimeter",
        type=Path,
        required=True,
        help="Production envelope GeoJSON, optionally rendered as a fire perimeter",
    )
    parser.add_argument(
        "--omit-fire-perimeter",
        action="store_true",
        help="Use the perimeter for clipping only and emit no incident boundary ring",
    )
    parser.add_argument(
        "--buildings", type=Path, help="Optional building polygon GeoJSON"
    )
    parser.add_argument(
        "--vegetation", type=Path, help="Optional vegetation polygon GeoJSON"
    )
    parser.add_argument(
        "--roads",
        type=Path,
        action="append",
        default=[],
        help="Optional road LineString GeoJSON; repeat for paginated local sources",
    )
    parser.add_argument(
        "--water-courses",
        type=Path,
        action="append",
        default=[],
        help="Optional named water-course LineString/MultiLineString GeoJSON; repeatable",
    )
    parser.add_argument(
        "--water-segments",
        type=Path,
        action="append",
        default=[],
        help="Optional hydrographic segment LineString GeoJSON; repeatable",
    )
    parser.add_argument(
        "--water-surfaces",
        type=Path,
        action="append",
        default=[],
        help="Optional hydrographic Polygon/MultiPolygon GeoJSON; repeatable",
    )
    parser.add_argument("--perimeter-crs", default="auto")
    parser.add_argument(
        "--perimeter-feature-id",
        help="Keep exactly one perimeter feature id/property id (for example EFFIS SYNTH-001)",
    )
    parser.add_argument(
        "--perimeter-swap-xy",
        action="store_true",
        help="Explicitly normalize a non-standard [latitude, longitude] source",
    )
    parser.add_argument("--buildings-crs", default="auto")
    parser.add_argument("--vegetation-crs", default="auto")
    parser.add_argument("--roads-crs", default="auto")
    parser.add_argument("--water-crs", default="auto")
    parser.add_argument("--road-offset-m", type=float, default=0.35)
    parser.add_argument("--water-course-offset-m", type=float, default=0.25)
    parser.add_argument("--water-segment-offset-m", type=float, default=0.20)
    parser.add_argument("--water-surface-offset-m", type=float, default=0.15)
    parser.add_argument("--vegetation-building-clearance-m", type=float, default=2.0)
    parser.add_argument("--vegetation-road-clearance-m", type=float, default=1.0)
    parser.add_argument("--vegetation-water-clearance-m", type=float, default=1.0)
    parser.add_argument("--mid-tree-min-height-m", type=float, default=3.0)
    parser.add_argument("--mid-tree-spacing-m", type=float, default=15.0)
    parser.add_argument("--mid-tree-local-max-radius-m", type=float, default=7.5)
    parser.add_argument("--mid-tree-max-count", type=int, default=200000)
    parser.add_argument("--buffer-m", type=float, default=1500.0)
    parser.add_argument(
        "--terrain-step",
        type=int,
        default=4,
        help="Keep one sample every N pixels; global delivery keeps the validated step 4",
    )
    parser.add_argument("--origin-x", type=float)
    parser.add_argument("--origin-y", type=float)
    parser.add_argument("--origin-z", type=float)
    parser.add_argument(
        "--default-building-height-m",
        type=float,
        help="Opt-in fallback; without it, buildings lacking a positive height are not extruded",
    )
    parser.add_argument(
        "--building-simplify-m",
        type=float,
        default=0.05,
        help="Building footprint simplification; global delivery keeps the validated 0.05 m",
    )
    parser.add_argument(
        "--minimum-visible-building-wall-m",
        type=float,
        default=2.70,
        help=(
            "Minimum roof clearance above the highest MNT sample in a footprint; "
            "prevents sloped terrain from leaving only a roof sliver visible"
        ),
    )
    parser.add_argument("--allow-partial-terrain", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, help="Destination .json.gz package")
    return parser


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    args = _parser().parse_args(list(argv)[1:])
    if args.buffer_m < 0:
        raise ValueError("--buffer-m must be non-negative")
    if args.terrain_step < 1:
        raise ValueError("--terrain-step must be at least 1")
    for option, value in (
        ("--road-offset-m", args.road_offset_m),
        ("--water-course-offset-m", args.water_course_offset_m),
        ("--water-segment-offset-m", args.water_segment_offset_m),
        ("--water-surface-offset-m", args.water_surface_offset_m),
        ("--vegetation-building-clearance-m", args.vegetation_building_clearance_m),
        ("--vegetation-road-clearance-m", args.vegetation_road_clearance_m),
        ("--vegetation-water-clearance-m", args.vegetation_water_clearance_m),
    ):
        if value < 0:
            raise ValueError(f"{option} must be non-negative")
    for option, value in (
        ("--default-building-height-m", args.default_building_height_m),
        (
            "--minimum-visible-building-wall-m",
            args.minimum_visible_building_wall_m,
        ),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{option} must be strictly positive when provided")
    for option, value in (
        ("--mid-tree-min-height-m", args.mid_tree_min_height_m),
        ("--mid-tree-spacing-m", args.mid_tree_spacing_m),
        ("--mid-tree-local-max-radius-m", args.mid_tree_local_max_radius_m),
    ):
        if value <= 0:
            raise ValueError(f"{option} must be strictly positive")
    if args.mid_tree_max_count <= 0:
        raise ValueError("--mid-tree-max-count must be strictly positive")
    if not args.validate_only and args.output is None:
        raise ValueError("--output is required unless --validate-only is used")
    return args


def _clip_features(
    features: Iterable[PolygonFeature], clip_geometry: Any
) -> list[PolygonFeature]:
    clipped: list[PolygonFeature] = []
    for feature in features:
        intersection = feature.geometry.intersection(clip_geometry)
        for index, polygon in enumerate(iter_polygon_parts(intersection)):
            if polygon.area > 0:
                clipped.append(
                    PolygonFeature(
                        feature_id=f"{feature.feature_id}:clip{index}",
                        geometry=polygon,
                        properties=feature.properties,
                    )
                )
    clipped.sort(
        key=lambda item: (item.feature_id, item.geometry.bounds, item.geometry.wkb_hex)
    )
    return clipped


def _clip_line_features(
    features: Iterable[LineFeature], clip_geometry: Any
) -> list[LineFeature]:
    clipped: list[LineFeature] = []
    from spatial_data import iter_line_parts

    for feature in features:
        intersection = feature.geometry.intersection(clip_geometry)
        for index, line in enumerate(iter_line_parts(intersection)):
            if line.length > 0 and len(line.coords) >= 2:
                clipped.append(
                    LineFeature(
                        feature_id=f"{feature.feature_id}:clip{index}",
                        geometry=line,
                        properties=feature.properties,
                    )
                )
    clipped.sort(
        key=lambda item: (item.feature_id, item.geometry.bounds, item.geometry.wkb_hex)
    )
    return clipped


def _rounded(value: float) -> float:
    return round(float(value), 3)


def _sample_elevation(
    terrain: Any, transform: Any, x: float, y: float, fallback: float
) -> float:
    import numpy as np

    column, row = (~transform) * (x, y)
    row_index = min(max(int(math.floor(row)), 0), terrain.shape[0] - 1)
    column_index = min(max(int(math.floor(column)), 0), terrain.shape[1] - 1)
    value = float(terrain[row_index, column_index])
    return value if np.isfinite(value) else fallback


def _assert_aligned_grids(
    mnt_shape: tuple[int, int],
    mnt_transform: Any,
    mns_shape: tuple[int, int],
    mns_transform: Any,
) -> None:
    """Reject any MNS that is not pixel-aligned with the MNT."""

    if tuple(mnt_shape) != tuple(mns_shape):
        raise ValueError(f"MNS/MNT shape mismatch: MNT={mnt_shape}, MNS={mns_shape}")
    mnt_coefficients = tuple(float(value) for value in mnt_transform)[:6]
    mns_coefficients = tuple(float(value) for value in mns_transform)[:6]
    if any(
        not math.isclose(mnt_value, mns_value, rel_tol=0.0, abs_tol=1e-9)
        for mnt_value, mns_value in zip(mnt_coefficients, mns_coefficients)
    ):
        raise ValueError(
            "MNS/MNT transform mismatch: rasters must have identical origin, pixel size and rotation"
        )


def _terrain_geometry(
    terrain: Any,
    transform: Any,
    valid_buffer_mask: Any,
    origin: tuple[float, float, float],
    step: int,
) -> dict[str, Any]:
    import numpy as np

    height, width = terrain.shape
    rows = list(range(0, height, step))
    columns = list(range(0, width, step))
    if rows[-1] != height - 1:
        rows.append(height - 1)
    if columns[-1] != width - 1:
        columns.append(width - 1)

    origin_x, origin_y, origin_z = origin
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    indices: dict[tuple[int, int], int] = {}

    def vertex(row: int, column: int) -> int:
        key = (row, column)
        existing = indices.get(key)
        if existing is not None:
            return existing
        x, y = transform * (column + 0.5, row + 0.5)
        index = len(vertices)
        vertices.append(
            [
                _rounded(x - origin_x),
                _rounded(y - origin_y),
                _rounded(terrain[row, column] - origin_z),
            ]
        )
        indices[key] = index
        return index

    for row_index in range(len(rows) - 1):
        north, south = rows[row_index], rows[row_index + 1]
        middle_row = (north + south) // 2
        for column_index in range(len(columns) - 1):
            west, east = columns[column_index], columns[column_index + 1]
            middle_column = (west + east) // 2
            if not valid_buffer_mask[middle_row, middle_column]:
                continue
            samples = (
                terrain[north, west],
                terrain[south, west],
                terrain[south, east],
                terrain[north, east],
            )
            if not all(np.isfinite(value) for value in samples):
                continue
            faces.append(
                [
                    vertex(north, west),
                    vertex(south, west),
                    vertex(south, east),
                    vertex(north, east),
                ]
            )
    return {"vertices": vertices, "faces": faces, "step_pixels": step}


def _regular_terrain_sampling_spec(
    terrain: Any,
    transform: Any,
    origin: tuple[float, float, float],
    step: int,
) -> dict[str, Any]:
    """Return the complete regular grid used by the rendered terrain preview.

    ``_terrain_geometry`` clips faces to the incident buffer and consequently
    stores a sparse vertex set. Vector draping needs the same sampled surface
    before clipping, so it can evaluate the explicit NW-SE terrain triangles
    at every road/water/building coordinate while remaining exactly aligned
    with ``TerrainPreview``.
    """

    height, width = terrain.shape
    rows = list(range(0, height, step))
    columns = list(range(0, width, step))
    if rows[-1] != height - 1:
        rows.append(height - 1)
    if columns[-1] != width - 1:
        columns.append(width - 1)
    origin_x, origin_y, origin_z = origin
    vertices: list[list[float]] = []
    for row in rows:
        for column in columns:
            x, y = transform * (column + 0.5, row + 0.5)
            vertices.append(
                [
                    _rounded(x - origin_x),
                    _rounded(y - origin_y),
                    _rounded(float(terrain[row, column]) - origin_z),
                ]
            )
    return {
        "vertices": vertices,
        "faces": [],
        "step_pixels": step,
        "row_count": len(rows),
        "column_count": len(columns),
    }


def _canopy_geometry(
    mnt: Any,
    mns: Any,
    transform: Any,
    vegetation_cell_mask: Any,
    origin: tuple[float, float, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a terrain-draped canopy shell on the native raster grid.

    Top quads follow the MNS. Boundary skirts follow the MNS at their upper
    edge and the MNT at their lower edge, preventing any floating vegetation
    while avoiding a hidden bottom face under every interior cell.
    """

    import numpy as np

    expected_shape = (mnt.shape[0] - 1, mnt.shape[1] - 1)
    if tuple(vegetation_cell_mask.shape) != expected_shape:
        raise ValueError(
            f"Vegetation cell mask shape mismatch: expected {expected_shape}, "
            f"got {vegetation_cell_mask.shape}"
        )

    finite_mnt = np.isfinite(mnt)
    finite_mns = np.isfinite(mns)
    valid_corners = (
        finite_mnt[:-1, :-1]
        & finite_mnt[1:, :-1]
        & finite_mnt[1:, 1:]
        & finite_mnt[:-1, 1:]
        & finite_mns[:-1, :-1]
        & finite_mns[1:, :-1]
        & finite_mns[1:, 1:]
        & finite_mns[:-1, 1:]
    )
    active_cells = np.asarray(vegetation_cell_mask, dtype=bool) & valid_corners

    origin_x, origin_y, origin_z = origin
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    top_indices = np.full(mnt.shape, -1, dtype=np.int32)
    bottom_indices = np.full(mnt.shape, -1, dtype=np.int32)
    top_vertex_count = 0
    bottom_vertex_count = 0
    clamped_top_vertices = 0
    zero_height_top_vertices = 0
    minimum_canopy_height = math.inf
    maximum_canopy_height = -math.inf

    def coordinates(row: int, column: int) -> tuple[float, float]:
        x, y = transform * (column + 0.5, row + 0.5)
        return float(x), float(y)

    def top_vertex(row: int, column: int) -> int:
        nonlocal clamped_top_vertices, zero_height_top_vertices, top_vertex_count
        nonlocal minimum_canopy_height, maximum_canopy_height
        existing = int(top_indices[row, column])
        if existing >= 0:
            return existing
        ground = float(mnt[row, column])
        raw_surface = float(mns[row, column])
        surface = max(raw_surface, ground)
        if raw_surface < ground:
            clamped_top_vertices += 1
        height = surface - ground
        if height <= 0:
            zero_height_top_vertices += 1
        minimum_canopy_height = min(minimum_canopy_height, height)
        maximum_canopy_height = max(maximum_canopy_height, height)
        x, y = coordinates(row, column)
        index = len(vertices)
        vertices.append(
            [
                _rounded(x - origin_x),
                _rounded(y - origin_y),
                _rounded(surface - origin_z),
            ]
        )
        top_indices[row, column] = index
        top_vertex_count += 1
        return index

    def bottom_vertex(row: int, column: int) -> int:
        nonlocal bottom_vertex_count
        existing = int(bottom_indices[row, column])
        if existing >= 0:
            return existing
        x, y = coordinates(row, column)
        index = len(vertices)
        vertices.append(
            [
                _rounded(x - origin_x),
                _rounded(y - origin_y),
                _rounded(float(mnt[row, column]) - origin_z),
            ]
        )
        bottom_indices[row, column] = index
        bottom_vertex_count += 1
        return index

    top_face_count = 0
    boundary_face_count = 0
    active_rows, active_columns = np.nonzero(active_cells)
    for row_value, column_value in zip(active_rows, active_columns):
        row, column = int(row_value), int(column_value)
        north_west = (row, column)
        south_west = (row + 1, column)
        south_east = (row + 1, column + 1)
        north_east = (row, column + 1)
        faces.append(
            [
                top_vertex(*north_west),
                top_vertex(*south_west),
                top_vertex(*south_east),
                top_vertex(*north_east),
            ]
        )
        top_face_count += 1

        boundary_edges = (
            (row, column - 1, north_west, south_west),
            (row + 1, column, south_west, south_east),
            (row, column + 1, south_east, north_east),
            (row - 1, column, north_east, north_west),
        )
        for neighbour_row, neighbour_column, start, end in boundary_edges:
            neighbour_active = (
                0 <= neighbour_row < active_cells.shape[0]
                and 0 <= neighbour_column < active_cells.shape[1]
                and bool(active_cells[neighbour_row, neighbour_column])
            )
            if neighbour_active:
                continue
            faces.append(
                [
                    top_vertex(*start),
                    bottom_vertex(*start),
                    bottom_vertex(*end),
                    top_vertex(*end),
                ]
            )
            boundary_face_count += 1

    statistics = {
        "source_polygon_cell_count": int(np.count_nonzero(vegetation_cell_mask)),
        "valid_canopy_cell_count": int(np.count_nonzero(active_cells)),
        "invalid_or_nodata_cell_count": int(
            np.count_nonzero(vegetation_cell_mask) - np.count_nonzero(active_cells)
        ),
        "top_vertex_count": top_vertex_count,
        "grounded_boundary_vertex_count": bottom_vertex_count,
        "top_face_count": top_face_count,
        "boundary_skirt_face_count": boundary_face_count,
        "clamped_mns_below_mnt_vertex_count": clamped_top_vertices,
        "zero_height_top_vertex_count": zero_height_top_vertices,
        "minimum_canopy_height_m": minimum_canopy_height if top_vertex_count else None,
        "maximum_canopy_height_m": maximum_canopy_height if top_vertex_count else None,
    }
    return {"vertices": vertices, "faces": faces, "grid_step_pixels": 1}, statistics


def _road_width_m(properties: dict[str, Any] | Any) -> tuple[float, str]:
    source_width = positive_numeric_property(
        properties,
        ("largeur_de_chaussee", "largeur_chaussee", "largeur", "width"),
    )
    if source_width is not None:
        return min(max(source_width, 1.5), 30.0), "source_width"
    importance = numeric_property(
        properties, ("importance", "classement", "road_class")
    )
    importance_widths = {1: 10.0, 2: 8.0, 3: 7.0, 4: 6.0, 5: 5.0, 6: 2.5, 7: 1.5}
    if importance is not None:
        rank = int(round(importance))
        if rank in importance_widths:
            return importance_widths[rank], "importance"
    return 4.0, "fallback_unclassified"


def _water_course_width_m(properties: dict[str, Any] | Any) -> tuple[float, str]:
    importance = numeric_property(properties, ("importance",))
    widths = {1: 14.0, 2: 11.0, 3: 8.0, 4: 6.0, 5: 4.0}
    if importance is not None and int(round(importance)) in widths:
        return widths[int(round(importance))], "water_course_importance"
    return 3.0, "water_course_unclassified"


def _water_segment_width_m(properties: dict[str, Any] | Any) -> tuple[float, str]:
    value = str(properties.get("classe_de_largeur", "")).casefold()
    if "plus de 50" in value:
        return 30.0, "hydro_width_class"
    if "15" in value and "50" in value:
        return 20.0, "hydro_width_class"
    if "5" in value and "15" in value:
        return 10.0, "hydro_width_class"
    if "0" in value and "5" in value:
        return 3.0, "hydro_width_class"
    return 2.0, "hydro_width_unclassified"


def _vegetation_exclusion_mask(
    buildings: Sequence[PolygonFeature],
    roads: Sequence[LineFeature],
    water_courses: Sequence[LineFeature],
    water_segments: Sequence[LineFeature],
    water_surfaces: Sequence[PolygonFeature],
    out_shape: tuple[int, int],
    cell_transform: Any,
    building_clearance_m: float,
    road_clearance_m: float,
    water_clearance_m: float,
) -> tuple[Any, dict[str, int]]:
    """Rasterize obstacles that must not be interpreted as canopy."""
    import numpy as np
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping

    def polygon_clearance(
        features: Sequence[PolygonFeature], clearance_m: float
    ) -> list[Any]:
        geometries: list[Any] = []
        for feature in features:
            geometry = (
                feature.geometry.buffer(clearance_m, join_style=2)
                if clearance_m
                else feature.geometry
            )
            if not geometry.is_empty:
                geometries.append(geometry)
        return geometries

    def line_clearance(
        features: Sequence[LineFeature],
        width_resolver: Any,
        clearance_m: float,
    ) -> list[Any]:
        geometries: list[Any] = []
        for feature in features:
            width_m, _ = width_resolver(feature.properties)
            geometry = feature.geometry.buffer(
                width_m / 2.0 + clearance_m, cap_style=2, join_style=2
            )
            if not geometry.is_empty:
                geometries.append(geometry)
        return geometries

    building_geometries = polygon_clearance(buildings, building_clearance_m)
    road_geometries = line_clearance(roads, _road_width_m, road_clearance_m)
    water_geometries = [
        *line_clearance(water_courses, _water_course_width_m, water_clearance_m),
        *line_clearance(water_segments, _water_segment_width_m, water_clearance_m),
        *polygon_clearance(water_surfaces, water_clearance_m),
    ]

    def rasterized(geometries: Sequence[Any]) -> Any:
        if not geometries:
            return np.zeros(out_shape, dtype=bool)
        return geometry_mask(
            [mapping(geometry) for geometry in geometries],
            out_shape=out_shape,
            transform=cell_transform,
            invert=True,
            all_touched=True,
        )

    building_mask = rasterized(building_geometries)
    road_mask = rasterized(road_geometries)
    water_mask = rasterized(water_geometries)
    combined = building_mask | road_mask | water_mask
    return combined, {
        "building_exclusion_cell_count": int(np.count_nonzero(building_mask)),
        "road_exclusion_cell_count": int(np.count_nonzero(road_mask)),
        "water_exclusion_cell_count": int(np.count_nonzero(water_mask)),
        "combined_exclusion_cell_count": int(np.count_nonzero(combined)),
    }


def _road_geometry(
    features: Sequence[LineFeature],
    terrain: Any,
    transform: Any,
    origin: tuple[float, float, float],
    offset_m: float,
    width_resolver: Any = _road_width_m,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build terrain-draped road ribbons with source-derived widths."""

    origin_x, origin_y, origin_z = origin
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    widths: list[float] = []
    width_methods: dict[str, int] = {}
    entity_ids: set[str] = set()
    skipped_degenerate = 0

    for feature in features:
        coordinates = [
            (float(value[0]), float(value[1])) for value in feature.geometry.coords
        ]
        deduplicated: list[tuple[float, float]] = []
        for coordinate in coordinates:
            if not deduplicated or coordinate != deduplicated[-1]:
                deduplicated.append(coordinate)
        if len(deduplicated) < 2:
            skipped_degenerate += 1
            continue
        width, width_method = width_resolver(feature.properties)
        widths.append(width)
        width_methods[width_method] = width_methods.get(width_method, 0) + 1
        entity_ids.add(feature.feature_id.split(":", 1)[0])
        half_width = width / 2.0
        left_indices: list[int] = []
        right_indices: list[int] = []

        for index, (x, y) in enumerate(deduplicated):
            if index == 0:
                tangent_x = deduplicated[1][0] - x
                tangent_y = deduplicated[1][1] - y
            elif index == len(deduplicated) - 1:
                tangent_x = x - deduplicated[index - 1][0]
                tangent_y = y - deduplicated[index - 1][1]
            else:
                tangent_x = deduplicated[index + 1][0] - deduplicated[index - 1][0]
                tangent_y = deduplicated[index + 1][1] - deduplicated[index - 1][1]
            tangent_length = math.hypot(tangent_x, tangent_y)
            if tangent_length <= 1e-9:
                tangent_x, tangent_y, tangent_length = 1.0, 0.0, 1.0
            perpendicular_x = -tangent_y / tangent_length
            perpendicular_y = tangent_x / tangent_length
            left_x, left_y = (
                x + perpendicular_x * half_width,
                y + perpendicular_y * half_width,
            )
            right_x, right_y = (
                x - perpendicular_x * half_width,
                y - perpendicular_y * half_width,
            )
            left_z = (
                _sample_elevation(terrain, transform, left_x, left_y, origin_z)
                + offset_m
            )
            right_z = (
                _sample_elevation(terrain, transform, right_x, right_y, origin_z)
                + offset_m
            )

            left_indices.append(len(vertices))
            vertices.append(
                [
                    _rounded(left_x - origin_x),
                    _rounded(left_y - origin_y),
                    _rounded(left_z - origin_z),
                ]
            )
            right_indices.append(len(vertices))
            vertices.append(
                [
                    _rounded(right_x - origin_x),
                    _rounded(right_y - origin_y),
                    _rounded(right_z - origin_z),
                ]
            )

        for index in range(len(deduplicated) - 1):
            faces.append(
                [
                    left_indices[index],
                    right_indices[index],
                    right_indices[index + 1],
                    left_indices[index + 1],
                ]
            )

    statistics = {
        "input_line_count": len(features),
        "input_entity_count": len(
            {feature.feature_id.split(":", 1)[0] for feature in features}
        ),
        "rendered_entity_count": len(entity_ids),
        "skipped_degenerate_line_count": skipped_degenerate,
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "minimum_width_m": min(widths) if widths else None,
        "maximum_width_m": max(widths) if widths else None,
        "width_method_counts": dict(sorted(width_methods.items())),
        "ground_offset_m": offset_m,
    }
    return {"vertices": vertices, "faces": faces}, statistics


def _water_surface_geometry(
    features: Sequence[PolygonFeature],
    terrain: Any,
    transform: Any,
    origin: tuple[float, float, float],
    offset_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Triangulate exact hydro polygons and drape every vertex on the MNT."""

    from shapely.ops import triangulate

    origin_x, origin_y, origin_z = origin
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    vertex_indices: dict[tuple[float, float], int] = {}
    rendered_entities: set[str] = set()
    skipped_polygon_count = 0

    def vertex(x: float, y: float) -> int:
        key = (round(float(x), 7), round(float(y), 7))
        existing = vertex_indices.get(key)
        if existing is not None:
            return existing
        z = (
            _sample_elevation(terrain, transform, float(x), float(y), origin_z)
            + offset_m
        )
        index = len(vertices)
        vertices.append(
            [
                _rounded(float(x) - origin_x),
                _rounded(float(y) - origin_y),
                _rounded(z - origin_z),
            ]
        )
        vertex_indices[key] = index
        return index

    for feature in features:
        polygon_triangle_count = 0
        for triangle in triangulate(feature.geometry):
            if triangle.area <= 0:
                continue
            covered_area = triangle.intersection(feature.geometry).area
            if covered_area / triangle.area < 0.999999:
                continue
            coordinates = list(triangle.exterior.coords)[:3]
            indices = [vertex(float(x), float(y)) for x, y, *_ in coordinates]
            a, b, c = (vertices[index] for index in indices)
            signed_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            faces.append(indices if signed_area >= 0 else list(reversed(indices)))
            polygon_triangle_count += 1
        if polygon_triangle_count:
            rendered_entities.add(feature.feature_id.split(":", 1)[0])
        else:
            skipped_polygon_count += 1

    statistics = {
        "input_polygon_count": len(features),
        "input_entity_count": len(
            {feature.feature_id.split(":", 1)[0] for feature in features}
        ),
        "rendered_entity_count": len(rendered_entities),
        "skipped_untriangulated_polygon_count": skipped_polygon_count,
        "vertex_count": len(vertices),
        "triangle_count": len(faces),
        "ground_offset_m": offset_m,
        "altitude_method": "mnt_draped",
    }
    return {"vertices": vertices, "faces": faces}, statistics


def _boundary_rings(
    geometry: Any,
    terrain: Any,
    transform: Any,
    origin: tuple[float, float, float],
    offset_m: float,
) -> list[list[list[float]]]:
    rings: list[list[list[float]]] = []
    origin_x, origin_y, origin_z = origin
    for polygon in iter_polygon_parts(geometry):
        source_rings = [polygon.exterior, *polygon.interiors]
        for source_ring in source_rings:
            coordinates = list(source_ring.coords)
            if coordinates and coordinates[0] == coordinates[-1]:
                coordinates = coordinates[:-1]
            ring: list[list[float]] = []
            for x, y, *_ in coordinates:
                z = _sample_elevation(terrain, transform, x, y, origin_z)
                ring.append(
                    [
                        _rounded(x - origin_x),
                        _rounded(y - origin_y),
                        _rounded(z - origin_z + offset_m),
                    ]
                )
            if len(ring) >= 3:
                rings.append(ring)
    return rings


def _prisms(
    features: Sequence[PolygonFeature],
    terrain: Any,
    transform: Any,
    origin: tuple[float, float, float],
    default_height: float | None,
    simplify_m: float,
    height_keys: Sequence[str],
    altitude_keys: Sequence[str],
    minimum_visible_wall_height_m: float = 2.70,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from shapely.geometry import Point
    from shapely.geometry.polygon import orient

    prisms: list[dict[str, Any]] = []
    input_entity_ids: set[str] = set()
    extruded_entity_ids: set[str] = set()
    skipped_height_ids: set[str] = set()
    skipped_geometry_ids: set[str] = set()
    default_height_entity_ids: set[str] = set()
    raised_roof_prism_count = 0
    source_clearance_shortfall_vertex_count = 0
    maximum_ground_above_roof_m = 0.0
    maximum_roof_raise_m = 0.0
    minimum_visible_wall_height_m = float(minimum_visible_wall_height_m)
    if (
        not math.isfinite(minimum_visible_wall_height_m)
        or minimum_visible_wall_height_m <= 0.0
    ):
        raise ValueError("minimum visible wall height must be finite and positive")
    origin_x, origin_y, origin_z = origin
    for feature in features:
        source_entity_id = feature.feature_id.split(":", 1)[0]
        input_entity_ids.add(source_entity_id)
        source_height = positive_numeric_property(feature.properties, height_keys)
        if source_height is None:
            if default_height is None:
                skipped_height_ids.add(source_entity_id)
                continue
            source_height = default_height
            default_height_entity_ids.add(source_entity_id)
        geometry = (
            feature.geometry.simplify(simplify_m, preserve_topology=True)
            if simplify_m
            else feature.geometry
        )
        feature_prism_count = 0
        for polygon in iter_polygon_parts(geometry):
            if polygon.area <= 0:
                continue
            polygon = orient(polygon, sign=1.0)
            representative = polygon.representative_point()
            altitude = numeric_property(feature.properties, altitude_keys)
            base_z = (
                altitude
                if altitude is not None
                else _sample_elevation(
                    terrain, transform, representative.x, representative.y, origin_z
                )
            )
            source_rings = [polygon.exterior, *polygon.interiors]
            rings: list[list[list[float]]] = []
            ground_z_rings: list[list[float]] = []
            source_roof_z = float(base_z) + float(source_height)
            sampled_foundation_z_values: list[float] = []
            for source_ring in source_rings:
                coordinates = list(source_ring.coords)
                if coordinates and coordinates[0] == coordinates[-1]:
                    coordinates = coordinates[:-1]
                ring = [
                    [_rounded(x - origin_x), _rounded(y - origin_y)]
                    for x, y, *_ in coordinates
                ]
                if len(ring) >= 3:
                    rings.append(ring)
                    ground_z_ring: list[float] = []
                    for x, y, *_ in coordinates:
                        sampled_ground_z = _sample_elevation(
                            terrain, transform, x, y, base_z
                        )
                        sampled_foundation_z_values.append(sampled_ground_z)
                        if sampled_ground_z > (
                            source_roof_z - minimum_visible_wall_height_m
                        ):
                            source_clearance_shortfall_vertex_count += 1
                        maximum_ground_above_roof_m = max(
                            maximum_ground_above_roof_m,
                            sampled_ground_z - source_roof_z,
                        )
                        ground_z_ring.append(_rounded(sampled_ground_z - origin_z))
                    ground_z_rings.append(ground_z_ring)
            if rings:
                # Boundary samples ground every wall vertex. Also inspect MNT
                # pixel centres inside the footprint so a local terrain peak
                # cannot pass through a flat roof between two vertices.
                inverse_transform = ~transform
                polygon_bounds = polygon.bounds
                raster_corners = [
                    inverse_transform * (x, y)
                    for x in (polygon_bounds[0], polygon_bounds[2])
                    for y in (polygon_bounds[1], polygon_bounds[3])
                ]
                column_min = max(
                    0, math.floor(min(point[0] for point in raster_corners)) - 1
                )
                column_max = min(
                    terrain.shape[1] - 1,
                    math.ceil(max(point[0] for point in raster_corners)) + 1,
                )
                row_min = max(
                    0, math.floor(min(point[1] for point in raster_corners)) - 1
                )
                row_max = min(
                    terrain.shape[0] - 1,
                    math.ceil(max(point[1] for point in raster_corners)) + 1,
                )
                for row in range(row_min, row_max + 1):
                    for column in range(column_min, column_max + 1):
                        x, y = transform * (column + 0.5, row + 0.5)
                        if not polygon.covers(Point(x, y)):
                            continue
                        elevation = float(terrain[row, column])
                        if math.isfinite(elevation):
                            sampled_foundation_z_values.append(elevation)
                representative_ground_z = _sample_elevation(
                    terrain,
                    transform,
                    representative.x,
                    representative.y,
                    base_z,
                )
                sampled_foundation_z_values.append(representative_ground_z)
                highest_ground_z = max(sampled_foundation_z_values, default=base_z)
                maximum_ground_above_roof_m = max(
                    maximum_ground_above_roof_m,
                    highest_ground_z - source_roof_z,
                )
                roof_z = max(
                    source_roof_z,
                    highest_ground_z + minimum_visible_wall_height_m,
                )
                roof_raise_m = roof_z - source_roof_z
                if roof_raise_m > 0.0:
                    raised_roof_prism_count += 1
                    maximum_roof_raise_m = max(maximum_roof_raise_m, roof_raise_m)
                prisms.append(
                    {
                        "feature_id": feature.feature_id,
                        "base_z": _rounded(base_z - origin_z),
                        "height": _rounded(source_height),
                        "rings": rings,
                        "ground_z_rings": ground_z_rings,
                        "roof_z": _rounded(roof_z - origin_z),
                        "roof_raise_m": _rounded(roof_raise_m),
                    }
                )
                feature_prism_count += 1
        if feature_prism_count == 0:
            skipped_geometry_ids.add(source_entity_id)
        else:
            extruded_entity_ids.add(source_entity_id)
    sorted_height_ids = sorted(skipped_height_ids)
    sorted_geometry_ids = sorted(skipped_geometry_ids)
    statistics = {
        "input_polygon_count": len(features),
        "input_entity_count": len(input_entity_ids),
        "extruded_prism_count": len(prisms),
        "extruded_entity_count": len(extruded_entity_ids),
        "not_extruded_no_positive_height_count": len(sorted_height_ids),
        "not_extruded_no_positive_height_feature_ids": sorted_height_ids,
        "not_extruded_empty_geometry_count": len(sorted_geometry_ids),
        "not_extruded_empty_geometry_feature_ids": sorted_geometry_ids,
        "default_height_m": default_height,
        "default_height_used_count": len(default_height_entity_ids),
        "draped_foundation_prism_count": len(prisms),
        "roof_reference": (
            "max_source_roof_and_highest_mnt_footprint_plus_minimum_wall"
        ),
        "foundation_grounding": "mnt_per_boundary_vertex_unclamped",
        "minimum_visible_wall_height_m": minimum_visible_wall_height_m,
        "raised_roof_prism_count": raised_roof_prism_count,
        "source_clearance_shortfall_vertex_count": (
            source_clearance_shortfall_vertex_count
        ),
        "maximum_ground_above_roof_m": maximum_ground_above_roof_m,
        "maximum_roof_raise_m": maximum_roof_raise_m,
    }
    return prisms, statistics


def prepare_package(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import rasterio
    from affine import Affine
    from rasterio.features import geometry_mask
    from shapely.geometry import box, mapping
    from shapely.ops import unary_union

    if args.vegetation is not None and args.mns is None:
        raise ValueError("--mns is required when --vegetation is provided")
    input_paths = [
        args.mnt,
        args.mns,
        args.perimeter,
        args.buildings,
        args.vegetation,
        *args.roads,
        *args.water_courses,
        *args.water_segments,
        *args.water_surfaces,
    ]
    for path in input_paths:
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)

    perimeter_features, perimeter_source_crs = read_polygon_features(
        args.perimeter,
        args.perimeter_crs,
        swap_xy=args.perimeter_swap_xy,
    )
    if args.perimeter_feature_id:
        wanted_id = str(args.perimeter_feature_id)
        perimeter_features = [
            feature
            for feature in perimeter_features
            if feature.feature_id.rsplit(":", 1)[0] == wanted_id
        ]
    if not perimeter_features:
        raise ValueError("Production perimeter contains no polygon")
    render_fire_perimeter = should_render_fire_perimeter(
        perimeter_features,
        explicitly_omitted=args.omit_fire_perimeter,
    )
    perimeter = unary_union([item.geometry for item in perimeter_features])
    buffer_geometry = perimeter.buffer(args.buffer_m)

    with rasterio.open(args.mnt) as dataset:
        crs_validation = lambert93_validation(dataset.crs)
        if crs_validation is None:
            raise ValueError(f"MNT must be EPSG:2154, got {dataset.crs}")
        transform = dataset.transform
        if not math.isclose(transform.b, 0.0, abs_tol=1e-12) or not math.isclose(
            transform.d, 0.0, abs_tol=1e-12
        ):
            raise ValueError("Rotated/sheared GeoTIFF transforms are not supported")
        terrain = (
            dataset.read(1, masked=True).filled(np.nan).astype("float64", copy=False)
        )
        raster_bounds = box(*dataset.bounds)
        raster_shape = (dataset.height, dataset.width)
        pixel_size = (abs(float(transform.a)), abs(float(transform.e)))

    surface_model = None
    mns_crs_validation = None
    if args.mns is not None:
        with rasterio.open(args.mns) as dataset:
            mns_crs_validation = lambert93_validation(dataset.crs)
            if mns_crs_validation is None:
                raise ValueError(f"MNS must be EPSG:2154, got {dataset.crs}")
            mns_transform = dataset.transform
            mns_shape = (dataset.height, dataset.width)
            _assert_aligned_grids(raster_shape, transform, mns_shape, mns_transform)
            surface_model = (
                dataset.read(1, masked=True)
                .filled(np.nan)
                .astype("float64", copy=False)
            )

    coverage = buffer_geometry.intersection(raster_bounds).area / buffer_geometry.area
    if coverage < 0.999 and not args.allow_partial_terrain:
        raise ValueError(
            f"MNT covers only {coverage:.3%} of the requested buffer; "
            "use a complete raster or explicitly pass --allow-partial-terrain"
        )
    valid_buffer_mask = geometry_mask(
        [mapping(buffer_geometry)],
        out_shape=terrain.shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )
    valid_values = terrain[valid_buffer_mask & np.isfinite(terrain)]
    if valid_values.size == 0:
        raise ValueError("MNT has no finite elevation inside the requested buffer")

    automatic_origin = choose_local_origin(
        buffer_geometry.bounds, float(valid_values.min())
    )
    origin = (
        args.origin_x if args.origin_x is not None else automatic_origin[0],
        args.origin_y if args.origin_y is not None else automatic_origin[1],
        args.origin_z if args.origin_z is not None else automatic_origin[2],
    )

    buildings: list[PolygonFeature] = []
    buildings_source_crs = None
    if args.buildings:
        buildings, buildings_source_crs = read_polygon_features(
            args.buildings, args.buildings_crs
        )
        buildings = _clip_features(buildings, buffer_geometry)
    vegetation: list[PolygonFeature] = []
    vegetation_source_crs = None
    if args.vegetation:
        vegetation, vegetation_source_crs = read_polygon_features(
            args.vegetation, args.vegetation_crs
        )
        vegetation = _clip_features(vegetation, buffer_geometry)

    roads: list[LineFeature] = []
    road_sources: list[dict[str, Any]] = []
    for path in args.roads:
        source_features, source_crs = read_line_features(path, args.roads_crs)
        road_sources.append(
            {
                "file_name": path.name,
                "sha256": sha256_file(path),
                "source_crs": source_crs,
                "source_line_count": len(source_features),
            }
        )
        roads.extend(source_features)
    roads = _clip_line_features(roads, buffer_geometry)

    water_courses: list[LineFeature] = []
    water_course_sources: list[dict[str, Any]] = []
    for path in args.water_courses:
        source_features, source_crs = read_line_features(path, args.water_crs)
        water_course_sources.append(
            {
                "file_name": path.name,
                "sha256": sha256_file(path),
                "source_crs": source_crs,
                "source_line_count": len(source_features),
            }
        )
        water_courses.extend(source_features)
    water_courses = _clip_line_features(water_courses, buffer_geometry)

    water_segments: list[LineFeature] = []
    water_segment_sources: list[dict[str, Any]] = []
    for path in args.water_segments:
        source_features, source_crs = read_line_features(path, args.water_crs)
        water_segment_sources.append(
            {
                "file_name": path.name,
                "sha256": sha256_file(path),
                "source_crs": source_crs,
                "source_line_count": len(source_features),
            }
        )
        water_segments.extend(source_features)
    water_segments = _clip_line_features(water_segments, buffer_geometry)

    water_surfaces: list[PolygonFeature] = []
    water_surface_sources: list[dict[str, Any]] = []
    for path in args.water_surfaces:
        source_features, source_crs = read_polygon_features(path, args.water_crs)
        water_surface_sources.append(
            {
                "file_name": path.name,
                "sha256": sha256_file(path),
                "source_crs": source_crs,
                "source_polygon_count": len(source_features),
            }
        )
        water_surfaces.extend(source_features)
    water_surfaces = _clip_features(water_surfaces, buffer_geometry)

    building_prisms, building_statistics = _prisms(
        buildings,
        terrain,
        transform,
        origin,
        args.default_building_height_m,
        args.building_simplify_m,
        ("block_height_m", "hauteur", "height", "building:levels"),
        ("base_elevation_m", "altitude_minimale_sol", "altitude_sol", "base_elevation"),
        args.minimum_visible_building_wall_m,
    )
    if vegetation:
        vegetation_geometries = [mapping(feature.geometry) for feature in vegetation]
        vegetation_cell_transform = transform * Affine.translation(0.5, 0.5)
        vegetation_cell_mask = geometry_mask(
            vegetation_geometries,
            out_shape=(terrain.shape[0] - 1, terrain.shape[1] - 1),
            transform=vegetation_cell_transform,
            invert=True,
            all_touched=False,
        )
        vegetation_source_cell_count = int(np.count_nonzero(vegetation_cell_mask))
        vegetation_exclusion_mask, vegetation_exclusion_statistics = (
            _vegetation_exclusion_mask(
                buildings,
                roads,
                water_courses,
                water_segments,
                water_surfaces,
                vegetation_cell_mask.shape,
                vegetation_cell_transform,
                args.vegetation_building_clearance_m,
                args.vegetation_road_clearance_m,
                args.vegetation_water_clearance_m,
            )
        )
        vegetation_excluded_cell_count = int(
            np.count_nonzero(vegetation_cell_mask & vegetation_exclusion_mask)
        )
        vegetation_cell_mask &= ~vegetation_exclusion_mask
        vegetation_mesh, vegetation_statistics = _canopy_geometry(
            terrain,
            surface_model,
            transform,
            vegetation_cell_mask,
            origin,
        )
        vegetation_statistics.update(
            {
                **vegetation_exclusion_statistics,
                "unfiltered_source_polygon_cell_count": vegetation_source_cell_count,
                "excluded_source_polygon_cell_count": vegetation_excluded_cell_count,
                "post_exclusion_polygon_cell_count": int(
                    np.count_nonzero(vegetation_cell_mask)
                ),
                "building_clearance_m": args.vegetation_building_clearance_m,
                "road_clearance_m": args.vegetation_road_clearance_m,
                "water_clearance_m": args.vegetation_water_clearance_m,
            }
        )

        # The continuous canopy shell remains the validated far-distance LOD.
        # Mid-distance proxies use the native MNT/MNS sample grid so each
        # representative is anchored to a measured ground and surface value.
        vegetation_sample_mask = geometry_mask(
            vegetation_geometries,
            out_shape=terrain.shape,
            transform=transform,
            invert=True,
            all_touched=False,
        )
        tree_exclusion_mask, tree_exclusion_statistics = _vegetation_exclusion_mask(
            buildings,
            roads,
            water_courses,
            water_segments,
            water_surfaces,
            terrain.shape,
            transform,
            args.vegetation_building_clearance_m,
            args.vegetation_road_clearance_m,
            args.vegetation_water_clearance_m,
        )
        vegetation_mid_lod = generate_mid_distance_vegetation_lod(
            terrain,
            surface_model,
            transform,
            vegetation_sample_mask,
            tree_exclusion_mask,
            origin,
            VegetationLodConfig(
                min_tree_height_m=args.mid_tree_min_height_m,
                local_max_radius_m=args.mid_tree_local_max_radius_m,
                min_spacing_m=args.mid_tree_spacing_m,
                max_proxy_count=args.mid_tree_max_count,
            ),
        )
        vegetation_mid_lod["statistics"]["exclusion_raster"] = tree_exclusion_statistics
    else:
        vegetation_mesh = {"vertices": [], "faces": [], "grid_step_pixels": 1}
        vegetation_mid_lod = None
        vegetation_statistics = {
            "source_polygon_cell_count": 0,
            "valid_canopy_cell_count": 0,
            "invalid_or_nodata_cell_count": 0,
            "top_vertex_count": 0,
            "grounded_boundary_vertex_count": 0,
            "top_face_count": 0,
            "boundary_skirt_face_count": 0,
            "clamped_mns_below_mnt_vertex_count": 0,
            "zero_height_top_vertex_count": 0,
            "minimum_canopy_height_m": None,
            "maximum_canopy_height_m": None,
        }

    road_meshes, road_statistics = build_mid_distance_road_geometry(
        roads,
        terrain,
        transform,
        origin,
        config=MidDistanceRoadConfig(pavement_offset_m=args.road_offset_m),
    )
    if water_segments:
        # BD TOPO named courses carry useful names/semantics but their geometry
        # is represented by the more complete hydrographic segment network.
        # Rendering both produced two coincident ribbons with different widths
        # and offsets throughout the secteur synthétique AOI.
        water_course_mesh = {"vertices": [], "faces": []}
        water_course_statistics = {
            "input_line_count": len(water_courses),
            "input_entity_count": len(
                {feature.feature_id.split(":", 1)[0] for feature in water_courses}
            ),
            "rendered_entity_count": 0,
            "skipped_degenerate_line_count": 0,
            "vertex_count": 0,
            "face_count": 0,
            "minimum_width_m": None,
            "maximum_width_m": None,
            "width_method_counts": {},
            "ground_offset_m": args.water_course_offset_m,
            "render_suppressed": True,
            "render_suppression_reason": (
                "named_courses_are_semantic_subset_of_hydro_segments"
            ),
        }
    else:
        water_course_mesh, water_course_statistics = _road_geometry(
            water_courses,
            terrain,
            transform,
            origin,
            args.water_course_offset_m,
            _water_course_width_m,
        )
    water_segment_mesh, water_segment_statistics = _road_geometry(
        water_segments,
        terrain,
        transform,
        origin,
        args.water_segment_offset_m,
        _water_segment_width_m,
    )
    water_surface_mesh, water_surface_statistics = _water_surface_geometry(
        water_surfaces,
        terrain,
        transform,
        origin,
        args.water_surface_offset_m,
    )

    # Medium-distance vectors must follow the surface Blender actually renders,
    # not the denser source raster hidden behind that preview.  Re-drape and
    # refine every layer on the exact ``terrain_step`` grid before serialising.
    terrain_preview_mesh = _terrain_geometry(
        terrain, transform, valid_buffer_mask, origin, args.terrain_step
    )
    preview_sampling_spec = _regular_terrain_sampling_spec(
        terrain, transform, origin, args.terrain_step
    )
    preview_sampler = TerrainMeshSampler.from_terrain_spec(preview_sampling_spec)
    preview_bounds_l93 = raster_bounds.bounds
    preview_bounds_local = (
        float(preview_bounds_l93[0]) - origin[0],
        float(preview_bounds_l93[1]) - origin[1],
        float(preview_bounds_l93[2]) - origin[0],
        float(preview_bounds_l93[3]) - origin[1],
    )
    medium_road_offsets = {
        "carriageway": args.road_offset_m,
        "left_shoulders": max(args.road_offset_m - 0.04, 0.0),
        "right_shoulders": max(args.road_offset_m - 0.04, 0.0),
        "center_markings": args.road_offset_m + 0.018,
    }
    medium_road_drape_statistics: dict[str, Any] = {}
    for layer, offset_m in medium_road_offsets.items():
        road_meshes[layer], medium_road_drape_statistics[layer] = (
            drape_ribbon_mesh_to_tile(
                road_meshes[layer],
                preview_sampler,
                preview_bounds_local,
                offset_m=offset_m,
                maximum_segment_length_m=5.0,
            )
        )
    road_statistics["active_render_surface_drape"] = {
        "terrain_step_pixels": args.terrain_step,
        "maximum_cell_edge_m": 5.0,
        "layers": medium_road_drape_statistics,
    }
    road_statistics["geometry_mode"] = "rendered_preview_surface_refined_ribbon_meshes"
    road_statistics["active_render_surface"] = "fixed_nw_se_triangle_planes"
    road_statistics["mesh_statistics"] = {
        layer: {
            "vertex_count": len(mesh["vertices"]),
            "face_count": len(mesh["faces"]),
        }
        for layer, mesh in road_meshes.items()
    }

    if water_course_mesh["faces"]:
        water_course_mesh, course_drape_statistics = drape_ribbon_mesh_to_tile(
            water_course_mesh,
            preview_sampler,
            preview_bounds_local,
            offset_m=args.water_course_offset_m,
            maximum_segment_length_m=5.0,
        )
        water_course_statistics["active_render_surface_drape"] = course_drape_statistics
    water_segment_mesh, segment_drape_statistics = drape_ribbon_mesh_to_tile(
        water_segment_mesh,
        preview_sampler,
        preview_bounds_local,
        offset_m=args.water_segment_offset_m,
        maximum_segment_length_m=5.0,
    )
    water_segment_statistics.update(
        {
            "vertex_count": len(water_segment_mesh["vertices"]),
            "face_count": len(water_segment_mesh["faces"]),
            "active_render_surface_drape": segment_drape_statistics,
            "altitude_method": (
                "rendered_terrain_preview_fixed_nw_se_triangles_refined"
            ),
        }
    )
    water_surface_mesh, surface_drape_statistics = drape_triangle_mesh_to_tile(
        water_surface_mesh,
        preview_sampler,
        preview_bounds_local,
        offset_m=args.water_surface_offset_m,
        maximum_edge_length_m=5.0,
    )
    water_surface_statistics.update(
        {
            "vertex_count": len(water_surface_mesh["vertices"]),
            "triangle_count": len(water_surface_mesh["faces"]),
            "active_render_surface_drape": surface_drape_statistics,
            "altitude_method": (
                "rendered_terrain_preview_fixed_nw_se_triangles_refined"
            ),
        }
    )

    building_prisms, preview_building_statistics = drape_building_prisms_to_tile(
        building_prisms,
        preview_sampler,
        preview_bounds_local,
        minimum_visible_wall_height_m=args.minimum_visible_building_wall_m,
        maximum_boundary_segment_length_m=5.0,
    )
    building_statistics.update(
        {
            "extruded_prism_count": len(building_prisms),
            "draped_foundation_prism_count": len(building_prisms),
            "foundation_grounding": ("rendered_terrain_preview_per_boundary_vertex"),
            "minimum_visible_wall_height_m": (args.minimum_visible_building_wall_m),
            "active_render_surface_drape": preview_building_statistics,
        }
    )

    metadata: dict[str, Any] = {
        "schema": "fireviewer.blender-control-scene.v2",
        "target_crs": TARGET_CRS,
        "axis_convention": "X=east, Y=north, Z=up",
        "linear_unit": "metre",
        "origin_l93_m": list(origin),
        "buffer_m": args.buffer_m,
        "buffer_area_m2": buffer_geometry.area,
        "terrain": {
            "file_name": args.mnt.name,
            "sha256": sha256_file(args.mnt),
            "shape": list(raster_shape),
            "pixel_size_m": list(pixel_size),
            "minimum_m": float(valid_values.min()),
            "maximum_m": float(valid_values.max()),
            "buffer_coverage_ratio": coverage,
            "preview_step_pixels": args.terrain_step,
            "crs_validation": crs_validation,
        },
        "surface_model": {
            "file_name": args.mns.name if args.mns else None,
            "sha256": sha256_file(args.mns) if args.mns else None,
            "shape": list(raster_shape) if args.mns else None,
            "pixel_size_m": list(pixel_size) if args.mns else None,
            "crs_validation": mns_crs_validation,
            "alignment": "pixel_exact_with_mnt" if args.mns else None,
        },
        "perimeter": {
            "file_name": args.perimeter.name,
            "sha256": sha256_file(args.perimeter),
            "source_crs": perimeter_source_crs,
            "selected_feature_id": args.perimeter_feature_id,
            "source_axis_swapped": args.perimeter_swap_xy,
            "area_m2": perimeter.area,
            "polygon_count": len(perimeter_features),
            "rendered_as_fire_perimeter": render_fire_perimeter,
            "rendering_policy": (
                "incident-ring"
                if render_fire_perimeter
                else "clipping-only-non-incident-envelope"
            ),
        },
        "buildings": {
            "file_name": args.buildings.name if args.buildings else None,
            "sha256": sha256_file(args.buildings) if args.buildings else None,
            "source_crs": buildings_source_crs,
            "clipped_polygon_count": len(buildings),
            "extrusion": building_statistics,
        },
        "vegetation": {
            "file_name": args.vegetation.name if args.vegetation else None,
            "sha256": sha256_file(args.vegetation) if args.vegetation else None,
            "source_crs": vegetation_source_crs,
            "clipped_polygon_count": len(vegetation),
            "canopy_mesh": vegetation_statistics,
            "mid_distance_lod": (
                vegetation_mid_lod["statistics"]
                if vegetation_mid_lod is not None
                else None
            ),
        },
        "roads": {
            "sources": road_sources,
            "clipped_line_count": len(roads),
            "mid_distance_surface_meshes": road_statistics,
        },
        "water": {
            "course_sources": water_course_sources,
            "segment_sources": water_segment_sources,
            "surface_sources": water_surface_sources,
            "courses": water_course_statistics,
            "segments": water_segment_statistics,
            "surfaces": water_surface_statistics,
        },
    }
    package = {
        "schema": PACKAGE_SCHEMA,
        "metadata": metadata,
        "terrain": terrain_preview_mesh,
        "fire_perimeter": {
            "rings": (
                []
                if not render_fire_perimeter
                else _boundary_rings(
                    perimeter, terrain, transform, origin, offset_m=3.0
                )
            )
        },
        "analysis_buffer": {
            "rings": _boundary_rings(
                buffer_geometry, terrain, transform, origin, offset_m=2.0
            )
        },
        "buildings": {
            "prisms": building_prisms,
            "statistics": building_statistics,
        },
        "vegetation": {
            "mesh": vegetation_mesh,
            "statistics": vegetation_statistics,
            **(
                {"mid_distance_lod": vegetation_mid_lod}
                if vegetation_mid_lod is not None
                else {}
            ),
        },
        "roads": {
            "meshes": road_meshes,
            "statistics": road_statistics,
        },
        "water": {
            "courses": {
                "mesh": water_course_mesh,
                "statistics": water_course_statistics,
            },
            "segments": {
                "mesh": water_segment_mesh,
                "statistics": water_segment_statistics,
            },
            "surfaces": {
                "mesh": water_surface_mesh,
                "statistics": water_surface_statistics,
            },
        },
    }
    metadata["preview_geometry"] = {
        "terrain_vertices": len(package["terrain"]["vertices"]),
        "terrain_faces": len(package["terrain"]["faces"]),
        "fire_rings": len(package["fire_perimeter"]["rings"]),
        "buffer_rings": len(package["analysis_buffer"]["rings"]),
        "building_prisms": len(package["buildings"]["prisms"]),
        "vegetation_vertices": len(package["vegetation"]["mesh"]["vertices"]),
        "vegetation_faces": len(package["vegetation"]["mesh"]["faces"]),
        "mid_tree_vertices": (
            len(package["vegetation"]["mid_distance_lod"]["mesh"]["vertices"])
            if "mid_distance_lod" in package["vegetation"]
            else 0
        ),
        "mid_tree_faces": (
            len(package["vegetation"]["mid_distance_lod"]["mesh"]["faces"])
            if "mid_distance_lod" in package["vegetation"]
            else 0
        ),
        "road_vertices": sum(
            len(mesh["vertices"]) for mesh in package["roads"]["meshes"].values()
        ),
        "road_faces": sum(
            len(mesh["faces"]) for mesh in package["roads"]["meshes"].values()
        ),
        "road_meshes": {
            key: {"vertices": len(mesh["vertices"]), "faces": len(mesh["faces"])}
            for key, mesh in package["roads"]["meshes"].items()
        },
        "water_course_faces": len(package["water"]["courses"]["mesh"]["faces"]),
        "water_segment_faces": len(package["water"]["segments"]["mesh"]["faces"]),
        "water_surface_triangles": len(package["water"]["surfaces"]["mesh"]["faces"]),
    }
    absolute_paths = find_absolute_local_paths(package)
    if absolute_paths:
        raise ValueError(
            "Preview package contains non-portable absolute paths at: "
            + ", ".join(absolute_paths)
        )
    return package


def write_package(package: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        package, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    output.write_bytes(gzip.compress(serialized, compresslevel=9, mtime=0))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_arguments(sys.argv if argv is None else argv)
        package = prepare_package(args)
        print(json.dumps(package["metadata"], indent=2, sort_keys=True))
        if not args.validate_only:
            write_package(package, args.output.resolve())
        return 0
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
