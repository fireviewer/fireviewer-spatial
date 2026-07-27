"""Tile-local detailed vector meshes draped on the active detail terrain.

The global control package deliberately keeps inexpensive vector meshes for the
medium-distance view.  Those meshes are sampled on the global MNT and must not
be reused unchanged when a 0.5 m terrain tile becomes active.  This module is
Blender-independent: it clips/refines the already serialised vector geometry
into one 500 m tile and samples every generated vertex on the exact terrain
mesh shipped by that tile.

All coordinates remain local to the shared Lambert-93 scene origin.  The
functions return JSON-compatible dictionaries that ``build_control_scene`` can
turn into Blender meshes without importing geospatial dependencies in Blender.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


DETAIL_ROAD_MAX_SEGMENT_M = 1.0
DETAIL_WATER_LINE_MAX_SEGMENT_M = 1.0
DETAIL_WATER_SURFACE_MAX_EDGE_M = 2.0
DETAIL_ROAD_OFFSETS_M = {
    "carriageway": 0.12,
    "left_shoulders": 0.10,
    "right_shoulders": 0.10,
    "center_markings": 0.14,
}
DETAIL_WATER_SEGMENT_OFFSET_M = 0.10
DETAIL_WATER_SURFACE_OFFSET_M = 0.08
DETAIL_BUILDING_MINIMUM_VISIBLE_WALL_M = 2.70
DETAIL_BUILDING_MAXIMUM_BOUNDARY_SEGMENT_M = 1.0
FOUNDATION_CLEARANCE_M = 0.005
SURFACE_CLEARANCE_SAFETY_M = 0.002


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(
        float(right[0]) - float(left[0]), float(right[1]) - float(left[1])
    )


def _interpolate(
    start: Sequence[float], end: Sequence[float], factor: float
) -> tuple[float, float]:
    return (
        float(start[0]) + (float(end[0]) - float(start[0])) * factor,
        float(start[1]) + (float(end[1]) - float(start[1])) * factor,
    )


def _inside_half_open(point: Sequence[float], bounds: Sequence[float]) -> bool:
    return float(bounds[0]) <= float(point[0]) < float(bounds[2]) and float(
        bounds[1]
    ) <= float(point[1]) < float(bounds[3])


def _bbox_intersects(
    points: Sequence[Sequence[float]], bounds: Sequence[float]
) -> bool:
    minimum_x = min(float(point[0]) for point in points)
    maximum_x = max(float(point[0]) for point in points)
    minimum_y = min(float(point[1]) for point in points)
    maximum_y = max(float(point[1]) for point in points)
    return not (
        maximum_x < float(bounds[0])
        or minimum_x >= float(bounds[2])
        or maximum_y < float(bounds[1])
        or minimum_y >= float(bounds[3])
    )


def _cross_2d(
    start: Sequence[float], end: Sequence[float], point: Sequence[float]
) -> float:
    return (float(end[0]) - float(start[0])) * (float(point[1]) - float(start[1])) - (
        float(end[1]) - float(start[1])
    ) * (float(point[0]) - float(start[0]))


def _clip_polygon_to_convex_triangle(
    polygon: Sequence[Sequence[float]],
    triangle: Sequence[Sequence[float]],
) -> list[tuple[float, float]]:
    """Clip a polygon against one counter-clockwise terrain triangle."""

    active = [(float(point[0]), float(point[1])) for point in polygon]
    for clip_start, clip_end in zip(triangle, (*triangle[1:], triangle[0])):
        if not active:
            break
        output: list[tuple[float, float]] = []
        previous = active[-1]
        previous_side = _cross_2d(clip_start, clip_end, previous)
        for current in active:
            current_side = _cross_2d(clip_start, clip_end, current)
            previous_inside = previous_side >= -1e-9
            current_inside = current_side >= -1e-9
            if current_inside != previous_inside:
                denominator = previous_side - current_side
                factor = (
                    0.0 if abs(denominator) <= 1e-15 else previous_side / denominator
                )
                output.append(_interpolate(previous, current, factor))
            if current_inside:
                output.append(current)
            previous = current
            previous_side = current_side
        active = output
    return active


def _triangle_interpolated_z(
    triangle: Sequence[Sequence[float]], point: Sequence[float]
) -> float:
    a, b, c = triangle
    denominator = _triangle_signed_double_area_xy(triangle)
    if abs(denominator) <= 1e-12:
        raise ValueError("vector face contains a degenerate triangle")
    first = (
        (float(b[1]) - float(c[1])) * (float(point[0]) - float(c[0]))
        + (float(c[0]) - float(b[0])) * (float(point[1]) - float(c[1]))
    ) / denominator
    second = (
        (float(c[1]) - float(a[1])) * (float(point[0]) - float(c[0]))
        + (float(a[0]) - float(c[0])) * (float(point[1]) - float(c[1]))
    ) / denominator
    third = 1.0 - first - second
    return first * float(a[2]) + second * float(b[2]) + third * float(c[2])


def _triangle_signed_double_area_xy(
    triangle: Sequence[Sequence[float]],
) -> float:
    a, b, c = triangle
    return (float(b[1]) - float(c[1])) * (float(a[0]) - float(c[0])) + (
        float(c[0]) - float(b[0])
    ) * (float(a[1]) - float(c[1]))


def _triangle_is_degenerate_xy(triangle: Sequence[Sequence[float]]) -> bool:
    return abs(_triangle_signed_double_area_xy(triangle)) <= 1e-12


def local_tile_bounds(
    bounds_l93_m: Sequence[float], origin_l93_m: Sequence[float]
) -> tuple[float, float, float, float]:
    if len(bounds_l93_m) != 4 or len(origin_l93_m) < 2:
        raise ValueError("tile bounds and origin have invalid dimensions")
    return (
        _finite(bounds_l93_m[0], "minimum X") - _finite(origin_l93_m[0], "origin X"),
        _finite(bounds_l93_m[1], "minimum Y") - _finite(origin_l93_m[1], "origin Y"),
        _finite(bounds_l93_m[2], "maximum X") - _finite(origin_l93_m[0], "origin X"),
        _finite(bounds_l93_m[3], "maximum Y") - _finite(origin_l93_m[1], "origin Y"),
    )


@dataclass(frozen=True)
class TerrainMeshSampler:
    """Fixed-triangle sampler over a regular serialised terrain mesh."""

    vertices: Sequence[Sequence[float]]
    x_values: tuple[float, ...]
    y_values_ascending: tuple[float, ...]
    column_count: int
    row_count: int

    @classmethod
    def from_terrain_spec(cls, terrain: Mapping[str, Any]) -> "TerrainMeshSampler":
        vertices = terrain.get("vertices")
        if not isinstance(vertices, list) or len(vertices) < 4:
            raise ValueError("detail terrain must contain a regular vertex grid")
        first_y = _finite(vertices[0][1], "terrain Y")
        column_count = 0
        for vertex in vertices:
            if not math.isclose(
                _finite(vertex[1], "terrain Y"), first_y, rel_tol=0.0, abs_tol=1e-7
            ):
                break
            column_count += 1
        if column_count < 2 or len(vertices) % column_count:
            raise ValueError(
                "detail terrain vertices are not a rectangular row-major grid"
            )
        row_count = len(vertices) // column_count
        if row_count < 2:
            raise ValueError("detail terrain requires at least two rows")
        x_values = tuple(
            _finite(vertex[0], "terrain X") for vertex in vertices[:column_count]
        )
        y_descending = tuple(
            _finite(vertices[row * column_count][1], "terrain Y")
            for row in range(row_count)
        )
        if any(right <= left for left, right in zip(x_values, x_values[1:])):
            raise ValueError("detail terrain X coordinates must increase")
        if any(right >= left for left, right in zip(y_descending, y_descending[1:])):
            raise ValueError("detail terrain Y coordinates must decrease")
        for row in range(row_count):
            row_start = row * column_count
            for column, expected_x in enumerate(x_values):
                vertex = vertices[row_start + column]
                if not math.isclose(
                    _finite(vertex[0], "terrain X"),
                    expected_x,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    raise ValueError("detail terrain rows do not share one X grid")
        return cls(
            vertices=vertices,
            x_values=x_values,
            y_values_ascending=tuple(reversed(y_descending)),
            column_count=column_count,
            row_count=row_count,
        )

    def _z(self, ascending_row: int, column: int) -> float:
        source_row = self.row_count - 1 - ascending_row
        return _finite(
            self.vertices[source_row * self.column_count + column][2], "terrain Z"
        )

    def sample(self, x: float, y: float) -> float:
        """Return Z on the fixed NW-SE triangulation rendered by Blender.

        A bilinear height-field sample is not the same surface as the two GPU
        triangles used to render a non-coplanar terrain quad.  On the steep
        secteur synthétique relief that mismatch reached several metres.  The terrain
        builders therefore split every source quad as ``(NW, SW, SE)`` and
        ``(NW, SE, NE)``; this sampler evaluates those exact planes.
        """

        active_x = min(max(_finite(x, "sample X"), self.x_values[0]), self.x_values[-1])
        active_y = min(
            max(_finite(y, "sample Y"), self.y_values_ascending[0]),
            self.y_values_ascending[-1],
        )
        column = min(
            max(bisect_right(self.x_values, active_x) - 1, 0),
            self.column_count - 2,
        )
        row = min(
            max(bisect_right(self.y_values_ascending, active_y) - 1, 0),
            self.row_count - 2,
        )
        x0, x1 = self.x_values[column], self.x_values[column + 1]
        y0, y1 = self.y_values_ascending[row], self.y_values_ascending[row + 1]
        x_fraction = 0.0 if x1 == x0 else (active_x - x0) / (x1 - x0)
        y_fraction = 0.0 if y1 == y0 else (active_y - y0) / (y1 - y0)
        z00 = self._z(row, column)
        z10 = self._z(row, column + 1)
        z01 = self._z(row + 1, column)
        z11 = self._z(row + 1, column + 1)
        if x_fraction + y_fraction <= 1.0:
            # South-west / south-east / north-west triangle.
            return z00 + x_fraction * (z10 - z00) + y_fraction * (z01 - z00)
        # North-east / north-west / south-east triangle.
        return z11 + (1.0 - x_fraction) * (z01 - z11) + (1.0 - y_fraction) * (z10 - z11)

    def _overlapping_cell_indices(
        self, points: Sequence[Sequence[float]]
    ) -> tuple[range, range]:
        minimum_x = max(min(float(point[0]) for point in points), self.x_values[0])
        maximum_x = min(max(float(point[0]) for point in points), self.x_values[-1])
        minimum_y = max(
            min(float(point[1]) for point in points), self.y_values_ascending[0]
        )
        maximum_y = min(
            max(float(point[1]) for point in points), self.y_values_ascending[-1]
        )
        first_column = min(
            max(bisect_right(self.x_values, minimum_x) - 1, 0),
            self.column_count - 2,
        )
        last_column = min(
            max(bisect_right(self.x_values, maximum_x) - 1, 0),
            self.column_count - 2,
        )
        first_row = min(
            max(bisect_right(self.y_values_ascending, minimum_y) - 1, 0),
            self.row_count - 2,
        )
        last_row = min(
            max(bisect_right(self.y_values_ascending, maximum_y) - 1, 0),
            self.row_count - 2,
        )
        return (
            range(first_column, last_column + 1),
            range(first_row, last_row + 1),
        )

    def maximum_triangle_surface_deficit(
        self,
        triangle: Sequence[Sequence[float]],
        *,
        vertical_offset_m: float,
    ) -> float:
        """Return the exact lift needed above all intersected terrain planes."""

        if len(triangle) != 3:
            raise ValueError("surface-clearance audit requires one triangle")
        columns, rows = self._overlapping_cell_indices(triangle)
        maximum_deficit = 0.0
        vector_xy = [(float(point[0]), float(point[1])) for point in triangle]
        target_offset = _finite(vertical_offset_m, "vertical offset")
        for row in rows:
            y0 = self.y_values_ascending[row]
            y1 = self.y_values_ascending[row + 1]
            for column in columns:
                x0 = self.x_values[column]
                x1 = self.x_values[column + 1]
                for terrain_triangle in (
                    ((x0, y0), (x1, y0), (x0, y1)),
                    ((x0, y1), (x1, y0), (x1, y1)),
                ):
                    overlap = _clip_polygon_to_convex_triangle(
                        vector_xy, terrain_triangle
                    )
                    for point in overlap:
                        vector_z = _triangle_interpolated_z(triangle, point)
                        deficit = (
                            self.sample(point[0], point[1]) + target_offset - vector_z
                        )
                        maximum_deficit = max(maximum_deficit, deficit)
        return maximum_deficit

    def segment_breakpoints(
        self,
        start: Sequence[float],
        end: Sequence[float],
        *,
        maximum_segment_length_m: float,
    ) -> list[tuple[float, float]]:
        """Split a line at every crossed grid/terrain-triangle boundary."""

        maximum_length = _finite(maximum_segment_length_m, "maximum segment length")
        if maximum_length <= 0.0:
            raise ValueError("maximum segment length must be strictly positive")
        start_xy = (float(start[0]), float(start[1]))
        end_xy = (float(end[0]), float(end[1]))
        segment_length = _distance(start_xy, end_xy)
        subdivisions = max(1, int(math.ceil(segment_length / maximum_length)))
        factors = {index / subdivisions for index in range(subdivisions + 1)}
        delta_x = end_xy[0] - start_xy[0]
        delta_y = end_xy[1] - start_xy[1]
        if abs(delta_x) > 1e-12:
            minimum_x, maximum_x = sorted((start_xy[0], end_xy[0]))
            first_x = bisect_right(self.x_values, minimum_x)
            last_x = bisect_right(self.x_values, maximum_x)
            for x_value in self.x_values[first_x:last_x]:
                factor = (x_value - start_xy[0]) / delta_x
                if 1e-9 < factor < 1.0 - 1e-9:
                    factors.add(round(factor, 12))
        if abs(delta_y) > 1e-12:
            minimum_y, maximum_y = sorted((start_xy[1], end_xy[1]))
            first_y = bisect_right(self.y_values_ascending, minimum_y)
            last_y = bisect_right(self.y_values_ascending, maximum_y)
            for y_value in self.y_values_ascending[first_y:last_y]:
                factor = (y_value - start_xy[1]) / delta_y
                if 1e-9 < factor < 1.0 - 1e-9:
                    factors.add(round(factor, 12))
        columns, rows = self._overlapping_cell_indices((start_xy, end_xy))
        for row in rows:
            y0 = self.y_values_ascending[row]
            y1 = self.y_values_ascending[row + 1]
            for column in columns:
                x0 = self.x_values[column]
                x1 = self.x_values[column + 1]
                start_side = (
                    (start_xy[0] - x0) / (x1 - x0)
                    + (start_xy[1] - y0) / (y1 - y0)
                    - 1.0
                )
                end_side = (
                    (end_xy[0] - x0) / (x1 - x0) + (end_xy[1] - y0) / (y1 - y0) - 1.0
                )
                denominator = end_side - start_side
                if abs(denominator) <= 1e-12:
                    continue
                factor = -start_side / denominator
                if not 1e-9 < factor < 1.0 - 1e-9:
                    continue
                point = _interpolate(start_xy, end_xy, factor)
                if (
                    x0 - 1e-8 <= point[0] <= x1 + 1e-8
                    and y0 - 1e-8 <= point[1] <= y1 + 1e-8
                ):
                    factors.add(round(factor, 12))
        return [_interpolate(start_xy, end_xy, factor) for factor in sorted(factors)]


class _MeshAccumulator:
    def __init__(self, sampler: TerrainMeshSampler, offset_m: float) -> None:
        self.sampler = sampler
        self.offset_m = _finite(offset_m, "vertical offset")
        self.vertices: list[list[float]] = []
        self.faces: list[list[int]] = []
        self._indices: dict[tuple[float, float], int] = {}
        self.clearance_lifted_face_count = 0
        self.maximum_clearance_lift_m = 0.0
        self.skipped_degenerate_face_count = 0

    def vertex(self, point: Sequence[float]) -> int:
        x = round(_finite(point[0], "vertex X"), 3)
        y = round(_finite(point[1], "vertex Y"), 3)
        key = (x, y)
        existing = self._indices.get(key)
        if existing is not None:
            return existing
        index = len(self.vertices)
        self.vertices.append(
            [x, y, round(self.sampler.sample(x, y) + self.offset_m, 3)]
        )
        self._indices[key] = index
        return index

    def face(self, points: Sequence[Sequence[float]]) -> int:
        serialized_points = [
            (
                round(_finite(point[0], "vertex X"), 3),
                round(_finite(point[1], "vertex Y"), 3),
            )
            for point in points
        ]
        point_triangles = (
            [serialized_points]
            if len(serialized_points) == 3
            else [
                [
                    serialized_points[0],
                    serialized_points[index],
                    serialized_points[index + 1],
                ]
                for index in range(1, len(serialized_points) - 1)
            ]
        )
        accepted_face_count = 0
        for active_points in point_triangles:
            if len(set(active_points)) < 3:
                self.skipped_degenerate_face_count += 1
                continue
            # Vertex coordinates are serialized to millimetres before the
            # clearance audit. Distinct source/index coordinates can therefore
            # collapse or become collinear here and no longer define a surface.
            if _triangle_is_degenerate_xy(active_points):
                self.skipped_degenerate_face_count += 1
                continue
            active_indices = [self.vertex(point) for point in active_points]
            triangle = [self.vertices[index] for index in active_indices]
            required_lift = self.sampler.maximum_triangle_surface_deficit(
                triangle,
                vertical_offset_m=self.offset_m,
            )
            if required_lift > 1e-9:
                applied_lift = required_lift + SURFACE_CLEARANCE_SAFETY_M
                for index in active_indices:
                    self.vertices[index][2] = round(
                        float(self.vertices[index][2]) + applied_lift, 3
                    )
                self.clearance_lifted_face_count += 1
                self.maximum_clearance_lift_m = max(
                    self.maximum_clearance_lift_m, applied_lift
                )
            self.faces.append(active_indices)
            accepted_face_count += 1
        return accepted_face_count

    def mesh(self) -> dict[str, list[list[float]] | list[list[int]]]:
        return {"vertices": self.vertices, "faces": self.faces}


def drape_ribbon_mesh_to_tile(
    mesh: Mapping[str, Any],
    sampler: TerrainMeshSampler,
    tile_bounds_local: Sequence[float],
    *,
    offset_m: float,
    maximum_segment_length_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refine serialised ribbon quads and drape them on one detail tile."""

    maximum_length = _finite(maximum_segment_length_m, "maximum segment length")
    if maximum_length <= 0.0:
        raise ValueError("maximum segment length must be strictly positive")
    source_vertices = mesh.get("vertices", [])
    source_faces = mesh.get("faces", [])
    accumulator = _MeshAccumulator(sampler, offset_m)
    selected_source_faces = 0
    generated_face_count = 0
    maximum_generated_length = 0.0
    for face in source_faces:
        if len(face) != 4:
            continue
        try:
            start_left, start_right, end_right, end_left = (
                source_vertices[int(index)] for index in face
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("ribbon mesh contains an invalid face index") from exc
        source_points = (start_left, start_right, end_right, end_left)
        if not _bbox_intersects(source_points, tile_bounds_local):
            continue
        longitudinal_length = max(
            _distance(start_left, end_left), _distance(start_right, end_right)
        )
        lateral_length = max(
            _distance(start_left, start_right), _distance(end_left, end_right)
        )
        longitudinal_subdivisions = max(
            1, int(math.ceil(longitudinal_length / maximum_length))
        )
        lateral_subdivisions = max(1, int(math.ceil(lateral_length / maximum_length)))
        source_selected = False
        for subdivision in range(longitudinal_subdivisions):
            start_factor = subdivision / longitudinal_subdivisions
            end_factor = (subdivision + 1) / longitudinal_subdivisions
            left_start = _interpolate(start_left, end_left, start_factor)
            right_start = _interpolate(start_right, end_right, start_factor)
            right_end = _interpolate(start_right, end_right, end_factor)
            left_end = _interpolate(start_left, end_left, end_factor)
            for lateral_subdivision in range(lateral_subdivisions):
                left_factor = lateral_subdivision / lateral_subdivisions
                right_factor = (lateral_subdivision + 1) / lateral_subdivisions
                cell_start_left = _interpolate(left_start, right_start, left_factor)
                cell_start_right = _interpolate(left_start, right_start, right_factor)
                cell_end_right = _interpolate(left_end, right_end, right_factor)
                cell_end_left = _interpolate(left_end, right_end, left_factor)
                cell = (
                    cell_start_left,
                    cell_start_right,
                    cell_end_right,
                    cell_end_left,
                )
                clipped_cell = _clip_ring_to_bounds(cell, tile_bounds_local)
                if len(clipped_cell) < 3:
                    continue
                accepted_face_count = accumulator.face(clipped_cell)
                generated_face_count += accepted_face_count
                source_selected = source_selected or accepted_face_count > 0
                for index in range(1, len(clipped_cell) - 1):
                    triangle = (
                        clipped_cell[0],
                        clipped_cell[index],
                        clipped_cell[index + 1],
                    )
                    maximum_generated_length = max(
                        maximum_generated_length,
                        *(
                            _distance(start, end)
                            for start, end in zip(
                                triangle, (*triangle[1:], triangle[0])
                            )
                        ),
                    )
        if source_selected:
            selected_source_faces += 1
    result = accumulator.mesh()
    return result, {
        "source_face_count": len(source_faces),
        "selected_source_face_count": selected_source_faces,
        "generated_face_count": generated_face_count,
        "generated_vertex_count": len(result["vertices"]),
        "maximum_segment_length_m": round(maximum_generated_length, 3),
        "target_maximum_segment_length_m": maximum_length,
        "vertical_offset_m": float(offset_m),
        "clearance_lifted_face_count": accumulator.clearance_lifted_face_count,
        "maximum_clearance_lift_m": round(accumulator.maximum_clearance_lift_m, 3),
        "skipped_degenerate_face_count": accumulator.skipped_degenerate_face_count,
        "clearance_validation": ("exact_overlap_with_fixed_terrain_triangles"),
        "altitude_method": (
            "tile_detail_terrain_fixed_nw_se_triangle_every_generated_vertex"
        ),
    }


def _triangle_maximum_edge(points: Sequence[Sequence[float]]) -> tuple[float, int]:
    edges = (
        (_distance(points[0], points[1]), 0),
        (_distance(points[1], points[2]), 1),
        (_distance(points[2], points[0]), 2),
    )
    return max(edges, key=lambda item: item[0])


def drape_triangle_mesh_to_tile(
    mesh: Mapping[str, Any],
    sampler: TerrainMeshSampler,
    tile_bounds_local: Sequence[float],
    *,
    offset_m: float,
    maximum_edge_length_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refine large hydro triangles until they can follow the detail terrain."""

    maximum_length = _finite(maximum_edge_length_m, "maximum edge length")
    if maximum_length <= 0.0:
        raise ValueError("maximum edge length must be strictly positive")
    source_vertices = mesh.get("vertices", [])
    source_faces = mesh.get("faces", [])
    accumulator = _MeshAccumulator(sampler, offset_m)
    selected_source_faces = 0
    generated_face_count = 0
    maximum_generated_edge = 0.0
    skipped_degenerate_source_face_count = 0
    for face in source_faces:
        if len(face) != 3:
            continue
        try:
            triangle = tuple(source_vertices[int(index)][:2] for index in face)
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("triangle mesh contains an invalid face index") from exc
        if _triangle_is_degenerate_xy(triangle):
            skipped_degenerate_source_face_count += 1
            continue
        if not _bbox_intersects(triangle, tile_bounds_local):
            continue
        stack = [triangle]
        source_selected = False
        while stack:
            active = stack.pop()
            if not _bbox_intersects(active, tile_bounds_local):
                continue
            longest, edge_index = _triangle_maximum_edge(active)
            if longest > maximum_length:
                first = edge_index
                second = (edge_index + 1) % 3
                opposite = (edge_index + 2) % 3
                midpoint = _interpolate(active[first], active[second], 0.5)
                stack.append((active[first], midpoint, active[opposite]))
                stack.append((midpoint, active[second], active[opposite]))
                continue
            clipped_triangle = _clip_ring_to_bounds(active, tile_bounds_local)
            if len(clipped_triangle) < 3:
                continue
            accepted_face_count = accumulator.face(clipped_triangle)
            source_selected = source_selected or accepted_face_count > 0
            generated_face_count += accepted_face_count
            for index in range(1, len(clipped_triangle) - 1):
                generated_triangle = (
                    clipped_triangle[0],
                    clipped_triangle[index],
                    clipped_triangle[index + 1],
                )
                maximum_generated_edge = max(
                    maximum_generated_edge,
                    *(
                        _distance(start, end)
                        for start, end in zip(
                            generated_triangle,
                            (*generated_triangle[1:], generated_triangle[0]),
                        )
                    ),
                )
        if source_selected:
            selected_source_faces += 1
    result = accumulator.mesh()
    return result, {
        "source_face_count": len(source_faces),
        "selected_source_face_count": selected_source_faces,
        "generated_face_count": generated_face_count,
        "generated_vertex_count": len(result["vertices"]),
        "maximum_edge_length_m": round(maximum_generated_edge, 3),
        "target_maximum_edge_length_m": maximum_length,
        "vertical_offset_m": float(offset_m),
        "clearance_lifted_face_count": accumulator.clearance_lifted_face_count,
        "maximum_clearance_lift_m": round(accumulator.maximum_clearance_lift_m, 3),
        "skipped_degenerate_source_face_count": skipped_degenerate_source_face_count,
        "skipped_degenerate_face_count": accumulator.skipped_degenerate_face_count,
        "clearance_validation": ("exact_overlap_with_fixed_terrain_triangles"),
        "altitude_method": (
            "tile_detail_terrain_fixed_nw_se_triangle_every_generated_vertex"
        ),
    }


def _drape_surface_mesh_to_tile(
    mesh: Mapping[str, Any],
    sampler: TerrainMeshSampler,
    tile_bounds_local: Sequence[float],
    *,
    offset_m: float,
    maximum_edge_length_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize ribbon quads / already-triangulated surfaces for near LOD.

    The medium package serialises explicit triangles so Blender cannot choose
    another diagonal. Legacy packages may still contain ribbon quads. A fan
    conversion keeps both inputs usable and lets the detail refiner operate on
    one deterministic representation.
    """

    source_faces = mesh.get("faces", [])
    triangle_faces: list[list[int]] = []
    skipped_face_count = 0
    for face in source_faces:
        if len(face) < 3:
            skipped_face_count += 1
            continue
        triangle_faces.extend(
            [int(face[0]), int(face[index]), int(face[index + 1])]
            for index in range(1, len(face) - 1)
        )
    result, statistics = drape_triangle_mesh_to_tile(
        {"vertices": mesh.get("vertices", []), "faces": triangle_faces},
        sampler,
        tile_bounds_local,
        offset_m=offset_m,
        maximum_edge_length_m=maximum_edge_length_m,
    )
    statistics.update(
        {
            "input_surface_face_count": len(source_faces),
            "normalized_triangle_face_count": len(triangle_faces),
            "skipped_subtriangle_face_count": skipped_face_count,
            "input_surface_normalization": "polygon_fan_to_explicit_triangles",
        }
    )
    return result, statistics


def _ring_area(ring: Sequence[Sequence[float]]) -> float:
    return 0.5 * sum(
        float(start[0]) * float(end[1]) - float(end[0]) * float(start[1])
        for start, end in zip(ring, (*ring[1:], ring[0]))
    )


def _clean_ring(ring: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for coordinate in ring:
        point = (
            _finite(coordinate[0], "building ring X"),
            _finite(coordinate[1], "building ring Y"),
        )
        if not result or _distance(result[-1], point) > 1e-8:
            result.append(point)
    if len(result) > 1 and _distance(result[0], result[-1]) <= 1e-8:
        result.pop()
    return result


def _clip_ring_to_bounds(
    ring: Sequence[Sequence[float]], bounds: Sequence[float]
) -> list[tuple[float, float]]:
    """Clip one ring to the terrain core rectangle.

    Detail tiles contain no sampling halo in their rendered geometry. Splitting
    a building at a tile edge is therefore safer than assigning the complete
    footprint to its centroid tile and clamping the opposite wall to that
    tile's last terrain sample.
    """

    points = _clean_ring(ring)
    if len(points) < 3:
        return []

    def clip_axis(
        active: list[tuple[float, float]],
        axis: int,
        limit: float,
        keep_greater: bool,
    ) -> list[tuple[float, float]]:
        if not active:
            return []

        def inside(point: tuple[float, float]) -> bool:
            return (
                point[axis] >= limit - 1e-9
                if keep_greater
                else point[axis] <= limit + 1e-9
            )

        def intersection(
            start: tuple[float, float], end: tuple[float, float]
        ) -> tuple[float, float]:
            delta = end[axis] - start[axis]
            if abs(delta) <= 1e-12:
                coordinate = list(start)
                coordinate[axis] = limit
                return float(coordinate[0]), float(coordinate[1])
            factor = min(max((limit - start[axis]) / delta, 0.0), 1.0)
            point = _interpolate(start, end, factor)
            coordinate = [point[0], point[1]]
            coordinate[axis] = limit
            return float(coordinate[0]), float(coordinate[1])

        output: list[tuple[float, float]] = []
        previous = active[-1]
        previous_inside = inside(previous)
        for current in active:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current))
            previous = current
            previous_inside = current_inside
        return _clean_ring(output)

    for axis, limit, keep_greater in (
        (0, float(bounds[0]), True),
        (0, float(bounds[2]), False),
        (1, float(bounds[1]), True),
        (1, float(bounds[3]), False),
    ):
        points = clip_axis(points, axis, limit, keep_greater)
    if len(points) < 3 or abs(_ring_area(points)) <= 1e-6:
        return []
    return points


def _densify_ring(
    ring: Sequence[Sequence[float]],
    maximum_segment_length_m: float,
    sampler: TerrainMeshSampler,
) -> list[tuple[float, float]]:
    dense: list[tuple[float, float]] = []
    for start, end in zip(ring, (*ring[1:], ring[0])):
        breakpoints = sampler.segment_breakpoints(
            start,
            end,
            maximum_segment_length_m=maximum_segment_length_m,
        )
        dense.extend(breakpoints[:-1])
    return dense


def drape_building_prisms_to_tile(
    prisms: Sequence[Mapping[str, Any]],
    sampler: TerrainMeshSampler,
    tile_bounds_local: Sequence[float],
    *,
    minimum_visible_wall_height_m: float = DETAIL_BUILDING_MINIMUM_VISIBLE_WALL_M,
    maximum_boundary_segment_length_m: float = (
        DETAIL_BUILDING_MAXIMUM_BOUNDARY_SEGMENT_M
    ),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Clip/densify building walls on detail MNT and keep useful height."""

    minimum_wall = _finite(minimum_visible_wall_height_m, "minimum wall height")
    if minimum_wall <= 0.0:
        raise ValueError("minimum visible wall height must be strictly positive")
    maximum_boundary_segment = _finite(
        maximum_boundary_segment_length_m,
        "maximum building boundary segment length",
    )
    if maximum_boundary_segment <= 0.0:
        raise ValueError(
            "maximum building boundary segment length must be strictly positive"
        )
    result: list[dict[str, Any]] = []
    raised_count = 0
    maximum_raise = 0.0
    clipped_count = 0
    suppressed_intersecting_hole_count = 0
    boundary_segment_count = 0
    maximum_realized_boundary_segment = 0.0
    for prism in prisms:
        source_rings = prism.get("rings", [])
        if not source_rings or len(source_rings[0]) < 3:
            continue
        outer_ring = _clean_ring(source_rings[0])
        clipped_outer = _clip_ring_to_bounds(outer_ring, tile_bounds_local)
        if len(clipped_outer) < 3:
            continue
        if len(clipped_outer) != len(outer_ring) or any(
            not (
                float(tile_bounds_local[0]) <= point[0] <= float(tile_bounds_local[2])
                and float(tile_bounds_local[1])
                <= point[1]
                <= float(tile_bounds_local[3])
            )
            for point in outer_ring
        ):
            clipped_count += 1

        clipped_rings: list[list[tuple[float, float]]] = [clipped_outer]
        for source_hole in source_rings[1:]:
            clean_hole = _clean_ring(source_hole)
            if len(clean_hole) < 3:
                continue
            hole_is_fully_inside = all(
                float(tile_bounds_local[0]) < point[0] < float(tile_bounds_local[2])
                and float(tile_bounds_local[1]) < point[1] < float(tile_bounds_local[3])
                for point in clean_hole
            )
            if hole_is_fully_inside:
                clipped_rings.append(clean_hole)
            elif _bbox_intersects(clean_hole, tile_bounds_local):
                # A clipped interior ring would touch the new exterior and is
                # invalid for Blender tessellation. Filling this rare sliver
                # is deterministic and preferable to a broken building mesh.
                suppressed_intersecting_hole_count += 1

        dense_rings = [
            _densify_ring(ring, maximum_boundary_segment, sampler)
            for ring in clipped_rings
        ]
        serialized_rings = [
            [[round(point[0], 3), round(point[1], 3)] for point in ring]
            for ring in dense_rings
            if len(ring) >= 3
        ]
        if not serialized_rings:
            continue
        ground_rings: list[list[float]] = []
        sampled_ground: list[float] = []
        for ring in serialized_rings:
            active_ring = []
            for x, y, *_ in ring:
                ground_z = sampler.sample(float(x), float(y)) + FOUNDATION_CLEARANCE_M
                active_ring.append(round(ground_z, 3))
                sampled_ground.append(ground_z)
            ground_rings.append(active_ring)
            following = (*ring[1:], ring[0])
            for start, end in zip(ring, following):
                boundary_segment_count += 1
                maximum_realized_boundary_segment = max(
                    maximum_realized_boundary_segment, _distance(start, end)
                )
        if not sampled_ground:
            continue
        original_roof = _finite(prism.get("roof_z", 0.0), "building roof Z")
        roof_z = max(original_roof, max(sampled_ground) + minimum_wall)
        roof_raise = roof_z - original_roof
        if roof_raise > 1e-9:
            raised_count += 1
            maximum_raise = max(maximum_raise, roof_raise)
        active = dict(prism)
        active["rings"] = serialized_rings
        active["base_z"] = round(min(sampled_ground), 3)
        active["ground_z_rings"] = ground_rings
        active["roof_z"] = round(roof_z, 3)
        active["detail_roof_raise_m"] = round(roof_raise, 3)
        active["detail_minimum_visible_wall_height_m"] = minimum_wall
        result.append(active)
    return result, {
        "input_prism_count": len(prisms),
        "selected_prism_count": len(result),
        "raised_roof_prism_count": raised_count,
        "maximum_roof_raise_m": round(maximum_raise, 3),
        "minimum_visible_wall_height_m": minimum_wall,
        "tile_boundary_clipped_prism_count": clipped_count,
        "suppressed_intersecting_hole_count": suppressed_intersecting_hole_count,
        "boundary_segment_count": boundary_segment_count,
        "maximum_boundary_segment_length_m": round(
            maximum_realized_boundary_segment, 3
        ),
        "target_maximum_boundary_segment_length_m": maximum_boundary_segment,
        "foundation_grounding": "tile_detail_mnt_per_boundary_vertex",
        "terrain_surface_sampling": "fixed_nw_se_triangle_planes",
    }


def build_detail_tile_vectors(
    global_package: Mapping[str, Any],
    tile_package: Mapping[str, Any],
    tile_bounds_l93_m: Sequence[float],
    origin_l93_m: Sequence[float],
) -> dict[str, Any]:
    """Build every detailed vector layer required by one active tile."""

    sampler = TerrainMeshSampler.from_terrain_spec(tile_package["terrain"])
    bounds_local = local_tile_bounds(tile_bounds_l93_m, origin_l93_m)
    road_meshes: dict[str, Any] = {}
    road_statistics: dict[str, Any] = {}
    for layer, offset in DETAIL_ROAD_OFFSETS_M.items():
        source = (
            global_package.get("roads", {})
            .get("meshes", {})
            .get(layer, {"vertices": [], "faces": []})
        )
        road_meshes[layer], road_statistics[layer] = _drape_surface_mesh_to_tile(
            source,
            sampler,
            bounds_local,
            offset_m=offset,
            maximum_edge_length_m=DETAIL_ROAD_MAX_SEGMENT_M,
        )
    water_segment_source = (
        global_package.get("water", {})
        .get("segments", {})
        .get("mesh", {"vertices": [], "faces": []})
    )
    water_surface_source = (
        global_package.get("water", {})
        .get("surfaces", {})
        .get("mesh", {"vertices": [], "faces": []})
    )
    water_segments, water_segment_statistics = _drape_surface_mesh_to_tile(
        water_segment_source,
        sampler,
        bounds_local,
        offset_m=DETAIL_WATER_SEGMENT_OFFSET_M,
        maximum_edge_length_m=DETAIL_WATER_LINE_MAX_SEGMENT_M,
    )
    water_surfaces, water_surface_statistics = drape_triangle_mesh_to_tile(
        water_surface_source,
        sampler,
        bounds_local,
        offset_m=DETAIL_WATER_SURFACE_OFFSET_M,
        maximum_edge_length_m=DETAIL_WATER_SURFACE_MAX_EDGE_M,
    )
    building_prisms, building_statistics = drape_building_prisms_to_tile(
        global_package.get("buildings", {}).get("prisms", []),
        sampler,
        bounds_local,
    )
    return {
        "schema": "fireviewer.detail-vector-lod.v1",
        "buildings": {"prisms": building_prisms, "statistics": building_statistics},
        "roads": {"meshes": road_meshes, "statistics": road_statistics},
        "water": {
            # BD TOPO named courses are a semantic/label subset of segments in
            # this AOI. Rendering both created two coincident ribbons.
            "rendered_linear_source": "segments_only",
            "courses_rendered": False,
            "segments": {
                "mesh": water_segments,
                "statistics": water_segment_statistics,
            },
            "surfaces": {
                "mesh": water_surfaces,
                "statistics": water_surface_statistics,
            },
        },
    }


__all__ = [
    "DETAIL_BUILDING_MAXIMUM_BOUNDARY_SEGMENT_M",
    "DETAIL_BUILDING_MINIMUM_VISIBLE_WALL_M",
    "DETAIL_ROAD_MAX_SEGMENT_M",
    "DETAIL_WATER_LINE_MAX_SEGMENT_M",
    "DETAIL_WATER_SURFACE_MAX_EDGE_M",
    "TerrainMeshSampler",
    "build_detail_tile_vectors",
    "drape_building_prisms_to_tile",
    "drape_ribbon_mesh_to_tile",
    "drape_triangle_mesh_to_tile",
    "local_tile_bounds",
]
