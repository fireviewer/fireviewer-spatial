"""Partition the lightweight global terrain on the detailed-tile grid.

The global terrain must remain visible around a near-distance working set, but
it must not remain below a loaded 0.5 m terrain tile.  This module cuts the
global mesh on the exact Lambert-93 boundaries declared by the production
manifest.  A renderer can then hide only the global chunks whose tile ids are
replaced by detailed terrain.

The implementation is Blender-independent.  Vertices stay in the package's
local coordinate system, UVs stay in face-loop order, and every output
triangle records the source face from which it came.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence


GLOBAL_CONTEXT_CHUNK_ID = "__global_context_outside_manifest__"
_DEFAULT_AREA_TOLERANCE_M2 = 1e-6
_GRID_TOLERANCE_M = 1e-6
_VERTEX_KEY_DECIMALS = 9

Bounds = tuple[float, float, float, float]
Vertex = tuple[float, float, float]
Uv = tuple[float, float]
Triangle = tuple[int, int, int]


@dataclass(frozen=True)
class TerrainChunk:
    """One independently hideable part of the lightweight global terrain."""

    chunk_id: str
    tile_id: str | None
    bounds_l93_m: Bounds | None
    vertices: tuple[Vertex, ...]
    faces: tuple[Triangle, ...]
    loop_uvs: tuple[Uv, ...] | None
    source_face_indices: tuple[int, ...]

    @property
    def is_empty(self) -> bool:
        return not self.faces


@dataclass(frozen=True)
class PartitionValidation:
    """Geometric conservation evidence for a completed partition."""

    source_face_count: int
    source_triangle_count: int
    covered_source_face_count: int
    output_triangle_count: int
    populated_tile_chunk_count: int
    context_triangle_count: int
    total_source_area_m2: float
    total_partitioned_area_m2: float
    maximum_source_face_area_error_m2: float
    lost_source_face_indices: tuple[int, ...]
    duplicated_source_face_indices: tuple[int, ...]

    @property
    def is_complete(self) -> bool:
        return (
            self.covered_source_face_count == self.source_face_count
            and not self.lost_source_face_indices
            and not self.duplicated_source_face_indices
        )


@dataclass(frozen=True)
class GlobalTerrainPartition:
    """Terrain chunks and the manifest mapping used to create them."""

    origin_l93_m: tuple[float, float, float]
    manifest_tile_ids: tuple[str, ...]
    tile_chunks: Mapping[str, TerrainChunk]
    context_chunk: TerrainChunk
    validation: PartitionValidation

    @property
    def populated_tile_ids(self) -> tuple[str, ...]:
        return tuple(
            tile_id
            for tile_id in self.manifest_tile_ids
            if not self.tile_chunks[tile_id].is_empty
        )


@dataclass(frozen=True)
class _ClipPoint:
    east_m: float
    north_m: float
    local_z_m: float
    uv: Uv | None


@dataclass(frozen=True)
class _ManifestGrid:
    tile_size_m: float
    anchor_east_m: float
    anchor_north_m: float
    tiles_by_cell: Mapping[tuple[int, int], tuple[str, Bounds]]
    tile_bounds: Mapping[str, Bounds]


class _ChunkAccumulator:
    def __init__(
        self,
        chunk_id: str,
        tile_id: str | None,
        bounds_l93_m: Bounds | None,
        *,
        has_uvs: bool,
        origin_l93_m: tuple[float, float, float],
    ) -> None:
        self.chunk_id = chunk_id
        self.tile_id = tile_id
        self.bounds_l93_m = bounds_l93_m
        self.has_uvs = has_uvs
        self.origin_l93_m = origin_l93_m
        self.vertices: list[Vertex] = []
        self.faces: list[Triangle] = []
        self.loop_uvs: list[Uv] | None = [] if has_uvs else None
        self.source_face_indices: list[int] = []
        self._vertex_indices: dict[Vertex, int] = {}

    def _vertex_index(self, point: _ClipPoint) -> int:
        local = (
            point.east_m - self.origin_l93_m[0],
            point.north_m - self.origin_l93_m[1],
            point.local_z_m,
        )
        key = tuple(round(value, _VERTEX_KEY_DECIMALS) for value in local)
        existing = self._vertex_indices.get(key)
        if existing is not None:
            return existing
        index = len(self.vertices)
        self.vertices.append(local)
        self._vertex_indices[key] = index
        return index

    def add_polygon(
        self,
        polygon: Sequence[_ClipPoint],
        *,
        source_face_index: int,
        area_tolerance_m2: float,
    ) -> int:
        emitted = 0
        for index in range(1, len(polygon) - 1):
            points = (polygon[0], polygon[index], polygon[index + 1])
            if _polygon_area_xy(points) <= area_tolerance_m2:
                continue
            self.faces.append(tuple(self._vertex_index(point) for point in points))
            self.source_face_indices.append(source_face_index)
            if self.loop_uvs is not None:
                if any(point.uv is None for point in points):
                    raise AssertionError("A UV-enabled terrain fragment lost its UV")
                self.loop_uvs.extend(
                    point.uv for point in points if point.uv is not None
                )
            emitted += 1
        return emitted

    def freeze(self) -> TerrainChunk:
        return TerrainChunk(
            chunk_id=self.chunk_id,
            tile_id=self.tile_id,
            bounds_l93_m=self.bounds_l93_m,
            vertices=tuple(self.vertices),
            faces=tuple(self.faces),
            loop_uvs=None if self.loop_uvs is None else tuple(self.loop_uvs),
            source_face_indices=tuple(self.source_face_indices),
        )


def _finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _origin(value: Sequence[float]) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("origin_l93_m must contain X, Y and Z")
    if len(value) != 3:
        raise ValueError("origin_l93_m must contain X, Y and Z")
    return tuple(
        _finite(component, f"origin_l93_m[{index}]")
        for index, component in enumerate(value)
    )


def _bounds(value: Any, field_name: str) -> Bounds:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must contain west, south, east and north")
    if len(value) != 4:
        raise ValueError(f"{field_name} must contain west, south, east and north")
    west, south, east, north = tuple(
        _finite(component, f"{field_name}[{index}]")
        for index, component in enumerate(value)
    )
    if east <= west or north <= south:
        raise ValueError(f"{field_name} must have a strictly positive area")
    return west, south, east, north


def _manifest_grid(
    manifest: Mapping[str, Any],
    origin_l93_m: tuple[float, float, float],
) -> _ManifestGrid:
    if manifest.get("crs") not in (None, "EPSG:2154"):
        raise ValueError("Global terrain partition requires an EPSG:2154 manifest")
    manifest_origin = manifest.get("origin_l93_m")
    if manifest_origin is not None:
        parsed_origin = _origin(manifest_origin)
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=0.001)
            for left, right in zip(parsed_origin, origin_l93_m, strict=True)
        ):
            raise ValueError("Manifest and terrain origins differ")

    tiling = manifest.get("tiling")
    if not isinstance(tiling, Mapping):
        raise ValueError("Manifest is missing tiling")
    tile_size_m = _finite(tiling.get("output_tile_size_m"), "tiling.output_tile_size_m")
    if tile_size_m <= 0.0:
        raise ValueError("tiling.output_tile_size_m must be strictly positive")

    tiles = manifest.get("tiles")
    if not isinstance(tiles, Sequence) or isinstance(tiles, (str, bytes)) or not tiles:
        raise ValueError("Manifest tiles must be a non-empty sequence")
    first = tiles[0]
    if not isinstance(first, Mapping):
        raise ValueError("tiles[0] must be an object")
    first_bounds = _bounds(first.get("bounds_l93_m"), "tiles[0].bounds_l93_m")
    anchor_east_m, anchor_north_m = first_bounds[0], first_bounds[1]

    tiles_by_cell: dict[tuple[int, int], tuple[str, Bounds]] = {}
    tile_bounds: dict[str, Bounds] = {}
    for index, tile in enumerate(tiles):
        if not isinstance(tile, Mapping):
            raise ValueError(f"tiles[{index}] must be an object")
        identifier = tile.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"tiles[{index}].id must be a non-empty string")
        if identifier == GLOBAL_CONTEXT_CHUNK_ID:
            raise ValueError(f"Tile id {identifier!r} is reserved")
        if identifier in tile_bounds:
            raise ValueError(f"Duplicate tile id: {identifier}")
        current_bounds = _bounds(
            tile.get("bounds_l93_m"), f"tiles[{index}].bounds_l93_m"
        )
        west, south, east, north = current_bounds
        if not math.isclose(
            east - west, tile_size_m, rel_tol=0.0, abs_tol=_GRID_TOLERANCE_M
        ) or not math.isclose(
            north - south, tile_size_m, rel_tol=0.0, abs_tol=_GRID_TOLERANCE_M
        ):
            raise ValueError(
                f"Tile {identifier!r} does not match the manifest grid size"
            )
        x_index = round((west - anchor_east_m) / tile_size_m)
        y_index = round((south - anchor_north_m) / tile_size_m)
        expected = (
            anchor_east_m + x_index * tile_size_m,
            anchor_north_m + y_index * tile_size_m,
            anchor_east_m + (x_index + 1) * tile_size_m,
            anchor_north_m + (y_index + 1) * tile_size_m,
        )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=_GRID_TOLERANCE_M)
            for left, right in zip(current_bounds, expected, strict=True)
        ):
            raise ValueError(f"Tile {identifier!r} is not aligned to the manifest grid")
        cell = (x_index, y_index)
        if cell in tiles_by_cell:
            raise ValueError(f"Multiple tile ids occupy manifest grid cell {cell}")
        tiles_by_cell[cell] = (identifier, current_bounds)
        tile_bounds[identifier] = current_bounds

    return _ManifestGrid(
        tile_size_m=tile_size_m,
        anchor_east_m=anchor_east_m,
        anchor_north_m=anchor_north_m,
        tiles_by_cell=MappingProxyType(tiles_by_cell),
        tile_bounds=MappingProxyType(tile_bounds),
    )


def _terrain_vertices(terrain: Mapping[str, Any]) -> tuple[Vertex, ...]:
    raw_vertices = terrain.get("vertices")
    if not isinstance(raw_vertices, Sequence) or isinstance(raw_vertices, (str, bytes)):
        raise ValueError("Terrain vertices must be a sequence")
    vertices: list[Vertex] = []
    for index, vertex in enumerate(raw_vertices):
        if not isinstance(vertex, Sequence) or isinstance(vertex, (str, bytes)):
            raise ValueError(f"Terrain vertex {index} must contain X, Y and Z")
        if len(vertex) != 3:
            raise ValueError(f"Terrain vertex {index} must contain X, Y and Z")
        vertices.append(
            tuple(
                _finite(component, f"terrain.vertices[{index}][{axis}]")
                for axis, component in enumerate(vertex)
            )
        )
    if not vertices:
        raise ValueError("Terrain has no vertices")
    return tuple(vertices)


def _terrain_faces(
    terrain: Mapping[str, Any], vertex_count: int
) -> tuple[tuple[int, ...], ...]:
    raw_faces = terrain.get("faces")
    if not isinstance(raw_faces, Sequence) or isinstance(raw_faces, (str, bytes)):
        raise ValueError("Terrain faces must be a sequence")
    faces: list[tuple[int, ...]] = []
    for face_index, face in enumerate(raw_faces):
        if not isinstance(face, Sequence) or isinstance(face, (str, bytes)):
            raise ValueError(f"Terrain face {face_index} must contain indices")
        if len(face) not in (3, 4):
            raise ValueError(
                f"Terrain face {face_index} must be a triangle or an NW-SE quad"
            )
        indices: list[int] = []
        for corner, value in enumerate(face):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"Terrain face {face_index} corner {corner} is not an integer"
                )
            if value < 0 or value >= vertex_count:
                raise ValueError(
                    f"Terrain face {face_index} references invalid vertex {value}"
                )
            indices.append(value)
        if len(set(indices)) != len(indices):
            raise ValueError(f"Terrain face {face_index} repeats a vertex")
        faces.append(tuple(indices))
    if not faces:
        raise ValueError("Terrain has no faces")
    return tuple(faces)


def _face_loop_uvs(
    faces: Sequence[Sequence[int]], loop_uvs: Sequence[Sequence[float]] | None
) -> tuple[tuple[Uv, ...], ...] | None:
    if loop_uvs is None:
        return None
    expected_count = sum(len(face) for face in faces)
    if len(loop_uvs) != expected_count:
        raise ValueError(
            f"loop_uvs contains {len(loop_uvs)} values; expected {expected_count}"
        )
    result: list[tuple[Uv, ...]] = []
    offset = 0
    for face_index, face in enumerate(faces):
        values: list[Uv] = []
        for corner in range(len(face)):
            raw_uv = loop_uvs[offset + corner]
            if not isinstance(raw_uv, Sequence) or isinstance(raw_uv, (str, bytes)):
                raise ValueError(
                    f"loop_uvs for face {face_index} corner {corner} is invalid"
                )
            if len(raw_uv) != 2:
                raise ValueError(
                    f"loop_uvs for face {face_index} corner {corner} is invalid"
                )
            values.append(
                (
                    _finite(raw_uv[0], "loop_uvs.u"),
                    _finite(raw_uv[1], "loop_uvs.v"),
                )
            )
        result.append(tuple(values))
        offset += len(face)
    return tuple(result)


def _texture_uv(east_m: float, north_m: float, bounds: Bounds) -> Uv:
    west, south, east, north = bounds
    return (
        (east_m - west) / (east - west),
        (north_m - south) / (north - south),
    )


def _polygon_area_xy(points: Sequence[_ClipPoint]) -> float:
    if len(points) < 3:
        return 0.0
    # Subtracting the first Lambert-93 coordinate before each cross product
    # avoids cancellation between products around 10^12 for a 20 m face.
    anchor = points[0]
    return (
        abs(
            sum(
                (points[index].east_m - anchor.east_m)
                * (points[index + 1].north_m - anchor.north_m)
                - (points[index + 1].east_m - anchor.east_m)
                * (points[index].north_m - anchor.north_m)
                for index in range(1, len(points) - 1)
            )
        )
        * 0.5
    )


def _interpolate(left: _ClipPoint, right: _ClipPoint, amount: float) -> _ClipPoint:
    uv = None
    if left.uv is not None and right.uv is not None:
        uv = (
            left.uv[0] + (right.uv[0] - left.uv[0]) * amount,
            left.uv[1] + (right.uv[1] - left.uv[1]) * amount,
        )
    return _ClipPoint(
        east_m=left.east_m + (right.east_m - left.east_m) * amount,
        north_m=left.north_m + (right.north_m - left.north_m) * amount,
        local_z_m=left.local_z_m + (right.local_z_m - left.local_z_m) * amount,
        uv=uv,
    )


def _clip_half_plane(
    polygon: Sequence[_ClipPoint],
    *,
    coordinate: str,
    boundary_m: float,
    keep_greater: bool,
) -> list[_ClipPoint]:
    if not polygon:
        return []

    def value(point: _ClipPoint) -> float:
        return point.east_m if coordinate == "east" else point.north_m

    def inside(point: _ClipPoint) -> bool:
        return (
            value(point) >= boundary_m if keep_greater else value(point) <= boundary_m
        )

    result: list[_ClipPoint] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            denominator = value(current) - value(previous)
            if math.isclose(denominator, 0.0, abs_tol=1e-15):
                raise AssertionError("A clipping edge cannot cross a parallel boundary")
            amount = (boundary_m - value(previous)) / denominator
            result.append(_interpolate(previous, current, amount))
        if current_inside:
            result.append(current)
        previous = current
        previous_inside = current_inside
    return _without_duplicate_neighbours(result)


def _without_duplicate_neighbours(points: Sequence[_ClipPoint]) -> list[_ClipPoint]:
    result: list[_ClipPoint] = []
    for point in points:
        if result and _same_point(result[-1], point):
            continue
        result.append(point)
    if len(result) > 1 and _same_point(result[0], result[-1]):
        result.pop()
    return result


def _same_point(left: _ClipPoint, right: _ClipPoint) -> bool:
    return (
        math.isclose(left.east_m, right.east_m, abs_tol=1e-9)
        and math.isclose(left.north_m, right.north_m, abs_tol=1e-9)
        and math.isclose(left.local_z_m, right.local_z_m, abs_tol=1e-9)
    )


def _clip_to_bounds(polygon: Sequence[_ClipPoint], bounds: Bounds) -> list[_ClipPoint]:
    west, south, east, north = bounds
    result = _clip_half_plane(
        polygon, coordinate="east", boundary_m=west, keep_greater=True
    )
    result = _clip_half_plane(
        result, coordinate="east", boundary_m=east, keep_greater=False
    )
    result = _clip_half_plane(
        result, coordinate="north", boundary_m=south, keep_greater=True
    )
    return _clip_half_plane(
        result, coordinate="north", boundary_m=north, keep_greater=False
    )


def _face_triangles(face: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    if len(face) == 3:
        return (tuple(face),)
    # Terrain packages order quads NW, SW, SE, NE.  This is the same explicit
    # diagonal used by the rendered global terrain and its CPU drape sampler.
    northwest, southwest, southeast, northeast = face
    return (
        (northwest, southwest, southeast),
        (northwest, southeast, northeast),
    )


def _face_uv_triangles(face_uvs: Sequence[Uv]) -> tuple[tuple[Uv, Uv, Uv], ...]:
    if len(face_uvs) == 3:
        return (tuple(face_uvs),)
    northwest, southwest, southeast, northeast = face_uvs
    return (
        (northwest, southwest, southeast),
        (northwest, southeast, northeast),
    )


def _cell_bounds(grid: _ManifestGrid, x_index: int, y_index: int) -> Bounds:
    west = grid.anchor_east_m + x_index * grid.tile_size_m
    south = grid.anchor_north_m + y_index * grid.tile_size_m
    return west, south, west + grid.tile_size_m, south + grid.tile_size_m


def partition_global_terrain(
    terrain: Mapping[str, Any],
    manifest: Mapping[str, Any],
    origin_l93_m: Sequence[float],
    *,
    loop_uvs: Sequence[Sequence[float]] | None = None,
    texture_bounds_l93_m: Sequence[float] | None = None,
    area_tolerance_m2: float = _DEFAULT_AREA_TOLERANCE_M2,
) -> GlobalTerrainPartition:
    """Cut a global terrain mesh into exact manifest-grid chunks.

    ``terrain`` uses package-local XYZ coordinates.  ``loop_uvs`` is optional
    and, when supplied, follows the input face-loop order.  Alternatively,
    ``texture_bounds_l93_m`` generates north-up georeferenced UVs.  Supplying
    both UV sources is rejected.

    Faces outside a manifest tile are retained in ``context_chunk``.  This is
    important at clipped AOI edges: hiding a detailed tile never makes the
    remainder of the global terrain disappear.
    """

    parsed_origin = _origin(origin_l93_m)
    grid = _manifest_grid(manifest, parsed_origin)
    vertices = _terrain_vertices(terrain)
    faces = _terrain_faces(terrain, len(vertices))
    if loop_uvs is not None and texture_bounds_l93_m is not None:
        raise ValueError("Provide loop_uvs or texture_bounds_l93_m, not both")
    face_uvs = _face_loop_uvs(faces, loop_uvs)
    texture_bounds = (
        None
        if texture_bounds_l93_m is None
        else _bounds(texture_bounds_l93_m, "texture_bounds_l93_m")
    )
    has_uvs = face_uvs is not None or texture_bounds is not None
    tolerance = _finite(area_tolerance_m2, "area_tolerance_m2")
    if tolerance <= 0.0:
        raise ValueError("area_tolerance_m2 must be strictly positive")

    accumulators = {
        tile_id: _ChunkAccumulator(
            tile_id,
            tile_id,
            bounds,
            has_uvs=has_uvs,
            origin_l93_m=parsed_origin,
        )
        for tile_id, bounds in grid.tile_bounds.items()
    }
    context = _ChunkAccumulator(
        GLOBAL_CONTEXT_CHUNK_ID,
        None,
        None,
        has_uvs=has_uvs,
        origin_l93_m=parsed_origin,
    )

    source_face_areas = [0.0] * len(faces)
    partitioned_face_areas = [0.0] * len(faces)
    covered_faces: set[int] = set()
    source_triangle_count = 0

    for face_index, face in enumerate(faces):
        uv_triangles = (
            None if face_uvs is None else _face_uv_triangles(face_uvs[face_index])
        )
        for triangle_number, triangle in enumerate(_face_triangles(face)):
            source_triangle_count += 1
            triangle_uvs = (
                None if uv_triangles is None else uv_triangles[triangle_number]
            )
            points: list[_ClipPoint] = []
            for corner, vertex_index in enumerate(triangle):
                local_x, local_y, local_z = vertices[vertex_index]
                east_m = parsed_origin[0] + local_x
                north_m = parsed_origin[1] + local_y
                uv = None
                if triangle_uvs is not None:
                    uv = triangle_uvs[corner]
                elif texture_bounds is not None:
                    uv = _texture_uv(east_m, north_m, texture_bounds)
                points.append(_ClipPoint(east_m, north_m, local_z, uv))

            source_area = _polygon_area_xy(points)
            if source_area <= tolerance:
                raise ValueError(
                    f"Terrain face {face_index} contains a degenerate triangle"
                )
            source_face_areas[face_index] += source_area
            minimum_east = min(point.east_m for point in points)
            maximum_east = max(point.east_m for point in points)
            minimum_north = min(point.north_m for point in points)
            maximum_north = max(point.north_m for point in points)
            minimum_x = math.floor(
                (minimum_east - grid.anchor_east_m) / grid.tile_size_m
            )
            maximum_x = math.floor(
                (maximum_east - grid.anchor_east_m) / grid.tile_size_m
            )
            minimum_y = math.floor(
                (minimum_north - grid.anchor_north_m) / grid.tile_size_m
            )
            maximum_y = math.floor(
                (maximum_north - grid.anchor_north_m) / grid.tile_size_m
            )

            triangle_partitioned_area = 0.0
            for x_index in range(minimum_x, maximum_x + 1):
                for y_index in range(minimum_y, maximum_y + 1):
                    cell = (x_index, y_index)
                    cell_bounds = _cell_bounds(grid, x_index, y_index)
                    clipped = _clip_to_bounds(points, cell_bounds)
                    clipped_area = (
                        _polygon_area_xy(clipped) if len(clipped) >= 3 else 0.0
                    )
                    if clipped_area <= tolerance:
                        continue
                    tile = grid.tiles_by_cell.get(cell)
                    accumulator = context if tile is None else accumulators[tile[0]]
                    emitted = accumulator.add_polygon(
                        clipped,
                        source_face_index=face_index,
                        area_tolerance_m2=tolerance,
                    )
                    if emitted == 0:
                        raise AssertionError(
                            "A positive terrain fragment emitted no triangle"
                        )
                    triangle_partitioned_area += clipped_area
                    covered_faces.add(face_index)
            partitioned_face_areas[face_index] += triangle_partitioned_area

    lost: list[int] = []
    duplicated: list[int] = []
    maximum_error = 0.0
    for face_index, (source_area, partitioned_area) in enumerate(
        zip(source_face_areas, partitioned_face_areas, strict=True)
    ):
        error = partitioned_area - source_area
        maximum_error = max(maximum_error, abs(error))
        allowed = max(tolerance, source_area * 1e-9)
        if error < -allowed:
            lost.append(face_index)
        elif error > allowed:
            duplicated.append(face_index)

    tile_chunks = {
        tile_id: accumulators[tile_id].freeze() for tile_id in sorted(accumulators)
    }
    context_chunk = context.freeze()
    validation = PartitionValidation(
        source_face_count=len(faces),
        source_triangle_count=source_triangle_count,
        covered_source_face_count=len(covered_faces),
        output_triangle_count=sum(len(chunk.faces) for chunk in tile_chunks.values())
        + len(context_chunk.faces),
        populated_tile_chunk_count=sum(
            1 for chunk in tile_chunks.values() if not chunk.is_empty
        ),
        context_triangle_count=len(context_chunk.faces),
        total_source_area_m2=sum(source_face_areas),
        total_partitioned_area_m2=sum(partitioned_face_areas),
        maximum_source_face_area_error_m2=maximum_error,
        lost_source_face_indices=tuple(lost),
        duplicated_source_face_indices=tuple(duplicated),
    )
    partition = GlobalTerrainPartition(
        origin_l93_m=parsed_origin,
        manifest_tile_ids=tuple(sorted(tile_chunks)),
        tile_chunks=MappingProxyType(tile_chunks),
        context_chunk=context_chunk,
        validation=validation,
    )
    validate_global_terrain_partition(partition, area_tolerance_m2=tolerance)
    return partition


def validate_global_terrain_partition(
    partition: GlobalTerrainPartition,
    *,
    area_tolerance_m2: float = _DEFAULT_AREA_TOLERANCE_M2,
) -> PartitionValidation:
    """Fail closed if chunks cannot prove complete, exclusive coverage."""

    tolerance = _finite(area_tolerance_m2, "area_tolerance_m2")
    if tolerance <= 0.0:
        raise ValueError("area_tolerance_m2 must be strictly positive")
    validation = partition.validation
    if not validation.is_complete:
        raise ValueError(
            "Global terrain partition is incomplete or duplicated: "
            f"lost={validation.lost_source_face_indices}, "
            f"duplicated={validation.duplicated_source_face_indices}"
        )
    total_error = abs(
        validation.total_partitioned_area_m2 - validation.total_source_area_m2
    )
    allowed_total_error = max(
        tolerance * max(1, validation.source_face_count),
        validation.total_source_area_m2 * 1e-9,
    )
    if total_error > allowed_total_error:
        raise ValueError("Global terrain partition does not conserve total XY area")
    if set(partition.tile_chunks) != set(partition.manifest_tile_ids):
        raise ValueError("Terrain partition tile mapping differs from the manifest")

    chunks = (*partition.tile_chunks.values(), partition.context_chunk)
    output_triangle_count = 0
    for chunk in chunks:
        if len(chunk.source_face_indices) != len(chunk.faces):
            raise ValueError(f"Chunk {chunk.chunk_id!r} lost source-face provenance")
        if chunk.loop_uvs is not None and len(chunk.loop_uvs) != 3 * len(chunk.faces):
            raise ValueError(f"Chunk {chunk.chunk_id!r} has misaligned loop UVs")
        for face in chunk.faces:
            if len(face) != 3 or any(
                index < 0 or index >= len(chunk.vertices) for index in face
            ):
                raise ValueError(f"Chunk {chunk.chunk_id!r} has an invalid triangle")
        if chunk.bounds_l93_m is not None:
            west, south, east, north = chunk.bounds_l93_m
            for local_x, local_y, _local_z in chunk.vertices:
                absolute_x = partition.origin_l93_m[0] + local_x
                absolute_y = partition.origin_l93_m[1] + local_y
                if not (
                    west - _GRID_TOLERANCE_M <= absolute_x <= east + _GRID_TOLERANCE_M
                    and south - _GRID_TOLERANCE_M
                    <= absolute_y
                    <= north + _GRID_TOLERANCE_M
                ):
                    raise ValueError(
                        f"Chunk {chunk.chunk_id!r} contains a vertex outside its bounds"
                    )
        output_triangle_count += len(chunk.faces)
    if output_triangle_count != validation.output_triangle_count:
        raise ValueError("Partition triangle statistics differ from chunk geometry")
    return validation


def global_context_visibility(
    partition: GlobalTerrainPartition,
    detailed_tile_ids: Sequence[str],
) -> Mapping[str, bool]:
    """Return chunk visibility with exact HD-tile exclusion.

    The residual context chunk is always visible.  Each populated global tile
    chunk is visible unless a detailed tile with the same manifest id replaces
    it.  Unknown detail ids fail closed instead of hiding unrelated terrain.
    """

    requested = {str(tile_id) for tile_id in detailed_tile_ids}
    unknown = sorted(requested.difference(partition.manifest_tile_ids))
    if unknown:
        raise ValueError(f"Unknown detailed tile ids: {unknown}")
    visibility = {
        tile_id: not chunk.is_empty and tile_id not in requested
        for tile_id, chunk in partition.tile_chunks.items()
    }
    visibility[GLOBAL_CONTEXT_CHUNK_ID] = not partition.context_chunk.is_empty
    return MappingProxyType(visibility)


__all__ = [
    "GLOBAL_CONTEXT_CHUNK_ID",
    "GlobalTerrainPartition",
    "PartitionValidation",
    "TerrainChunk",
    "global_context_visibility",
    "partition_global_terrain",
    "validate_global_terrain_partition",
]
