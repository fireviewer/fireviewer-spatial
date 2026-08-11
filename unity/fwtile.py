"""Versioned, deterministic and corruption-detecting FireViewer tile container.

The container is deliberately platform-neutral.  Coordinates remain metric
Lambert-93 (east, north, up); the Unity adapter decides how to map those axes
into a scene.  Large logical sections are compressed independently so a
runtime can validate and decode only what it needs.
"""

from __future__ import annotations

from array import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any, Iterable, Mapping, Sequence
import zlib


MAGIC = b"FWTILE1\0"
VERSION_MAJOR = 1
VERSION_MINOR = 0
PREFIX = struct.Struct("<8sHHI")
TREE_RECORD = struct.Struct("<IIiHHBH")
VECTOR_VERTEX = struct.Struct("<HHH")
VECTOR_INDEX_U16 = struct.Struct("<H")
VECTOR_INDEX = struct.Struct("<I")
MAX_HEADER_BYTES = 16 * 1024 * 1024


class FWTileError(ValueError):
    """Raised when input geometry or a container violates the contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _little_endian_u16(values: Iterable[int]) -> bytes:
    result = array("H", values)
    if result.itemsize != 2:
        raise FWTileError("this Python runtime has no 16-bit unsigned short")
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FWTileError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise FWTileError(f"{label} is not finite")
    return result


def _quantize_u16(values: Sequence[float]) -> tuple[bytes, dict[str, Any]]:
    if not values:
        raise FWTileError("cannot quantize an empty elevation sequence")
    minimum = min(values)
    maximum = max(values)
    step = (maximum - minimum) / 65535.0 if maximum > minimum else 0.0
    if step == 0.0:
        encoded = [0] * len(values)
        maximum_error = 0.0
    else:
        encoded = [
            max(0, min(65535, int(round((value - minimum) / step)))) for value in values
        ]
        maximum_error = max(
            abs(value - (minimum + quantized * step))
            for value, quantized in zip(values, encoded, strict=True)
        )
    return _little_endian_u16(encoded), {
        "component_type": "UInt16LE",
        "minimum_m": minimum,
        "maximum_m": maximum,
        "step_m": step,
        "maximum_observed_error_m": maximum_error,
    }


def encode_detail_terrain(
    terrain: Mapping[str, Any],
    bounds_l93_m: Sequence[float],
    origin_l93_m: Sequence[float],
) -> tuple[bytes, dict[str, Any]]:
    """Encode the exact regular tile grid as implicit XY plus UInt16 Z."""

    if len(bounds_l93_m) != 4 or len(origin_l93_m) != 3:
        raise FWTileError("terrain bounds/origin dimensions are invalid")
    tile_xmin, tile_ymin, tile_xmax, tile_ymax = map(float, bounds_l93_m)
    spacing = terrain.get("sample_spacing_m")
    if not isinstance(spacing, Sequence) or len(spacing) != 2:
        raise FWTileError("terrain has no two-dimensional sample spacing")
    spacing_x = _finite_float(spacing[0], "terrain spacing X")
    spacing_y = _finite_float(spacing[1], "terrain spacing Y")
    if spacing_x <= 0.0 or spacing_y <= 0.0:
        raise FWTileError("terrain sample spacing must be positive")
    vertices = terrain.get("vertices")
    if not isinstance(vertices, list) or len(vertices) < 4:
        raise FWTileError("terrain has no usable regular vertex grid")
    first_y = _finite_float(vertices[0][1], "first terrain Y")
    columns = 0
    for vertex in vertices:
        if abs(_finite_float(vertex[1], "terrain Y") - first_y) > 1e-7:
            break
        columns += 1
    if columns < 2 or len(vertices) % columns:
        raise FWTileError("terrain vertices are not a rectangular row-major grid")
    rows = len(vertices) // columns
    if rows < 2:
        raise FWTileError("terrain grid has fewer than two rows")
    faces = terrain.get("faces")
    if not isinstance(faces, list) or len(faces) != (rows - 1) * (columns - 1):
        raise FWTileError("terrain face count is not a complete regular grid")
    origin_x, origin_y, _origin_z = map(float, origin_l93_m)
    actual_bounds = [
        origin_x + _finite_float(vertices[0][0], "terrain west X"),
        origin_y + _finite_float(vertices[(rows - 1) * columns][1], "terrain south Y"),
        origin_x + _finite_float(vertices[columns - 1][0], "terrain east X"),
        origin_y + first_y,
    ]
    if (
        actual_bounds[0] < tile_xmin - 0.002
        or actual_bounds[1] < tile_ymin - 0.002
        or actual_bounds[2] > tile_xmax + 0.002
        or actual_bounds[3] > tile_ymax + 0.002
    ):
        raise FWTileError("terrain grid extends outside its owning tile")
    geometric_bounds = terrain.get("geometric_bounds_l93_m", actual_bounds)
    if len(geometric_bounds) != 4 or any(
        abs(float(left) - float(right)) > 0.002
        for left, right in zip(geometric_bounds, actual_bounds, strict=True)
    ):
        raise FWTileError("terrain geometric bounds do not match its actual grid")
    face_index = 0
    for row in range(rows - 1):
        for column in range(columns - 1):
            northwest = row * columns + column
            expected_face = [
                northwest,
                northwest + 1,
                northwest + columns + 1,
                northwest + columns,
            ]
            if [int(value) for value in faces[face_index]] != expected_face:
                raise FWTileError(
                    f"terrain face {face_index} breaks the fixed grid topology"
                )
            face_index += 1
    elevations: list[float] = []
    tolerance_m = 0.002
    for index, vertex in enumerate(vertices):
        if not isinstance(vertex, Sequence) or len(vertex) < 3:
            raise FWTileError(f"terrain vertex {index} is invalid")
        row, column = divmod(index, columns)
        expected_x = actual_bounds[0] - origin_x + column * spacing_x
        expected_y = actual_bounds[3] - origin_y - row * spacing_y
        actual_x = _finite_float(vertex[0], f"terrain vertex {index} X")
        actual_y = _finite_float(vertex[1], f"terrain vertex {index} Y")
        if (
            abs(actual_x - expected_x) > tolerance_m
            or abs(actual_y - expected_y) > tolerance_m
        ):
            raise FWTileError(
                f"terrain vertex {index} breaks the regular Lambert-93 grid"
            )
        elevations.append(_finite_float(vertex[2], f"terrain vertex {index} Z"))
    encoded, quantization = _quantize_u16(elevations)
    return encoded, {
        "encoding": "regular-grid-z-u16.v1",
        "rows": rows,
        "columns": columns,
        "sample_spacing_m": [spacing_x, spacing_y],
        "geometric_bounds_l93_m": actual_bounds,
        "row_order": "north_to_south",
        "column_order": "west_to_east",
        "triangulation": "fixed_nw_se_diagonal",
        "vertex_count": len(vertices),
        "triangle_count": 2 * (rows - 1) * (columns - 1),
        "elevation_quantization": quantization,
        "source_pixel_size_m": terrain.get("source_pixel_size_m"),
        "boundary_sampling": terrain.get("boundary_sampling"),
        "adjacent_edge_contract": terrain.get("adjacent_edge_contract"),
    }


def encode_far_grid(
    elevations: Sequence[float],
    validity: Sequence[bool],
    *,
    rows: int,
    columns: int,
    pixel_size_m: Sequence[float],
    outer_bounds_l93_m: Sequence[float],
) -> tuple[bytes, dict[str, Any]]:
    """Encode a rectangular far MNT plus a bit mask for AOI/nodata cells."""

    if rows <= 0 or columns <= 0 or len(elevations) != rows * columns:
        raise FWTileError("far terrain grid dimensions are inconsistent")
    if len(validity) != len(elevations):
        raise FWTileError("far terrain validity mask length is inconsistent")
    valid_values = [
        _finite_float(value, f"far elevation {index}")
        for index, (value, valid) in enumerate(zip(elevations, validity, strict=True))
        if valid
    ]
    if not valid_values:
        raise FWTileError("far terrain contains no valid elevations")
    valid_raw, quantization = _quantize_u16(valid_values)
    valid_encoded = array("H")
    valid_encoded.frombytes(valid_raw)
    if sys.byteorder != "little":
        valid_encoded.byteswap()
    iterator = iter(valid_encoded)
    encoded_grid = [next(iterator) if valid else 0 for valid in validity]
    terrain_raw = _little_endian_u16(encoded_grid)
    mask = bytearray((len(validity) + 7) // 8)
    for index, valid in enumerate(validity):
        if valid:
            mask[index // 8] |= 1 << (index % 8)
    pixel_x = _finite_float(pixel_size_m[0], "far pixel size X")
    pixel_y = _finite_float(pixel_size_m[1], "far pixel size Y")
    if pixel_x <= 0.0 or pixel_y <= 0.0:
        raise FWTileError("far terrain pixel size must be positive")
    return terrain_raw + mask, {
        "encoding": "masked-regular-grid-z-u16.v1",
        "rows": rows,
        "columns": columns,
        "sample_spacing_m": [pixel_x, pixel_y],
        "outer_bounds_l93_m": list(map(float, outer_bounds_l93_m)),
        "sample_centres": True,
        "row_order": "north_to_south",
        "column_order": "west_to_east",
        "triangulation": "skip_triangles_touching_invalid_samples",
        "elevation_bytes": len(terrain_raw),
        "validity_mask_offset_bytes": len(terrain_raw),
        "validity_mask_bytes": len(mask),
        "validity_mask_bit_order": "least_significant_bit_first",
        "valid_sample_count": len(valid_values),
        "elevation_quantization": quantization,
    }


def encode_tree_instances(
    package: Mapping[str, Any],
    bounds_l93_m: Sequence[float],
    origin_l93_m: Sequence[float],
) -> tuple[bytes, dict[str, Any]]:
    """Encode every detected tree; no density thinning is performed here."""

    attributes = package.get("instances", {}).get("attributes")
    values = package.get("instances", {}).get("values")
    expected = [
        "local_x_m",
        "local_y_m",
        "local_ground_z_m",
        "height_m",
        "crown_diameter_m",
        "visual_variant",
        "rotation_degrees",
    ]
    if attributes != expected or not isinstance(values, list):
        raise FWTileError("tree instance attributes do not match the v1 contract")
    accepted_count = package.get("statistics", {}).get("accepted_instance_count")
    if int(accepted_count) != len(values):
        raise FWTileError("tree instance list is incomplete")
    xmin, ymin, xmax, ymax = map(float, bounds_l93_m)
    origin_x, origin_y, origin_z = map(float, origin_l93_m)
    output = bytearray()
    for index, record in enumerate(values):
        if not isinstance(record, Sequence) or len(record) != 7:
            raise FWTileError(f"tree instance {index} is malformed")
        local_x, local_y, ground_z, height, crown, variant, rotation = record
        east = origin_x + _finite_float(local_x, f"tree {index} X")
        north = origin_y + _finite_float(local_y, f"tree {index} Y")
        if not (xmin <= east <= xmax and ymin <= north <= ymax):
            raise FWTileError(f"tree instance {index} is outside its owning tile")
        x_mm = int(round((east - xmin) * 1000.0))
        y_mm = int(round((north - ymin) * 1000.0))
        ground_mm = int(round(_finite_float(ground_z, f"tree {index} ground") * 1000))
        # UInt16 millimetres tops out at 65.535 m.  The HD LiDAR packages do
        # contain legitimate or retained source outliers above that limit, so
        # dimensions use centimetres in v2.  Positions and ground elevation
        # remain millimetric; no instance is clamped or discarded.
        height_cm = int(round(_finite_float(height, f"tree {index} height") * 100))
        crown_cm = int(round(_finite_float(crown, f"tree {index} crown") * 100))
        variant_i = int(variant)
        rotation_centidegrees = (
            int(
                round((_finite_float(rotation, f"tree {index} rotation") % 360.0) * 100)
            )
            % 36000
        )
        if not (0 <= x_mm <= 0xFFFFFFFF and 0 <= y_mm <= 0xFFFFFFFF):
            raise FWTileError(f"tree instance {index} position overflows UInt32")
        if not (-0x80000000 <= ground_mm <= 0x7FFFFFFF):
            raise FWTileError(f"tree instance {index} ground overflows Int32")
        if not (0 < height_cm <= 0xFFFF and 0 < crown_cm <= 0xFFFF):
            raise FWTileError(f"tree instance {index} dimensions overflow UInt16")
        if not 0 <= variant_i <= 0xFF:
            raise FWTileError(f"tree instance {index} variant overflows UInt8")
        output.extend(
            TREE_RECORD.pack(
                x_mm,
                y_mm,
                ground_mm,
                height_cm,
                crown_cm,
                variant_i,
                rotation_centidegrees,
            )
        )
    tree_contract = package.get("tree_instances", {})
    return bytes(output), {
        "encoding": "tree-instance-position-mm-dimension-cm.v2",
        "record_stride_bytes": TREE_RECORD.size,
        "count": len(values),
        "position_origin_l93_m": [xmin, ymin, origin_z],
        "fields": [
            ["east_offset_mm", "UInt32LE"],
            ["north_offset_mm", "UInt32LE"],
            ["ground_up_offset_mm", "Int32LE"],
            ["height_cm", "UInt16LE"],
            ["crown_diameter_cm", "UInt16LE"],
            ["visual_variant", "UInt8"],
            ["rotation_centidegrees", "UInt16LE"],
        ],
        "prototypes": tree_contract.get("prototypes", []),
        "material_slots": tree_contract.get("material_slots", []),
        "completeness_claim": package.get("statistics", {}).get("completeness_claim"),
        "thinning": "none_in_exporter",
    }


def triangulate_faces(faces: Iterable[Sequence[int]]) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    for face_index, face in enumerate(faces):
        indices = [int(value) for value in face]
        if len(indices) < 3:
            raise FWTileError(f"mesh face {face_index} has fewer than three indices")
        for offset in range(1, len(indices) - 1):
            triangle = (indices[0], indices[offset], indices[offset + 1])
            if len(set(triangle)) != 3:
                raise FWTileError(
                    f"mesh face {face_index} contains a degenerate triangle"
                )
            triangles.append(triangle)
    return triangles


def prism_to_mesh(prism: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one grounded building prism into deterministic wall/roof triangles."""

    try:
        from shapely import make_valid
        from shapely.geometry import Polygon
        from shapely.ops import triangulate
    except ImportError as exc:  # pragma: no cover - dependency is in repo requirements
        raise FWTileError("Shapely is required to triangulate building roofs") from exc

    rings = prism.get("rings")
    if not isinstance(rings, list) or not rings or len(rings[0]) < 3:
        raise FWTileError("building prism has no valid exterior ring")
    ground_rings = prism.get("ground_z_rings")
    base_z = _finite_float(prism.get("base_z"), "building base Z")
    top_z = _finite_float(
        prism.get("roof_z", base_z + _finite_float(prism.get("height"), "height")),
        "building roof Z",
    )
    clean_rings: list[list[tuple[float, float]]] = []
    clean_ground_rings: list[list[float]] = []
    for ring_index, ring in enumerate(rings):
        clean = [
            (
                _finite_float(point[0], "building X"),
                _finite_float(point[1], "building Y"),
            )
            for point in ring
        ]
        if len(clean) > 1 and clean[0] == clean[-1]:
            clean.pop()
        if len(clean) < 3:
            raise FWTileError("building prism contains a short ring")
        clean_rings.append(clean)
        ground_values = (
            ground_rings[ring_index]
            if isinstance(ground_rings, list) and ring_index < len(ground_rings)
            else None
        )
        clean_ground_rings.append(
            [
                _finite_float(ground_values[point_index], "building ground Z")
                if isinstance(ground_values, list) and point_index < len(ground_values)
                else base_z
                for point_index in range(len(clean))
            ]
        )

    polygon = Polygon(clean_rings[0], clean_rings[1:])
    if polygon.is_empty:
        raise FWTileError("building roof polygon is invalid")
    repaired = (
        polygon if polygon.is_valid and polygon.area > 0.0 else make_valid(polygon)
    )

    def polygon_parts(geometry: Any) -> list[Any]:
        if geometry.geom_type == "Polygon":
            return [geometry] if geometry.area > 1e-10 else []
        result: list[Any] = []
        for child in getattr(geometry, "geoms", ()):  # MultiPolygon/collection
            result.extend(polygon_parts(child))
        return result

    parts = polygon_parts(repaired)
    if not parts:
        raise FWTileError("building roof repair produced no polygon")
    parts.sort(
        key=lambda item: (
            tuple(round(value, 6) for value in item.bounds),
            round(item.area, 9),
        )
    )

    source_segments: list[tuple[float, float, float, float, float, float]] = []
    for ring, grounds in zip(clean_rings, clean_ground_rings, strict=True):
        for index, (start_x, start_y) in enumerate(ring):
            following = (index + 1) % len(ring)
            end_x, end_y = ring[following]
            source_segments.append(
                (
                    start_x,
                    start_y,
                    end_x,
                    end_y,
                    grounds[index],
                    grounds[following],
                )
            )

    def ground_at(x: float, y: float) -> float:
        candidates: list[float] = []
        for start_x, start_y, end_x, end_y, start_z, end_z in source_segments:
            delta_x, delta_y = end_x - start_x, end_y - start_y
            squared_length = delta_x * delta_x + delta_y * delta_y
            if squared_length <= 1e-16:
                continue
            factor = (
                (x - start_x) * delta_x + (y - start_y) * delta_y
            ) / squared_length
            if not -1e-7 <= factor <= 1.0 + 1e-7:
                continue
            projected_x = start_x + factor * delta_x
            projected_y = start_y + factor * delta_y
            if math.hypot(projected_x - x, projected_y - y) <= 2e-6:
                candidates.append(start_z + factor * (end_z - start_z))
        if not candidates:
            raise FWTileError("repaired building vertex is not on its source boundary")
        return max(candidates)

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    top_lookup: dict[tuple[float, float], int] = {}
    for active_polygon in parts:
        active_rings = [active_polygon.exterior, *active_polygon.interiors]
        for active_ring in active_rings:
            coordinates = [
                (float(x), float(y)) for x, y in list(active_ring.coords)[:-1]
            ]
            bottoms: list[int] = []
            tops: list[int] = []
            for x, y in coordinates:
                bottoms.append(len(vertices))
                vertices.append((x, y, ground_at(x, y)))
                tops.append(len(vertices))
                vertices.append((x, y, top_z))
                top_lookup[(round(x, 6), round(y, 6))] = tops[-1]
            for index in range(len(coordinates)):
                following = (index + 1) % len(coordinates)
                faces.append((bottoms[index], bottoms[following], tops[following]))
                faces.append((bottoms[index], tops[following], tops[index]))

        roof_triangles = [
            item
            for item in triangulate(active_polygon)
            if active_polygon.covers(item) and item.area > 1e-10
        ]
        roof_triangles.sort(
            key=lambda item: tuple(
                sorted(
                    (round(x, 6), round(y, 6))
                    for x, y in list(item.exterior.coords)[:-1]
                )
            )
        )
        for triangle in roof_triangles:
            indices: list[int] = []
            for x, y in list(triangle.exterior.coords)[:-1]:
                key = (round(float(x), 6), round(float(y), 6))
                index = top_lookup.get(key)
                if index is None:
                    index = len(vertices)
                    vertices.append((float(x), float(y), top_z))
                    top_lookup[key] = index
                indices.append(index)
            a, b, c = (vertices[index] for index in indices)
            signed_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if signed_area < 0.0:
                indices.reverse()
            faces.append(tuple(indices))
    return {
        "name": f"building/{prism.get('feature_id', 'unknown')}",
        "vertices": vertices,
        "faces": faces,
    }


def encode_mesh_section(
    meshes: Sequence[Mapping[str, Any]],
    bounds_l93_m: Sequence[float],
    origin_l93_m: Sequence[float],
) -> tuple[bytes, dict[str, Any]]:
    """Encode named meshes with bounded sub-centimetre position error."""

    xmin, ymin, xmax, ymax = map(float, bounds_l93_m)
    origin_x, origin_y, origin_z = map(float, origin_l93_m)
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0.0 or height <= 0.0:
        raise FWTileError("mesh section tile bounds are invalid")
    all_up = [
        _finite_float(vertex[2], f"{mesh.get('name', 'mesh')} vertex Z")
        for mesh in meshes
        for vertex in mesh.get("vertices", [])
    ]
    minimum_up = min(all_up) if all_up else origin_z
    maximum_up = max(all_up) if all_up else origin_z
    up_step = (maximum_up - minimum_up) / 65535.0 if maximum_up > minimum_up else 0.0
    raw = bytearray()
    descriptors: list[dict[str, Any]] = []
    maximum_xy_error = 0.0
    maximum_up_error = 0.0
    for mesh in meshes:
        name = str(mesh.get("name", "mesh"))
        vertices = mesh.get("vertices", [])
        faces = mesh.get("faces", [])
        if not isinstance(vertices, list) or not isinstance(faces, list):
            raise FWTileError(f"mesh {name!r} has invalid arrays")
        vertex_offset = len(raw)
        for vertex_index, vertex in enumerate(vertices):
            if not isinstance(vertex, Sequence) or len(vertex) < 3:
                raise FWTileError(f"mesh {name!r} vertex {vertex_index} is invalid")
            east = origin_x + _finite_float(vertex[0], f"{name} vertex X")
            north = origin_y + _finite_float(vertex[1], f"{name} vertex Y")
            up = _finite_float(vertex[2], f"{name} vertex Z")
            if not (
                xmin - 0.005 <= east <= xmax + 0.005
                and ymin - 0.005 <= north <= ymax + 0.005
            ):
                raise FWTileError(
                    f"mesh {name!r} vertex {vertex_index} lies outside tile bounds"
                )
            encoded_x = max(0, min(65535, int(round((east - xmin) / width * 65535.0))))
            encoded_y = max(
                0, min(65535, int(round((north - ymin) / height * 65535.0)))
            )
            encoded_z = (
                max(
                    0,
                    min(65535, int(round((up - minimum_up) / up_step))),
                )
                if up_step > 0.0
                else 0
            )
            decoded_x = xmin + encoded_x * width / 65535.0
            decoded_y = ymin + encoded_y * height / 65535.0
            decoded_z = minimum_up + encoded_z * up_step
            maximum_xy_error = max(
                maximum_xy_error, abs(decoded_x - east), abs(decoded_y - north)
            )
            maximum_up_error = max(maximum_up_error, abs(decoded_z - up))
            raw.extend(VECTOR_VERTEX.pack(encoded_x, encoded_y, encoded_z))
        triangles = triangulate_faces(faces)
        index_offset = len(raw)
        index_struct = VECTOR_INDEX_U16 if len(vertices) <= 0xFFFF else VECTOR_INDEX
        for triangle in triangles:
            for index in triangle:
                if not 0 <= index < len(vertices):
                    raise FWTileError(f"mesh {name!r} has an out-of-range index")
                raw.extend(index_struct.pack(index))
        descriptors.append(
            {
                "name": name,
                "vertex_count": len(vertices),
                "triangle_count": len(triangles),
                "vertex_offset_bytes": vertex_offset,
                "index_offset_bytes": index_offset,
                "end_offset_bytes": len(raw),
                "index_component_type": (
                    "UInt16LE" if index_struct is VECTOR_INDEX_U16 else "UInt32LE"
                ),
                "index_stride_bytes": index_struct.size,
            }
        )
    return bytes(raw), {
        "encoding": "mesh-position-u16-quantized-index-adaptive.v1",
        "position_quantization": {
            "component_type": "UInt16LE",
            "east_minimum_m": xmin,
            "east_step_m": width / 65535.0,
            "north_minimum_m": ymin,
            "north_step_m": height / 65535.0,
            "up_minimum_m": minimum_up,
            "up_step_m": up_step,
            "maximum_observed_horizontal_error_m": maximum_xy_error,
            "maximum_observed_vertical_error_m": maximum_up_error,
        },
        "vertex_stride_bytes": VECTOR_VERTEX.size,
        "meshes": descriptors,
        "mesh_count": len(descriptors),
        "vertex_count": sum(item["vertex_count"] for item in descriptors),
        "triangle_count": sum(item["triangle_count"] for item in descriptors),
    }


def build_vector_sections(
    detail_vectors: Mapping[str, Any],
    bounds_l93_m: Sequence[float],
    origin_l93_m: Sequence[float],
) -> dict[str, tuple[bytes, dict[str, Any]]]:
    buildings = [
        prism_to_mesh(prism) for prism in detail_vectors["buildings"]["prisms"]
    ]
    roads = [
        {"name": f"road/{name}", **mesh}
        for name, mesh in sorted(detail_vectors["roads"]["meshes"].items())
        if mesh.get("vertices") or mesh.get("faces")
    ]
    water = []
    for name in ("segments", "surfaces"):
        mesh = detail_vectors["water"][name]["mesh"]
        if mesh.get("vertices") or mesh.get("faces"):
            water.append({"name": f"water/{name}", **mesh})
    return {
        "buildings": encode_mesh_section(buildings, bounds_l93_m, origin_l93_m),
        "roads": encode_mesh_section(roads, bounds_l93_m, origin_l93_m),
        "water": encode_mesh_section(water, bounds_l93_m, origin_l93_m),
    }


def build_container(
    *,
    kind: str,
    tile_id: str,
    bounds_l93_m: Sequence[float],
    origin_l93_m: Sequence[float],
    sections: Sequence[tuple[str, bytes, Mapping[str, Any]]],
    metadata: Mapping[str, Any] | None = None,
) -> bytes:
    if not sections:
        raise FWTileError("a tile container must have at least one section")
    body = bytearray()
    section_headers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, raw, section_metadata in sections:
        if name in seen:
            raise FWTileError(f"duplicate section {name!r}")
        seen.add(name)
        compressed = zlib.compress(raw, level=9)
        section_headers.append(
            {
                "name": name,
                "codec": "zlib",
                "offset_bytes": len(body),
                "stored_bytes": len(compressed),
                "raw_bytes": len(raw),
                "stored_sha256": sha256_bytes(compressed),
                "raw_sha256": sha256_bytes(raw),
                "metadata": dict(section_metadata),
            }
        )
        body.extend(compressed)
    header = {
        "schema": "fireviewer.fwtile.v1",
        "kind": kind,
        "tile_id": tile_id,
        "crs": "EPSG:2154",
        "linear_unit": "metre",
        "axis_convention": "X=east,Y=north,Z=up",
        "bounds_l93_m": list(map(float, bounds_l93_m)),
        "origin_l93_m": list(map(float, origin_l93_m)),
        "sections": section_headers,
        "metadata": dict(metadata or {}),
    }
    header_bytes = canonical_json_bytes(header)
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise FWTileError("container header is too large")
    return (
        PREFIX.pack(MAGIC, VERSION_MAJOR, VERSION_MINOR, len(header_bytes))
        + header_bytes
        + body
    )


def _safe_decompress(value: bytes, expected_size: int) -> bytes:
    decompressor = zlib.decompressobj()
    result = decompressor.decompress(value, expected_size + 1)
    if len(result) > expected_size or decompressor.unconsumed_tail:
        raise FWTileError("compressed section expands beyond its declared length")
    result += decompressor.flush(expected_size - len(result) + 1)
    if (
        len(result) != expected_size
        or decompressor.unconsumed_tail
        or not decompressor.eof
    ):
        raise FWTileError("compressed section length is invalid")
    return result


def read_container(value: bytes, *, decode_sections: bool = True) -> dict[str, Any]:
    if len(value) < PREFIX.size:
        raise FWTileError("container is truncated before its prefix")
    magic, major, minor, header_length = PREFIX.unpack_from(value)
    if magic != MAGIC or (major, minor) != (VERSION_MAJOR, VERSION_MINOR):
        raise FWTileError("container magic/version is unsupported")
    if header_length <= 0 or header_length > MAX_HEADER_BYTES:
        raise FWTileError("container header length is invalid")
    header_end = PREFIX.size + header_length
    if header_end > len(value):
        raise FWTileError("container is truncated inside its header")
    try:
        header = json.loads(value[PREFIX.size : header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FWTileError("container header JSON is invalid") from exc
    if header.get("schema") != "fireviewer.fwtile.v1":
        raise FWTileError("container header schema is unsupported")
    body = value[header_end:]
    expected_offset = 0
    decoded: dict[str, bytes] = {}
    for section in header.get("sections", []):
        offset = int(section["offset_bytes"])
        stored_bytes = int(section["stored_bytes"])
        raw_bytes = int(section["raw_bytes"])
        if offset != expected_offset or stored_bytes < 0 or raw_bytes < 0:
            raise FWTileError("container section layout is not contiguous")
        end = offset + stored_bytes
        if end > len(body):
            raise FWTileError("container is truncated inside a section")
        stored = body[offset:end]
        if sha256_bytes(stored) != section["stored_sha256"]:
            raise FWTileError(f"stored section {section['name']!r} checksum mismatch")
        if decode_sections:
            raw = _safe_decompress(stored, raw_bytes)
            if sha256_bytes(raw) != section["raw_sha256"]:
                raise FWTileError(f"raw section {section['name']!r} checksum mismatch")
            decoded[str(section["name"])] = raw
        expected_offset = end
    if expected_offset != len(body):
        raise FWTileError("container has unreferenced trailing bytes")
    return {"header": header, "sections": decoded}


def decode_u16_elevations(raw: bytes, metadata: Mapping[str, Any]) -> list[float]:
    if len(raw) % 2:
        raise FWTileError("terrain UInt16 section has an odd byte count")
    values = array("H")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    quantization = metadata["elevation_quantization"]
    minimum = float(quantization["minimum_m"])
    step = float(quantization["step_m"])
    return [minimum + value * step for value in values]


__all__ = [
    "FWTileError",
    "MAGIC",
    "TREE_RECORD",
    "build_container",
    "build_vector_sections",
    "canonical_json_bytes",
    "decode_u16_elevations",
    "encode_detail_terrain",
    "encode_far_grid",
    "encode_mesh_section",
    "encode_tree_instances",
    "prism_to_mesh",
    "read_container",
    "sha256_bytes",
    "sha256_file",
    "triangulate_faces",
]
