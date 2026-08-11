"""Deterministic fixed-grid terrain compiler and FVTG v1 codec.

The compiler consumes one exact 500 m Lambert-93 tile sampled every 2 m and
one 10 m acquisition halo.  Input rows run south-to-north and columns run
west-to-east.  Heights are quantized to millimetres before interpolation.

LOD0 is sampled at global EPSG:2154 coordinates with integer bilinear
interpolation.  LOD1 and LOD2 are strict subsets of LOD0.  A separate 10 m
skirt section is emitted for rendering transitions; skirt geometry is never
part of the terrain core, collision, or simulation surface.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Sequence

import numpy as np


CONTRACT_SCHEMA = "fireviewer.terrain-fixed-grid-contract.v1"
FVTG_SCHEMA = "fireviewer.terrain-fixed-grid.v1"
FVTG_MAGIC = b"FVTG"
FVTG_VERSION = 1
CRS = "EPSG:2154"

TILE_SIZE_MM = 500_000
SOURCE_RESOLUTION_MM = 2_000
SOURCE_SAMPLE_COUNT = 251
ACQUISITION_HALO_MM = 10_000
ACQUISITION_HALO_SAMPLE_COUNT = 261
NORMAL_HALO_MM = 2_000
NORMAL_HALO_SAMPLE_COUNT = 253
SKIRT_DEPTH_MM = 10_000

LOD0_GRID_SIZE = 129
LOD_SPECS = (
    (0, 129, 1, 32_768),
    (1, 33, 4, 2_048),
    (2, 9, 16, 128),
)
EDGE_ORDER = ("west", "east", "south", "north")

_HEADER = struct.Struct("<4sHBBqqIIHHIiii32s32s32s32s")
_EDGE_VALUE = struct.Struct("<qiii3h")


@dataclass(frozen=True, slots=True)
class FixedLodMesh:
    """One regular terrain core and its separately indexed render skirt."""

    lod: int
    grid_size: int
    relative_heights_mm: tuple[int, ...]
    gradients_mm_per_4m: tuple[tuple[int, int], ...]
    normals_snorm16: tuple[tuple[int, int, int], ...]
    core_triangles: tuple[tuple[int, int, int], ...]
    skirt_core_vertex_indices: tuple[int, ...]
    skirt_relative_heights_mm: tuple[int, ...]
    skirt_triangles: tuple[tuple[int, int, int], ...]

    @property
    def core_vertex_count(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def core_triangle_count(self) -> int:
        return len(self.core_triangles)

    @property
    def skirt_vertex_count(self) -> int:
        return len(self.skirt_core_vertex_indices)

    @property
    def skirt_triangle_count(self) -> int:
        return len(self.skirt_triangles)


@dataclass(frozen=True, slots=True)
class FixedTerrainTile:
    """Canonical 2 m terrain source plus three deterministic regular LODs."""

    tile_origin_mm: tuple[int, int]
    z_origin_mm: int
    normal_halo_heights_mm: tuple[int, ...]
    lods: tuple[FixedLodMesh, FixedLodMesh, FixedLodMesh]
    contract_sha256: bytes
    source_grid_sha256: bytes
    normal_halo_sha256: bytes


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def load_contract(path: Path | None = None) -> tuple[dict[str, object], bytes]:
    """Load and strictly validate the fixed-grid v1 contract."""

    contract_path = path or Path(__file__).with_name(
        "fixed_terrain_grid_contract.v1.json"
    )
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("Unsupported fixed terrain contract")
    source = payload.get("source")
    geometry = payload.get("geometry")
    gradients = payload.get("gradients")
    normals = payload.get("normals")
    binary = payload.get("binary_format")
    quantization = payload.get("height_quantization")
    lods = payload.get("lods")
    if not all(
        isinstance(section, dict)
        for section in (
            source,
            geometry,
            gradients,
            normals,
            binary,
            quantization,
        )
    ) or not isinstance(lods, list):
        raise ValueError("Fixed terrain contract is incomplete")
    assert isinstance(source, dict)
    assert isinstance(geometry, dict)
    assert isinstance(gradients, dict)
    assert isinstance(normals, dict)
    assert isinstance(binary, dict)
    assert isinstance(quantization, dict)
    expected_lods = [
        {
            "lod": lod,
            "grid_shape": [size, size],
            "lod0_stride": stride,
            "core_triangle_count": triangles,
        }
        for lod, size, stride, triangles in LOD_SPECS
    ]
    observed_lods = []
    for item in lods:
        if not isinstance(item, dict):
            raise ValueError("Fixed terrain LOD entries must be objects")
        observed_lods.append(
            {
                "lod": item.get("lod"),
                "grid_shape": item.get("grid_shape"),
                "lod0_stride": item.get("lod0_stride"),
                "core_triangle_count": item.get("core_triangle_count"),
            }
        )
    if (
        payload.get("format_schema") != FVTG_SCHEMA
        or payload.get("crs") != CRS
        or float(payload.get("tile_size_m", 0.0)) != TILE_SIZE_MM / 1_000
        or float(source.get("resolution_m", 0.0)) != SOURCE_RESOLUTION_MM / 1_000
        or source.get("core_shape") != [SOURCE_SAMPLE_COUNT, SOURCE_SAMPLE_COUNT]
        or float(source.get("acquisition_halo_m", 0.0)) != ACQUISITION_HALO_MM / 1_000
        or source.get("acquisition_halo_shape")
        != [ACQUISITION_HALO_SAMPLE_COUNT, ACQUISITION_HALO_SAMPLE_COUNT]
        or source.get("acquisition_core_slice") != [5, 256, 5, 256]
        or float(source.get("retained_normal_halo_m", 0.0)) != NORMAL_HALO_MM / 1_000
        or source.get("retained_normal_halo_shape")
        != [NORMAL_HALO_SAMPLE_COUNT, NORMAL_HALO_SAMPLE_COUNT]
        or source.get("retained_from_acquisition_slice") != [4, 257, 4, 257]
        or source.get("retained_core_slice") != [1, 252, 1, 252]
        or source.get("row_order") != "south_to_north"
        or source.get("column_order") != "west_to_east"
        or source.get("orthophoto_dependency") != "forbidden"
        or quantization.get("unit") != "millimetre"
        or quantization.get("rounding") != "half_away_from_zero"
        or quantization.get("storage") != "signed_int32"
        or float(geometry.get("skirt_depth_m", 0.0)) != SKIRT_DEPTH_MM / 1_000
        or geometry.get("core_and_skirt_sections") != "separate"
        or geometry.get("lod0_derivation")
        != "integer_bilinear_interpolation_at_global_EPSG2154_coordinates"
        or geometry.get("coarser_lod_derivation") != "strict_subsampling_of_lod0"
        or geometry.get("triangle_diagonal") != "southwest_to_northeast"
        or geometry.get("skirt_collision") != "forbidden"
        or geometry.get("skirt_shadow") != "forbidden"
        or geometry.get("skirt_main_camera_visibility") != "forbidden"
        or gradients.get("storage") != "two_signed_int32"
        or gradients.get("unit") != "millimetre_height_difference_per_4m"
        or gradients.get("method")
        != "centred_difference_at_plus_minus_2m_from_retained_halo"
        or normals.get("storage") != "three_signed_int16_snorm"
        or normals.get("range") != [-32_767, 32_767]
        or normals.get("method")
        != "integer_fixed_point_normalization_of_negative_gradient_x_negative_gradient_y_positive_4000mm"
        or observed_lods != expected_lods
        or binary.get("schema") != FVTG_SCHEMA
        or binary.get("extension") != ".fvtg"
        or binary.get("magic_ascii") != FVTG_MAGIC.decode("ascii")
        or int(binary.get("version", -1)) != FVTG_VERSION
        or binary.get("endianness") != "little"
        or binary.get("integrity") != "sha256_of_zero_digest_header_and_payload"
        or binary.get("derived_geometry_storage")
        != "implicit_and_rebuilt_deterministically_from_the_retained_halo"
        or binary.get("unknown_trailing_payload") != "reject"
        or binary.get("contract_hash_mismatch") != "reject"
    ):
        raise ValueError("Fixed terrain contract constants are invalid")
    canonical = _canonical_json_bytes(payload)
    return payload, hashlib.sha256(canonical).digest()


def _metres_to_mm(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("Metric coordinates and heights must be finite")
    scaled = value * 1_000.0
    if scaled >= 0.0:
        result = int(math.floor(scaled + 0.5))
    else:
        result = int(math.ceil(scaled - 0.5))
    if result < np.iinfo("int64").min or result > np.iinfo("int64").max:
        raise ValueError("Metric coordinate exceeds signed int64 millimetres")
    return result


def _quantize_grid_mm(
    heights_m: np.ndarray, expected_shape: tuple[int, int], label: str
) -> np.ndarray:
    values = np.asarray(heights_m, dtype="float64")
    if values.shape != expected_shape:
        raise ValueError(f"{label} must have shape {expected_shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must not contain nodata")
    scaled = values * 1_000.0
    rounded = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    if rounded.min() < np.iinfo("int32").min or rounded.max() > np.iinfo("int32").max:
        raise ValueError(f"{label} exceeds signed int32 millimetres")
    return rounded.astype("<i4")


def quantize_heights_mm(heights_m: np.ndarray) -> np.ndarray:
    """Quantize the canonical 251 x 251 MNT core to signed millimetres."""

    return _quantize_grid_mm(
        heights_m,
        (SOURCE_SAMPLE_COUNT, SOURCE_SAMPLE_COUNT),
        "Canonical 2 m MNT core",
    )


def quantize_source_halo_mm(heights_m: np.ndarray) -> np.ndarray:
    """Quantize the 10 m acquisition halo to signed millimetres."""

    return _quantize_grid_mm(
        heights_m,
        (ACQUISITION_HALO_SAMPLE_COUNT, ACQUISITION_HALO_SAMPLE_COUNT),
        "Canonical 2 m MNT acquisition halo",
    )


def _canonical_normal_halo_mm(values_mm: np.ndarray) -> np.ndarray:
    values = np.asarray(values_mm)
    expected = (NORMAL_HALO_SAMPLE_COUNT, NORMAL_HALO_SAMPLE_COUNT)
    if values.shape != expected:
        raise ValueError(f"Canonical MNT normal halo must have shape {expected}")
    if values.dtype.kind not in {"i", "u"}:
        raise ValueError("Canonical MNT normal halo must contain integer millimetres")
    values64 = np.asarray(values, dtype="int64")
    if (
        int(values64.min()) < np.iinfo("int32").min
        or int(values64.max()) > np.iinfo("int32").max
    ):
        raise ValueError("Canonical MNT normal halo exceeds signed int32 millimetres")
    return values64.astype("<i4")


def _tile_origin_mm(value: Sequence[float]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError("tile_origin_l93_m must contain easting and northing")
    origin = (_metres_to_mm(float(value[0])), _metres_to_mm(float(value[1])))
    if any(coordinate % TILE_SIZE_MM for coordinate in origin):
        raise ValueError("Tile origin must align to the 500 m Lambert-93 grid")
    return origin


def _round_div_half_away(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("Interpolation denominator must be positive")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def _sample_global_mm(
    grid: np.ndarray,
    *,
    grid_origin_scaled: tuple[int, int],
    point_scaled: tuple[int, int],
) -> int:
    """Integer bilinear sample in global millimetres scaled by 128."""

    step = SOURCE_RESOLUTION_MM * (LOD0_GRID_SIZE - 1)
    x_delta = point_scaled[0] - grid_origin_scaled[0]
    y_delta = point_scaled[1] - grid_origin_scaled[1]
    maximum_x = (grid.shape[1] - 1) * step
    maximum_y = (grid.shape[0] - 1) * step
    if not (0 <= x_delta <= maximum_x and 0 <= y_delta <= maximum_y):
        raise ValueError("Fixed terrain sampling escaped the retained halo")
    column0 = min(x_delta // step, grid.shape[1] - 1)
    row0 = min(y_delta // step, grid.shape[0] - 1)
    remainder_x = x_delta - column0 * step
    remainder_y = y_delta - row0 * step
    column1 = min(column0 + 1, grid.shape[1] - 1)
    row1 = min(row0 + 1, grid.shape[0] - 1)
    if column0 == column1:
        remainder_x = 0
    if row0 == row1:
        remainder_y = 0
    weight_x0 = step - remainder_x
    weight_y0 = step - remainder_y
    numerator = (
        int(grid[row0, column0]) * weight_x0 * weight_y0
        + int(grid[row0, column1]) * remainder_x * weight_y0
        + int(grid[row1, column0]) * weight_x0 * remainder_y
        + int(grid[row1, column1]) * remainder_x * remainder_y
    )
    return _round_div_half_away(numerator, step * step)


def _normal_snorm16(gradient_x: int, gradient_y: int) -> tuple[int, int, int]:
    components = (-gradient_x, -gradient_y, 4_000)
    length_squared = sum(component * component for component in components)
    # Sixteen fixed-point fractional bits make normalization deterministic and
    # avoid relying on platform floating-point square roots.
    fixed_scale = 1 << 16
    length_fixed = math.isqrt(length_squared * fixed_scale * fixed_scale)
    values = tuple(
        max(
            -32_767,
            min(
                32_767,
                _round_div_half_away(
                    component * 32_767 * fixed_scale,
                    length_fixed,
                ),
            ),
        )
        for component in components
    )
    return values  # type: ignore[return-value]


def _core_triangles(grid_size: int) -> tuple[tuple[int, int, int], ...]:
    triangles: list[tuple[int, int, int]] = []
    for row in range(grid_size - 1):
        for column in range(grid_size - 1):
            southwest = row * grid_size + column
            southeast = southwest + 1
            northwest = southwest + grid_size
            northeast = northwest + 1
            triangles.append((southwest, southeast, northeast))
            triangles.append((southwest, northeast, northwest))
    return tuple(triangles)


def _perimeter_indices(grid_size: int) -> tuple[int, ...]:
    south = tuple(range(grid_size))
    east = tuple(row * grid_size + grid_size - 1 for row in range(1, grid_size))
    north = tuple(
        (grid_size - 1) * grid_size + column for column in range(grid_size - 2, -1, -1)
    )
    west = tuple(row * grid_size for row in range(grid_size - 2, 0, -1))
    return south + east + north + west


def _skirt_triangles(
    core_vertex_count: int, skirt_vertex_count: int
) -> tuple[tuple[int, int, int], ...]:
    triangles: list[tuple[int, int, int]] = []
    for index in range(skirt_vertex_count):
        following = (index + 1) % skirt_vertex_count
        core_a = index
        core_b = following
        # Core perimeter indices are remapped by the caller after construction.
        skirt_a = core_vertex_count + index
        skirt_b = core_vertex_count + following
        triangles.append((core_a, skirt_a, core_b))
        triangles.append((core_b, skirt_a, skirt_b))
    return tuple(triangles)


def _build_lod_mesh(
    lod: int,
    grid_size: int,
    stride: int,
    lod0_absolute_heights: tuple[int, ...],
    lod0_gradients: tuple[tuple[int, int], ...],
    lod0_normals: tuple[tuple[int, int, int], ...],
    z_origin_mm: int,
) -> FixedLodMesh:
    selected = tuple(
        row * LOD0_GRID_SIZE + column
        for row in range(0, LOD0_GRID_SIZE, stride)
        for column in range(0, LOD0_GRID_SIZE, stride)
    )
    if len(selected) != grid_size * grid_size:
        raise AssertionError("Fixed LOD subsampling produced an invalid grid")
    relative_heights = tuple(
        lod0_absolute_heights[index] - z_origin_mm for index in selected
    )
    if (
        min(relative_heights) < np.iinfo("int32").min
        or max(relative_heights) > np.iinfo("int32").max
    ):
        raise ValueError("Fixed terrain relative height exceeds signed int32")
    gradients = tuple(lod0_gradients[index] for index in selected)
    normals = tuple(lod0_normals[index] for index in selected)
    perimeter = _perimeter_indices(grid_size)
    skirt_heights = tuple(
        relative_heights[index] - SKIRT_DEPTH_MM for index in perimeter
    )
    if (
        min(skirt_heights) < np.iinfo("int32").min
        or max(skirt_heights) > np.iinfo("int32").max
    ):
        raise ValueError("Fixed terrain skirt height exceeds signed int32")
    raw_skirt_triangles = _skirt_triangles(grid_size * grid_size, len(perimeter))
    skirt_triangles = tuple(
        (
            perimeter[triangle[0]],
            triangle[1],
            perimeter[triangle[2]],
        )
        if triangle_number % 2 == 0
        else (
            perimeter[triangle[0]],
            triangle[1],
            triangle[2],
        )
        for triangle_number, triangle in enumerate(raw_skirt_triangles)
    )
    return FixedLodMesh(
        lod=lod,
        grid_size=grid_size,
        relative_heights_mm=relative_heights,
        gradients_mm_per_4m=gradients,
        normals_snorm16=normals,
        core_triangles=_core_triangles(grid_size),
        skirt_core_vertex_indices=perimeter,
        skirt_relative_heights_mm=skirt_heights,
        skirt_triangles=skirt_triangles,
    )


def _build_lods(
    normal_halo: np.ndarray,
    tile_origin_mm: tuple[int, int],
    z_origin_mm: int,
) -> tuple[FixedLodMesh, FixedLodMesh, FixedLodMesh]:
    scale = LOD0_GRID_SIZE - 1
    grid_origin_scaled = (
        (tile_origin_mm[0] - NORMAL_HALO_MM) * scale,
        (tile_origin_mm[1] - NORMAL_HALO_MM) * scale,
    )
    lod0_heights: list[int] = []
    lod0_gradients: list[tuple[int, int]] = []
    lod0_normals: list[tuple[int, int, int]] = []
    gradient_offset = SOURCE_RESOLUTION_MM * scale
    for row in range(LOD0_GRID_SIZE):
        global_y = tile_origin_mm[1] * scale + row * TILE_SIZE_MM
        for column in range(LOD0_GRID_SIZE):
            global_x = tile_origin_mm[0] * scale + column * TILE_SIZE_MM
            point = (global_x, global_y)
            height = _sample_global_mm(
                normal_halo,
                grid_origin_scaled=grid_origin_scaled,
                point_scaled=point,
            )
            gradient_x = _sample_global_mm(
                normal_halo,
                grid_origin_scaled=grid_origin_scaled,
                point_scaled=(global_x + gradient_offset, global_y),
            ) - _sample_global_mm(
                normal_halo,
                grid_origin_scaled=grid_origin_scaled,
                point_scaled=(global_x - gradient_offset, global_y),
            )
            gradient_y = _sample_global_mm(
                normal_halo,
                grid_origin_scaled=grid_origin_scaled,
                point_scaled=(global_x, global_y + gradient_offset),
            ) - _sample_global_mm(
                normal_halo,
                grid_origin_scaled=grid_origin_scaled,
                point_scaled=(global_x, global_y - gradient_offset),
            )
            if not (
                np.iinfo("int32").min <= gradient_x <= np.iinfo("int32").max
                and np.iinfo("int32").min <= gradient_y <= np.iinfo("int32").max
            ):
                raise ValueError("Fixed terrain gradient exceeds signed int32")
            lod0_heights.append(height)
            lod0_gradients.append((gradient_x, gradient_y))
            lod0_normals.append(_normal_snorm16(gradient_x, gradient_y))
    absolute = tuple(lod0_heights)
    gradients = tuple(lod0_gradients)
    normals = tuple(lod0_normals)
    meshes = tuple(
        _build_lod_mesh(
            lod,
            grid_size,
            stride,
            absolute,
            gradients,
            normals,
            z_origin_mm,
        )
        for lod, grid_size, stride, _triangles in LOD_SPECS
    )
    return meshes  # type: ignore[return-value]


def _grid_sha256(
    domain: bytes, grid: np.ndarray, tile_origin_mm: tuple[int, int]
) -> bytes:
    canonical = np.asarray(grid, dtype="<i4")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(
        struct.pack(
            "<qqHH",
            tile_origin_mm[0],
            tile_origin_mm[1],
            canonical.shape[0],
            canonical.shape[1],
        )
    )
    digest.update(canonical.tobytes(order="C"))
    return digest.digest()


def source_grid_mm(tile: FixedTerrainTile) -> np.ndarray:
    """Return a defensive 251 x 251 copy of the canonical MNT core."""

    halo = np.asarray(tile.normal_halo_heights_mm, dtype="<i4").reshape(
        NORMAL_HALO_SAMPLE_COUNT, NORMAL_HALO_SAMPLE_COUNT
    )
    return halo[1:-1, 1:-1].copy()


def lod_absolute_heights_mm(tile: FixedTerrainTile, lod: int) -> np.ndarray:
    """Return one south-to-north regular LOD height grid in absolute mm."""

    try:
        mesh = tile.lods[lod]
    except (IndexError, TypeError) as error:
        raise ValueError("LOD must be 0, 1, or 2") from error
    if mesh.lod != lod:
        raise ValueError("Fixed terrain LODs are not ordered")
    values = np.asarray(mesh.relative_heights_mm, dtype="int64") + tile.z_origin_mm
    return values.reshape(mesh.grid_size, mesh.grid_size)


def compile_fixed_terrain(
    heights_m: np.ndarray,
    *,
    source_halo_heights_m: np.ndarray,
    tile_origin_l93_m: Sequence[float],
    contract_path: Path | None = None,
) -> FixedTerrainTile:
    """Compile the three fixed LODs for one exact 500 m Lambert-93 tile."""

    source = quantize_heights_mm(heights_m)
    acquisition_halo = quantize_source_halo_mm(source_halo_heights_m)
    if not np.array_equal(source, acquisition_halo[5:256, 5:256]):
        raise ValueError("The 10 m MNT halo core must exactly match the 251 grid")
    normal_halo = acquisition_halo[4:257, 4:257].copy()
    return compile_fixed_terrain_from_canonical_mm(
        normal_halo,
        tile_origin_l93_m=tile_origin_l93_m,
        contract_path=contract_path,
    )


def compile_fixed_terrain_from_normal_halo(
    normal_halo_heights_m: np.ndarray,
    *,
    tile_origin_l93_m: Sequence[float],
    contract_path: Path | None = None,
) -> FixedTerrainTile:
    """Compile directly from the retained 253 x 253 halo in metres."""

    normal_halo = _quantize_grid_mm(
        normal_halo_heights_m,
        (NORMAL_HALO_SAMPLE_COUNT, NORMAL_HALO_SAMPLE_COUNT),
        "Canonical 2 m MNT normal halo",
    )
    return compile_fixed_terrain_from_canonical_mm(
        normal_halo,
        tile_origin_l93_m=tile_origin_l93_m,
        contract_path=contract_path,
    )


def compile_fixed_terrain_from_canonical_mm(
    normal_halo_mm: np.ndarray,
    *,
    tile_origin_l93_m: Sequence[float],
    contract_path: Path | None = None,
) -> FixedTerrainTile:
    """Compile from the acquisition pipeline's canonical 253 x 253 `.npy`."""

    normal_halo = _canonical_normal_halo_mm(normal_halo_mm)
    source = normal_halo[1:-1, 1:-1]
    tile_origin = _tile_origin_mm(tile_origin_l93_m)
    _contract, contract_hash = load_contract(contract_path)
    source_hash = _grid_sha256(b"FVTGSOURCE1", source, tile_origin)
    normal_halo_hash = _grid_sha256(b"FVTGNORMAL1", normal_halo, tile_origin)
    z_origin = int(source.min())
    tile = FixedTerrainTile(
        tile_origin_mm=tile_origin,
        z_origin_mm=z_origin,
        normal_halo_heights_mm=tuple(int(value) for value in normal_halo.flat),
        lods=_build_lods(normal_halo, tile_origin, z_origin),
        contract_sha256=contract_hash,
        source_grid_sha256=source_hash,
        normal_halo_sha256=normal_halo_hash,
    )
    validate_fixed_terrain(tile, contract_path=contract_path)
    return tile


def validate_fixed_terrain(
    tile: FixedTerrainTile, *, contract_path: Path | None = None
) -> None:
    """Fail closed unless every retained and derived value matches v1."""

    _contract, expected_contract_hash = load_contract(contract_path)
    if tile.contract_sha256 != expected_contract_hash:
        raise ValueError("FVTG contract SHA-256 mismatch")
    if any(
        not isinstance(value, bytes) or len(value) != 32
        for value in (
            tile.contract_sha256,
            tile.source_grid_sha256,
            tile.normal_halo_sha256,
        )
    ):
        raise ValueError("FVTG hashes must contain exactly 32 bytes")
    if len(tile.tile_origin_mm) != 2 or any(
        coordinate % TILE_SIZE_MM for coordinate in tile.tile_origin_mm
    ):
        raise ValueError("FVTG tile origin is not on the 500 m grid")
    expected_halo_values = NORMAL_HALO_SAMPLE_COUNT * NORMAL_HALO_SAMPLE_COUNT
    if len(tile.normal_halo_heights_mm) != expected_halo_values:
        raise ValueError("FVTG retained normal halo has an invalid size")
    halo64 = np.asarray(tile.normal_halo_heights_mm, dtype="int64")
    if (
        int(halo64.min()) < np.iinfo("int32").min
        or int(halo64.max()) > np.iinfo("int32").max
    ):
        raise ValueError("FVTG retained normal halo exceeds signed int32")
    halo = halo64.astype("<i4").reshape(
        NORMAL_HALO_SAMPLE_COUNT, NORMAL_HALO_SAMPLE_COUNT
    )
    source = halo[1:-1, 1:-1]
    if tile.z_origin_mm != int(source.min()):
        raise ValueError("FVTG Z origin is not the source minimum")
    if tile.source_grid_sha256 != _grid_sha256(
        b"FVTGSOURCE1", source, tile.tile_origin_mm
    ):
        raise ValueError("FVTG source grid SHA-256 mismatch")
    if tile.normal_halo_sha256 != _grid_sha256(
        b"FVTGNORMAL1", halo, tile.tile_origin_mm
    ):
        raise ValueError("FVTG retained halo SHA-256 mismatch")
    expected_lods = _build_lods(halo, tile.tile_origin_mm, tile.z_origin_mm)
    if tile.lods != expected_lods:
        raise ValueError("FVTG LOD geometry does not match its canonical source")
    for mesh, (lod, grid_size, _stride, triangle_count) in zip(
        tile.lods, LOD_SPECS, strict=True
    ):
        perimeter_count = 4 * (grid_size - 1)
        if (
            mesh.lod != lod
            or mesh.grid_size != grid_size
            or mesh.core_vertex_count != grid_size * grid_size
            or mesh.core_triangle_count != triangle_count
            or mesh.skirt_vertex_count != perimeter_count
            or mesh.skirt_triangle_count != 2 * perimeter_count
        ):
            raise ValueError(f"FVTG LOD{lod} counts violate the fixed contract")


def edge_signature(tile: FixedTerrainTile, lod: int, edge: str) -> bytes:
    """Hash a globally ordered core edge for exact neighbor comparisons."""

    validate_fixed_terrain(tile)
    if edge not in EDGE_ORDER:
        raise ValueError(f"Edge must be one of {EDGE_ORDER}")
    if lod not in (0, 1, 2):
        raise ValueError("LOD must be 0, 1, or 2")
    mesh = tile.lods[lod]
    size = mesh.grid_size
    if edge == "west":
        indices = tuple(row * size for row in range(size))
    elif edge == "east":
        indices = tuple(row * size + size - 1 for row in range(size))
    elif edge == "south":
        indices = tuple(range(size))
    else:
        indices = tuple((size - 1) * size + column for column in range(size))
    scale = size - 1
    digest = hashlib.sha256()
    digest.update(b"FVTGEDGE1")
    digest.update(struct.pack("<B", lod))
    for index in indices:
        row, column = divmod(index, size)
        parameter_scaled = (
            tile.tile_origin_mm[1] * scale + row * TILE_SIZE_MM
            if edge in ("west", "east")
            else tile.tile_origin_mm[0] * scale + column * TILE_SIZE_MM
        )
        gradient_x, gradient_y = mesh.gradients_mm_per_4m[index]
        normal_x, normal_y, normal_z = mesh.normals_snorm16[index]
        digest.update(
            _EDGE_VALUE.pack(
                parameter_scaled,
                mesh.relative_heights_mm[index] + tile.z_origin_mm,
                gradient_x,
                gradient_y,
                normal_x,
                normal_y,
                normal_z,
            )
        )
    return digest.digest()


def encode_fixed_terrain(
    tile: FixedTerrainTile, *, contract_path: Path | None = None
) -> bytes:
    """Encode one complete tile as deterministic little-endian FVTG v1."""

    validate_fixed_terrain(tile, contract_path=contract_path)
    halo = np.asarray(tile.normal_halo_heights_mm, dtype="<i4").reshape(
        NORMAL_HALO_SAMPLE_COUNT, NORMAL_HALO_SAMPLE_COUNT
    )
    source = halo[1:-1, 1:-1]
    body = bytearray(halo.tobytes(order="C"))
    header_values = (
        FVTG_MAGIC,
        FVTG_VERSION,
        len(tile.lods),
        0,
        tile.tile_origin_mm[0],
        tile.tile_origin_mm[1],
        TILE_SIZE_MM,
        SOURCE_RESOLUTION_MM,
        SOURCE_SAMPLE_COUNT,
        NORMAL_HALO_SAMPLE_COUNT,
        SKIRT_DEPTH_MM,
        tile.z_origin_mm,
        int(source.min()),
        int(source.max()),
        tile.contract_sha256,
        tile.source_grid_sha256,
        tile.normal_halo_sha256,
    )
    zeroed_header = _HEADER.pack(*header_values, bytes(32))
    file_digest = hashlib.sha256(zeroed_header + body).digest()
    return _HEADER.pack(*header_values, file_digest) + body


def _take(payload: bytes, offset: int, size: int) -> tuple[bytes, int]:
    end = offset + size
    if size < 0 or end > len(payload):
        raise ValueError("Truncated FVTG payload")
    return payload[offset:end], end


def _decode_scalars(
    payload: bytes, offset: int, count: int, dtype: str
) -> tuple[tuple[int, ...], int]:
    item_size = np.dtype(dtype).itemsize
    raw, offset = _take(payload, offset, count * item_size)
    values = np.frombuffer(raw, dtype=dtype)
    return tuple(int(value) for value in values), offset


def decode_fixed_terrain(
    payload: bytes, *, contract_path: Path | None = None
) -> FixedTerrainTile:
    """Decode FVTG v1 and reject any hash, contract, or semantic mismatch."""

    header_raw, offset = _take(payload, 0, _HEADER.size)
    unpacked = _HEADER.unpack(header_raw)
    (
        magic,
        version,
        lod_count,
        flags,
        origin_x,
        origin_y,
        tile_size,
        source_resolution,
        source_count,
        normal_halo_count,
        skirt_depth,
        z_origin,
        minimum_height,
        maximum_height,
        contract_hash,
        source_hash,
        normal_halo_hash,
        observed_digest,
    ) = unpacked
    if (
        magic != FVTG_MAGIC
        or version != FVTG_VERSION
        or lod_count != len(LOD_SPECS)
        or flags != 0
        or tile_size != TILE_SIZE_MM
        or source_resolution != SOURCE_RESOLUTION_MM
        or source_count != SOURCE_SAMPLE_COUNT
        or normal_halo_count != NORMAL_HALO_SAMPLE_COUNT
        or skirt_depth != SKIRT_DEPTH_MM
    ):
        raise ValueError("Unsupported FVTG header")
    zeroed_header = _HEADER.pack(*unpacked[:-1], bytes(32))
    if (
        hashlib.sha256(zeroed_header + payload[_HEADER.size :]).digest()
        != observed_digest
    ):
        raise ValueError("FVTG file SHA-256 mismatch")
    _contract, expected_contract_hash = load_contract(contract_path)
    if contract_hash != expected_contract_hash:
        raise ValueError("FVTG contract SHA-256 mismatch")
    halo_count = NORMAL_HALO_SAMPLE_COUNT * NORMAL_HALO_SAMPLE_COUNT
    halo_values, offset = _decode_scalars(payload, offset, halo_count, "<i4")
    halo = np.asarray(halo_values, dtype="<i4").reshape(
        NORMAL_HALO_SAMPLE_COUNT, NORMAL_HALO_SAMPLE_COUNT
    )
    source = halo[1:-1, 1:-1]
    if (
        z_origin != int(source.min())
        or minimum_height != int(source.min())
        or maximum_height != int(source.max())
    ):
        raise ValueError("FVTG source height bounds mismatch")
    if offset != len(payload):
        raise ValueError("FVTG contains an unknown trailing payload")
    tile_origin = (origin_x, origin_y)
    tile = FixedTerrainTile(
        tile_origin_mm=tile_origin,
        z_origin_mm=z_origin,
        normal_halo_heights_mm=halo_values,
        lods=_build_lods(halo, tile_origin, z_origin),
        contract_sha256=contract_hash,
        source_grid_sha256=source_hash,
        normal_halo_sha256=normal_halo_hash,
    )
    validate_fixed_terrain(tile, contract_path=contract_path)
    return tile


def write_fixed_terrain(
    tile: FixedTerrainTile,
    path: Path,
    *,
    contract_path: Path | None = None,
) -> bytes:
    """Atomically write FVTG, refusing to replace different existing bytes."""

    payload = encode_fixed_terrain(tile, contract_path=contract_path)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_bytes(payload)
        if destination.exists():
            if destination.read_bytes() != payload:
                raise FileExistsError(
                    f"Refusing to replace different fixed terrain: {destination}"
                )
            temporary.unlink()
        else:
            temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).digest()


def read_fixed_terrain(
    path: Path, *, contract_path: Path | None = None
) -> FixedTerrainTile:
    """Read and fully validate one FVTG tile."""

    return decode_fixed_terrain(Path(path).read_bytes(), contract_path=contract_path)


def _require_d_output(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if resolved.drive.upper() != "D:":
        raise ValueError(f"Fixed terrain output must stay on D:, got {resolved}")
    return resolved


def _cli_summary(
    tile: FixedTerrainTile, path: Path, file_sha256: bytes
) -> dict[str, object]:
    return {
        "schema": "fireviewer.terrain-fixed-grid-build-receipt.v1",
        "format_schema": FVTG_SCHEMA,
        "path": str(path),
        "sha256": file_sha256.hex(),
        "tile_origin_l93_m": [
            tile.tile_origin_mm[0] / 1_000,
            tile.tile_origin_mm[1] / 1_000,
        ],
        "source_grid_sha256": tile.source_grid_sha256.hex(),
        "normal_halo_sha256": tile.normal_halo_sha256.hex(),
        "contract_sha256": tile.contract_sha256.hex(),
        "lods": [
            {
                "lod": mesh.lod,
                "grid_size": mesh.grid_size,
                "core_triangles": mesh.core_triangle_count,
                "skirt_triangles": mesh.skirt_triangle_count,
            }
            for mesh in tile.lods
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Compile or verify an FVTG using the acquisition pipeline's `.npy`."""

    parser = argparse.ArgumentParser(
        description="Compile and verify deterministic FireViewer fixed terrain"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser(
        "compile", help="compile one canonical MNT halo"
    )
    compile_parser.add_argument("--normal-halo", type=Path, required=True)
    compile_parser.add_argument("--origin-x", type=float, required=True)
    compile_parser.add_argument("--origin-y", type=float, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--contract", type=Path)
    verify_parser = commands.add_parser("verify", help="verify one FVTG payload")
    verify_parser.add_argument("--input", type=Path, required=True)
    verify_parser.add_argument("--contract", type=Path)
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    if arguments.command == "compile":
        output = _require_d_output(arguments.output)
        loaded = np.load(arguments.normal_halo, allow_pickle=False)
        if not isinstance(loaded, np.ndarray):
            raise ValueError("--normal-halo must identify one NumPy array")
        tile = compile_fixed_terrain_from_canonical_mm(
            loaded,
            tile_origin_l93_m=(arguments.origin_x, arguments.origin_y),
            contract_path=arguments.contract,
        )
        file_sha256 = write_fixed_terrain(
            tile, output, contract_path=arguments.contract
        )
        summary = _cli_summary(tile, output, file_sha256)
    else:
        input_path = arguments.input.resolve(strict=True)
        payload = input_path.read_bytes()
        tile = decode_fixed_terrain(payload, contract_path=arguments.contract)
        summary = _cli_summary(tile, input_path, hashlib.sha256(payload).digest())
    print(
        json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )
    return 0


__all__ = [
    "ACQUISITION_HALO_SAMPLE_COUNT",
    "CONTRACT_SCHEMA",
    "CRS",
    "EDGE_ORDER",
    "FVTG_MAGIC",
    "FVTG_SCHEMA",
    "FVTG_VERSION",
    "FixedLodMesh",
    "FixedTerrainTile",
    "LOD_SPECS",
    "NORMAL_HALO_SAMPLE_COUNT",
    "SKIRT_DEPTH_MM",
    "SOURCE_SAMPLE_COUNT",
    "compile_fixed_terrain",
    "compile_fixed_terrain_from_canonical_mm",
    "compile_fixed_terrain_from_normal_halo",
    "decode_fixed_terrain",
    "edge_signature",
    "encode_fixed_terrain",
    "load_contract",
    "lod_absolute_heights_mm",
    "quantize_heights_mm",
    "quantize_source_halo_mm",
    "read_fixed_terrain",
    "source_grid_mm",
    "validate_fixed_terrain",
    "write_fixed_terrain",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
