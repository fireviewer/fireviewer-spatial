"""Deterministic adaptive terrain compiler and FVTQ v1 codec.

The compiler consumes one canonical 500 m height grid sampled every 2 m.
Input rows run from south to north and columns from west to east. Heights are
quantized to millimetres before any interpolation or subdivision decision.
All later calculations use integer arithmetic so worker order, paths and
floating-point implementation details cannot change the output.

This module only authors terrain geometry. Ground materials, buildings,
vegetation and simulation content are deliberately outside its contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Iterable, Mapping, Sequence

import numpy as np


CONTRACT_SCHEMA = "fireviewer.terrain-quadtree-contract.v1"
FVTQ_SCHEMA = "fireviewer.terrain-quadtree.v1"
FVTQ_MAGIC = b"FVTQ"
FVTQ_VERSION = 1
TILE_SIZE_MM = 500_000
SOURCE_RESOLUTION_MM = 2_000
SOURCE_SAMPLE_COUNT = TILE_SIZE_MM // SOURCE_RESOLUTION_MM + 1
NORMAL_HALO_SAMPLE_COUNT = SOURCE_SAMPLE_COUNT + 2
MAX_DEPTH = 8
GRID_UNITS = 1 << MAX_DEPTH
LOD0_TRIANGLE_BUDGET_WITHOUT_BREAKLINES = 32_768
EDGE_ORDER = ("west", "east", "south", "north")
STITCH_MASK_BITS = {"west": 1, "east": 2, "south": 4, "north": 8}
# A cross-LOD edge is always separated by one quadtree depth (2:1).  The
# coarser payloads therefore keep a narrow, deterministic boundary ring that
# is finer than their interior minimum.  Skipping directly from 7 to 5 (or 5
# to 3) would be a 4:1 boundary and can exceed the fine LOD error contract
# before the two payloads even meet.
LOD_EDGE_DEPTHS = (7, 6, 5)
LOD_MAXIMUM_TRIANGLES_WITH_BREAKLINES = (131_072, 4_096, 1_024)

NODE_LEAF = 1 << 0
NODE_SPLIT_ERROR = 1 << 1
NODE_SPLIT_MAX_CELL = 1 << 2
NODE_SPLIT_BREAKLINE = 1 << 3
NODE_SPLIT_EDGE = 1 << 4
NODE_SPLIT_BALANCE = 1 << 5
NODE_FLAGS_MASK = (
    NODE_LEAF
    | NODE_SPLIT_ERROR
    | NODE_SPLIT_MAX_CELL
    | NODE_SPLIT_BREAKLINE
    | NODE_SPLIT_EDGE
    | NODE_SPLIT_BALANCE
)

_HEADER = struct.Struct("<4sHBBqqIiiiIIIII4I32s32s32s128s32s")
_NODE = struct.Struct("<IBBHHI")
_VERTEX = struct.Struct("<HHi")
_GRADIENT = struct.Struct("<ii")
_TRIANGLE = struct.Struct("<III")
_CONSTRAINT_HEADER = struct.Struct("<HH")
_CONSTRAINT_POINT = struct.Struct("<ii")
_EDGE_VALUE = struct.Struct("<Hiii")
_STITCH_HEADER = struct.Struct("<B3xIII128s")


@dataclass(frozen=True, slots=True)
class LodPolicy:
    lod: int
    maximum_error_mm: int
    base_depth: int
    maximum_regular_depth: int
    maximum_breakline_depth: int
    edge_depth: int
    maximum_triangles_with_breaklines: int


@dataclass(frozen=True, slots=True)
class Breakline:
    """One deterministic tile-local breakline in millimetres."""

    feature_id: str
    points_mm: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.feature_id or len(self.points_mm) < 2:
            raise ValueError("A breakline needs an id and at least two points")
        for point in self.points_mm:
            if len(point) != 2 or any(
                value < 0 or value > TILE_SIZE_MM for value in point
            ):
                raise ValueError("Breakline points must be inside the 500 m tile")

    @classmethod
    def from_metres(
        cls, feature_id: str, points: Sequence[Sequence[float]]
    ) -> "Breakline":
        return cls(
            feature_id=feature_id,
            points_mm=tuple(
                (_metres_to_mm(float(point[0])), _metres_to_mm(float(point[1])))
                for point in points
            ),
        )


@dataclass(frozen=True, slots=True)
class QuadtreeNode:
    morton: int
    depth: int
    flags: int
    x_index: int
    y_index: int
    maximum_error_mm: int

    @property
    def is_leaf(self) -> bool:
        return bool(self.flags & NODE_LEAF)


@dataclass(frozen=True, slots=True)
class StitchVariant:
    mask: int
    removed_triangle_indices: tuple[int, ...]
    replacement_triangles: tuple[tuple[int, int, int], ...]
    effective_edge_signatures: tuple[bytes, bytes, bytes, bytes]
    maximum_error_mm: int

    def triangle_count(self, base_triangle_count: int) -> int:
        return (
            base_triangle_count
            - len(self.removed_triangle_indices)
            + len(self.replacement_triangles)
        )


@dataclass(frozen=True, slots=True)
class FvtqMesh:
    lod: int
    tile_origin_mm: tuple[int, int]
    z_origin_mm: int
    minimum_relative_height_mm: int
    maximum_relative_height_mm: int
    maximum_final_error_mm: int
    nodes: tuple[QuadtreeNode, ...]
    vertices: tuple[tuple[int, int, int], ...]
    vertex_gradients_mm_per_4m: tuple[tuple[int, int], ...]
    triangles: tuple[tuple[int, int, int], ...]
    edge_vertex_indices: tuple[tuple[int, ...], ...]
    edge_signatures: tuple[bytes, ...]
    stitch_variants: tuple[StitchVariant, ...]
    breaklines: tuple[Breakline, ...]
    contract_sha256: bytes
    source_grid_sha256: bytes
    normal_halo_sha256: bytes

    @property
    def leaf_count(self) -> int:
        return sum(node.is_leaf for node in self.nodes)


@dataclass(frozen=True, slots=True)
class AdaptiveTerrainTile:
    tile_origin_mm: tuple[int, int]
    source_grid_sha256: bytes
    normal_halo_sha256: bytes
    contract_sha256: bytes
    lods: tuple[FvtqMesh, FvtqMesh, FvtqMesh]


@dataclass(slots=True)
class _NodeState:
    depth: int
    x_index: int
    y_index: int
    maximum_error_mm: int
    intersects_breakline: bool
    flags: int

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.depth, self.x_index, self.y_index)

    @property
    def is_leaf(self) -> bool:
        return bool(self.flags & NODE_LEAF)


def _metres_to_mm(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("Metric coordinates and heights must be finite")
    scaled = value * 1_000.0
    if scaled >= 0:
        return int(math.floor(scaled + 0.5))
    return int(math.ceil(scaled - 0.5))


def quantize_heights_mm(heights_m: np.ndarray) -> np.ndarray:
    values = np.asarray(heights_m, dtype="float64")
    expected = (SOURCE_SAMPLE_COUNT, SOURCE_SAMPLE_COUNT)
    if values.shape != expected:
        raise ValueError(f"The canonical 2 m grid must have shape {expected}")
    if not np.isfinite(values).all():
        raise ValueError("The canonical 2 m grid must not contain nodata")
    scaled = values * 1_000.0
    rounded = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    if rounded.min() < np.iinfo("int32").min or rounded.max() > np.iinfo("int32").max:
        raise ValueError("Quantized terrain heights exceed signed int32")
    return rounded.astype("<i4")


def quantize_normal_halo_mm(heights_m: np.ndarray) -> np.ndarray:
    values = np.asarray(heights_m, dtype="float64")
    expected = (NORMAL_HALO_SAMPLE_COUNT, NORMAL_HALO_SAMPLE_COUNT)
    if values.shape != expected:
        raise ValueError(f"The canonical 2 m normal halo must have shape {expected}")
    if not np.isfinite(values).all():
        raise ValueError("The canonical 2 m normal halo must not contain nodata")
    scaled = values * 1_000.0
    rounded = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    if rounded.min() < np.iinfo("int32").min or rounded.max() > np.iinfo("int32").max:
        raise ValueError("Quantized terrain normal halo exceeds signed int32")
    return rounded.astype("<i4")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def load_contract(
    path: Path | None = None,
) -> tuple[dict[str, object], bytes, tuple[LodPolicy, LodPolicy, LodPolicy]]:
    contract_path = path or Path(__file__).with_name(
        "terrain_quadtree_contract.v1.json"
    )
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("Unsupported adaptive terrain contract")
    source = payload.get("source_working_grid")
    normal_halo = payload.get("canonical_normal_halo")
    quadtree = payload.get("quadtree")
    if (
        not isinstance(source, dict)
        or not isinstance(normal_halo, dict)
        or not isinstance(quadtree, dict)
    ):
        raise ValueError("Adaptive terrain contract is incomplete")
    if (
        payload.get("crs") != "EPSG:2154"
        or float(payload.get("tile_size_m", 0.0)) != 500.0
        or float(source.get("resolution_m", 0.0)) != 2.0
        or source.get("shape") != [251, 251]
        or source.get("wms_pixel_shape_with_10m_halo") != [260, 260]
        or normal_halo.get("shape") != [253, 253]
        or normal_halo.get("core_slice") != [1, 252, 1, 252]
        or int(quadtree.get("maximum_depth", -1)) != MAX_DEPTH
        or quadtree.get("balance") != "2:1"
        or quadtree.get("stitch_mask_bits") != STITCH_MASK_BITS
        or int(quadtree.get("maximum_adjacent_lod_delta", -1)) != 1
    ):
        raise ValueError("Adaptive terrain contract constants are invalid")
    raw_lods = payload.get("lods")
    if not isinstance(raw_lods, list) or len(raw_lods) != 3:
        raise ValueError("The adaptive terrain contract must define three LODs")
    policies = tuple(
        LodPolicy(
            lod=int(item["lod"]),
            maximum_error_mm=_metres_to_mm(float(item["maximum_vertical_error_m"])),
            base_depth=int(item["base_depth"]),
            maximum_regular_depth=int(item["maximum_regular_depth"]),
            maximum_breakline_depth=int(item["maximum_breakline_depth"]),
            edge_depth=int(item["edge_depth"]),
            maximum_triangles_with_breaklines=int(
                item["maximum_triangles_with_breaklines"]
            ),
        )
        for item in raw_lods
        if isinstance(item, dict)
    )
    if len(policies) != 3 or tuple(policy.lod for policy in policies) != (0, 1, 2):
        raise ValueError("LOD policies must be ordered 0, 1, 2")
    expected = (
        LodPolicy(0, 500, 4, 7, 8, 7, 131_072),
        LodPolicy(1, 2_000, 2, 5, 5, 6, 4_096),
        LodPolicy(2, 8_000, 0, 3, 3, 5, 1_024),
    )
    if policies != expected:
        raise ValueError("LOD policies do not match the FVTQ v1 geometry contract")
    if (
        not isinstance(raw_lods[0], dict)
        or int(raw_lods[0].get("maximum_triangles_without_breaklines", -1))
        != LOD0_TRIANGLE_BUDGET_WITHOUT_BREAKLINES
    ):
        raise ValueError("LOD0 triangle budget does not match the FVTQ v1 contract")
    canonical = _canonical_json_bytes(payload)
    return payload, hashlib.sha256(canonical).digest(), policies  # type: ignore[return-value]


def _round_div_half_away(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("The interpolation denominator must be positive")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def _sample_height_mm(grid: np.ndarray, x_scaled: int, y_scaled: int) -> int:
    """Sample the quantized source with exact integer bilinear interpolation.

    Coordinates are tile-local millimetres multiplied by ``GRID_UNITS``.
    """

    source_step = SOURCE_RESOLUTION_MM * GRID_UNITS
    maximum = TILE_SIZE_MM * GRID_UNITS
    if not (0 <= x_scaled <= maximum and 0 <= y_scaled <= maximum):
        raise ValueError("Terrain sampling escaped the tile")
    column = min(x_scaled // source_step, SOURCE_SAMPLE_COUNT - 1)
    row = min(y_scaled // source_step, SOURCE_SAMPLE_COUNT - 1)
    if column == SOURCE_SAMPLE_COUNT - 1:
        column0 = column1 = column
        remainder_x = 0
    else:
        column0, column1 = column, column + 1
        remainder_x = x_scaled - column * source_step
    if row == SOURCE_SAMPLE_COUNT - 1:
        row0 = row1 = row
        remainder_y = 0
    else:
        row0, row1 = row, row + 1
        remainder_y = y_scaled - row * source_step
    inverse_x = source_step - remainder_x
    inverse_y = source_step - remainder_y
    numerator = (
        int(grid[row0, column0]) * inverse_x * inverse_y
        + int(grid[row0, column1]) * remainder_x * inverse_y
        + int(grid[row1, column0]) * inverse_x * remainder_y
        + int(grid[row1, column1]) * remainder_x * remainder_y
    )
    return _round_div_half_away(numerator, source_step * source_step)


def _sample_normal_halo_height_mm(
    grid: np.ndarray, x_scaled: int, y_scaled: int
) -> int:
    """Sample the canonical one-vertex halo at exact integer coordinates."""

    source_step = SOURCE_RESOLUTION_MM * GRID_UNITS
    shifted_x = x_scaled + source_step
    shifted_y = y_scaled + source_step
    maximum = (TILE_SIZE_MM + 2 * SOURCE_RESOLUTION_MM) * GRID_UNITS
    if not (0 <= shifted_x <= maximum and 0 <= shifted_y <= maximum):
        raise ValueError("Terrain normal sampling escaped the canonical halo")
    column = min(shifted_x // source_step, NORMAL_HALO_SAMPLE_COUNT - 1)
    row = min(shifted_y // source_step, NORMAL_HALO_SAMPLE_COUNT - 1)
    if column == NORMAL_HALO_SAMPLE_COUNT - 1:
        column0 = column1 = column
        remainder_x = 0
    else:
        column0, column1 = column, column + 1
        remainder_x = shifted_x - column * source_step
    if row == NORMAL_HALO_SAMPLE_COUNT - 1:
        row0 = row1 = row
        remainder_y = 0
    else:
        row0, row1 = row, row + 1
        remainder_y = shifted_y - row * source_step
    inverse_x = source_step - remainder_x
    inverse_y = source_step - remainder_y
    numerator = (
        int(grid[row0, column0]) * inverse_x * inverse_y
        + int(grid[row0, column1]) * remainder_x * inverse_y
        + int(grid[row1, column0]) * inverse_x * remainder_y
        + int(grid[row1, column1]) * remainder_x * remainder_y
    )
    return _round_div_half_away(numerator, source_step * source_step)


def _vertex_gradient_mm_per_4m(
    normal_halo: np.ndarray, x_grid: int, y_grid: int
) -> tuple[int, int]:
    x_scaled = x_grid * TILE_SIZE_MM
    y_scaled = y_grid * TILE_SIZE_MM
    delta = SOURCE_RESOLUTION_MM * GRID_UNITS
    dz_dx = _sample_normal_halo_height_mm(
        normal_halo, x_scaled + delta, y_scaled
    ) - _sample_normal_halo_height_mm(normal_halo, x_scaled - delta, y_scaled)
    dz_dy = _sample_normal_halo_height_mm(
        normal_halo, x_scaled, y_scaled + delta
    ) - _sample_normal_halo_height_mm(normal_halo, x_scaled, y_scaled - delta)
    minimum, maximum = np.iinfo("int32").min, np.iinfo("int32").max
    if not (minimum <= dz_dx <= maximum and minimum <= dz_dy <= maximum):
        raise ValueError("Terrain normal gradient exceeds signed int32")
    return dz_dx, dz_dy


def _node_scaled_bounds(
    depth: int, x_index: int, y_index: int
) -> tuple[int, int, int, int]:
    span = GRID_UNITS >> depth
    x0 = x_index * span * TILE_SIZE_MM
    y0 = y_index * span * TILE_SIZE_MM
    return x0, y0, x0 + span * TILE_SIZE_MM, y0 + span * TILE_SIZE_MM


def _triangle_prediction_mm(
    corners: tuple[int, int, int, int],
    bounds: tuple[int, int, int, int],
    x_scaled: int,
    y_scaled: int,
) -> int:
    z_sw, z_se, z_nw, z_ne = corners
    x0, y0, x1, _y1 = bounds
    denominator = x1 - x0
    dx = x_scaled - x0
    dy = y_scaled - y0
    if dx >= dy:
        numerator = z_sw * denominator + (z_se - z_sw) * dx + (z_ne - z_se) * dy
    else:
        numerator = z_sw * denominator + (z_ne - z_nw) * dx + (z_nw - z_sw) * dy
    return _round_div_half_away(numerator, denominator)


def _node_error_mm(grid: np.ndarray, depth: int, x_index: int, y_index: int) -> int:
    bounds = _node_scaled_bounds(depth, x_index, y_index)
    x0, y0, x1, y1 = bounds
    corners = (
        _sample_height_mm(grid, x0, y0),
        _sample_height_mm(grid, x1, y0),
        _sample_height_mm(grid, x0, y1),
        _sample_height_mm(grid, x1, y1),
    )
    source_step = SOURCE_RESOLUTION_MM * GRID_UNITS
    first_column = (x0 + source_step - 1) // source_step
    last_column = x1 // source_step
    first_row = (y0 + source_step - 1) // source_step
    last_row = y1 // source_step
    samples: set[tuple[int, int]] = {
        ((x0 + x1) // 2, (y0 + y1) // 2),
        ((x0 + x1) // 2, y0),
        ((x0 + x1) // 2, y1),
        (x0, (y0 + y1) // 2),
        (x1, (y0 + y1) // 2),
    }
    for row in range(first_row, last_row + 1):
        for column in range(first_column, last_column + 1):
            samples.add((column * source_step, row * source_step))
    maximum = 0
    for x_scaled, y_scaled in samples:
        actual = _sample_height_mm(grid, x_scaled, y_scaled)
        predicted = _triangle_prediction_mm(corners, bounds, x_scaled, y_scaled)
        maximum = max(maximum, abs(actual - predicted))
    return maximum


def _orientation(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (value > 0) - (value < 0)


def _on_segment(a: tuple[int, int], b: tuple[int, int], point: tuple[int, int]) -> bool:
    return (
        min(a[0], b[0]) <= point[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= point[1] <= max(a[1], b[1])
        and _orientation(a, b, point) == 0
    )


def _segments_intersect(
    a: tuple[int, int],
    b: tuple[int, int],
    c: tuple[int, int],
    d: tuple[int, int],
) -> bool:
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    if ab_c != ab_d and cd_a != cd_b:
        return True
    return any(
        (
            ab_c == 0 and _on_segment(a, b, c),
            ab_d == 0 and _on_segment(a, b, d),
            cd_a == 0 and _on_segment(c, d, a),
            cd_b == 0 and _on_segment(c, d, b),
        )
    )


def _segment_intersects_box(
    a: tuple[int, int], b: tuple[int, int], bounds: tuple[int, int, int, int]
) -> bool:
    x0, y0, x1, y1 = bounds
    if max(a[0], b[0]) < x0 or min(a[0], b[0]) > x1:
        return False
    if max(a[1], b[1]) < y0 or min(a[1], b[1]) > y1:
        return False
    if (x0 <= a[0] <= x1 and y0 <= a[1] <= y1) or (
        x0 <= b[0] <= x1 and y0 <= b[1] <= y1
    ):
        return True
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    return any(
        _segments_intersect(a, b, corners[index], corners[(index + 1) % 4])
        for index in range(4)
    )


def _node_intersects_breakline(
    depth: int, x_index: int, y_index: int, breaklines: Sequence[Breakline]
) -> bool:
    bounds = _node_scaled_bounds(depth, x_index, y_index)
    for breakline in breaklines:
        scaled = tuple((x * GRID_UNITS, y * GRID_UNITS) for x, y in breakline.points_mm)
        if any(
            _segment_intersects_box(start, end, bounds)
            for start, end in zip(scaled, scaled[1:], strict=False)
        ):
            return True
    return False


def _touches_tile_edge(depth: int, x_index: int, y_index: int) -> bool:
    count = 1 << depth
    return x_index == 0 or y_index == 0 or x_index == count - 1 or y_index == count - 1


def _requires_breakline_split(state: _NodeState, policy: LodPolicy) -> bool:
    if not state.intersects_breakline or state.depth >= policy.maximum_breakline_depth:
        return False
    # Stitch payloads bridge exactly one quadtree depth.  A constrained LOD0
    # cell may reach depth 8 in the tile interior, but refining a boundary cell
    # past its canonical edge depth would put it directly beside LOD1 depth 6
    # at runtime (4:1).  Keep that one-cell boundary at depth 7; the constraint
    # remains stored in the payload and full depth-8 refinement resumes inside.
    return (
        not _touches_tile_edge(state.depth, state.x_index, state.y_index)
        or state.depth < policy.edge_depth
    )


def _morton(x: int, y: int) -> int:
    value = 0
    for bit in range(MAX_DEPTH + 1):
        value |= ((x >> bit) & 1) << (2 * bit)
        value |= ((y >> bit) & 1) << (2 * bit + 1)
    return value


def _node_morton(depth: int, x_index: int, y_index: int) -> int:
    shift = MAX_DEPTH - depth
    return _morton(x_index << shift, y_index << shift)


def _evaluate_node(
    grid: np.ndarray,
    breaklines: Sequence[Breakline],
    depth: int,
    x_index: int,
    y_index: int,
) -> _NodeState:
    return _NodeState(
        depth=depth,
        x_index=x_index,
        y_index=y_index,
        maximum_error_mm=_node_error_mm(grid, depth, x_index, y_index),
        intersects_breakline=_node_intersects_breakline(
            depth, x_index, y_index, breaklines
        ),
        flags=NODE_LEAF,
    )


def _split_node(
    nodes: dict[tuple[int, int, int], _NodeState],
    state: _NodeState,
    grid: np.ndarray,
    breaklines: Sequence[Breakline],
    reason: int,
) -> tuple[_NodeState, _NodeState, _NodeState, _NodeState]:
    state.flags = (state.flags & ~NODE_LEAF) | reason
    children = tuple(
        _evaluate_node(
            grid,
            breaklines,
            state.depth + 1,
            state.x_index * 2 + dx,
            state.y_index * 2 + dy,
        )
        for dy in (0, 1)
        for dx in (0, 1)
    )
    for child in children:
        nodes[child.key] = child
    return children  # type: ignore[return-value]


def _initial_tree(
    grid: np.ndarray, breaklines: Sequence[Breakline], policy: LodPolicy
) -> dict[tuple[int, int, int], _NodeState]:
    root = _evaluate_node(grid, breaklines, 0, 0, 0)
    nodes = {root.key: root}
    pending = [root]
    while pending:
        state = pending.pop()
        reason = 0
        if state.depth < policy.base_depth:
            reason |= NODE_SPLIT_MAX_CELL
        if (
            state.maximum_error_mm > policy.maximum_error_mm
            and state.depth < policy.maximum_regular_depth
        ):
            reason |= NODE_SPLIT_ERROR
        if _requires_breakline_split(state, policy):
            reason |= NODE_SPLIT_BREAKLINE
        if _touches_tile_edge(state.depth, state.x_index, state.y_index) and (
            state.depth < policy.edge_depth
        ):
            reason |= NODE_SPLIT_EDGE
        if reason:
            pending.extend(_split_node(nodes, state, grid, breaklines, reason))
    return nodes


def _leaf_occupancy(
    nodes: Mapping[tuple[int, int, int], _NodeState],
) -> tuple[np.ndarray, list[_NodeState]]:
    leaves = sorted(
        (node for node in nodes.values() if node.is_leaf),
        key=lambda node: (
            _node_morton(node.depth, node.x_index, node.y_index),
            node.depth,
        ),
    )
    occupancy = np.full((GRID_UNITS, GRID_UNITS), -1, dtype="int32")
    for index, leaf in enumerate(leaves):
        span = GRID_UNITS >> leaf.depth
        x0 = leaf.x_index * span
        y0 = leaf.y_index * span
        occupancy[y0 : y0 + span, x0 : x0 + span] = index
    if (occupancy < 0).any():
        raise AssertionError("Quadtree leaves do not cover the complete tile")
    return occupancy, leaves


def _balance_tree(
    nodes: dict[tuple[int, int, int], _NodeState],
    grid: np.ndarray,
    breaklines: Sequence[Breakline],
) -> None:
    for _iteration in range(MAX_DEPTH + 1):
        occupancy, leaves = _leaf_occupancy(nodes)
        depths = np.asarray([leaf.depth for leaf in leaves], dtype="int16")
        split_indices: set[int] = set()
        for left, right in (
            (occupancy[:, :-1], occupancy[:, 1:]),
            (occupancy[:-1, :], occupancy[1:, :]),
        ):
            mask = np.abs(depths[left] - depths[right]) > 1
            if not mask.any():
                continue
            for left_index, right_index in zip(left[mask], right[mask], strict=True):
                if depths[left_index] < depths[right_index]:
                    split_indices.add(int(left_index))
                else:
                    split_indices.add(int(right_index))
        if not split_indices:
            return
        for index in sorted(split_indices):
            leaf = leaves[index]
            if leaf.depth >= MAX_DEPTH or not leaf.is_leaf:
                continue
            _split_node(nodes, leaf, grid, breaklines, NODE_SPLIT_BALANCE)
    raise AssertionError("Quadtree balancing did not converge")


def _clone_state(state: _NodeState) -> _NodeState:
    return _NodeState(
        depth=state.depth,
        x_index=state.x_index,
        y_index=state.y_index,
        maximum_error_mm=state.maximum_error_mm,
        intersects_breakline=state.intersects_breakline,
        flags=NODE_LEAF,
    )


def _split_node_from_master(
    nodes: dict[tuple[int, int, int], _NodeState],
    state: _NodeState,
    master: Mapping[tuple[int, int, int], _NodeState],
    reason: int,
) -> tuple[_NodeState, _NodeState, _NodeState, _NodeState]:
    state.flags = (state.flags & ~NODE_LEAF) | reason
    child_keys = tuple(
        (state.depth + 1, state.x_index * 2 + dx, state.y_index * 2 + dy)
        for dy in (0, 1)
        for dx in (0, 1)
    )
    try:
        children = tuple(_clone_state(master[key]) for key in child_keys)
    except KeyError as error:
        raise AssertionError(
            "A coarser LOD requested a node absent from LOD0"
        ) from error
    for child in children:
        nodes[child.key] = child
    return children  # type: ignore[return-value]


def _collapsed_tree_from_lod0(
    master: Mapping[tuple[int, int, int], _NodeState], policy: LodPolicy
) -> dict[tuple[int, int, int], _NodeState]:
    """Collapse the evaluated LOD0 tree without sampling the source again."""

    root = _clone_state(master[(0, 0, 0)])
    nodes = {root.key: root}
    pending = [root]
    while pending:
        state = pending.pop()
        reason = 0
        if state.depth < policy.base_depth:
            reason |= NODE_SPLIT_MAX_CELL
        if (
            state.maximum_error_mm > policy.maximum_error_mm
            and state.depth < policy.maximum_regular_depth
        ):
            reason |= NODE_SPLIT_ERROR
        if _requires_breakline_split(state, policy):
            reason |= NODE_SPLIT_BREAKLINE
        if _touches_tile_edge(state.depth, state.x_index, state.y_index) and (
            state.depth < policy.edge_depth
        ):
            reason |= NODE_SPLIT_EDGE
        if reason:
            pending.extend(_split_node_from_master(nodes, state, master, reason))

    for _iteration in range(MAX_DEPTH + 1):
        occupancy, leaves = _leaf_occupancy(nodes)
        depths = np.asarray([leaf.depth for leaf in leaves], dtype="int16")
        split_indices: set[int] = set()
        for left, right in (
            (occupancy[:, :-1], occupancy[:, 1:]),
            (occupancy[:-1, :], occupancy[1:, :]),
        ):
            mask = np.abs(depths[left] - depths[right]) > 1
            if not mask.any():
                continue
            for left_index, right_index in zip(left[mask], right[mask], strict=True):
                coarse = (
                    int(left_index)
                    if depths[left_index] < depths[right_index]
                    else int(right_index)
                )
                split_indices.add(coarse)
        if not split_indices:
            return nodes
        for index in sorted(split_indices):
            leaf = leaves[index]
            if leaf.depth >= MAX_DEPTH or not leaf.is_leaf:
                continue
            _split_node_from_master(nodes, leaf, master, NODE_SPLIT_BALANCE)
    raise AssertionError("Collapsed quadtree balancing did not converge")


def _leaf_transition_edges(
    leaf: _NodeState, occupancy: np.ndarray, leaves: Sequence[_NodeState]
) -> tuple[bool, bool, bool, bool]:
    span = GRID_UNITS >> leaf.depth
    x0 = leaf.x_index * span
    y0 = leaf.y_index * span
    x1 = x0 + span
    y1 = y0 + span
    depths = np.asarray([item.depth for item in leaves], dtype="int16")

    def finer(values: np.ndarray) -> bool:
        return bool(values.size and int(depths[values].max()) > leaf.depth)

    west = x0 > 0 and finer(occupancy[y0:y1, x0 - 1])
    east = x1 < GRID_UNITS and finer(occupancy[y0:y1, x1])
    south = y0 > 0 and finer(occupancy[y0 - 1, x0:x1])
    north = y1 < GRID_UNITS and finer(occupancy[y1, x0:x1])
    return west, east, south, north


def _build_mesh(
    grid: np.ndarray,
    normal_halo: np.ndarray,
    nodes: Mapping[tuple[int, int, int], _NodeState],
    z_origin_mm: int,
    master_heights: Mapping[tuple[int, int], int] | None = None,
    master_gradients: Mapping[tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[
    tuple[tuple[int, int, int], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int, int], ...],
    tuple[tuple[int, ...], ...],
    tuple[bytes, ...],
]:
    occupancy, leaves = _leaf_occupancy(nodes)
    triangle_points: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    points: set[tuple[int, int]] = set()
    for leaf in leaves:
        span = GRID_UNITS >> leaf.depth
        x0 = leaf.x_index * span
        y0 = leaf.y_index * span
        x1, y1 = x0 + span, y0 + span
        west, east, south, north = _leaf_transition_edges(leaf, occupancy, leaves)
        sw, se, ne, nw = (x0, y0), (x1, y0), (x1, y1), (x0, y1)
        if not any((west, east, south, north)):
            cell_triangles = ((sw, se, ne), (sw, ne, nw))
        else:
            if span % 2:
                raise AssertionError("A maximum-depth leaf cannot border a finer leaf")
            half = span // 2
            perimeter: list[tuple[int, int]] = [sw]
            if south:
                perimeter.append((x0 + half, y0))
            perimeter.append(se)
            if east:
                perimeter.append((x1, y0 + half))
            perimeter.append(ne)
            if north:
                perimeter.append((x0 + half, y1))
            perimeter.append(nw)
            if west:
                perimeter.append((x0, y0 + half))
            center = (x0 + half, y0 + half)
            cell_triangles = tuple(
                (center, perimeter[index], perimeter[(index + 1) % len(perimeter)])
                for index in range(len(perimeter))
            )
        triangle_points.extend(cell_triangles)
        for triangle in cell_triangles:
            points.update(triangle)

    ordered_points = sorted(
        points, key=lambda point: (_morton(*point), point[1], point[0])
    )
    point_to_index = {point: index for index, point in enumerate(ordered_points)}
    if master_heights is None:
        absolute_heights = {
            (x, y): _sample_height_mm(grid, x * TILE_SIZE_MM, y * TILE_SIZE_MM)
            for x, y in ordered_points
        }
    else:
        missing = [point for point in ordered_points if point not in master_heights]
        if missing:
            raise AssertionError("A collapsed LOD vertex is absent from the LOD0 mesh")
        absolute_heights = {point: master_heights[point] for point in ordered_points}
    vertices = tuple(
        (x, y, absolute_heights[(x, y)] - z_origin_mm) for x, y in ordered_points
    )
    if master_gradients is None:
        gradients = tuple(
            _vertex_gradient_mm_per_4m(normal_halo, x, y) for x, y in ordered_points
        )
    else:
        missing_gradients = [
            point for point in ordered_points if point not in master_gradients
        ]
        if missing_gradients:
            raise AssertionError(
                "A collapsed LOD gradient is absent from the LOD0 mesh"
            )
        gradients = tuple(master_gradients[point] for point in ordered_points)
    triangles = tuple(
        tuple(point_to_index[point] for point in triangle)  # type: ignore[misc]
        for triangle in triangle_points
    )
    edge_indices: list[tuple[int, ...]] = []
    signatures: list[bytes] = []
    for edge in EDGE_ORDER:
        if edge == "west":
            indices = [index for index, (x, _y) in enumerate(ordered_points) if x == 0]
            indices.sort(key=lambda index: ordered_points[index][1])
            parameter_axis = 1
        elif edge == "east":
            indices = [
                index for index, (x, _y) in enumerate(ordered_points) if x == GRID_UNITS
            ]
            indices.sort(key=lambda index: ordered_points[index][1])
            parameter_axis = 1
        elif edge == "south":
            indices = [index for index, (_x, y) in enumerate(ordered_points) if y == 0]
            indices.sort(key=lambda index: ordered_points[index][0])
            parameter_axis = 0
        else:
            indices = [
                index for index, (_x, y) in enumerate(ordered_points) if y == GRID_UNITS
            ]
            indices.sort(key=lambda index: ordered_points[index][0])
            parameter_axis = 0
        digest = hashlib.sha256()
        digest.update(b"FVTQEDGE1")
        for index in indices:
            point = ordered_points[index]
            absolute_height = vertices[index][2] + z_origin_mm
            gradient_x, gradient_y = gradients[index]
            digest.update(
                _EDGE_VALUE.pack(
                    point[parameter_axis], absolute_height, gradient_x, gradient_y
                )
            )
        edge_indices.append(tuple(indices))
        signatures.append(digest.digest())
    return vertices, gradients, triangles, tuple(edge_indices), tuple(signatures)


def _edge_signature_from_indices(
    vertices: Sequence[tuple[int, int, int]],
    gradients: Sequence[tuple[int, int]],
    z_origin_mm: int,
    edge_number: int,
    indices: Sequence[int],
) -> bytes:
    edge = EDGE_ORDER[edge_number]
    parameter_axis = 1 if edge in ("west", "east") else 0
    digest = hashlib.sha256()
    digest.update(b"FVTQEDGE1")
    for index in indices:
        point = vertices[index]
        gradient_x, gradient_y = gradients[index]
        digest.update(
            _EDGE_VALUE.pack(
                point[parameter_axis],
                point[2] + z_origin_mm,
                gradient_x,
                gradient_y,
            )
        )
    return digest.digest()


def _triangle_area_grid(
    vertices: Sequence[tuple[int, int, int]], triangle: tuple[int, int, int]
) -> int:
    a, b, c = (vertices[index] for index in triangle)
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _triangle_lattice_error_mm(
    grid: np.ndarray,
    vertices: Sequence[tuple[int, int, int]],
    z_origin_mm: int,
    triangle: tuple[int, int, int],
    coverage: np.ndarray | None = None,
) -> int:
    """Exhaustively compare one final triangle on the 257 x 257 lattice."""

    a, b, c = (vertices[index] for index in triangle)
    area = _triangle_area_grid(vertices, triangle)
    if area <= 0:
        raise ValueError("Final terrain triangle is degenerate or clockwise")
    maximum = 0
    for y in range(min(a[1], b[1], c[1]), max(a[1], b[1], c[1]) + 1):
        for x in range(min(a[0], b[0], c[0]), max(a[0], b[0], c[0]) + 1):
            weight_a = (b[0] - x) * (c[1] - y) - (b[1] - y) * (c[0] - x)
            weight_b = (c[0] - x) * (a[1] - y) - (c[1] - y) * (a[0] - x)
            weight_c = (a[0] - x) * (b[1] - y) - (a[1] - y) * (b[0] - x)
            if weight_a < 0 or weight_b < 0 or weight_c < 0:
                continue
            predicted_relative = _round_div_half_away(
                a[2] * weight_a + b[2] * weight_b + c[2] * weight_c,
                area,
            )
            actual = _sample_height_mm(grid, x * TILE_SIZE_MM, y * TILE_SIZE_MM)
            maximum = max(maximum, abs(actual - (predicted_relative + z_origin_mm)))
            if coverage is not None:
                coverage[y, x] = True
    # The 257 quadtree lattice and the canonical 251-sample source grid are
    # deliberately not coincident (1.953125 m versus 2 m).  Checking only one
    # of them can miss the maximum error that originally triggered refinement.
    source_step = SOURCE_RESOLUTION_MM * GRID_UNITS
    scaled_vertices = tuple(
        (vertex[0] * TILE_SIZE_MM, vertex[1] * TILE_SIZE_MM, vertex[2])
        for vertex in (a, b, c)
    )
    scaled_area = area * TILE_SIZE_MM * TILE_SIZE_MM
    minimum_x = min(vertex[0] for vertex in scaled_vertices)
    maximum_x = max(vertex[0] for vertex in scaled_vertices)
    minimum_y = min(vertex[1] for vertex in scaled_vertices)
    maximum_y = max(vertex[1] for vertex in scaled_vertices)
    first_column = (minimum_x + source_step - 1) // source_step
    last_column = maximum_x // source_step
    first_row = (minimum_y + source_step - 1) // source_step
    last_row = maximum_y // source_step
    scaled_a, scaled_b, scaled_c = scaled_vertices
    for row in range(first_row, last_row + 1):
        y_scaled = row * source_step
        for column in range(first_column, last_column + 1):
            x_scaled = column * source_step
            weight_a = (scaled_b[0] - x_scaled) * (scaled_c[1] - y_scaled) - (
                scaled_b[1] - y_scaled
            ) * (scaled_c[0] - x_scaled)
            weight_b = (scaled_c[0] - x_scaled) * (scaled_a[1] - y_scaled) - (
                scaled_c[1] - y_scaled
            ) * (scaled_a[0] - x_scaled)
            weight_c = (scaled_a[0] - x_scaled) * (scaled_b[1] - y_scaled) - (
                scaled_a[1] - y_scaled
            ) * (scaled_b[0] - x_scaled)
            if weight_a < 0 or weight_b < 0 or weight_c < 0:
                continue
            predicted_relative = _round_div_half_away(
                scaled_a[2] * weight_a
                + scaled_b[2] * weight_b
                + scaled_c[2] * weight_c,
                scaled_area,
            )
            maximum = max(
                maximum,
                abs(int(grid[row, column]) - (predicted_relative + z_origin_mm)),
            )
    return maximum


def _canonical_triangle_key(triangle: tuple[int, int, int]) -> tuple[int, int, int]:
    rotations = (
        triangle,
        (triangle[1], triangle[2], triangle[0]),
        (triangle[2], triangle[0], triangle[1]),
    )
    return min(rotations)


def _point_in_or_on_ccw_triangle(
    point: tuple[int, int],
    a: tuple[int, int],
    b: tuple[int, int],
    c: tuple[int, int],
) -> bool:
    def cross(
        start: tuple[int, int], end: tuple[int, int], value: tuple[int, int]
    ) -> int:
        return (end[0] - start[0]) * (value[1] - start[1]) - (end[1] - start[1]) * (
            value[0] - start[0]
        )

    return (
        cross(a, b, point) >= 0 and cross(b, c, point) >= 0 and cross(c, a, point) >= 0
    )


def _triangulate_simple_polygon(
    vertices: Sequence[tuple[int, int, int]], indices: Sequence[int]
) -> tuple[tuple[int, int, int], ...]:
    """Triangulate one CCW stitch patch with deterministic integer ear clipping."""

    remaining = list(indices)
    if len(remaining) < 3 or len(set(remaining)) != len(remaining):
        raise ValueError("Stitch patch polygon is empty or repeats a vertex")
    polygon_area = sum(
        vertices[remaining[index]][0]
        * vertices[remaining[(index + 1) % len(remaining)]][1]
        - vertices[remaining[(index + 1) % len(remaining)]][0]
        * vertices[remaining[index]][1]
        for index in range(len(remaining))
    )
    if polygon_area <= 0:
        raise ValueError("Stitch patch polygon must be counter-clockwise")
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following = remaining[(position + 1) % len(remaining)]
            candidate = (previous, current, following)
            if _triangle_area_grid(vertices, candidate) <= 0:
                continue
            a, b, c = ((vertices[index][0], vertices[index][1]) for index in candidate)
            if any(
                other not in candidate
                and _point_in_or_on_ccw_triangle(
                    (vertices[other][0], vertices[other][1]), a, b, c
                )
                for other in remaining
            ):
                continue
            triangles.append(candidate)
            del remaining[position]
            break
        else:
            raise ValueError(
                "Stitch patch polygon cannot be triangulated deterministically"
            )
    final_triangle = tuple(remaining)
    if _triangle_area_grid(vertices, final_triangle) <= 0:
        raise ValueError("Stitch patch polygon ended in a degenerate triangle")
    triangles.append(final_triangle)  # type: ignore[arg-type]
    return tuple(triangles)


def _build_stitch_patch_triangles(
    *,
    lod: int,
    mask: int,
    vertices: Sequence[tuple[int, int, int]],
) -> tuple[tuple[int, int, int], ...]:
    """Retriangulate deterministic fine-edge bands against one coarser LOD."""

    if not mask or lod == 2:
        return ()
    fine_width = GRID_UNITS >> LOD_EDGE_DEPTHS[lod]
    coarse_spacing = GRID_UNITS >> LOD_EDGE_DEPTHS[lod + 1]
    point_to_index = {(x, y): index for index, (x, y, _z) in enumerate(vertices)}
    vertical: dict[int, list[int]] = {}
    horizontal: dict[int, list[int]] = {}
    for index, (x, y, _z) in enumerate(vertices):
        vertical.setdefault(x, []).append(index)
        horizontal.setdefault(y, []).append(index)
    for values in vertical.values():
        values.sort(key=lambda index: vertices[index][1])
    for values in horizontal.values():
        values.sort(key=lambda index: vertices[index][0])

    def segment(
        start: tuple[int, int],
        end: tuple[int, int],
        *,
        endpoints_only: bool = False,
    ) -> tuple[int, ...]:
        if endpoints_only:
            try:
                return (point_to_index[start], point_to_index[end])
            except KeyError as error:
                raise AssertionError(
                    "A stitch anchor is absent from the base mesh"
                ) from error
        if start[0] == end[0]:
            candidates = vertical.get(start[0], [])
            axis = 1
        elif start[1] == end[1]:
            candidates = horizontal.get(start[1], [])
            axis = 0
        else:
            raise AssertionError("Stitch patch segments must be axis aligned")
        minimum, maximum = sorted((start[axis], end[axis]))
        selected = [
            index for index in candidates if minimum <= vertices[index][axis] <= maximum
        ]
        if start[axis] > end[axis]:
            selected.reverse()
        if not selected or (
            (vertices[selected[0]][0], vertices[selected[0]][1]) != start
            or (vertices[selected[-1]][0], vertices[selected[-1]][1]) != end
        ):
            raise AssertionError("A stitch patch boundary is absent from the base mesh")
        return tuple(selected)

    polygons: list[tuple[int, ...]] = []

    def add_polygon(
        paths: Sequence[tuple[tuple[int, int], tuple[int, int], bool]],
    ) -> None:
        polygon: list[int] = []
        for start, end, endpoints_only in paths:
            values = segment(start, end, endpoints_only=endpoints_only)
            polygon.extend(values if not polygon else values[1:])
        if polygon[-1] != polygon[0]:
            raise AssertionError("Stitch patch paths do not form a closed polygon")
        polygons.append(tuple(polygon[:-1]))

    west = bool(mask & STITCH_MASK_BITS["west"])
    east = bool(mask & STITCH_MASK_BITS["east"])
    south = bool(mask & STITCH_MASK_BITS["south"])
    north = bool(mask & STITCH_MASK_BITS["north"])

    if west:
        start = coarse_spacing if south else 0
        end = GRID_UNITS - coarse_spacing if north else GRID_UNITS
        for lower in range(start, end, coarse_spacing):
            upper = lower + coarse_spacing
            add_polygon(
                (
                    ((0, lower), (fine_width, lower), False),
                    ((fine_width, lower), (fine_width, upper), False),
                    ((fine_width, upper), (0, upper), False),
                    ((0, upper), (0, lower), True),
                )
            )
    if east:
        inner = GRID_UNITS - fine_width
        start = coarse_spacing if south else 0
        end = GRID_UNITS - coarse_spacing if north else GRID_UNITS
        for lower in range(start, end, coarse_spacing):
            upper = lower + coarse_spacing
            add_polygon(
                (
                    ((inner, lower), (GRID_UNITS, lower), False),
                    ((GRID_UNITS, lower), (GRID_UNITS, upper), True),
                    ((GRID_UNITS, upper), (inner, upper), False),
                    ((inner, upper), (inner, lower), False),
                )
            )
    if south:
        start = coarse_spacing if west else 0
        end = GRID_UNITS - coarse_spacing if east else GRID_UNITS
        for lower in range(start, end, coarse_spacing):
            upper = lower + coarse_spacing
            add_polygon(
                (
                    ((lower, 0), (upper, 0), True),
                    ((upper, 0), (upper, fine_width), False),
                    ((upper, fine_width), (lower, fine_width), False),
                    ((lower, fine_width), (lower, 0), False),
                )
            )
    if north:
        inner = GRID_UNITS - fine_width
        start = coarse_spacing if west else 0
        end = GRID_UNITS - coarse_spacing if east else GRID_UNITS
        for lower in range(start, end, coarse_spacing):
            upper = lower + coarse_spacing
            add_polygon(
                (
                    ((lower, inner), (upper, inner), False),
                    ((upper, inner), (upper, GRID_UNITS), False),
                    ((upper, GRID_UNITS), (lower, GRID_UNITS), True),
                    ((lower, GRID_UNITS), (lower, inner), False),
                )
            )

    if west and south:
        add_polygon(
            (
                ((0, 0), (coarse_spacing, 0), True),
                ((coarse_spacing, 0), (coarse_spacing, fine_width), False),
                ((coarse_spacing, fine_width), (fine_width, fine_width), False),
                ((fine_width, fine_width), (fine_width, coarse_spacing), False),
                ((fine_width, coarse_spacing), (0, coarse_spacing), False),
                ((0, coarse_spacing), (0, 0), True),
            )
        )
    if east and south:
        inner = GRID_UNITS - fine_width
        anchor = GRID_UNITS - coarse_spacing
        add_polygon(
            (
                ((anchor, 0), (GRID_UNITS, 0), True),
                ((GRID_UNITS, 0), (GRID_UNITS, coarse_spacing), True),
                ((GRID_UNITS, coarse_spacing), (inner, coarse_spacing), False),
                ((inner, coarse_spacing), (inner, fine_width), False),
                ((inner, fine_width), (anchor, fine_width), False),
                ((anchor, fine_width), (anchor, 0), False),
            )
        )
    if west and north:
        inner = GRID_UNITS - fine_width
        anchor = GRID_UNITS - coarse_spacing
        add_polygon(
            (
                ((0, anchor), (fine_width, anchor), False),
                ((fine_width, anchor), (fine_width, inner), False),
                ((fine_width, inner), (coarse_spacing, inner), False),
                ((coarse_spacing, inner), (coarse_spacing, GRID_UNITS), False),
                ((coarse_spacing, GRID_UNITS), (0, GRID_UNITS), True),
                ((0, GRID_UNITS), (0, anchor), True),
            )
        )
    if east and north:
        inner = GRID_UNITS - fine_width
        anchor = GRID_UNITS - coarse_spacing
        add_polygon(
            (
                ((anchor, inner), (inner, inner), False),
                ((inner, inner), (inner, anchor), False),
                ((inner, anchor), (GRID_UNITS, anchor), False),
                ((GRID_UNITS, anchor), (GRID_UNITS, GRID_UNITS), True),
                ((GRID_UNITS, GRID_UNITS), (anchor, GRID_UNITS), True),
                ((anchor, GRID_UNITS), (anchor, inner), False),
            )
        )

    return tuple(
        triangle
        for polygon in polygons
        for triangle in _triangulate_simple_polygon(vertices, polygon)
    )


def _materialize_variant_triangles(
    base_triangles: Sequence[tuple[int, int, int]], variant: StitchVariant
) -> tuple[tuple[int, int, int], ...]:
    removed = set(variant.removed_triangle_indices)
    return (
        tuple(
            triangle
            for index, triangle in enumerate(base_triangles)
            if index not in removed
        )
        + variant.replacement_triangles
    )


def _validate_closed_tile_topology(
    vertices: Sequence[tuple[int, int, int]],
    triangles: Sequence[tuple[int, int, int]],
) -> None:
    directed_by_edge: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for triangle in triangles:
        for start, end in zip(triangle, triangle[1:] + triangle[:1], strict=True):
            key = (min(start, end), max(start, end))
            directed_by_edge.setdefault(key, []).append((start, end))
    boundary_lengths = {edge: 0 for edge in EDGE_ORDER}
    for directions in directed_by_edge.values():
        if len(directions) == 2:
            if directions[0] != (directions[1][1], directions[1][0]):
                raise ValueError(
                    "Terrain triangles overlap or disagree on an interior edge"
                )
            continue
        if len(directions) != 1:
            raise ValueError("Terrain edge is referenced by more than two triangles")
        start, end = directions[0]
        a, b = vertices[start], vertices[end]
        if a[0] == b[0] == 0:
            edge, length = "west", abs(a[1] - b[1])
        elif a[0] == b[0] == GRID_UNITS:
            edge, length = "east", abs(a[1] - b[1])
        elif a[1] == b[1] == 0:
            edge, length = "south", abs(a[0] - b[0])
        elif a[1] == b[1] == GRID_UNITS:
            edge, length = "north", abs(a[0] - b[0])
        else:
            raise ValueError("Terrain triangulation contains an internal crack")
        boundary_lengths[edge] += length
    if any(length != GRID_UNITS for length in boundary_lengths.values()):
        raise ValueError(
            "Terrain triangulation does not cover every tile boundary once"
        )


def _build_stitch_variants(
    *,
    lod: int,
    grid: np.ndarray,
    vertices: tuple[tuple[int, int, int], ...],
    gradients: tuple[tuple[int, int], ...],
    triangles: tuple[tuple[int, int, int], ...],
    edge_vertex_indices: tuple[tuple[int, ...], ...],
    z_origin_mm: int,
) -> tuple[tuple[StitchVariant, ...], int]:
    coverage = np.zeros((GRID_UNITS + 1, GRID_UNITS + 1), dtype="bool")
    base_errors = tuple(
        _triangle_lattice_error_mm(grid, vertices, z_origin_mm, triangle, coverage)
        for triangle in triangles
    )
    if not coverage.all():
        raise AssertionError("Final base triangles do not cover the complete lattice")
    base_signatures = tuple(
        _edge_signature_from_indices(
            vertices, gradients, z_origin_mm, edge_number, indices
        )
        for edge_number, indices in enumerate(edge_vertex_indices)
    )
    variants: list[StitchVariant] = []
    for mask in range(16):
        removed: list[int] = []
        replacements = list(
            _build_stitch_patch_triangles(lod=lod, mask=mask, vertices=vertices)
        )
        if mask and lod < 2:
            fine_width = GRID_UNITS >> LOD_EDGE_DEPTHS[lod]
            for triangle_index, triangle in enumerate(triangles):
                points = [vertices[index] for index in triangle]
                in_selected_band = any(
                    (edge == "west" and max(point[0] for point in points) <= fine_width)
                    or (
                        edge == "east"
                        and min(point[0] for point in points) >= GRID_UNITS - fine_width
                    )
                    or (
                        edge == "south"
                        and max(point[1] for point in points) <= fine_width
                    )
                    or (
                        edge == "north"
                        and min(point[1] for point in points) >= GRID_UNITS - fine_width
                    )
                    for edge in EDGE_ORDER
                    if mask & STITCH_MASK_BITS[edge]
                )
                if in_selected_band:
                    removed.append(triangle_index)

        provisional = StitchVariant(
            mask=mask,
            removed_triangle_indices=tuple(removed),
            replacement_triangles=tuple(replacements),
            effective_edge_signatures=(bytes(32),) * 4,
            maximum_error_mm=0,
        )
        materialized = _materialize_variant_triangles(triangles, provisional)
        if sum(
            _triangle_area_grid(vertices, triangle) for triangle in materialized
        ) != (2 * GRID_UNITS * GRID_UNITS):
            raise ValueError("Stitch variant does not cover the exact tile area")
        used_indices = {index for triangle in materialized for index in triangle}
        effective_signatures: list[bytes] = []
        for edge_number, indices in enumerate(edge_vertex_indices):
            effective_indices = tuple(
                index for index in indices if index in used_indices
            )
            effective_signatures.append(
                _edge_signature_from_indices(
                    vertices,
                    gradients,
                    z_origin_mm,
                    edge_number,
                    effective_indices,
                )
            )
            edge = EDGE_ORDER[edge_number]
            expected_indices = tuple(indices)
            if lod < 2 and mask & STITCH_MASK_BITS[edge]:
                parameter_axis = 1 if edge in ("west", "east") else 0
                coarse_spacing = GRID_UNITS >> LOD_EDGE_DEPTHS[lod + 1]
                expected_indices = tuple(
                    index
                    for index in indices
                    if vertices[index][parameter_axis] % coarse_spacing == 0
                )
            if effective_indices != expected_indices:
                raise ValueError(
                    f"Stitch mask {mask} does not expose the canonical {edge} edge"
                )
        removed_set = set(removed)
        unchanged_maximum = max(
            (
                error
                for index, error in enumerate(base_errors)
                if index not in removed_set
            ),
            default=0,
        )
        replacement_maximum = max(
            (
                _triangle_lattice_error_mm(grid, vertices, z_origin_mm, triangle)
                for triangle in replacements
            ),
            default=0,
        )
        variants.append(
            StitchVariant(
                mask=mask,
                removed_triangle_indices=tuple(removed),
                replacement_triangles=tuple(replacements),
                effective_edge_signatures=tuple(effective_signatures),  # type: ignore[arg-type]
                maximum_error_mm=max(unchanged_maximum, replacement_maximum),
            )
        )
    maximum_error = max(variant.maximum_error_mm for variant in variants)
    if variants[0].effective_edge_signatures != base_signatures:
        raise AssertionError("Mask-zero stitch signatures differ from the base mesh")
    return tuple(variants), maximum_error


def materialize_stitch_triangles(
    mesh: FvtqMesh, mask: int
) -> tuple[tuple[int, int, int], ...]:
    if not 0 <= mask < 16:
        raise ValueError("stitch mask must be between 0 and 15")
    return _materialize_variant_triangles(mesh.triangles, mesh.stitch_variants[mask])


def stitch_mask_for_neighbors(lod: int, neighbor_lods: Mapping[str, int | None]) -> int:
    if lod not in (0, 1, 2):
        raise ValueError("terrain LOD must be 0, 1 or 2")
    if set(neighbor_lods) - set(EDGE_ORDER):
        raise ValueError("unknown terrain neighbor edge")
    mask = 0
    for edge in EDGE_ORDER:
        neighbor = neighbor_lods.get(edge)
        if neighbor is None:
            continue
        if neighbor not in (0, 1, 2) or abs(neighbor - lod) > 1:
            raise ValueError("adjacent terrain LOD delta must not exceed one")
        if neighbor == lod + 1:
            mask |= STITCH_MASK_BITS[edge]
    if lod == 2 and mask:
        raise ValueError("LOD2 cannot stitch to a coarser terrain payload")
    return mask


def _normal_halo_hash(normal_halo: np.ndarray) -> bytes:
    digest = hashlib.sha256()
    digest.update(b"FVTQ-NORMAL-HALO-2M-MM-V1\0")
    digest.update(
        struct.pack(
            "<III",
            NORMAL_HALO_SAMPLE_COUNT,
            NORMAL_HALO_SAMPLE_COUNT,
            SOURCE_RESOLUTION_MM,
        )
    )
    digest.update(np.asarray(normal_halo, dtype="<i4").tobytes(order="C"))
    return digest.digest()


def _source_grid_hash(grid: np.ndarray, normal_halo: np.ndarray) -> bytes:
    digest = hashlib.sha256()
    digest.update(b"FVTQ-SOURCE-GRID-AND-NORMAL-HALO-2M-V1\0")
    digest.update(
        struct.pack(
            "<III", SOURCE_SAMPLE_COUNT, SOURCE_SAMPLE_COUNT, SOURCE_RESOLUTION_MM
        )
    )
    digest.update(np.asarray(grid, dtype="<i4").tobytes(order="C"))
    digest.update(_normal_halo_hash(normal_halo))
    return digest.digest()


def _mesh_from_states(
    grid: np.ndarray,
    normal_halo: np.ndarray,
    states: Mapping[tuple[int, int, int], _NodeState],
    tile_origin_mm: tuple[int, int],
    z_origin_mm: int,
    source_hash: bytes,
    normal_halo_hash: bytes,
    contract_hash: bytes,
    breaklines: tuple[Breakline, ...],
    policy: LodPolicy,
    master_heights: Mapping[tuple[int, int], int] | None = None,
    master_gradients: Mapping[tuple[int, int], tuple[int, int]] | None = None,
) -> FvtqMesh:
    ordered_states = sorted(
        states.values(),
        key=lambda state: (
            _node_morton(state.depth, state.x_index, state.y_index),
            state.depth,
        ),
    )
    nodes = tuple(
        QuadtreeNode(
            morton=_node_morton(state.depth, state.x_index, state.y_index),
            depth=state.depth,
            flags=state.flags,
            x_index=state.x_index,
            y_index=state.y_index,
            maximum_error_mm=state.maximum_error_mm,
        )
        for state in ordered_states
    )
    vertices, gradients, triangles, edge_indices, edge_signatures = _build_mesh(
        grid,
        normal_halo,
        states,
        z_origin_mm,
        master_heights,
        master_gradients,
    )
    if len(triangles) > policy.maximum_triangles_with_breaklines:
        raise ValueError(
            f"LOD{policy.lod} base mesh exceeds its explicit "
            f"{policy.maximum_triangles_with_breaklines}-triangle budget "
            "with declared breaklines"
        )
    stitch_variants, maximum_final_error = _build_stitch_variants(
        lod=policy.lod,
        grid=grid,
        vertices=vertices,
        gradients=gradients,
        triangles=triangles,
        edge_vertex_indices=edge_indices,
        z_origin_mm=z_origin_mm,
    )
    maximum_variant_triangles = max(
        variant.triangle_count(len(triangles)) for variant in stitch_variants
    )
    if maximum_variant_triangles > policy.maximum_triangles_with_breaklines:
        raise ValueError(
            f"LOD{policy.lod} final stitch variant exceeds its explicit "
            f"{policy.maximum_triangles_with_breaklines}-triangle budget "
            "with declared breaklines"
        )
    if maximum_final_error > policy.maximum_error_mm:
        raise ValueError(
            f"LOD{policy.lod} final triangulation error is {maximum_final_error} mm, "
            f"limit {policy.maximum_error_mm} mm"
        )
    relative_heights = [vertex[2] for vertex in vertices]
    mesh = FvtqMesh(
        lod=policy.lod,
        tile_origin_mm=tile_origin_mm,
        z_origin_mm=z_origin_mm,
        minimum_relative_height_mm=min(relative_heights),
        maximum_relative_height_mm=max(relative_heights),
        maximum_final_error_mm=maximum_final_error,
        nodes=nodes,
        vertices=vertices,
        vertex_gradients_mm_per_4m=gradients,
        triangles=triangles,
        edge_vertex_indices=edge_indices,
        edge_signatures=edge_signatures,
        stitch_variants=stitch_variants,
        breaklines=breaklines,
        contract_sha256=contract_hash,
        source_grid_sha256=source_hash,
        normal_halo_sha256=normal_halo_hash,
    )
    validate_fvtq_mesh(mesh)
    return mesh


def compile_adaptive_tile(
    heights_m: np.ndarray,
    *,
    normal_halo_heights_m: np.ndarray,
    tile_origin_l93_m: tuple[float, float],
    breaklines: Iterable[Breakline] = (),
    contract_path: Path | None = None,
) -> AdaptiveTerrainTile:
    """Compile the three nested LODs for one exact 500 m Lambert-93 tile."""

    grid = quantize_heights_mm(heights_m)
    normal_halo = quantize_normal_halo_mm(normal_halo_heights_m)
    if not np.array_equal(grid, normal_halo[1:-1, 1:-1]):
        raise ValueError("The normal halo core must exactly match the terrain grid")
    origin_mm = (
        _metres_to_mm(tile_origin_l93_m[0]),
        _metres_to_mm(tile_origin_l93_m[1]),
    )
    if any(coordinate % TILE_SIZE_MM for coordinate in origin_mm):
        raise ValueError("The tile origin must be aligned to the 500 m Lambert-93 grid")
    ordered_breaklines = tuple(
        sorted(breaklines, key=lambda item: (item.feature_id, item.points_mm))
    )
    if len({item.feature_id for item in ordered_breaklines}) != len(ordered_breaklines):
        raise ValueError("Breakline feature ids must be unique within a tile")
    _contract, contract_hash, policies = load_contract(contract_path)
    normal_halo_hash = _normal_halo_hash(normal_halo)
    source_hash = _source_grid_hash(grid, normal_halo)
    z_origin_mm = int(grid.min())
    lod0_states = _initial_tree(grid, ordered_breaklines, policies[0])
    _balance_tree(lod0_states, grid, ordered_breaklines)
    lod0 = _mesh_from_states(
        grid,
        normal_halo,
        lod0_states,
        origin_mm,
        z_origin_mm,
        source_hash,
        normal_halo_hash,
        contract_hash,
        ordered_breaklines,
        policies[0],
    )
    if (
        not ordered_breaklines
        and len(lod0.triangles) > LOD0_TRIANGLE_BUDGET_WITHOUT_BREAKLINES
    ):
        raise ValueError(
            "LOD0 exceeds the 32768-triangle budget without a declared breakline"
        )
    master_heights = {
        (x, y): relative_height + lod0.z_origin_mm
        for x, y, relative_height in lod0.vertices
    }
    master_gradients = {
        (x, y): gradient
        for (x, y, _relative_height), gradient in zip(
            lod0.vertices, lod0.vertex_gradients_mm_per_4m, strict=True
        )
    }
    lod1_states = _collapsed_tree_from_lod0(lod0_states, policies[1])
    lod2_states = _collapsed_tree_from_lod0(lod0_states, policies[2])
    lod1 = _mesh_from_states(
        grid,
        normal_halo,
        lod1_states,
        origin_mm,
        z_origin_mm,
        source_hash,
        normal_halo_hash,
        contract_hash,
        ordered_breaklines,
        policies[1],
        master_heights,
        master_gradients,
    )
    lod2 = _mesh_from_states(
        grid,
        normal_halo,
        lod2_states,
        origin_mm,
        z_origin_mm,
        source_hash,
        normal_halo_hash,
        contract_hash,
        ordered_breaklines,
        policies[2],
        master_heights,
        master_gradients,
    )
    lods = (lod0, lod1, lod2)
    for coarse, fine in ((lods[2], lods[1]), (lods[1], lods[0])):
        coarse_vertices = {(x, y, z) for x, y, z in coarse.vertices}
        fine_vertices = {(x, y, z) for x, y, z in fine.vertices}
        if not coarse_vertices.issubset(fine_vertices):
            raise AssertionError("FVTQ LOD vertices are not strictly nested")
    return AdaptiveTerrainTile(
        tile_origin_mm=origin_mm,
        source_grid_sha256=source_hash,
        normal_halo_sha256=normal_halo_hash,
        contract_sha256=contract_hash,
        lods=lods,  # type: ignore[arg-type]
    )


def _edge_signature(mesh: FvtqMesh, edge_number: int) -> bytes:
    edge = EDGE_ORDER[edge_number]
    digest = hashlib.sha256()
    digest.update(b"FVTQEDGE1")
    for index in mesh.edge_vertex_indices[edge_number]:
        x, y, relative_height = mesh.vertices[index]
        parameter = y if edge in ("west", "east") else x
        gradient_x, gradient_y = mesh.vertex_gradients_mm_per_4m[index]
        digest.update(
            _EDGE_VALUE.pack(
                parameter,
                relative_height + mesh.z_origin_mm,
                gradient_x,
                gradient_y,
            )
        )
    return digest.digest()


def validate_fvtq_mesh(mesh: FvtqMesh) -> None:
    if mesh.lod not in (0, 1, 2):
        raise ValueError("FVTQ LOD must be 0, 1 or 2")
    if any(coordinate % TILE_SIZE_MM for coordinate in mesh.tile_origin_mm):
        raise ValueError("FVTQ tile origin must align to the 500 m Lambert-93 grid")
    if (
        len(mesh.contract_sha256) != 32
        or len(mesh.source_grid_sha256) != 32
        or len(mesh.normal_halo_sha256) != 32
    ):
        raise ValueError("FVTQ hashes must contain 32 bytes")
    expected_nodes = sorted(mesh.nodes, key=lambda node: (node.morton, node.depth))
    if list(mesh.nodes) != expected_nodes or len(
        {(node.depth, node.x_index, node.y_index) for node in mesh.nodes}
    ) != len(mesh.nodes):
        raise ValueError("FVTQ nodes are not unique Morton-ordered nodes")
    for node in mesh.nodes:
        if node.flags & ~NODE_FLAGS_MASK:
            raise ValueError("FVTQ node contains unknown flags")
        if node.is_leaf:
            if node.flags != NODE_LEAF:
                raise ValueError("FVTQ leaf node cannot carry split flags")
        elif node.flags == 0:
            raise ValueError("FVTQ internal node must record a split reason")
    if not mesh.vertices or not mesh.triangles:
        raise ValueError("FVTQ terrain geometry is empty")
    if len(set(mesh.vertices)) != len(mesh.vertices):
        raise ValueError("FVTQ vertices are not unique")
    if len(mesh.vertex_gradients_mm_per_4m) != len(mesh.vertices):
        raise ValueError("FVTQ must carry one halo gradient per vertex")
    for x, y, relative_height in mesh.vertices:
        if not (0 <= x <= GRID_UNITS and 0 <= y <= GRID_UNITS):
            raise ValueError("FVTQ vertex lies outside the tile")
        if not np.iinfo("int32").min <= relative_height <= np.iinfo("int32").max:
            raise ValueError("FVTQ relative height exceeds int32")
    for gradient_x, gradient_y in mesh.vertex_gradients_mm_per_4m:
        if not (
            np.iinfo("int32").min <= gradient_x <= np.iinfo("int32").max
            and np.iinfo("int32").min <= gradient_y <= np.iinfo("int32").max
        ):
            raise ValueError("FVTQ terrain gradient exceeds int32")
    for triangle in mesh.triangles:
        if len(set(triangle)) != 3 or any(
            index < 0 or index >= len(mesh.vertices) for index in triangle
        ):
            raise ValueError("FVTQ triangle indices are invalid")
        a, b, c = (mesh.vertices[index] for index in triangle)
        signed_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if signed_area <= 0:
            raise ValueError(
                "FVTQ triangles must be non-degenerate and counter-clockwise"
            )
    if len(mesh.edge_vertex_indices) != 4 or len(mesh.edge_signatures) != 4:
        raise ValueError("FVTQ must carry four edge lists and signatures")
    for edge_number, indices in enumerate(mesh.edge_vertex_indices):
        if tuple(indices) != tuple(dict.fromkeys(indices)) or not indices:
            raise ValueError("FVTQ edge indices are empty or duplicated")
        edge = EDGE_ORDER[edge_number]
        coordinates = [mesh.vertices[index] for index in indices]
        parameters = [
            value[1] if edge in ("west", "east") else value[0] for value in coordinates
        ]
        if (
            parameters != sorted(parameters)
            or parameters[0] != 0
            or parameters[-1] != GRID_UNITS
        ):
            raise ValueError("FVTQ edge vertices are not globally ordered and complete")
        if edge == "west" and any(value[0] != 0 for value in coordinates):
            raise ValueError("FVTQ west edge list is invalid")
        if edge == "east" and any(value[0] != GRID_UNITS for value in coordinates):
            raise ValueError("FVTQ east edge list is invalid")
        if edge == "south" and any(value[1] != 0 for value in coordinates):
            raise ValueError("FVTQ south edge list is invalid")
        if edge == "north" and any(value[1] != GRID_UNITS for value in coordinates):
            raise ValueError("FVTQ north edge list is invalid")
        if mesh.edge_signatures[edge_number] != _edge_signature(mesh, edge_number):
            raise ValueError("FVTQ edge signature does not match its vertices")
    heights = [vertex[2] for vertex in mesh.vertices]
    if (
        min(heights) != mesh.minimum_relative_height_mm
        or max(heights) != mesh.maximum_relative_height_mm
    ):
        raise ValueError("FVTQ height bounds are inconsistent")
    if not 0 <= mesh.maximum_final_error_mm <= np.iinfo("uint32").max:
        raise ValueError("FVTQ final terrain error exceeds uint32")
    if len(mesh.stitch_variants) != 16 or tuple(
        variant.mask for variant in mesh.stitch_variants
    ) != tuple(range(16)):
        raise ValueError("FVTQ must carry stitch variants ordered by mask 0..15")
    materialized_areas: list[int] = []
    for variant in mesh.stitch_variants:
        if not 0 <= variant.maximum_error_mm <= np.iinfo("uint32").max:
            raise ValueError("FVTQ stitch error exceeds uint32")
        removed = variant.removed_triangle_indices
        if removed != tuple(sorted(set(removed))) or any(
            index < 0 or index >= len(mesh.triangles) for index in removed
        ):
            raise ValueError("FVTQ stitch removals are invalid")
        if len(variant.effective_edge_signatures) != 4 or any(
            len(signature) != 32 for signature in variant.effective_edge_signatures
        ):
            raise ValueError(
                "FVTQ stitch edge signatures must contain four SHA-256 values"
            )
        for triangle in variant.replacement_triangles:
            if len(set(triangle)) != 3 or any(
                index < 0 or index >= len(mesh.vertices) for index in triangle
            ):
                raise ValueError("FVTQ stitch replacement indices are invalid")
            if _triangle_area_grid(mesh.vertices, triangle) <= 0:
                raise ValueError("FVTQ stitch replacements must be counter-clockwise")
        materialized = materialize_stitch_triangles(mesh, variant.mask)
        _validate_closed_tile_topology(mesh.vertices, materialized)
        materialized_areas.append(
            sum(
                _triangle_area_grid(mesh.vertices, triangle)
                for triangle in materialized
            )
        )
        used_indices = {index for triangle in materialized for index in triangle}
        expected_used_edges: list[tuple[int, ...]] = []
        for edge_number, indices in enumerate(mesh.edge_vertex_indices):
            edge = EDGE_ORDER[edge_number]
            if mesh.lod < 2 and variant.mask & STITCH_MASK_BITS[edge]:
                parameter_axis = 1 if edge in ("west", "east") else 0
                coarse_spacing = GRID_UNITS >> LOD_EDGE_DEPTHS[mesh.lod + 1]
                expected_used_edges.append(
                    tuple(
                        index
                        for index in indices
                        if mesh.vertices[index][parameter_axis] % coarse_spacing == 0
                    )
                )
            else:
                expected_used_edges.append(tuple(indices))
        observed_used_edges = [
            tuple(index for index in indices if index in used_indices)
            for indices in mesh.edge_vertex_indices
        ]
        if observed_used_edges != expected_used_edges:
            raise ValueError("FVTQ stitch mask exposes an invalid boundary vertex set")
        observed_signatures = tuple(
            _edge_signature_from_indices(
                mesh.vertices,
                mesh.vertex_gradients_mm_per_4m,
                mesh.z_origin_mm,
                edge_number,
                tuple(index for index in indices if index in used_indices),
            )
            for edge_number, indices in enumerate(mesh.edge_vertex_indices)
        )
        if observed_signatures != variant.effective_edge_signatures:
            raise ValueError("FVTQ stitch edge signature does not match its triangles")
    if any(area != 2 * GRID_UNITS * GRID_UNITS for area in materialized_areas):
        raise ValueError("FVTQ stitch variant does not cover the exact tile area")
    base_variant = mesh.stitch_variants[0]
    if base_variant.removed_triangle_indices or base_variant.replacement_triangles:
        raise ValueError("FVTQ stitch mask zero must preserve the base triangles")
    if base_variant.effective_edge_signatures != mesh.edge_signatures:
        raise ValueError("FVTQ stitch mask zero must preserve base edge signatures")
    if mesh.maximum_final_error_mm != max(
        variant.maximum_error_mm for variant in mesh.stitch_variants
    ):
        raise ValueError(
            "FVTQ maximum final error is inconsistent with stitch variants"
        )


def encode_fvtq(mesh: FvtqMesh) -> bytes:
    validate_fvtq_mesh(mesh)
    body = bytearray()
    for node in mesh.nodes:
        body.extend(
            _NODE.pack(
                node.morton,
                node.depth,
                node.flags,
                node.x_index,
                node.y_index,
                node.maximum_error_mm,
            )
        )
    for vertex in mesh.vertices:
        body.extend(_VERTEX.pack(*vertex))
    for gradient in mesh.vertex_gradients_mm_per_4m:
        body.extend(_GRADIENT.pack(*gradient))
    for triangle in mesh.triangles:
        body.extend(_TRIANGLE.pack(*triangle))
    for indices in mesh.edge_vertex_indices:
        for index in indices:
            body.extend(struct.pack("<I", index))
    for breakline in mesh.breaklines:
        feature_id = breakline.feature_id.encode("utf-8")
        if (
            len(feature_id) > np.iinfo("uint16").max
            or len(breakline.points_mm) > np.iinfo("uint16").max
        ):
            raise ValueError("FVTQ breakline exceeds v1 binary limits")
        body.extend(_CONSTRAINT_HEADER.pack(len(feature_id), len(breakline.points_mm)))
        body.extend(feature_id)
        for point in breakline.points_mm:
            body.extend(_CONSTRAINT_POINT.pack(*point))
    for variant in mesh.stitch_variants:
        body.extend(
            _STITCH_HEADER.pack(
                variant.mask,
                variant.maximum_error_mm,
                len(variant.removed_triangle_indices),
                len(variant.replacement_triangles),
                b"".join(variant.effective_edge_signatures),
            )
        )
        for triangle_index in variant.removed_triangle_indices:
            body.extend(struct.pack("<I", triangle_index))
        for triangle in variant.replacement_triangles:
            body.extend(_TRIANGLE.pack(*triangle))
    edge_blob = b"".join(mesh.edge_signatures)
    header_values = (
        FVTQ_MAGIC,
        FVTQ_VERSION,
        mesh.lod,
        0,
        mesh.tile_origin_mm[0],
        mesh.tile_origin_mm[1],
        TILE_SIZE_MM,
        mesh.z_origin_mm,
        mesh.minimum_relative_height_mm,
        mesh.maximum_relative_height_mm,
        mesh.maximum_final_error_mm,
        len(mesh.nodes),
        len(mesh.vertices),
        len(mesh.triangles),
        len(mesh.breaklines),
        *(len(indices) for indices in mesh.edge_vertex_indices),
        mesh.contract_sha256,
        mesh.source_grid_sha256,
        mesh.normal_halo_sha256,
        edge_blob,
    )
    header_without_digest = _HEADER.pack(*header_values, bytes(32))
    file_digest = hashlib.sha256(header_without_digest + body).digest()
    return _HEADER.pack(*header_values, file_digest) + body


def write_fvtq(mesh: FvtqMesh, path: Path) -> bytes:
    payload = encode_fvtq(mesh)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return hashlib.sha256(payload).digest()


def _take(payload: bytes, offset: int, size: int) -> tuple[bytes, int]:
    end = offset + size
    if end > len(payload):
        raise ValueError("Truncated FVTQ payload")
    return payload[offset:end], end


def decode_fvtq(payload: bytes) -> FvtqMesh:
    header_raw, offset = _take(payload, 0, _HEADER.size)
    unpacked = _HEADER.unpack(header_raw)
    (
        magic,
        version,
        lod,
        flags,
        origin_x,
        origin_y,
        tile_size,
        z_origin,
        minimum_height,
        maximum_height,
        maximum_final_error,
        node_count,
        vertex_count,
        triangle_count,
        constraint_count,
        west_count,
        east_count,
        south_count,
        north_count,
        contract_hash,
        source_hash,
        normal_halo_hash,
        edge_blob,
        observed_digest,
    ) = unpacked
    if (
        magic != FVTQ_MAGIC
        or version != FVTQ_VERSION
        or flags != 0
        or tile_size != TILE_SIZE_MM
    ):
        raise ValueError("Unsupported FVTQ header")
    zeroed_header = _HEADER.pack(*unpacked[:-1], bytes(32))
    if (
        hashlib.sha256(zeroed_header + payload[_HEADER.size :]).digest()
        != observed_digest
    ):
        raise ValueError("FVTQ file digest mismatch")
    nodes: list[QuadtreeNode] = []
    for _ in range(node_count):
        raw, offset = _take(payload, offset, _NODE.size)
        morton, depth, node_flags, x_index, y_index, maximum_error = _NODE.unpack(raw)
        nodes.append(
            QuadtreeNode(morton, depth, node_flags, x_index, y_index, maximum_error)
        )
    vertices: list[tuple[int, int, int]] = []
    for _ in range(vertex_count):
        raw, offset = _take(payload, offset, _VERTEX.size)
        vertices.append(_VERTEX.unpack(raw))
    gradients: list[tuple[int, int]] = []
    for _ in range(vertex_count):
        raw, offset = _take(payload, offset, _GRADIENT.size)
        gradients.append(_GRADIENT.unpack(raw))
    triangles: list[tuple[int, int, int]] = []
    for _ in range(triangle_count):
        raw, offset = _take(payload, offset, _TRIANGLE.size)
        triangles.append(_TRIANGLE.unpack(raw))
    edge_indices: list[tuple[int, ...]] = []
    for count in (west_count, east_count, south_count, north_count):
        values: list[int] = []
        for _ in range(count):
            raw, offset = _take(payload, offset, 4)
            values.append(struct.unpack("<I", raw)[0])
        edge_indices.append(tuple(values))
    breaklines: list[Breakline] = []
    for _ in range(constraint_count):
        raw, offset = _take(payload, offset, _CONSTRAINT_HEADER.size)
        id_length, point_count = _CONSTRAINT_HEADER.unpack(raw)
        raw_id, offset = _take(payload, offset, id_length)
        points: list[tuple[int, int]] = []
        for _ in range(point_count):
            raw_point, offset = _take(payload, offset, _CONSTRAINT_POINT.size)
            points.append(_CONSTRAINT_POINT.unpack(raw_point))
        try:
            feature_id = raw_id.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("FVTQ breakline id is not UTF-8") from error
        breaklines.append(Breakline(feature_id, tuple(points)))
    stitch_variants: list[StitchVariant] = []
    for expected_mask in range(16):
        raw, offset = _take(payload, offset, _STITCH_HEADER.size)
        (
            mask,
            maximum_error,
            removed_count,
            replacement_count,
            effective_edge_blob,
        ) = _STITCH_HEADER.unpack(raw)
        if mask != expected_mask:
            raise ValueError("FVTQ stitch variants are not ordered by mask")
        removed: list[int] = []
        for _ in range(removed_count):
            raw_index, offset = _take(payload, offset, 4)
            removed.append(struct.unpack("<I", raw_index)[0])
        replacements: list[tuple[int, int, int]] = []
        for _ in range(replacement_count):
            raw_triangle, offset = _take(payload, offset, _TRIANGLE.size)
            replacements.append(_TRIANGLE.unpack(raw_triangle))
        stitch_variants.append(
            StitchVariant(
                mask=mask,
                removed_triangle_indices=tuple(removed),
                replacement_triangles=tuple(replacements),
                effective_edge_signatures=tuple(
                    effective_edge_blob[index * 32 : (index + 1) * 32]
                    for index in range(4)
                ),  # type: ignore[arg-type]
                maximum_error_mm=maximum_error,
            )
        )
    if offset != len(payload):
        raise ValueError("FVTQ contains an unknown trailing payload")
    signatures = tuple(edge_blob[index * 32 : (index + 1) * 32] for index in range(4))
    mesh = FvtqMesh(
        lod=lod,
        tile_origin_mm=(origin_x, origin_y),
        z_origin_mm=z_origin,
        minimum_relative_height_mm=minimum_height,
        maximum_relative_height_mm=maximum_height,
        maximum_final_error_mm=maximum_final_error,
        nodes=tuple(nodes),
        vertices=tuple(vertices),
        vertex_gradients_mm_per_4m=tuple(gradients),
        triangles=tuple(triangles),
        edge_vertex_indices=tuple(edge_indices),  # type: ignore[arg-type]
        edge_signatures=signatures,  # type: ignore[arg-type]
        stitch_variants=tuple(stitch_variants),
        breaklines=tuple(breaklines),
        contract_sha256=contract_hash,
        source_grid_sha256=source_hash,
        normal_halo_sha256=normal_halo_hash,
    )
    validate_fvtq_mesh(mesh)
    return mesh


def read_fvtq(path: Path) -> FvtqMesh:
    return decode_fvtq(path.read_bytes())


__all__ = [
    "AdaptiveTerrainTile",
    "Breakline",
    "CONTRACT_SCHEMA",
    "EDGE_ORDER",
    "FVTQ_SCHEMA",
    "FvtqMesh",
    "GRID_UNITS",
    "LodPolicy",
    "LOD0_TRIANGLE_BUDGET_WITHOUT_BREAKLINES",
    "LOD_MAXIMUM_TRIANGLES_WITH_BREAKLINES",
    "NORMAL_HALO_SAMPLE_COUNT",
    "QuadtreeNode",
    "STITCH_MASK_BITS",
    "StitchVariant",
    "SOURCE_SAMPLE_COUNT",
    "compile_adaptive_tile",
    "decode_fvtq",
    "encode_fvtq",
    "load_contract",
    "materialize_stitch_triangles",
    "quantize_heights_mm",
    "quantize_normal_halo_mm",
    "read_fvtq",
    "stitch_mask_for_neighbors",
    "validate_fvtq_mesh",
    "write_fvtq",
]
