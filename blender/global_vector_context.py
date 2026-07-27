"""Partition lightweight global vectors on the spatial LOD tile grid.

The near LOD replaces only a bounded set of 500 m cells.  Global roads, water
and buildings therefore cannot stay as monolithic Blender collections: hiding
the monolith would remove context outside the detailed footprint, while
keeping it would overlap the detailed vectors.  This module creates one
exclusive global-vector owner per manifest cell plus a residual owner for
features outside the irregular manifest mask.

Road and water triangle meshes are clipped exactly on the grid through the
same conservative partition used by the global terrain.  Buildings are kept
as complete prisms and assigned once by their outer-ring centroid.  A 500 m
cell is much larger than a building footprint; keeping the prism intact avoids
introducing invalid roof polygons while still providing deterministic,
non-duplicated simple-model ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from global_terrain_context import (
    GLOBAL_CONTEXT_CHUNK_ID,
    GlobalTerrainPartition,
    _manifest_grid,
    _origin,
    partition_global_terrain,
)


GLOBAL_VECTOR_CONTEXT_SCHEMA = "fireviewer.global-vector-context.v1"


@dataclass(frozen=True)
class PrismOwnershipPartition:
    """Exactly-once ownership of simple building prisms."""

    manifest_tile_ids: tuple[str, ...]
    prisms_by_tile: Mapping[str, tuple[Mapping[str, Any], ...]]
    context_prisms: tuple[Mapping[str, Any], ...]
    source_prism_count: int

    @property
    def assigned_prism_count(self) -> int:
        return sum(len(items) for items in self.prisms_by_tile.values())

    @property
    def is_complete(self) -> bool:
        return (
            self.assigned_prism_count + len(self.context_prisms)
            == self.source_prism_count
        )


def _outer_ring(prism: Mapping[str, Any], index: int) -> Sequence[Sequence[float]]:
    rings = prism.get("rings")
    if not isinstance(rings, Sequence) or isinstance(rings, (str, bytes)) or not rings:
        raise ValueError(f"buildings.prisms[{index}].rings must be non-empty")
    ring = rings[0]
    if not isinstance(ring, Sequence) or isinstance(ring, (str, bytes)):
        raise ValueError(f"buildings.prisms[{index}].rings[0] must be a sequence")
    if len(ring) < 3:
        raise ValueError(
            f"buildings.prisms[{index}] outer ring has fewer than 3 points"
        )
    return ring


def _ring_centroid(
    ring: Sequence[Sequence[float]], field_name: str
) -> tuple[float, float]:
    """Return a stable polygon centroid, falling back to the vertex mean."""

    coordinates: list[tuple[float, float]] = []
    for point_index, point in enumerate(ring):
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)):
            raise ValueError(f"{field_name}[{point_index}] must contain X and Y")
        if len(point) != 2:
            raise ValueError(f"{field_name}[{point_index}] must contain X and Y")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"{field_name}[{point_index}] must be finite")
        coordinates.append((x, y))

    twice_area = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    for left, right in zip(
        coordinates, (*coordinates[1:], coordinates[0]), strict=True
    ):
        cross = left[0] * right[1] - right[0] * left[1]
        twice_area += cross
        weighted_x += (left[0] + right[0]) * cross
        weighted_y += (left[1] + right[1]) * cross
    if abs(twice_area) <= 1e-12:
        return (
            sum(point[0] for point in coordinates) / len(coordinates),
            sum(point[1] for point in coordinates) / len(coordinates),
        )
    return weighted_x / (3.0 * twice_area), weighted_y / (3.0 * twice_area)


def partition_building_prisms(
    prisms: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    origin_l93_m: Sequence[float],
) -> PrismOwnershipPartition:
    """Assign every complete simple building prism to exactly one grid owner."""

    if not isinstance(prisms, Sequence) or isinstance(prisms, (str, bytes)):
        raise ValueError("buildings.prisms must be a sequence")
    origin = _origin(origin_l93_m)
    grid = _manifest_grid(manifest, origin)
    owned: dict[str, list[Mapping[str, Any]]] = {
        tile_id: [] for tile_id in sorted(grid.tile_bounds)
    }
    context: list[Mapping[str, Any]] = []
    for index, prism in enumerate(prisms):
        if not isinstance(prism, Mapping):
            raise ValueError(f"buildings.prisms[{index}] must be an object")
        local_x, local_y = _ring_centroid(
            _outer_ring(prism, index), f"buildings.prisms[{index}].rings[0]"
        )
        east_m = origin[0] + local_x
        north_m = origin[1] + local_y
        cell = (
            math.floor((east_m - grid.anchor_east_m) / grid.tile_size_m),
            math.floor((north_m - grid.anchor_north_m) / grid.tile_size_m),
        )
        tile = grid.tiles_by_cell.get(cell)
        (context if tile is None else owned[tile[0]]).append(prism)

    partition = PrismOwnershipPartition(
        manifest_tile_ids=tuple(sorted(owned)),
        prisms_by_tile=MappingProxyType(
            {tile_id: tuple(items) for tile_id, items in owned.items()}
        ),
        context_prisms=tuple(context),
        source_prism_count=len(prisms),
    )
    if not partition.is_complete:
        raise AssertionError("Building prism ownership is incomplete")
    return partition


def partition_vector_surface(
    mesh: Mapping[str, Any],
    manifest: Mapping[str, Any],
    origin_l93_m: Sequence[float],
) -> GlobalTerrainPartition:
    """Clip one triangle surface layer into exact, exclusive grid chunks."""

    partition = partition_global_terrain(mesh, manifest, origin_l93_m)
    if not partition.validation.is_complete:
        raise ValueError("Global vector surface partition is incomplete")
    return partition


def vector_context_visibility(
    manifest_tile_ids: Sequence[str], detailed_tile_ids: Sequence[str]
) -> Mapping[str, bool]:
    """Return exact simple-vector visibility for a detailed publication."""

    known = {str(tile_id) for tile_id in manifest_tile_ids}
    detailed = {str(tile_id) for tile_id in detailed_tile_ids}
    unknown = sorted(detailed.difference(known))
    if unknown:
        raise ValueError(f"Unknown detailed tile ids: {unknown}")
    result = {tile_id: tile_id not in detailed for tile_id in sorted(known)}
    result[GLOBAL_CONTEXT_CHUNK_ID] = True
    return MappingProxyType(result)


__all__ = [
    "GLOBAL_VECTOR_CONTEXT_SCHEMA",
    "PrismOwnershipPartition",
    "partition_building_prisms",
    "partition_vector_surface",
    "vector_context_visibility",
]
