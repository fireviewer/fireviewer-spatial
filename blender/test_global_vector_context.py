from __future__ import annotations

import pytest

from global_terrain_context import GLOBAL_CONTEXT_CHUNK_ID
from global_vector_context import (
    partition_building_prisms,
    partition_vector_surface,
    vector_context_visibility,
)


ORIGIN = [1000.0, 2000.0, 50.0]
MANIFEST = {
    "crs": "EPSG:2154",
    "origin_l93_m": ORIGIN,
    "tiling": {"output_tile_size_m": 10.0},
    "tiles": [
        {"id": "west", "bounds_l93_m": [1000.0, 2000.0, 1010.0, 2010.0]},
        {"id": "east", "bounds_l93_m": [1010.0, 2000.0, 1020.0, 2010.0]},
    ],
}


def _prism_at(x: float, y: float) -> dict[str, object]:
    return {
        "base_z": 1.0,
        "height": 4.0,
        "rings": [[[x, y], [x + 1.0, y], [x + 1.0, y + 1.0], [x, y + 1.0]]],
    }


def test_building_prisms_have_exactly_one_owner_including_residual_context() -> None:
    west = _prism_at(1.0, 1.0)
    east = _prism_at(11.0, 1.0)
    outside = _prism_at(1.0, 11.0)

    partition = partition_building_prisms([west, east, outside], MANIFEST, ORIGIN)

    assert partition.is_complete
    assert partition.source_prism_count == 3
    assert partition.prisms_by_tile["west"] == (west,)
    assert partition.prisms_by_tile["east"] == (east,)
    assert partition.context_prisms == (outside,)


def test_vector_surface_is_clipped_without_loss_or_overlap() -> None:
    mesh = {
        "vertices": [
            [5.0, 2.0, 3.0],
            [15.0, 2.0, 3.0],
            [15.0, 8.0, 4.0],
        ],
        "faces": [[0, 1, 2]],
    }

    partition = partition_vector_surface(mesh, MANIFEST, ORIGIN)

    assert partition.validation.is_complete
    assert partition.tile_chunks["west"].faces
    assert partition.tile_chunks["east"].faces
    assert not partition.context_chunk.faces
    assert partition.validation.total_partitioned_area_m2 == pytest.approx(
        partition.validation.total_source_area_m2, abs=1e-9
    )


def test_visibility_keeps_context_and_excludes_only_published_cells() -> None:
    visibility = vector_context_visibility(["west", "east"], ["west"])

    assert visibility == {
        "west": False,
        "east": True,
        GLOBAL_CONTEXT_CHUNK_ID: True,
    }


def test_visibility_rejects_unknown_detailed_cell() -> None:
    with pytest.raises(ValueError, match="Unknown detailed tile ids"):
        vector_context_visibility(["west", "east"], ["north"])
