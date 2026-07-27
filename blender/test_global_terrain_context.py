from __future__ import annotations

import math

import pytest

from global_terrain_context import (
    GLOBAL_CONTEXT_CHUNK_ID,
    global_context_visibility,
    partition_global_terrain,
    validate_global_terrain_partition,
)


ORIGIN = (1_000.0, 2_000.0, 100.0)


def _tile(identifier: str, west: float, south: float) -> dict[str, object]:
    return {
        "id": identifier,
        "bounds_l93_m": [west, south, west + 10.0, south + 10.0],
    }


def _manifest(*tiles: dict[str, object]) -> dict[str, object]:
    return {
        "crs": "EPSG:2154",
        "origin_l93_m": list(ORIGIN),
        "tiling": {"output_tile_size_m": 10.0},
        "tiles": list(tiles),
    }


def _triangle_area(
    vertices: tuple[tuple[float, float, float], ...], face: tuple[int, ...]
) -> float:
    a, b, c = (vertices[index] for index in face)
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) * 0.5


def test_partition_cuts_crossing_quad_and_conserves_faces_and_uvs() -> None:
    # One 16 x 8 m terrain quad crosses the exact X=1010 tile boundary.  Its
    # NW-SE diagonal is intentionally non-planar so the split must retain the
    # explicit render triangulation and interpolate Z/UV on that surface.
    terrain = {
        "vertices": [
            [2.0, 9.0, 1.0],
            [2.0, 1.0, 2.0],
            [18.0, 1.0, 5.0],
            [18.0, 9.0, 9.0],
        ],
        "faces": [[0, 1, 2, 3]],
    }
    loop_uvs = [(0.1, 0.9), (0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]
    partition = partition_global_terrain(
        terrain,
        _manifest(
            _tile("west", 1_000.0, 2_000.0),
            _tile("east", 1_010.0, 2_000.0),
        ),
        ORIGIN,
        loop_uvs=loop_uvs,
    )

    validation = validate_global_terrain_partition(partition)
    assert validation.is_complete
    assert validation.source_face_count == 1
    assert validation.source_triangle_count == 2
    assert validation.covered_source_face_count == 1
    assert validation.lost_source_face_indices == ()
    assert validation.duplicated_source_face_indices == ()
    assert math.isclose(validation.total_source_area_m2, 128.0, abs_tol=1e-8)
    assert math.isclose(
        validation.total_partitioned_area_m2,
        validation.total_source_area_m2,
        abs_tol=1e-8,
    )
    assert partition.context_chunk.is_empty

    west = partition.tile_chunks["west"]
    east = partition.tile_chunks["east"]
    assert west.faces and east.faces
    assert west.loop_uvs is not None and len(west.loop_uvs) == len(west.faces) * 3
    assert east.loop_uvs is not None and len(east.loop_uvs) == len(east.faces) * 3
    assert all(vertex[0] <= 10.0 + 1e-9 for vertex in west.vertices)
    assert all(vertex[0] >= 10.0 - 1e-9 for vertex in east.vertices)
    boundary_vertices = {
        (round(vertex[1], 6), round(vertex[2], 6))
        for chunk in (west, east)
        for vertex in chunk.vertices
        if math.isclose(vertex[0], 10.0, abs_tol=1e-9)
    }
    assert boundary_vertices == {(1.0, 3.5), (5.0, 3.0), (9.0, 5.0)}
    boundary_uvs = {
        (round(uv[0], 6), round(uv[1], 6))
        for chunk in (west, east)
        for face_index, face in enumerate(chunk.faces)
        for corner, vertex_index in enumerate(face)
        if math.isclose(chunk.vertices[vertex_index][0], 10.0, abs_tol=1e-9)
        for uv in (chunk.loop_uvs[face_index * 3 + corner],)
    }
    assert boundary_uvs == {(0.5, 0.1), (0.5, 0.5), (0.5, 0.9)}
    assert math.isclose(
        sum(_triangle_area(west.vertices, face) for face in west.faces),
        64.0,
        abs_tol=1e-8,
    )
    assert math.isclose(
        sum(_triangle_area(east.vertices, face) for face in east.faces),
        64.0,
        abs_tol=1e-8,
    )


def test_unmanifested_grid_remainder_stays_in_always_visible_context() -> None:
    terrain = {
        "vertices": [
            [-5.0, 8.0, 0.0],
            [-5.0, 2.0, 0.0],
            [5.0, 2.0, 0.0],
            [5.0, 8.0, 0.0],
        ],
        "faces": [[0, 1, 2, 3]],
    }
    partition = partition_global_terrain(
        terrain,
        _manifest(_tile("inside", 1_000.0, 2_000.0)),
        ORIGIN,
        texture_bounds_l93_m=(990.0, 1_990.0, 1_020.0, 2_020.0),
    )

    assert partition.tile_chunks["inside"].faces
    assert partition.context_chunk.faces
    assert partition.context_chunk.loop_uvs is not None
    assert math.isclose(
        partition.validation.total_source_area_m2,
        partition.validation.total_partitioned_area_m2,
        abs_tol=1e-8,
    )
    visibility = global_context_visibility(partition, ["inside"])
    assert visibility["inside"] is False
    assert visibility[GLOBAL_CONTEXT_CHUNK_ID] is True


def test_visibility_keeps_all_other_complete_global_chunks() -> None:
    terrain = {
        "vertices": [
            [1.0, 9.0, 0.0],
            [1.0, 1.0, 0.0],
            [19.0, 1.0, 0.0],
            [19.0, 9.0, 0.0],
        ],
        "faces": [[0, 1, 2, 3]],
    }
    partition = partition_global_terrain(
        terrain,
        _manifest(
            _tile("west", 1_000.0, 2_000.0),
            _tile("east", 1_010.0, 2_000.0),
            _tile("empty", 1_020.0, 2_000.0),
        ),
        ORIGIN,
    )

    visibility = global_context_visibility(partition, ["west"])
    assert visibility["west"] is False
    assert visibility["east"] is True
    assert visibility["empty"] is False
    assert visibility[GLOBAL_CONTEXT_CHUNK_ID] is False
    with pytest.raises(ValueError, match="Unknown detailed tile ids"):
        global_context_visibility(partition, ["missing"])


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (
            _manifest(
                _tile("duplicate-cell-a", 1_000.0, 2_000.0),
                _tile("duplicate-cell-b", 1_000.0, 2_000.0),
            ),
            "occupy manifest grid cell",
        ),
        (
            {
                **_manifest(_tile("misaligned", 1_000.5, 2_000.0)),
                "tiles": [
                    _tile("anchor", 1_000.0, 2_000.0),
                    _tile("misaligned", 1_010.5, 2_000.0),
                ],
            },
            "not aligned",
        ),
    ],
)
def test_invalid_manifest_grid_fails_closed(
    manifest: dict[str, object], message: str
) -> None:
    terrain = {
        "vertices": [[1.0, 2.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        "faces": [[0, 1, 2]],
    }
    with pytest.raises(ValueError, match=message):
        partition_global_terrain(terrain, manifest, ORIGIN)


def test_invalid_mesh_and_uv_contracts_fail_closed() -> None:
    manifest = _manifest(_tile("tile", 1_000.0, 2_000.0))
    degenerate = {
        "vertices": [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]],
        "faces": [[0, 1, 2]],
    }
    with pytest.raises(ValueError, match="degenerate triangle"):
        partition_global_terrain(degenerate, manifest, ORIGIN)

    terrain = {
        "vertices": [[1.0, 2.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        "faces": [[0, 1, 2]],
    }
    with pytest.raises(ValueError, match="expected 3"):
        partition_global_terrain(terrain, manifest, ORIGIN, loop_uvs=[(0.0, 0.0)])
    with pytest.raises(ValueError, match="not both"):
        partition_global_terrain(
            terrain,
            manifest,
            ORIGIN,
            loop_uvs=[(0.0, 0.0)] * 3,
            texture_bounds_l93_m=(999.0, 1_999.0, 1_020.0, 2_020.0),
        )
