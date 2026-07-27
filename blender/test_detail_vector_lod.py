from __future__ import annotations

import math

from detail_vector_lod import (
    TerrainMeshSampler,
    build_detail_tile_vectors,
    drape_building_prisms_to_tile,
    drape_ribbon_mesh_to_tile,
    drape_triangle_mesh_to_tile,
)


def _terrain_spec() -> dict[str, object]:
    # Row-major north-to-south grid. Z = X + 2Y gives an exact bilinear plane.
    vertices = []
    for y in (10.0, 5.0, 0.0):
        for x in (0.0, 5.0, 10.0):
            vertices.append([x, y, x + 2.0 * y])
    return {
        "vertices": vertices,
        "faces": [
            [0, 1, 4, 3],
            [1, 2, 5, 4],
            [3, 4, 7, 6],
            [4, 5, 8, 7],
        ],
    }


def test_regular_detail_sampler_uses_fixed_triangle_and_clamps_outer_rim() -> None:
    sampler = TerrainMeshSampler.from_terrain_spec(_terrain_spec())

    assert sampler.sample(2.5, 2.5) == 7.5
    assert sampler.sample(7.5, 7.5) == 22.5
    assert sampler.sample(-0.25, 4.0) == 8.0
    assert sampler.sample(10.25, 4.0) == 18.0

    non_coplanar = TerrainMeshSampler.from_terrain_spec(
        {
            # NW, NE, SW, SE; only NE is elevated. The fixed NW-SE
            # triangulation keeps the south-west triangle exactly flat.
            "vertices": [
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 10.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            "faces": [[0, 2, 3, 1]],
        }
    )
    assert non_coplanar.sample(0.25, 0.25) == 0.0
    assert non_coplanar.sample(0.75, 0.75) == 5.0


def test_ribbon_is_refined_both_longitudinally_and_laterally() -> None:
    sampler = TerrainMeshSampler.from_terrain_spec(_terrain_spec())
    source = {
        "vertices": [
            [0.0, 3.0, -50.0],
            [0.0, 7.0, -50.0],
            [10.0, 7.0, -50.0],
            [10.0, 3.0, -50.0],
        ],
        "faces": [[0, 1, 2, 3]],
    }

    mesh, statistics = drape_ribbon_mesh_to_tile(
        source,
        sampler,
        (0.0, 0.0, 10.0, 10.0),
        offset_m=0.12,
        maximum_segment_length_m=1.0,
    )

    assert statistics["generated_face_count"] == 80
    assert statistics["maximum_segment_length_m"] <= math.sqrt(2.0) + 0.001
    assert mesh["faces"]
    for x, y, z in mesh["vertices"]:
        assert math.isclose(z, x + 2.0 * y + 0.12, abs_tol=0.001)


def test_hydro_triangles_are_refined_and_draped_on_detail_mnt() -> None:
    sampler = TerrainMeshSampler.from_terrain_spec(_terrain_spec())
    source = {
        "vertices": [[0.0, 0.0, -10.0], [10.0, 0.0, -10.0], [0.0, 10.0, -10.0]],
        "faces": [[0, 1, 2]],
    }

    mesh, statistics = drape_triangle_mesh_to_tile(
        source,
        sampler,
        (0.0, 0.0, 10.0, 10.0),
        offset_m=0.08,
        maximum_edge_length_m=2.0,
    )

    assert statistics["generated_face_count"] > 1
    assert statistics["maximum_edge_length_m"] <= 2.0
    for x, y, z in mesh["vertices"]:
        assert math.isclose(z, x + 2.0 * y + 0.08, abs_tol=0.001)


def test_triangle_draping_skips_degenerate_source_and_serialized_faces() -> None:
    sampler = TerrainMeshSampler.from_terrain_spec(_terrain_spec())
    source = {
        "vertices": [
            # Non-zero source area, but millimetre serialization collapses the
            # three X coordinates onto one line.
            [1.0001, 1.0, 0.0],
            [1.0004, 1.0, 0.0],
            [1.0004, 2.0, 0.0],
            # Degenerate before refinement.
            [2.0, 2.0, 0.0],
            [3.0, 3.0, 0.0],
            [4.0, 4.0, 0.0],
            # One valid control face must still be emitted.
            [2.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
            [2.0, 4.0, 0.0],
        ],
        "faces": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    }

    mesh, statistics = drape_triangle_mesh_to_tile(
        source,
        sampler,
        (0.0, 0.0, 10.0, 10.0),
        offset_m=0.08,
        maximum_edge_length_m=10.0,
    )

    assert statistics["skipped_degenerate_source_face_count"] == 1
    assert statistics["skipped_degenerate_face_count"] == 1
    assert statistics["selected_source_face_count"] == 1
    assert statistics["generated_face_count"] == 1
    assert len(mesh["faces"]) == 1
    assert len(mesh["vertices"]) == 3


def test_vector_triangles_are_lifted_above_every_crossed_terrain_plane() -> None:
    sampler = TerrainMeshSampler.from_terrain_spec(
        {
            "vertices": [
                [0.0, 10.0, 10.0],
                [10.0, 10.0, 0.0],
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            "faces": [[0, 2, 3, 1]],
        }
    )
    # This ribbon's implicit diagonal is SW-NE, opposite to the terrain's
    # fixed NW-SE diagonal, so vertex-only draping would intersect the ground.
    source = {
        "vertices": [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
        ],
        "faces": [[0, 1, 2, 3]],
    }

    mesh, statistics = drape_ribbon_mesh_to_tile(
        source,
        sampler,
        (0.0, 0.0, 10.0, 10.0),
        offset_m=0.12,
        maximum_segment_length_m=10.0,
    )

    assert statistics["clearance_lifted_face_count"] > 0
    for face in mesh["faces"]:
        triangle = [mesh["vertices"][index] for index in face]
        assert (
            sampler.maximum_triangle_surface_deficit(triangle, vertical_offset_m=0.12)
            <= 0.001
        )


def test_detail_building_roof_keeps_2m70_visible_above_highest_ground() -> None:
    sampler = TerrainMeshSampler.from_terrain_spec(_terrain_spec())
    prisms = [
        {
            "feature_id": "building:1",
            "base_z": 0.0,
            "height": 5.0,
            "rings": [[[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]]],
            "roof_z": 5.0,
        }
    ]

    result, statistics = drape_building_prisms_to_tile(
        prisms, sampler, (0.0, 0.0, 10.0, 10.0)
    )

    assert statistics["selected_prism_count"] == 1
    assert statistics["raised_roof_prism_count"] == 1
    assert result[0]["ground_z_rings"] == [
        [6.005, 7.005, 8.005, 10.005, 12.005, 11.005, 10.005, 8.005]
    ]
    assert result[0]["roof_z"] == 14.705
    assert statistics["maximum_boundary_segment_length_m"] == 1.0


def test_detail_building_is_clipped_to_tile_before_ground_sampling() -> None:
    sampler = TerrainMeshSampler.from_terrain_spec(_terrain_spec())
    prisms = [
        {
            "feature_id": "building:edge",
            "base_z": 0.0,
            "height": 5.0,
            "rings": [[[-2.0, 2.0], [2.0, 2.0], [2.0, 4.0], [-2.0, 4.0]]],
            "roof_z": 8.0,
        }
    ]

    result, statistics = drape_building_prisms_to_tile(
        prisms, sampler, (0.0, 0.0, 10.0, 10.0)
    )

    assert statistics["selected_prism_count"] == 1
    assert statistics["tile_boundary_clipped_prism_count"] == 1
    assert all(
        0.0 <= x <= 10.0 and 0.0 <= y <= 10.0
        for ring in result[0]["rings"]
        for x, y in ring
    )
    assert statistics["maximum_boundary_segment_length_m"] <= 1.0


def test_complete_tile_contract_renders_segments_once_and_omits_courses() -> None:
    ribbon = {
        "vertices": [
            [0.0, 4.0, 0.0],
            [0.0, 6.0, 0.0],
            [10.0, 6.0, 0.0],
            [10.0, 4.0, 0.0],
        ],
        "faces": [[0, 1, 2, 3]],
    }
    explicit_surface = {
        "vertices": ribbon["vertices"],
        "faces": [[0, 1, 2], [0, 2, 3]],
    }
    surface = {
        "vertices": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
        "faces": [[0, 1, 2]],
    }
    package = {
        "buildings": {"prisms": []},
        "roads": {
            "meshes": {
                "carriageway": explicit_surface,
                "left_shoulders": {"vertices": [], "faces": []},
                "right_shoulders": {"vertices": [], "faces": []},
                "center_markings": {"vertices": [], "faces": []},
            }
        },
        "water": {
            "courses": {"mesh": ribbon},
            "segments": {"mesh": explicit_surface},
            "surfaces": {"mesh": surface},
        },
    }
    tile_package = {"terrain": _terrain_spec()}

    result = build_detail_tile_vectors(
        package, tile_package, (100.0, 200.0, 110.0, 210.0), (100.0, 200.0, 0.0)
    )

    assert result["water"]["rendered_linear_source"] == "segments_only"
    assert result["water"]["courses_rendered"] is False
    assert "courses" not in result["water"]
    assert result["water"]["segments"]["mesh"]["faces"]
    assert result["roads"]["meshes"]["carriageway"]["faces"]
