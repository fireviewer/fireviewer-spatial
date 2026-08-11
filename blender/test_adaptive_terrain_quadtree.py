from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

import adaptive_terrain_quadtree as terrain_quadtree
from adaptive_terrain_quadtree import (
    Breakline,
    EDGE_ORDER,
    FVTQ_MAGIC,
    GRID_UNITS,
    LOD0_TRIANGLE_BUDGET_WITHOUT_BREAKLINES,
    NODE_SPLIT_BREAKLINE,
    NORMAL_HALO_SAMPLE_COUNT,
    STITCH_MASK_BITS,
    compile_adaptive_tile,
    decode_fvtq,
    encode_fvtq,
    load_contract,
    materialize_stitch_triangles,
    quantize_heights_mm,
    read_fvtq,
    stitch_mask_for_neighbors,
    write_fvtq,
)


@pytest.fixture(scope="module")
def synthetic_surface() -> tuple[np.ndarray, np.ndarray, tuple[Breakline, ...]]:
    """Smooth terrain plus declared constraints on the canonical halo grid."""

    halo_axis = np.linspace(-2.0, 502.0, NORMAL_HALO_SAMPLE_COUNT)
    northing, easting = np.meshgrid(halo_axis, halo_axis, indexing="ij")
    halo = 120.0 + 0.01 * easting + 0.005 * northing
    halo += 1.5 * np.sin(easting / 90.0) + 0.8 * np.cos(northing / 110.0)
    surface = halo[1:-1, 1:-1].copy()
    breaklines = (
        Breakline.from_metres("ridge-main", [(255.0, 0.0), (255.0, 500.0)]),
        Breakline.from_metres("ravine-main", [(0.0, 80.0), (500.0, 305.0)]),
    )
    return surface, halo, breaklines


@pytest.fixture(scope="module")
def compiled_synthetic(
    synthetic_surface: tuple[np.ndarray, np.ndarray, tuple[Breakline, ...]],
):
    surface, halo, breaklines = synthetic_surface
    return compile_adaptive_tile(
        surface,
        normal_halo_heights_m=halo,
        tile_origin_l93_m=(700_000.0, 6_300_000.0),
        breaklines=reversed(breaklines),
    )


def _leaf_depth_occupancy(mesh) -> np.ndarray:
    occupancy = np.full((GRID_UNITS, GRID_UNITS), -1, dtype="int16")
    for node in mesh.nodes:
        if not node.is_leaf:
            continue
        span = GRID_UNITS >> node.depth
        x0 = node.x_index * span
        y0 = node.y_index * span
        occupancy[y0 : y0 + span, x0 : x0 + span] = node.depth
    return occupancy


def _twice_triangle_area(mesh, triangles=None) -> int:
    total = 0
    for triangle in mesh.triangles if triangles is None else triangles:
        a, b, c = (mesh.vertices[index] for index in triangle)
        total += (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return total


def _resign_fvtq(payload: bytearray) -> bytes:
    digest_offset = terrain_quadtree._HEADER.size - 32
    payload[digest_offset : terrain_quadtree._HEADER.size] = bytes(32)
    digest = hashlib.sha256(payload).digest()
    payload[digest_offset : terrain_quadtree._HEADER.size] = digest
    return bytes(payload)


def test_contract_locks_500m_2m_grid_and_adaptive_lods() -> None:
    contract, digest, policies = load_contract()

    assert contract["tile_size_m"] == 500.0
    assert contract["source_working_grid"]["resolution_m"] == 2.0
    assert contract["source_working_grid"]["shape"] == [251, 251]
    assert contract["source_working_grid"]["wms_pixel_shape_with_10m_halo"] == [
        260,
        260,
    ]
    assert contract["canonical_normal_halo"]["shape"] == [253, 253]
    assert contract["source_working_grid"]["orthophoto_dependency"] == "forbidden"
    assert len(digest) == 32
    assert [policy.maximum_error_mm for policy in policies] == [500, 2_000, 8_000]
    assert [policy.base_depth for policy in policies] == [4, 2, 0]
    assert [policy.maximum_regular_depth for policy in policies] == [7, 5, 3]
    assert [policy.maximum_breakline_depth for policy in policies] == [8, 5, 3]
    assert [policy.edge_depth for policy in policies] == [7, 6, 5]
    assert all(
        fine.edge_depth - coarse.edge_depth == 1
        for fine, coarse in zip(policies[:-1], policies[1:], strict=True)
    )
    assert [policy.maximum_triangles_with_breaklines for policy in policies] == [
        131_072,
        4_096,
        1_024,
    ]
    assert contract["quadtree"]["stitch_mask_bits"] == STITCH_MASK_BITS
    assert (
        contract["quadtree"]["boundary_breakline_refinement"]
        == "capped_at_lod_edge_depth_to_preserve_2_to_1_runtime_stitching"
    )
    assert contract["lods"][0]["maximum_triangles_without_breaklines"] == 32_768


def test_compiles_nested_balanced_lods_for_flat_ridge_and_ravine(
    compiled_synthetic,
) -> None:
    tile = compiled_synthetic

    assert [mesh.lod for mesh in tile.lods] == [0, 1, 2]
    assert all(mesh.source_grid_sha256 == tile.source_grid_sha256 for mesh in tile.lods)
    assert all(mesh.contract_sha256 == tile.contract_sha256 for mesh in tile.lods)
    assert [
        max(node.depth for node in mesh.nodes if node.is_leaf) for mesh in tile.lods
    ] == [
        8,
        6,
        5,
    ]
    assert any(node.flags & NODE_SPLIT_BREAKLINE for node in tile.lods[0].nodes)

    for mesh, minimum_edge_count in zip(tile.lods, (129, 33, 9), strict=True):
        occupancy = _leaf_depth_occupancy(mesh)
        assert (occupancy >= 0).all()
        assert int(np.abs(np.diff(occupancy, axis=0)).max()) <= 1
        assert int(np.abs(np.diff(occupancy, axis=1)).max()) <= 1
        assert _twice_triangle_area(mesh) == 2 * GRID_UNITS * GRID_UNITS
        assert all(len(edge) >= minimum_edge_count for edge in mesh.edge_vertex_indices)
        assert len(mesh.vertex_gradients_mm_per_4m) == len(mesh.vertices)
        assert len(mesh.stitch_variants) == 16
        assert [variant.mask for variant in mesh.stitch_variants] == list(range(16))
        assert mesh.maximum_final_error_mm == max(
            variant.maximum_error_mm for variant in mesh.stitch_variants
        )
        for variant in mesh.stitch_variants:
            assert (
                _twice_triangle_area(
                    mesh, materialize_stitch_triangles(mesh, variant.mask)
                )
                == 2 * GRID_UNITS * GRID_UNITS
            )

    assert set(tile.lods[2].vertices).issubset(tile.lods[1].vertices)
    assert set(tile.lods[1].vertices).issubset(tile.lods[0].vertices)
    lod0_nodes = {
        (node.depth, node.x_index, node.y_index): node for node in tile.lods[0].nodes
    }
    for collapsed in tile.lods[1:]:
        for node in collapsed.nodes:
            master = lod0_nodes[(node.depth, node.x_index, node.y_index)]
            assert node.maximum_error_mm == master.maximum_error_mm


def test_fvtq_is_bitwise_reproducible_and_roundtrips(
    synthetic_surface: tuple[np.ndarray, np.ndarray, tuple[Breakline, ...]],
    compiled_synthetic,
    tmp_path: Path,
) -> None:
    surface, halo, breaklines = synthetic_surface
    rebuilt = compile_adaptive_tile(
        surface.copy(),
        normal_halo_heights_m=halo.copy(),
        tile_origin_l93_m=(700_000.0, 6_300_000.0),
        breaklines=breaklines,
    )

    for expected, observed in zip(compiled_synthetic.lods, rebuilt.lods, strict=True):
        expected_bytes = encode_fvtq(expected)
        observed_bytes = encode_fvtq(observed)
        assert expected_bytes == observed_bytes
        assert expected_bytes[:4] == FVTQ_MAGIC
        assert expected_bytes[4:6] == b"\x01\x00"
        assert decode_fvtq(expected_bytes) == expected

        first_path = tmp_path / "first" / f"terrain-lod{expected.lod}.fvtq"
        second_path = tmp_path / "second" / f"terrain-lod{expected.lod}.fvtq"
        first_hash = write_fvtq(expected, first_path)
        second_hash = write_fvtq(observed, second_path)
        assert first_hash == second_hash
        assert first_path.read_bytes() == second_path.read_bytes()
        assert read_fvtq(first_path) == expected


def _effective_edge_payload(mesh, mask: int, edge_number: int):
    used = {
        index
        for triangle in materialize_stitch_triangles(mesh, mask)
        for index in triangle
    }
    edge = EDGE_ORDER[edge_number]
    payload = []
    for index in mesh.edge_vertex_indices[edge_number]:
        if index not in used:
            continue
        x, y, relative_height = mesh.vertices[index]
        payload.append(
            (
                y if edge in ("west", "east") else x,
                relative_height + mesh.z_origin_mm,
                mesh.vertex_gradients_mm_per_4m[index],
            )
        )
    return payload


def _maximum_leaf_depth_on_edge(mesh, edge: str) -> int:
    def touches(node) -> bool:
        maximum_index = (1 << node.depth) - 1
        return {
            "west": node.x_index == 0,
            "east": node.x_index == maximum_index,
            "south": node.y_index == 0,
            "north": node.y_index == maximum_index,
        }[edge]

    return max(node.depth for node in mesh.nodes if node.is_leaf and touches(node))


def test_global_axis_edge_signatures_and_lod_stitches_match_adjacent_tiles() -> None:
    halo_axis = np.linspace(-2.0, 502.0, NORMAL_HALO_SAMPLE_COUNT)
    northing, local_easting = np.meshgrid(halo_axis, halo_axis, indexing="ij")
    west_halo = 90.0 + 0.012 * local_easting + 0.006 * northing
    east_halo = 90.0 + 0.012 * (local_easting + 500.0) + 0.006 * northing

    west_tile = compile_adaptive_tile(
        west_halo[1:-1, 1:-1],
        normal_halo_heights_m=west_halo,
        tile_origin_l93_m=(700_000.0, 6_300_000.0),
    )
    east_tile = compile_adaptive_tile(
        east_halo[1:-1, 1:-1],
        normal_halo_heights_m=east_halo,
        tile_origin_l93_m=(700_500.0, 6_300_000.0),
    )

    for west_lod, east_lod in zip(west_tile.lods, east_tile.lods, strict=True):
        assert west_lod.edge_signatures[1] == east_lod.edge_signatures[0]
        assert _effective_edge_payload(west_lod, 0, 1) == _effective_edge_payload(
            east_lod, 0, 0
        )

    east_stitch = STITCH_MASK_BITS["east"]
    west_stitch = STITCH_MASK_BITS["west"]
    for fine_lod, coarse_lod in ((0, 1), (1, 2)):
        fine_west = west_tile.lods[fine_lod]
        coarse_east = east_tile.lods[coarse_lod]
        assert (
            fine_west.stitch_variants[east_stitch].effective_edge_signatures[1]
            == coarse_east.edge_signatures[0]
        )
        assert _effective_edge_payload(fine_west, east_stitch, 1) == (
            _effective_edge_payload(coarse_east, 0, 0)
        )

        coarse_west = west_tile.lods[coarse_lod]
        fine_east = east_tile.lods[fine_lod]
        assert (
            fine_east.stitch_variants[west_stitch].effective_edge_signatures[0]
            == coarse_west.edge_signatures[1]
        )
        assert _effective_edge_payload(fine_east, west_stitch, 0) == (
            _effective_edge_payload(coarse_west, 0, 1)
        )

    assert stitch_mask_for_neighbors(0, {"east": 1, "north": 0}) == east_stitch
    assert stitch_mask_for_neighbors(1, {"west": 2}) == west_stitch
    with pytest.raises(ValueError, match="delta"):
        stitch_mask_for_neighbors(0, {"east": 2})
    assert len(west_tile.lods[0].triangles) <= LOD0_TRIANGLE_BUDGET_WITHOUT_BREAKLINES
    assert len(east_tile.lods[0].triangles) <= LOD0_TRIANGLE_BUDGET_WITHOUT_BREAKLINES


def test_breakline_crossing_seam_preserves_two_to_one_runtime_edge() -> None:
    halo_axis = np.linspace(-2.0, 502.0, NORMAL_HALO_SAMPLE_COUNT)
    northing, local_easting = np.meshgrid(halo_axis, halo_axis, indexing="ij")
    west_halo = 90.0 + 0.012 * local_easting + 0.006 * northing
    east_halo = 90.0 + 0.012 * (local_easting + 500.0) + 0.006 * northing
    feature_id = "global-breakline-crossing-seam"

    west_tile = compile_adaptive_tile(
        west_halo[1:-1, 1:-1],
        normal_halo_heights_m=west_halo,
        tile_origin_l93_m=(700_000.0, 6_300_000.0),
        breaklines=(
            Breakline.from_metres(feature_id, [(450.0, 250.0), (500.0, 250.0)]),
        ),
    )
    east_tile = compile_adaptive_tile(
        east_halo[1:-1, 1:-1],
        normal_halo_heights_m=east_halo,
        tile_origin_l93_m=(700_500.0, 6_300_000.0),
        breaklines=(Breakline.from_metres(feature_id, [(0.0, 250.0), (50.0, 250.0)]),),
    )

    west_fine = west_tile.lods[0]
    east_coarse = east_tile.lods[1]
    assert _maximum_leaf_depth_on_edge(west_fine, "east") == 7
    assert _maximum_leaf_depth_on_edge(east_coarse, "west") == 6
    assert len(west_fine.edge_vertex_indices[1]) == 129
    assert len(east_coarse.edge_vertex_indices[0]) == 65
    assert (
        max(
            node.depth
            for node in west_fine.nodes
            if node.is_leaf and node.x_index != (1 << node.depth) - 1
        )
        == 8
    )
    assert (
        _maximum_leaf_depth_on_edge(west_fine, "east")
        - _maximum_leaf_depth_on_edge(east_coarse, "west")
        == 1
    )

    east_stitch = STITCH_MASK_BITS["east"]
    assert _effective_edge_payload(west_fine, east_stitch, 1) == (
        _effective_edge_payload(east_coarse, 0, 0)
    )


def test_flat_lod_costs_decrease_and_all_stitch_masks_are_deterministic() -> None:
    halo = np.zeros((NORMAL_HALO_SAMPLE_COUNT, NORMAL_HALO_SAMPLE_COUNT))
    first = compile_adaptive_tile(
        halo[1:-1, 1:-1],
        normal_halo_heights_m=halo,
        tile_origin_l93_m=(700_000.0, 6_300_000.0),
    )
    second = compile_adaptive_tile(
        halo[1:-1, 1:-1],
        normal_halo_heights_m=halo.copy(),
        tile_origin_l93_m=(700_000.0, 6_300_000.0),
    )

    maximum_counts = [
        max(
            variant.triangle_count(len(mesh.triangles))
            for variant in mesh.stitch_variants
        )
        for mesh in first.lods
    ]
    assert maximum_counts[1] < maximum_counts[0] // 2
    assert maximum_counts[2] < maximum_counts[1] // 2
    for expected, observed in zip(first.lods, second.lods, strict=True):
        assert expected.stitch_variants == observed.stitch_variants
        assert encode_fvtq(expected) == encode_fvtq(observed)


def test_rejects_bad_source_grid_and_binary_corruption(compiled_synthetic) -> None:
    with pytest.raises(ValueError, match="shape"):
        quantize_heights_mm(np.zeros((250, 251), dtype="float64"))
    broken = np.zeros((251, 251), dtype="float64")
    broken[10, 10] = np.nan
    with pytest.raises(ValueError, match="nodata"):
        quantize_heights_mm(broken)
    with pytest.raises(ValueError, match="500 m Lambert-93 grid"):
        compile_adaptive_tile(
            np.zeros((251, 251), dtype="float64"),
            normal_halo_heights_m=np.zeros((253, 253), dtype="float64"),
            tile_origin_l93_m=(700_001.0, 6_300_000.0),
        )
    with pytest.raises(ValueError, match="normal halo core"):
        compile_adaptive_tile(
            np.zeros((251, 251), dtype="float64"),
            normal_halo_heights_m=np.ones((253, 253), dtype="float64"),
            tile_origin_l93_m=(700_000.0, 6_300_000.0),
        )

    payload = bytearray(encode_fvtq(compiled_synthetic.lods[0]))
    payload[-1] ^= 1
    with pytest.raises(ValueError, match="digest mismatch"):
        decode_fvtq(bytes(payload))


def test_decode_rejects_unknown_node_flags_and_unaligned_origin(
    compiled_synthetic,
) -> None:
    encoded = encode_fvtq(compiled_synthetic.lods[0])

    unknown_flags = bytearray(encoded)
    first_node_flag_offset = terrain_quadtree._HEADER.size + 5
    unknown_flags[first_node_flag_offset] |= 0x80
    with pytest.raises(ValueError, match="unknown flags"):
        decode_fvtq(_resign_fvtq(unknown_flags))

    unpacked = list(
        terrain_quadtree._HEADER.unpack(encoded[: terrain_quadtree._HEADER.size])
    )
    unpacked[4] += 1
    unpacked[-1] = bytes(32)
    unaligned = bytearray(
        terrain_quadtree._HEADER.pack(*unpacked)
        + encoded[terrain_quadtree._HEADER.size :]
    )
    with pytest.raises(ValueError, match="500 m Lambert-93 grid"):
        decode_fvtq(_resign_fvtq(unaligned))


def test_lod0_triangle_budget_fails_closed_without_breaklines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_bundle = load_contract()
    monkeypatch.setattr(terrain_quadtree, "LOD0_TRIANGLE_BUDGET_WITHOUT_BREAKLINES", 1)
    monkeypatch.setattr(
        terrain_quadtree,
        "load_contract",
        lambda _path=None: contract_bundle,
    )
    with pytest.raises(ValueError, match="32768-triangle budget"):
        compile_adaptive_tile(
            np.zeros((251, 251), dtype="float64"),
            normal_halo_heights_m=np.zeros((253, 253), dtype="float64"),
            tile_origin_l93_m=(700_000.0, 6_300_000.0),
        )


def test_explicit_breakline_stitch_budget_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, digest, policies = load_contract()
    constrained = (
        replace(policies[0], maximum_triangles_with_breaklines=1),
        policies[1],
        policies[2],
    )
    monkeypatch.setattr(
        terrain_quadtree,
        "load_contract",
        lambda _path=None: (contract, digest, constrained),
    )
    halo = np.zeros((NORMAL_HALO_SAMPLE_COUNT, NORMAL_HALO_SAMPLE_COUNT))
    with pytest.raises(ValueError, match="explicit 1-triangle budget"):
        compile_adaptive_tile(
            halo[1:-1, 1:-1],
            normal_halo_heights_m=halo,
            tile_origin_l93_m=(700_000.0, 6_300_000.0),
            breaklines=(
                Breakline.from_metres("budget-breakline", [(0.0, 0.0), (500.0, 500.0)]),
            ),
        )


def test_unachievable_spike_error_fails_closed() -> None:
    halo = np.zeros((253, 253), dtype="float64")
    halo[126, 126] = 100.0
    with pytest.raises(
        ValueError, match=r"LOD0 final triangulation error.*limit 500 mm"
    ):
        compile_adaptive_tile(
            halo[1:-1, 1:-1],
            normal_halo_heights_m=halo,
            tile_origin_l93_m=(700_000.0, 6_300_000.0),
        )


def test_seed_12345_final_triangle_regression_fails_closed() -> None:
    rng = np.random.default_rng(12345)
    grid_mm = rng.integers(-2_000, 2_001, size=(251, 251), dtype=np.int32)
    # This depth-7 node is individually below the 500 mm contract, while the
    # final transition triangulation on the same deterministic surface is not.
    assert terrain_quadtree._node_error_mm(grid_mm, 7, 52, 106) < 500
    halo = np.pad(grid_mm, 1, mode="edge").astype("float64") / 1_000.0
    with pytest.raises(
        ValueError, match=r"LOD0 final triangulation error.*limit 500 mm"
    ):
        compile_adaptive_tile(
            halo[1:-1, 1:-1],
            normal_halo_heights_m=halo,
            tile_origin_l93_m=(700_000.0, 6_300_000.0),
        )
