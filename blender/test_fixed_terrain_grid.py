from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import fixed_terrain_grid as fixed_grid
from fixed_terrain_grid import (
    ACQUISITION_HALO_SAMPLE_COUNT,
    FVTG_MAGIC,
    LOD_SPECS,
    NORMAL_HALO_SAMPLE_COUNT,
    SKIRT_DEPTH_MM,
    SOURCE_SAMPLE_COUNT,
    compile_fixed_terrain,
    compile_fixed_terrain_from_canonical_mm,
    decode_fixed_terrain,
    edge_signature,
    encode_fixed_terrain,
    load_contract,
    lod_absolute_heights_mm,
    main,
    quantize_heights_mm,
    read_fixed_terrain,
    source_grid_mm,
    validate_fixed_terrain,
    write_fixed_terrain,
)


ORIGIN = (700_000.0, 6_300_000.0)


def _sampled_surface(origin: tuple[float, float], *, halo_m: float) -> np.ndarray:
    sample_count = SOURCE_SAMPLE_COUNT + int(halo_m)
    axis = np.arange(sample_count, dtype="float64") * 2.0 - halo_m
    northing, easting = np.meshgrid(
        origin[1] + axis,
        origin[0] + axis,
        indexing="ij",
    )
    return 125.0 + 0.01 * (easting - ORIGIN[0]) + 0.02 * (northing - ORIGIN[1])


def _inputs(
    origin: tuple[float, float] = ORIGIN,
) -> tuple[np.ndarray, np.ndarray]:
    return _sampled_surface(origin, halo_m=0.0), _sampled_surface(origin, halo_m=10.0)


@pytest.fixture(scope="module")
def plane_tile():
    core, halo = _inputs()
    return compile_fixed_terrain(
        core,
        source_halo_heights_m=halo,
        tile_origin_l93_m=ORIGIN,
    )


def test_contract_locks_fixed_grid_source_lods_and_skirts() -> None:
    contract, digest = load_contract()

    assert contract["schema"] == "fireviewer.terrain-fixed-grid-contract.v1"
    assert contract["format_schema"] == "fireviewer.terrain-fixed-grid.v1"
    assert contract["crs"] == "EPSG:2154"
    assert contract["tile_size_m"] == 500.0
    assert contract["source"]["core_shape"] == [251, 251]
    assert contract["source"]["acquisition_halo_shape"] == [261, 261]
    assert contract["source"]["retained_normal_halo_shape"] == [253, 253]
    assert contract["source"]["orthophoto_dependency"] == "forbidden"
    assert [item[1] for item in LOD_SPECS] == [129, 33, 9]
    assert [item[3] for item in LOD_SPECS] == [32_768, 2_048, 128]
    assert contract["geometry"]["skirt_depth_m"] == 10.0
    assert contract["geometry"]["core_and_skirt_sections"] == "separate"
    assert len(digest) == 32


def test_quantization_is_half_away_from_zero_and_rejects_nodata() -> None:
    values = np.zeros((SOURCE_SAMPLE_COUNT, SOURCE_SAMPLE_COUNT), dtype="float64")
    values[0, :4] = [1.0004, 1.0005, -1.0004, -1.0005]

    quantized = quantize_heights_mm(values)

    assert quantized[0, :4].tolist() == [1_000, 1_001, -1_000, -1_001]
    values[50, 50] = np.nan
    with pytest.raises(ValueError, match="nodata"):
        quantize_heights_mm(values)


def test_compiles_exact_regular_nested_lods_with_separate_skirts(plane_tile) -> None:
    assert plane_tile.tile_origin_mm == (700_000_000, 6_300_000_000)
    assert source_grid_mm(plane_tile).shape == (251, 251)
    assert len(plane_tile.normal_halo_heights_mm) == 253 * 253
    assert [mesh.grid_size for mesh in plane_tile.lods] == [129, 33, 9]
    assert [mesh.core_triangle_count for mesh in plane_tile.lods] == [
        32_768,
        2_048,
        128,
    ]

    lod0_heights = lod_absolute_heights_mm(plane_tile, 0)
    lod1_heights = lod_absolute_heights_mm(plane_tile, 1)
    lod2_heights = lod_absolute_heights_mm(plane_tile, 2)
    assert np.array_equal(lod1_heights, lod0_heights[::4, ::4])
    assert np.array_equal(lod2_heights, lod0_heights[::16, ::16])

    lod0_gradients = np.asarray(plane_tile.lods[0].gradients_mm_per_4m).reshape(
        129, 129, 2
    )
    lod0_normals = np.asarray(plane_tile.lods[0].normals_snorm16).reshape(129, 129, 3)
    assert np.array_equal(
        np.asarray(plane_tile.lods[1].gradients_mm_per_4m).reshape(33, 33, 2),
        lod0_gradients[::4, ::4],
    )
    assert np.array_equal(
        np.asarray(plane_tile.lods[2].normals_snorm16).reshape(9, 9, 3),
        lod0_normals[::16, ::16],
    )
    assert set(map(tuple, lod0_gradients.reshape(-1, 2))) == {(40, 80)}
    assert len(set(map(tuple, lod0_normals.reshape(-1, 3)))) == 1
    assert np.all(lod0_normals[:, :, 2] > 0)

    for mesh, (_lod, size, _stride, core_triangle_count) in zip(
        plane_tile.lods, LOD_SPECS, strict=True
    ):
        core_vertex_count = size * size
        perimeter_count = 4 * (size - 1)
        assert len(mesh.relative_heights_mm) == core_vertex_count
        assert len(mesh.core_triangles) == core_triangle_count
        assert len(mesh.skirt_core_vertex_indices) == perimeter_count
        assert len(set(mesh.skirt_core_vertex_indices)) == perimeter_count
        assert len(mesh.skirt_triangles) == 2 * perimeter_count
        assert all(
            skirt_height == mesh.relative_heights_mm[core_index] - SKIRT_DEPTH_MM
            for core_index, skirt_height in zip(
                mesh.skirt_core_vertex_indices,
                mesh.skirt_relative_heights_mm,
                strict=True,
            )
        )
        assert all(
            max(triangle) < core_vertex_count for triangle in mesh.core_triangles
        )
        assert all(
            any(index >= core_vertex_count for index in triangle)
            for triangle in mesh.skirt_triangles
        )
        assert all(
            max(triangle) < core_vertex_count + perimeter_count
            for triangle in mesh.skirt_triangles
        )


def test_flat_surface_has_exact_up_normals() -> None:
    core = np.full((251, 251), 200.0)
    halo = np.full((261, 261), 200.0)

    tile = compile_fixed_terrain(
        core,
        source_halo_heights_m=halo,
        tile_origin_l93_m=ORIGIN,
    )

    assert set(tile.lods[0].gradients_mm_per_4m) == {(0, 0)}
    assert set(tile.lods[0].normals_snorm16) == {(0, 0, 32_767)}


def test_global_edges_heights_gradients_and_normals_match_neighbors() -> None:
    west_core, west_halo = _inputs(ORIGIN)
    east_origin = (ORIGIN[0] + 500.0, ORIGIN[1])
    east_core, east_halo = _inputs(east_origin)
    west = compile_fixed_terrain(
        west_core,
        source_halo_heights_m=west_halo,
        tile_origin_l93_m=ORIGIN,
    )
    east = compile_fixed_terrain(
        east_core,
        source_halo_heights_m=east_halo,
        tile_origin_l93_m=east_origin,
    )

    for lod, size, _stride, _triangles in LOD_SPECS:
        west_mesh = west.lods[lod]
        east_mesh = east.lods[lod]
        west_indices = [row * size + size - 1 for row in range(size)]
        east_indices = [row * size for row in range(size)]
        west_payload = [
            (
                west_mesh.relative_heights_mm[index] + west.z_origin_mm,
                west_mesh.gradients_mm_per_4m[index],
                west_mesh.normals_snorm16[index],
            )
            for index in west_indices
        ]
        east_payload = [
            (
                east_mesh.relative_heights_mm[index] + east.z_origin_mm,
                east_mesh.gradients_mm_per_4m[index],
                east_mesh.normals_snorm16[index],
            )
            for index in east_indices
        ]
        assert west_payload == east_payload
        assert edge_signature(west, lod, "east") == edge_signature(east, lod, "west")


def test_codec_is_bitwise_reproducible_roundtrips_and_writes_atomically(
    plane_tile, tmp_path: Path
) -> None:
    core, halo = _inputs()
    rebuilt = compile_fixed_terrain(
        core.copy(),
        source_halo_heights_m=halo.copy(),
        tile_origin_l93_m=ORIGIN,
    )
    first_payload = encode_fixed_terrain(plane_tile)
    second_payload = encode_fixed_terrain(rebuilt)

    assert first_payload == second_payload
    assert first_payload[:4] == FVTG_MAGIC
    assert first_payload[4:6] == b"\x01\x00"
    assert len(first_payload) == fixed_grid._HEADER.size + 253 * 253 * 4
    assert decode_fixed_terrain(first_payload) == plane_tile

    first_path = tmp_path / "first" / "terrain.fvtg"
    second_path = tmp_path / "second" / "terrain.fvtg"
    first_hash = write_fixed_terrain(plane_tile, first_path)
    second_hash = write_fixed_terrain(rebuilt, second_path)
    repeated_hash = write_fixed_terrain(plane_tile, first_path)

    assert first_hash == second_hash == repeated_hash
    assert first_path.read_bytes() == second_path.read_bytes()
    assert read_fixed_terrain(first_path) == plane_tile


def test_compiles_directly_from_existing_canonical_millimetre_npy(plane_tile) -> None:
    normal_halo = np.asarray(plane_tile.normal_halo_heights_mm, dtype="<i4").reshape(
        253, 253
    )

    rebuilt = compile_fixed_terrain_from_canonical_mm(
        normal_halo,
        tile_origin_l93_m=ORIGIN,
    )

    assert rebuilt == plane_tile
    assert encode_fixed_terrain(rebuilt) == encode_fixed_terrain(plane_tile)


def test_cli_compiles_and_verifies_canonical_npy_on_d(
    plane_tile, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    normal_halo = np.asarray(plane_tile.normal_halo_heights_mm, dtype="<i4").reshape(
        253, 253
    )
    source_path = tmp_path / "mnt-canonical-normal-halo-2m-mm.npy"
    output_path = tmp_path / "terrain.fvtg"
    np.save(source_path, normal_halo, allow_pickle=False)

    result = main(
        [
            "compile",
            "--normal-halo",
            str(source_path),
            "--origin-x",
            str(ORIGIN[0]),
            "--origin-y",
            str(ORIGIN[1]),
            "--output",
            str(output_path),
        ]
    )
    compile_receipt = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output_path.is_file()
    assert compile_receipt["format_schema"] == "fireviewer.terrain-fixed-grid.v1"
    assert [item["core_triangles"] for item in compile_receipt["lods"]] == [
        32_768,
        2_048,
        128,
    ]

    assert main(["verify", "--input", str(output_path)]) == 0
    verify_receipt = json.loads(capsys.readouterr().out)
    assert verify_receipt["sha256"] == compile_receipt["sha256"]


def _resign_fvtg(payload: bytearray) -> bytes:
    digest_offset = fixed_grid._HEADER.size - 32
    payload[digest_offset : fixed_grid._HEADER.size] = bytes(32)
    digest = hashlib.sha256(payload).digest()
    payload[digest_offset : fixed_grid._HEADER.size] = digest
    return bytes(payload)


def test_codec_fails_closed_on_sha_and_semantic_corruption(plane_tile) -> None:
    payload = bytearray(encode_fixed_terrain(plane_tile))
    payload[-1] ^= 1
    with pytest.raises(ValueError, match="file SHA-256"):
        decode_fixed_terrain(bytes(payload))

    resigned = bytearray(encode_fixed_terrain(plane_tile))
    retained_height_offset = fixed_grid._HEADER.size + (100 * 253 + 100) * 4
    resigned[retained_height_offset] ^= 1
    with pytest.raises(ValueError, match="source grid SHA-256"):
        decode_fixed_terrain(_resign_fvtg(resigned))

    malformed_mesh = replace(
        plane_tile.lods[0],
        normals_snorm16=((1, 0, 32_767),) + plane_tile.lods[0].normals_snorm16[1:],
    )
    malformed_tile = replace(
        plane_tile,
        lods=(malformed_mesh, plane_tile.lods[1], plane_tile.lods[2]),
    )
    with pytest.raises(ValueError, match="LOD geometry"):
        validate_fixed_terrain(malformed_tile)


def test_rejects_shape_nodata_misalignment_and_halo_core_mismatch() -> None:
    core, halo = _inputs()
    with pytest.raises(ValueError, match="shape"):
        compile_fixed_terrain(
            core[:-1],
            source_halo_heights_m=halo,
            tile_origin_l93_m=ORIGIN,
        )
    bad_halo = halo.copy()
    bad_halo[5, 5] += 0.001
    with pytest.raises(ValueError, match="exactly match"):
        compile_fixed_terrain(
            core,
            source_halo_heights_m=bad_halo,
            tile_origin_l93_m=ORIGIN,
        )
    with pytest.raises(ValueError, match="align"):
        compile_fixed_terrain(
            core,
            source_halo_heights_m=halo,
            tile_origin_l93_m=(ORIGIN[0] + 1.0, ORIGIN[1]),
        )


def test_rejects_modified_contract(tmp_path: Path) -> None:
    contract_path = Path(fixed_grid.__file__).with_name(
        "fixed_terrain_grid_contract.v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["geometry"]["skirt_depth_m"] = 9.0
    modified = tmp_path / "invalid-contract.json"
    modified.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="constants"):
        load_contract(modified)


def test_public_shape_constants_match_contract() -> None:
    assert SOURCE_SAMPLE_COUNT == 251
    assert ACQUISITION_HALO_SAMPLE_COUNT == 261
    assert NORMAL_HALO_SAMPLE_COUNT == 253
