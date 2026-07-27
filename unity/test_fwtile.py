from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

import pytest


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from export_remote_catalog import (  # noqa: E402
    COMPATIBLE_EXPORTER_SHA256S,
    FWTileError,
    _crop_to_largest_measured_regular_grid,
    _prioritize_tiles,
    build_catalog,
    create_context,
    export_tile,
    validate_catalog,
)
from fwtile import (  # noqa: E402
    TREE_RECORD,
    build_container,
    build_vector_sections,
    decode_u16_elevations,
    encode_detail_terrain,
    encode_far_grid,
    encode_tree_instances,
    prism_to_mesh,
    read_container,
    sha256_file,
)


def _terrain() -> dict:
    vertices = []
    for row, north in enumerate((2.0, 1.0, 0.0)):
        for east in (0.0, 1.0, 2.0):
            vertices.append([east, north, 100.0 + east * 2.0 + row])
    return {
        "vertices": vertices,
        "faces": [
            [0, 1, 4, 3],
            [1, 2, 5, 4],
            [3, 4, 7, 6],
            [4, 5, 8, 7],
        ],
        "sample_spacing_m": [1.0, 1.0],
        "source_pixel_size_m": [0.5, 0.5],
        "boundary_sampling": "test_exact_edges",
        "adjacent_edge_contract": "test_coincident_edges",
    }


def _package() -> dict:
    return {
        "metadata": {
            "bounds_l93_m": [0.0, 0.0, 2.0, 2.0],
            "origin_l93_m": [0.0, 0.0, 0.0],
        },
        "statistics": {
            "accepted_instance_count": 2,
            "completeness_claim": "detected_crowns_not_field_inventory",
        },
        "terrain": _terrain(),
        "instances": {
            "attributes": [
                "local_x_m",
                "local_y_m",
                "local_ground_z_m",
                "height_m",
                "crown_diameter_m",
                "visual_variant",
                "rotation_degrees",
            ],
            "values": [
                [0.5, 0.5, 101.25, 8.125, 4.5, 1, 30.25],
                [1.5, 1.0, 104.0, 12.0, 6.0, 3, 359.99],
            ],
        },
        "tree_instances": {
            "prototypes": [{"id": 1, "kind": "deciduous"}],
            "material_slots": ["foliage"],
        },
    }


def _detail_vectors() -> dict:
    prism = {
        "feature_id": "building.test",
        "base_z": 101.0,
        "roof_z": 106.0,
        "height": 5.0,
        "rings": [[[0.2, 0.2], [1.2, 0.2], [1.2, 1.2], [0.2, 1.2]]],
        "ground_z_rings": [[101.0, 101.1, 101.2, 101.1]],
    }
    return {
        "schema": "fireviewer.detail-vector-lod.v1",
        "buildings": {"prisms": [prism]},
        "roads": {
            "meshes": {
                "carriageway": {
                    "vertices": [
                        [0.0, 0.0, 100.1],
                        [2.0, 0.0, 104.1],
                        [0.0, 1.0, 101.1],
                    ],
                    "faces": [[0, 1, 2]],
                },
                "center_markings": {"vertices": [], "faces": []},
            }
        },
        "water": {
            "segments": {
                "mesh": {
                    "vertices": [
                        [0.0, 1.0, 101.08],
                        [2.0, 1.0, 105.08],
                        [0.0, 2.0, 102.08],
                    ],
                    "faces": [[0, 1, 2]],
                }
            },
            "surfaces": {"mesh": {"vertices": [], "faces": []}},
        },
    }


def _detail_container() -> bytes:
    package = _package()
    bounds = package["metadata"]["bounds_l93_m"]
    origin = package["metadata"]["origin_l93_m"]
    terrain = encode_detail_terrain(package["terrain"], bounds, origin)
    trees = encode_tree_instances(package, bounds, origin)
    vectors = build_vector_sections(_detail_vectors(), bounds, origin)
    return build_container(
        kind="detail_tile",
        tile_id="x0_y0_s2",
        bounds_l93_m=bounds,
        origin_l93_m=origin,
        sections=[
            ("terrain", *terrain),
            ("trees", *trees),
            ("buildings", *vectors["buildings"]),
            ("roads", *vectors["roads"]),
            ("water", *vectors["water"]),
        ],
    )


def test_round_trip_preserves_grid_all_trees_and_vector_meshes() -> None:
    payload = _detail_container()
    parsed = read_container(payload)
    headers = {item["name"]: item for item in parsed["header"]["sections"]}
    decoded = decode_u16_elevations(
        parsed["sections"]["terrain"], headers["terrain"]["metadata"]
    )
    expected = [vertex[2] for vertex in _terrain()["vertices"]]
    maximum_error = headers["terrain"]["metadata"]["elevation_quantization"][
        "maximum_observed_error_m"
    ]
    assert (
        max(abs(left - right) for left, right in zip(decoded, expected, strict=True))
        <= maximum_error + 1e-12
    )
    assert len(parsed["sections"]["trees"]) == 2 * TREE_RECORD.size
    assert headers["trees"]["metadata"]["count"] == 2
    assert headers["buildings"]["metadata"]["mesh_count"] == 1
    assert headers["buildings"]["metadata"]["triangle_count"] == 10
    assert headers["roads"]["metadata"]["triangle_count"] == 1
    assert headers["water"]["metadata"]["triangle_count"] == 1


def test_container_is_byte_deterministic_and_detects_corruption() -> None:
    first = _detail_container()
    second = _detail_container()
    assert first == second
    corrupted = bytearray(first)
    corrupted[-1] ^= 0x01
    with pytest.raises(FWTileError, match="checksum"):
        read_container(bytes(corrupted))


def test_far_grid_encodes_nodata_mask_and_quantization() -> None:
    raw, metadata = encode_far_grid(
        [10.0, 11.0, 0.0, 13.0],
        [True, True, False, True],
        rows=2,
        columns=2,
        pixel_size_m=[5.0, 5.0],
        outer_bounds_l93_m=[0.0, 0.0, 10.0, 10.0],
    )
    assert metadata["valid_sample_count"] == 3
    assert metadata["validity_mask_bytes"] == 1
    assert len(raw) == 2 * 4 + 1
    assert raw[-1] == 0b00001011


def test_detail_terrain_may_keep_an_exact_regular_subgrid_at_source_edge() -> None:
    terrain = _terrain()
    terrain["vertices"] = terrain["vertices"][:6]
    terrain["faces"] = terrain["faces"][:2]
    terrain["geometric_bounds_l93_m"] = [0.0, 1.0, 2.0, 2.0]
    _raw, metadata = encode_detail_terrain(
        terrain, [0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 0.0]
    )
    assert metadata["rows"] == 2
    assert metadata["columns"] == 3
    assert metadata["geometric_bounds_l93_m"] == [0.0, 1.0, 2.0, 2.0]


def test_tree_ground_z_is_already_local_to_the_shared_origin() -> None:
    package = _package()
    package["metadata"]["origin_l93_m"] = [0.0, 0.0, 320.0]
    raw, _metadata = encode_tree_instances(
        package, [0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 320.0]
    )
    first = TREE_RECORD.unpack_from(raw)
    assert first[2] == 101_250


def test_tree_dimensions_above_65_metres_are_preserved_without_clamping() -> None:
    package = _package()
    package["instances"]["values"][0][3] = 79.434
    raw, metadata = encode_tree_instances(
        package, [0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 0.0]
    )
    first = TREE_RECORD.unpack_from(raw)
    assert metadata["encoding"] == "tree-instance-position-mm-dimension-cm.v2"
    assert first[3] == 7_943
    assert first[4] == 450


def test_missing_source_corner_is_cropped_to_largest_measured_grid() -> None:
    terrain = _terrain()
    terrain["vertices"] = terrain["vertices"][:-1]
    terrain["faces"] = terrain["faces"][:-1]
    cropped = _crop_to_largest_measured_regular_grid(terrain)
    assert cropped["vertex_count"] == 6
    assert cropped["face_count"] == 2
    assert {vertex[1] for vertex in cropped["vertices"]} == {1.0, 2.0}


def test_self_touching_clipped_building_is_repaired_without_dropping_its_mesh() -> None:
    prism = {
        "feature_id": "building.boundary-touch",
        "base_z": 10.0,
        "roof_z": 15.0,
        "height": 5.0,
        "rings": [
            [
                [0.0, 4.0],
                [-1.0, 3.0],
                [-1.0, 2.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [-0.5, -0.5],
                [0.0, -1.0],
                [0.0, 1.0],
            ]
        ],
        "ground_z_rings": [[10.0] * 8],
    }
    mesh = prism_to_mesh(prism)
    assert mesh["vertices"]
    assert mesh["faces"]
    assert all(len(set(face)) == 3 for face in mesh["faces"])


def test_repaired_roof_interpolates_new_boundary_vertex_ground_height() -> None:
    prism = {
        "feature_id": "building.crossing",
        "base_z": 9.0,
        "roof_z": 20.0,
        "height": 11.0,
        "rings": [[[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]]],
        "ground_z_rings": [[10.0, 12.0, 14.0, 16.0]],
    }

    mesh = prism_to_mesh(prism)

    crossing_ground_heights = [
        z for x, y, z in mesh["vertices"] if (x, y) == (1.0, 1.0) and z != 20.0
    ]
    assert crossing_ground_heights
    assert set(crossing_ground_heights) == {15.0}
    assert mesh["faces"]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _export_fixture(
    tmp_path: Path, *, include_image: bool = True, near_lod_disabled: bool = False
):
    manifest_dir = tmp_path / "global-05m"
    tile_id = "x0_y0_s2"
    tile_dir = manifest_dir / "tiles" / tile_id
    tile_dir.mkdir(parents=True)
    mid_path = tile_dir / "mid.json.gz"
    with gzip.open(mid_path, "wt", encoding="utf-8") as stream:
        json.dump(_package(), stream)
    image_path = tile_dir / "ortho.jpg"
    if include_image:
        image_path.write_bytes(b"\xff\xd8synthetic-jpeg\xff\xd9")
    assets = {
        "mid_package": {
            "path": f"tiles/{tile_id}/mid.json.gz",
            "sha256": sha256_file(mid_path),
            "byte_count": mid_path.stat().st_size,
        },
        "near_orthophoto_image": {
            "path": f"tiles/{tile_id}/ortho.jpg",
            "sha256": sha256_file(image_path) if include_image else "0" * 64,
            "byte_count": image_path.stat().st_size if include_image else 1,
        },
        "orthophoto_image": {
            "path": f"tiles/{tile_id}/ortho.jpg",
            "sha256": sha256_file(image_path) if include_image else "0" * 64,
            "byte_count": image_path.stat().st_size if include_image else 1,
        },
    }
    manifest = {
        "schema": "fireviewer.global-05m-production-manifest.v1",
        "status": "ready",
        "plan_id": "test-plan",
        "origin_l93_m": [0.0, 0.0, 0.0],
        "tiles": [
            {
                "id": tile_id,
                "bounds_l93_m": [0.0, 0.0, 2.0, 2.0],
                "origin_l93_m": [0.0, 0.0, 0.0],
                "status": {"state": "ready"},
                "near_orthophoto_status": {"state": "ready"},
                "production_statistics": {"accepted_instance_count": 2},
                "assets": assets,
            }
        ],
    }
    manifest_path = manifest_dir / "production-manifest.json"
    _write_json(manifest_path, manifest)
    global_path = tmp_path / "global-vectors.json.gz"
    with gzip.open(global_path, "wt", encoding="utf-8") as stream:
        json.dump({"schema": "fireviewer.blender-preview-package.v2"}, stream)
    calls = []

    def builder(global_package, package, bounds, origin):
        calls.append((global_package, bounds, origin))
        return _detail_vectors()

    context = create_context(
        manifest_path,
        global_path,
        tmp_path / "remote",
        near_lod_disabled=near_lod_disabled,
        vector_builder=builder,
    )
    return context, manifest["tiles"][0], calls


def test_export_is_atomic_content_addressed_and_resumes_from_receipt(
    tmp_path: Path,
) -> None:
    context, tile, calls = _export_fixture(tmp_path)
    first = export_tile(context, tile)
    second = export_tile(context, tile)
    assert first == second
    assert len(calls) == 1
    assert first["payload"]["url"].endswith(".fwtile")
    assert first["imagery"]["url"].endswith(".jpg")
    assert len(first["payload"]["sha256"]) == 64
    assert not list(context.output_root.rglob("*.part"))
    payload_path = context.output_root / Path(*first["payload"]["url"].split("/"))
    parsed = read_container(payload_path.read_bytes(), decode_sections=False)
    assert [item["name"] for item in parsed["header"]["sections"]] == [
        "terrain",
        "trees",
        "buildings",
        "roads",
        "water",
    ]


def test_resume_adopts_a_byte_compatible_exporter_receipt(tmp_path: Path) -> None:
    context, tile, calls = _export_fixture(tmp_path)
    first = export_tile(context, tile)
    receipt_path = context.output_root / "receipts" / f"{tile['id']}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["inputs"]["exporter_sha256"] = next(iter(COMPATIBLE_EXPORTER_SHA256S))
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    resumed = export_tile(context, tile)

    assert resumed == first
    assert len(calls) == 1
    adopted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert adopted["inputs"]["exporter_sha256"] == context.exporter_sha256


def test_export_fails_closed_when_selected_orthophoto_is_missing(
    tmp_path: Path,
) -> None:
    context, tile, _calls = _export_fixture(tmp_path, include_image=False)
    with pytest.raises(FWTileError, match="missing"):
        export_tile(context, tile)


def test_catalog_has_two_lods_and_enforces_global_budget(tmp_path: Path) -> None:
    context, tile, _calls = _export_fixture(tmp_path)
    export_tile(context, tile)
    digest = "a" * 64
    far = {
        "terrain": {
            "url": "far/global.fwterrain",
            "sha256": digest,
            "byte_count": 10,
            "resolution_m": [5.0, 5.0],
        },
        "imagery": {
            "url": "far/global.jpg",
            "sha256": "b" * 64,
            "byte_count": 10,
            "resolution_m": 2.0,
        },
        "bounds_l93_m": [0.0, 0.0, 2.0, 2.0],
    }
    catalog = build_catalog(context, far)
    assert catalog["lod_policy"]["far"]["role"] == "always_available_global_fallback"
    assert catalog["lod_policy"]["detail"]["maximum_resident_tile_count"] == 16
    assert catalog["lod_policy"]["detail"]["near_disabled"] is False
    assert catalog["exported_detail_tile_count"] == 1
    catalog["lod_policy"]["detail"]["maximum_resident_tile_count"] = 17
    with pytest.raises(FWTileError, match="16 tile"):
        validate_catalog(catalog)

    catalog = build_catalog(context, far)
    catalog["lod_policy"]["detail"]["publish_distance_m"] = 601.0
    with pytest.raises(FWTileError, match="600 m"):
        validate_catalog(catalog)


def test_disabled_near_lod_keeps_mid_imagery_and_declares_policy(
    tmp_path: Path,
) -> None:
    context, tile, _calls = _export_fixture(tmp_path, near_lod_disabled=True)
    record = export_tile(context, tile)
    digest = "a" * 64
    catalog = build_catalog(
        context,
        {
            "terrain": {
                "url": "far/global.fwterrain",
                "sha256": digest,
                "byte_count": 10,
                "resolution_m": [5.0, 5.0],
            },
            "imagery": {
                "url": "far/global.jpg",
                "sha256": digest,
                "byte_count": 10,
                "resolution_m": 2.0,
            },
            "bounds_l93_m": [0.0, 0.0, 2.0, 2.0],
        },
    )

    assert record["imagery"]["resolution_m"] == 0.5
    assert catalog["lod_policy"]["detail"]["near_disabled"] is True
    validate_catalog(catalog)


def test_detail_zones_are_exported_first_in_contract_order() -> None:
    zones = [
        ("montmaur", [10.5, 10.5, 20.5, 20.5]),
        ("barsac", [30.5, 30.5, 40.5, 40.5]),
        ("ausson", [50.5, 50.5, 60.5, 60.5]),
    ]

    def tile(identifier: str, bounds: list[float]) -> dict:
        return {"id": identifier, "bounds_l93_m": bounds}

    unordered = [
        tile("outside-a", [70.0, 70.0, 80.0, 80.0]),
        tile("ausson-a", [50.0, 50.0, 55.0, 55.0]),
        tile("montmaur-b", [15.0, 15.0, 20.0, 20.0]),
        tile("barsac-a", [30.0, 30.0, 35.0, 35.0]),
        tile("touch-only", [0.5, 10.5, 10.5, 20.5]),
        tile("montmaur-a", [10.0, 10.0, 15.0, 15.0]),
    ]

    ordered = _prioritize_tiles(unordered, zones)

    assert [item["id"] for item in ordered] == [
        "montmaur-a",
        "montmaur-b",
        "barsac-a",
        "ausson-a",
        "outside-a",
        "touch-only",
    ]
