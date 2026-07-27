from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from build_control_scene import (
    _assert_collection_has_no_realize_instances,
    _resident_tile_ids_within_budget,
    _triangulated_terrain_faces,
    _validate_detail_terrain_core_contract,
    apply_scene_distance_lod,
    apply_tiled_collection_visibility,
    scene_distance_lod_strategy,
    tiled_compositing_strategy,
)
from tiled_scene import (
    load_global_05m_manifest,
    ready_tiles,
    select_tile_asset,
    tile_distance_to_point_m,
    tile_is_visible,
)


def _tile(identifier: str, *, state: str = "ready") -> dict[str, object]:
    return {
        "id": identifier,
        "bounds_l93_m": [880_000.0, 6_400_000.0, 880_500.0, 6_400_500.0],
        "processing_bounds_l93_m": [
            879_990.0,
            6_399_990.0,
            880_510.0,
            6_400_510.0,
        ],
        "origin_l93_m": [879_000.0, 6_399_000.0, 300.0],
        "aoi_intersection_area_m2": 200_000.0,
        "source_tile_ids": ["0880_6401"],
        "status": {"state": state, "attempt_count": 1, "last_error": None},
        "assets": {
            "mid_package": {"path": f"tiles/{identifier}/mid.json.gz"},
            "orthophoto_source": {"path": f"tiles/{identifier}/ortho.source.json"},
            "orthophoto_image": {"path": f"tiles/{identifier}/ortho.jpg"},
            "orthophoto_geotiff": {"path": f"tiles/{identifier}/ortho.tif"},
            "blender_library": {"path": f"tiles/{identifier}/tile.blend"},
        },
        "visibility": {
            "default_visible": False,
            "activation": "camera_or_selected_attention_zone",
            "lod": "mid_0m50",
        },
    }


def _manifest(*tiles: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "fireviewer.global-05m-production-manifest.v1",
        "crs": "EPSG:2154",
        "origin_l93_m": [879_000.0, 6_399_000.0, 300.0],
        "aoi": {
            "bounds_l93_m": [879_500.0, 6_399_500.0, 881_000.0, 6_401_000.0],
            "area_m2": 1_500_000.0,
            "sha256": "a" * 64,
        },
        "tiling": {
            "source_tile_size_m": 1000,
            "output_tile_size_m": 500,
            "halo_m": 10,
            "ownership_rule": "apex_in_half_open_core_min_inclusive_max_exclusive",
        },
        "source_tiles": [],
        "tiles": list(tiles),
    }


def test_manifest_and_explicit_ready_selection_are_deterministic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "index.json"
    path.write_text(
        json.dumps(
            _manifest(
                _tile("tile_b"), _tile("tile_a"), _tile("tile_c", state="planned")
            )
        ),
        encoding="utf-8",
    )
    loaded = load_global_05m_manifest(path)

    assert [tile["id"] for tile in ready_tiles(loaded)] == ["tile_a", "tile_b"]
    assert [tile["id"] for tile in ready_tiles(loaded, ["tile_b", "tile_a"])] == [
        "tile_a",
        "tile_b",
    ]
    with pytest.raises(ValueError, match="not ready"):
        ready_tiles(loaded, ["tile_c"])


def test_detail_terrain_requires_exact_bounds_and_deterministic_edges() -> None:
    tile = _tile("tile_a")
    terrain = {
        "geometric_bounds_l93_m": tile["bounds_l93_m"],
        "boundary_sampling": (
            "bilinear_processing_halo_at_exact_lambert93_core_coordinates"
        ),
        "adjacent_edge_contract": ("coincident_xy_and_identical_sample_coordinates"),
    }
    _validate_detail_terrain_core_contract(tile, terrain)

    without_bounds = dict(terrain)
    without_bounds.pop("geometric_bounds_l93_m")
    with pytest.raises(ValueError, match="no exact geometric core bounds"):
        _validate_detail_terrain_core_contract(tile, without_bounds)

    inset = dict(terrain)
    inset["geometric_bounds_l93_m"] = [
        880_000.25,
        6_400_000.25,
        880_499.75,
        6_400_499.75,
    ]
    with pytest.raises(ValueError, match="does not reach its 500 m core"):
        _validate_detail_terrain_core_contract(tile, inset)


def test_library_is_preferred_then_source_packages_are_verified(tmp_path: Path) -> None:
    tile = _tile("tile_a")
    root = tmp_path / "tiles" / "tile_a"
    root.mkdir(parents=True)
    mid = root / "mid.json.gz"
    mid.write_bytes(b"mid-package")
    source = root / "ortho.source.json"
    source.write_text("{}", encoding="utf-8")
    tile["assets"]["mid_package"]["sha256"] = hashlib.sha256(b"mid-package").hexdigest()
    manifest_path = tmp_path / "index.json"
    manifest_path.write_text(json.dumps(_manifest(tile)), encoding="utf-8")

    fallback = select_tile_asset(manifest_path, tile)
    assert fallback.kind == "source_packages"
    assert fallback.primary_path == mid
    assert fallback.orthophoto_source_path == source
    assert fallback.orthophoto_resolution_m == 0.5
    assert fallback.orthophoto_lod == "mid_0m50_fallback"

    library = root / "tile.blend"
    library.write_bytes(b"BLENDER")
    preferred = select_tile_asset(manifest_path, tile)
    assert preferred.kind == "blender_library"
    assert preferred.primary_path == library

    native = root / "ortho-0m20.source.json"
    native.write_text("{}", encoding="utf-8")
    tile["assets"]["near_orthophoto_source"] = {
        "path": "tiles/tile_a/ortho-0m20.source.json",
        "sha256": hashlib.sha256(b"{}").hexdigest(),
    }
    tile["near_orthophoto_request"] = {"resolution_m": 0.2}
    near = select_tile_asset(manifest_path, tile)
    assert near.kind == "source_packages"
    assert near.primary_path == mid
    assert near.orthophoto_source_path == native
    assert near.orthophoto_resolution_m == 0.2
    assert near.orthophoto_lod == "near_native_0m20"


def test_ready_tile_rejects_corrupt_mid_package(tmp_path: Path) -> None:
    tile = _tile("tile_a")
    root = tmp_path / "tiles" / "tile_a"
    root.mkdir(parents=True)
    (root / "mid.json.gz").write_bytes(b"wrong")
    (root / "ortho.source.json").write_text("{}", encoding="utf-8")
    tile["assets"]["mid_package"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        select_tile_asset(tmp_path / "index.json", tile)


def test_visibility_uses_distance_to_tile_bounds_not_center() -> None:
    tile = _tile("tile_a")
    assert (
        tile_distance_to_point_m(tile["bounds_l93_m"], (879_900.0, 6_400_250.0))
        == 100.0
    )
    assert tile_is_visible(tile, focus_l93_m=(879_900.0, 6_400_250.0), radius_m=100.0)
    assert not tile_is_visible(
        tile, focus_l93_m=(879_899.0, 6_400_250.0), radius_m=100.0
    )
    assert tile_is_visible(tile, explicitly_selected=True)
    assert not tile_is_visible(tile)


def test_manifest_rejects_duplicate_ids_and_non_portable_paths(tmp_path: Path) -> None:
    first = _tile("tile_a")
    duplicate = _tile("tile_a")
    path = tmp_path / "index.json"
    path.write_text(json.dumps(_manifest(first, duplicate)), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate tile id"):
        load_global_05m_manifest(path)

    first["assets"]["mid_package"]["path"] = "C:/absolute/mid.json.gz"
    path.write_text(json.dumps(_manifest(first)), encoding="utf-8")
    with pytest.raises(ValueError, match="relative to the manifest"):
        load_global_05m_manifest(path)


class _FakeCollection(dict):
    def __init__(self, name: str, **properties: object) -> None:
        super().__init__(properties)
        self.name = name
        self.hide_viewport = False
        self.hide_render = False
        self.objects: list[object] = []
        self.children: list[object] = []
        self.all_objects: list[object] = []


class _FakeObject(dict):
    def __init__(self, name: str, **properties: object) -> None:
        super().__init__(properties)
        self.name = name
        self.hide_viewport = False
        self.hide_render = False


class _FakeCollections(list[_FakeCollection]):
    def get(self, name: str) -> _FakeCollection | None:
        return next((item for item in self if item.name == name), None)


def test_tiled_compositing_strategy_starts_from_global_without_tree_proxies() -> None:
    strategy = tiled_compositing_strategy(["tile_b", "tile_a", "tile_b"])

    assert strategy == {
        "schema": "fireviewer.tiled-compositing-strategy.v3",
        "global_base_visibility_strategy": (
            "global_monolith_or_partitioned_context_plus_complete_detail"
        ),
        "global_terrain_visible": True,
        "global_context_visible": False,
        "global_vegetation_visible": False,
        "visible_detail_tile_ids": ["tile_a", "tile_b"],
        "detail_terrain_source_z_offset_m": 0.0,
        "detail_terrain_render_z_offset_m": 0.0,
    }


def test_terrain_quads_have_one_explicit_northwest_southeast_diagonal() -> None:
    assert _triangulated_terrain_faces([[0, 3, 4, 1], [8, 9, 10]]) == [
        (0, 3, 4),
        (0, 4, 1),
        (8, 9, 10),
    ]


def test_scene_distance_lod_strategy_has_exact_boundary_contract() -> None:
    near = scene_distance_lod_strategy(0.0)
    assert near["band"] == "near"
    assert near["vegetation_mode"] == "none"
    assert near["near_content"] == (
        "complete_0m50_tiles_plus_detail_vectors_when_covered"
    )
    assert near["terrain_lod_requested"] == "detail_0m50"
    assert scene_distance_lod_strategy(1_499.999)["band"] == "near"
    medium = scene_distance_lod_strategy(1_500.0)
    assert medium["band"] == "medium"
    assert medium["vegetation_mode"] == "none"
    assert medium["medium_content"] == ("mnt_plus_simple_models_without_tree_geometry")
    assert scene_distance_lod_strategy(5_999.999)["band"] == "medium"
    far = scene_distance_lod_strategy(6_000.0)
    assert far["band"] == "far"
    assert far["far_content"] == "mnt_only"
    assert far["terrain_visible"] is True
    assert far["simple_models_visible"] is False
    assert far["detail_tiles_visible"] is False
    assert far["terrain_lod_requested"] == "global_2m"

    with pytest.raises(ValueError, match="non-negative"):
        scene_distance_lod_strategy(-0.1)
    with pytest.raises(ValueError, match="greater than near"):
        scene_distance_lod_strategy(2_000.0, near_max_m=2_000.0, far_min_m=2_000.0)


def test_resident_tile_budget_rejects_before_materializing_seventeenth_tile() -> None:
    sixteen = [f"tile_{index:02d}" for index in range(16)]
    assert _resident_tile_ids_within_budget([], sixteen, 16) == sixteen
    assert _resident_tile_ids_within_budget(sixteen[:8], sixteen[8:], 16) == sixteen
    with pytest.raises(ValueError, match="requested 17, maximum 16"):
        _resident_tile_ids_within_budget(sixteen, ["tile_16"], 16)
    with pytest.raises(ValueError, match="positive integer"):
        _resident_tile_ids_within_budget([], [], 0)


def test_scene_distance_lod_applies_far_medium_and_near_visibility() -> None:
    tile = _FakeCollection(
        "GlobalTile_loaded",
        fireviewer_tile_id="loaded",
        fireviewer_core_bounds_l93_json=json.dumps(
            [880_000.0, 6_400_000.0, 880_500.0, 6_400_500.0]
        ),
        fireviewer_tile_loaded=True,
        detail_vector_lod_complete=True,
    )
    terrain = _FakeCollection("Terrain")
    context_tile = _FakeObject(
        "GlobalTerrainContext_loaded", fireviewer_tile_id="loaded"
    )
    global_context = _FakeCollection("GlobalTerrainContext")
    global_context.objects = [context_tile]
    vector_context_tile = _FakeCollection(
        "GlobalVectorContext_loaded",
        fireviewer_global_vector_context_chunk_id="loaded",
        fireviewer_tile_id="loaded",
        fireviewer_has_content=True,
    )
    vector_context_residual = _FakeCollection(
        "GlobalVectorContext___global_context_outside_manifest__",
        fireviewer_global_vector_context_chunk_id=(
            "__global_context_outside_manifest__"
        ),
        fireviewer_tile_id="",
        fireviewer_has_content=False,
    )
    vector_context = _FakeCollection("GlobalVectorContext")
    vector_context.children = [vector_context_tile, vector_context_residual]
    detail_parent = _FakeCollection("GlobalTiles")
    global_vegetation = _FakeCollection("Vegetation")
    simple = [
        _FakeCollection(name)
        for name in ("FirePerimeter", "Buildings", "Roads", "Water")
    ]
    scene: dict[str, object] = {}
    bpy = SimpleNamespace(
        data=SimpleNamespace(
            collections=_FakeCollections(
                [
                    tile,
                    terrain,
                    global_context,
                    vector_context,
                    global_vegetation,
                    detail_parent,
                    *simple,
                ]
            )
        ),
        context=SimpleNamespace(scene=scene),
    )

    far = apply_scene_distance_lod(bpy, 6_000.0)
    assert far["band"] == "far"
    assert not terrain.hide_viewport
    assert global_context.hide_viewport
    assert vector_context.hide_viewport
    assert detail_parent.hide_viewport and tile.hide_viewport
    assert global_context.hide_viewport
    assert all(collection.hide_viewport for collection in simple)
    assert global_vegetation.hide_viewport

    medium = apply_scene_distance_lod(bpy, 2_500.0)
    assert medium["band"] == "medium"
    assert detail_parent.hide_viewport and tile.hide_viewport
    assert not simple[0].hide_viewport
    assert all(collection.hide_viewport for collection in simple[1:])
    assert not vector_context.hide_viewport
    assert not vector_context_tile.hide_viewport
    assert global_vegetation.hide_viewport
    assert medium["vegetation_mode"] == "none"

    near = apply_scene_distance_lod(
        bpy,
        5.0,
        focus_l93_m=(880_250.0, 6_400_250.0),
        detail_radius_m=10.0,
    )
    assert near["band"] == "near"
    assert terrain.hide_viewport
    assert not global_context.hide_viewport and context_tile.hide_viewport
    assert not detail_parent.hide_viewport and not tile.hide_viewport
    assert global_vegetation.hide_viewport
    assert near["detail_coverage_complete"] is True
    assert near["vegetation_mode"] == "detailed_0m50_tiles"
    assert near["terrain_mode"] == "detail_0m50_tiles_with_global_context"
    assert near["global_terrain_visible"] is False
    assert near["global_context_visible"] is True
    assert near["global_context_coverage_complete"] is True
    assert near["global_vector_context_coverage_complete"] is True
    assert near["visible_global_context_tile_count"] == 0
    assert near["detail_terrain_visible"] is True
    assert near["terrain_overlap_active"] is False
    assert near["detail_vector_contract_declared"] is True
    assert near["detail_vector_coverage_complete"] is True
    assert near["global_vector_models_visible"] is True
    assert near["legacy_global_vector_models_visible"] is False
    assert near["global_vector_context_visible"] is True
    assert vector_context_tile.hide_viewport
    assert not vector_context_residual.hide_viewport
    assert near["detail_vector_models_visible"] is True
    assert near["vector_model_overlap_active"] is False
    assert near["detail_required_coverage_radius_m"] == 6.0
    assert near["visible_detail_tile_ids"] == ["loaded"]
    assert not simple[0].hide_viewport
    assert all(collection.hide_viewport for collection in simple[1:])
    assert scene["scene_distance_lod_schema"] == "fireviewer.scene-distance-lod.v3"
    assert scene["scene_distance_lod_global_vegetation_visible"] is False
    assert scene["scene_distance_lod_vegetation_mode"] == "detailed_0m50_tiles"
    assert scene["scene_distance_lod_terrain_mode"] == (
        "detail_0m50_tiles_with_global_context"
    )
    assert scene["scene_distance_lod_global_context_visible"] is True
    assert scene["scene_distance_lod_terrain_overlap_active"] is False
    assert scene["scene_distance_lod_detail_vector_contract_declared"] is True
    assert scene["scene_distance_lod_detail_vector_coverage_complete"] is True
    assert scene["scene_distance_lod_global_vector_models_visible"] is True
    assert scene["scene_distance_lod_legacy_global_vector_models_visible"] is False
    assert scene["scene_distance_lod_global_vector_context_visible"] is True
    assert scene["scene_distance_lod_detail_vector_models_visible"] is True
    assert scene["scene_distance_lod_vector_model_overlap_active"] is False
    assert scene["global_base_hidden_for_detail_tiles"] is True
    assert scene["global_context_continuous_around_detail_tiles"] is True
    assert scene["global_05m_visible_tile_count"] == 1

    with pytest.raises(ValueError, match="requires a Lambert-93 focus"):
        apply_scene_distance_lod(bpy, 1_000.0)


def test_near_detail_fails_closed_when_radius_does_not_cover_view() -> None:
    tile = _FakeCollection(
        "GlobalTile_loaded",
        fireviewer_tile_id="loaded",
        fireviewer_core_bounds_l93_json=json.dumps(
            [880_000.0, 6_400_000.0, 880_500.0, 6_400_500.0]
        ),
        fireviewer_tile_loaded=True,
    )
    global_vegetation = _FakeCollection("Vegetation")
    terrain = _FakeCollection("Terrain")
    detail_parent = _FakeCollection("GlobalTiles")
    scene: dict[str, object] = {}
    bpy = SimpleNamespace(
        data=SimpleNamespace(
            collections=_FakeCollections(
                [tile, terrain, global_vegetation, detail_parent]
            )
        ),
        context=SimpleNamespace(scene=scene),
    )

    result = apply_scene_distance_lod(
        bpy,
        1_200.0,
        focus_l93_m=(880_250.0, 6_400_250.0),
        detail_radius_m=750.0,
    )

    assert result["band"] == "near"
    assert result["detail_tiles_requested_by_band"] is True
    assert result["detail_view_radius_covered"] is False
    assert result["detail_required_coverage_radius_m"] == 1_440.0
    assert result["detail_coverage_complete"] is False
    assert result["detail_coverage_reason"] == "view_exceeds_loaded_radius"
    assert result["vegetation_mode"] == "none"
    assert result["terrain_mode"] == "global_2m"
    assert result["global_terrain_visible"] is True
    assert result["visible_detail_tile_ids"] == []
    assert tile.hide_viewport and detail_parent.hide_viewport
    assert not terrain.hide_viewport
    assert global_vegetation.hide_viewport


def test_near_detail_fails_closed_without_complete_tile_local_vectors() -> None:
    tile = _FakeCollection(
        "GlobalTile_without_vectors",
        fireviewer_tile_id="without_vectors",
        fireviewer_core_bounds_l93_json=json.dumps(
            [880_000.0, 6_400_000.0, 880_500.0, 6_400_500.0]
        ),
        fireviewer_tile_loaded=True,
    )
    terrain = _FakeCollection("Terrain")
    detail_parent = _FakeCollection("GlobalTiles")
    buildings = _FakeCollection("Buildings")
    roads = _FakeCollection("Roads")
    water = _FakeCollection("Water")
    scene: dict[str, object] = {}
    bpy = SimpleNamespace(
        data=SimpleNamespace(
            collections=_FakeCollections(
                [tile, terrain, detail_parent, buildings, roads, water]
            )
        ),
        context=SimpleNamespace(scene=scene),
    )

    result = apply_scene_distance_lod(
        bpy,
        5.0,
        focus_l93_m=(880_250.0, 6_400_250.0),
        detail_radius_m=10.0,
    )

    assert result["detail_vector_contract_declared"] is False
    assert result["detail_vector_coverage_complete"] is False
    assert result["detail_coverage_complete"] is False
    assert result["detail_coverage_reason"] == "detail_vector_coverage_incomplete"
    assert result["global_terrain_visible"] is True
    assert result["global_vector_models_visible"] is True
    assert result["detail_vector_models_visible"] is False
    assert result["vector_model_overlap_active"] is False
    assert not terrain.hide_viewport
    assert all(not collection.hide_viewport for collection in (buildings, roads, water))
    assert tile.hide_viewport and detail_parent.hide_viewport
    assert scene["scene_distance_lod_detail_vector_coverage_complete"] is False


def test_near_detail_fails_closed_instead_of_showing_partial_tree_tiles() -> None:
    bounds = json.dumps([880_000.0, 6_400_000.0, 880_500.0, 6_400_500.0])
    loaded = _FakeCollection(
        "GlobalTile_loaded",
        fireviewer_tile_id="loaded",
        fireviewer_core_bounds_l93_json=bounds,
        fireviewer_tile_loaded=True,
    )
    missing = _FakeCollection(
        "GlobalTile_missing",
        fireviewer_tile_id="missing",
        fireviewer_core_bounds_l93_json=bounds,
        fireviewer_tile_loaded=False,
    )
    detail_parent = _FakeCollection("GlobalTiles")
    bpy = SimpleNamespace(
        data=SimpleNamespace(
            collections=_FakeCollections([loaded, missing, detail_parent])
        ),
        context=SimpleNamespace(scene={}),
    )

    result = apply_scene_distance_lod(
        bpy,
        5.0,
        focus_l93_m=(880_250.0, 6_400_250.0),
        detail_radius_m=10.0,
    )

    assert result["detail_resident_coverage_complete"] is False
    assert result["detail_coverage_reason"] == "resident_tile_coverage_incomplete"
    assert result["missing_detail_tile_ids"] == ["missing"]
    assert result["visible_detail_tile_ids"] == []
    assert (
        loaded.hide_viewport and missing.hide_viewport and detail_parent.hide_viewport
    )


def test_near_detail_never_exceeds_sixteen_visible_tiles() -> None:
    bounds = json.dumps([880_000.0, 6_400_000.0, 880_500.0, 6_400_500.0])
    tiles = [
        _FakeCollection(
            f"GlobalTile_{index:02d}",
            fireviewer_tile_id=f"tile_{index:02d}",
            fireviewer_core_bounds_l93_json=bounds,
            fireviewer_tile_loaded=True,
        )
        for index in range(17)
    ]
    detail_parent = _FakeCollection("GlobalTiles")
    bpy = SimpleNamespace(
        data=SimpleNamespace(collections=_FakeCollections([*tiles, detail_parent])),
        context=SimpleNamespace(scene={}),
    )

    result = apply_scene_distance_lod(
        bpy,
        5.0,
        focus_l93_m=(880_250.0, 6_400_250.0),
        detail_radius_m=10.0,
    )

    assert result["detail_tile_budget_satisfied"] is False
    assert result["detail_coverage_reason"] == "detail_tile_budget_exceeded"
    assert result["visible_detail_tile_count"] == 0
    assert all(tile.hide_viewport for tile in tiles)
    assert detail_parent.hide_viewport

    with pytest.raises(ValueError, match="hard ceiling of 16"):
        apply_scene_distance_lod(
            bpy,
            5.0,
            focus_l93_m=(880_250.0, 6_400_250.0),
            detail_radius_m=10.0,
            maximum_detail_tile_count=17,
        )


def test_collection_culling_keeps_global_soil_but_hides_tree_proxies_pre_lod() -> None:
    loaded = _FakeCollection(
        "GlobalTile_loaded",
        fireviewer_tile_id="loaded",
        fireviewer_core_bounds_l93_json=json.dumps(
            [880_000.0, 6_400_000.0, 880_500.0, 6_400_500.0]
        ),
        fireviewer_tile_loaded=True,
    )
    placeholder = _FakeCollection(
        "GlobalTile_placeholder",
        fireviewer_tile_id="placeholder",
        fireviewer_core_bounds_l93_json=json.dumps(
            [880_500.0, 6_400_000.0, 881_000.0, 6_400_500.0]
        ),
        fireviewer_tile_loaded=False,
    )
    terrain = _FakeCollection("Terrain")
    vegetation = _FakeCollection("Vegetation")
    scene: dict[str, object] = {}
    bpy = SimpleNamespace(
        data=SimpleNamespace(
            collections=_FakeCollections([loaded, placeholder, terrain, vegetation])
        ),
        context=SimpleNamespace(scene=scene),
    )

    visible = apply_tiled_collection_visibility(bpy, (880_250.0, 6_400_250.0), 10.0)
    assert visible == ["loaded"]
    assert not loaded.hide_viewport
    assert placeholder.hide_viewport
    assert not terrain.hide_viewport and vegetation.hide_viewport
    assert scene["global_05m_visible_tile_count"] == 1
    assert scene["global_base_hidden_for_detail_tiles"] is False
    assert scene["global_base_visibility_strategy"] == (
        "global_monolith_or_partitioned_context_plus_complete_detail"
    )
    assert scene["detail_terrain_source_z_offset_m"] == 0.0
    assert scene["detail_terrain_render_z_offset_m"] == 0.0

    visible = apply_tiled_collection_visibility(bpy, (900_000.0, 6_500_000.0), 10.0)
    assert visible == []
    assert not terrain.hide_viewport and vegetation.hide_viewport

    with pytest.raises(ValueError, match="prove complete detail coverage"):
        apply_tiled_collection_visibility(
            bpy,
            (880_250.0, 6_400_250.0),
            10.0,
            hide_global_base=True,
        )


def test_realize_instances_guard_rejects_node_and_realized_flag() -> None:
    collection = _FakeCollection("Tile")
    collection.all_objects = [
        {
            "instances_realized": True,
        }
    ]
    with pytest.raises(ValueError, match="realized instances"):
        _assert_collection_has_no_realize_instances(collection)

    realize_node = SimpleNamespace(bl_idname="GeometryNodeRealizeInstances")
    collection.all_objects = [
        SimpleNamespace(
            get=lambda key, default=None: False,
            modifiers=[
                SimpleNamespace(node_group=SimpleNamespace(nodes=[realize_node]))
            ],
        )
    ]
    with pytest.raises(ValueError, match="Realize Instances"):
        _assert_collection_has_no_realize_instances(collection)
