from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import sys
import threading
import time
from types import SimpleNamespace
import uuid
import zlib

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import build_measured_scene_usd as measured  # noqa: E402


D_TEST_ROOT = (
    Path(
        os.environ.get(
            "FIREVIEWER_TEST_ROOT",
            "D:/Dev/project/fireviewer-repositories/fireviewer-work/temp/pytest",
        )
    )
    / "measured-scene"
)


def _png_bytes() -> bytes:
    def chunk(name: bytes, payload: bytes) -> bytes:
        return (
            len(payload).to_bytes(4, "big")
            + name
            + payload
            + (zlib.crc32(name + payload) & 0xFFFFFFFF).to_bytes(4, "big")
        )

    row = b"\0" + bytes((58, 112, 47, 255))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row))
        + chunk(b"IEND", b"")
    )


PROTOTYPE_BYTES = b"""#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 0.001
    upAxis = "Y"
)
def Xform "Asset"
{
}
"""
TEXTURE_BYTES = _png_bytes()


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_json(value: object) -> str:
    return _hash_bytes(measured.canonical_json_bytes(value))


def _asset(
    asset_id: str,
    category: str,
    *,
    usd_sha256: str,
    qualification: dict | None = None,
) -> dict:
    return {
        "asset_id": asset_id,
        "category": category,
        "usd": {
            "root": "review_batch",
            "path": "prototypes/source.usda",
            "byte_count": len(PROTOTYPE_BYTES),
            "sha256": usd_sha256,
        },
        "texture": {
            "root": "review_batch",
            "path": "prototypes/textures/source.png",
            "byte_count": len(TEXTURE_BYTES),
            "sha256": _hash_bytes(TEXTURE_BYTES),
        },
        "source_bounds": {
            "status": "reported",
            "coordinate_space": "source_glb_unscaled",
            "minimum": [-1.0, -0.5, -2.0],
            "maximum": [1.0, 3.5, 2.0],
            "diagonal": 6.0,
        },
        "qualification": qualification
        or {
            "dimensions": {"status": "pending", "value_m": None},
            "ground_anchor": {"status": "pending", "offset_m": None},
        },
    }


def _library(usd_sha256: str) -> dict:
    assets = [
        *(
            _asset(f"building_{index:02d}", "building", usd_sha256=usd_sha256)
            for index in range(24)
        ),
        *(
            _asset(f"tree_{index:02d}", "tree", usd_sha256=usd_sha256)
            for index in range(18)
        ),
        *(
            _asset(f"road_{index:02d}", "road_equipment", usd_sha256=usd_sha256)
            for index in range(8)
        ),
        *(
            _asset(f"vehicle_{index:02d}", "vehicle", usd_sha256=usd_sha256)
            for index in range(2)
        ),
        _asset("pasture_00", "pasture_equipment", usd_sha256=usd_sha256),
    ]
    assets.sort(key=lambda value: value["asset_id"])
    return {
        "schema": "fireviewer.asset-library.v1",
        "catalog_revision": _hash_bytes(b"synthetic-53-catalog"),
        "asset_count": 53,
        "assets": assets,
        "selection_pools": {
            "building": sorted(
                asset["asset_id"] for asset in assets if asset["category"] == "building"
            ),
            "tree": sorted(
                asset["asset_id"] for asset in assets if asset["category"] == "tree"
            ),
        },
    }


def _inventory() -> dict:
    buildings = [
        {
            "candidate_id": "building-valid-footprint",
            "status": "valid",
            "reason_codes": [],
            "anchor_l93_m": [700_020.0, 6_600_020.0],
            "ground_elevation_mm": 100_125,
            "height_cm": 800,
            "footprint_geojson": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [700_015.0, 6_600_017.0],
                        [700_025.0, 6_600_017.0],
                        [700_025.0, 6_600_023.0],
                        [700_015.0, 6_600_023.0],
                        [700_015.0, 6_600_017.0],
                    ]
                ],
            },
        },
        {
            "candidate_id": "building-valid-component",
            "status": "valid",
            "reason_codes": [],
            "anchor_l93_m": [700_080.0, 6_600_090.0],
            "ground_elevation_mm": 101_000,
            "height_cm": 900,
            "component_footprint_width_m": 12.0,
            "component_footprint_depth_m": 5.0,
            "component_orientation_rad": 0.25,
        },
        {
            "candidate_id": "building-ambiguous",
            "status": "ambiguous",
            "reason_codes": ["footprint_exceeds_processing_halo"],
            "anchor_l93_m": [700_100.0, 6_600_100.0],
            "ground_elevation_mm": 100_000,
            "height_cm": 700,
        },
    ]
    trees = [
        {
            "candidate_id": "tree-valid",
            "status": "valid",
            "reason_codes": [],
            "position_l93_m": [700_200.5, 6_600_250.5],
            "ground_elevation_mm": 99_500,
            "height_cm": 1_000,
            "equivalent_crown_radius_m": 3.0,
        },
        {
            "candidate_id": "tree-rejected",
            "status": "rejected",
            "reason_codes": ["crown_area_below_4m2"],
            "position_l93_m": [700_210.5, 6_600_260.5],
            "ground_elevation_mm": 99_600,
            "height_cm": 300,
            "equivalent_crown_radius_m": 0.5,
        },
    ]
    payload = {
        "schema": "fireviewer.mns-mnt-placement-inventory.v1",
        "build_id": _hash_bytes(b"synthetic-placement-build"),
        "zone_id": "FR-30-00001",
        "crs": "EPSG:2154",
        "grid": {"core_bounds_l93_m": [700_000, 6_600_000, 700_500, 6_600_500]},
        "buildings": {
            "source_count": 3,
            "valid_count": 2,
            "ambiguous_count": 1,
            "rejected_count": 0,
            "placement_ready_count": 2,
            "placement_blocked_count": 1,
            "instantiated_asset_count": 0,
            "candidates": buildings,
        },
        "trees": {
            "source_count": 2,
            "valid_count": 1,
            "ambiguous_count": 0,
            "rejected_count": 1,
            "placement_ready_count": 1,
            "placement_blocked_count": 1,
            "instantiated_asset_count": 0,
            "candidates": trees,
        },
    }
    payload["inventory_sha256"] = _hash_json(payload)
    return payload


def test_factual_tree_uses_measured_crown_width_and_support_elevation(
    tmp_path: Path,
) -> None:
    prototype = measured._Prototype(
        family="trees",
        asset_id="tree-oak",
        reference="prototype.usda",
        source_path=tmp_path / "prototype.usda",
        source_relative="prototype.usda",
        source_sha256=_hash_bytes(b"prototype"),
        source_byte_count=9,
        texture_path=None,
        texture_relative=None,
        texture_sha256=None,
        texture_byte_count=None,
        wrapper_relative="tree-oak/prototype.usda",
        wrapper_bytes=b"",
        material_policy="test",
        source_up_axis="Y",
        native_min_y=-0.5,
        native_extents=(2.0, 4.0, 4.0),
        qualification_blockers=(),
        availability="real_usd",
        fallback_resolution=None,
    )
    candidate = {
        "candidate_id": "tree-factual",
        "position_l93_m": [700_010.0, 6_600_020.0],
        "ground_elevation_mm": 100_000,
        "support_elevation_mm": 101_250,
        "height_cm": 1_000,
        "equivalent_crown_radius_m": 3.0,
        "geometry_scale_policy": "uniform_fit_inside_measured_crown_and_height_bounds",
    }

    instance = measured._instance_from_candidate(
        candidate,
        family="trees",
        asset_id="tree-oak",
        asset_category="tree",
        selection_seed=1,
        prototype=prototype,
        terrain=measured.TerrainReference(
            tmp_path / "terrain.usda", (700_000.0, 6_600_000.0)
        ),
    )

    assert instance.scale == (1.5, 1.5, 1.5)
    assert instance.position == (10.0, 20.0, 101.25)
    assert instance.measured_horizontal_m == (6.0, 6.0)


def test_tree_semantic_tags_distinguish_conifers_oaks_and_mixed_forest() -> None:
    conifer_metadata = measured._candidate_selection_metadata(
        "trees",
        {
            "source_properties": {"nature": "Forêt fermée de conifères"},
            "context_classification": "vegetation_prior",
        },
        "tree",
    )
    assert conifer_metadata["semantic_tags"] == ["conifer"]
    oak_tags = measured._semantic_tags(
        measured._metadata_terms({"libelle2": "FUTAIE DE CHENES DECIDUS PURS"})
    )
    assert {"broadleaf", "oak"}.issubset(oak_tags)
    mixed_tags = measured._semantic_tags(
        measured._metadata_terms({"nature": "Forêt fermée mixte"})
    )
    assert {"broadleaf", "conifer"}.issubset(mixed_tags)


@pytest.mark.parametrize(
    ("height_cm", "area_m2", "expected_form"),
    [
        (650, 90.0, "low_rise_house"),
        (900, 140.0, "mid_rise_residential"),
        (1_400, 140.0, "multi_storey_residential"),
        (1_050, 300.0, "multi_storey_residential"),
    ],
)
def test_building_selection_metadata_uses_measured_form_without_special_evidence(
    height_cm: int, area_m2: float, expected_form: str
) -> None:
    metadata = measured._candidate_selection_metadata(
        "buildings",
        {
            "height_cm": height_cm,
            "footprint_area_m2": area_m2,
            "source_properties": {"nature": "Bâtiment indifférencié"},
            "footprint_geojson": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [12, 0], [12, 8], [0, 8], [0, 0]]],
            },
        },
        "building",
    )

    assert metadata["semantic_tags"] == ["residential"]
    assert metadata["measured_dimensions_m"] == [12.0, height_cm / 100.0, 8.0]
    assert metadata["building_form"] == expected_form


def test_building_instance_is_centered_and_uniformly_fitted_inside_footprint(
    tmp_path: Path,
) -> None:
    prototype = measured._Prototype(
        family="buildings",
        asset_id="house",
        reference="prototype.usda",
        source_path=tmp_path / "prototype.usda",
        source_relative="prototype.usda",
        source_sha256=_hash_bytes(b"prototype"),
        source_byte_count=9,
        texture_path=None,
        texture_relative=None,
        texture_sha256=None,
        texture_byte_count=None,
        wrapper_relative="house/prototype.usda",
        wrapper_bytes=b"",
        material_policy="test",
        source_up_axis="Y",
        native_min_y=0.0,
        native_extents=(2.0, 4.0, 4.0),
        qualification_blockers=(),
        availability="real_usd",
        fallback_resolution=None,
    )
    candidate = {
        "candidate_id": "building-adjacent",
        "anchor_l93_m": [700_001.0, 6_600_001.0],
        "ground_elevation_mm": 100_000,
        "height_cm": 800,
        "footprint_geojson": {
            "type": "Polygon",
            "coordinates": [[
                [700_010.0, 6_600_020.0],
                [700_020.0, 6_600_020.0],
                [700_020.0, 6_600_026.0],
                [700_010.0, 6_600_026.0],
                [700_010.0, 6_600_020.0],
            ]],
        },
    }

    instance = measured._instance_from_candidate(
        candidate,
        family="buildings",
        asset_id="house",
        asset_category="building",
        selection_seed=1,
        prototype=prototype,
        terrain=measured.TerrainReference(
            tmp_path / "terrain.usda", (700_000.0, 6_600_000.0)
        ),
    )

    assert instance.position == (15.0, 23.0, 100.0)
    assert instance.scale == (2.5, 2.5, 2.5)
    assert instance.measured_horizontal_m == (10.0, 6.0)


def test_short_measured_crown_stays_a_tree_candidate() -> None:
    assert (
        measured._candidate_category(
            "trees",
            {"height_cm": 175},
            {
                "schema": measured.REFERENCE_CATALOG_SCHEMA,
                "selection_pools": {"tree": ["oak"], "vegetation": ["grass"]},
            },
        )
        == "tree"
    )


def test_factual_tree_selection_metadata_locks_conifer_or_oak_forms() -> None:
    metadata = measured._candidate_selection_metadata(
        "trees",
        {
            "source_properties": {"nature": "Bois"},
            "asset_selection_policy": (
                "current_bdtopo_composition_else_bdforet_v1_then_"
                "conifer_or_oak_only"
            ),
        },
        "tree",
    )

    assert metadata["tree_form_policy"] == "conifer_or_oak_only"


def test_self_contained_usdz_keeps_scoped_pbr_and_omits_redundant_texture(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    premium = asset_root / "premium-usdz" / "premium.usdz"
    preview = asset_root / "premium-usdz" / "textures" / "premium.jpg"
    premium.parent.mkdir(parents=True)
    preview.parent.mkdir(parents=True)
    premium.write_bytes(b"self-contained-usdz")
    preview.write_bytes(b"catalog-preview-only")
    asset = {
        "asset_id": "premium_building",
        "availability": "real_usd",
        "material": {
            "policy": "source_package_pbr",
            "source_package": True,
            "pbr_preserved": True,
        },
        "usd_stage": {
            "status": "inspected",
            "up_axis": "Z",
            "meters_per_unit": 1.0,
            "default_prim": "/Asset",
        },
        "usd": {
            "root": "review_batch",
            "path": "premium-usdz/premium.usdz",
            "byte_count": premium.stat().st_size,
            "sha256": measured.sha256_file(premium),
        },
        "texture": {
            "root": "review_batch",
            "path": "premium-usdz/textures/premium.jpg",
            "byte_count": preview.stat().st_size,
            "sha256": measured.sha256_file(preview),
        },
    }
    output = tmp_path / "scene"
    prototype = measured._plan_prototype_bundle(
        asset,
        family="buildings",
        asset_roots={"review_batch": asset_root},
        portable_root=tmp_path,
        bundle_root=output / "prototypes",
        output_root=output,
        native_min_y=0.0,
        native_extents=(1.0, 2.0, 1.0),
        qualification_blockers=(),
    )
    payloads = measured._prototype_payloads(prototype)
    assert set(payloads) == {
        "premium_building/source.usdz",
        "premium_building/prototype.usda",
    }
    assert b"source_package_pbr" in prototype.wrapper_bytes
    assert b"@source.usdz@" in prototype.wrapper_bytes
    assert b"xformOp:rotateX = -90" in prototype.wrapper_bytes


def test_normalized_usd_keeps_scoped_pbr_texture_and_upright_transform(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    normalized = asset_root / "simready-normalized" / "tree.usd"
    texture = asset_root / "simready-normalized" / "textures" / "tree.png"
    normalized.parent.mkdir(parents=True)
    texture.parent.mkdir(parents=True)
    normalized.write_bytes(PROTOTYPE_BYTES)
    texture.write_bytes(TEXTURE_BYTES)
    asset = {
        "asset_id": "normalized_tree",
        "availability": "real_usd",
        "material": {
            "policy": "scoped_source_pbr",
            "source_package": False,
            "pbr_preserved": True,
        },
        "usd_stage": {
            "status": "inspected",
            "up_axis": "Z",
            "meters_per_unit": 1.0,
            "default_prim": "/Asset",
        },
        "usd": {
            "root": "review_batch",
            "path": "simready-normalized/tree.usd",
            "byte_count": normalized.stat().st_size,
            "sha256": measured.sha256_file(normalized),
        },
        "texture": {
            "root": "review_batch",
            "path": "simready-normalized/textures/tree.png",
            "byte_count": texture.stat().st_size,
            "sha256": measured.sha256_file(texture),
        },
    }
    output = tmp_path / "scene"
    prototype = measured._plan_prototype_bundle(
        asset,
        family="trees",
        asset_roots={"review_batch": asset_root},
        portable_root=tmp_path,
        bundle_root=output / "prototypes",
        output_root=output,
        native_min_y=0.0,
        native_extents=(2.0, 6.0, 2.0),
        qualification_blockers=(),
    )
    payloads = measured._prototype_payloads(prototype)
    assert set(payloads) == {
        "normalized_tree/source.usd",
        "normalized_tree/textures/tree.png",
        "normalized_tree/prototype.usda",
    }
    assert b"scoped_source_pbr" in prototype.wrapper_bytes
    assert b"@source.usd@" in prototype.wrapper_bytes
    assert b"xformOp:rotateX = -90" in prototype.wrapper_bytes


def test_scoped_source_pbr_receipt_revalidates_the_exact_wrapper(
    fixture_root,
) -> None:
    root, terrain, prototype = fixture_root
    library = _library(_hash_bytes(prototype.read_bytes()))
    for asset in library["assets"]:
        asset["material"] = {
            "policy": "scoped_source_pbr",
            "source_package": False,
            "pbr_preserved": True,
        }
        asset["usd_stage"] = {
            "status": "inspected",
            "up_axis": "Y",
            "meters_per_unit": 0.001,
            "default_prim": "/Asset",
        }

    package = _build(root, terrain, prototype, "scene-scoped-pbr", library=library)
    receipt = measured.validate_measured_scene_package(package.output_root)

    assert receipt["prototype_count"] > 0
    assert {
        record["material"]["implementation"] for record in receipt["prototypes"]
    } == {"scoped_source_pbr"}
    for record in receipt["prototypes"]:
        assert record["material"] == measured._prototype_material_receipt(
            "scoped_source_pbr"
        )
        wrapper = (
            package.output_root
            / receipt["prototype_bundle"]["root_reference"]
            / record["wrapper"]["path"]
        )
        assert b"scoped_source_pbr" in wrapper.read_bytes()


def _selector(calls: list | None = None):
    def select(library, *, category, zone, candidate, rule_version, usage):
        if calls is not None:
            calls.append((category, zone, candidate, rule_version, usage))
        pool = sorted(library["selection_pools"][category])
        seed = int.from_bytes(
            hashlib.sha256(
                "\x00".join(
                    (zone, candidate, library["catalog_revision"], rule_version)
                ).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        return {
            "asset_id": pool[seed % len(pool)],
            "category": category,
            "selection_seed": seed,
            "usage_status": usage,
        }

    return select


@pytest.fixture
def fixture_root():
    root = D_TEST_ROOT / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        terrain = root / "terrain" / "terrain.usda"
        terrain.parent.mkdir()
        terrain.write_text(
            '#usda 1.0\n(def Xform "Terrain")\n', encoding="utf-8", newline="\n"
        )
        prototype = root / "assets" / "prototypes" / "source.usda"
        prototype.parent.mkdir(parents=True)
        prototype.write_bytes(PROTOTYPE_BYTES)
        texture = prototype.parent / "textures" / "source.png"
        texture.parent.mkdir()
        texture.write_bytes(TEXTURE_BYTES)
        yield root, terrain, prototype
    finally:
        if root.is_dir() and root.resolve().is_relative_to(D_TEST_ROOT.resolve()):
            shutil.rmtree(root)


def _build(root: Path, terrain: Path, prototype: Path, output_name: str, **kwargs):
    library = kwargs.pop("library", None)
    if library is None:
        library = _library(_hash_bytes(prototype.read_bytes()))
    inventory = kwargs.pop("inventory", _inventory())
    terrain_vertical_origin_mm = kwargs.pop("terrain_vertical_origin_mm", 0)
    return measured.build_measured_scene_usd(
        measured.TerrainReference(
            terrain,
            (700_000.0, 6_600_000.0),
            terrain_vertical_origin_mm,
        ),
        inventory,
        library,
        root / output_name,
        portable_root=root,
        asset_roots={"review_batch": root / "assets"},
        selection_api=kwargs.pop("selection_api", _selector()),
        **kwargs,
    )


def test_builds_bit_stable_measured_scene_without_quota_or_fallback(
    fixture_root,
) -> None:
    root, terrain, prototype = fixture_root
    calls: list = []
    first = _build(root, terrain, prototype, "scene-a", selection_api=_selector(calls))
    second = _build(root, terrain, prototype, "scene-b")

    assert first.scene.read_bytes() == second.scene.read_bytes()
    assert first.building_instance_count == 2
    assert first.tree_instance_count == 1
    assert len(calls) == 3
    assert sorted(call[2] for call in calls) == [
        "building-valid-component",
        "building-valid-footprint",
        "tree-valid",
    ]
    text = first.scene.read_text(encoding="utf-8")
    assert 'def PointInstancer "Buildings"' in text
    assert 'def PointInstancer "Trees"' in text
    assert 'custom string fireviewer:category = "terrain"' in text
    assert text.count('custom string fireviewer:category = "building"') == 1
    assert text.count('custom string fireviewer:category = "tree"') == 1
    assert "custom int fireviewer:count = 2" in text
    assert "custom int fireviewer:count = 1" in text
    assert "def Mesh" not in text
    assert "def Cube" not in text
    assert "fallback" not in text.casefold()
    assert "@../terrain/terrain.usda@" in text
    assert "@../assets/prototypes/source.usda@" not in text
    assert "@prototypes/" in text
    assert "/prototype.usda@" in text
    assert "custom bool fireviewer:quota_applied = false" in text
    assert "custom bool fireviewer:thinning_applied = false" in text
    assert "(20, 20, 100.125)" in text
    assert "(200.5, 250.5, 99.5)" in text
    assert "(0, 0.5, 0)" in text  # native minY is grounded before rotation
    assert "(2.5, 2.5, 2.5)" in text  # 10 m measured height / 4 m native height

    receipt = measured.validate_measured_scene_package(first.output_root)
    assert receipt["status"] == "technical_pilot_non_final"
    assert receipt["accepted_final"] is False
    assert receipt["reconciliation"]["buildings"]["source_count"] == 3
    assert receipt["reconciliation"]["buildings"]["instance_count"] == 2
    assert receipt["reconciliation"]["buildings"]["blocked_candidates"] == [
        {
            "candidate_id": "building-ambiguous",
            "reason_codes": ["footprint_exceeds_processing_halo"],
            "status": "ambiguous",
        }
    ]
    assert receipt["reconciliation"]["trees"]["source_count"] == 2
    assert receipt["reconciliation"]["trees"]["instance_count"] == 1
    assert receipt["placement_policy"]["fallback_primitive_used"] is False
    assert receipt["placement_policy"]["tree_scale"] == "uniform_from_mns_mnt_height"
    assert receipt["placement_policy"]["tree_yaw"] == "deterministic_selection_seed"
    assert receipt["prototype_bundle"]["root_reference"] == "prototypes"
    assert receipt["prototype_bundle"]["scope"] == "output_local"
    assert receipt["prototype_bundle"]["unused_catalog_assets_copied"] == 0
    assert (
        len(list((first.output_root / "prototypes").glob("*/prototype.usda")))
        == receipt["prototype_count"]
    )
    for wrapper in (first.output_root / "prototypes").glob("*/prototype.usda"):
        wrapper_text = wrapper.read_text(encoding="utf-8")
        assert 'uniform token info:id = "UsdPreviewSurface"' in wrapper_text
        assert 'bindMaterialAs = "strongerThanDescendants"' in wrapper_text
        assert "asset inputs:file = @textures/source.png@" in wrapper_text
    assert receipt["placement_policy"]["building_scale"] == (
        "uniform_fit_inside_measured_footprint_bounds"
    )
    assert receipt["placement_policy"]["non_uniform_building_scale_candidate_ids"] == []


def test_reference_catalog_is_validated_once_per_scene(
    monkeypatch, fixture_root
) -> None:
    root, terrain, prototype = fixture_root
    library = _library(_hash_bytes(prototype.read_bytes()))
    library["schema"] = measured.REFERENCE_CATALOG_SCHEMA
    validation_calls: list[dict] = []
    selection_calls: list = []
    delegate = _selector(selection_calls)

    def validate_catalog(payload):
        validation_calls.append(payload)

    def select_validated(payload, **kwargs):
        kwargs.pop("metadata", None)
        return delegate(payload, **kwargs)

    def select_public(*_args, **_kwargs):
        raise AssertionError("the scene hot path must use the prevalidated selector")

    monkeypatch.setitem(
        sys.modules,
        "build_reference_usd_asset_library",
        SimpleNamespace(
            validate_reference_asset_library=validate_catalog,
            _select_asset_for_candidate_from_validated_library=select_validated,
            select_asset_for_candidate=select_public,
        ),
    )

    package = _build(
        root,
        terrain,
        prototype,
        "scene-reference-catalog",
        library=library,
        selection_api=None,
    )

    assert package.building_instance_count == 2
    assert package.tree_instance_count == 1
    assert len(selection_calls) == 3
    assert len(validation_calls) == 1


def test_fixed_context_asset_bypasses_selection_and_keeps_exact_asset_id(
    fixture_root,
) -> None:
    root, terrain, prototype = fixture_root
    inventory = _inventory()
    inventory["context_assets"] = {
        "source_count": 1,
        "valid_count": 1,
        "ambiguous_count": 0,
        "rejected_count": 0,
        "placement_ready_count": 1,
        "placement_blocked_count": 0,
        "instantiated_asset_count": 0,
        "candidates": [
            {
                "candidate_id": "fixed-church-main",
                "status": "valid",
                "reason_codes": [],
                "fixed_placement_id": "church-main",
                "fixed_asset_id": "building_03",
                "asset_category": "building",
                "selection_context": "fixed_user_coordinate",
                "position_l93_m": [700_125.0, 6_600_175.0],
                "ground_elevation_mm": 100_750,
                "yaw_rad": 0.5,
            }
        ],
    }
    inventory.pop("inventory_sha256")
    inventory["inventory_sha256"] = _hash_json(inventory)
    calls: list = []
    result = _build(
        root,
        terrain,
        prototype,
        "scene-fixed",
        inventory=inventory,
        selection_api=_selector(calls),
    )
    assert result.context_asset_instance_count == 1
    assert len(calls) == 3
    text = result.scene.read_text(encoding="utf-8")
    assert "building_03" in text
    receipt = json.loads(result.receipt.read_text(encoding="utf-8"))
    assert any(record["asset_id"] == "building_03" for record in receipt["prototypes"])


def test_one_asset_can_be_reused_across_measured_and_fixed_families(
    fixture_root,
) -> None:
    root, terrain, prototype = fixture_root
    inventory = _inventory()
    inventory["context_assets"] = {
        "source_count": 1,
        "valid_count": 1,
        "ambiguous_count": 0,
        "rejected_count": 0,
        "placement_ready_count": 1,
        "placement_blocked_count": 0,
        "instantiated_asset_count": 0,
        "candidates": [
            {
                "candidate_id": "fixed-building-reuse",
                "status": "valid",
                "reason_codes": [],
                "fixed_placement_id": "fixed-building-reuse",
                "fixed_asset_id": "building_03",
                "asset_category": "building",
                "selection_context": "fixed_user_coordinate",
                "position_l93_m": [700_125.0, 6_600_175.0],
                "ground_elevation_mm": 100_750,
                "yaw_rad": 0.5,
            }
        ],
    }
    inventory.pop("inventory_sha256")
    inventory["inventory_sha256"] = _hash_json(inventory)

    def select(library, *, category, zone, candidate, rule_version, usage):
        del zone, candidate, rule_version
        asset_id = "building_03" if category == "building" else "tree_00"
        return {
            "asset_id": asset_id,
            "category": category,
            "selection_seed": 1,
            "usage_status": usage,
        }

    package = _build(
        root,
        terrain,
        prototype,
        "scene-reused-asset",
        inventory=inventory,
        selection_api=select,
        asset_bundle_root=root / "shared" / "prototype-bundles",
    )
    receipt = measured.validate_measured_scene_package(package.output_root)
    reused = [
        record
        for record in receipt["prototypes"]
        if record["asset_id"] == "building_03"
    ]
    assert {record["family"] for record in reused} == {"buildings", "context_assets"}
    assert receipt["prototype_bundle"]["scope"] == "explicit_shared"
    assert len(list((root / "shared" / "prototype-bundles").glob("building_03"))) == 1


def test_instance_altitudes_share_the_absolute_z_frame_with_terrain(
    fixture_root,
) -> None:
    root, terrain, prototype = fixture_root
    result = _build(
        root,
        terrain,
        prototype,
        "scene-with-datum",
        terrain_vertical_origin_mm=100_000,
    )
    text = result.scene.read_text(encoding="utf-8")
    assert "custom int fireviewer:vertical_origin_mm = 100000" in text
    assert "(20, 20, 100.125)" in text
    assert "(200.5, 250.5, 99.5)" in text
    assert "(20, 20, 0.125)" not in text


def test_candidate_and_catalog_order_do_not_change_scene(fixture_root) -> None:
    root, terrain, prototype = fixture_root
    library = _library(_hash_bytes(prototype.read_bytes()))
    inventory = _inventory()
    first = _build(
        root, terrain, prototype, "ordered", library=library, inventory=inventory
    )

    reordered_library = copy.deepcopy(library)
    reordered_library["assets"].reverse()
    reordered_library["selection_pools"]["building"].reverse()
    reordered_library["selection_pools"]["tree"].reverse()
    reordered_inventory = copy.deepcopy(inventory)
    reordered_inventory["buildings"]["candidates"].reverse()
    reordered_inventory["trees"]["candidates"].reverse()
    reordered_inventory.pop("inventory_sha256")
    reordered_inventory["inventory_sha256"] = _hash_json(reordered_inventory)
    second = _build(
        root,
        terrain,
        prototype,
        "reordered",
        library=reordered_library,
        inventory=reordered_inventory,
    )
    assert first.scene.read_bytes() == second.scene.read_bytes()


def test_final_usage_replaces_technically_blocked_assets_with_procedural_geometry(
    fixture_root,
) -> None:
    root, terrain, prototype = fixture_root
    package = _build(
        root, terrain, prototype, "final-resilient", usage="final_scene"
    )
    receipt = measured.validate_measured_scene_package(package.output_root)
    assert receipt["status"] == "assembled_final_candidate"
    assert receipt["procedural_instance_count"] == 3
    assert receipt["placement_policy"]["procedural_fallback_used"] is True
    assert receipt["final_blockers"] == []


def test_missing_or_escaping_asset_uses_traced_procedural_fallback(fixture_root) -> None:
    root, terrain, prototype = fixture_root
    library = _library(_hash_bytes(prototype.read_bytes()))
    selected_id = _selector()(
        library,
        category="tree",
        zone="FR-30-00001",
        candidate="tree-valid",
        rule_version="fireviewer.measured-tree-selection.v1",
        usage="technical_pilot_non_final",
    )["asset_id"]
    selected = next(
        asset for asset in library["assets"] if asset["asset_id"] == selected_id
    )
    selected["usd"]["path"] = "../outside.usda"
    escaped = _build(root, terrain, prototype, "escape", library=library)
    escaped_receipt = measured.validate_measured_scene_package(escaped.output_root)
    assert escaped_receipt["procedural_instance_count"] >= 1
    assert escaped_receipt["placement_policy"]["procedural_fallback_used"] is True

    library = _library(_hash_bytes(prototype.read_bytes()))
    prototype.unlink()
    missing = _build(root, terrain, prototype, "missing", library=library)
    missing_receipt = measured.validate_measured_scene_package(missing.output_root)
    assert missing_receipt["procedural_instance_count"] >= 1
    assert missing_receipt["placement_policy"]["procedural_fallback_used"] is True


def test_overwrite_c_drive_and_receipt_tamper_are_blocked(fixture_root) -> None:
    root, terrain, prototype = fixture_root
    package = _build(root, terrain, prototype, "scene")
    with pytest.raises(measured.MeasuredSceneError, match="overwrite"):
        _build(root, terrain, prototype, "scene")
    with pytest.raises(measured.MeasuredSceneError, match="forbidden on C"):
        measured.build_measured_scene_usd(
            measured.TerrainReference(terrain, (700_000.0, 6_600_000.0)),
            _inventory(),
            _library(_hash_bytes(prototype.read_bytes())),
            Path("C:/fireviewer/scene"),
            portable_root=root,
            asset_roots={"review_batch": root / "assets"},
            selection_api=_selector(),
        )
    receipt = json.loads(package.receipt.read_text(encoding="utf-8"))
    receipt["reconciliation"]["trees"]["instance_count"] = 0
    package.receipt.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(measured.MeasuredSceneError, match="receipt hash mismatch"):
        measured.validate_measured_scene_package(package.output_root)


def test_default_adapter_calls_public_catalog_selection_api(monkeypatch) -> None:
    calls: list = []
    selector = _selector(calls)
    monkeypatch.setitem(
        sys.modules,
        "build_asset_library_53",
        SimpleNamespace(select_asset_for_candidate=selector),
    )
    library = {
        "catalog_revision": _hash_bytes(b"catalog"),
        "selection_pools": {"building": ["building_00"]},
    }
    result = measured._default_selection_api(
        library,
        category="building",
        zone="FR-30-00001",
        candidate="building-1",
        rule_version="fireviewer.measured-building-selection.v1",
        usage="technical_pilot_non_final",
    )
    assert result["asset_id"] == "building_00"
    assert calls == [
        (
            "building",
            "FR-30-00001",
            "building-1",
            "fireviewer.measured-building-selection.v1",
            "technical_pilot_non_final",
        )
    ]


def test_contract_locks_measured_sources_and_quantity_preservation() -> None:
    contract = json.loads(
        Path(__file__)
        .with_name("measured_scene_usd_contract.v1.json")
        .read_text(encoding="utf-8")
    )
    assert contract["schema"] == measured.CONTRACT_SCHEMA
    assert contract["placement"]["quota"] == "forbidden"
    assert contract["placement"]["thinning"] == "forbidden"
    assert contract["placement"]["fallback_primitive"] == "forbidden"
    assert contract["outputs"]["storage_drive"] == "D"
    assert contract["prototype_bundle"]["scope"] == "selected_assets_only"
    implementations = contract["prototype_material"]["implementations"]
    assert implementations["fallback"]["implementation"] == "UsdPreviewSurface"
    assert implementations["scoped_source_pbr"] == {
        "binding_strength": "authored_below_default_prim",
        "texture_role": "source_usd_dependency",
        "source_color_space": "authored",
    }
    assert implementations["fallback"]["binding_strength"] == "strongerThanDescendants"
    assert contract["prototype_material"]["pbr_atlas"] == "forbidden"
    assert contract["normalization"]["building_scale"] == (
        "one_uniform_factor_fitted_inside_measured_oriented_footprint_bounds"
    )
    assert contract["normalization"]["non_uniform_building_scale"] == "forbidden"
    assert contract["selection"]["rule_versions"]["building"].endswith(".v2")
    assert contract["acceptance"]["builder_never_grants_final_acceptance"] is True


@pytest.mark.parametrize("artifact", ["source_usd", "texture", "wrapper"])
def test_each_bundled_prototype_artifact_is_tamper_evident(
    fixture_root, artifact: str
) -> None:
    root, terrain, prototype = fixture_root
    package = _build(root, terrain, prototype, f"tamper-{artifact}")
    receipt = json.loads(package.receipt.read_text(encoding="utf-8"))
    target = (
        package.output_root
        / receipt["prototype_bundle"]["root_reference"]
        / receipt["prototypes"][0][artifact]["path"]
    )
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(measured.MeasuredSceneError, match="bundle bytes differ"):
        measured.validate_measured_scene_package(package.output_root)


def test_bundle_remains_valid_without_original_asset_tree_and_after_move(
    fixture_root,
) -> None:
    root, terrain, prototype = fixture_root
    package = _build(root, terrain, prototype, "portable")
    shutil.rmtree(root / "assets")
    measured.validate_measured_scene_package(package.output_root)

    moved = root / "moved-package"
    shutil.copytree(package.output_root, moved)
    receipt = measured.validate_measured_scene_package(moved)
    assert receipt["prototype_bundle"]["absolute_asset_paths"] is False
    scene_text = (moved / "scene.usda").read_text(encoding="utf-8")
    assert str(root) not in scene_text
    assert "@prototypes/" in scene_text


def test_unused_file_in_local_bundle_is_rejected(fixture_root) -> None:
    root, terrain, prototype = fixture_root
    package = _build(root, terrain, prototype, "extra-file")
    (package.output_root / "prototypes" / "unused.txt").write_text(
        "not selected", encoding="utf-8"
    )
    with pytest.raises(measured.MeasuredSceneError, match="missing or unused"):
        measured.validate_measured_scene_package(package.output_root)


def test_zero_buildings_is_an_exact_valid_reconciliation(fixture_root) -> None:
    root, terrain, prototype = fixture_root
    inventory = _inventory()
    inventory["buildings"] = {
        "source_count": 0,
        "valid_count": 0,
        "ambiguous_count": 0,
        "rejected_count": 0,
        "placement_ready_count": 0,
        "placement_blocked_count": 0,
        "instantiated_asset_count": 0,
        "candidates": [],
    }
    inventory.pop("inventory_sha256")
    inventory["inventory_sha256"] = _hash_json(inventory)
    package = _build(root, terrain, prototype, "zero-buildings", inventory=inventory)
    receipt = measured.validate_measured_scene_package(package.output_root)
    assert package.building_instance_count == 0
    assert receipt["reconciliation"]["buildings"]["instance_count"] == 0
    assert receipt["reconciliation"]["trees"]["instance_count"] == 1


def test_explicit_shared_bundle_is_immutable_idempotent_and_not_duplicated(
    fixture_root,
) -> None:
    root, terrain, prototype = fixture_root
    shared = root / "shared" / "scene-prototypes"
    first = _build(
        root,
        terrain,
        prototype,
        "shared-scene-a",
        asset_bundle_root=shared,
    )
    first_files = {
        path.relative_to(shared).as_posix(): _hash_bytes(path.read_bytes())
        for path in shared.rglob("*")
        if path.is_file()
    }
    second = _build(
        root,
        terrain,
        prototype,
        "shared-scene-b",
        asset_bundle_root=shared,
    )
    second_files = {
        path.relative_to(shared).as_posix(): _hash_bytes(path.read_bytes())
        for path in shared.rglob("*")
        if path.is_file()
    }
    assert second_files == first_files
    for package in (first, second):
        receipt = measured.validate_measured_scene_package(package.output_root)
        assert receipt["prototype_bundle"]["scope"] == "explicit_shared"
        assert receipt["prototype_bundle"]["root_reference"].startswith("../shared/")
        assert not (package.output_root / "prototypes").exists()
        assert {path.name for path in package.output_root.iterdir()} == {
            "scene.usda",
            "scene.done.json",
        }


@pytest.mark.skipif(os.name == "nt", reason="linked bundle is a Linux worker mode")
def test_linked_bundle_uses_startup_validated_embedded_assets_without_rehash(
    fixture_root, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, terrain, prototype = fixture_root
    shared = root / "shared" / "scene-prototypes"
    texture = prototype.parent / "textures" / "source.png"
    prototype_hash = _hash_bytes(prototype.read_bytes())
    texture_hash = _hash_bytes(texture.read_bytes())
    measured.remember_validated_file_hash(prototype, prototype_hash)
    measured.remember_validated_file_hash(texture, texture_hash)
    original_sha256_file = measured.sha256_file
    calls = {prototype.resolve(): 0, texture.resolve(): 0}

    def counted_sha256_file(path: Path) -> str:
        resolved = path.resolve()
        if resolved in calls:
            calls[resolved] += 1
        return original_sha256_file(path)

    monkeypatch.setenv(measured.PROTOTYPE_BUNDLE_MODE_ENV, "linked")
    monkeypatch.setattr(measured, "sha256_file", counted_sha256_file)
    package = _build(
        root,
        terrain,
        prototype,
        "linked-scene",
        asset_bundle_root=shared,
    )
    receipt = measured.validate_measured_scene_package(package.output_root)
    source_link = shared / receipt["prototypes"][0]["source_usd"]["path"]
    texture_link = shared / receipt["prototypes"][0]["texture"]["path"]

    assert source_link.is_symlink()
    assert texture_link.is_symlink()
    assert source_link.resolve() == prototype.resolve()
    assert texture_link.resolve() == texture.resolve()
    assert calls == {prototype.resolve(): 0, texture.resolve(): 0}


@pytest.mark.skipif(os.name == "nt", reason="linked bundle is a Linux worker mode")
def test_linked_bundle_validates_lexical_source_name_and_detects_target_tamper(
    fixture_root, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, terrain, prototype = fixture_root
    embedded = prototype.with_name("embedded-normalized-model.usdc")
    prototype.rename(embedded)
    library = _library(_hash_bytes(embedded.read_bytes()))
    for asset in library["assets"]:
        asset["usd"]["path"] = "prototypes/embedded-normalized-model.usdc"
    texture = embedded.parent / "textures" / "source.png"
    measured.remember_validated_file_hash(embedded, _hash_bytes(embedded.read_bytes()))
    measured.remember_validated_file_hash(texture, _hash_bytes(texture.read_bytes()))
    shared = root / "shared" / "lexical-name-prototypes"
    monkeypatch.setenv(measured.PROTOTYPE_BUNDLE_MODE_ENV, "linked")

    package = _build(
        root,
        terrain,
        embedded,
        "linked-lexical-name-scene",
        library=library,
        asset_bundle_root=shared,
    )
    receipt = measured.validate_measured_scene_package(package.output_root)

    for record in receipt["prototypes"]:
        source_link = shared / record["source_usd"]["path"]
        wrapper = shared / record["wrapper"]["path"]
        assert source_link.name == "source.usdc"
        assert source_link.resolve() == embedded.resolve()
        assert source_link.name != embedded.name
        assert b"@source.usdc@" in wrapper.read_bytes()

    embedded.write_bytes(embedded.read_bytes() + b"\n# tampered\n")
    with pytest.raises(measured.MeasuredSceneError, match="bundle bytes differ"):
        measured.validate_measured_scene_package(package.output_root)


@pytest.mark.skipif(os.name == "nt", reason="linked bundle is a Linux worker mode")
def test_linked_bundle_rejects_a_symlinked_parent_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    outside = tmp_path / "outside"
    bundle.mkdir()
    outside.mkdir()
    target = outside / "source.usd"
    target.write_bytes(PROTOTYPE_BYTES)
    (bundle / "tree").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv(measured.PROTOTYPE_BUNDLE_MODE_ENV, "linked")

    with pytest.raises(measured.MeasuredSceneError, match="escapes portable root"):
        measured._validate_bundle_artifact(
            bundle,
            {
                "path": "tree/source.usd",
                "sha256": _hash_bytes(PROTOTYPE_BYTES),
                "byte_count": len(PROTOTYPE_BYTES),
            },
            label="prototype tree source USD",
            expected_prefix=PurePosixPath("tree"),
        )


def test_concurrent_shared_bundle_hashes_each_source_artifact_once(
    fixture_root, monkeypatch
) -> None:
    root, terrain, prototype = fixture_root
    shared = root / "shared" / "scene-prototypes"
    texture = prototype.parent / "textures" / "source.png"
    original_sha256_file = measured.sha256_file
    calls = {prototype.resolve(): 0, texture.resolve(): 0}
    calls_lock = threading.Lock()

    def counted_sha256_file(path: Path) -> str:
        resolved = path.resolve()
        if resolved in calls:
            with calls_lock:
                calls[resolved] += 1
            time.sleep(0.05)
        return original_sha256_file(path)

    monkeypatch.setattr(measured, "sha256_file", counted_sha256_file)
    with ThreadPoolExecutor(max_workers=8) as executor:
        packages = list(
            executor.map(
                lambda index: _build(
                    root,
                    terrain,
                    prototype,
                    f"concurrent-shared-scene-{index}",
                    asset_bundle_root=shared,
                ),
                range(8),
            )
        )

    assert calls == {prototype.resolve(): 1, texture.resolve(): 1}
    assert measured._SHARED_PROTOTYPE_LOCKS == {}
    for package in packages:
        receipt = measured.validate_measured_scene_package(package.output_root)
        assert receipt["prototype_bundle"]["scope"] == "explicit_shared"


def test_cross_process_style_publication_converges_without_shared_part_name(
    fixture_root, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _terrain, source = fixture_root
    shared = root / "shared" / "race-safe-prototypes"
    output = root / "scene-plan"
    asset = _library(_hash_bytes(source.read_bytes()))["assets"][0]
    prototype = measured._plan_prototype_bundle(
        asset,
        family="buildings",
        asset_roots={"review_batch": root / "assets"},
        portable_root=root,
        bundle_root=shared,
        output_root=output,
        native_min_y=0.0,
        native_extents=(2.0, 4.0, 4.0),
        qualification_blockers=(),
    )
    real_replace = measured.os.replace
    both_ready = threading.Barrier(2)

    def racing_replace(source_path: Path, target_path: Path) -> None:
        both_ready.wait(timeout=2.0)
        real_replace(source_path, target_path)

    monkeypatch.setattr(measured.os, "replace", racing_replace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(
            executor.map(
                lambda _index: measured._publish_shared_prototype_locked(
                    shared, prototype
                ),
                range(2),
            )
        )

    artifacts = measured._prototype_artifact_hashes(prototype)
    measured._validate_published_shared_prototype(
        shared / prototype.asset_id, prototype, artifacts
    )
    assert not list(shared.glob(".*.part"))


def test_unrelated_shared_prototypes_publish_without_one_global_lock(
    tmp_path: Path, monkeypatch
) -> None:
    entered = threading.Barrier(2)
    published: list[str] = []
    published_lock = threading.Lock()

    def publish(_bundle_root: Path, prototype: SimpleNamespace) -> None:
        entered.wait(timeout=2.0)
        with published_lock:
            published.append(prototype.asset_id)

    monkeypatch.setattr(measured, "_publish_shared_prototype_locked", publish)
    prototypes = [
        SimpleNamespace(asset_id="tree_a"),
        SimpleNamespace(asset_id="building_b"),
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(
            executor.map(
                lambda prototype: measured._publish_shared_prototype(
                    tmp_path / "shared", prototype
                ),
                prototypes,
            )
        )

    assert sorted(published) == ["building_b", "tree_a"]
    assert measured._SHARED_PROTOTYPE_LOCKS == {}


def test_one_scene_publishes_its_prototype_batch_in_parallel(
    tmp_path: Path, monkeypatch
) -> None:
    entered = threading.Barrier(2)
    published: list[str] = []
    progress: list[tuple[int, int, str]] = []
    published_lock = threading.Lock()

    def publish(_bundle_root: Path, prototype: SimpleNamespace) -> None:
        entered.wait(timeout=2.0)
        with published_lock:
            published.append(prototype.asset_id)

    monkeypatch.setattr(measured, "_PROTOTYPE_WORKER_LIMIT", 2)
    monkeypatch.setattr(measured, "_publish_shared_prototype_with_slot", publish)
    prototypes = [
        SimpleNamespace(family="trees", asset_id="tree_a"),
        SimpleNamespace(family="buildings", asset_id="building_b"),
    ]

    measured._publish_shared_prototypes(
        tmp_path / "shared",
        prototypes,
        progress_callback=lambda completed, total, asset_id: progress.append(
            (completed, total, asset_id)
        ),
    )

    assert sorted(published) == ["building_b", "tree_a"]
    assert [item[0] for item in progress] == [1, 2]
    assert {item[1] for item in progress} == {2}
    assert {item[2] for item in progress} == {"building_b", "tree_a"}


def test_shared_bundle_texture_tamper_invalidates_existing_scene(fixture_root) -> None:
    root, terrain, prototype = fixture_root
    shared = root / "shared" / "scene-prototypes"
    package = _build(
        root,
        terrain,
        prototype,
        "shared-scene",
        asset_bundle_root=shared,
    )
    receipt = json.loads(package.receipt.read_text(encoding="utf-8"))
    texture = shared / receipt["prototypes"][0]["texture"]["path"]
    before = texture.stat()
    content = texture.read_bytes()
    texture.write_bytes(bytes((content[0] ^ 1,)) + content[1:])
    os.utime(
        texture,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000),
    )
    with pytest.raises(measured.MeasuredSceneError, match="bundle bytes differ"):
        measured.validate_measured_scene_package(package.output_root)
