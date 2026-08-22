from __future__ import annotations

import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import build_asset_library_53 as reviewed_library
import build_reference_usd_asset_library as library
import jsonschema
import pytest

OMNIVERSE_ROOT = Path(__file__).resolve().parents[1] / "omniverse"
if str(OMNIVERSE_ROOT) not in sys.path:
    sys.path.insert(0, str(OMNIVERSE_ROOT))
import build_measured_scene_usd as measured  # noqa: E402


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_portable_basename_accepts_historical_windows_receipts() -> None:
    assert (
        library._portable_basename(r"D:\archive\assets\001_tree\tree.usd") == "tree.usd"
    )
    assert library._portable_basename("/archive/assets/001_tree/tree.usd") == "tree.usd"


def _reference(asset_id: str, path: str, route: str = "hunyuan3d") -> dict:
    content = f"reference:{path}".encode()
    return {
        "asset_id": asset_id,
        "route": route,
        "source_relative": path,
        "source_bytes": len(content),
        "source_sha256": _sha(content),
        "seed": 1,
    }


def _reviewed(reference: dict, category: str) -> dict:
    usd = b"reviewed-usd"
    texture = b"reviewed-texture"
    return {
        "asset_id": reference["asset_id"],
        "category": category,
        "source": {
            "root": "reference_assets",
            "path": reference["source_relative"],
            "byte_count": reference["source_bytes"],
            "sha256": reference["source_sha256"],
        },
        "usd": {
            "root": "review_batch",
            "path": f"usd/{reference['asset_id']}.usd",
            "byte_count": len(usd),
            "sha256": _sha(usd),
        },
        "texture": {
            "root": "review_batch",
            "path": f"usd/textures/{reference['asset_id']}.png",
            "byte_count": len(texture),
            "sha256": _sha(texture),
        },
        "source_bounds": {
            "status": "reported",
            "coordinate_space": "source_glb_unscaled",
            "minimum": [-1.0, -1.0, -1.0],
            "maximum": [1.0, 1.0, 1.0],
            "diagonal": 3.464,
        },
        "usd_stage": {
            "status": "inspected",
            "up_axis": "Y",
            "meters_per_unit": 0.001,
            "default_prim": "/Asset",
        },
        "qualification": {
            "dimensions": {"status": "pending", "value_m": None},
            "ground_anchor": {"status": "pending", "offset_m": None},
            "visual": {"status": "pending", "accepted": False},
        },
    }


def test_runtime_tree_selection_respects_conifer_and_oak_evidence() -> None:
    assets = [
        {
            "asset_id": "oak",
            "category": "tree",
            "reference": {"path": "01_arbres/Chene pedoncule.png"},
            "placement": library._placement_profile(
                "01_arbres/Chene pedoncule.png", "tree"
            ),
        },
        {
            "asset_id": "pine",
            "category": "tree",
            "reference": {"path": "01_arbres/Pin sylvestre.png"},
            "placement": library._placement_profile(
                "01_arbres/Pin sylvestre.png", "tree"
            ),
        },
    ]
    catalog = {
        "schema": library.SCHEMA,
        "catalog_revision": _sha(b"tree-selection-catalog"),
        "assets": assets,
        "selection_pools": {"tree": ["oak", "pine"]},
    }

    oak = library._select_asset_for_candidate_from_validated_library(
        catalog,
        category="tree",
        zone="GPS-SPECIES",
        candidate="tree-1",
        rule_version="tree-v2",
        usage="technical_pilot_non_final",
        metadata={
            "context": "measured_woody_canopy",
            "semantic_tags": ["broadleaf", "oak"],
            "reference_terms": ["chene"],
            "tree_form_policy": "conifer_or_oak_only",
        },
    )
    conifer = library._select_asset_for_candidate_from_validated_library(
        catalog,
        category="tree",
        zone="GPS-SPECIES",
        candidate="tree-2",
        rule_version="tree-v2",
        usage="technical_pilot_non_final",
        metadata={
            "context": "measured_woody_canopy",
            "semantic_tags": ["conifer"],
            "reference_terms": ["pin", "noir"],
            "tree_form_policy": "conifer_or_oak_only",
        },
    )

    assert oak["asset_id"] == "oak"
    assert conifer["asset_id"] == "pine"


def test_runtime_tree_selection_never_uses_unrelated_broadleaf_or_natural_prop() -> None:
    assets = [
        {
            "asset_id": "oak",
            "category": "tree",
            "reference": {"path": "01_arbres/Chene pedoncule.png"},
            "placement": library._placement_profile(
                "01_arbres/Chene pedoncule.png", "tree"
            ),
        },
        {
            "asset_id": "pine",
            "category": "tree",
            "reference": {"path": "01_arbres/Pin sylvestre.png"},
            "placement": library._placement_profile(
                "01_arbres/Pin sylvestre.png", "tree"
            ),
        },
        {
            "asset_id": "plane",
            "category": "tree",
            "reference": {"path": "01_arbres/Platane.png"},
            "placement": library._placement_profile("01_arbres/Platane.png", "tree"),
        },
        {
            "asset_id": "rock",
            "category": "vegetation",
            "reference": {"path": "04_vegetaux/Grand rocher moussu.png"},
            "placement": library._placement_profile(
                "04_vegetaux/Grand rocher moussu.png", "vegetation"
            ),
        },
    ]
    catalog = {
        "schema": library.SCHEMA,
        "catalog_revision": _sha(b"tree-form-catalog"),
        "assets": assets,
        "selection_pools": {
            "tree": ["oak", "pine", "plane"],
            "vegetation": ["rock"],
        },
    }

    selected = {
        library._select_asset_for_candidate_from_validated_library(
            catalog,
            category="tree",
            zone="GPS-SPECIES",
            candidate=f"tree-{index}",
            rule_version="tree-v2",
            usage="technical_pilot_non_final",
            metadata={
                "context": "measured_woody_canopy",
                "semantic_tags": [],
                "reference_terms": [],
                "tree_form_policy": "conifer_or_oak_only",
            },
        )["asset_id"]
        for index in range(32)
    }

    assert selected <= {"oak", "pine"}
    assert selected == {"oak", "pine"}


def test_runtime_forest_terms_do_not_collapse_compatible_conifer_variety() -> None:
    assets = [
        {
            "asset_id": asset_id,
            "category": "tree",
            "reference": {"path": path},
            "placement": library._placement_profile(path, "tree"),
        }
        for asset_id, path in (
            ("douglas", "01_arbres/Douglas.png"),
            ("pine", "01_arbres/Pin sylvestre.png"),
            ("spruce", "01_arbres/Epicea commun.png"),
        )
    ]
    # Reproduce the accidental lexical advantage seen in the real r11
    # catalogue.  Forest-composition terms are evidence for the conifer form,
    # not for selecting this one species for every measured crown.
    assets[0]["placement"]["reference_terms"].extend(["foret", "fermee"])
    catalog = {
        "schema": library.SCHEMA,
        "catalog_revision": _sha(b"tree-variety-catalog"),
        "assets": assets,
        "selection_pools": {"tree": [asset["asset_id"] for asset in assets]},
    }

    selected = {
        library._select_asset_for_candidate_from_validated_library(
            catalog,
            category="tree",
            zone="GPS-FOREST",
            candidate=f"tree-{index}",
            rule_version="tree-v2",
            usage="technical_pilot_non_final",
            metadata={
                "context": "measured_woody_canopy",
                "semantic_tags": ["conifer"],
                "reference_terms": ["foret", "fermee", "coniferes"],
                "tree_form_policy": "conifer_or_oak_only",
            },
        )["asset_id"]
        for index in range(64)
    }

    assert selected == {"douglas", "pine", "spruce"}


def test_frozen_station_service_profile_is_enriched_without_catalog_rebuild() -> None:
    source = (
        "02_lot_2_services_et_habitat/"
        "08_station_service_rurale/reference.png"
    )
    current = library._placement_profile(source, "building")
    assert "commercial" in current["semantic_tags"]

    frozen = copy.deepcopy(current)
    frozen["semantic_tags"].remove("commercial")
    assert library._placement_profile_matches(source, "building", frozen)

    equipment_source = (
        "Lot_12_elements_complementaires_indispensables_autour_du_bati/"
        "04_station_service_rurale_modernisee.png"
    )
    equipment = library._placement_profile(equipment_source, "public_equipment")
    equipment["semantic_tags"].remove("commercial")
    assert library._placement_profile_matches(
        equipment_source, "public_equipment", equipment
    )

    malformed = copy.deepcopy(frozen)
    malformed["selection"] = "uncontrolled"
    assert not library._placement_profile_matches(source, "building", malformed)


def _runtime_building_asset(
    asset_id: str, path: str, extents: tuple[float, float, float]
) -> dict:
    return {
        "asset_id": asset_id,
        "category": "building",
        "reference": {"path": path},
        "placement": library._placement_profile(path, "building"),
        "source_bounds": {
            "minimum": [0.0, 0.0, 0.0],
            "maximum": list(extents),
        },
    }


def test_generic_buildings_are_residential_height_classed_and_never_stations() -> None:
    assets = [
        _runtime_building_asset(
            "house-fit", "02_batiments/maison_individuelle.png", (10.0, 7.0, 6.0)
        ),
        _runtime_building_asset(
            "house-wrong-shape", "02_batiments/pavillon_etroit.png", (4.0, 15.0, 4.0)
        ),
        _runtime_building_asset(
            "apartment", "02_batiments/immeuble_collectif.png", (10.0, 15.0, 6.0)
        ),
        _runtime_building_asset(
            "station",
            "02_lot_2_services_et_habitat/station_service_rurale.png",
            (10.0, 7.0, 6.0),
        ),
        _runtime_building_asset(
            "supermarket", "Lot_06_batiments_commerciaux/supermarche.png", (10.0, 7.0, 6.0)
        ),
        _runtime_building_asset(
            "mixed-warehouse",
            "02_batiments/habitat_commerce_industrie/entrepot_commercial.png",
            (10.0, 7.0, 6.0),
        ),
    ]
    catalog = {
        "schema": library.SCHEMA,
        "catalog_revision": _sha(b"strict-building-selection"),
        "assets": assets,
        "selection_pools": {"building": [asset["asset_id"] for asset in assets]},
    }
    common = {
        "category": "building",
        "zone": "FR-26",
        "rule_version": "building-v2",
        "usage": "technical_pilot_non_final",
    }

    low = {
        library._select_asset_for_candidate_from_validated_library(
            catalog,
            candidate=f"house-{index}",
            metadata={
                "context": "building",
                "semantic_tags": ["residential"],
                "reference_terms": ["batiment", "indifferencie"],
                "building_form": "low_rise_house",
                "measured_dimensions_m": [10.0, 7.0, 6.0],
            },
            **common,
        )["asset_id"]
        for index in range(32)
    }
    high = library._select_asset_for_candidate_from_validated_library(
        catalog,
        candidate="collective-1",
        metadata={
            "context": "building",
            "semantic_tags": ["residential"],
            "reference_terms": ["batiment", "indifferencie"],
            "building_form": "multi_storey_residential",
            "measured_dimensions_m": [10.0, 15.0, 6.0],
        },
        **common,
    )
    commercial = library._select_asset_for_candidate_from_validated_library(
        catalog,
        candidate="commerce-1",
        metadata={
            "context": "building",
            "semantic_tags": ["commercial"],
            "reference_terms": ["commerce"],
            "building_form": "low_rise_house",
            "measured_dimensions_m": [10.0, 7.0, 6.0],
        },
        **common,
    )
    station = library._select_asset_for_candidate_from_validated_library(
        catalog,
        candidate="station-1",
        metadata={
            "context": "building",
            "semantic_tags": ["commercial"],
            "reference_terms": ["station", "service"],
            "building_form": "low_rise_house",
            "measured_dimensions_m": [10.0, 7.0, 6.0],
        },
        **common,
    )

    assert low == {"house-fit"}
    assert high["asset_id"] == "apartment"
    assert commercial["asset_id"] == "supermarket"
    assert station["asset_id"] == "station"


def _fixture() -> tuple[dict, dict, dict, dict]:
    tree = _reference("111111111111_tree", "01_arbres/Lot 1/Tree.png")
    building = _reference(
        "222222222222_house",
        "02_batiments/01_petite_ville_rurale/House.png",
    )
    tile = _reference(
        "333333333333_tile",
        "pack_dalles_terrain_2D_53_assets/01/Tile.png",
        "terrain_2d",
    )
    documentation = _reference(
        "444444444444_doc",
        "validation_documentation/Preview.png",
        "documentation",
    )
    manifest = {
        "schema_version": 1,
        "asset_count": 4,
        "route_counts": {"hunyuan3d": 2, "terrain_2d": 1, "documentation": 1},
        "assets": [documentation, tile, building, tree],
    }
    reviewed = {
        "schema": "fireviewer.asset-library.v1",
        "assets": [_reviewed(tree, "tree"), _reviewed(building, "building")],
    }
    return manifest, reviewed, tree, building


def _inspection(
    path: Path,
    *,
    scope_safe: bool = True,
    up_axis: str = "Y",
    meters_per_unit: float = 1.0,
) -> dict:
    basis = {
        "schema": "fireviewer.usd-candidate-inspection.v1",
        "artifacts": [
            {
                "source_name": path.name,
                "source_sha256": _sha(path.read_bytes()),
                "source_bounds": {
                    "status": "reported",
                    "coordinate_space": "usd_authored_world",
                    "minimum": [-1.0, 0.0, -1.0],
                    "maximum": [1.0, 2.0, 1.0],
                },
                "usd_stage": {
                    "status": "inspected",
                    "up_axis": up_axis,
                    "meters_per_unit": meters_per_unit,
                    "default_prim": "/Asset",
                },
                "mesh_count": 1,
                "material_count": 1,
                "bound_material_mesh_count": 1,
                "material_scope_safe": scope_safe,
            }
        ],
    }
    basis["content_sha256"] = _sha(library._canonical_bytes(basis))
    return basis


def _simready_root(
    root: Path,
    *,
    active_reference: dict,
    rejected_reference: dict,
) -> Path:
    asset_id = active_reference["asset_id"]
    rejected_id = rejected_reference["asset_id"]
    directory = root / "assets" / f"001_{asset_id}"
    texture = directory / "textures" / f"{asset_id}.png"
    usd = directory / f"{asset_id}.usd"
    glb = directory / f"{asset_id}.glb"
    texture.parent.mkdir(parents=True)
    usd.write_bytes(b"#usda 1.0 normalized")
    glb.write_bytes(b"normalized glb")
    texture.write_bytes(b"normalized png")
    active = {
        "schema_version": 1,
        "status": "validated_omniverse_minimal_placeable_visual",
        "property_assignment_intent": "skip",
        "asset_count": 1,
        "rejected_count": 7,
        "meters_per_unit": 1.0,
        "scale_policy": "uniform_root_scale_only",
        "color_policy": "adaptive_sRGB_gamma_contrast_saturation",
        "usd_up_axis": "Z",
        "usd_root_rotation_degrees": 0.0,
        "assets": [
            {
                "index": 1,
                "asset_id": asset_id,
                "glb": str(glb),
                "usd": str(usd),
                "texture": str(texture),
                "scale_color": {
                    "target_m": 4.0,
                    "axis_scale_ratios": [2.0, 2.0, 2.0],
                    "corrected_sha256": _sha(glb.read_bytes()),
                    "after_geometry": {"bounds": [[-1.0, 0.0, -0.5], [1.0, 4.0, 0.5]]},
                },
                "usd_material_restore": {
                    "passed": True,
                    "source_glb_sha256": _sha(glb.read_bytes()),
                    "usd_sha256": _sha(usd.read_bytes()),
                    "meters_per_unit": 1.0,
                    "up_axis": "Z",
                    "root_rotation_x_degrees": 0.0,
                    "structural_validation": {
                        "texture_sha256": _sha(texture.read_bytes())
                    },
                },
                "passed": True,
            }
        ],
    }
    rejected_assets = [
        {"index": index, "asset_id": value}
        for index, value in (
            (2, rejected_id),
            (3, "reject-3"),
            (4, "reject-4"),
            (5, "reject-5"),
            (6, "reject-6"),
            (7, "reject-7"),
            (8, "reject-8"),
        )
    ]
    validation = {
        "schema_version": 1,
        "expected_active_count": 1,
        "expected_rejected_indices": list(range(2, 9)),
        "asset_count": 1,
        "passed_count": 1,
        "failed_count": 0,
        "library_errors": [],
        "passed": True,
        "assets": [
            {
                "asset_id": asset_id,
                "passed": True,
                "errors": [],
                "evidence": {
                    "glb_sha256": _sha(glb.read_bytes()),
                    "usd_sha256": _sha(usd.read_bytes()),
                    "texture_sha256": _sha(texture.read_bytes()),
                    "meters_per_unit": 1.0,
                    "up_axis": "Z",
                    "usd_extents_m": [2.0, 1.0, 4.0],
                },
            }
        ],
    }
    omniverse = {
        "status": "PASS",
        "features": [
            {
                "id": "com.nvidia.usd.minimal_placeable_visual",
                "status": "PASS",
            }
        ],
    }
    rejected = {
        "schema_version": 1,
        "status": "user_rejected",
        "asset_count": 7,
        "assets": rejected_assets,
    }
    for name, payload in (
        ("active-assets.json", active),
        ("simready-validation.json", validation),
        ("omniverse-asset-validator.json", omniverse),
        ("rejected-assets.json", rejected),
    ):
        (root / name).write_text(json.dumps(payload), encoding="utf-8")
    return root


def _final_simready_root(
    root: Path,
    *,
    active_reference: dict,
    rejected_reference: dict,
) -> tuple[Path, Path]:
    asset_id = active_reference["asset_id"]
    rejected_id = rejected_reference["asset_id"]
    directory = root / "assets" / f"001_{asset_id}"
    texture = directory / "textures" / f"{asset_id}.png"
    usd = directory / f"{asset_id}.usd"
    glb = directory / f"{asset_id}.glb"
    texture.parent.mkdir(parents=True)
    usd.write_bytes(b"final controlled usd")
    texture.write_bytes(b"final controlled texture")
    record = {
        "schema_version": 1,
        "index": 1,
        "asset_id": asset_id,
        "status": "retained",
        "source_library": r"D:\archives\simready_reviewed",
        "source_asset_report": r"D:\archives\simready_reviewed\asset-report.json",
        "glb": str(glb),
        "usd": str(usd),
        "texture": str(texture),
        "scale_policy": "unchanged_from_postprocess",
        "color_policy": "unchanged_from_postprocess",
        "passed": True,
    }
    (directory / "asset-report.json").write_text(json.dumps(record), encoding="utf-8")
    active = {
        "schema_version": 1,
        "status": "final_merged",
        "range": {"start": 1, "end": 2},
        "asset_count": 1,
        "rejected_count": 1,
        "operation": "merge_only_no_asset_modification",
        "sources": [],
        "assets": [record],
    }
    rejected = {
        "schema_version": 1,
        "status": "excluded_from_final_library",
        "asset_count": 1,
        "archive_policy": "provenance only",
        "assets": [
            {
                "index": 2,
                "asset_id": rejected_id,
                "reason": "rejected",
                "active_library_included": False,
            }
        ],
    }
    merge = {
        "schema_version": 1,
        "status": "passed",
        "operation": "merge_only_no_asset_modification",
        "range_count": 2,
        "active_count": 1,
        "rejected_count": 1,
        "active_directory_count": 1,
        "glb_count": 1,
        "usd_count": 1,
        "texture_count": 1,
        "active_indices": [1],
        "rejected_indices": [2],
    }
    for name, payload in (
        ("active-assets.json", active),
        ("merge-validation.json", merge),
        ("rejected-assets.json", rejected),
    ):
        (root / name).write_text(json.dumps(payload), encoding="utf-8")
    return root, usd


def test_images_define_expected_assets_but_never_enter_runtime() -> None:
    manifest, reviewed, tree, building = _fixture()
    payload = library.build_reference_asset_library(manifest, reviewed)

    assert payload["asset_count"] == 2
    assert payload["availability_counts"] == {"real_usd": 2}
    assert payload["fallback_policy"]["fallback_asset_count"] == 0
    assert [asset["asset_id"] for asset in payload["assets"]] == sorted(
        (tree["asset_id"], building["asset_id"])
    )
    assert payload["reference_manifest"]["runtime_images_embedded"] is False
    assert payload["selection_pools"]["tree"] == [tree["asset_id"]]
    assert payload["selection_pools"]["building"] == [building["asset_id"]]
    assert payload["available_asset_pools"]["tree"] == [tree["asset_id"]]
    assert payload["available_asset_pools"]["building"] == [building["asset_id"]]
    assert all(
        "pack_dalles_terrain_2D" not in asset["reference"]["path"]
        for asset in payload["assets"]
    )


def test_simready_normalized_replaces_reviewed_and_blocks_rejected_hunyuan(
    tmp_path: Path,
) -> None:
    active = _reference("111111111111_tree", "01_arbres/Lot 1/Normalized Tree.png")
    rejected = _reference("222222222222_tree", "01_arbres/Lot 1/Rejected Tree.png")
    extra_references = [
        _reference(f"reject-{index}", f"01_arbres/Lot 1/Reject {index}.png")
        for index in range(3, 9)
    ]
    manifest = {
        "schema_version": 1,
        "asset_count": 8,
        "route_counts": {"hunyuan3d": 8},
        "assets": [active, rejected, *extra_references],
    }
    reviewed = {
        "schema": "fireviewer.asset-library.v1",
        "assets": [_reviewed(active, "tree"), _reviewed(rejected, "tree")],
    }
    simready = _simready_root(
        tmp_path / "simready",
        active_reference=active,
        rejected_reference=rejected,
    )
    candidates = library.discover_candidate_assets(manifest, simready_root=simready)
    payload = library.build_reference_asset_library(
        manifest, reviewed, candidate_assets=candidates
    )
    indexed = {asset["asset_id"]: asset for asset in payload["assets"]}

    assert candidates.rejected_asset_ids == {
        rejected["asset_id"],
        *(reference["asset_id"] for reference in extra_references),
    }
    assert payload["source_counts"] == {"simready_normalized": 1}
    assert indexed[active["asset_id"]]["source_selection"]["tier"] == (
        "simready_normalized"
    )
    assert indexed[active["asset_id"]]["material"] == {
        "policy": "scoped_source_pbr",
        "source_package": False,
        "pbr_preserved": True,
    }
    assert indexed[active["asset_id"]]["usd_stage"]["up_axis"] == "Z"
    assert (
        indexed[active["asset_id"]]["qualification"]["dimensions"]["status"]
        == "accepted"
    )
    assert indexed[rejected["asset_id"]]["fallback_resolution"]["used"] is True
    assert (
        indexed[rejected["asset_id"]]["fallback_resolution"]["donor_asset_id"]
        == active["asset_id"]
    )

    output = tmp_path / "published"
    result = library.write_reference_asset_library(
        payload,
        review_batch_root=output,
        output_catalog=output / "reference-asset-library.v1.json",
        candidate_assets=candidates,
    )
    assert result["materialized_simready_normalized_count"] == 1
    assert (output / "simready-normalized" / f"{active['asset_id']}.usd").is_file()
    assert (
        output / "simready-normalized" / "textures" / f"{active['asset_id']}.png"
    ).is_file()

    manifest_path = tmp_path / "reference-manifest.json"
    reviewed_path = tmp_path / "reviewed-library.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    reviewed_path.write_text(json.dumps(reviewed), encoding="utf-8")
    cli_output = tmp_path / "cli-published"
    assert (
        library.main(
            [
                "--reference-manifest",
                str(manifest_path),
                "--reviewed-library",
                str(reviewed_path),
                "--review-batch-root",
                str(cli_output),
                "--reviewed-source-root",
                str(tmp_path / "unused-reviewed-source"),
                "--output-catalog",
                str(cli_output / "reference-asset-library.v1.json"),
                "--simready-root",
                str(simready),
                "--execute",
            ]
        )
        == 0
    )
    assert (cli_output / "reference-asset-library.v1.json").is_file()


def test_final_simready_replaces_uncontrolled_sources_and_preserves_rejections(
    tmp_path: Path,
) -> None:
    active = _reference("111111111111_tree", "01_arbres/Lot 1/Final Tree.png")
    rejected = _reference("222222222222_tree", "01_arbres/Lot 1/Rejected Tree.png")
    manifest = {
        "schema_version": 1,
        "asset_count": 2,
        "route_counts": {"hunyuan3d": 2},
        "assets": [active, rejected],
    }
    final_root, usd = _final_simready_root(
        tmp_path / "simready_final_0001_0294",
        active_reference=active,
        rejected_reference=rejected,
    )
    candidates = library.discover_candidate_assets(
        manifest,
        simready_root=final_root,
        candidate_inspection=_inspection(usd, up_axis="Z"),
    )
    payload = library.build_reference_asset_library(
        manifest,
        {"schema": "fireviewer.asset-library.v1", "assets": []},
        candidate_assets=candidates,
    )
    indexed = {asset["asset_id"]: asset for asset in payload["assets"]}

    assert candidates.rejected_asset_ids == {rejected["asset_id"]}
    assert payload["source_counts"] == {"simready_normalized": 1}
    assert indexed[active["asset_id"]]["usd"]["sha256"] == _sha(usd.read_bytes())
    assert indexed[rejected["asset_id"]]["fallback_resolution"]["used"] is True
    assert (
        indexed[rejected["asset_id"]]["fallback_resolution"]["donor_asset_id"]
        == active["asset_id"]
    )

    output = tmp_path / "published-final"
    result = library.write_reference_asset_library(
        payload,
        review_batch_root=output,
        output_catalog=output / "reference-asset-library.v1.json",
        candidate_assets=candidates,
    )
    assert result["materialized_simready_normalized_count"] == 1
    assert (output / "simready-normalized" / f"{active['asset_id']}.usd").is_file()
    assert not (output / "usd").exists()


def test_final_simready_legacy_stage_uses_local_material_and_metric_qualification(
    tmp_path: Path,
) -> None:
    active = _reference("111111111111_tree", "01_arbres/Lot 1/Final Tree.png")
    rejected = _reference("222222222222_tree", "01_arbres/Lot 1/Rejected Tree.png")
    manifest = {
        "schema_version": 1,
        "asset_count": 2,
        "route_counts": {"hunyuan3d": 2},
        "assets": [active, rejected],
    }
    final_root, usd = _final_simready_root(
        tmp_path / "simready_final_0001_0294",
        active_reference=active,
        rejected_reference=rejected,
    )
    candidates = library.discover_candidate_assets(
        manifest,
        simready_root=final_root,
        candidate_inspection=_inspection(
            usd, scope_safe=False, up_axis="Y", meters_per_unit=0.001
        ),
    )
    payload = library.build_reference_asset_library(
        manifest,
        {"schema": "fireviewer.asset-library.v1", "assets": []},
        candidate_assets=candidates,
    )
    selected = next(
        asset for asset in payload["assets"] if asset["asset_id"] == active["asset_id"]
    )

    assert selected["usd_stage"]["up_axis"] == "Y"
    assert selected["usd_stage"]["meters_per_unit"] == 0.001
    assert selected["material"] == {
        "policy": "fireviewer_color_override",
        "source_package": False,
        "pbr_preserved": False,
    }
    assert selected["texture"]["sha256"] == _sha(
        (
            final_root
            / "assets"
            / f"001_{active['asset_id']}"
            / "textures"
            / f"{active['asset_id']}.png"
        ).read_bytes()
    )
    assert selected["qualification"]["dimensions"] == {
        "status": "accepted",
        "value_m": [0.002, 0.002, 0.002],
    }


def test_missing_asset_resolves_to_compatible_real_usd_without_black_cube() -> None:
    donor = _reference(
        "aaaaaaaaaaaa_farm",
        "Lot_09_ferme_ancienne_renovee_et_habitat_agricole_realiste/01_corps_de_ferme.png",
    )
    missing = _reference(
        "bbbbbbbbbbbb_barn",
        "Lot_09_ferme_ancienne_renovee_et_habitat_agricole_realiste/02_grange.png",
    )
    manifest = {
        "schema_version": 1,
        "asset_count": 2,
        "route_counts": {"hunyuan3d": 2},
        "assets": [missing, donor],
    }
    payload = library.build_reference_asset_library(
        manifest,
        {
            "schema": "fireviewer.asset-library.v1",
            "assets": [_reviewed(donor, "building")],
        },
    )
    resolved = next(
        asset for asset in payload["assets"] if asset["asset_id"] == missing["asset_id"]
    )

    assert payload["availability_counts"] == {"real_usd": 2}
    assert payload["fallback_policy"]["fallback_asset_count"] == 1
    assert resolved["fallback_resolution"]["used"] is True
    assert resolved["fallback_resolution"]["donor_asset_id"] == donor["asset_id"]
    assert resolved["fallback_resolution"]["compatibility_mode"] == "exact_category"
    assert resolved["usd"]["path"] == f"usd/{donor['asset_id']}.usd"
    assert "placeholders" not in resolved["usd"]["path"]


def test_fallback_resolution_is_schema_valid_deterministic_and_tamper_evident() -> None:
    donor = _reference(
        "aaaaaaaaaaaa_farm",
        "Lot_09_ferme_ancienne_renovee_et_habitat_agricole_realiste/01_ferme.png",
    )
    missing = _reference(
        "bbbbbbbbbbbb_barn",
        "Lot_09_ferme_ancienne_renovee_et_habitat_agricole_realiste/02_grange.png",
    )
    manifest = {
        "schema_version": 1,
        "asset_count": 2,
        "route_counts": {"hunyuan3d": 2},
        "assets": [missing, donor],
    }
    reviewed = {
        "schema": "fireviewer.asset-library.v1",
        "assets": [_reviewed(donor, "building")],
    }
    first = library.build_reference_asset_library(manifest, reviewed)
    second = library.build_reference_asset_library(manifest, reviewed)
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "contracts"
        / "terrain"
        / "v1"
        / "reference-asset-library.v1.schema.json"
    )

    assert library._canonical_bytes(first) == library._canonical_bytes(second)
    jsonschema.validate(first, json.loads(schema_path.read_text(encoding="utf-8")))

    tampered = copy.deepcopy(first)
    fallback = next(
        asset
        for asset in tampered["assets"]
        if asset["asset_id"] == missing["asset_id"]
    )
    fallback["fallback_resolution"]["metadata_match_score"] += 1
    without_revision = dict(tampered)
    without_revision.pop("catalog_revision")
    tampered["catalog_revision"] = _sha(library._canonical_bytes(without_revision))
    with pytest.raises(
        library.ReferenceAssetLibraryError,
        match="fallback resolution differs",
    ):
        library.validate_reference_asset_library(tampered)


def test_real_usd_replaces_same_identifier_on_next_build() -> None:
    donor = _reference(
        "aaaaaaaaaaaa_existing_house",
        "02_batiments/01_petite_ville_rurale/Existing House.png",
    )
    building = _reference(
        "bbbbbbbbbbbb_new_house",
        "02_batiments/01_petite_ville_rurale/New House.png",
    )
    manifest = {
        "schema_version": 1,
        "asset_count": 2,
        "route_counts": {"hunyuan3d": 2},
        "assets": [building, donor],
    }
    reviewed = {
        "schema": "fireviewer.asset-library.v1",
        "assets": [_reviewed(donor, "building")],
    }
    pending = library.build_reference_asset_library(manifest, reviewed)
    pending_record = next(
        asset
        for asset in pending["assets"]
        if asset["asset_id"] == building["asset_id"]
    )
    assert pending_record["availability"] == "real_usd"
    assert pending_record["fallback_resolution"]["used"] is True
    assert pending_record["fallback_resolution"]["donor_asset_id"] == donor["asset_id"]

    updated = copy.deepcopy(reviewed)
    updated["assets"].append(_reviewed(building, "building"))
    rebuilt = library.build_reference_asset_library(manifest, updated)
    real_record = next(
        asset
        for asset in rebuilt["assets"]
        if asset["asset_id"] == building["asset_id"]
    )
    assert real_record["availability"] == "real_usd"
    assert real_record["fallback_resolution"]["used"] is False
    assert real_record["replacement"]["key"] == pending_record["replacement"]["key"]
    assert real_record["usd"]["path"] == f"usd/{building['asset_id']}.usd"
    expected_ids = sorted((donor["asset_id"], building["asset_id"]))
    assert rebuilt["selection_pools"]["building"] == expected_ids
    assert rebuilt["available_asset_pools"]["building"] == expected_ids


def test_premium_usdz_replaces_added_and_reviewed_hunyuan_atomically(
    tmp_path: Path,
) -> None:
    reference = _reference(
        "7546cb9a6b30_02_petite_caserne_sdis",
        "02_lot_2_services_et_habitat/02_petite_caserne_sdis.png",
    )
    manifest = {
        "schema_version": 1,
        "asset_count": 1,
        "route_counts": {"hunyuan3d": 1},
        "assets": [reference],
    }
    reviewed = {
        "schema": "fireviewer.asset-library.v1",
        "assets": [_reviewed(reference, "building")],
    }
    batch = tmp_path / "candidates" / "batch-0053-0057"
    usd = batch / "usd" / f"{reference['asset_id']}.usd"
    texture = batch / "usd" / "textures" / f"{reference['asset_id']}.png"
    receipt = batch / "reports" / "usd" / f"{reference['asset_id']}-usd.json"
    usd.parent.mkdir(parents=True)
    texture.parent.mkdir(parents=True)
    receipt.parent.mkdir(parents=True)
    usd.write_bytes(b"added-hunyuan-usd")
    texture.write_bytes(b"added-hunyuan-texture")
    receipt.write_text(
        json.dumps(
            {
                "asset": reference["asset_id"],
                "passed": True,
                "usd_sha256": _sha(usd.read_bytes()),
                "structural_validation": {"texture_sha256": _sha(texture.read_bytes())},
            }
        ),
        encoding="utf-8",
    )
    (batch / "reports" / "glb-validation.json").write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "path": f"/remote/{reference['asset_id']}.glb",
                        "passed": True,
                        "bounds": [[-1, 0, -1], [1, 2, 1]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    premium_root = tmp_path / "premium"
    premium_root.mkdir()
    premium = premium_root / "02_petite_caserne_sdis.usdz"
    with zipfile.ZipFile(premium, "w") as archive:
        archive.writestr("asset.usdc", b"premium-usdc")
        archive.writestr("textures/premium_0.jpg", b"premium-base-color")

    candidates = library.discover_candidate_assets(
        manifest,
        hunyuan_roots=[tmp_path / "candidates"],
        premium_usdz_root=premium_root,
        candidate_inspection=_inspection(premium, up_axis="Z"),
    )
    assert candidates[reference["asset_id"]].tier == "premium_usdz"
    payload = library.build_reference_asset_library(
        manifest,
        reviewed,
        candidate_assets=candidates,
    )
    asset = payload["assets"][0]
    assert payload["source_counts"] == {"premium_usdz": 1}
    assert asset["usd"]["path"].endswith(".usdz")
    assert asset["material"] == {
        "policy": "source_package_pbr",
        "source_package": True,
        "pbr_preserved": True,
    }
    assert asset["usd_stage"]["up_axis"] == "Z"
    assert asset["source_bounds"]["coordinate_space"] == "usd_canonical_y_up_from_z_up"
    output_root = tmp_path / "published"
    catalog = tmp_path / "reference-library.v1.json"
    result = library.write_reference_asset_library(
        payload,
        review_batch_root=output_root,
        output_catalog=catalog,
        candidate_assets=candidates,
    )
    assert result["materialized_premium_usdz_count"] == 1
    assert (output_root / asset["usd"]["path"]).read_bytes() == premium.read_bytes()
    assert (
        output_root / asset["texture"]["path"]
    ).read_bytes() == b"premium-base-color"


def test_premium_usdz_requires_hash_bound_openusd_inspection(tmp_path: Path) -> None:
    reference = _reference(
        "febc96eb56d2_06_chalet_alpin",
        "02_lot_2_services_et_habitat/06_chalet_alpin.png",
    )
    manifest = {
        "schema_version": 1,
        "asset_count": 1,
        "route_counts": {"hunyuan3d": 1},
        "assets": [reference],
    }
    premium_root = tmp_path / "premium"
    premium_root.mkdir()
    premium = premium_root / "chalet+house+3d+model.usdz"
    with zipfile.ZipFile(premium, "w") as archive:
        archive.writestr("asset.usdc", b"premium-usdc")
        archive.writestr("textures/premium_0.jpg", b"premium-base-color")
    candidates = library.discover_candidate_assets(
        manifest,
        premium_usdz_root=premium_root,
    )
    assert candidates[reference["asset_id"]].tier == "premium_usdz"
    with pytest.raises(
        library.ReferenceAssetLibraryError,
        match="bounds inspection is required",
    ):
        library.build_reference_asset_library(
            manifest,
            {"schema": "fireviewer.asset-library.v1", "assets": []},
            candidate_assets=candidates,
        )


def test_metadata_selection_is_repeatable_and_does_not_consume_an_asset() -> None:
    supermarket = _reference(
        "aaaaaaaaaaaa_supermarket",
        "Lot_06_batiments_commerciaux_du_quotidien/06_superette_de_quartier_plus_realiste.png",
    )
    farm = _reference(
        "bbbbbbbbbbbb_farm",
        "Lot_09_ferme_ancienne_renovee_et_habitat_agricole_realiste/01_corps_de_ferme_en_u_avec_cour_interieure.png",
    )
    tree = _reference("cccccccccccc_tree", "01_arbres/Lot 1/Tilleul.png")
    manifest = {
        "schema_version": 1,
        "asset_count": 3,
        "route_counts": {"hunyuan3d": 3},
        "assets": [supermarket, farm, tree],
    }
    payload = library.build_reference_asset_library(
        manifest,
        {
            "schema": "fireviewer.asset-library.v1",
            "assets": [
                _reviewed(supermarket, "building"),
                _reviewed(farm, "building"),
                _reviewed(tree, "tree"),
            ],
        },
    )
    metadata = {
        "context": "building",
        "semantic_tags": ["commercial"],
        "reference_terms": ["superette"],
    }

    first = library.select_asset_for_candidate(
        payload,
        category="building",
        zone="FR-30",
        candidate="building-a",
        rule_version="building-v1",
        usage="technical_pilot_non_final",
        metadata=metadata,
    )
    second = library.select_asset_for_candidate(
        payload,
        category="building",
        zone="FR-30",
        candidate="building-b",
        rule_version="building-v1",
        usage="technical_pilot_non_final",
        metadata=metadata,
    )

    assert first["asset_id"] == supermarket["asset_id"]
    assert second["asset_id"] == supermarket["asset_id"]
    assert first["repeatable"] is True
    assert second["repeatable"] is True


def test_public_selector_revalidates_and_rejects_tampered_catalog(monkeypatch) -> None:
    manifest, reviewed, _tree, _building = _fixture()
    payload = library.build_reference_asset_library(manifest, reviewed)
    validation_calls: list[dict] = []
    validate = library.validate_reference_asset_library

    def counted_validate(candidate):
        validation_calls.append(candidate)
        return validate(candidate)

    monkeypatch.setattr(library, "validate_reference_asset_library", counted_validate)
    selected = library.select_asset_for_candidate(
        payload,
        category="building",
        zone="FR-30",
        candidate="building-a",
        rule_version="building-v1",
        usage="technical_pilot_non_final",
    )
    assert selected["category"] == "building"
    assert len(validation_calls) == 1

    tampered = copy.deepcopy(payload)
    tampered["asset_count"] += 1
    with pytest.raises(
        library.ReferenceAssetLibraryError,
        match="catalog revision differs",
    ):
        library.select_asset_for_candidate(
            tampered,
            category="building",
            zone="FR-30",
            candidate="building-b",
            rule_version="building-v1",
            usage="technical_pilot_non_final",
        )
    assert len(validation_calls) == 2


def test_writer_is_atomic_idempotent_and_tamper_evident(tmp_path: Path) -> None:
    manifest, reviewed, _tree, building = _fixture()
    payload = library.build_reference_asset_library(manifest, reviewed)
    batch = tmp_path / "review-batch"
    output = tmp_path / "reference-library.v1.json"

    first = library.write_reference_asset_library(
        payload, review_batch_root=batch, output_catalog=output
    )
    second = library.write_reference_asset_library(
        payload, review_batch_root=batch, output_catalog=output
    )
    assert first == second
    assert first["placeholder_usd_count"] == 0
    assert not (batch / "placeholders").exists()
    assert building["asset_id"] in {asset["asset_id"] for asset in payload["assets"]}
    assert (
        json.loads(output.read_text(encoding="utf-8"))["catalog_revision"]
        == payload["catalog_revision"]
    )

    output.write_text("{}", encoding="utf-8")
    with pytest.raises(
        library.ReferenceAssetLibraryError,
        match="existing catalogue is immutable and differs",
    ):
        library.write_reference_asset_library(
            payload, review_batch_root=batch, output_catalog=output
        )


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parents[2]
        / "fireviewer-sdg"
        / "asset4sim"
        / "generated_hunyuan3d_v2"
        / "reference-manifest.json"
    ).is_file(),
    reason="The sibling 294-reference source catalogue is not available",
)
def test_current_reference_manifest_yields_294_usd_entries() -> None:
    repository = Path(__file__).resolve().parents[2]
    generated_root = (
        repository / "fireviewer-sdg" / "asset4sim" / "generated_hunyuan3d_v2"
    )
    reviewed = reviewed_library.build_asset_library(
        generated_root / "reference-manifest.json",
        generated_root / "review_batch_53",
    )
    payload = library.build_reference_asset_library(
        generated_root / "reference-manifest.json",
        reviewed,
    )
    assert payload["asset_count"] == 294
    assert payload["availability_counts"] == {"real_usd": 294}
    assert payload["fallback_policy"]["direct_asset_count"] == 53
    assert payload["fallback_policy"]["fallback_asset_count"] == 241
    assert payload["fallback_policy"]["black_placeholder_forbidden"] is True
    assert payload["category_counts"]["building"] == 148
    assert payload["category_counts"]["tree"] == 30
    assert all(
        not asset["reference"]["path"].startswith("pack_dalles_terrain_2D")
        for asset in payload["assets"]
    )


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parents[2]
        / "fireviewer-sdg"
        / "asset4sim"
        / "generated_hunyuan3d_v2"
        / "reference-manifest.json"
    ).is_file(),
    reason="The sibling 294-reference source catalogue is not available",
)
def test_current_added_assets_match_references_with_premium_precedence() -> None:
    repository = Path(__file__).resolve().parents[2]
    manifest = (
        repository
        / "fireviewer-sdg"
        / "asset4sim"
        / "generated_hunyuan3d_v2"
        / "reference-manifest.json"
    )
    candidates = library.discover_candidate_assets(
        manifest,
        hunyuan_roots=[
            repository
            / "fireviewer-sdg"
            / "asset4sim"
            / "generated_hunyuan3d_v2"
            / "production_batch_53_plus"
        ],
        premium_usdz_root=(
            repository
            / "fireviewer-sdg"
            / "asset4sim"
            / "generated_otherway_betterquality"
        ),
    )
    assert len(candidates) == 69
    assert sum(item.tier == "premium_usdz" for item in candidates.values()) == 26
    assert sum(item.tier == "added_hunyuan" for item in candidates.values()) == 43
    assert (
        candidates["febc96eb56d2_06_chalet_alpin"].source_name
        == "chalet+house+3d+model.usdz"
    )


def _materialize_reviewed(batch: Path, record: dict) -> None:
    for role, content in (("usd", b"reviewed-usd"), ("texture", b"reviewed-texture")):
        target = batch.joinpath(*record[role]["path"].split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _family(candidates: list[dict]) -> dict:
    return {
        "source_count": len(candidates),
        "valid_count": len(candidates),
        "ambiguous_count": 0,
        "rejected_count": 0,
        "placement_ready_count": len(candidates),
        "placement_blocked_count": 0,
        "instantiated_asset_count": 0,
        "candidates": candidates,
    }


def _select(asset_id: str):
    def selection(_library: object, **kwargs: object) -> dict:
        return {
            "asset_id": asset_id,
            "category": kwargs["category"],
            "selection_seed": 1,
            "usage_status": "technical_pilot_non_final",
            "metadata_match_score": 0,
            "repeatable": True,
        }

    return selection


def test_missing_selected_asset_packages_real_donor_usd(tmp_path: Path) -> None:
    donor = _reference("aaaaaaaaaaaa_tilleul", "01_arbres/Lot 1/Tilleul.png")
    missing = _reference("bbbbbbbbbbbb_pin", "01_arbres/Lot 1/Pin maritime.png")
    building = _reference(
        "cccccccccccc_house",
        "02_batiments/01_petite_ville_rurale/House.png",
    )
    manifest = {
        "schema_version": 1,
        "asset_count": 3,
        "route_counts": {"hunyuan3d": 3},
        "assets": [missing, donor, building],
    }
    reviewed_record = _reviewed(donor, "tree")
    building_record = _reviewed(building, "building")
    payload = library.build_reference_asset_library(
        manifest,
        {
            "schema": "fireviewer.asset-library.v1",
            "assets": [reviewed_record, building_record],
        },
    )
    batch = tmp_path / "review-batch"
    _materialize_reviewed(batch, reviewed_record)
    _materialize_reviewed(batch, building_record)
    catalog = tmp_path / "reference-library.v1.json"
    library.write_reference_asset_library(
        payload, review_batch_root=batch, output_catalog=catalog
    )
    terrain = tmp_path / "terrain.usda"
    terrain.write_text('#usda 1.0\n(def Xform "Terrain")\n', encoding="utf-8")
    inventory = {
        "schema": "fireviewer.mns-mnt-placement-inventory.v1",
        "build_id": _sha(b"inventory-build"),
        "zone_id": "FR-30-test",
        "crs": "EPSG:2154",
        "grid": {"core_bounds_l93_m": [700_000, 6_600_000, 700_500, 6_600_500]},
        "buildings": _family([]),
        "trees": _family(
            [
                {
                    "candidate_id": "tree-measured-1",
                    "status": "valid",
                    "reason_codes": [],
                    "position_l93_m": [700_100.5, 6_600_100.5],
                    "ground_elevation_mm": 100_000,
                    "height_cm": 500,
                    "equivalent_crown_radius_m": 2.0,
                }
            ]
        ),
    }
    inventory["inventory_sha256"] = _sha(measured.canonical_json_bytes(inventory))

    result = measured.build_measured_scene_usd(
        measured.TerrainReference(terrain, (700_000.0, 6_600_000.0)),
        inventory,
        catalog,
        tmp_path / "scene",
        portable_root=tmp_path,
        asset_roots={"review_batch": batch},
        selection_api=_select(missing["asset_id"]),
    )
    receipt = measured.validate_measured_scene_package(result.output_root)
    prototype = receipt["prototypes"][0]
    source = result.output_root / "prototypes" / prototype["source_usd"]["path"]

    assert receipt["placeholder_prototype_count"] == 0
    assert receipt["placeholder_instance_count"] == 0
    assert receipt["placement_policy"]["catalog_placeholder_usd_used"] is False
    assert prototype["asset_id"] == missing["asset_id"]
    assert prototype["availability"] == "real_usd"
    assert prototype["fallback_resolution"]["used"] is True
    assert prototype["fallback_resolution"]["donor_asset_id"] == donor["asset_id"]
    assert source.read_bytes() == b"reviewed-usd"
    assert b'def Cube "MissingAsset"' not in source.read_bytes()


def test_compatible_equipment_usd_can_serve_multiple_instances(tmp_path: Path) -> None:
    donor = _reference(
        "aaaaaaaaaaaa_road_barrier",
        "01_Lot_3D_T1_bordures_routieres_securite/08_barriere_pivotante.png",
    )
    missing = _reference(
        "bbbbbbbbbbbb_sports_fence",
        "06_Lot_3D_T6_sports_parkings_exterieurs/01_cloture_sportive.png",
    )
    tree = _reference("cccccccccccc_tree", "01_arbres/Lot 1/Tilleul.png")
    building = _reference(
        "dddddddddddd_house",
        "02_batiments/01_petite_ville_rurale/House.png",
    )
    manifest = {
        "schema_version": 1,
        "asset_count": 4,
        "route_counts": {"hunyuan3d": 4},
        "assets": [missing, donor, tree, building],
    }
    reviewed_record = _reviewed(donor, "road_equipment")
    tree_record = _reviewed(tree, "tree")
    building_record = _reviewed(building, "building")
    payload = library.build_reference_asset_library(
        manifest,
        {
            "schema": "fireviewer.asset-library.v1",
            "assets": [reviewed_record, tree_record, building_record],
        },
    )
    fallback = next(
        asset for asset in payload["assets"] if asset["asset_id"] == missing["asset_id"]
    )
    assert fallback["fallback_resolution"]["compatibility_mode"] == (
        "compatible_equipment"
    )
    batch = tmp_path / "review-batch"
    _materialize_reviewed(batch, reviewed_record)
    _materialize_reviewed(batch, tree_record)
    _materialize_reviewed(batch, building_record)
    catalog = tmp_path / "reference-library.v1.json"
    library.write_reference_asset_library(
        payload, review_batch_root=batch, output_catalog=catalog
    )
    terrain = tmp_path / "terrain.usda"
    terrain.write_text('#usda 1.0\n(def Xform "Terrain")\n', encoding="utf-8")
    context_candidates = [
        {
            "candidate_id": f"sports-feature-{index}",
            "status": "valid",
            "reason_codes": [],
            "asset_category": "sports_equipment",
            "selection_context": "sports_ground",
            "context_role": "sports",
            "source_ids": [f"SPORT-{index}"],
            "source_properties": {"nature": "Terrain de sport"},
            "position_l93_m": [700_100.0 + index * 20.0, 6_600_100.0],
            "ground_elevation_mm": 100_000,
            "yaw_rad": 0.0,
        }
        for index in range(2)
    ]
    inventory = {
        "schema": "fireviewer.mns-mnt-placement-inventory.v1",
        "build_id": _sha(b"context-inventory-build"),
        "zone_id": "FR-30-test",
        "crs": "EPSG:2154",
        "grid": {"core_bounds_l93_m": [700_000, 6_600_000, 700_500, 6_600_500]},
        "buildings": _family([]),
        "trees": _family([]),
        "context_assets": _family(context_candidates),
    }
    inventory["inventory_sha256"] = _sha(measured.canonical_json_bytes(inventory))

    result = measured.build_measured_scene_usd(
        measured.TerrainReference(terrain, (700_000.0, 6_600_000.0)),
        inventory,
        catalog,
        tmp_path / "scene-context",
        portable_root=tmp_path,
        asset_roots={"review_batch": batch},
        selection_api=_select(missing["asset_id"]),
    )
    receipt = measured.validate_measured_scene_package(result.output_root)

    assert result.context_asset_instance_count == 2
    assert receipt["prototype_count"] == 1
    assert receipt["placeholder_prototype_count"] == 0
    assert receipt["placeholder_instance_count"] == 0
    assert receipt["reconciliation"]["context_assets"]["instance_count"] == 2
    assert receipt["reconciliation"]["context_assets"]["asset_category_counts"] == {
        "sports_equipment": 2
    }
    scene = result.scene.read_text(encoding="utf-8")
    assert 'def PointInstancer "ContextAssets"' in scene
