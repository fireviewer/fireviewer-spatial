from __future__ import annotations

import json
from pathlib import Path

import prepare_incident_terrains as incident
from prepare_global_05m import completion_receipt
from prepare_incident_terrains import (
    CASES,
    SAFETY_MARGIN_M,
    square_aoi,
    validate_case,
    write_case_plan,
)


def _write_test_ground_contracts(
    tmp_path: Path, monkeypatch
) -> tuple[Path, Path, Path, dict]:
    modes = ["ground_blend"] * 8 + [
        "directional_area",
        "linear_overlay",
        "watercourse_overlay",
        "slope_cliff_overlay",
    ]
    monkeypatch.setattr(incident, "GROUND_MICRO_SOURCE_COUNT", 1)
    monkeypatch.setattr(incident, "GROUND_PROFILE_COUNT", len(modes))
    profiles = [
        {
            "id": f"test-ground-{index:02d}",
            "application_mode": mode,
            "compatible_incidents": ["FR-26-00001"],
        }
        for index, mode in enumerate(modes)
    ]
    catalog = {
        "schema": incident.GROUND_SURFACE_SCHEMA,
        "status": "generated_pending_omniverse_visual_acceptance",
        "micro_source_count": 1,
        "profile_count": len(profiles),
        "runtime_texture_count": 4,
        "micro_source_runtime_import": "forbidden",
        "orthophoto_dependency": "forbidden",
        "profiles": profiles,
        "qa": {"automated": "passed"},
    }
    catalog["catalog_sha256"] = incident._canonical_sha256(catalog)
    catalog_path = tmp_path / "ground-surface-atlas-catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    families = []
    for mode in sorted(set(modes)):
        count = 8 if mode == "ground_blend" else 1
        families.append(
            {
                "application_mode": mode,
                "variant_ids": [f"variant-{index}" for index in range(count)],
            }
        )
    runtime = {
        "schema": incident.GROUND_RUNTIME_SCHEMA,
        "status": "specified_not_accepted",
        "orthophoto_dependency": "forbidden",
        "runtime_atlas": {"runtime_texture_count": 4},
        "scale_contract": {"direct_source_image_import": "forbidden"},
        "composition": {
            "ground_blend": {"maximum_profiles_per_tile": 4},
            "rail_geometry": {
                "steel_rails": "required_separate_future_3d_geometry"
            },
        },
        "determinism": {"silent_fallback": "forbidden"},
        "profile_families": families,
    }
    runtime_path = tmp_path / "ground-surface-runtime-contract.json"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    context_root = tmp_path / "context"
    context_cases = []
    for case in CASES:
        case_root = context_root / case.fire_id.lower()
        case_root.mkdir(parents=True)
        package = case_root / "ground-context.gpkg"
        package.write_bytes(f"context-{case.fire_id}".encode())
        package_hash = incident.sha256_file(package)
        manifest = {
            "fire_id": case.fire_id,
            "profile_binding_count": len(modes),
            "package": {"sha256": package_hash},
            "source_cleanup": {
                "status": "completed_after_package_validation"
            },
        }
        manifest_path = case_root / "ground-context-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        context_cases.append(
            {
                "fire_id": case.fire_id,
                "package_path": package.relative_to(context_root).as_posix(),
                "package_sha256": package_hash,
                "package_byte_count": package.stat().st_size,
                "manifest_path": manifest_path.relative_to(context_root).as_posix(),
                "manifest_sha256": incident.sha256_file(manifest_path),
                "feature_count": 1,
            }
        )
    context = {
        "schema": incident.GROUND_CONTEXT_CATALOG_SCHEMA,
        "status": "validated_six_case_context",
        "crs": incident.SOURCE_CRS,
        "orthophoto_dependency": "forbidden",
        "case_count": 6,
        "profile_binding_count": len(modes),
        "total_feature_count": 6,
        "cases": context_cases,
    }
    context["catalog_sha256"] = incident._canonical_sha256(context)
    context_path = context_root / "ground-context-catalog.json"
    context_path.write_text(json.dumps(context), encoding="utf-8")
    return catalog_path, runtime_path, context_path, context


def test_six_square_cases_cover_reference_bounds_with_margin() -> None:
    assert len(CASES) == 6
    assert len({case.fire_id for case in CASES}) == 6
    for case in CASES:
        validate_case(case)
        west, south, east, north = case.square_bounds_epsg2154_m
        source_west, source_south, source_east, source_north = (
            case.source_bounds_epsg2154_m
        )
        assert east - west == north - south
        assert west <= source_west - SAFETY_MARGIN_M
        assert south <= source_south - SAFETY_MARGIN_M
        assert east >= source_east + SAFETY_MARGIN_M
        assert north >= source_north + SAFETY_MARGIN_M


def test_die_is_replanned_without_legacy_terrain_or_placements(
    tmp_path: Path, monkeypatch
) -> None:
    die = next(case for case in CASES if case.fire_id == "FR-26-00001")
    catalog_path, runtime_path, context_path, context = _write_test_ground_contracts(
        tmp_path, monkeypatch
    )
    result = write_case_plan(
        die,
        tmp_path,
        catalog_path,
        runtime_path,
        context_path,
        context,
        overwrite=False,
    )
    root = tmp_path / result["root"]
    aoi = json.loads((root / result["aoi"]).read_text(encoding="utf-8"))
    manifest = json.loads((root / result["manifest"]).read_text(encoding="utf-8"))

    assert aoi == square_aoi(die)
    assert manifest["legacy_reuse"] == {
        "terrain": "forbidden",
        "ground_texture": "forbidden",
        "assets": "forbidden",
        "placements": "forbidden",
        "die_retained_logic_only": ["counts", "composition_structure"],
    }
    assert manifest["summary"]["orthophoto_request_count"] == 0
    assert manifest["worker_contract"]["source_products"] == ["mnt", "mns"]
    assert manifest["worker_contract"]["ground_2d"]["orthophoto_dependency"] == "forbidden"
    assert manifest["worker_contract"]["terrain_tile_contract"].startswith("bare-mnt")
    assert "mid_package_terrain_contract" not in manifest["worker_contract"]
    assert "orthophoto_resolution_m" not in manifest["tiling"]
    assert "ownership_rule" not in manifest["tiling"]
    assert "segmentation_rule" not in manifest["tiling"]
    assert all("orthophoto_request" not in tile for tile in manifest["tiles"])
    assert manifest["status"] == "blocked_pending_ground_surface_visual_acceptance"
    gate = manifest["ground_surface_gate"]
    assert gate["runtime_texture_count"] == 4
    assert gate["ground_context"]["status"] == "validated"
    assert gate["ground_context"]["feature_count"] == 1
    assert set(gate["candidate_profile_ids_by_application_mode"]) == {
        "ground_blend",
        "directional_area",
        "linear_overlay",
        "watercourse_overlay",
        "slope_cliff_overlay",
    }
    assert gate["railway_steel_rails"] == "required_separate_future_3d_geometry"
    assert all(
        set(tile["assets"])
        == {"terrain_package", "ground_material_map", "completion_receipt"}
        for tile in manifest["tiles"]
    )
    tile = manifest["tiles"][0]
    for name in ("terrain_package", "ground_material_map"):
        path = root / tile["assets"][name]["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"validated-{name}".encode())
    receipt = completion_receipt(tile, root, producer="terrain-test")
    assert receipt["terrain_tile_contract"].startswith("bare-mnt")
    assert "mid_package_terrain_contract" not in receipt
