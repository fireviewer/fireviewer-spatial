from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from run_production import (  # noqa: E402
    ProductionError,
    build_stages,
    load_contract,
    main,
    plan_record,
)


PROFILE = MODULE_DIRECTORY / "profiles/unity-v1-accepted.json"


def _feature_collection(path: Path, geometry: dict[str, object] | None) -> None:
    features = (
        []
        if geometry is None
        else [{"type": "Feature", "properties": {}, "geometry": geometry}]
    )
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )


def _polygon() -> dict[str, object]:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [700000.0, 6600000.0],
                [702000.0, 6600000.0],
                [702000.0, 6602000.0],
                [700000.0, 6602000.0],
                [700000.0, 6600000.0],
            ]
        ],
    }


def _line() -> dict[str, object]:
    return {
        "type": "LineString",
        "coordinates": [[700100.0, 6600100.0], [701900.0, 6601900.0]],
    }


def _config(tmp_path: Path) -> Path:
    source = tmp_path / "sources"
    source.mkdir()
    for name in ("aoi", "envelope", "buildings", "vegetation"):
        _feature_collection(source / f"{name}.geojson", _polygon())
    _feature_collection(source / "roads.geojson", _line())
    _feature_collection(source / "water.geojson", None)
    config = {
        "schema": "fireviewer.france-map-production.v1",
        "quality_profile": str(PROFILE),
        "zone": {
            "zone_id": "TEST-FR-01",
            "revision": 1,
            "package_id": "fireviewer-test-fr-01-r1-v1",
            "artifact_slug": "test-fr-01-r1",
            "label": "Zone de test",
            "origin_l93_m": [701000.0, 6601000.0, 200.0],
        },
        "inputs": {
            "aoi_l93": "sources/aoi.geojson",
            "production_envelope_l93": "sources/envelope.geojson",
            "buildings_l93": "sources/buildings.geojson",
            "vegetation_l93": "sources/vegetation.geojson",
            "roads_l93": ["sources/roads.geojson"],
            "water_courses_l93": ["sources/water.geojson"],
            "water_segments_l93": [],
            "water_surfaces_l93": [],
        },
        "attention_zones": [
            {
                "id": "centre",
                "label": "Centre",
                "bounds_l93_m": [700500.0, 6600500.0, 701500.0, 6601500.0],
            }
        ],
        "execution": {
            "artifact_root": "artifacts/test-fr-01-r1",
            "python_executable": "python",
            "blender_executable": None,
            "build_blender_scene": False,
            "near_lod_enabled": True,
            "expected_source_tile_count": None,
            "unity_validation_receipt": "artifacts/test-fr-01-r1/unity-validation-receipt.json",
            "unity_preview_png": "artifacts/test-fr-01-r1/unity-validation-preview.png",
        },
    }
    path = tmp_path / "zone.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_plan_is_complete_generic_and_does_not_create_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _config(tmp_path)
    contract = load_contract(config_path)
    stages = build_stages(contract)
    record = plan_record(contract, stages)

    assert record["stage_count"] == 10
    assert [stage["name"] for stage in stages] == [
        "plan_05m",
        "produce_05m",
        "near_imagery",
        "far_rasters",
        "far_imagery",
        "vector_package",
        "blender_scene",
        "unity_catalog",
        "validate_catalog",
        "site_upload",
    ]
    unity_command = next(
        stage["command"] for stage in stages if stage["name"] == "unity_catalog"
    )
    assert "--production-manifest" in unity_command
    assert "--global-vector-package" in unity_command
    assert "--detail-zones" in unity_command
    assert "--batch-size" in unity_command
    upload_command = next(
        stage["command"] for stage in stages if stage["name"] == "site_upload"
    )
    assert "--unity-validation-receipt" in upload_command
    assert "--unity-preview-png" in upload_command
    produce = next(
        stage["command"] for stage in stages if stage["name"] == "produce_05m"
    )
    assert produce.count("--exclude-polygons") == 1
    assert produce.count("--exclude-lines") == 2

    assert main(["--config", str(config_path), "--plan"]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == (
        "fireviewer.france-map-production-plan.v1"
    )
    assert not contract["artifact_root"].exists()


def test_preflight_requires_an_explicit_hydrographic_source(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["inputs"]["water_courses_l93"] = []
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ProductionError, match="fichier hydrographique"):
        load_contract(config_path)


def test_preflight_rejects_quality_profile_drift(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    profile["imagery"]["brightness"] = 1.0
    local_profile = tmp_path / "modified-profile.json"
    local_profile.write_text(json.dumps(profile), encoding="utf-8")
    config["quality_profile"] = str(local_profile)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ProductionError, match="plus identique"):
        load_contract(config_path)


def test_contract_hash_locks_local_source_contents(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    initial = load_contract(config_path)["config_hash"]
    water = tmp_path / "sources/water.geojson"
    water.write_text(water.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    assert load_contract(config_path)["config_hash"] != initial


def test_disabling_near_lod_skips_02m_download_and_marks_unity_export(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    initial_hash = load_contract(config_path)["config_hash"]
    config["execution"]["near_lod_enabled"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")

    contract = load_contract(config_path)
    stages = build_stages(contract)
    near = next(stage for stage in stages if stage["name"] == "near_imagery")
    unity = next(stage for stage in stages if stage["name"] == "unity_catalog")

    assert contract["config_hash"] == initial_hash
    assert contract["delivery_policy"] == {"near_lod_enabled": False}
    assert near["command"] is None
    assert near["optional"] is True
    assert "--disable-near-lod" in unity["command"]


def test_global_only_uses_checked_lidar_sources_without_detail_stages(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    profile["delivery"] = {"mode": "global_only"}
    profile["terrain"]["required_native_lidar_resolution_m"] = 0.2
    profile_path = tmp_path / "global-only-profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    config["quality_profile"] = str(profile_path)
    config["inputs"] = {"aoi_l93": config["inputs"]["aoi_l93"]}
    config["execution"]["delivery_mode"] = "global_only"
    config["execution"]["near_lod_enabled"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")

    contract = load_contract(config_path)
    stages = build_stages(contract)

    assert contract["delivery_policy"] == {
        "near_lod_enabled": False,
        "mode": "global_only",
        "required_native_lidar_resolution_m": 0.2,
    }
    assert [stage["name"] for stage in stages] == [
        "plan_05m",
        "source_lidar",
        "far_rasters",
        "far_imagery",
        "unity_catalog",
        "validate_catalog",
        "site_upload",
    ]
    source_stage = next(stage for stage in stages if stage["name"] == "source_lidar")
    assert source_stage["command"][-3:] == [
        "--required-elevation-resolution-m",
        "0.2",
        "--execute",
    ]
    catalog_stage = next(stage for stage in stages if stage["name"] == "unity_catalog")
    assert "--global-only" in catalog_stage["command"]
